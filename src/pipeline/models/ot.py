"""Off-target / Cas9-variant model (``MyUcrisprML``).

A linear energy model on top of frozen SIM contact features plus a frozen
residual-CNN feature.  It scores off-target activity for a (guide, target) pair
and transfers to mutant Cas9 variants without re-training.

Feature stack::

    [ 240 one-hot mismatch energy,      # 20 positions x 12 non-identical base pairs
        1  sgRNA ensemble folding energy (wdG),
        8  extra: wg1, wt1, wg2, wt2 + consecutive-mismatch counts CMM_2..5,
        2  longE: the residual CNN applied to the guide and to the target
    ]

The only trainable parameters are ``linear(251, 1, bias=False)`` and a
per-dataset affine pair ``ab(Embedding(n_trsets, 2))``.  ``matchE`` -- a scalar
SIM contact energy from ``sites_data`` and the frozen per-residue energies --
is added to ``linear(X_ss)`` before the output::

    pred = sigmoid(a * (linear(X_ss) + matchE) + b)
"""

import os
import pickle as pk
import subprocess
from itertools import groupby

import numpy as np
import torch
import torch.nn as nn

from .residual_cnn import LongCorrectNet, two_mer_encode

BASES = "ATCG"
MM_PAIRS = [n1 + n2 for n1 in BASES for n2 in BASES if n1 != n2]   # 12
MM_PAIR_IDX = {p: i for i, p in enumerate(MM_PAIRS)}

# Feature names (in linear-weight order), after the 240 mismatch weights.
CORE_FEAT_NAMES = ["wdG", "wg1", "wt1", "wg2", "wt2",
                   "CMM_2", "CMM_3", "CMM_4", "CMM_5"]
LONG_FEAT_NAMES = ["wlong1", "wlong2"]

SCAFFOLD = ("GUUUUAGAGCUAGAAAUAGCAAGUUAAAAUAAGGCUAGUCCGUUAUCAACUUG"
            "AAAAAGUGGCACCGAGUCGGUGCU")


class MyUcrisprML(nn.Module):
    """Off-target model with FIXED SIM/long features and a trainable linear head."""

    def __init__(self, savepath, traindir_path, device="cuda", rs_path=None,
                 temp_dir=None, onparams_posweights=None, onparams_resenergies=None,
                 respair_encode=None, sites_data=None, ensE_data=None,
                 long_weights=None):
        super().__init__()

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.traindir_path = traindir_path
        self.savepath = savepath
        self.rs_path = rs_path
        self.temp_dir = temp_dir or "./temp"
        self.long_weights = long_weights

        # dataset -> stable index
        self.file_to_idx = {
            f[:-4]: i for i, f in enumerate(sorted(os.listdir(self.traindir_path)))
        }
        self.n_trsets = len(self.file_to_idx)

        self.mmpairs = MM_PAIRS
        self.mmpair_idx = MM_PAIR_IDX
        self.base_pair = {"A": "T", "T": "A", "C": "G", "G": "C"}
        self.scaffold = SCAFFOLD

        # ---- frozen SIM contact parameters ----
        self.onparams_posweights = onparams_posweights
        self.onparams_resenergies = onparams_resenergies
        self.respair_encode = respair_encode
        self.sites_data = sites_data or {}
        self.ensE_data = ensE_data or {}

        # ---- frozen residual-CNN: the two "longE" features ----
        if long_weights is None:
            raise ValueError("long_weights (residual CNN checkpoint) is required")
        self._long_model = LongCorrectNet()
        self._long_model.load_state_dict(torch.load(long_weights, map_location=self.device))
        self._long_model.to(self.device).eval()
        for p in self._long_model.parameters():
            p.requires_grad = False

        # ---- network head ----
        n_feat = 240 + 1 + 8 + 2
        self.linear = nn.Linear(n_feat, 1, bias=False)
        self.ab = nn.Embedding(self.n_trsets, 2)
        self.feat_dim = n_feat

    # ------------------------------------------------------------------
    # sgRNA ensemble energy (RNAstructure), with cache
    # ------------------------------------------------------------------
    def _features(self, gRNA, tarDNA, testsites=None, mutdata=None):
        """Build the full feature vector X_ss (and matchE) for one (gRNA, tarDNA)."""
        tarDNA = tarDNA[:23]
        gRNA = gRNA[:20] + tarDNA[20:23]

        pam = tarDNA[-2:]
        if pam != "GG":
            return None

        X_mmene = torch.zeros((20, 12))
        X_wdG = torch.zeros(1)
        X_extra = torch.zeros(8)
        X_long = torch.zeros(2)

        matchE = 0.0

        for i in range(20):
            site_pair = f"{gRNA[i]}{tarDNA[i]}"

            if testsites is None:
                on_site_name1 = f"{i+1}{gRNA[i]}_{i+2}{gRNA[i+1]}"
                on_site_name2 = f"{i+1}{tarDNA[i]}_{i+2}{tarDNA[i+1]}"
            else:
                on_site_name1 = testsites[0][i]
                on_site_name2 = testsites[1][i]

            if gRNA[i:i + 2] == tarDNA[i:i + 2]:
                matchE += self._sum_matchE_singlesite(on_site_name1, mutdata)

            elif gRNA[i] != tarDNA[i] and gRNA[i + 1] == tarDNA[i + 1]:
                X_mmene[i, self.mmpair_idx[site_pair]] = 1
                X_extra[0] += self._sum_matchE_singlesite(on_site_name1, mutdata)
                X_extra[1] += self._sum_matchE_singlesite(on_site_name2, mutdata)

            elif gRNA[i] == tarDNA[i] and gRNA[i + 1] != tarDNA[i + 1]:
                X_extra[0] += self._sum_matchE_singlesite(on_site_name1, mutdata)
                X_extra[1] += self._sum_matchE_singlesite(on_site_name2, mutdata)

            else:
                X_mmene[i, self.mmpair_idx[site_pair]] = 1
                X_extra[2] += self._sum_matchE_singlesite(on_site_name1, mutdata)
                X_extra[3] += self._sum_matchE_singlesite(on_site_name2, mutdata)

        NCMM_count = self._count_consecutive_mismatch(gRNA[:20], tarDNA)
        for n in [2, 3, 4, 5]:
            X_extra[n + 2] = NCMM_count[n]

        X_wdG[0] = self.get_dG_ensRNA(gRNA[:20])

        gRNA_encoded = torch.tensor([two_mer_encode(gRNA)], dtype=torch.float32, device=self.device)
        tarDNA_encoded = torch.tensor([two_mer_encode(tarDNA)], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            X_long[0] = self._long_model(gRNA_encoded).squeeze().item()
            X_long[1] = self._long_model(tarDNA_encoded).squeeze().item()

        X_ss = torch.cat([X_mmene.flatten(), X_wdG, X_extra, X_long])
        return X_ss, matchE

    def feature_names(self):
        return CORE_FEAT_NAMES + LONG_FEAT_NAMES

    def get_dG_ensRNA(self, gRNA):
        sgRNA = gRNA[:20].replace("T", "U") + self.scaffold
        if sgRNA in self.ensE_data:
            return self.ensE_data[sgRNA]

        temp_dir = self.temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        seq_path = os.path.join(temp_dir, "sgRNA.seq")
        out_path = os.path.join(temp_dir, "sgRNA.out")
        with open(seq_path, "w") as fw:
            fw.write(sgRNA + "\n")

        cmd = f"{self.rs_path}/exe/EnsembleEnergy {seq_path} -s --sequence > {out_path}"
        subprocess.run(cmd, shell=True)

        with open(out_path, "r") as f:
            outline = f.readline().strip()
        energy = float(outline.split()[-2])
        self.ensE_data[sgRNA] = energy

        os.remove(seq_path)
        os.remove(out_path)
        return energy

    def _sum_matchE_singlesite(self, site_name, additional_data=None):
        site_matchE = 0.0
        site_data_split = site_name.split("_")
        if len(site_data_split) == 2:                 # on-target / wt
            site_data = self.sites_data[site_name]
        else:                                         # on-target with mutation
            assert additional_data is not None, f"mut site {site_name} needs mutdata"
            site_data = additional_data[site_name]

        for code, posi in site_data:
            code = int(code)
            if code < 90:
                site_matchE += self._code_to_energy(code) * float(self.onparams_posweights[int(posi) - 1])
            else:
                raise Warning(f"code > 90 found: {code}")
        return site_matchE / 10.0

    def _code_to_energy(self, code):
        # code -> energy via precomputed mapping (built in builder fx)
        return self._energy_of_code[code]

    def _count_consecutive_mismatch(self, gRNA, tarDNA):
        mismatches = [r != d for r, d in zip(gRNA, tarDNA[:20])]
        NCMM_count = {2: 0, 3: 0, 4: 0, 5: 0}
        for k, g in groupby(mismatches):
            if k:
                length = len(list(g))
                if length >= 5:
                    NCMM_count[5] += 1
                elif length in NCMM_count:
                    NCMM_count[length] += 1
        return NCMM_count

    # ------------------------------------------------------------------
    # prediction
    # ------------------------------------------------------------------
    def pred_single(self, gRNA, tarDNA, testsites=None, mutdata=None):
        self.eval()
        with torch.no_grad():
            out = self._features(gRNA, tarDNA, testsites, mutdata)
            if out is None:
                return None
            X_ss, matchE = out
            X_ss = X_ss.to(self.linear.weight.device)
            E = self.linear(X_ss) + matchE
            return torch.sigmoid(E).item()

    # ------------------------------------------------------------------
    # text checkpoint (load only)
    # ------------------------------------------------------------------
    def load_params_txt(self, filename=None):
        """Load a text checkpoint (``<savepath>/checkpoint.txt`` by default)."""
        if filename is None:
            filename = os.path.join(self.savepath, "checkpoint.txt")
        param_dict = {}
        with open(filename, "r") as fr:
            for line in fr:
                parts = line.strip().split()
                name = parts[1]
                val = float(parts[0])
                if name not in param_dict:
                    param_dict[name] = val
                else:
                    param_dict[name] = val  # last-write wins (a_ duplicates)

        feat_len = self.linear.weight.shape[1]
        fnames = self.feature_names()

        weight_array = np.zeros(feat_len)
        k = 0
        for pos in range(1, 21):
            for m in self.mmpairs:
                weight_array[k] = param_dict[f"{m}_{pos}"]
                k += 1
        for fn in fnames:
            weight_array[k] = param_dict[fn]
            k += 1
        device = self.linear.weight.device
        dtype = self.linear.weight.dtype

        wt_tensor = torch.tensor(weight_array, dtype=dtype, device=device).view_as(self.linear.weight.data)
        self.linear.weight.data.copy_(wt_tensor)

        ab_np = np.zeros((self.n_trsets, 2))
        for idx, fname in enumerate(sorted(self.file_to_idx.keys())):
            ab_np[idx, 0] = param_dict[f"a_{fname}"]
            ab_np[idx, 1] = param_dict[f"b_{fname}"]
        self.ab.weight.data.copy_(torch.tensor(ab_np, dtype=dtype, device=device))

        print("Off-target parameters loaded")

    # ------------------------------------------------------------------
    # evaluation / testing
    # ------------------------------------------------------------------
    def test_all_datasets(self, datapath=None):
        import scipy.stats as ss
        self.eval()
        if datapath is None:
            # data/offtarget — one level up from train_u/
            datapath = os.path.dirname(self.traindir_path)
        out = {}
        for fname in sorted(os.listdir(datapath)):
            if not fname.endswith(".dat"):
                continue
            if fname[:-4] in self.file_to_idx:
                continue
            fp = os.path.join(datapath, fname)
            with open(fp) as f:
                lines = f.readlines()[1:]
            y_true, y_pred = [], []
            for line in lines:
                gRNA, tarDNA, A_exp = line.strip().split()
                pred_val = self.pred_single(gRNA, tarDNA)
                if pred_val is None:
                    continue
                y_true.append(float(A_exp))
                y_pred.append(pred_val)
            if len(set(y_true)) > 1:
                corr = abs(ss.pearsonr(y_true, y_pred)[0])
            else:
                corr = float("nan")
            out[fname[:-4]] = corr
            print(f"{fname[:-4]} {len(y_true)} {corr:.4f}")
        return out

    def test_one_dataset_with_mut(self, test_prefix, mutantsroot, slc_mut="all",
                                  return_data=False):
        """Correlation of one Cas9-variant benchmark.

        ``mutantsroot/<test_prefix>/`` holds ``testset_0_0_0.pkl`` (rows of
        guide, target, mutation label, efficiency, per-site names) and
        ``mut_data_0_0_0.pkl`` (the mutation-resolved contact tables).
        ``return_data=True`` additionally returns the raw points.
        """
        import scipy.stats as ss
        self.eval()
        testfile = np.load(os.path.join(mutantsroot, test_prefix, "testset_0_0_0.pkl"), allow_pickle=True)
        mutdata = np.load(os.path.join(mutantsroot, test_prefix, "mut_data_0_0_0.pkl"), allow_pickle=True)

        allmuts = set()
        allseqs = set()
        for line in testfile:
            gRNA, tarDNA, muts, eff, testsites = line
            if eff == "nan":
                continue
            allmuts.add(muts)
            allseqs.add((gRNA, tarDNA))

        def _eval_subset(lines_subset):
            y_true, y_pred, grnas, tardnas = [], [], [], []
            for line in lines_subset:
                gRNA, tarDNA, muts, eff, testsites = line
                pred_val = self.pred_single(gRNA, tarDNA, testsites, mutdata)
                if pred_val is None:
                    continue
                y_true.append(float(eff))
                y_pred.append(pred_val)
                grnas.append(str(gRNA))
                tardnas.append(str(tarDNA))
            return y_true, y_pred, grnas, tardnas

        results = []
        if slc_mut == "each mut":
            for curr_muts in allmuts:
                subset = [l for l in testfile if l[3] != "nan" and l[2] == curr_muts]
                results.append((curr_muts, subset))
        elif slc_mut == "each seq":
            for curr_seq in allseqs:
                subset = [l for l in testfile if l[3] != "nan" and (l[0], l[1]) == curr_seq]
                results.append((str(curr_seq), subset))
        else:
            subset = []
            for line in testfile:
                gRNA, tarDNA, muts, eff, testsites = line
                if eff == "nan":
                    continue
                if slc_mut != "all":
                    assert isinstance(slc_mut, list)
                    if muts not in slc_mut:
                        continue
                subset.append(line)
            results.append((slc_mut if isinstance(slc_mut, str) else str(slc_mut), subset))

        out = {}
        for label, subset in results:
            if len(subset) == 0:
                continue
            y_true, y_pred, grnas, tardnas = _eval_subset(subset)
            if len(set(y_true)) > 1:
                corr = abs(ss.pearsonr(y_true, y_pred)[0])
                spr = ss.spearmanr(y_true, y_pred)[0]
            else:
                corr = spr = float("nan")
            print(f"{test_prefix} {label} {len(y_true)} pearson={corr:.3f} spearman={spr:.3f}")
            rec = dict(n=len(y_true), pearson=float(corr), spearman=float(spr))
            if return_data:
                rec.update(y_true=y_true, y_pred=y_pred, grnas=grnas, tardnas=tardnas)
            out[f"{test_prefix} {label}"] = rec
        return out


def build_from_checkpoint(weights_dir, traindir_path, contacts_on, ot_data_dir,
                          long_weights, device="cuda", rs_path=None):
    """Build the off-target head and load every frozen input it needs.

    ``weights_dir``  ``runs/fold<k>/ot/weights``  (checkpoint.txt, posweights.pkl, resenergies.pkl)
    ``traindir_path````data/offtarget/train_u``   (dataset order -> per-dataset a, b)
    ``contacts_on``  ``data/contact/sites_data.pkl``
    ``ot_data_dir``  ``data/ot``                 (sites_data_off.pkl, respair_encode.pkl,
                                                  sgRNA_ens_energies.pkl)
    ``long_weights`` ``runs/fold<k>/residual/weights/checkpoint.pth``
    """
    with open(os.path.join(weights_dir, "posweights.pkl"), "rb") as f:
        onparams_posweights = pk.load(f)
    with open(os.path.join(weights_dir, "resenergies.pkl"), "rb") as f:
        onparams_resenergies = pk.load(f)
    with open(os.path.join(ot_data_dir, "respair_encode.pkl"), "rb") as f:
        respair_encode_raw = pk.load(f)
    respair_encode = {k: v for k, v in respair_encode_raw.items() if k.endswith("0")}

    with open(contacts_on, "rb") as f:
        sites_data_on = pk.load(f)
    with open(os.path.join(ot_data_dir, "sites_data_off.pkl"), "rb") as f:
        sites_data_off = pk.load(f)
    sites_data = {**sites_data_on, **sites_data_off}

    with open(os.path.join(ot_data_dir, "sgRNA_ens_energies.pkl"), "rb") as f:
        ensE_data = pk.load(f)

    model = MyUcrisprML(
        savepath=weights_dir, traindir_path=traindir_path, device=device,
        rs_path=rs_path,
        onparams_posweights=onparams_posweights,
        onparams_resenergies=onparams_resenergies,
        respair_encode=respair_encode,
        sites_data=sites_data, ensE_data=ensE_data,
        long_weights=long_weights,
    )
    # code -> energy mapping (0..89)
    model._energy_of_code = {
        int(respair_encode[k]): float(onparams_resenergies[k]) for k in onparams_resenergies
    }
    return model
