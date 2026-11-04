"""Helpers shared by the four test notebooks (01-04)."""

import contextlib
import io

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from .. import config
from ..data.sites import MutationSGdataset, SGdataset
from ..models.ot import build_from_checkpoint
from ..models.residual_cnn import ResidualCNN, two_mer_encode
from ..models.sim import SGmodel


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_sim(fold, dev):
    net = SGmodel().to(dev)
    net.load_state_dict(torch.load(str(config.sim_weights_dir(fold) / "checkpoint.pth"),
                                   map_location=dev))
    net.eval()
    return net


def load_residual(fold, dev):
    res = ResidualCNN().to(dev)
    res.load_state_dict(torch.load(str(config.residual_weights_dir(fold) / "checkpoint.pth"),
                                   map_location=dev))
    res.eval()
    return res


def predict(net, res, arr, dev, idx=None):
    """``arr`` = (N, 2) [seq, eff].  Returns (y_true, y_pred)."""
    ds = SGdataset(arr)
    ys, zs = [], []
    for seq, y, c, p in torch.utils.data.DataLoader(ds, batch_size=1):
        with torch.no_grad():
            e = net(c.to(dev), p.to(dev))
            if res is None:
                z = e
            else:
                enc = torch.tensor([two_mer_encode(seq[0])],
                                   dtype=torch.float32, device=dev)
                r, ab = res(enc, torch.LongTensor([0 if idx is None else idx]).to(dev))
                if idx is None:
                    z = e + r.squeeze()
                else:
                    z = ab[0, 0] * (e + r.squeeze()) + ab[0, 1]
        ys.append(y.item())
        zs.append(z.item())
    return np.array(ys, dtype=float), np.array(zs, dtype=float)


def rho(y, z, kind="spearman"):
    if len(np.unique(y)) < 2 or len(np.unique(z)) < 2:
        return float("nan")
    f = spearmanr if kind == "spearman" else pearsonr
    return float(f(y, z)[0])


def build_ot(fold, dev):
    """Off-target / Cas9-variant head of one fold."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m = build_from_checkpoint(
            weights_dir=str(config.ot_weights_dir(fold)),
            traindir_path=str(config.ot_traindir(fold)),
            contacts_on=str(config.CONTACTS_ON),
            ot_data_dir=str(config.OT_UTILS),
            long_weights=str(config.residual_weights_dir(fold) / "checkpoint.pth"),
            device=dev, rs_path=config.RSPATH)
        m.to(dev)
        m.load_params_txt()
    return m


def predict_mut_table(net, res, rows, mut_data, sites_data, dev, idx=None):
    """``rows`` = (N, 23) [seq, mut, eff, 20 site names] -> (y_true, y_pred).

    Compound site names (``A_B_MUT``) resolve through ``mut_data``, plain ones
    through ``sites_data``.
    """
    ys, zs = [], []
    for line in rows:
        seq, eff = line[0], float(line[2])
        names = line[3:]
        codes, poss = [], []
        for pos in range(20):
            sn = names[pos]
            tab = mut_data if len(sn.split("_")) > 2 else sites_data
            a = np.asarray(tab[sn], dtype=int)
            codes.extend(a[:, 0].tolist())
            poss.extend((a[:, 1] - 1).tolist())
        with torch.no_grad():
            e = net(torch.LongTensor(codes).unsqueeze(0).to(dev),
                    torch.LongTensor(poss).unsqueeze(0).to(dev))
            if res is None:
                z = e
            else:
                enc = torch.tensor([two_mer_encode(seq)], dtype=torch.float32, device=dev)
                r, ab = res(enc, torch.LongTensor([0 if idx is None else idx]).to(dev))
                z = e + r.squeeze() if idx is None else ab[0, 0] * (e + r.squeeze()) + ab[0, 1]
        ys.append(eff)
        zs.append(z.item())
    return np.array(ys, dtype=float), np.array(zs, dtype=float)


def mut_dataset(lib, idx="0_0_0"):
    """Variant benchmark table of one set, built without console noise."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ds = MutationSGdataset(config.SIM_MUTDATA / lib, idx)
    return ds


def variant_groups(fold, dev, name="Kleinstiver"):
    """Per-variant-group off-target predictions of one fold."""
    model = build_ot(fold, dev)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = model.test_one_dataset_with_mut(name, str(config.OTDATA_VARIANTS),
                                              slc_mut="each mut", return_data=True)
    return out


def offtarget_scores(fold, dev):
    """Pearson r of every off-target benchmark set (``{set: r}``)."""
    model = build_ot(fold, dev)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return model.test_all_datasets(datapath=str(config.OT_DATASETS))
