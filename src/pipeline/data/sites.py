"""SIM contact-feature datasets.

Converts a 23-bp guide (or a (guide, target, mutation) triple) into flat
contact/position token lists consumed by :class:`SGmodel`::

    site = "{pos+1}{base}_{pos+2}{next}"  lookup -> [[code, poslabel], ...]

- ``code`` : contact-code index (0..89), a base-amino-acid or base-base interaction.
- ``poslabel`` : 1-based position; stored as ``poslabel - 1`` (0..20) for the model.

Sequences that are not exactly 23 bp or that do not end in ``GG`` are skipped.
"""

import os
import pickle as pk

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import SIM_GETDATA


def load_sites_data(path=None):
    if path is None:
        path = SIM_GETDATA / "sites_data.pkl"
    with open(str(path), "rb") as f:
        return pk.load(f)


def _site_of(seq, pos):
    return f"{pos+1}{seq[pos]}_{pos+2}{seq[pos+1]}"


class SGdataset(Dataset):
    """Sequence dataset -> ``(seq, eff, contacts, pos)`` per item.

    Parameters
    ----------
    datafile : ndarray | str
        ``(N, 2)`` array of ``[seq, eff]`` rows, or a ``.npy`` path.
    sites_data : dict
        The contact table (defaults to ``sites_data.pkl``).
    """

    def __init__(self, datafile, sites_data=None):
        if sites_data is None:
            sites_data = load_sites_data()

        if isinstance(datafile, np.ndarray):
            all_seq_data = datafile
        elif isinstance(datafile, (str, os.PathLike)):
            all_seq_data = np.load(str(datafile), allow_pickle=True)
        else:
            raise TypeError(f"unsupported datafile type {type(datafile)}")

        seq_data = all_seq_data

        self.seqs, self.effs, self.contacts, self.poss = [], [], [], []

        for line in seq_data:
            seq, eff = line
            if len(seq) != 23 or seq[-2:] != "GG":
                continue

            _seqdata, _posdata = [], []
            try:
                for pos in range(20):
                    site_name = _site_of(seq, pos)
                    pos_contacts = np.array(sites_data[site_name], dtype=int)
                    _seqdata.extend(pos_contacts[:, 0].astype(int).tolist())
                    _posdata.extend((pos_contacts[:, 1].astype(int) - 1).tolist())

                self.contacts.append(_seqdata)
                self.poss.append(_posdata)
                self.seqs.append(seq)
                self.effs.append(float(eff))
            except KeyError:
                continue

        # Precompute once: building the full-length tensor per item is O(N).
        self._eff_tensor = torch.FloatTensor(self.effs)

    def __len__(self):
        return len(self.effs)

    def __getitem__(self, index):
        return (self.seqs[index],
                self._eff_tensor[index],
                torch.LongTensor(self.contacts[index]),
                torch.LongTensor(self.poss[index]))


class MutationSGdataset(Dataset):
    """Mutation / off-target test set -> ``(seq, mut, eff, contacts, pos)``.

    Reads ``<testdir>/<name>_testset_<idx>.npy`` (23 columns:
    ``[seq, mut, eff, 20 site_names]``) and resolves each position's contacts
    by name: a compound ``_``-joined name (len > 2, e.g. ``1A_2T_Q695A``) comes
    from ``<testdir>/mut_data_<idx>.pkl``; a plain name from ``sites_data.pkl``.
    """

    def __init__(self, testdir, idx, sites_data=None):
        if sites_data is None:
            sites_data = load_sites_data()

        testname = os.path.basename(testdir)
        testnpyname = f"{testdir}/{testname}_testset_{idx}.npy"
        with open(f"{testdir}/mut_data_{idx}.pkl", "rb") as f:
            mut_data = pk.load(f)
        test_data = np.load(testnpyname, allow_pickle=True)

        self.seqs, self.muts, self.effs, self.contacts, self.poss = [], [], [], [], []

        for line in test_data:
            seq, mut, eff = line[:3]
            site_names = line[3:]

            _seqdata, _posdata = [], []
            for pos in range(20):
                site_name = site_names[pos]
                if len(site_name.split("_")) > 2:
                    pos_contacts = np.array(mut_data[site_name], dtype=int)
                else:
                    pos_contacts = np.array(sites_data[site_name], dtype=int)
                _seqdata.extend(pos_contacts[:, 0].astype(int).tolist())
                _posdata.extend((pos_contacts[:, 1].astype(int) - 1).tolist())

            self.contacts.append(_seqdata)
            self.poss.append(_posdata)
            self.seqs.append(seq)
            self.muts.append(mut)
            self.effs.append(float(eff))

        self._eff_tensor = torch.FloatTensor(self.effs)   # avoid O(N) per item

    def __len__(self):
        return len(self.effs)

    def __getitem__(self, index):
        return (self.seqs[index],
                self.muts[index],
                self._eff_tensor[index],
                torch.LongTensor(self.contacts[index]),
                torch.LongTensor(self.poss[index]))
