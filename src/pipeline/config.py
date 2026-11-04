"""Paths for the CasSEE test code.

Everything is resolved from this file, so the repository is relocatable::

    <repo>/
      src/pipeline/            model + data code
      data/
        contact/sites_data.pkl  contact features (320 dinucleotide sites)
        seqtest/                22 on-target benchmark files (4 sets x 5 folds + 2 external)
        muttest/{lib}/          3 Cas9-variant benchmarks (testset + mut_data)
        hf1/                    SpCas9-HF1 (5 per-fold slices + mut_data)
        offtarget/              66 off-target benchmark .dat + train_u/ (18)
        ot/                     fold-independent off-target inputs
        ot_variants/Kleinstiver/  Cas9-variant library (testset + mut_data)
      runs/fold{0..4}/
        sim/weights/            structure-informed module  (checkpoint.pth)
        residual/weights/       residual CNN              (checkpoint.pth)
        ot/weights/             off-target head + frozen per-fold inputs
      notebooks/                01-04 test notebooks
"""

import os
from pathlib import Path

PIPE_ROOT = Path(__file__).resolve().parent        # <repo>/src/pipeline
SRC_ROOT = PIPE_ROOT.parent                        # <repo>/src
TREE = SRC_ROOT.parent                             # <repo>

DATA = TREE / "data"
RUNS = TREE / "runs"

SIM_GETDATA = DATA / "contact"
SIM_SEQTEST = DATA / "seqtest"
SIM_MUTDATA = DATA / "muttest"
HF1_DIR = DATA / "hf1"

OT_DATASETS = DATA / "offtarget"
OTDATA_VARIANTS = DATA / "ot_variants"
OT_UTILS = DATA / "ot"

CONTACTS_ON = SIM_GETDATA / "sites_data.pkl"

# RNAstructure binary directory.  Only consulted when an sgRNA ensemble energy
# is missing from the cached ``data/ot/sgRNA_ens_energies.pkl``; the shipped
# cache covers every sequence in the off-target benchmarks, so normally unused.
RSPATH = os.environ.get("RNASTRUCTURE_PATH", "")


def fold_root(k):
    return RUNS / ("fold%d" % k)


def sim_weights_dir(k):
    return fold_root(k) / "sim" / "weights"


def residual_weights_dir(k):
    return fold_root(k) / "residual" / "weights"


def ot_weights_dir(k):
    return fold_root(k) / "ot" / "weights"


def ot_traindir(k):
    return OT_DATASETS / "train_u"
