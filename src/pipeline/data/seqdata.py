"""Readers for the on-target test sets under ``data/seqtest``.

Each ``*.dat`` file starts with a ``#`` header, followed by whitespace-delimited
``<30mer> <activity> <original_guide>`` rows. ``original_guide`` is normally the
23 bp target site; when it is the 20 bp spacer only, the 23 bp window is located
inside the 30-mer.
"""

import numpy as np


def load_seqtest_dat(path):
    """Parse a ``seqtest/*.dat`` file into an ``(N, 2)`` ``[seq, eff]`` array.

    30-mers whose original is 23 bp are used directly; 20-mers are embedded in
    their 30-mer to recover the 23 bp; unparseable rows are skipped.
    """
    with open(str(path)) as f:
        lines = f.readlines()[1:]  # drop header

    rows = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        _30, y, orig = parts[0], float(parts[1]), parts[2]
        if len(orig) == 23:
            seq = orig
        elif len(orig) == 20:
            seq = None
            for j in range(len(_30)):
                if _30[j:j + 20] == orig:
                    seq = _30[j:j + 23]
                    break
        else:
            continue
        if seq is None:
            continue
        rows.append([seq, y])
    return np.array(rows, dtype=object)


def load_seqtest_file(path):
    """Dispatch on extension: ``.npy`` -> raw array, ``.dat`` -> parsed array."""
    path = str(path)
    if path.endswith(".npy"):
        return np.load(path, allow_pickle=True)
    if path.endswith(".dat"):
        return load_seqtest_dat(path)
    raise ValueError(f"unsupported seqtest extension: {path}")
