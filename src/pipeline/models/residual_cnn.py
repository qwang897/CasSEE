"""Residual CNN on top of the SIM energy.

A 2-mer 1D-CNN predicts a *residual* logit ``r`` that is added to the SIM
free energy ``E``; the combined logit is then affine-scaled per-dataset
``(a, b)`` and passed through a sigmoid to get activity.

    final_logit = a * (SIM_E + r) + b
    r = CNN(one_hot(2-mers of 23bp seq))

The CNN sees ONLY the raw sequence (overlapping dinucleotide one-hot, 22 x 16);
it never sees the structural contacts/positions.

Parameter layout (must match ``runs/fold<k>/residual/weights/checkpoint.pth``):
    conv1(16->32, k3, pad1), conv2(32->64, k3, pad1),
    fc1(64*22 -> 128), fc2(128 -> 1), ab(Embedding(5, 2))
"""

import torch
import torch.nn as nn

KMERS = [a + b for a in "ATCG" for b in "ATCG"]
KMER_TO_IDX = {k: i for i, k in enumerate(KMERS)}


def two_mer_encode(seq: str):
    """Encode a DNA sequence as overlapping-dinucleotide one-hot vectors.

    Returns a ``(len(seq)-1, 16)`` one-hot (for a 23 bp seq -> 22 x 16).
    """
    vecs = []
    for i in range(len(seq) - 1):
        onehot = [0.0] * 16
        onehot[KMER_TO_IDX[seq[i:i + 2]]] = 1.0
        vecs.append(onehot)
    return vecs


class ResidualCNN(nn.Module):
    """Conv stack predicting a residual logit ``r`` for a 23 bp sequence.

    ``forward(x, i)`` returns ``(r, ab)``; the caller combines
    ``sigmoid(a * (SIM_E + r) + b)`` with the SIM free energy ``E``.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 22, 128)
        self.fc2 = nn.Linear(128, 1)
        self.ab = nn.Embedding(5, 2)

    def conv_stack(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)                       # (B,22,16) -> (B,16,22)
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = x.view(x.size(0), -1)                    # (B, 1408)
        x = self.leaky_relu(self.fc1(x))
        x = self.fc2(x)                              # (B, 1) residual
        return x

    def forward(self, x, i):
        """Return the CNN residual ``(B, 1)`` and the per-dataset ``(a, b)`` pair."""
        return self.conv_stack(x), self.ab(i)


class LongCorrectNet(nn.Module):
    """The residual CNN's conv stack, used as a frozen 2-dim feature extractor.

    ``LongCorrectNet`` is the same conv/fc stack as ``ResidualNet`` but *without*
    the ``ab`` embedding in ``forward``; it returns the raw ``fc2`` scalar for
    each input sequence (sgRNA and target DNA separately, giving 2 features).
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 22, 128)
        self.fc2 = nn.Linear(128, 1)
        self.ab = nn.Embedding(5, 2)   # unused at inference; kept for state_dict parity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x
