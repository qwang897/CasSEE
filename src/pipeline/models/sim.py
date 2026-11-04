"""SIM -- Structure-Informed Model.

Per-base contact energy model for sgRNA/Cas9 on-target activity.

The model consumes a flat list of *contact tokens* (one per structure-derived
contact: a contact code plus a base position) and returns a scalar logit ``E``
(free energy, before the sigmoid)::

    E = sum_i( softplus(contact_embd[code_i]) * pos_weight[pos_i] ) + bias

- ``contact_embd``: an ``nn.Embedding(90, 1)`` mapping each of the 90
  structure-derived contact codes to a scalar energy.  ``softplus`` keeps every
  contact energy positive (contacts destabilize the complex).
- ``pos_weight``: a positive per-position weight that is ordered across three
  segments -- positions 1-13 (PAM-distal), 14-17 (core) and 18-21
  (PAM-proximal) -- and free inside a segment.  The ordering is enforced by
  construction: three learned boundaries confine each weight to its segment's
  interval through a sigmoid deviation, so it holds at every evaluation, not
  just after convergence.
- ``bias``: a single learnable offset.

Only ~112 parameters (90 + 3 + 21 + 1); the science lives in the contact
features, not in the network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentMonotonicEmbedding(nn.Module):
    """Segment-ordered positive per-index weights.

    The 21 positions split into three fixed segments -- I: idx 0-12, II:
    13-16, III: 17-20 (0-based; index 0 = leftmost / PAM-distal).  Within a
    segment the weights are free (no ordering); every weight in a later
    segment is >= every weight in an earlier one, and all weights stay >= 0
    (in every trained fold they came out > 0).

    Enforced *by construction*, not by a loss term: learned boundaries
    ``0 < b1 < b2 < b3`` (softplus increments) confine each weight to its
    segment's interval via a sigmoid deviation, so the ordering holds at
    every optimisation step.  24 parameters (3 bounds + 21 deviations).

    The convention across segments is *non-strict* (``min(II) >= max(I)``,
    ``min(III) >= max(II)``): in float32 ``sigmoid(30) == 1.0`` exactly, so a
    saturating deviation places a weight on its interval endpoint.
    """

    def __init__(self, num_embeddings: int = 21):
        super().__init__()
        if num_embeddings != 21:
            raise ValueError("segment layout is fixed to 21 positions (1-13 / 14-17 / 18-21)")
        self.raw_bounds = nn.Parameter(torch.randn(3))             # -> 0 < b1 < b2 < b3
        self.raw_devs = nn.Parameter(torch.randn(num_embeddings))  # free within segment

    def weights(self) -> torch.Tensor:
        inc = F.softplus(self.raw_bounds)
        b1 = inc[0]
        b2 = b1 + inc[1]
        b3 = b2 + inc[2]
        lo = torch.cat([torch.zeros(13, device=b1.device), b1.expand(4), b2.expand(4)])
        hi = torch.cat([b1.expand(13), b2.expand(4), b3.expand(4)])
        return lo + (hi - lo) * torch.sigmoid(self.raw_devs)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.weights()[indices]


class SGmodel(nn.Module):
    """Scalar free energy (logit) from flat contact-code / position token lists."""

    def __init__(self):
        super().__init__()
        self.contact_embd = nn.Embedding(90, 1)          # 90 contact codes -> energy
        self.pos_weight = SegmentMonotonicEmbedding(21)  # 21 segment-ordered weights
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, contacts: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # contacts: (1, L) LongTensor of code indices; pos: (1, L) 0-based positions
        energies = F.softplus(self.contact_embd(contacts).squeeze(-1).squeeze(0))
        weights = self.pos_weight(pos).squeeze(-1).squeeze(0)
        return torch.sum(energies * weights) + self.bias
