"""CasSEE: activity prediction for SpCas9 variants.

Modules
-------
config              paths and runtime settings
data.sites          contact-encoded sequence datasets
data.seqdata        test-set readers
models.sim          structure-informed module (per-position contact energy)
models.residual_cnn 2-mer convolutional residual module
models.ot           off-target / variant head
test.casee_eval     evaluation helpers used by ``notebooks/``
"""

from . import config  # noqa: F401
