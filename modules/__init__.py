"""R.I.N.A (Residual-Integrated Neural Architecture) — PyTorch Modules."""

from .residual_pursuit import (
    ResidualBinaryPursuit,
    decode_from_bases,
    differential_encode_decode,
    adaptive_encode_matrix,
    encode_matrix,
)
from .svd_noise_shaping import (
    SVDNoiseShaper,
    compute_q_covariance,
    compute_nullspace_projector,
    compute_per_head_nullspace_projectors,
    compute_shared_nullspace_projector,
)
from .differential_cancellation import (
    DifferentialCanceller,
    PerturbationStrategy,
)

__all__ = [
    "ResidualBinaryPursuit",
    "decode_from_bases",
    "differential_encode_decode",
    "adaptive_encode_matrix",
    "encode_matrix",
    "SVDNoiseShaper",
    "compute_q_covariance",
    "compute_nullspace_projector",
    "compute_per_head_nullspace_projectors",
    "compute_shared_nullspace_projector",
    "DifferentialCanceller",
    "PerturbationStrategy",
]