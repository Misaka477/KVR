"""RINA utilities — orthogonal transforms, diagnostics, helpers."""

from rina.utils.walsh_hadamard import fwht, ifwht
from rina.utils.transforms import (
    TransformMode,
    dct_2d,
    idct_2d,
    dwt_haar_2d,
    idwt_haar_2d,
    apply_transform,
    apply_inverse_transform,
    compute_tile_diagnostics,
)

__all__ = [
    "fwht",
    "ifwht",
    "TransformMode",
    "dct_2d",
    "idct_2d",
    "dwt_haar_2d",
    "idwt_haar_2d",
    "apply_transform",
    "apply_inverse_transform",
    "compute_tile_diagnostics",
]
