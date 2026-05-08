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
from rina.utils.tile_ops import (
    tile_count,
    pad_to_tile_multiple,
    pad_rows_to_tile_multiple,
    unpad_matrix,
    unfold_to_tiles,
    fold_from_tiles,
    reshape_for_cross_token,
    unreshape_cross_token,
)
from rina.utils.transform_pipeline import (
    TransformContext,
    TransformPipeline,
    resolve_transform_mode,
)
from rina.utils.bit_packing import (
    BITS_PER_PACK,
    pack_bases,
    unpack_bases,
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
    "tile_count",
    "pad_to_tile_multiple",
    "pad_rows_to_tile_multiple",
    "unpad_matrix",
    "unfold_to_tiles",
    "fold_from_tiles",
    "reshape_for_cross_token",
    "unreshape_cross_token",
    "TransformContext",
    "TransformPipeline",
    "resolve_transform_mode",
    "BITS_PER_PACK",
    "pack_bases",
    "unpack_bases",
]
