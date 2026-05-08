"""
§1 Tile geometry operations — pad, unfold, fold, cross-token reshape.

Centralises tile-manipulation primitives that were previously duplicated
across residual_pursuit.py, ds_kv_cache.py, and incremental_decode.py.

All functions are pure and stateless; they operate on tensors and return
(tensor, metadata) tuples.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def tile_count(shape_2d: Tuple[int, int], tile_size: int) -> Tuple[int, int]:
    """Number of tiles in (rows, cols) directions for a padded matrix.

    Parameters
    ----------
    shape_2d: (H, W) of the (possibly padded) matrix.
    tile_size: Square tile dimension.

    Returns
    -------
    (n_tile_rows, n_tile_cols).
    """
    return (
        math.ceil(shape_2d[0] / tile_size),
        math.ceil(shape_2d[1] / tile_size),
    )


def pad_to_tile_multiple(
    w: torch.Tensor, tile_size: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Right-/bottom-pad *w* so both dims are multiples of *tile_size*.

    Parameters
    ----------
    w: ``(..., H, W)``.
    tile_size: Square tile dimension.

    Returns
    -------
    (w_padded, (pad_rows, pad_cols)).
    """
    rows, cols = w.shape[-2], w.shape[-1]
    pad_r = (tile_size - rows % tile_size) % tile_size
    pad_c = (tile_size - cols % tile_size) % tile_size
    if pad_r == 0 and pad_c == 0:
        return w, (0, 0)
    return F.pad(w, (0, pad_c, 0, pad_r)), (pad_r, pad_c)


def pad_rows_to_tile_multiple(
    mat: torch.Tensor, tile_size: int
) -> Tuple[torch.Tensor, int]:
    """Pad only the row dimension so it is a multiple of *tile_size*.

    Used when the column dimension (d_head) must not change, e.g. before
    2-D transform unfold where tile_size² must divide total elements.

    Parameters
    ----------
    mat: ``(N, d_head)``.
    tile_size: Square tile dimension.

    Returns
    -------
    (mat_padded, pad_rows).
    """
    N, d = mat.shape
    pad = (tile_size - N % tile_size) % tile_size
    if pad == 0:
        return mat, 0
    return F.pad(mat, (0, 0, 0, pad), mode='constant', value=0.0), pad


def unpad_matrix(w: torch.Tensor, orig_shape: Tuple[int, ...]) -> torch.Tensor:
    """Remove bottom/right padding added by pad_to_tile_multiple.

    Parameters
    ----------
    w: Padded tensor ``(..., H_pad, W_pad)``.
    orig_shape: Original shape (before padding).

    Returns
    -------
    Cropped tensor matching *orig_shape*.
    """
    return w[..., : orig_shape[-2], : orig_shape[-1]]


def unfold_to_tiles(
    mat: torch.Tensor, tile_size: int
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """Unfold a 2-D matrix into flat tiles via ``F.unfold``.

    (N, d_head) → (n_tiles, tile_size²)

    The matrix is padded to tile boundaries first so that every element
    belongs to exactly one tile.

    Parameters
    ----------
    mat: ``(N, d_head)``.
    tile_size: Square tile dimension.

    Returns
    -------
    (tiles, (n_tile_rows, n_tile_cols, pad_rows)).
        *tiles* has shape ``(n_tiles, tile_size²)``.
        *n_tile_rows*, *n_tile_cols* describe the 2-D tile grid.
        *pad_rows* is the number of zero-rows added so ``N_padded`` is
        divisible by *tile_size*.
    """
    N_orig, d_orig = mat.shape
    mat_padded, pad_rows = pad_rows_to_tile_multiple(mat, tile_size)
    N_pad = mat_padded.shape[0]

    n_tile_rows = N_pad // tile_size
    n_tile_cols = d_orig // tile_size if d_orig % tile_size == 0 else 1

    if n_tile_cols > 1 and d_orig % tile_size == 0:
        # 2-D tile grid: (N_pad, d_orig) → (n_tile_rows*tile_size, n_tile_cols*tile_size)
        patches = F.unfold(
            mat_padded.unsqueeze(0).unsqueeze(0),
            kernel_size=tile_size,
            stride=tile_size,
        )  # (1, tile_size², n_tiles)
        tiles = patches.squeeze(0).t()  # (n_tiles, tile_size²)
    else:
        # d_orig not divisible by tile_size → tile each row independently
        # Reshape: (N_pad, d_orig) → (n_tile_rows, tile_size, d_orig)
        # Then each tile is a single row of length d_orig
        M = tile_size * tile_size
        total_elems = N_pad * d_orig
        if total_elems % M != 0:
            # Pad total elements to be divisible by tile_size²
            needed = ((total_elems + M - 1) // M) * M
            pad_elems = needed - total_elems
            mat_padded = F.pad(mat_padded.reshape(-1), (0, pad_elems), mode='constant', value=0.0)
            # Recalculate tiles based on total elements
            n_tiles = needed // M
            tiles = mat_padded.reshape(n_tiles, M)
            return tiles, (n_tiles, 1, pad_rows)

        n_tiles = total_elems // M
        tiles = mat_padded.reshape(n_tiles, M)

    return tiles, (n_tile_rows, n_tile_cols, pad_rows)


def fold_from_tiles(
    tiles: torch.Tensor,
    output_shape: Tuple[int, int],
    tile_size: int,
    n_tile_rows: int = 0,
    n_tile_cols: int = 0,
) -> torch.Tensor:
    """Fold flat tiles back into a 2-D matrix.

    Inverse of ``unfold_to_tiles``.

    Parameters
    ----------
    tiles: ``(n_tiles, tile_size²)``.
    output_shape: Target ``(N_orig, d_orig)``.
    tile_size: Square tile dimension.
    n_tile_rows: Number of tile rows in the grid (0 = auto-detect).
    n_tile_cols: Number of tile cols in the grid (0 = auto-detect).

    Returns
    -------
    ``(N_orig, d_orig)`` matrix.
    """
    N_orig, d_orig = output_shape
    M = tile_size * tile_size

    if n_tile_cols <= 0 and d_orig % tile_size == 0:
        n_tile_cols = d_orig // tile_size
    elif n_tile_cols <= 0:
        n_tile_cols = 1

    if n_tile_rows <= 0:
        n_tile_rows = tiles.shape[0]

    if n_tile_cols > 1 and d_orig % tile_size == 0:
        # 2-D fold: F.fold reconstructs the padded matrix
        patches = tiles.t().unsqueeze(0)  # (1, M, n_tiles)
        N_padded = n_tile_rows * tile_size
        d_padded = n_tile_cols * tile_size
        mat_padded = F.fold(
            patches,
            output_size=(N_padded, d_padded),
            kernel_size=tile_size,
            stride=tile_size,
        ).squeeze(0).squeeze(0)  # (N_padded, d_padded)
    else:
        # d_orig not aligned → simple reshape
        mat_padded = tiles.reshape(-1, d_orig)

    # Crop to original size
    return mat_padded[:N_orig, :d_orig]


def reshape_for_cross_token(
    mat: torch.Tensor, group: int
) -> Tuple[torch.Tensor, int]:
    """Reshape (N, d) → (N//G, G*d) for cross-token joint encoding.

    Parameters
    ----------
    mat: ``(N, d)``.
    group: Token grouping factor (1 = no grouping).

    Returns
    -------
    (reshaped, pad_tokens).  *pad_tokens* = 0 when N divisible by *group*.
    """
    if group <= 1:
        return mat, 0
    N, d = mat.shape
    pad = (group - (N % group)) % group
    if pad > 0:
        mat = F.pad(mat, (0, 0, 0, pad), mode='constant', value=0.0)
        N += pad
    return mat.reshape(N // group, group * d), pad


def unreshape_cross_token(
    mat: torch.Tensor, group: int, n_tokens_original: int, pad_tokens: int = 0
) -> torch.Tensor:
    """Inverse of reshape_for_cross_token.

    Parameters
    ----------
    mat: ``(N//G, G*d)``.
    group: Token grouping factor.
    n_tokens_original: Original token count before grouping.
    pad_tokens: Number of padding tokens added during forward reshape.

    Returns
    -------
    ``(n_tokens_original, d)``.
    """
    if group <= 1:
        return mat
    N_enc, Gd = mat.shape
    d = Gd // group
    flat = mat.reshape(N_enc * group, d)
    if pad_tokens > 0:
        flat = flat[:-pad_tokens]
    return flat[:n_tokens_original]


__all__ = [
    "tile_count",
    "pad_to_tile_multiple",
    "pad_rows_to_tile_multiple",
    "unpad_matrix",
    "unfold_to_tiles",
    "fold_from_tiles",
    "reshape_for_cross_token",
    "unreshape_cross_token",
]
