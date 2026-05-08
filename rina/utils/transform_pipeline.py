"""
§2 TransformPipeline — unified forward/inverse transform with state tracking.

Single entry-point for all DCT/DWT/Hybrid/AUTO transform operations,
replacing 5+ scattered transform-invocation sites across ds_kv_cache.py
and incremental_decode.py.

Design:
    TransformContext captures the complete state needed to invert a
    transform (mode, per-tile decisions, original dimensions, pad rows).
    It provides store_dict/restore_dict serialization so the context can
    be persisted on DSKVCacheStore and used during decode.

    Forward:  (N, d_head) × config → (n_tiles, tile_size²) × context
    Inverse:  (n_tiles, tile_size²) × context → (N, d_head)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from rina.utils.transforms import (
    TransformMode,
    apply_transform,
    apply_inverse_transform,
)


@dataclass
class TransformContext:
    """Complete state required to invert an orthogonal transform.

    Persisted on DSKVCacheStore so reconstruct_all() can correctly
    apply the inverse transform without guessing from matrix shape.
    """

    mode: TransformMode = TransformMode.NONE
    """Forward transform mode (NONE = no transform active)."""

    decisions: Optional[List[str]] = None
    """Per-tile decisions (required for AUTO / hybrid modes)."""

    original_mat_shape: Optional[Tuple[int, int]] = None
    """(N_orig, d_head) — shape before any transform padding/reshape."""

    transform_pad_rows: int = 0
    """Zero-pad rows added so total elements are divisible by tile_size²."""

    tile_size: int = 16
    """Tile size used during the forward transform."""

    @property
    def is_active(self) -> bool:
        """True when a non-trivial transform is active."""
        return self.mode not in (TransformMode.NONE, None)

    def to_store_dict(self) -> dict:
        """Serialize to a dict suitable for DSKVCacheStore persistence."""
        return {
            "transform_mode": self.mode.value if isinstance(self.mode, TransformMode) else str(self.mode),
            "transform_decisions": list(self.decisions) if self.decisions else None,
            "_original_mat_shape": self.original_mat_shape,
            "transform_pad_rows": self.transform_pad_rows,
            "tile_size": self.tile_size,
        }

    @classmethod
    def from_store_dict(cls, d: dict) -> "TransformContext":
        """Restore from dict (allows partial fields — missing → defaults)."""
        mode_str = d.get("transform_mode", "none")
        if isinstance(mode_str, TransformMode):
            mode = mode_str
        elif mode_str and mode_str not in ("none", "", None):
            try:
                mode = TransformMode[mode_str.upper()]
            except KeyError:
                mode = TransformMode(mode_str)
        else:
            mode = TransformMode.NONE
        return cls(
            mode=mode,
            decisions=d.get("transform_decisions"),
            original_mat_shape=d.get("_original_mat_shape"),
            transform_pad_rows=d.get("transform_pad_rows", 0),
            tile_size=d.get("tile_size", 16),
        )


class TransformPipeline:
    """Unified forward/inverse transform with consistent state tracking.

    Usage::

        pipeline = TransformPipeline(tile_size=16)
        ctx = TransformContext(mode=TransformMode.DCT, tile_size=16)
        transformed, ctx = pipeline.forward(matrix, ctx)
        # ... encode transformed ...
        recovered = pipeline.inverse(transformed, ctx)
    """

    def __init__(self, tile_size: int = 16):
        self.tile_size = tile_size

    def forward(
        self,
        mat: torch.Tensor,
        ctx: TransformContext,
        *,
        smooth_threshold: float = 0.05,
        outlier_threshold: float = 3.0,
    ) -> Tuple[torch.Tensor, TransformContext]:
        """Apply forward transform, updating context with state.

        Parameters
        ----------
        mat: ``(N, d_head)`` — raw spatial-domain matrix.
        ctx: Current transform context (mode, etc.).

        Returns
        -------
        (transformed, ctx): *transformed* is ``(n_tiles, tile_size²)``
        in the transform domain.  *ctx* is updated with decisions, pad
        rows, and original shape.
        """
        if not ctx.is_active:
            # No transform: unfold to flat tiles directly
            from rina.utils.tile_ops import unfold_to_tiles, pad_rows_to_tile_multiple
            mat_padded, pad_rows = pad_rows_to_tile_multiple(mat, self.tile_size)
            tiles, (_, _, _) = unfold_to_tiles(mat_padded, self.tile_size)
            ctx.original_mat_shape = (mat.shape[0], mat.shape[1])
            ctx.transform_pad_rows = pad_rows
            ctx.decisions = ["none"] * tiles.shape[0]
            return tiles, ctx

        N_orig, d_head = mat.shape
        ctx.original_mat_shape = (N_orig, d_head)

        # Forward transform: handles padding + unfolding to 2-D internally
        transformed, decisions = apply_transform(
            mat,
            mode=ctx.mode,
            tile_size=self.tile_size,
            smooth_threshold=smooth_threshold,
            outlier_threshold=outlier_threshold,
        )
        ctx.decisions = decisions
        # compute pad rows: n_tiles * tile_size² must be >= N_orig * d_head
        M = self.tile_size * self.tile_size
        n_tiles = transformed.shape[0]
        total_padded_elems = n_tiles * M
        pad_elems = total_padded_elems - N_orig * d_head
        ctx.transform_pad_rows = max(0, (pad_elems + d_head - 1) // d_head) if d_head > 0 else 0

        return transformed, ctx

    def inverse(
        self,
        transformed: torch.Tensor,
        ctx: TransformContext,
    ) -> torch.Tensor:
        """Apply inverse transform and restore to original matrix shape.

        Parameters
        ----------
        transformed: ``(n_tiles, tile_size²)`` — transform-domain tiles.
        ctx: Context from the forward pass (mode, decisions, original shape).

        Returns
        -------
        ``(N_orig, d_head)`` — spatial-domain reconstruction, cropped
        to the original dimensions recorded in *ctx*.
        """
        M = self.tile_size * self.tile_size
        n_tiles = transformed.shape[0]

        if not ctx.is_active:
            # No transform: fold tiles back to 2-D matrix
            if ctx.original_mat_shape is not None:
                N_orig, d_orig = ctx.original_mat_shape
                # simple reshape: total elements = n_tiles * M
                total = n_tiles * M
                # Discard padding extras (if any)
                mat = transformed.reshape(-1)[:N_orig * d_orig].reshape(N_orig, d_orig)
                return mat
            return transformed

        if ctx.original_mat_shape is not None:
            N_orig, d_head = ctx.original_mat_shape
        else:
            # Fallback: infer from tile geometry
            d_head = M  # Default: square tiles
            N_padded = n_tiles * M // d_head if d_head > 0 else n_tiles
            N_orig = N_padded
            if ctx.transform_pad_rows > 0:
                N_orig = N_padded - ctx.transform_pad_rows

        # Compute assert-compatible original shape for apply_inverse_transform
        if ctx.original_mat_shape is not None:
            orig_shape = ctx.original_mat_shape
        else:
            orig_shape = None

        result = apply_inverse_transform(
            transformed,
            mode=ctx.mode,
            tile_size=self.tile_size,
            decisions=ctx.decisions,
            original_shape=orig_shape,
        )

        return result

    @staticmethod
    def inverse_tiles_to_raw(
        transformed: torch.Tensor,
        ctx: TransformContext,
    ) -> torch.Tensor:
        """Inverse transform WITHOUT reshaping to original (N, d_head).

        Returns ``(n_tiles, tile_size²)`` in the RAW spatial domain.
        Useful when merging old and new tile sets that are in different
        domains.
        """
        if not ctx.is_active:
            return transformed
        return apply_inverse_transform(
            transformed,
            mode=ctx.mode,
            tile_size=ctx.tile_size,
            decisions=ctx.decisions,
            original_shape=None,  # keep tile-space
        )


def resolve_transform_mode(mode) -> TransformMode:
    """Resolve a string or TransformMode to a TransformMode enum."""
    if isinstance(mode, TransformMode):
        return mode
    if mode and mode not in ("none", "", None):
        try:
            return TransformMode[mode.upper()]
        except KeyError:
            return TransformMode(mode)
    return TransformMode.NONE


__all__ = [
    "TransformContext",
    "TransformPipeline",
    "resolve_transform_mode",
]
