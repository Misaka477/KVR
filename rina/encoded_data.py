"""
§4 EncodedData — immutable snapshot of 1-bit encoded K/V data.

Encapsulates bit-packed bases, FP16 alphas, residual differential
data, and the logical shape they represent.  This is the persistent
"data payload" of a DSKVCacheStore — all encoding parameters and
transform state live in Metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

from modules.residual_pursuit import ResidualAlphas


@dataclass
class EncodedData:
    """Immutable encoded representation of a K or V matrix.

    All tensors are on-device.  The packed bases use int32 bit-packing
    (32 signs per int32 element) for ~32× storage reduction relative
    to FP16 bases.
    """

    bases: Optional[torch.Tensor] = None
    """Bit-packed bases ``(N_steps, n_tiles, M_packed)`` int32."""

    bases_shape_M: Optional[int] = None
    """Original M (tile_size²) before bit-packing — used during unpack."""

    alphas: Optional[ResidualAlphas] = None
    """FP16 scaling factors ``(N_steps, n_tiles)``."""

    orig_shape: Optional[Tuple[int, int]] = None
    """Logical ``(n_encoded_tokens, d_head)`` of the encoded matrix."""

    # ── Residual differential ───────────────────────────────────────────
    bases_residual: Optional[torch.Tensor] = None
    bases_shape_M_residual: Optional[int] = None
    alphas_residual: Optional[ResidualAlphas] = None

    @property
    def n_tiles(self) -> int:
        """Number of encoded tiles."""
        if self.bases is None:
            return 0
        return self.bases.shape[1]

    @property
    def n_steps(self) -> int:
        """Number of Σ-Δ steps (bases)."""
        if self.bases is None:
            return 0
        return self.bases.shape[0]

    @property
    def has_residual(self) -> bool:
        """True if a differential residual stage is present."""
        return self.bases_residual is not None and self.alphas_residual is not None

    @property
    def d_head(self) -> int:
        """Head dimension from orig_shape."""
        if self.orig_shape is not None:
            return self.orig_shape[1]
        return 0

    def as_primary(self) -> "EncodedData":
        """Return a copy with only the primary bases (strip residual)."""
        return EncodedData(
            bases=self.bases,
            bases_shape_M=self.bases_shape_M,
            alphas=self.alphas,
            orig_shape=self.orig_shape,
        )


__all__ = ["EncodedData"]
