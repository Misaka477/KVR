"""
§5 Metadata — encoding parameters and transform state for decode.

Metadata captures everything needed to correctly reconstruct a matrix
from its EncodedData: cross-token reshaping, orthogonal transform state,
V rotation, differential blend coefficient, tile padding, adaptive
masking decisions, and weighted reconstruction weights.

This is intentionally a flat dataclass (no nesting) for simplicity of
serialization and fast attribute access during the hot decode path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from rina.utils.transform_pipeline import TransformContext


@dataclass
class Metadata:
    """All encoding parameters needed to decode an EncodedData payload.

    Fields are grouped by purpose:
      - tile_* : tile geometry
      - cross_token_* : cross-token joint encoding (§8.1.5)
      - transform_* : orthogonal transform state (§8.2)
      - v_rotation_* : V orthogonal rotation
      - tile_pad_* : dynamic tile padding (§8.1.10)
      - masking_* : adaptive bit-rate masking (Roadmap 1)
      - differential_* : two-stage residual blending
      - recon_* : weighted reconstruction (§8.1.7)
      - protected : bypass encoding (§8.1.8)
    """

    # ── Tile geometry ────────────────────────────────────────────────────
    tile_size: int = 16

    # ── Cross-token joint encoding (§8.1.5) ─────────────────────────────
    cross_token_group: int = 1
    """Tokens grouped per matrix row before tile encoding (1 = per-token)."""

    original_n_tokens: int = 0
    """Pre-reshape token count; used for cross-token unreshape."""

    cross_token_pad: int = 0
    """Zero-pad tokens added for cross-token divisibility."""

    # ── Orthogonal transform state (§8.2 / Roadmap 3) ───────────────────
    transform_context: Optional[TransformContext] = None
    """Complete forward-transform state (mode, decisions, original shape, pad rows)."""

    use_fwht: bool = False
    """Legacy FWHT flag (DEPRECATED, superseded by transform_context)."""

    # ── V orthogonal rotation ───────────────────────────────────────────
    v_rotation_matrix: Optional[torch.Tensor] = None
    """Square (d_head, d_head) orthogonal rotation for V path."""

    # ── Dynamic tile padding (§8.1.10) ──────────────────────────────────
    tile_pad_counts: Optional[List[int]] = None
    """Per-tile pad row counts from dynamic tile size encoding."""

    # ── Adaptive bit-rate masking (Roadmap 1) ───────────────────────────
    masking_decisions: Optional[List[bool]] = None
    """Per-tile sensitivity flags (True = boosted encoding budget)."""

    # ── Differential residual ───────────────────────────────────────────
    diff_gamma: float = 0.0
    """Blending coefficient for residual stage decode."""

    # ── Weighted reconstruction (§8.1.7) ─────────────────────────────────
    recon_weights: Optional[torch.Tensor] = None
    """Per-step reconstruction weights ``(N_steps,)``."""

    # ── Protected mode (§8.1.8) ──────────────────────────────────────────
    protected: bool = False
    """If True, all data is stored as raw FP16 (no 1-bit encoding)."""

    def has_transform(self) -> bool:
        """True when an orthogonal transform (DCT/DWT/Hybrid/AUTO) is active."""
        if self.transform_context is not None:
            return self.transform_context.is_active
        return False

    def get_n_tokens(self, n_encoded_tokens: int = 0) -> int:
        """Total logical tokens (encoded + buffered)."""
        if self.cross_token_group > 1 and self.original_n_tokens > 0:
            return self.original_n_tokens
        return n_encoded_tokens


__all__ = ["Metadata"]
