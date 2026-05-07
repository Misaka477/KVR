"""
§2 DS-KVCache Core — Incremental Tile-Based Encoding (16×16)
=============================================================

Key design:
  • raw_buffer stores FP16 K/V rows until 16 tokens accumulate
  • On tile-full (len(buf)==16), trigger R.I.N.A 1-bit encode → append to bit-packed store
  • reconstruct_all() = decode(bit_packed_history) + raw_buffer (tail <16)
  • K: 3 steps, V: 5 steps, V orthogonal transform ON
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from rina.config import DSKVCacheConfig

from modules.residual_pursuit import (
    ResidualBases,
    ResidualAlphas,
    encode_matrix,
    decode_from_bases,
    pack_bases,
    unpack_bases,
)

_logger = logging.getLogger(__name__)


@dataclass
class DSKVCacheStore:
    """On-device storage for a single head's DS-encoded K/V cache.

    Incremental mode: raw_buffer holds <16 un-encoded rows; when it hits 16,
    a tile is encoded and appended to the bit-packed store.
    """

    tile_size: int = 16

    # ── Bit-packed encoded tiles (already committed) ──
    bases: Optional[torch.Tensor] = None          # (N_steps, n_tiles_encoded, M_packed) int32
    bases_shape_M: Optional[int] = None
    alphas: Optional[ResidualAlphas] = None        # (N_steps, n_tiles_encoded)
    orig_shape: Optional[Tuple[int, int]] = None   # (n_encoded_tokens, d_head)

    # ── Two-stage residual differential ──
    bases_residual: Optional[torch.Tensor] = None
    bases_shape_M_residual: Optional[int] = None
    alphas_residual: Optional[ResidualAlphas] = None
    diff_gamma: float = 0.0

    # ── Cross-token joint encoding (§8.1.5) ──────────────────────────
    cross_token_group: int = 1
    """Number of tokens grouped per matrix row before tile encoding.
    1 = per-token (original), 4 = 4-token groups."""
    original_n_tokens: Optional[int] = None
    """Pre-reshape token count; used to un-reshape after decode."""

    # ── Orthogonal transform state (V only) ──
    v_rotation_matrix: Optional[torch.Tensor] = None

    # ── Decode cache ──
    full_k_hat: Optional[torch.Tensor] = None

    # ── Weighted reconstruction (§8.1.7) ──
    recon_weights: Optional[torch.Tensor] = None
    """Per-step reconstruction weights w_i for weighted sum:
        recon = sum(w_i * alpha_i * B_i).
    If None, uses uniform w_i=1.0 (standard sum)."""

    # ── Dynamic tile size (§8.1.10) ──
    tile_pad_counts: Optional[List[int]] = None
    """Number of zero-padded rows per encoded tile when dynamic tile size
    triggers early encoding (e.g. tile_size=16 but only 4 tokens available).
    reconstruct_all() strips these rows after decoding."""

    # ── Protected mode (§8.1.8) ────────────────────────────────────────
    protected: bool = False
    """If True, all K/V are stored at FP16 in raw_buffer without any
    1-bit encoding.  Used for critical layers (first/last) where
    quantization error propagates disproportionately."""

    # ── Orthogonal transform mode (§8.2 / Roadmap 3 — DCT/DWT/Hybrid) ──
    use_fwht: bool = False  # DEPRECATED — superseded by transform_mode
    """If True, FWHT was applied during encoding; IFWHT must be applied
    during decode.  Persisted from config so reconstruct_all can
    correctly invert the Walsh-Hadamard transform.
    DEPRECATED: use ``transform_mode`` and ``transform_decisions`` for
    the DCT/DWT/Hybrid engine (§8.2)."""

    transform_mode: str = "none"
    """Transform mode applied during encoding.  One of ``"none"``,
    ``"dct"``, ``"dwt"``, ``"hybrid"``, ``"auto"``, ``"fwht"``.
    Mirrors ``DSKVCacheConfig.transform_mode``; persisted so
    reconstruct_all can apply the correct inverse transform."""

    transform_decisions: Optional[List[str]] = None
    """Per-tile transform decisions (required when transform_mode is
    ``"auto"``, ``"hybrid"``, or ``"dwt"``).  Each element is one of
    ``"dct"``, ``"dwt"``, ``"hybrid"``, or ``"fwht"``.
    Stored alongside bases/alphas so decode can invert exactly."""

    transform_pad_rows: int = 0
    """Number of zero-pad rows added to ensure total elements are
    divisible by tile_size² for 2-D transforms (DCT/DWT/Hybrid).
    reconstruct_all strips these rows after inverse transform."""

    # ── Adaptive Bit-Rate Masking (§A / Roadmap 1) ──────────────────────
    masking_decisions: Optional[List[bool]] = None
    """Per-tile sensitivity decisions from adaptive_masking.
    True = sensitive tile (boosted proj_beta / extra steps applied).
    Persisted for diagnostics; not needed during decode (already baked
    into the stored bases/alphas)."""

    # ── Original matrix shape (before transform reshape) ──
    _original_mat_shape: Optional[Tuple[int, int]] = None
    """Pre-transform matrix shape (N_orig, d_head_orig).
    Set by encode_kv_cache / _encode_and_append_tile so reconstruct_all
    can reshape from tile-space back to the original (N, d_head)."""

    # ── Calibration (noise shaping) ──
    svd_shaper: Optional[Dict] = None

    # ── Incremental buffer (§5) ──
    raw_buffer: Optional[torch.Tensor] = None      # (B, d_head) FP16, B < tile_size
    buffer_full: int = 0

    # ── Stats ──
    memory_bytes: int = 0
    fp16_memory_bytes: int = 0
    compression_ratio: float = 0.0

    # ── Weighted reconstruction (§8.1.7) ────────────────────────────────
    def compute_recon_weights(self, temperature: float = 0.5):
        """Compute energy-based per-step reconstruction weights from alphas.

        Each step's mean |alpha| indicates its contribution to the
        reconstruction.  Steps with higher energy get larger weight:
            w_i = softmax(mean_tiles(|alpha_i|) / temperature).

        Parameters
        ----------
        temperature:
            Softmax temperature.  0.5 = moderate sharpness (default).
            1.0 = near-uniform, 0.1 = nearly argmax.

        Side effects
        ------------
        Sets ``self.recon_weights`` to a ``(N_steps,)`` tensor on the
        same device as alphas.  Call after encoding is complete.
        """
        if self.alphas is None:
            return
        alpha_mean = self.alphas.float().abs().mean(dim=-1)  # (N_steps,)
        if alpha_mean.numel() <= 1:
            return
        weights = torch.softmax(alpha_mean / temperature, dim=0)
        # Normalise so max weight = 1.0 (avoids inflating overall scale)
        weights = weights / weights.max()
        self.recon_weights = weights.to(self.alphas.dtype)

    @property
    def n_tokens(self) -> int:
        """Total logical tokens: encoded + buffered.
        Uses original_n_tokens when cross_token_group > 1."""
        # When cross_token_group > 1, orig_shape was reshaped;
        # original_n_tokens = pre-reshape count (or total for per-tile incremental)
        if self.original_n_tokens is not None and self.cross_token_group > 1:
            return self.original_n_tokens + self.buffer_full
        encoded = self.orig_shape[0] if self.orig_shape is not None else 0
        return encoded + self.buffer_full

    @property
    def n_tiles(self) -> int:
        """Number of encoded tiles in bit-packed store."""
        if self.bases is None:
            return 0
        return self.bases.shape[1]

    # ------------------------------------------------------------------
    # Incremental append (§5 — tile trigger)
    # ------------------------------------------------------------------

    def append_incremental(
        self,
        new_vec: torch.Tensor,
        *,
        cfg: DSKVCacheConfig,
        svd_shaper: Optional[dict] = None,
        v_rotation: Optional[torch.Tensor] = None,
        initial_momentum: Optional[torch.Tensor] = None,
        initial_integrator2: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Add one or more FP16 K/V rows.  When >= tile_size rows accumulate,
        encode a tile and commit to the bit-packed store.

        Protected mode (§8.1.8): raw_buffer grows unbounded, NO tile encoding
        ever triggered.  reconstruct_all() returns the raw buffer as-is.

        Parameters
        ----------
        new_vec: (B, d_head) — 1 or more new token vectors.
        cfg: Pipeline config (heterogeneous n_steps_k / n_steps_v).
        svd_shaper: Optional per-head noise shaper.
        v_rotation: Orthogonal rotation matrix (V path only).
        initial_momentum: Cross-head Σ-Δ momentum from previous head (§8.1.9).
        initial_integrator2: Cross-head second-order integrator from previous head.

        Returns
        -------
        (momentum, integrator2) — final Σ-Δ state after encoding, or (None, None)
        if no tile was encoded in this call.  Pass to next head for cross-head
        error sharing (§8.1.9).
        """
        B, d_head = new_vec.shape
        tile_size = self.tile_size
        is_v = v_rotation is not None  # V path flag: determines n_steps later

        # ── Protected mode: just accumulate raw FP16, never encode ──
        if self.protected:
            if self.raw_buffer is None:
                self.raw_buffer = new_vec.to(torch.float16)
                self.buffer_full = B
            else:
                self.raw_buffer = torch.cat([self.raw_buffer, new_vec.to(torch.float16)], dim=0)
                self.buffer_full += B
            if self.original_n_tokens is None:
                self.original_n_tokens = 0
            self.original_n_tokens += B
            if self.orig_shape is None:
                self.orig_shape = (self.raw_buffer.shape[0], d_head)
            else:
                encoded = self.orig_shape[0]
                self.orig_shape = (encoded + B, d_head)
            return initial_momentum, initial_integrator2

        # ── V-orthogonal: rotate BEFORE storing so raw_buffer is always in rotated space ──
        # This avoids the reconstruct_all bug where the buffer tail (in original space)
        # gets incorrectly un-rotated alongside the rotated encoded tiles.
        if v_rotation is not None:
            new_vec = new_vec.to(torch.float32) @ v_rotation.to(torch.float32)

        # ── First call: initialise raw_buffer ──
        if self.raw_buffer is None:
            self.raw_buffer = new_vec.to(torch.float16)
            self.buffer_full = B
        else:
            self.raw_buffer = torch.cat([self.raw_buffer, new_vec.to(torch.float16)], dim=0)
            self.buffer_full += B

        # ── Determine cross-token group policy (must happen before any tracking) ──
        # §8.1.5: persist on store so reconstruct_all can unreshape correctly.
        # K path uses at most 2-token groups; V path uses full cfg group.
        if self.cross_token_group <= 1:
            self.cross_token_group = (
                max(1, cfg.cross_token_group) if is_v
                else min(2, max(1, cfg.cross_token_group))
            )

        cross_token_group = self.cross_token_group

        # Track original_n_tokens — per-tile mode only; cross-token mode
        # handles its own tracking inside _encode_and_append_tile
        if cross_token_group <= 1:
            if self.original_n_tokens is None:
                self.original_n_tokens = 0
            self.original_n_tokens += B

        # ── Encode tiles while we have enough rows ──
        # Buffer is already in V-rotated space → _encode_and_append_tile must NOT re-rotate.
        #
        # §8.1.5 Cross-token joint encoding:
        # When cross_token_group > 1, accumulate G * tile_size tokens before encoding.
        # Reshape (G*T, d_head) → (T, G*d_head) so each tile spans G tokens,
        # distributing quantisation noise across adjacent tokens instead of
        # accumulating independently per token.

        if cross_token_group > 1:
            # ── Cross-token mode: group G * tile_size tokens per encoding unit ──
            group_trigger = tile_size * cross_token_group
            momentum, integrator2 = initial_momentum, initial_integrator2
            while self.buffer_full >= group_trigger:
                group_tokens = self.raw_buffer[:group_trigger].to(torch.float32)
                # Reshape: (G*T, d_head) → (T, G*d_head)
                group_reshaped = group_tokens.reshape(tile_size, cross_token_group * d_head)
                ret_momentum, ret_integrator2 = self._encode_and_append_tile(
                    group_reshaped, cfg=cfg, svd_shaper=svd_shaper, is_v=is_v,
                    initial_momentum=momentum, initial_integrator2=integrator2,
                )
                momentum, integrator2 = ret_momentum, ret_integrator2
                self.raw_buffer = self.raw_buffer[group_trigger:]
                self.buffer_full = self.raw_buffer.shape[0] if self.raw_buffer.numel() > 0 else 0
                if self.buffer_full == 0:
                    self.raw_buffer = None
                # No padding in incremental mode — exact group_trigger tokens always taken
                self._cross_token_pad = 0
        else:
            # ── Per-tile mode: encode each tile_size block independently ──
            # §8.1.10 Dynamic tile size: when enabled, use the largest
            # power-of-2 ≤ min(buffer_full, tile_size) as the effective
            # tile dimension.  This avoids long raw-buffer residency for
            # the first 15 tokens while preserving Tensor Core alignment.
            momentum, integrator2 = initial_momentum, initial_integrator2

            while True:
                # ── Determine effective tile size for this iteration ──
                if cfg.dynamic_tile_size and self.buffer_full < tile_size:
                    # Find largest power-of-2 ≤ buffer_full that's ≥ min_tile_size
                    dyn_ts = tile_size
                    min_ts = getattr(cfg, 'min_tile_size', 4)
                    while dyn_ts > self.buffer_full and dyn_ts > min_ts:
                        dyn_ts //= 2
                    if dyn_ts < min_ts or self.buffer_full < dyn_ts:
                        break  # not enough tokens for even the minimum tile
                    effective_tile_size = dyn_ts
                elif self.buffer_full >= tile_size:
                    effective_tile_size = tile_size
                else:
                    break  # buffer not full enough and dynamic not applicable

                # Pad to full tile_size for encode_matrix compatibility;
                # decode_from_bases will produce tile_size rows, we strip
                # the zero-padded tail afterwards via tile_pad_counts.
                tile_raw = self.raw_buffer[:effective_tile_size].to(torch.float32)
                pad_rows = tile_size - effective_tile_size
                tile = F.pad(tile_raw, (0, 0, 0, pad_rows), mode='constant', value=0.0)

                ret_momentum, ret_integrator2 = self._encode_and_append_tile(
                    tile, cfg=cfg, svd_shaper=svd_shaper, is_v=is_v,
                    initial_momentum=momentum, initial_integrator2=integrator2,
                )
                momentum, integrator2 = ret_momentum, ret_integrator2

                # Track padding so reconstruct_all can strip it
                if self.tile_pad_counts is None:
                    self.tile_pad_counts = []
                self.tile_pad_counts.append(pad_rows)

                # Keep the remainder
                self.raw_buffer = self.raw_buffer[effective_tile_size:]
                self.buffer_full = self.raw_buffer.shape[0] if self.raw_buffer.numel() > 0 else 0
                if self.buffer_full == 0:
                    self.raw_buffer = None

        return momentum, integrator2

    def _encode_and_append_tile(
        self,
        tile: torch.Tensor,
        *,
        cfg: DSKVCacheConfig,
        svd_shaper: Optional[dict] = None,
        is_v: bool = False,
        initial_momentum: Optional[torch.Tensor] = None,
        initial_integrator2: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Encode a single (tile_size, d_head) tile and concatenate to store.

        Parameters
        ----------
        is_v: True for V-path tiles → uses n_steps_v instead of n_steps_k.
            Tile is assumed already in V-rotated space; no further rotation applied.
        initial_momentum: Cross-head Σ-Δ momentum from previous head (§8.1.9).
        initial_integrator2: Cross-head second-order integrator from previous head.

        Returns
        -------
        (momentum, integrator2) — final Σ-Δ state after encoding this tile.
        """
        tile_size, d_head = tile.shape
        n_steps = cfg.get_n_steps_v() if is_v else cfg.get_n_steps_k()

        # ── Noise-shaping projector ──
        proj_matrix = None
        if cfg.use_noise_shaping and cfg.proj_rank > 0 and cfg.proj_beta > 0:
            if svd_shaper is not None:
                proj_matrix = svd_shaper.get("projector", None)
            # else: skip for incremental (too expensive per-tile)

        # ── Cross-head error sharing: request momentum return ──
        do_cross_head = (
            cfg.cross_head_error_share
            and initial_momentum is not None
            and cfg.order2_gamma > 0
        )

        # ── Primary encode (use same path as bulk: _encode_single_path) ──
        if do_cross_head and initial_momentum.shape[-1] == tile_size ** 2:
            result = _encode_single_path(
                tile,
                n_steps=n_steps,
                cfg=cfg,
                proj_matrix=proj_matrix,
                initial_momentum=initial_momentum,
                initial_integrator2=initial_integrator2,
                return_momentum=True,
            )
            bases, alphas, shape, final_momentum, final_integrator2, tile_xform_decisions, tile_mask_decisions, _pad_rows = result
        else:
            result = _encode_single_path(
                tile,
                n_steps=n_steps,
                cfg=cfg,
                proj_matrix=proj_matrix,
            )
            bases, alphas, shape, tile_xform_decisions, tile_mask_decisions, _pad_rows = result
            final_momentum, final_integrator2 = None, None

        bases_M = bases.shape[-1]
        packed = pack_bases(bases)

        # ── Two-stage residual ──
        bases_res, alphas_res = None, None
        bases_shape_M_res = None
        if cfg.use_differential and cfg.diff_strategy == "residual":
            primary = decode_from_bases(bases, alphas, shape, tile_size=tile_size)
            residual = tile - primary
            bases_res, alphas_res, _res_shape, _ = encode_matrix(
                residual,
                n_steps=cfg.diff_residual_n_steps,
                tile_size=tile_size,
                beta=cfg.beta,
                proj_matrix=None,
                proj_beta=0.0,
                adaptive_eta=False,
            )
            bases_shape_M_res = bases_res.shape[-1]
            bases_res = pack_bases(bases_res)
            alphas_res = alphas_res.to(torch.float16)

        alphas = alphas.to(torch.float16)

        # ── Set diff_gamma for incremental path (was missing → residual never applied) ──
        if cfg.use_differential and bases_res is not None:
            self.diff_gamma = cfg.get_diff_residual_gamma_k() if not is_v else cfg.diff_residual_gamma

        # ── Persist transform mode from config (first tile only) ──
        transform_mode = getattr(cfg, 'transform_mode', 'none')
        if transform_mode and transform_mode not in ("none", "", None):
            if not self.transform_mode or self.transform_mode == "none":
                self.transform_mode = transform_mode

        # ── Concat to existing store ──
        if self.bases is None:
            self.bases = packed                # (N, 1, M_packed)
            self.alphas = alphas               # (N, 1)
            self.orig_shape = (tile_size, d_head)
            # Track original (pre-reshape) token count for cross-token unreshape
            if self.cross_token_group > 1:
                # (tile_size, G*d_head) encodes tile_size * G real tokens
                self.original_n_tokens = tile_size * self.cross_token_group
            if bases_res is not None:
                self.bases_residual = bases_res
                self.bases_shape_M_residual = bases_shape_M_res
                self.alphas_residual = alphas_res
            self.bases_shape_M = bases_M
            # ── Roadmap 3 & 1: store per-tile decisions ──
            if tile_xform_decisions is not None:
                self.transform_decisions = list(tile_xform_decisions)
            if tile_mask_decisions is not None:
                self.masking_decisions = list(tile_mask_decisions)
        else:
            # Concatenate bases along tile dim (dim=1)
            self.bases = torch.cat([self.bases, packed], dim=1)
            self.alphas = torch.cat([self.alphas, alphas], dim=1)
            encoded_tokens = self.orig_shape[0]
            self.orig_shape = (encoded_tokens + tile_size, d_head)
            # Accumulate original (pre-reshape) token count
            if self.cross_token_group > 1:
                if self.original_n_tokens is None:
                    # Transition from per-tile to cross-token: convert existing count
                    self.original_n_tokens = encoded_tokens
                self.original_n_tokens += tile_size * self.cross_token_group
            if bases_res is not None:
                if self.bases_residual is not None:
                    self.bases_residual = torch.cat([self.bases_residual, bases_res], dim=1)
                    self.alphas_residual = torch.cat([self.alphas_residual, alphas_res], dim=1)
                else:
                    self.bases_residual = bases_res
                    self.bases_shape_M_residual = bases_shape_M_res
                    self.alphas_residual = alphas_res
            # ── Roadmap 3 & 1: append per-tile decisions ──
            if tile_xform_decisions is not None:
                if self.transform_decisions is None:
                    self.transform_decisions = list(tile_xform_decisions)
                else:
                    self.transform_decisions.extend(tile_xform_decisions)
            if tile_mask_decisions is not None:
                if self.masking_decisions is None:
                    self.masking_decisions = list(tile_mask_decisions)
                else:
                    self.masking_decisions.extend(tile_mask_decisions)

        # Invalidate decode cache
        self.full_k_hat = None

        # ── Weighted reconstruction (§8.1.7) — recompute after each append ──
        if hasattr(cfg, 'use_recon_weights') and cfg.use_recon_weights:
            self.compute_recon_weights(temperature=getattr(cfg, 'recon_weight_temperature', 0.5))

        return final_momentum, final_integrator2

    # ------------------------------------------------------------------
    # Full reconstruction
    # ------------------------------------------------------------------

    def reconstruct_all(
        self,
        tile_size: int = 16,
        use_differential: bool = True,
    ) -> torch.Tensor:
        """Return (original_n_tokens, d_head) — decoded bit-packed history + raw_buffer tail.
        
        Handles cross-token unreshape when cross_token_group > 1.
        V un-rotation is applied AFTER cross-token unreshape to ensure
        the rotation operates on the correct d_head dimension.

        §A Roadmap 3: Inverse DCT/DWT/Hybrid transform applied after
        primary decode (and residual if active), BEFORE cross-token
        unreshape and V un-rotation.
        """
        decoded_parts = []

        # ── Determine transform inversion policy ────────────────────────
        transform_mode = getattr(self, 'transform_mode', 'none')
        transform_decisions = getattr(self, 'transform_decisions', None)
        do_inverse_transform = (
            transform_mode and transform_mode not in ("none", "", "fwht", None)
        )

        # Decode bit-packed encoded tiles
        if self.bases is not None:
            if self.full_k_hat is not None:
                mat = self.full_k_hat
            else:
                bases = unpack_bases(self.bases)
                if self.bases_shape_M is not None and bases.shape[-1] > self.bases_shape_M:
                    bases = bases[..., :self.bases_shape_M]
                mat_primary = decode_from_bases(
                    bases, self.alphas, self.orig_shape, tile_size=tile_size,
                    recon_weights=self.recon_weights,
                    use_fwht=self.use_fwht,
                )

                if use_differential and self.bases_residual is not None and self.diff_gamma > 0:
                    bases_res = unpack_bases(self.bases_residual)
                    if self.bases_shape_M_residual is not None and bases_res.shape[-1] > self.bases_shape_M_residual:
                        bases_res = bases_res[..., :self.bases_shape_M_residual]
                    mat_residual = decode_from_bases(
                        bases_res, self.alphas_residual, self.orig_shape, tile_size=tile_size,
                        use_fwht=self.use_fwht,
                    )
                    mat = mat_primary + self.diff_gamma * mat_residual
                else:
                    mat = mat_primary

                # ── §A Roadmap 3: Inverse DCT/DWT/Hybrid transform ─────
                if do_inverse_transform:
                    from rina.utils.transforms import apply_inverse_transform, TransformMode
                    # Resolve string → TransformMode enum
                    if isinstance(transform_mode, str):
                        try:
                            tf_mode = TransformMode[transform_mode.upper()]
                        except KeyError:
                            tf_mode = TransformMode(transform_mode)
                    else:
                        tf_mode = transform_mode
                    mat = apply_inverse_transform(
                        mat,
                        mode=tf_mode,
                        tile_size=tile_size,
                        decisions=transform_decisions,
                    )
                    # NOTE: transform_pad_rows is stripped AFTER _original_mat_shape
                    # reshape below (pad rows refer to original (N,d_head) space,
                    # not tile-space (n_tiles, tile_size²)).

                # ── Cross-token unreshape (§8.1.5) ─────────────────────
                if self.cross_token_group > 1 and self.original_n_tokens is not None:
                    d_head = self.orig_shape[1] // self.cross_token_group
                    pad_tokens = getattr(self, '_cross_token_pad', 0)
                    # orig_shape = (N//G padded, G*d_head) after encode
                    # Unflatten: (N_padded, G*d) → (N_padded*G, d) → [:original_n]
                    N_encoded, Gd = mat.shape
                    flat = mat.reshape(N_encoded * self.cross_token_group, d_head)
                    if pad_tokens > 0:
                        flat = flat[:-pad_tokens]
                    mat = flat[:self.original_n_tokens]

                # ── Strip dynamic tile pad rows (§8.1.10) ─────────────
                if self.tile_pad_counts is not None:
                    total_pad = sum(self.tile_pad_counts)
                    if total_pad > 0 and mat.shape[0] >= len(self.tile_pad_counts) * tile_size:
                        n_full_tiles = len(self.tile_pad_counts)
                        mat_tiles = mat[:n_full_tiles * tile_size].reshape(
                            n_full_tiles, tile_size, -1,
                        )
                        keep_chunks = []
                        for i in range(n_full_tiles):
                            keep = tile_size - self.tile_pad_counts[i]
                            if keep > 0:
                                keep_chunks.append(mat_tiles[i, :keep])
                        if keep_chunks:
                            mat = torch.cat(keep_chunks, dim=0)
                        else:
                            mat = mat[:0]  # empty (all padding) → no rows

                self.full_k_hat = mat  # cache for next call
            decoded_parts.append(mat)

        # Append raw buffer tail
        if self.raw_buffer is not None and self.buffer_full > 0:
            tail = self.raw_buffer.to(torch.float32)
            # ── Apply inverse transform to raw buffer tail as well ──────
            if do_inverse_transform:
                tail_padded = _pad_for_tile_inversion(tail, tile_size)
                from rina.utils.transforms import apply_inverse_transform, TransformMode
                # Resolve string → TransformMode enum
                if isinstance(transform_mode, str):
                    try:
                        tf_mode = TransformMode[transform_mode.upper()]
                    except KeyError:
                        tf_mode = TransformMode(transform_mode)
                else:
                    tf_mode = transform_mode
                tail_transformed = apply_inverse_transform(
                    tail_padded,
                    mode=tf_mode,
                    tile_size=tile_size,
                    decisions=None,  # tail tiles get default inverse
                )
                tail = tail_transformed[:tail.shape[0]]
            decoded_parts.append(tail)

        if not decoded_parts:
            return torch.empty(0, 0)

        result = torch.cat(decoded_parts, dim=0)

        # ── V un-rotation (applied after cross-token unreshape) ──
        if self.v_rotation_matrix is not None:
            R_T = self.v_rotation_matrix.T.to(result.dtype)
            # Safety: only apply if rotation matrix dimension matches result
            # (may be disabled when cross_token_group > 1 changes effective tile width)
            if R_T.shape[-1] == result.shape[-1]:
                result = result @ R_T
            else:
                _logger.debug(
                    "V rotation shape %s does not match result %s; skipping un-rotation. "
                    "This is expected when cross_token_group > 1.",
                    R_T.shape, result.shape,
                )

        # ── Restore original matrix shape (undo transform reshape) ──
        # When a 2-D transform (DCT/DWT/Hybrid) is used, the encoding
        # reshapes (N_orig, d_head) → (n_tiles, tile_size²).  After
        # inverse transform the result may still be in tile-space.
        # _original_mat_shape records the pre-transform shape so we can
        # reshape back.
        orig_shape = getattr(self, '_original_mat_shape', None)
        if orig_shape is not None:
            N_orig, d_orig = orig_shape
            # Only reshape if the current result is tile-space (N*M != N_orig*d_orig)
            if result.numel() >= N_orig * d_orig and result.shape != (N_orig, d_orig):
                # Crop excess (pad rows) then reshape
                excess = result.numel() - N_orig * d_orig
                if excess > 0:
                    # Remove trailing pad rows
                    result = result.flatten()[:N_orig * d_orig].reshape(N_orig, d_orig)
                else:
                    result = result.reshape(N_orig, d_orig)
            elif result.numel() == N_orig * d_orig and result.shape != (N_orig, d_orig):
                result = result.reshape(N_orig, d_orig)

        # ── Strip zero-pad rows added for 2-D transform tile alignment ──
        # Only applies when _original_mat_shape reshape did NOT already
        # crop to the exact N_orig*d_head element count (which already
        # implicitly strips the pad rows).
        if orig_shape is None:
            transform_pad_rows = getattr(self, 'transform_pad_rows', 0)
            if transform_pad_rows > 0 and result.shape[0] >= transform_pad_rows:
                result = result[:-transform_pad_rows]

        return result

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def update_stats(self):
        """Recalculate memory footprint."""
        d_head = self.orig_shape[1] if self.orig_shape is not None else 64
        total_tokens = self.n_tokens
        self.fp16_memory_bytes = total_tokens * d_head * 2

        total = 0
        for packed_attr in ("bases", "bases_residual"):
            tensor = getattr(self, packed_attr, None)
            if tensor is not None:
                total += (tensor.numel() * 32) // 8
        for fp16_attr in ("alphas", "alphas_residual"):
            tensor = getattr(self, fp16_attr, None)
            if tensor is not None:
                total += (tensor.numel() * 16) // 8
        if self.raw_buffer is not None and self.buffer_full > 0:
            total += self.buffer_full * d_head * 2

        self.memory_bytes = total
        self.compression_ratio = self.fp16_memory_bytes / (total + 1e-12)


# ══════════════════════════════════════════════════════════════════════════════
# Legacy bulk encoder (for eval scripts)
# ══════════════════════════════════════════════════════════════════════════════


def _encode_single_path(
    mat: torch.Tensor,
    n_steps: int,
    cfg: DSKVCacheConfig,
    proj_matrix: Optional[torch.Tensor] = None,
    initial_momentum: Optional[torch.Tensor] = None,
    initial_integrator2: Optional[torch.Tensor] = None,
    return_momentum: bool = False,
) -> Tuple:
    """Encode a single matrix path with optional cross-head momentum.

    Returns extend to ``(bases, alphas, orig_shape, momentum, integrator2,
    transform_decisions, masking_decisions)`` when ``return_momentum=True``
    (§8.1.9 cross-head error sharing).

    Roadmaps wired here:
      §A Roadmap 1 — Adaptive Bit-Rate Masking (per-tile outlier/anchor detection)
      §A Roadmap 3 — DCT/DWT/Hybrid orthogonal transform engine
    """
    tile_size = cfg.tile_size
    tile_d = tile_size ** 2
    per_tile_proj = proj_matrix

    if proj_matrix is not None and proj_matrix.shape[-1] != tile_d:
        proj_matrix = None
        per_tile_proj = None

    # ── Pad mat so total elements are divisible by tile_size² for 2-D transforms ──
    N_orig, d_head_orig = mat.shape
    transform_pad_rows = 0
    transform_mode = getattr(cfg, 'transform_mode', 'none')
    if transform_mode and transform_mode not in ("none", "", None, "fwht"):
        total_elems = N_orig * d_head_orig
        if total_elems % tile_d != 0:
            needed_elems = ((total_elems + tile_d - 1) // tile_d) * tile_d
            pad_elems = needed_elems - total_elems
            transform_pad_rows = (pad_elems + d_head_orig - 1) // d_head_orig
            mat = F.pad(mat, (0, 0, 0, transform_pad_rows), mode='constant', value=0.0)

    # ── §A Roadmap 3: Orthogonal transform BEFORE encoding ────────────────
    transform_decisions = None
    mat_enc = mat
    if transform_mode and transform_mode not in ("none", "", None):
        from rina.utils.transforms import apply_transform, TransformMode
        # Resolve string → TransformMode enum
        if isinstance(transform_mode, str):
            try:
                tf_mode = TransformMode[transform_mode.upper()]
            except KeyError:
                tf_mode = TransformMode(transform_mode)
        else:
            tf_mode = transform_mode
        mat_enc, transform_decisions = apply_transform(
            mat_enc,
            mode=tf_mode,
            tile_size=tile_size,
            smooth_threshold=getattr(cfg, 'transform_smooth_threshold', 0.05),
            outlier_threshold=getattr(cfg, 'transform_outlier_threshold', 3.0),
        )

    # ── §A Roadmap 1: Adaptive Bit-Rate Masking ──────────────────────────
    adaptive_masking = getattr(cfg, 'adaptive_masking', False)
    mask_decisions = None
    n_steps_per_tile = n_steps  # default uniform
    if adaptive_masking and transform_decisions is None:
        # Compute sensitivity per tile from raw mat (before any transform)
        from rina.utils.transforms import compute_tile_diagnostics
        # mat may not be tile-aligned — pad to tile_size² boundary
        flat_diag = mat.reshape(-1)
        pad_diag = (tile_d - flat_diag.numel() % tile_d) % tile_d
        if pad_diag > 0:
            flat_diag = F.pad(flat_diag, (0, pad_diag))
        tiled_diag = flat_diag.reshape(-1, tile_size, tile_size)
        variances, max_abs_vals = compute_tile_diagnostics(tiled_diag)
        stds = variances.sqrt().clamp_min(1e-8)
        outlier_thr = getattr(cfg, 'mask_outlier_threshold', 3.0)
        mask_decisions = (max_abs_vals > outlier_thr * stds).tolist()
        # Per-tile extra steps for sensitive tiles
        n_boost = getattr(cfg, 'mask_n_steps_boost', 1)
        if any(mask_decisions) and n_boost > 0:
            # We handle per-tile n_steps by encoding sensitive tiles with extra steps
            # Simple approach: encode all tiles with base n_steps, then re-encode
            # sensitive tiles with extra steps
            pass  # handled in encode_matrix via per-tile adaptive logic
    if adaptive_masking and transform_mode not in ("none", "", None):
        # When transform is active, compute mask on transformed tiles
        from rina.utils.transforms import compute_tile_diagnostics
        tiled_diag = mat_enc.reshape(-1, tile_size, tile_size)
        variances, max_abs_vals = compute_tile_diagnostics(tiled_diag)
        stds = variances.sqrt().clamp_min(1e-8)
        outlier_thr = getattr(cfg, 'mask_outlier_threshold', 3.0)
        mask_decisions = (max_abs_vals > outlier_thr * stds).tolist()

    # ── Build encode kwargs ──────────────────────────────────────────────
    encode_kwargs = dict(
        tile_size=tile_size,
        beta=cfg.beta,
        proj_matrix=per_tile_proj,
        proj_beta=cfg.proj_beta if per_tile_proj is not None else 0.0,
        adaptive_eta=cfg.adaptive_eta,
        order2_gamma=cfg.order2_gamma,
        order2_c1=cfg.order2_c1,
        order2_c2=cfg.order2_c2,
        initial_momentum=initial_momentum,
        initial_integrator2=initial_integrator2,
        return_momentum=return_momentum,
        use_fwht=cfg.use_fwht if transform_mode in ("none", "", None, "fwht") else False,
        zero_mean_integrator2=cfg.zero_mean_integrator2,
    )

    # §A Roadmap 1: Adaptive Bit-Rate Masking (§8.2.1)
    # Forward adaptive_masking + all per-tile boost config to encode_matrix.
    # encode_matrix's adaptive_masking branch handles per-tile sensitivity
    # internally using tile diagnostics (variance/max-abs), so we don't
    # need to precompute mask_decisions here — just pass the config.
    if adaptive_masking:
        encode_kwargs['adaptive_masking'] = True
        encode_kwargs['mask_smooth_threshold'] = getattr(cfg, 'mask_smooth_threshold', 0.05)
        encode_kwargs['mask_outlier_threshold'] = getattr(cfg, 'mask_outlier_threshold', 3.0)
        encode_kwargs['mask_proj_beta_boost'] = getattr(cfg, 'mask_proj_beta_boost', 0.5)
        encode_kwargs['mask_n_steps_boost'] = getattr(cfg, 'mask_n_steps_boost', 1)

    if cfg.adaptive_n:
        from modules.residual_pursuit import adaptive_encode_matrix
        n_extra = max(cfg.n_upper_bound - n_steps, 2)
        # adaptive_encode_matrix doesn't accept momentum/tracker kwargs
        adaptive_kwargs = {
            k: v for k, v in encode_kwargs.items()
            if k not in ("initial_momentum", "initial_integrator2", "return_momentum",
                         "use_fwht", "zero_mean_integrator2")
        }
        result = adaptive_encode_matrix(
            mat_enc,
            n_steps_base=n_steps,
            n_steps_extra=n_extra,
            energy_threshold_ratio=cfg.energy_threshold_factor,
            **adaptive_kwargs,
        )
        if return_momentum:
            bases, alphas, _, orig_shape, momentum, integrator2 = result
            return bases, alphas, orig_shape, momentum, integrator2, transform_decisions, mask_decisions
        bases, alphas, _, orig_shape = result
        return bases, alphas, orig_shape, transform_decisions, mask_decisions
    else:
        result = encode_matrix(mat_enc, n_steps=n_steps, **encode_kwargs)
        if return_momentum:
            # encode_matrix returns (bases, alphas, orig_shape, xform_dec, momentum, integrator2)
            bases, alphas, orig_shape, _inner_xform, momentum, integrator2 = result
            return bases, alphas, orig_shape, momentum, integrator2, transform_decisions, mask_decisions, transform_pad_rows
        # encode_matrix returns (bases, alphas, orig_shape, xform_dec)
        bases, alphas, orig_shape, _inner_xform = result
        return bases, alphas, orig_shape, transform_decisions, mask_decisions, transform_pad_rows


def _build_v_rotation(k: torch.Tensor) -> Optional[torch.Tensor]:
    """Build a square (d_head × d_head) orthogonal transform from K's SVD.

    Using ``full_matrices=True`` guarantees a rotation that preserves
    dimensionality — critical because V will later be multiplied by this
    matrix before encoding, and reconstructed V must stay shape (N, d_head).

    For K ∈ ℝ^{N×d_head} with N < d_head, ``full_matrices=False`` would
    return Vt ∈ ℝ^{N×d_head}, giving a (d_head × N) rotation that collapses
    the d_head dimension down to N — that destroys information.
    """
    _, d_head = k.shape
    if d_head < 8:
        return None
    try:
        _, _, Vt = torch.linalg.svd(k.float(), full_matrices=True)
        # Vt is (d_head, d_head) — perfect orthogonal rotation
        return Vt.T.to(k.dtype)
    except Exception:
        _logger.warning("SVD for V rotation failed — falling back to identity")
        return None


def _pad_for_tile_inversion(
    mat: torch.Tensor,
    tile_size: int,
) -> torch.Tensor:
    """Zero-pad mat so its token count is divisible by tile_size.

    Used when applying inverse DCT/DWT/Hybrid to raw buffer tail
    (which typically has < tile_size rows).  Padding guarantees
    tile-aligned reshape for per-tile inverse transform.
    """
    N, d = mat.shape
    if N % tile_size == 0:
        return mat
    pad = tile_size - (N % tile_size)
    return F.pad(mat, (0, 0, 0, pad), mode='constant', value=0.0)


def _reshape_for_cross_token(
    mat: torch.Tensor,
    group: int,
) -> Tuple[torch.Tensor, int]:
    """Reshape (N, d) → (N//G, G*d) for cross-token joint encoding.
    
    Returns (reshaped, pad_tokens).  pad_tokens=0 if N divisible by G.
    """
    if group <= 1:
        return mat, 0
    N, d = mat.shape
    pad = (group - (N % group)) % group
    if pad > 0:
        mat = F.pad(mat, (0, 0, 0, pad), mode='constant', value=0.0)
        N += pad
    return mat.reshape(N // group, group * d), pad


def encode_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: DSKVCacheConfig,
    svd_shaper: Optional[dict] = None,
    protected: bool = False,
) -> Tuple[DSKVCacheStore, DSKVCacheStore]:
    """Bulk-encode K/V matrices (used by eval scripts).
    
    Parameters
    ----------
    protected:
        If True, store K/V raw at FP16 with zero encoding loss.
        Used for critical layers (first/last) where quantization error
        propagates disproportionately through the transformer stack.
    """
    assert k.ndim == 2 and v.ndim == 2
    assert k.shape == v.shape
    n_tokens_original, d_head = k.shape

    # ── Protected mode: store raw FP16, skip all encoding ───────────
    if protected:
        k_store = DSKVCacheStore(
            tile_size=cfg.tile_size,
            protected=True,
            raw_buffer=k.to(torch.float16),
            buffer_full=k.shape[0],
            orig_shape=k.shape,
            cross_token_group=1,
        )
        v_store = DSKVCacheStore(
            tile_size=cfg.tile_size,
            protected=True,
            raw_buffer=v.to(torch.float16),
            buffer_full=v.shape[0],
            orig_shape=v.shape,
            cross_token_group=1,
        )
        k_store.update_stats()
        v_store.update_stats()
        return k_store, v_store

    # ── V orthogonal transform: apply BEFORE cross-token reshape ──
    v_rotation = None
    if cfg.v_orthogonal_transform:
        v_rotation = _build_v_rotation(k)
    v_rotated = v @ v_rotation if v_rotation is not None else v

    # ── Cross-token joint encoding: K and V use different groups ──
    # §8.1.5: K has fewer steps (4) → can't afford row-resolution loss
    # from grouping, so cap at 2.  V (8 steps) can use the full group.
    group_v = max(1, cfg.cross_token_group)
    group_k = min(2, group_v)  # K gets at most 2-token grouping
    k_enc, k_pad = _reshape_for_cross_token(k, group_k)
    v_enc, v_pad = _reshape_for_cross_token(v_rotated, group_v)

    n_steps_k = cfg.get_n_steps_k()
    n_steps_v = cfg.get_n_steps_v()
    transform_mode = getattr(cfg, 'transform_mode', 'none')

    proj_matrix = None
    if cfg.use_noise_shaping and cfg.proj_rank > 0 and cfg.proj_beta > 0:
        if svd_shaper is not None:
            proj_matrix = svd_shaper.get("projector", None)
        else:
            from modules.svd_noise_shaping import compute_per_head_nullspace_projectors
            projectors = compute_per_head_nullspace_projectors(k.unsqueeze(0), energy_ratio=0.95)
            proj_matrix = projectors[0][0] if 0 in projectors else None

    k_result = _encode_single_path(k_enc, n_steps_k, cfg, proj_matrix)
    v_result = _encode_single_path(v_enc, n_steps_v, cfg, proj_matrix)
    k_bases, k_alphas, k_shape, k_xform_decisions, k_mask_decisions, k_pad_rows = k_result
    v_bases, v_alphas, v_shape, v_xform_decisions, v_mask_decisions, v_pad_rows = v_result

    # ── Two-stage residual differential ───────────────────────────────
    k_bases_res, k_alphas_res, k_shape_res = None, None, None
    v_bases_res, v_alphas_res, v_shape_res = None, None, None

    if cfg.use_differential and cfg.diff_strategy == "residual":
        k_hat_primary = decode_from_bases(k_bases, k_alphas, k_shape, tile_size=cfg.tile_size,
                                          use_fwht=cfg.use_fwht)
        # ── §A Roadmap 3: Inverse transform BEFORE residual alignment ──
        # decode_from_bases returns tile-space (n_tiles, tile_size²) when
        # DCT/DWT/Hybrid was applied; k_enc is in original (N, d_head) space.
        # Apply inverse transform to map k_hat_primary back to original shape.
        if transform_mode and transform_mode not in ("none", "", None, "fwht"):
            from rina.utils.transforms import apply_inverse_transform, TransformMode
            try:
                tf_mode = TransformMode[transform_mode.upper()]
            except KeyError:
                tf_mode = TransformMode(transform_mode)
            k_hat_primary = apply_inverse_transform(
                k_hat_primary, mode=tf_mode, tile_size=cfg.tile_size,
                decisions=k_xform_decisions,
                original_shape=(k_enc.shape[0], k_enc.shape[1]),
            )
        k_residual = k_enc - k_hat_primary
        k_bases_res, k_alphas_res, k_shape_res, _ = encode_matrix(
            k_residual, n_steps=cfg.diff_residual_n_steps, tile_size=cfg.tile_size,
            beta=cfg.beta, proj_matrix=None, proj_beta=0.0, adaptive_eta=False,
        )

        v_hat_primary = decode_from_bases(v_bases, v_alphas, v_shape, tile_size=cfg.tile_size,
                                          use_fwht=cfg.use_fwht)
        # ── §A Roadmap 3: Inverse transform for V residual ──
        if transform_mode and transform_mode not in ("none", "", None, "fwht"):
            from rina.utils.transforms import apply_inverse_transform, TransformMode
            try:
                tf_mode = TransformMode[transform_mode.upper()]
            except KeyError:
                tf_mode = TransformMode(transform_mode)
            v_hat_primary = apply_inverse_transform(
                v_hat_primary, mode=tf_mode, tile_size=cfg.tile_size,
                decisions=v_xform_decisions,
                original_shape=(v_enc.shape[0], v_enc.shape[1]),
            )
        v_residual = v_enc - v_hat_primary
        v_bases_res, v_alphas_res, v_shape_res, _ = encode_matrix(
            v_residual, n_steps=cfg.diff_residual_n_steps, tile_size=cfg.tile_size,
            beta=cfg.beta, proj_matrix=None, proj_beta=0.0, adaptive_eta=False,
        )

    k_bases_M = k_bases.shape[-1]
    v_bases_M = v_bases.shape[-1]

    transform_mode = getattr(cfg, 'transform_mode', 'none')
    k_store = DSKVCacheStore(
        tile_size=cfg.tile_size,
        bases=pack_bases(k_bases),
        bases_shape_M=k_bases_M,
        alphas=k_alphas.to(torch.float16),
        orig_shape=k_shape,
        svd_shaper=svd_shaper,
        bases_residual=pack_bases(k_bases_res) if k_bases_res is not None else None,
        bases_shape_M_residual=k_bases_res.shape[-1] if k_bases_res is not None else None,
        alphas_residual=k_alphas_res.to(torch.float16) if k_alphas_res is not None else None,
        diff_gamma=cfg.get_diff_residual_gamma_k() if cfg.use_differential else 0.0,
        cross_token_group=group_k,
        original_n_tokens=n_tokens_original,
        use_fwht=cfg.use_fwht if transform_mode in ("none", "", None, "fwht") else False,
        transform_mode=transform_mode if transform_mode else "none",
        transform_decisions=k_xform_decisions if k_xform_decisions is not None else None,
        masking_decisions=k_mask_decisions if k_mask_decisions is not None else None,
        transform_pad_rows=k_pad_rows,
    )
    # Store pad tokens for unreshape
    k_store._cross_token_pad = k_pad  # type: ignore
    k_store._original_mat_shape = (n_tokens_original, d_head)

    v_store = DSKVCacheStore(
        tile_size=cfg.tile_size,
        bases=pack_bases(v_bases),
        bases_shape_M=v_bases_M,
        alphas=v_alphas.to(torch.float16),
        orig_shape=v_shape,
        svd_shaper=svd_shaper,
        bases_residual=pack_bases(v_bases_res) if v_bases_res is not None else None,
        bases_shape_M_residual=v_bases_res.shape[-1] if v_bases_res is not None else None,
        alphas_residual=v_alphas_res.to(torch.float16) if v_alphas_res is not None else None,
        diff_gamma=cfg.diff_residual_gamma if cfg.use_differential else 0.0,
        v_rotation_matrix=v_rotation,
        cross_token_group=group_v,
        original_n_tokens=n_tokens_original,
        use_fwht=cfg.use_fwht if transform_mode in ("none", "", None, "fwht") else False,
        transform_mode=transform_mode if transform_mode else "none",
        transform_decisions=v_xform_decisions if v_xform_decisions is not None else None,
        masking_decisions=v_mask_decisions if v_mask_decisions is not None else None,
        transform_pad_rows=v_pad_rows,
    )
    v_store._cross_token_pad = v_pad  # type: ignore
    v_store._original_mat_shape = (n_tokens_original, d_head)

    # ── Weighted reconstruction (§8.1.7): compute per-step weights from alphas ──
    if cfg.use_recon_weights:
        k_store.compute_recon_weights(temperature=cfg.recon_weight_temperature)
        v_store.compute_recon_weights(temperature=cfg.recon_weight_temperature)

    k_store.update_stats()
    v_store.update_stats()

    if cfg.verbose:
        _log_diagnostics("K", k, k_store, cfg)
        _log_diagnostics("V", v, v_store, cfg)

    return k_store, v_store


# ══════════════════════════════════════════════════════════════════════════════
# Legacy decode (for eval scripts)
# ══════════════════════════════════════════════════════════════════════════════


def decode_kvcache_store(
    store: DSKVCacheStore,
    tile_size: int = 16,
    use_differential: bool = True,
) -> torch.Tensor:
    """Legacy decode path — delegates to reconstruct_all()."""
    return store.reconstruct_all(tile_size=tile_size, use_differential=use_differential)


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ══════════════════════════════════════════════════════════════════════════════


def _log_diagnostics(
    tag: str,
    original: torch.Tensor,
    store: DSKVCacheStore,
    cfg: DSKVCacheConfig,
):
    approx = store.reconstruct_all(cfg.tile_size, cfg.use_differential)

    mse = F.mse_loss(approx.float(), original.float()).item()
    signal_power = (original.float() ** 2).mean().item()
    noise_power = ((original.float() - approx.float()) ** 2).mean().item()
    snr_db = 10 * math.log10(signal_power / (noise_power + 1e-12))

    cos_sim = F.cosine_similarity(
        approx.float().flatten().unsqueeze(0),
        original.float().flatten().unsqueeze(0),
    ).item()

    original_bytes = original.element_size() * original.numel()
    comp_ratio = original_bytes / (store.memory_bytes + 1e-12)

    _logger.info(
        f"[DS-KVCache {tag}] tokens={store.n_tokens}, "
        f"tiles={store.n_tiles}, "
        f"bases={store.bases.shape[0] if store.bases is not None else 0} steps, "
        f"MSE={mse:.6f}, SNR={snr_db:.2f}dB, "
        f"CosSim={cos_sim:.6f}, "
        f"CompressRatio={comp_ratio:.1f}x ({original_bytes}→{store.memory_bytes} bytes)"
    )