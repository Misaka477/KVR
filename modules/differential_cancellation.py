"""
§7 Differential Noise Cancellation — DS-KVCache Component Three
================================================================

Integrates differential noise cancellation into the DS-KVCache
encoding pipeline.  Pairs attention heads (or applies dual-path
encoding to the same head) and averages the two 1-bit encodings
to cancel partially independent quantisation errors.

This module wraps the core differential_encode_decode from
residual_pursuit.py and integrates it with SVDNoiseShaper
from svd_noise_shaping.py for noise-shaped differential paths.

Key concepts
------------
- Dual-path encoding: encode K (or V) twice with different strategies
  (momentum perturbation, extra-step, adaptive η shift) and average
- Head pairing: heads with complementary Q distributions are paired
  so that one acts as the "differential positive" and the other as
  the "differential negative" — the downstream attention acts as
  the averaging node.
- Strategy diversity: the two paths must differ sufficiently for
  their quantisation errors to be decorrelated (→ positive NRR).
  Momentum-perturbation is the default; extra-step and η-shift are
  alternatives.

Reference: R.I.N.A / DS-KVCache Whitepaper §7 (previously §8.2)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Perturbation strategies
# ---------------------------------------------------------------------------

class PerturbationStrategy:
    """Namespaced perturbation strategy identifiers."""

    MOMENTUM_SHIFT = "momentum_shift"   # β_B = β + Δβ
    EXTRA_STEP = "extra_step"            # N_B = N + 1
    ETA_SHIFT = "eta_shift"              # η_B = η + Δη
    SIGN_COMPLEMENT = "sign_complement"  # Path B uses sign_flip = -1.0 (⚠️ risky)
    RESIDUAL = "residual"                # Path B encodes Path A's residual (1-step, additive)

    _ALL = {MOMENTUM_SHIFT, EXTRA_STEP, ETA_SHIFT, SIGN_COMPLEMENT, RESIDUAL}

    @classmethod
    def is_valid(cls, name: str) -> bool:
        return name in cls._ALL


# ---------------------------------------------------------------------------
# Main differential cancellation module
# ---------------------------------------------------------------------------


class DifferentialCanceller(nn.Module):
    """Dual-path differential encoder for KV-cache 1-bit compression.

    Usage::

        canceller = DifferentialCanceller(
            n_heads=32, d_head=128,
            strategy="momentum_shift",
            perturbation_strength=0.15,
            noise_shaper=svd_shaper,   # optional
        )

        # Encode K for a batch of tokens per head
        result = canceller.encode_kv_tensor(
            tensor=K_h,  # (seq_len, d_head) or (batch, d_head)
            head_idx=0,
            token_slice=...,  # for incremental decode
        )

    Parameters
    ----------
    n_heads:
        Number of attention heads.
    d_head:
        Dimension per head.
    strategy:
        Perturbation strategy for the second encoding path:
        - ``"momentum_shift"``: Path B uses β + perturbation_strength.
        - ``"extra_step"``: Path B uses N + 1 steps.
        - ``"eta_shift"``: Path B increases proj_beta by perturbation_strength.
        - ``"sign_complement"``: Path B flips sign (use with caution).
    perturbation_strength:
        Magnitude of the perturbation.  For momentum_shift, 0.10-0.20
        works well.  For extra_step, it's the number of extra steps.
        For eta_shift, 0.10-0.30.
    noise_shaper:
        Optional ``SVDNoiseShaper`` instance.  If provided, both encoding
        paths receive the same noise-shaping projection (per-head for
        per-head encoding, or shared for GQA).
    n_steps:
        Base number of 1-bit bases (N).  Default 5.
    tile_size:
        Tile dimension for block-wise encoding.  Default 16.
    beta:
        Base momentum coefficient.  Default 0.0 (first-order Σ-Δ).
    proj_beta:
        Base noise-shaping strength ∈ [0, 1].  Default 0.0 = off.
    adaptive_eta:
        If True, ramps proj_beta from 0 → peak (§8.1.1).
    order2_gamma:
        Second-order Σ-Δ coupling strength (§8.1.2).
    order2_c1, order2_c2:
        Gain coefficients for integrator cascading.
    head_pairs:
        Optional list of ``(h_a, h_b)`` tuples for explicit head pairing.
        If provided, ``head_idx`` in encode calls is resolved to a pair.
        This enables *asymmetric differential* where two different heads'
        K tensors (with complementary Q subspaces) cancel each other's
        quantisation noise through shared K-cache readout.
    """

    def __init__(
        self,
        n_heads: int,
        d_head: int,
        *,
        strategy: str = PerturbationStrategy.MOMENTUM_SHIFT,
        perturbation_strength: float = 0.15,
        noise_shaper: Optional[nn.Module] = None,
        n_steps: int = 5,
        tile_size: int = 16,
        beta: float = 0.0,
        proj_beta: float = 0.0,
        adaptive_eta: bool = False,
        eta_peak_step: Optional[int] = None,
        order2_gamma: float = 0.0,
        order2_c1: float = 1.0,
        order2_c2: float = 0.5,
        head_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        super().__init__()
        if not PerturbationStrategy.is_valid(strategy):
            raise ValueError(
                f"Unknown perturbation strategy '{strategy}'. "
                f"Choose from {PerturbationStrategy._ALL}"
            )

        self.n_heads = n_heads
        self.d_head = d_head
        self.strategy = strategy
        self.perturbation_strength = perturbation_strength
        self.noise_shaper = noise_shaper
        self.n_steps = n_steps
        self.tile_size = tile_size
        self.beta = beta
        self.proj_beta = proj_beta
        self.adaptive_eta = adaptive_eta
        self.eta_peak_step = eta_peak_step
        self.order2_gamma = order2_gamma
        self.order2_c1 = order2_c1
        self.order2_c2 = order2_c2

        self._head_pairs: Dict[int, int] = {}
        if head_pairs:
            for ha, hb in head_pairs:
                self._head_pairs[ha] = hb
                self._head_pairs[hb] = ha

    # ------------------------------------------------------------------
    # Head pairing
    # ------------------------------------------------------------------

    def find_partner(self, head_idx: int) -> int:
        """Return the partner head index, or *head_idx* if unpaired."""
        return self._head_pairs.get(head_idx, head_idx)

    def is_paired(self, head_idx: int) -> bool:
        return head_idx in self._head_pairs

    # ------------------------------------------------------------------
    # Perturbation methods
    # ------------------------------------------------------------------

    def _get_path_b_params(self) -> dict:
        """Return the parameter modifications for Path B."""
        s = self.strategy
        ps = self.perturbation_strength

        if s == PerturbationStrategy.MOMENTUM_SHIFT:
            return {"beta_B": self.beta + ps}
        elif s == PerturbationStrategy.EXTRA_STEP:
            extra = max(1, int(math.ceil(ps)))
            return {"n_steps_B": self.n_steps + extra}
        elif s == PerturbationStrategy.ETA_SHIFT:
            return {"proj_beta_B": min(1.0, self.proj_beta + ps)}
        elif s == PerturbationStrategy.SIGN_COMPLEMENT:
            return {"sign_flip_B": -1.0}
        elif s == PerturbationStrategy.RESIDUAL:
            return {"n_steps_B": 1}  # Path B encodes only the residual
        return {}

    def _resolve_noise_shaping_params(
        self, head_idx: int
    ) -> Tuple[Optional[torch.Tensor], float]:
        """Return (proj_matrix, proj_beta) for the given head.

        If SVDNoiseShaper is available, uses its per-head nullspace
        projector.  Otherwise falls back to the module-level *proj_beta*
        with ``proj_matrix=None`` (uniform noise shaping without
        directionality).
        """
        if self.noise_shaper is not None and self.noise_shaper.is_calibrated:
            proj_matrix, _k = self.noise_shaper.get_projector(head_idx)
            return proj_matrix, self.proj_beta

        return None, self.proj_beta

    # ------------------------------------------------------------------
    # Public encode API
    # ------------------------------------------------------------------

    def encode_dual(
        self,
        w: torch.Tensor,
        head_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Dual-path encode a 2-D matrix (K or V for one head / token group).

        This is the core operation: encodes *w* twice with different
        parameters, then returns the averaged reconstruction.

        Parameters
        ----------
        w:
            ``(rows, cols)`` — K or V slice to encode.  For a full
            sequence K matrix, ``(seq_len, d_head)``.  For incremental
            decode, ``(n_new_tokens, d_head)``.
        head_idx:
            Attention head index for noise-shaping lookup.

        Returns
        -------
        w_diff:
            ``(rows, cols)`` — differential reconstruction (averaged).
        diag:
            Diagnostic dictionary with NRR, cross-correlation, MSE,
            cosine similarity, and SNR for both paths and differential.
        """
        from .residual_pursuit import differential_encode_decode

        proj_matrix, proj_beta = self._resolve_noise_shaping_params(head_idx)

        path_b_params = self._get_path_b_params()
        beta_b = path_b_params.get("beta_B", self.beta)
        n_steps_b = path_b_params.get("n_steps_B", self.n_steps)
        proj_beta_b = path_b_params.get("proj_beta_B", proj_beta)
        sign_flip_b = path_b_params.get("sign_flip_B", +1.0)

        # If Path B uses a different N or sign_flip, we need a custom
        # two-path encode (differential_encode_decode only supports
        # momentum_shift and eta_shift natively via beta_B).
        # For extra_step, sign_complement, and residual we delegate to
        # _custom_dual_encode below.
        use_custom = (
            self.strategy == PerturbationStrategy.EXTRA_STEP
            or self.strategy == PerturbationStrategy.SIGN_COMPLEMENT
            or self.strategy == PerturbationStrategy.RESIDUAL
        )

        if use_custom:
            return self._custom_dual_encode(
                w,
                head_idx=head_idx,
                n_steps_b=n_steps_b,
                sign_flip_b=sign_flip_b,
                proj_matrix=proj_matrix,
                proj_beta=proj_beta,
                proj_beta_b=proj_beta_b,
            )

        # Standard momentum_shift or eta_shift — delegate to existing
        # differential_encode_decode with beta_B.
        return differential_encode_decode(
            w,
            n_steps=self.n_steps,
            tile_size=self.tile_size,
            beta=self.beta,
            proj_matrix=proj_matrix,
            proj_beta=proj_beta,
            adaptive_eta=self.adaptive_eta,
            eta_peak_step=self.eta_peak_step,
            order2_gamma=self.order2_gamma,
            order2_c1=self.order2_c1,
            order2_c2=self.order2_c2,
        )

    def _custom_dual_encode(
        self,
        w: torch.Tensor,
        *,
        head_idx: int,
        n_steps_b: int,
        sign_flip_b: float,
        proj_matrix: Optional[torch.Tensor],
        proj_beta: float,
        proj_beta_b: float,
    ) -> Tuple[torch.Tensor, dict]:
        """Custom dual-path encode for extra_step / sign_complement / residual strategies."""
        from .residual_pursuit import (
            _pad_to_tile_multiple,
            encode_matrix,
            decode_from_bases,
        )
        import torch.nn.functional as F

        rows, cols = w.shape
        is_residual = (self.strategy == PerturbationStrategy.RESIDUAL)

        # ---- Path A: base parameters ----
        bases_a, alphas_a, shape_a = encode_matrix(
            w,
            n_steps=self.n_steps,
            tile_size=self.tile_size,
            beta=self.beta,
            proj_matrix=proj_matrix,
            proj_beta=proj_beta,
            adaptive_eta=self.adaptive_eta,
            eta_peak_step=self.eta_peak_step,
            order2_gamma=self.order2_gamma,
            order2_c1=self.order2_c1,
            order2_c2=self.order2_c2,
        )
        ŵ_a = decode_from_bases(
            bases_a, alphas_a, shape_a, tile_size=self.tile_size
        )

        if is_residual:
            # §7.3 Residual Differential Encoding:
            # Path B encodes Path A's RECONSTRUCTION RESIDUAL with 1 step.
            # Decode: ŵ = ŵ_a + ŵ_b  (additive, not averaged).
            residual = w - ŵ_a
            bases_b, alphas_b, shape_b = encode_matrix(
                residual,
                n_steps=1,          # single step on the residual
                tile_size=self.tile_size,
                beta=self.beta,
                proj_matrix=proj_matrix,
                proj_beta=proj_beta,
                adaptive_eta=False,  # single step → no adaptive needed
                eta_peak_step=None,
                order2_gamma=self.order2_gamma,
                order2_c1=self.order2_c1,
                order2_c2=self.order2_c2,
            )
            ŵ_b = decode_from_bases(
                bases_b, alphas_b, shape_b, tile_size=self.tile_size
            )
            ŵ_diff = ŵ_a + ŵ_b  # additive reconstruction (not averaged)
        else:
            # ---- Path B: modified parameters ----
            bases_b, alphas_b, shape_b = encode_matrix(
                w,
                n_steps=n_steps_b,
                tile_size=self.tile_size,
                beta=self.beta,  # keep same β; perturbation via N or sign
                proj_matrix=proj_matrix,
                proj_beta=proj_beta_b,
                adaptive_eta=self.adaptive_eta,
                eta_peak_step=self.eta_peak_step,
                order2_gamma=self.order2_gamma,
                order2_c1=self.order2_c1,
                order2_c2=self.order2_c2,
            )
            # If sign complement, we need to encode with sign_flip=-1 inside
            # residual_pursuit_nd.  But encode_matrix doesn't expose sign_flip.
            # We handle this separately via manual pursuit below.
            if sign_flip_b < 0:
                ŵ_b = self._sign_flip_encode_and_decode(w, proj_matrix, proj_beta_b)
            else:
                ŵ_b = decode_from_bases(
                    bases_b, alphas_b, shape_b, tile_size=self.tile_size
                )
            # ---- Differential reconstruction ----
            ŵ_diff = (ŵ_a + ŵ_b) / 2

        # ---- Diagnostics ----
        w_flat = w.reshape(-1)
        wa = ŵ_a.reshape(-1)
        wb = ŵ_b.reshape(-1)
        wd = ŵ_diff.reshape(-1)

        eps_a = w_flat - wa
        eps_b = w_flat - wb
        eps_diff = w_flat - wd

        norm_a = eps_a.norm().item()
        norm_b = eps_b.norm().item()
        norm_diff = eps_diff.norm().item()

        mean_norm = (norm_a + norm_b) / 2
        nrr = 1.0 - (norm_diff / max(mean_norm, 1e-12))
        cross_corr = torch.dot(eps_a, eps_b).item() / max(norm_a * norm_b, 1e-12)

        mse_a = F.mse_loss(ŵ_a, w).item()
        mse_b = F.mse_loss(ŵ_b, w).item()
        mse_diff = F.mse_loss(ŵ_diff, w).item()

        def _cos(h: torch.Tensor) -> float:
            return F.cosine_similarity(
                w.reshape(-1).unsqueeze(0), h.reshape(-1).unsqueeze(0),
            ).item()

        def _snr(h: torch.Tensor) -> float:
            noise = ((w - h) ** 2).mean()
            signal = (w ** 2).mean()
            return 10 * math.log10(
                max(signal.item() / max(noise.item(), 1e-12), 1e-12)
            )

        # Pack bases for storage (fixes full_k_hat leak)
        from .residual_pursuit import pack_bases as _pack_bases
        bases_a_storage = _pack_bases(bases_a)
        bases_b_storage = _pack_bases(bases_b)
        bases_shape_M = bases_a.shape[-1]

        diag = {
            "nrr": nrr,
            "cross_corr": cross_corr,
            "mse_a": mse_a,
            "mse_b": mse_b,
            "mse_diff": mse_diff,
            "cosine_a": _cos(ŵ_a),
            "cosine_b": _cos(ŵ_b),
            "cosine_diff": _cos(ŵ_diff),
            "snr_a_db": _snr(ŵ_a),
            "snr_b_db": _snr(ŵ_b),
            "snr_diff_db": _snr(ŵ_diff),
            "strategy_a": f"N={self.n_steps},β={self.beta}",
            "strategy_b": f"N={n_steps_b},sign_flip={sign_flip_b}",
            # Dual-path bases for storage (fixes full_k_hat leak)
            "bases_a": bases_a_storage,
            "bases_b": bases_b_storage,
            "alphas_a": alphas_a,
            "alphas_b": alphas_b,
            "bases_shape_M": bases_shape_M,
        }
        return ŵ_diff, diag

    def _sign_flip_encode_and_decode(
        self,
        w: torch.Tensor,
        proj_matrix: Optional[torch.Tensor],
        proj_beta: float,
    ) -> torch.Tensor:
        """Encode with sign_flip=-1 via direct pursuit call."""
        from .residual_pursuit import (
            _pad_to_tile_multiple,
            _unpad,
            residual_pursuit_nd,
        )
        import torch.nn.functional as F

        rows, cols = w.shape
        w_padded, (pad_r, pad_c) = _pad_to_tile_multiple(w, self.tile_size)
        w_4d = w_padded.unsqueeze(0).unsqueeze(0)

        patches = (
            F.unfold(w_4d, kernel_size=self.tile_size, stride=self.tile_size)
            .squeeze(0)
            .transpose(0, 1)
            .contiguous()
        )  # (n_tiles, M)

        _bases, _alphas, w_hat_tiles = residual_pursuit_nd(
            patches,
            n_steps=self.n_steps,
            beta=self.beta,
            return_bases=False,
            proj_matrix=proj_matrix,
            proj_beta=proj_beta,
            sign_flip=-1.0,
            adaptive_eta=self.adaptive_eta,
            eta_peak_step=self.eta_peak_step,
            order2_gamma=self.order2_gamma,
            order2_c1=self.order2_c1,
            order2_c2=self.order2_c2,
        )
        # w_hat_tiles: (n_tiles, M)
        padded_h = rows + pad_r
        padded_w = cols + pad_c
        rec_4d = F.fold(
            w_hat_tiles.transpose(0, 1).unsqueeze(0),
            output_size=(padded_h, padded_w),
            kernel_size=self.tile_size,
            stride=self.tile_size,
        )
        rec = rec_4d.squeeze(0).squeeze(0)
        return _unpad(rec, (rows, cols))

    # ------------------------------------------------------------------
    # Head-pair differential encoding (for asymmetric pairing)
    # ------------------------------------------------------------------

    def encode_head_pair(
        self,
        K_pair: torch.Tensor,
        V_pair: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode K (and optionally V) with head-pair differential strategy.

        When two heads share the same KV cache (GQA), encoding them with
        complementary strategies and summing their attention scores
        achieves the same noise cancellation as the S/W dual-path method.

        Parameters
        ----------
        K_pair:
            ``(2, seq_len, d_head)`` — K tensors for both heads.
        V_pair:
            Optional ``(2, seq_len, d_head)`` — V tensors for both heads.

        Returns
        -------
        result:
            ``{"K_combined": ..., "V_combined": ..., "diag_k": ..., "diag_v": ...}``
        """
        assert K_pair.shape[0] == 2, "K_pair must have shape (2, seq_len, d_head)"

        K_diff_a, diag_k = self.encode_dual(K_pair[0], head_idx=-1)
        K_diff_b = K_pair[1]  # partner head uses raw (or separately encoded)
        # Default: average the two
        K_combined = (K_diff_a + K_diff_b) / 2

        result = {"K_combined": K_combined, "diag_k": diag_k}

        if V_pair is not None:
            V_diff_a, diag_v = self.encode_dual(V_pair[0], head_idx=-1)
            V_combined = (V_diff_a + V_pair[1]) / 2
            result["V_combined"] = V_combined
            result["diag_v"] = diag_v

        return result

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_quality(
        self,
        w: torch.Tensor,
        w_hat: torch.Tensor,
    ) -> dict:
        """Compute reconstruction quality metrics."""
        from .residual_pursuit import ResidualBinaryPursuit
        return ResidualBinaryPursuit.compute_metrics_static(w, w_hat)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"n_heads={self.n_heads}, d_head={self.d_head}, "
            f"strategy={self.strategy}, "
            f"perturbation={self.perturbation_strength}, "
            f"N={self.n_steps}, β={self.beta}, η={self.proj_beta}, "
            f"adaptive_eta={self.adaptive_eta}, "
            f"pairs={len(self._head_pairs)}"
        )