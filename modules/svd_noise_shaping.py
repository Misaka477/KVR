"""
§6 SVD Noise Shaping Projection — DS-KVCache Component Two
==========================================================

Pre-computes attention-aware nullspace projection matrices from
Q-statistics samples, then applies the projection during online
KV-cache encoding to push quantisation noise into directions
that softmax(QK^T/√d) does not perceive.

Key concepts
------------
- P_signal = U_{:k} U_{:k}^T  → projects onto the top-k principal
  directions of the Q sample distribution.
- P_null = I - P_signal       → projects quantisation error into
  the perceptual nullspace (complement of signal space).
- Per-head calibration: each attention head sees a different
  subspace of Q, so P_null is computed per head.

Reference: R.I.N.A / DS-KVCache Whitepaper §6
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pre-computation helpers
# ---------------------------------------------------------------------------


def compute_q_covariance(
    Q_samples: torch.Tensor,
) -> torch.Tensor:
    """Compute covariance of Q samples.

    Parameters
    ----------
    Q_samples:
        ``(n_samples, d_head)`` or ``(n_heads, n_samples, d_head)``
        tensor of Q vectors collected from a short calibration
        forward pass (100-500 tokens typically suffice).

    Returns
    -------
    Σ_Q:
        If input is ``(n_samples, d)`` → ``(d, d)``.
        If input is ``(n_heads, n_samples, d)`` → ``(n_heads, d, d)``.
    """
    if Q_samples.dim() == 2:
        # Single head / pooled across heads
        Q = Q_samples.float()
        Q_c = Q - Q.mean(dim=0, keepdim=True)
        n = Q.shape[0]
        return (Q_c.T @ Q_c) / n

    # Per-head: (H, N, d)
    H, N, d = Q_samples.shape
    Q = Q_samples.float()
    Q_c = Q - Q.mean(dim=1, keepdim=True)  # (H, N, d)
    # Batched covariance: (H, d, d)
    Σ = torch.bmm(Q_c.transpose(1, 2), Q_c) / N
    return Σ


def compute_nullspace_projector(
    Σ_Q: torch.Tensor,
    energy_ratio: float = 0.95,
) -> Tuple[torch.Tensor, int]:
    """Build P_null = I - P_signal from Q covariance matrix.

    Parameters
    ----------
    Σ_Q:
        ``(d, d)`` Q-covariance matrix (single head).
    energy_ratio:
        Fraction of spectral energy to retain in the signal subspace.
        Default 0.95 → top singular vectors covering 95% energy.

    Returns
    -------
    P_null:
        ``(d, d)`` nullspace projector.
    k:
        Number of retained principal components.
    """
    d = Σ_Q.shape[0]
    device = Σ_Q.device

    # Eigen-decomposition of symmetric positive semi-definite matrix
    Λ, U = torch.linalg.eigh(Σ_Q)  # ascending order: λ₁ < λ₂ < ... < λ_d

    # Descending order
    Λ = torch.flip(Λ, dims=[0])
    U = torch.flip(U, dims=[1])

    # Cumulative energy
    total_energy = Λ.sum()
    if total_energy < 1e-12:
        # Degenerate case: return identity (no nullspace)
        return torch.eye(d, device=device, dtype=torch.float32), 0

    cum_energy = torch.cumsum(Λ, dim=0)
    # Find k: smallest k such that cum_energy[k-1] / total_energy >= energy_ratio
    k_mask = cum_energy / total_energy >= energy_ratio
    k = int(k_mask.float().argmax().item()) + 1
    k = min(k, d - 1)  # at least 1 dim for nullspace
    k = max(k, 1)       # at least 1 dim for signal

    # Signal subspace projector
    U_k = U[:, :k]           # (d, k)
    P_signal = U_k @ U_k.T   # (d, d)

    # Nullspace projector
    P_null = torch.eye(d, device=device, dtype=torch.float32) - P_signal

    return P_null, k


def compute_per_head_nullspace_projectors(
    Q_samples: torch.Tensor,
    energy_ratio: float = 0.95,
) -> Dict[int, Tuple[torch.Tensor, int]]:
    """Compute P_null and rank k for each attention head.

    Parameters
    ----------
    Q_samples:
        ``(n_heads, n_samples, d_head)`` — calibration Q vectors
        collected per head.
    energy_ratio:
        Fraction of spectral energy retained in signal subspace.

    Returns
    -------
    projectors:
        ``{head_idx: (P_null, k)}`` mapping.
    """
    assert Q_samples.dim() == 3, (
        f"Expected (n_heads, n_samples, d_head), got shape {Q_samples.shape}"
    )
    n_heads = Q_samples.shape[0]
    projectors: Dict[int, Tuple[torch.Tensor, int]] = {}

    Σ_all = compute_q_covariance(Q_samples)  # (n_heads, d, d)

    for h in range(n_heads):
        P_null, k = compute_nullspace_projector(Σ_all[h], energy_ratio)
        projectors[h] = (P_null, k)

    return projectors


def compute_shared_nullspace_projector(
    Q_samples: torch.Tensor,
    energy_ratio: float = 0.95,
) -> Tuple[torch.Tensor, int]:
    """Compute a single shared P_null by pooling Q across all heads.

    Useful for GQA (Grouped-Query Attention) where multiple heads
    share the same K/V projections.
    """
    if Q_samples.dim() == 3:
        # Pool: (n_heads, n_samples, d_head) → (n_heads * n_samples, d_head)
        Q_pooled = Q_samples.reshape(-1, Q_samples.shape[-1])
    else:
        Q_pooled = Q_samples

    Σ_Q = compute_q_covariance(Q_pooled)
    return compute_nullspace_projector(Σ_Q, energy_ratio)


# ---------------------------------------------------------------------------
# Main module: pre-computes and caches projectors
# ---------------------------------------------------------------------------


class SVDNoiseShaper(nn.Module):
    """Attention-aware SVD noise-shaping projection module.

    Pre-computes P_null per attention head from Q calibration data,
    then exposes an online projection method for KV-cache encoding.

    Usage::

        # --- Calibration (model load time, once) ---
        shaper = SVDNoiseShaper(n_heads=32, d_head=128, energy_ratio=0.95)
        Q_calib = collect_q_samples(model, calib_data)  # (H, N_samples, d)
        shaper.calibrate(Q_calib)

        # --- Online (during inference) ---
        # For each head h, push quantisation error into nullspace:
        error_shaped = shaper.project_noise(head_idx=h, error=error_h)
        # ... feed error_shaped into DS modulator residual update

    Parameters
    ----------
    n_heads:
        Number of attention heads.
    d_head:
        Dimension per head.
    energy_ratio:
        Fraction of spectral energy retained in signal subspace.
        Lower = more aggressive noise pushing into nullspace.
        Recommended: 0.90–0.98.
    """

    def __init__(
        self,
        n_heads: int,
        d_head: int,
        energy_ratio: float = 0.95,
        share_projector: bool = False,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.energy_ratio = energy_ratio
        self.share_projector = share_projector

        # Will be populated by .calibrate()
        self._calibrated: bool = False
        self._projectors: Dict[int, Tuple[torch.Tensor, int]] = {}
        self._P_null_shared: Optional[torch.Tensor] = None
        self._k_shared: int = 0

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    # ------------------------------------------------------------------
    # Calibration API
    # ------------------------------------------------------------------

    def calibrate(
        self,
        Q_samples: torch.Tensor,
    ) -> None:
        """Compute and cache P_null from Q calibration data.

        Parameters
        ----------
        Q_samples:
            ``(n_heads, n_samples, d_head)`` — Q vectors from a short
            calibration forward pass.
        """
        assert Q_samples.dim() == 3, (
            f"Expected (n_heads, n_samples, d_head), got {Q_samples.shape}"
        )
        assert Q_samples.shape[0] == self.n_heads, (
            f"Q_samples n_heads={Q_samples.shape[0]} != module n_heads={self.n_heads}"
        )
        assert Q_samples.shape[2] == self.d_head, (
            f"Q_samples d_head={Q_samples.shape[2]} != module d_head={self.d_head}"
        )

        if self.share_projector:
            P_null, k = compute_shared_nullspace_projector(
                Q_samples, self.energy_ratio
            )
            self._P_null_shared = P_null
            self._k_shared = k
        else:
            self._projectors = compute_per_head_nullspace_projectors(
                Q_samples, self.energy_ratio
            )

        self._calibrated = True

    def calibrate_from_pooled(
        self,
        Q_pooled: torch.Tensor,
    ) -> None:
        """Calibrate with single pooled P_null (GQA-friendly).

        Parameters
        ----------
        Q_pooled:
            ``(n_samples, d_head)`` — Q vectors pooled across heads.
        """
        P_null, k = compute_nullspace_projector(
            compute_q_covariance(Q_pooled), self.energy_ratio
        )
        self._P_null_shared = P_null
        self._k_shared = k
        self.share_projector = True
        self._calibrated = True

    def get_projector(self, head_idx: int) -> Tuple[torch.Tensor, int]:
        """Get (P_null, k) for a given head.

        Raises RuntimeError if not calibrated.
        """
        if not self._calibrated:
            raise RuntimeError(
                "SVDNoiseShaper not calibrated. Call .calibrate(Q_samples) first."
            )

        if self.share_projector:
            return self._P_null_shared, self._k_shared

        return self._projectors[head_idx]

    # ------------------------------------------------------------------
    # Online projection
    # ------------------------------------------------------------------

    def project_noise(
        self,
        head_idx: int,
        error: torch.Tensor,
    ) -> torch.Tensor:
        """Project quantisation error into the attention nullspace.

        Parameters
        ----------
        head_idx:
            Which attention head (0-indexed).
        error:
            ``(..., d_head)`` — quantisation error from the current
            DS modulation step.

        Returns
        -------
        error_null:
            ``(..., d_head)`` — error projected into nullspace.
            Components in signal directions are suppressed.
        """
        P_null, _ = self.get_projector(head_idx)
        P_null = P_null.to(device=error.device, dtype=error.dtype)
        # error_null = P_null @ error  (matrix multiply on last dim)
        return torch.matmul(error, P_null.T)

    def project_noise_flat(
        self,
        head_idx: int,
        error_flat: torch.Tensor,
    ) -> torch.Tensor:
        """Project tiled/flat error into nullspace.

        Parameters
        ----------
        head_idx:
            Attention head index.
        error_flat:
            ``(n_tiles, M)`` or ``(..., M)`` where M = tile_size² is
            the flattened tile dimension.  Each tile is projected
            through the nullspace matrix for that head.

        Returns
        -------
        error_null:
            Same shape as input, projected into nullspace.

        Note
        ----
        This method uses the *full* ``(d_head, d_head)`` projector
        directly, which works for per-token K/V vectors.  For tile-
        based encoding of K/V matrices, use :meth:`project_noise` on
        the per-tile residual.
        """
        P_null, _ = self.get_projector(head_idx)
        P_null = P_null.to(device=error_flat.device, dtype=error_flat.dtype)
        # (..., M) @ (M, M) → (..., M)
        return torch.matmul(error_flat, P_null.T)

    def forward(
        self,
        head_idx: int,
        error: torch.Tensor,
    ) -> torch.Tensor:
        """Alias for :meth:`project_noise`."""
        return self.project_noise(head_idx, error)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def ranks(self) -> Dict[int, int]:
        """Return per-head (or shared) signal-subspace ranks."""
        if self.share_projector:
            return {-1: self._k_shared}
        return {h: k for h, (_, k) in self._projectors.items()}

    def effective_compression_ratio(self) -> float:
        """Approximate fraction of dimensions suppressed into nullspace."""
        total_k = 0
        if self.share_projector:
            total_k = self._k_shared
        else:
            total_k = sum(k for _, k in self._projectors.values())
        avg_k = total_k / max(self.n_heads, 1)
        return 1.0 - (avg_k / self.d_head)

    def extra_repr(self) -> str:
        comp = ""
        if self._calibrated:
            if self.share_projector:
                comp = f", shared_k={self._k_shared}"
            else:
                avg_k = (
                    sum(k for _, k in self._projectors.values()) / self.n_heads
                )
                comp = f", avg_k={avg_k:.1f}"
        return (
            f"n_heads={self.n_heads}, d_head={self.d_head}, "
            f"energy_ratio={self.energy_ratio}, "
            f"calibrated={self._calibrated}{comp}"
        )