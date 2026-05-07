"""
§8.2 Hybrid DCT + DWT Transform Engine
=======================================

Replaces FWHT with a smarter pair of orthogonal transforms:

* **DCT (Discrete Cosine Transform - Type II)**: Excellent energy compaction for
  smooth, locally-correlated signals.  Pushes information into the lowest
  frequency components, giving Σ-Δ a well-structured "main signal + ripple".

* **DWT (Discrete Wavelet Transform - Haar)**: Zero-cost localisation.  A single
  outlier only affects the coefficients of its own decomposition path, unlike
  DCT/FWHT where outlier energy spreads globally (Gibbs phenomenon / flat spectrum).

The adaptive hybrid selector chooses per tile:
  - Low-variance tile → DCT (maximum energy compaction)
  - High-max outlier tile → DWT (localised isolation)
  - Medium tile → DCT + DWT cascade (DCT for body, DWT for residual)

This is the core signal-processing improvement over FWHT (§8.1.11 experiment:
FWHT degrades match_rate from 0.1477→0.1023 because it destroys the structured
energy distribution that Σ-Δ depends on).

Reference: R.I.N.A Whitepaper §8.2
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple

import torch
import torch.fft


# ---------------------------------------------------------------------------
# DCT Type-II (2-D, separable via 1-D DCTs)
# ---------------------------------------------------------------------------

def _dct_1d(x: torch.Tensor) -> torch.Tensor:
    """DCT Type-II along the **last dimension**.

    Uses the FFT-based method:
        DCT-II[x] = Re[ FFT( interleave(x, reverse(x)) ) * exp(-j·π·k / 2N) ]

    All 1-D DCT coefficients by real-valued RFFT.
    """
    N = x.shape[-1]

    # Mirror extension: [x_0, x_1, ..., x_{N-1}, x_{N-1}, ..., x_0]
    x_mirror = torch.cat([x, x.flip(-1)], dim=-1)

    # RFFT of length 2N; keep only real component
    X = torch.fft.rfft(x_mirror, dim=-1)  # complex

    # Twiddle factor: exp(-j * pi * k / (2N))  for k = 0..N-1
    k = torch.arange(N, device=x.device, dtype=x.dtype)
    twiddle = torch.exp(
        -1j * torch.pi * k / (2.0 * N)
    ).to(x.device)

    # Apply twiddle and take real (DCT is real-valued)
    X_dct = (X[..., :N] * twiddle).real
    return X_dct


def _idct_1d(X: torch.Tensor) -> torch.Tensor:
    """Inverse DCT Type-II (i.e. DCT Type-III) along the last dimension.

    X has shape (..., N), returns (..., N).
    """
    N = X.shape[-1]
    k = torch.arange(N, device=X.device, dtype=X.dtype)

    # Build complex signal: X_k * exp(+j * pi * k / (2N))
    twiddle = torch.exp(
        1j * torch.pi * k / (2.0 * N)
    ).to(X.device)
    Z = X.to(torch.complex64) * twiddle  # (..., N)

    # Zero-pad to 2N for IRFFT
    Z_pad = torch.cat([
        Z,
        torch.zeros(*Z.shape[:-1], 1, device=Z.device, dtype=Z.dtype),
    ], dim=-1)  # (..., N+1) — RFFT length 2N needs N+1 non-redundant

    # IRFFT: 2N real → N samples (discard mirror)
    x = torch.fft.irfft(Z_pad, n=2 * N, dim=-1)  # (..., 2N)
    x = x[..., :N]  # take first half

    # Normalise
    x = x / (2.0 * N)

    # First element: special correction
    x[..., 0] = x[..., 0] / 2.0

    return x.real.to(X.dtype)


def dct_2d(x: torch.Tensor) -> torch.Tensor:
    """2-D DCT Type-II: row DCT followed by column DCT.

    Parameters
    ----------
    x: Tensor of shape ``(..., H, W)``.

    Returns
    -------
    X_dct: Same shape, DCT coefficients.
    """
    # Row DCT along last dim
    X = _dct_1d(x)          # (..., H, W)
    # Column DCT along second-last dim
    X = _dct_1d(X.transpose(-1, -2)).transpose(-1, -2)  # (..., H, W)
    return X


def idct_2d(X: torch.Tensor) -> torch.Tensor:
    """2-D Inverse DCT Type-II.

    Parameters
    ----------
    X: DCT coefficients, shape ``(..., H, W)``.

    Returns
    -------
    x: Spatial-domain reconstruction, same shape.
    """
    # Column IDCT
    x = _idct_1d(X.transpose(-1, -2)).transpose(-1, -2)
    # Row IDCT
    x = _idct_1d(x)
    return x


# ---------------------------------------------------------------------------
# Haar Discrete Wavelet Transform (2-D, separable)
# ---------------------------------------------------------------------------

def _haar_1d(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-level 1-D Haar decomposition along the last dimension.

    Parameters
    ----------
    x: Tensor of shape ``(..., N)`` where N must be even.

    Returns
    -------
    approx: (..., N//2) — approximation (low-pass) coefficients.
    detail: (..., N//2) — detail (high-pass) coefficients.
    """
    N = x.shape[-1]
    assert N % 2 == 0, f"Haar requires even length, got {N}"

    x_even = x[..., 0::2]   # x[0], x[2], x[4], ...
    x_odd  = x[..., 1::2]   # x[1], x[3], x[5], ...

    approx = (x_even + x_odd) / 2.0      # scaling function
    detail = (x_even - x_odd) / 2.0      # wavelet function

    return approx, detail


def _ihaar_1d(approx: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
    """Inverse 1-D Haar: reconstruct x from approx + detail.

    Parameters
    ----------
    approx: (..., N//2) — low-pass coefficients.
    detail: (..., N//2) — high-pass coefficients.

    Returns
    -------
    x: (..., N) — reconstructed signal.
    """
    N2 = approx.shape[-1]
    even = approx + detail    # x[0], x[2], ...
    odd  = approx - detail    # x[1], x[3], ...

    # Interleave: [even_0, odd_0, even_1, odd_1, ...]
    x = torch.stack([even, odd], dim=-1)  # (..., N2, 2)
    x = x.reshape(*approx.shape[:-1], 2 * N2)  # (..., N)
    return x


def dwt_haar_2d(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-level 2-D Haar wavelet decomposition.

    Decomposes (H, W) image into 4 subbands:
        LL (low-low), LH (low-high), HL (high-low), HH (high-high).

    Parameters
    ----------
    x: ``(..., H, W)`` where H and W must be even.

    Returns
    -------
    LL, LH, HL, HH: Each ``(..., H/2, W/2)``.
    """
    # Row decomposition
    L_row, H_row = _haar_1d(x)           # (..., H, W/2)

    # Column decomposition of each row-band
    LL = _haar_1d(L_row.transpose(-1, -2))[0].transpose(-1, -2)   # low-low
    LH = _haar_1d(L_row.transpose(-1, -2))[1].transpose(-1, -2)   # low-high
    HL = _haar_1d(H_row.transpose(-1, -2))[0].transpose(-1, -2)   # high-low
    HH = _haar_1d(H_row.transpose(-1, -2))[1].transpose(-1, -2)   # high-high

    return LL, LH, HL, HH


def idwt_haar_2d(
    LL: torch.Tensor,
    LH: torch.Tensor,
    HL: torch.Tensor,
    HH: torch.Tensor,
) -> torch.Tensor:
    """Inverse 2-D Haar wavelet.

    Parameters
    ----------
    LL, LH, HL, HH: Each ``(..., H/2, W/2)`` subband.

    Returns
    -------
    x: ``(..., H, W)`` reconstruction.
    """
    # Reconstruct rows from LL+LH and HL+HH
    L_row = _ihaar_1d(LL.transpose(-1, -2), LH.transpose(-1, -2)).transpose(-1, -2)
    H_row = _ihaar_1d(HL.transpose(-1, -2), HH.transpose(-1, -2)).transpose(-1, -2)

    # Reconstruct full image from L_row + H_row
    x = _ihaar_1d(L_row, H_row)
    return x


# ---------------------------------------------------------------------------
# Hybrid adaptive transform selector
# ---------------------------------------------------------------------------

class TransformMode(Enum):
    """Transform mode for tile pre-processing before Σ-Δ encoding."""
    AUTO = "auto"       # Adaptive: choose DCT, DWT, or hybrid per tile
    DCT = "dct"         # Force DCT on all tiles
    DWT = "dwt"         # Force Haar DWT on all tiles
    HYBRID = "hybrid"   # Force DCT+DWT cascade on all tiles
    NONE = "none"       # No transform (raw spatial domain)
    FWHT = "fwht"       # Legacy FWHT (kept for ablation)


def apply_transform(
    tiles: torch.Tensor,
    mode: TransformMode,
    tile_size: int,
    *,
    smooth_threshold: float = 0.05,
    outlier_threshold: float = 3.0,
) -> Tuple[torch.Tensor, list]:
    """Apply the selected transform to a batch of tiles.

    Parameters
    ----------
    tiles: ``(n_tiles, tile_size, tile_size)`` or ``(n_tiles, tile_size**2)``.
        If flat (2-D), tiles are reshaped to (n_tiles, tile_size, tile_size)
        for 2-D transforms and flattened back.
    mode: TransformMode
        The transform strategy to use.  AUTO does per-tile adaptive selection.
    tile_size: int
        Tile dimension (must be power-of-2 for DWT, even at minimum).
    smooth_threshold: float
        Variance threshold for AUTO mode: tiles below this use DCT.
    outlier_threshold: float
        Max-abs threshold for AUTO mode: tiles above this use DWT.

    Returns
    -------
    transformed: ``(n_tiles, tile_size**2)`` — flat transformed tiles.
    decisions: list of str — per-tile decision record (for diagnostics).
    """
    flat_input = (tiles.dim() == 2)
    if flat_input:
        tiles_2d = tiles.reshape(-1, tile_size, tile_size)
    else:
        tiles_2d = tiles

    n_tiles = tiles_2d.shape[0]
    M = tile_size * tile_size
    decisions = []

    if mode == TransformMode.NONE:
        return (tiles_2d.reshape(n_tiles, M) if flat_input else tiles.reshape(n_tiles, M)), ["none"] * n_tiles

    if mode == TransformMode.FWHT:
        from rina.utils.walsh_hadamard import fwht as _fwht
        flat = tiles_2d.reshape(n_tiles, M)
        return _fwht(flat), ["fwht"] * n_tiles

    if mode == TransformMode.DCT:
        X = dct_2d(tiles_2d.to(torch.float32))
        return X.reshape(n_tiles, M), ["dct"] * n_tiles

    if mode == TransformMode.DWT:
        ll, lh, hl, hh = dwt_haar_2d(tiles_2d.to(torch.float32))
        half = tile_size // 2
        top = torch.cat([ll, lh], dim=-1)     # (n_tiles, half, tile_size)
        bot = torch.cat([hl, hh], dim=-1)     # (n_tiles, half, tile_size)
        X = torch.cat([top, bot], dim=-2)      # (n_tiles, tile_size, tile_size)
        return X.reshape(n_tiles, M), ["dwt"] * n_tiles

    if mode == TransformMode.HYBRID:
        # NOTE: Hybrid cascade (DCT low-freq + DWT residual) has a
        # structural packing problem for square tiles:
        # both DCT tile and DWT tile produce tile_size² elements each
        # but M//2 only fits tile_size²/2.  Fall back to pure DCT.
        X = dct_2d(tiles_2d.to(torch.float32))
        return X.reshape(n_tiles, M), ["dct"] * n_tiles

    # AUTO: per-tile adaptive selection
    if mode == TransformMode.AUTO:
        tile_vars = tiles_2d.reshape(n_tiles, -1).var(dim=-1)        # (n_tiles,)
        tile_maxabs = tiles_2d.reshape(n_tiles, -1).abs().max(dim=-1).values  # (n_tiles,)

        result_parts = []
        for i in range(n_tiles):
            v = tile_vars[i].item()
            m = tile_maxabs[i].item()
            t = tiles_2d[i:i+1].to(torch.float32)
            M = tile_size * tile_size
            half = tile_size // 2

            if v < smooth_threshold:
                # Low variance: smooth → DCT
                X = dct_2d(t).reshape(1, M)
                decisions.append("dct")
            elif m > outlier_threshold:
                # High outlier → DWT
                ll, lh, hl, hh = dwt_haar_2d(t)
                top = torch.cat([ll, lh], dim=-1)
                bot = torch.cat([hl, hh], dim=-1)
                X = torch.cat([top, bot], dim=-2).reshape(1, M)
                decisions.append("dwt")
            else:
                # Medium → DCT (Hybrid cascade disabled — the packed format
                # M/2 per sublayer requires tile_size²/2 elements but both
                # DCT tile and DWT tile produce tile_size² elements each,
                # so packing both into one row is structurally broken for
                # square tiles.  DCT alone is the safe fallback.)
                X = dct_2d(t).reshape(1, M)
                decisions.append("dct")

            result_parts.append(X)

        return torch.cat(result_parts, dim=0), decisions

    raise ValueError(f"Unknown TransformMode: {mode}")


def apply_inverse_transform(
    transformed: torch.Tensor,
    mode: TransformMode,
    tile_size: int,
    decisions: Optional[list] = None,
    original_shape: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Inverse of apply_transform: reconstruct spatial-domain tiles.

    Parameters
    ----------
    transformed: ``(n_tiles, tile_size**2)`` — flat transformed tiles.
    mode: TransformMode — must match the forward transform mode.
    tile_size: int.
    decisions: Optional per-tile decision list (needed for AUTO mode).
    original_shape: ``(N, d_head)`` | None.
        When provided, the output is reshaped from ``(n_tiles, tile_size**2)``
        to ``(N_padded, d_head)`` after per-tile inverse transform, then
        cropped to the first ``N`` rows.  This eliminates the fragile
        reshape/detection logic that previously lived in callers.

    Returns
    -------
    tiles: ``(n_tiles, tile_size**2)`` — flat spatial-domain tiles,
           OR ``(N, d_head)`` when *original_shape* is provided.
    """
    n_tiles = transformed.shape[0]
    M = tile_size * tile_size
    half = tile_size // 2

    if mode == TransformMode.NONE:
        result = transformed
    elif mode == TransformMode.FWHT:
        from rina.utils.walsh_hadamard import ifwht as _ifwht
        result = _ifwht(transformed)
    elif mode == TransformMode.DCT:
        tiles_2d = transformed.reshape(n_tiles, tile_size, tile_size)
        X = idct_2d(tiles_2d)
        result = X.reshape(n_tiles, M)
    elif mode == TransformMode.DWT:
        tiles_2d = transformed.reshape(n_tiles, tile_size, tile_size)
        ll  = tiles_2d[:, :half, :half]
        lh  = tiles_2d[:, :half, half:]
        hl  = tiles_2d[:, half:, :half]
        hh  = tiles_2d[:, half:, half:]
        X = idwt_haar_2d(ll, lh, hl, hh)
        result = X.reshape(n_tiles, M)
    elif mode == TransformMode.HYBRID:
        result_parts = []
        for i in range(n_tiles):
            t = transformed[i]  # (M,)
            # First half: DCT low-freq
            dct_low = t[:M // 2].reshape(tile_size, tile_size)
            # Second half: DWT residual
            dwt_r = t[M // 2:].reshape(tile_size, tile_size)
            ll  = dwt_r[:half, :half]
            lh  = dwt_r[:half, half:]
            hl  = dwt_r[half:, :half]
            hh  = dwt_r[half:, half:]
            residual = idwt_haar_2d(
                ll.unsqueeze(0), lh.unsqueeze(0),
                hl.unsqueeze(0), hh.unsqueeze(0),
            ).squeeze(0)
            body = idct_2d(dct_low.unsqueeze(0)).squeeze(0)
            X = body + residual
            result_parts.append(X.reshape(1, M))
        result = torch.cat(result_parts, dim=0)
    elif mode == TransformMode.AUTO:
        assert decisions is not None, "AUTO inverse requires per-tile decisions"
        result_parts = []
        for i in range(n_tiles):
            t = transformed[i]  # (M,)
            dec = decisions[i]

            if dec == "dct":
                X = idct_2d(t.reshape(tile_size, tile_size))
            elif dec == "dwt":
                t2d = t.reshape(tile_size, tile_size)
                ll  = t2d[:half, :half]
                lh  = t2d[:half, half:]
                hl  = t2d[half:, :half]
                hh  = t2d[half:, half:]
                X = idwt_haar_2d(
                    ll.unsqueeze(0), lh.unsqueeze(0),
                    hl.unsqueeze(0), hh.unsqueeze(0),
                ).squeeze(0)
            elif dec == "hybrid":
                dct_low = t[:M // 2].reshape(tile_size, tile_size)
                dwt_r = t[M // 2:].reshape(tile_size, tile_size)
                ll  = dwt_r[:half, :half]
                lh  = dwt_r[:half, half:]
                hl  = dwt_r[half:, :half]
                hh  = dwt_r[half:, half:]
                residual = idwt_haar_2d(
                    ll.unsqueeze(0), lh.unsqueeze(0),
                    hl.unsqueeze(0), hh.unsqueeze(0),
                ).squeeze(0)
                body = idct_2d(dct_low.unsqueeze(0)).squeeze(0)
                X = body + residual
            else:
                X = t.reshape(tile_size, tile_size)

            result_parts.append(X.reshape(1, M))
        result = torch.cat(result_parts, dim=0)
    else:
        raise ValueError(f"Unknown TransformMode: {mode}")

    # ── Reshape to original spatial layout if requested ──
    if original_shape is not None:
        N_orig, d_head = original_shape
        N_padded = n_tiles * tile_size  # rows after inverse transform
        # Reshape flat (n_tiles, M) → (N_padded, d_head)
        # Assert: n_tiles * M == N_padded * d_head
        result = result.reshape(N_padded, d_head)
        # Crop padding rows
        if N_padded > N_orig:
            result = result[:N_orig]
        return result

    return result


def compute_tile_diagnostics(
    tiles_2d: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-tile variance and max-abs for adaptive decision.

    Parameters
    ----------
    tiles_2d: ``(n_tiles, tile_size, tile_size)`` or ``(n_tiles, M)``.

    Returns
    -------
    variances: ``(n_tiles,)``.
    max_abs_vals: ``(n_tiles,)``.
    """
    if tiles_2d.dim() == 2:
        flat = tiles_2d
    else:
        flat = tiles_2d.reshape(tiles_2d.shape[0], -1)
    variances = flat.var(dim=-1)
    max_abs_vals = flat.abs().max(dim=-1).values
    return variances, max_abs_vals


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "TransformMode",
    "dct_2d",
    "idct_2d",
    "dwt_haar_2d",
    "idwt_haar_2d",
    "apply_transform",
    "apply_inverse_transform",
    "compute_tile_diagnostics",
]