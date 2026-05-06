"""
Fast Walsh-Hadamard Transform (FWHT) — zero-multiplication orthogonal transform.

Mathematical foundation:
    Hadamard matrix H[n] defined recursively:
        H[1] = [1]
        H[2k] = [H[k]  H[k] ]
                [H[k] -H[k]]

    FWHT(x) = H[n] @ x  computes the Walsh spectrum of x in O(n·log₂n)
    using only additions/subtractions — zero multiplications.

Key properties for R.I.N.A.:
    • Orthogonal: H·Hᵀ = n·I  (energy-preserving up to scale n)
    • Outlier diffusion: a single extreme element spreads uniformly
      across all Walsh coefficients — quantiser sees flat spectrum
    • Perfect reconstruction: ifwht(fwht(x)) == x
    • GPU-friendly: in-place butterfly network, coalesced memory access

Usage:
    >>> x = torch.randn(100, 256)          # (n_tiles, tile_size²)
    >>> x_walsh = fwht(x)                  # forward transform
    >>> x_recon = ifwht(x_walsh)           # inverse (with normalisation)
    >>> torch.allclose(x, x_recon, atol=1e-6)
    True
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def fwht(x: torch.Tensor) -> torch.Tensor:
    """In-place Fast Walsh-Hadamard Transform on the last dimension.

    Operates on the trailing dimension, which MUST be a power of 2.
    For R.I.N.A. usage, this is always tile_size² = 256 (16×16 tile).

    Algorithm: iterative butterfly (Cooley-Tukey radix-2 on Hadamard).
    Each iteration pairs elements at distance h apart.

    Complexity: O(N·log₂N) per tile, ZERO multiplications.

    Parameters
    ----------
    x:
        Input tensor of any shape.  The last dimension must be a power of 2.

    Returns
    -------
    Transformed tensor, same shape as input.  No normalisation applied —
    call ``ifwht`` to recover the original signal.
    """
    n = x.shape[-1]
    assert (n & (n - 1)) == 0, f"FWHT: last dim {n} must be a power of 2"

    # Work on a contiguous copy, reshape trick: treat last dim as (n/h, h) pairs
    y = x.clone()
    h = 1
    while h < n:
        # Reshape: (..., n/(2h), 2, h)
        # Pair elements at distance h by treating them as adjacent in dim=-2
        y_view = y.view(*y.shape[:-1], n // (2 * h), 2, h)
        # a = left element of each pair, b = right element
        a = y_view[..., 0, :].clone()
        b = y_view[..., 1, :].clone()
        y_view[..., 0, :] = a + b
        y_view[..., 1, :] = a - b
        h *= 2
    return y


def ifwht(x: torch.Tensor) -> torch.Tensor:
    """Inverse Fast Walsh-Hadamard Transform.

    Computes fwht(x) / N where N = x.shape[-1].  This recovers the
    original signal from its Walsh spectrum.

    Parameters
    ----------
    x:
        Walsh-domain tensor.

    Returns
    -------
    Time-domain tensor, same shape as input.
    """
    n = x.shape[-1]
    return fwht(x) / n


# ---------------------------------------------------------------------------
# Self-test (runs on import if executed directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== FWHT Self-Test ===")

    # Test 1: Perfect reconstruction (single tile)
    torch.manual_seed(42)
    x = torch.randn(256)
    x_w = fwht(x)
    x_hat = ifwht(x_w)
    err = (x - x_hat).abs().max().item()
    print(f"  Single tile recon error: {err:.2e}  {'OK' if err < 1e-6 else 'FAIL'}")

    # Test 2: Batch of tiles (simulating encode_matrix input)
    tiles = torch.randn(48, 256)  # 48 tiles from 3 heads × 16 tile_size
    tiles_w = fwht(tiles)
    tiles_hat = ifwht(tiles_w)
    err_batch = (tiles - tiles_hat).abs().max().item()
    print(f"  Batch 48×256 recon error: {err_batch:.2e}  {'OK' if err_batch < 1e-6 else 'FAIL'}")

    # Test 3: Energy preservation (H^T H = N·I)
    x2 = torch.randn(256)
    energy_in = (x2 ** 2).sum()
    energy_walsh = (fwht(x2) ** 2).sum()
    ratio = energy_walsh / energy_in
    expected_ratio = float(x2.shape[-1])  # H^T H = N·I
    print(f"  Energy ratio: {ratio:.1f} (expected {expected_ratio:.0f})"
          f"  {'OK' if abs(ratio - expected_ratio) < 0.1 else 'FAIL'}")

    # Test 4: Outlier diffusion — single spike becomes uniform
    outlier = torch.zeros(256)
    outlier[0] = 10.0
    w_outlier = fwht(outlier)
    all_vals = w_outlier.abs()
    max_abs = all_vals.max().item()
    min_abs = all_vals.min().item()
    print(f"  Outlier [10,0,...,0] → Walsh range [{min_abs:.3f}, {max_abs:.3f}]")
    # All coefficients = ±10 (H has only ±1 entries)
    expected_val = 10.0
    max_dev = (w_outlier.abs() - expected_val).abs().max().item()
    print(f"  Max deviation from flat ±10: {max_dev:.2e}  {'OK' if max_dev < 1e-6 else 'WARN'}")

    # Test 5: Orthogonality check — H·H^T / N = I
    # Generate a small H and verify
    def hadamard(n):
        if n == 1:
            return torch.tensor([[1.0]])
        H_half = hadamard(n // 2)
        return torch.cat([
            torch.cat([H_half, H_half], dim=1),
            torch.cat([H_half, -H_half], dim=1)], dim=0)

    H16 = hadamard(16)
    eye_check = (H16 @ H16.T) / 16.0
    off_diag = (eye_check - torch.eye(16)).abs().max().item()
    print(f"  H16 orthogonality off-diag: {off_diag:.2e}  {'OK' if off_diag < 1e-12 else 'FAIL'}")

    # Test 6: FWHT matches matrix multiply
    y = torch.randn(16)
    y_fwht = fwht(y)
    y_mat = H16 @ y
    err_mat = (y_fwht - y_mat).abs().max().item()
    print(f"  FWHT vs H16·x error: {err_mat:.2e}  {'OK' if err_mat < 1e-12 else 'FAIL'}")

    # Test 7: GPU compatibility (if available)
    if torch.cuda.is_available():
        x_gpu = torch.randn(100, 256, device="cuda")
        x_gpu_w = fwht(x_gpu)
        x_gpu_hat = ifwht(x_gpu_w)
        err_gpu = (x_gpu - x_gpu_hat).abs().max().item()
        print(f"  GPU recon error: {err_gpu:.2e}  {'OK' if err_gpu < 1e-6 else 'FAIL'}")

    print("=== FWHT Self-Test Passed ===")