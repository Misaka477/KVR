#!/usr/bin/env python
"""Second-order Σ-Δ sweep: explore (β, γ) grid across matrix archetypes (§8.1.2).

Produces a CSV table with:
  N, matrix_type, beta, gamma, mse, snr_db, cosine_similarity

Matrix archetypes correspond to different neural-network weight regimes:
  - random_iid    — pure i.i.d. Gaussian (whitepaper baseline)
  - low_rank_4    — rank-4 signal + small noise (attention Q/K/V projection)
  - layernorm_scale — tiny std ~0.005 (LayerNorm / final projection)
  - per_tile_varying — different energy per tile (adaptive-N scenario)

Usage:
    python scripts/sweep_order2.py [--output results/sweep_order2.csv]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from pathlib import Path
from typing import Iterator

import torch

# Ensure repo root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.residual_pursuit import residual_pursuit_nd  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
TILE_DIM = 256        # 16×16 tile elements
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# Sweep grid
N_STEPS_VALUES = [1, 3, 5, 8, 10]
BETA_VALUES = [0.0, 0.15, 0.3, 0.5, 0.7]
GAMMA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.5]

# Matrix archetypes
ARCHETYPES = ["random_iid", "low_rank_4", "layernorm_scale", "per_tile_varying"]


# ---------------------------------------------------------------------------
# Matrix generators
# ---------------------------------------------------------------------------

def make_random_iid(n_tiles: int, std: float = 0.02) -> torch.Tensor:
    """i.i.d. Gaussian tiles (whitepaper baseline)."""
    return torch.randn(n_tiles, TILE_DIM, device=DEVICE, dtype=DTYPE) * std


def make_low_rank_4(n_tiles: int, std: float = 0.02) -> torch.Tensor:
    """Rank-4 signal + small noise per tile (attention projection)."""
    torch.manual_seed(SEED + 100)
    U = torch.randn(n_tiles, 4, device=DEVICE, dtype=DTYPE)
    V = torch.randn(4, TILE_DIM, device=DEVICE, dtype=DTYPE)
    signal = U @ V  # (n_tiles, 4) @ (4, 256) → (n_tiles, 256)
    signal = signal * (std / signal.std(dim=1, keepdim=True).clamp(min=1e-8))
    noise = torch.randn(n_tiles, TILE_DIM, device=DEVICE, dtype=DTYPE) * std * 0.2
    return signal + noise


def make_layernorm_scale(n_tiles: int, std: float = 0.005) -> torch.Tensor:
    """Very small std weights (LayerNorm / final projection regime)."""
    return torch.randn(n_tiles, TILE_DIM, device=DEVICE, dtype=DTYPE) * std


def make_per_tile_varying(n_tiles: int) -> torch.Tensor:
    """Tiles with log-uniform energy spread (adaptive-N scenario)."""
    torch.manual_seed(SEED + 200)
    # Log-uniform energy from 0.001 to 0.1
    log_energies = torch.linspace(-3.0, -1.0, n_tiles, device=DEVICE, dtype=DTYPE)
    scales = (10 ** log_energies).sqrt()  # std ∝ sqrt(energy)
    data = torch.randn(n_tiles, TILE_DIM, device=DEVICE, dtype=DTYPE)
    return data * scales.unsqueeze(1)


MATRIX_BUILDERS = {
    "random_iid": make_random_iid,
    "low_rank_4": make_low_rank_4,
    "layernorm_scale": make_layernorm_scale,
    "per_tile_varying": make_per_tile_varying,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(w: torch.Tensor, w_hat: torch.Tensor) -> dict:
    """Compute MSE, SNR (dB), and cosine similarity."""
    eps = 1e-12
    mse = torch.nn.functional.mse_loss(w_hat, w).item()
    signal_power = w.pow(2).mean().item()
    noise_power = max(mse, eps)
    snr_db = 10.0 * math.log10(signal_power / noise_power)
    cos_sim = torch.nn.functional.cosine_similarity(
        w.reshape(1, -1), w_hat.reshape(1, -1)
    ).item()
    return {"mse": mse, "snr_db": snr_db, "cosine_similarity": cos_sim}


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep(n_tiles: int = 256) -> Iterator[dict]:
    """Yield one row per (N, archetype, beta, gamma) combination."""
    # Pre-build matrices (same matrices for all param combos)
    matrices = {
        name: builder(n_tiles)
        for name, builder in MATRIX_BUILDERS.items()
    }

    total = len(N_STEPS_VALUES) * len(ARCHETYPES) * len(BETA_VALUES) * len(GAMMA_VALUES)
    count = 0

    for n in N_STEPS_VALUES:
        for arch_name in ARCHETYPES:
            w = matrices[arch_name]
            for beta in BETA_VALUES:
                for gamma in GAMMA_VALUES:
                    _, _, w_hat = residual_pursuit_nd(
                        w, n_steps=n, beta=beta, order2_gamma=gamma,
                        return_bases=False,
                    )
                    m = compute_metrics(w, w_hat)
                    row = {
                        "N": n,
                        "matrix_type": arch_name,
                        "beta": beta,
                        "gamma": gamma,
                        **m,
                    }
                    yield row
                    count += 1
                    if count % 200 == 0:
                        print(f"  ... {count}/{total} rows", flush=True)


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def print_best_rows(rows: list[dict], n: int = 5):
    """Print top-N rows per matrix type sorted by SNR."""
    from collections import defaultdict

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["matrix_type"]].append(r)

    for arch in ARCHETYPES:
        arch_rows = sorted(by_type[arch], key=lambda r: r["snr_db"], reverse=True)
        print(f"\n{'='*80}")
        print(f"  Matrix type: {arch}")
        print(f"  {'N':>4}  {'β':>6}  {'γ':>6}  {'MSE':>10}  {'SNR':>8}  {'CosSim':>8}")
        print(f"  {'-'*56}")
        for r in arch_rows[:n]:
            print(f"  {r['N']:4d}  {r['beta']:6.2f}  {r['gamma']:6.2f}  "
                  f"{r['mse']:10.2e}  {r['snr_db']:7.2f}  {r['cosine_similarity']:8.5f}")


def find_best_gamma_per_beta(rows: list[dict]):
    """For each (N, matrix_type, beta), find the best gamma."""
    from collections import defaultdict

    groups = defaultdict(list)
    for r in rows:
        key = (r["N"], r["matrix_type"], r["beta"])
        groups[key].append(r)

    print(f"\n{'='*80}")
    print(f"  Best γ per (N, matrix_type, β) — by SNR")
    print(f"  {'N':>4}  {'Type':<18}  {'β':>6}  {'best γ':>8}  {'SNR':>8}  {'CosSim':>8}")
    print(f"  {'-'*64}")

    for key in sorted(groups.keys()):
        candidates = groups[key]
        best = max(candidates, key=lambda r: r["snr_db"])
        print(f"  {best['N']:4d}  {best['matrix_type']:<18}  "
              f"{best['beta']:6.2f}  {best['gamma']:8.2f}  "
              f"{best['snr_db']:7.2f}  {best['cosine_similarity']:8.5f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Order-2 Σ-Δ sweep")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output CSV")
    parser.add_argument("--n-tiles", type=int, default=256,
                        help="Number of tiles to simulate (default: 256)")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of top rows to show per matrix type")
    args = parser.parse_args()

    print(f"Device: {DEVICE}, tiles: {args.n_tiles}")
    print(f"Sweep: N={N_STEPS_VALUES}, β={BETA_VALUES}, γ={GAMMA_VALUES}")
    print(f"Archetypes: {ARCHETYPES}")
    total_combos = len(N_STEPS_VALUES) * len(ARCHETYPES) * len(BETA_VALUES) * len(GAMMA_VALUES)
    print(f"Total combinations: {total_combos}")

    rows = list(run_sweep(args.n_tiles))
    print(f"\n  Done. {len(rows)} rows collected.")

    # Print summary
    print_best_rows(rows, n=args.top_n)
    find_best_gamma_per_beta(rows)

    # Compute aggregate stats
    gamma0_rows = [r for r in rows if r["gamma"] == 0.0]
    best_rows = []
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        key = (r["N"], r["matrix_type"], r["beta"])
        groups[key].append(r)
    for key, candidates in groups.items():
        best_rows.append(max(candidates, key=lambda r: r["snr_db"]))

    avg_gain = 0.0
    count_gain = 0
    for r_best, r_g0 in zip(
        sorted(best_rows, key=lambda r: (r["N"], r["matrix_type"], r["beta"])),
        sorted(gamma0_rows, key=lambda r: (r["N"], r["matrix_type"], r["beta"])),
    ):
        gain = r_best["snr_db"] - r_g0["snr_db"]
        if gain > 0:
            avg_gain += gain
            count_gain += 1

    if count_gain > 0:
        print(f"\n  Average SNR gain from best γ vs γ=0: {avg_gain/count_gain:.2f} dB "
              f"(across {count_gain}/{len(gamma0_rows)} configs where γ>0 helps)")

    # Save CSV if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["N", "matrix_type", "beta", "gamma", "mse", "snr_db", "cosine_similarity"]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  CSV saved to: {output_path}")


if __name__ == "__main__":
    main()