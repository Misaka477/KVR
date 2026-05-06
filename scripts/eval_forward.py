r"""
DS-KVCache End-to-End Forward Verification
===========================================

Validates the full DS-KVCache pipeline without requiring a HF model:

  1.  Synthetic K/V generation (random or sinusoid)
  2.  Bulk encode → decode → metric comparison
  3.  Incremental encode (ring buffer) → finalize → metric comparison
  4.  Compression-ratio report
  5.  A/B comparison: vanilla RBP vs noise-shaped RBP vs differential RBP

Run::

    python scripts/eval_forward.py

    python scripts/eval_forward.py --n-tokens 512 --d-head 128 --n-steps 6 --diff
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from tabulate import tabulate

from rina.config import DSKVCacheConfig
from rina.ds_kv_cache import (
    DSKVCacheStore,
    encode_kv_cache,
    decode_kvcache_store,
)
from rina.incremental_decode import (
    init_incremental_store,
    incremental_encode_step,
    finalize_store,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("eval_forward")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def make_synthetic_kv(
    n_tokens: int,
    d_head: int,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic K, V mimicking real attention projections.

    Uses sinusoid features + noise — approximates real attention
    statistics better than pure Gaussian.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    # Position-dependent sinusoid
    pos = torch.arange(n_tokens, dtype=torch.float32).unsqueeze(1)
    freqs = torch.linspace(1.0, 10.0, d_head // 2).unsqueeze(0)
    sin_part = torch.sin(pos * freqs)
    cos_part = torch.cos(pos * freqs)
    structure = torch.cat([sin_part, cos_part], dim=1)[:, :d_head]

    # Noise floor (real attention matrices are structured, not pure Gaussian)
    noise_k = torch.randn(n_tokens, d_head, generator=gen) * 0.3
    noise_v = torch.randn(n_tokens, d_head, generator=gen) * 0.3

    k = structure + noise_k
    v = structure * 0.8 + noise_v
    return k, v


def compute_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    tag: str = "",
) -> dict:
    """Return MSE, SNR, CosSim for a (N_tokens, d_head) pair."""
    o = original.float()
    r = reconstructed.float()

    mse = F.mse_loss(r, o).item()
    signal_power = (o**2).mean().item()
    noise_power = ((o - r)**2).mean().item()
    snr_db = 10 * math.log10(signal_power / (noise_power + 1e-12))

    # Cosine similarity (flattened)
    cos_sim = F.cosine_similarity(
        r.flatten().unsqueeze(0),
        o.flatten().unsqueeze(0),
    ).item()

    return {
        "tag": tag,
        "MSE": mse,
        "SNR_dB": snr_db,
        "CosSim": cos_sim,
    }


def _mem_mb(t: torch.Tensor) -> float:
    """Tensor → MB."""
    return t.element_size() * t.numel() / (1024**2)


# ══════════════════════════════════════════════════════════════════════════════
# Test cases
# ══════════════════════════════════════════════════════════════════════════════


def test_bulk_encode_decode(
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: DSKVCacheConfig,
) -> dict:
    """Encode and decode K/V in bulk, return metrics."""
    k_store, v_store = encode_kv_cache(k, v, cfg)

    k_hat = decode_kvcache_store(k_store, cfg.tile_size, cfg.use_differential)
    v_hat = decode_kvcache_store(v_store, cfg.tile_size, cfg.use_differential)

    metrics_k = compute_metrics(k, k_hat, "K")
    metrics_v = compute_metrics(v, v_hat, "V")

    # Compression report
    orig_mb = _mem_mb(k) + _mem_mb(v)
    ds_mb = (k_store.memory_bytes + v_store.memory_bytes) / (1024**2)

    return {
        **{f"K_{k}": v for k, v in metrics_k.items()},
        **{f"V_{k}": v for k, v in metrics_v.items()},
        "CompRatio": orig_mb / (ds_mb + 1e-12),
        "Orig_MB": orig_mb,
        "DS_MB": ds_mb,
    }


def test_incremental_encode_decode(
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: DSKVCacheConfig,
) -> dict:
    """Incrementally encode K/V token-by-token and finalize."""
    n_tokens, d_head = k.shape

    k_store = init_incremental_store(d_head, cfg)
    v_store = init_incremental_store(d_head, cfg)

    for t in range(n_tokens):
        k_store = incremental_encode_step(k[t], k_store, cfg, is_key=True)
        v_store = incremental_encode_step(v[t], v_store, cfg, is_key=False)

    k_store = finalize_store(k_store, cfg, is_key=True)
    v_store = finalize_store(v_store, cfg, is_key=False)

    k_hat = k_store.full_k_hat
    v_hat = v_store.full_k_hat

    metrics_k = compute_metrics(k, k_hat, "K")
    metrics_v = compute_metrics(v, v_hat, "V")

    orig_mb = _mem_mb(k) + _mem_mb(v)
    ds_mb = (k_store.memory_bytes + v_store.memory_bytes) / (1024**2)

    return {
        **{f"K_{k}": v for k, v in metrics_k.items()},
        **{f"V_{k}": v for k, v in metrics_v.items()},
        "CompRatio": orig_mb / (ds_mb + 1e-12),
        "Orig_MB": orig_mb,
        "DS_MB": ds_mb,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def main():
    p = argparse.ArgumentParser(description="DS-KVCache forward verification")
    p.add_argument("--n-tokens", type=int, default=512,
                   help="Sequence length (tokens)")
    p.add_argument("--d-head", type=int, default=128,
                   help="Head dimension")
    p.add_argument("--n-steps", type=int, default=5,
                   help="Number of 1-bit bases (oversampling ratio)")
    p.add_argument("--tile-size", type=int, default=16,
                   help="Tile size for block encoding")
    p.add_argument("--beta", type=float, default=0.15,
                   help="Σ-Δ momentum coefficient")
    p.add_argument("--no-ns", action="store_true",
                   help="Disable noise shaping")
    p.add_argument("--no-adaptive-n", action="store_true",
                   help="Disable adaptive N scheduling")
    p.add_argument("--diff", action="store_true",
                   help="Enable differential dual-path encoding")
    p.add_argument("--diff-strategy", type=str, default="momentum_shift",
                   choices=["momentum_shift", "extra_step", "eta_shift"],
                   help="Differential perturbation strategy")
    p.add_argument("--proj-rank", type=int, default=8,
                   help="Noise-shaping projection rank")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--all-configs", action="store_true",
                   help="Run A/B comparison of all configs")
    args = p.parse_args()

    # ── Synthetic data ────────────────────────────────────────────────
    k, v = make_synthetic_kv(args.n_tokens, args.d_head, args.seed)
    _logger.info(
        f"Synthetic KV: shape=({args.n_tokens}, {args.d_head}), "
        f"K energy={k.var().item():.4f}, V energy={v.var().item():.4f}"
    )

    # ── Base config ───────────────────────────────────────────────────
    base_cfg = DSKVCacheConfig(
        n_steps=args.n_steps,
        tile_size=args.tile_size,
        beta=args.beta,
        use_noise_shaping=not args.no_ns,
        proj_rank=args.proj_rank,
        proj_beta=0.3 if not args.no_ns else 0.0,
        adaptive_eta=not args.no_ns,
        adaptive_n=not args.no_adaptive_n,
        n_upper_bound=args.n_steps + 5,
        use_differential=args.diff,
        diff_strategy=args.diff_strategy, order2_gamma=0.0,
        base_dtype="fp16",
        verbose=True,
    )

    # ── Single test or A/B ────────────────────────────────────────────
    if args.all_configs:
        configs = _build_ab_configs(args)
        rows = []
        for name, cfg in configs:
            _logger.info(f"\n─── {name} ───")
            result = test_bulk_encode_decode(k, v, cfg)
            rows.append([
                name,
                f"{result['K_CosSim']:.4f}",
                f"{result['V_CosSim']:.4f}",
                f"{result['K_SNR_dB']:.1f}",
                f"{result['V_SNR_dB']:.1f}",
                f"{result['CompRatio']:.1f}x",
                f"{result['DS_MB']:.3f}",
            ])
        print("\n")
        print(tabulate(
            rows,
            headers=["Config", "K CosSim", "V CosSim", "K SNR", "V SNR", "Compress", "DS MB"],
            tablefmt="grid",
        ))
        print()
    else:
        _logger.info("\n─── Bulk ───")
        bulk = test_bulk_encode_decode(k, v, base_cfg)
        _logger.info(
            f"K: CosSim={bulk['K_CosSim']:.4f}, SNR={bulk['K_SNR_dB']:.1f}dB | "
            f"V: CosSim={bulk['V_CosSim']:.4f}, SNR={bulk['V_SNR_dB']:.1f}dB | "
            f"Compress: {bulk['CompRatio']:.1f}x ({bulk['Orig_MB']:.3f}→{bulk['DS_MB']:.3f} MB)"
        )

        _logger.info("\n─── Incremental ───")
        inc = test_incremental_encode_decode(k, v, base_cfg)
        _logger.info(
            f"K: CosSim={inc['K_CosSim']:.4f}, SNR={inc['K_SNR_dB']:.1f}dB | "
            f"V: CosSim={inc['V_CosSim']:.4f}, SNR={inc['V_SNR_dB']:.1f}dB | "
            f"Compress: {inc['CompRatio']:.1f}x ({inc['Orig_MB']:.3f}→{inc['DS_MB']:.3f} MB)"
        )

    print("\n[OK] DS-KVCache forward verification complete.")


def _build_ab_configs(args):
    """Return list of (name, DSKVCacheConfig) for A/B comparison."""
    cfgs = []

    # 1. Vanilla RBP (no noise shaping, no diff)
    cfgs.append(("Vanilla RBP", DSKVCacheConfig(
        n_steps=args.n_steps, tile_size=args.tile_size,
        beta=0.0, use_noise_shaping=False, use_differential=False,
        adaptive_n=False, base_dtype="fp16",
    )))

    # 2. RBP + Σ-Δ momentum
    cfgs.append(("RBP + Σ-Δ", DSKVCacheConfig(
        n_steps=args.n_steps, tile_size=args.tile_size,
        beta=args.beta, use_noise_shaping=False, use_differential=False,
        adaptive_n=False, base_dtype="fp16",
    )))

    # 3. RBP + Noise Shaping
    cfgs.append(("RBP + NS", DSKVCacheConfig(
        n_steps=args.n_steps, tile_size=args.tile_size,
        beta=0.0, use_noise_shaping=True, proj_rank=args.proj_rank,
        proj_beta=0.3, adaptive_eta=True, use_differential=False,
        adaptive_n=False, base_dtype="fp16",
    )))

    # 4. Full RINA: Σ-Δ + NS + Adaptive-N
    cfgs.append(("Full RINA (Σ-Δ+NS+AdpN)", DSKVCacheConfig(
        n_steps=args.n_steps, tile_size=args.tile_size,
        beta=args.beta, use_noise_shaping=True, proj_rank=args.proj_rank,
        proj_beta=0.3, adaptive_eta=True, use_differential=False,
        adaptive_n=True, n_upper_bound=args.n_steps + 5,
        base_dtype="fp16",
    )))

    # 5. Full + Differential
    cfgs.append(("RINA+Diff", DSKVCacheConfig(
        n_steps=args.n_steps, tile_size=args.tile_size,
        beta=args.beta, use_noise_shaping=True, proj_rank=args.proj_rank,
        proj_beta=0.3, adaptive_eta=True, use_differential=True,
        adaptive_n=True, n_upper_bound=args.n_steps + 5,
        diff_strategy="momentum_shift", base_dtype="fp16",
    )))

    return cfgs


if __name__ == "__main__":
    main()