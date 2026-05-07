"""
§A Roadmap 3 — DCT vs FWHT vs None Transform Ablation
=====================================================

Runs eval_precision.run_evaluation() with three transform_mode values:
    none   — no transform (spatial-domain baseline)
    dct    — DCT Type-II energy compaction
    fwht   — legacy Walsh-Hadamard (flat spectrum)

Llama 3.2 1B, greedy argmax, 50 tokens.  Logs KL, cos_sim, token_match_rate,
first_divergence_position, and wall-clock time per mode.

Usage:
    python _ablation_transform_roadmap3.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.eval_precision import run_evaluation

logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("ablation_transform_roadmap3")


MODEL_PATH = "D:/Software_Development/Project/models/Llama-3.2-1B"
PROMPT = "The future of artificial intelligence lies in"
MAX_TOKENS = 50


def run_single(mode: str) -> tuple[dict, float]:
    """Run eval_precision.run_evaluation with transform_mode=mode.
    Returns (results_dict, elapsed_seconds).
    """
    # Disable differential residual + cross-token grouping when testing transforms:
    #   1) Differential path: decode(bases) is in transform domain while
    #      original is spatial → shape mismatch in Σ-Δ residual loop.
    #   2) Cross-token grouping (G=4) reshapes (N,64)→(N//2,128) for K before
    #      DCT; inverse transform + unreshape can interact badly during decode.
    cfg_override = {
        "transform_mode": mode,
        "use_differential": False,
        "cross_token_group": 1,
        "v_orthogonal_transform": False,  # V rotation interacts badly with
                                          # cross-token grouping + DCT shape
    }
    _logger.info(f"\n{'─'*60}")
    _logger.info(f"Mode: transform_mode='{mode}'")
    _logger.info(f"{'─'*60}")

    t0 = time.perf_counter()
    results = run_evaluation(
        model_path=MODEL_PATH,
        prompt=PROMPT,
        max_new_tokens=MAX_TOKENS,
        temperature=1.0,
        do_sample=False,          # greedy argmax for deterministic comparison
        quality="balanced",
        cfg_override=cfg_override,
    )
    elapsed = time.perf_counter() - t0
    return results, elapsed


def summary_row(results: dict, elapsed: float) -> str:
    """One-row summary for the comparison table."""
    lp = results["logit_precision"]
    tp = results["token_precision"]
    co = results["compression"]
    la = results["latency"]

    kl = lp["KL_divergence"]["mean"]
    cos = lp["Cosine_similarity"]["mean"]
    match = tp["token_match_rate"] * 100
    first_div = tp["first_divergence_position"]
    ds_us = la["ds_us_per_token"]
    ds_overhead = la["ds_overhead_pct"]
    comp = co["compression_ratio"]
    return (
        f"  {kl:>9.6f}  {cos:>9.6f}  {match:>6.2f}%  "
        f"{str(first_div):>9}  {ds_us:>8.0f}  {ds_overhead:>+7.1f}%  "
        f"{comp:>5.1f}x"
    )


def main():
    modes = [
        ("none", "No transform (spatial)"),
        ("dct",  "DCT Type-II"),
        ("fwht", "FWHT (Walsh-Hadamard)"),
    ]

    headers = [
        "Mode",
        "KL(mean)  Cos(mean)  Match%  1stDiv  µs/tok  Overhead  Comp",
    ]

    print("\n" + "=" * 80)
    print("  §A Roadmap 3 — Transform Mode Ablation (Llama 3.2 1B)")
    print("=" * 80)

    rows = []
    full = {}
    for mode, label in modes:
        results, elapsed = run_single(mode)
        row = summary_row(results, elapsed)
        rows.append((label, row))
        full[mode] = {
            "label": label,
            "elapsed_seconds": elapsed,
            "results": results,
        }
        _logger.info(f"  → Done in {elapsed:.1f}s")

    print("\n" + "=" * 80)
    print("  Comparison Summary")
    print("=" * 80)
    print(f"  {'─'*62}")
    print(f"  {headers[0]:<20} {headers[1]}")
    print(f"  {'─'*62}")
    for label, row in rows:
        print(f"  {label:<20} {row}")
    print(f"  {'─'*62}")
    print()

    print("=" * 80)
    print("  Full per-mode reports:")
    print("=" * 80)
    from scripts.eval_precision import print_report
    for mode, label in modes:
        d = full[mode]
        print(f"\n═══ {label} (transform_mode='{mode}') ═══")
        print_report(d["results"])

    # Save raw JSON
    out_path = Path("ablation_transform_results.json")
    out_path.write_text(json.dumps(full, indent=2, default=str), encoding="utf-8")
    _logger.info(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()