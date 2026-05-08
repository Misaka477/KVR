"""
R1 vs Baseline vs Mask-Gating Ablation
======================================

Quantitatively validate that mask-based gating improves reconstruction
quality, especially for R1 (adaptive_masking) with partially-filled tiles.

Uses a SINGLE model instance across all routes to eliminate
reload-induced non-determinism.  Fixed random seeds.

Run::
    python scripts/eval_padding_masking.py
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

sys.path.insert(0, ".")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rina.config import DSKVCacheConfig
from rina.model_wrapper import DSKVCacheModel

# ── Determinism: fix ALL random backends ──────────────────────────────
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_logger = logging.getLogger("eval_mask_gating")

PROMPTS = [
    "The future of AI is",
    "Once upon a time in a galaxy far far away",
    "The capital of France is",
    "To solve the equation x^2 + 2x + 1 = 0 we",
    "Deep learning has revolutionized computer vision by",
    "The three laws of thermodynamics state that",
    "In quantum mechanics the Schr\u00f6dinger equation describes",
    "Python is a high-level programming language that",
]


@dataclass
class MaskRoute:
    name: str
    label: str
    adaptive_masking: bool = False
    use_mask_gating: bool = False
    extra_kwargs: dict = field(default_factory=dict)


def build_routes() -> List[MaskRoute]:
    return [
        MaskRoute("native",       "native",         adaptive_masking=False, use_mask_gating=False),
        MaskRoute("baseline",     "baseline",       adaptive_masking=False, use_mask_gating=False),
        MaskRoute("baseline_mask","baseline_mask",  adaptive_masking=False, use_mask_gating=True),
        MaskRoute("r1",           "r1",             adaptive_masking=True,  use_mask_gating=False),
        MaskRoute("r1_mask",      "r1_mask",        adaptive_masking=True,  use_mask_gating=True),
    ]


def make_config(route: MaskRoute, cross_token_group: int = 2) -> Optional[DSKVCacheConfig]:
    if route.name == "native":
        return None
    return DSKVCacheConfig(
        n_steps=5,
        n_steps_k=5,
        n_steps_v=5,
        tile_size=16,
        beta=0.12,
        use_noise_shaping=True,
        proj_rank=8,
        proj_beta=0.4,
        adaptive_eta=True,
        adaptive_n=False,
        use_differential=True,
        diff_strategy="residual",
        diff_residual_gamma=0.25,
        diff_residual_n_steps=2,
        v_orthogonal_transform=True,
        order2_gamma=0.15,
        cross_token_group=cross_token_group,
        use_recon_weights=False,
        cross_head_error_share=False,
        transform_mode="none",
        adaptive_masking=route.adaptive_masking,
        mask_outlier_threshold=route.extra_kwargs.get("mask_outlier_threshold", 3.0),
        mask_n_steps_boost=route.extra_kwargs.get("mask_n_steps_boost", 0),
        mask_proj_beta_boost=route.extra_kwargs.get("mask_proj_beta_boost", 0.0),
        use_mask_gating=route.use_mask_gating,
        base_dtype="fp16",
        verbose=False,
    )


def char_match_ratio(a: str, b: str) -> float:
    mn = min(len(a), len(b))
    mx = max(len(a), len(b))
    if mx == 0:
        return 1.0
    matches = sum(1 for i in range(mn) if a[i] == b[i])
    return matches / mx


def prefix_match_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            n += 1
        else:
            break
    return n


def _run_route_greedy(
    wrapper: DSKVCacheModel,
    tokenizer,
    prompts: List[str],
    max_tokens: int,
) -> Dict[str, str]:
    """Generate text for all prompts using greedy decoding (deterministic)."""
    outputs: Dict[str, str] = {}
    for prompt in prompts:
        _logger.info(f"    Prompt: \"{prompt[:40]}...\"")
        try:
            gen_text = wrapper.generate(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=False,
            )
        except Exception as e:
            _logger.error(f"    GENERATION FAILED: {e}", exc_info=True)
            gen_text = f"[ERROR: {e}]"
        outputs[prompt] = gen_text
    return outputs


def main():
    p = argparse.ArgumentParser(description="Mask-gating ablation (single-model, deterministic)")
    p.add_argument("--model", type=str,
                   default="D:/Software_Development/Project/models/Llama-3.2-1B")
    p.add_argument("--max-tokens", type=int, default=50)
    p.add_argument("--prompts", type=str, nargs="*", default=None,
                   help="Override prompt list")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--cross-token-group", type=int, default=2,
                   help="Cross-token grouping size (1=off, 2=default)")
    args = p.parse_args()

    prompts = args.prompts if args.prompts else PROMPTS
    routes = build_routes()
    ctg = args.cross_token_group

    _logger.info(f"Model: {args.model}")
    _logger.info(f"Max tokens: {args.max_tokens}")
    _logger.info(f"Prompts: {len(prompts)}")
    _logger.info(f"cross_token_group: {ctg}")
    _logger.info(f"Seed: {SEED}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load model ONCE, share across all DS routes ───────────────────
    _logger.info("Loading model (shared across all routes) ...")
    shared_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    shared_model.eval()

    # ── Native FP16 baseline (greedy, same model instance) ────────────
    native_outputs: Dict[str, str] = {}
    _logger.info("Native baseline (greedy, same model) ...")
    for prompt in prompts:
        _logger.info(f"  Native: \"{prompt[:40]}...\"")
        inputs = tokenizer(prompt, return_tensors="pt").to(shared_model.device)
        with torch.no_grad():
            out = shared_model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        native_outputs[prompt] = text
        _logger.info(f"    => \"{text[len(prompt):][:60]}...\"")
    _logger.info("Native baseline complete.")

    # ── DS routes: share model instance, reset between routes ─────────
    results: List[dict] = []
    ds_routes = [r for r in routes if r.name != "native"]

    for ri, route in enumerate(ds_routes):
        cfg = make_config(route, cross_token_group=ctg)
        label = route.label

        _logger.info(f"[{ri+1}/{len(ds_routes)}] {label}: "
                     f"adaptive_masking={route.adaptive_masking} "
                     f"use_mask_gating={route.use_mask_gating}")

        t0 = time.time()
        try:
            # Create fresh wrapper — this is the only "reset" between routes.
            # The wrapper creates fresh DSKVCacheStore instances internally
            # via _bulk_encode_from_prefill, so no stale KV persists.
            wrapper = DSKVCacheModel(shared_model, tokenizer, cfg=cfg)
            route_outputs = _run_route_greedy(wrapper, tokenizer, prompts, args.max_tokens)

            for prompt in prompts:
                gen_text = route_outputs.get(prompt, f"[MISSING: {prompt[:20]}]")
                native_text = native_outputs.get(prompt, "")
                prompt_only = prompt
                gen_new = gen_text[len(prompt_only):] if gen_text.startswith(prompt_only) else gen_text
                nat_new = native_text[len(prompt_only):] if native_text.startswith(prompt_only) else native_text

                results.append({
                    "route": label,
                    "prompt": prompt_only[:40],
                    "char_match": round(char_match_ratio(gen_new, nat_new), 4) if native_text else None,
                    "prefix_match": prefix_match_len(gen_new, nat_new) if native_text else None,
                    "time_s": round(time.time() - t0, 1),
                    "route_output": gen_new[:200],
                    "native_output": nat_new[:200],
                })

            # Release wrapper (and its internal DS stores)
            del wrapper

        except Exception as e:
            _logger.error(f"Route {label} FAILED: {e}", exc_info=True)
            for prompt in prompts:
                results.append({
                    "route": label,
                    "prompt": prompt[:40],
                    "char_match": None,
                    "prefix_match": None,
                    "time_s": round(time.time() - t0, 1),
                    "route_output": f"[ROUTE ERROR: {e}]",
                    "native_output": "",
                })

    # ── Native entries (already computed) ─────────────────────────────
    for prompt in prompts:
        native_text = native_outputs.get(prompt, "")
        prompt_only = prompt
        nat_new = native_text[len(prompt_only):] if native_text.startswith(prompt_only) else native_text
        results.append({
            "route": "native",
            "prompt": prompt_only[:40],
            "char_match": 1.0,
            "prefix_match": None,
            "time_s": 0.0,
            "route_output": nat_new[:200],
            "native_output": nat_new[:200],
        })

    # ── Clean up shared model ──────────────────────────────────────────
    del shared_model
    torch.cuda.empty_cache()

    # ── Print summary table ────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("=== R1 vs Baseline -> Mask Gating Ablation ===")
    print(f"{'Route':<18s} {'char_match':>10s} {'prefix_match':>12s} {'time':>8s}")
    print(f"{'-'*50}")

    summary: Dict[str, List[float]] = defaultdict(list)
    summary_prefix: Dict[str, List[int]] = defaultdict(list)
    summary_time: Dict[str, List[float]] = defaultdict(list)

    for r in results:
        if r["char_match"] is not None:
            summary[r["route"]].append(r["char_match"])
        if r["prefix_match"] is not None:
            summary_prefix[r["route"]].append(r["prefix_match"])
        summary_time[r["route"]].append(r["time_s"])

    for label in ["native", "baseline", "baseline_mask", "r1", "r1_mask"]:
        vals = summary.get(label, [])
        prefix_vals = summary_prefix.get(label, [])
        time_vals = summary_time.get(label, [])
        if vals:
            avg_char = sum(vals) / len(vals)
            avg_prefix = sum(prefix_vals) / len(prefix_vals) if prefix_vals else 0
            avg_time = sum(time_vals) / len(time_vals) if time_vals else 0
            # Show per-prompt details safely (avoid unicode issues)
            for rp in [r for r in results if r["route"] == label]:
                try:
                    gen_snippet = rp["route_output"][:30].encode('ascii', errors='replace').decode('ascii')
                except Exception:
                    gen_snippet = "[?]"
                print(f"  [{label}] {rp['prompt'][:30]:<30s} char={rp['char_match']} "
                      f"pref={rp['prefix_match']} gen=\"{gen_snippet}\"")
            print(f"{label:<18s} {avg_char:>10.4f} {avg_prefix:>12.1f} {avg_time:>8.1f}")

    print(f"\n[OK] eval_padding_masking complete.")


if __name__ == "__main__":
    main()
