"""
§A 三路线 Generation 端到端对比
================================

12 组配置 × 同一 prompt 的自回归生成对比 + 原生 Llama FP16 基线。

运行::
    python _eval_generation_routes.py

依赖:
    pip install rouge-score  # 若无则自动 fallback 到 char-level match
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, ".")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rina.config import DSKVCacheConfig
from rina.model_wrapper import DSKVCacheModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_logger = logging.getLogger("eval_gen_routes")

# ══════════════════════════════════════════════════════════════════════════════
# Prompt bank
# ══════════════════════════════════════════════════════════════════════════════

PROMPTS = [
    "The future of AI is",
    "Once upon a time in a galaxy far far away",
    "The capital of France is",
    "To solve the equation x^2 + 2x + 1 = 0 we",
    "Deep learning has revolutionized computer vision by",
    "The three laws of thermodynamics state that",
    "In quantum mechanics the Schrödinger equation describes",
    "Python is a high-level programming language that",
]

# ══════════════════════════════════════════════════════════════════════════════
# Route config builder
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RouteSpec:
    """One experimental configuration to evaluate."""
    name: str
    label: str
    transform_mode: str = "none"
    adaptive_masking: bool = False
    cross_head_error_share: bool = False
    extra_kwargs: dict = field(default_factory=dict)


def build_routes() -> List[RouteSpec]:
    """Return the 12-route ablation matrix."""
    return [
        # ── Baseline ──
        RouteSpec("baseline",      "0_Baseline",         "none",  False, False),
        # ── R3: Transform ──
        RouteSpec("R3_DCT",        "R3_DCT",             "dct",   False, False),
        RouteSpec("R3_DWT",        "R3_DWT",             "dwt",   False, False),
        RouteSpec("R3_AUTO",       "R3_AUTO",            "auto",  False, False),
        # ── R1: Adaptive masking ──
        RouteSpec("R1_Mask",       "R1_AdaptMask",       "none",  True,  False),
        RouteSpec("R1_MaskStrong", "R1_MaskStrong",      "none",  True,  False,
                  extra_kwargs={"mask_outlier_threshold": 2.0, "mask_n_steps_boost": 2}),
        # ── R1+R3 combo ──
        RouteSpec("R1R3_AUTO",     "R1+R3_AUTO+Mask",    "auto",  True,  False),
        RouteSpec("R1R3_DCT",      "R1+R3_DCT+Mask",     "dct",   True,  False),
        # ── R2: Cross-head residual ──
        RouteSpec("R2_CrossHead",  "R2_CrossHeadRes",    "none",  False, True),
        # ── R2+R3 ──
        RouteSpec("R2R3_DCT",      "R2+DCT_CrossHead",   "dct",   False, True),
        # ── R1+R2+R3 full stack ──
        RouteSpec("R1R2R3",        "R1+R2+R3_Full",      "dct",   True,  True),
        # ── R1+R2+R3 (AUTO) ──
        RouteSpec("R1R2R3_AUTO",   "R1+R2+R3_AUTO",      "auto",  True,  True),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Config factory
# ══════════════════════════════════════════════════════════════════════════════

def make_config(route: RouteSpec) -> DSKVCacheConfig:
    """Build a DSKVCacheConfig from a route spec."""
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
        cross_token_group=2,
        use_recon_weights=False,
        cross_head_error_share=route.cross_head_error_share,
        transform_mode=route.transform_mode,
        adaptive_masking=route.adaptive_masking,
        mask_outlier_threshold=route.extra_kwargs.get("mask_outlier_threshold", 3.0),
        mask_n_steps_boost=route.extra_kwargs.get("mask_n_steps_boost", 1),
        mask_proj_beta_boost=route.extra_kwargs.get("mask_proj_beta_boost", 0.5),
        base_dtype="fp16",
        verbose=False,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Text comparison utilities
# ══════════════════════════════════════════════════════════════════════════════

def char_match_ratio(a: str, b: str) -> float:
    """Character-level overlap ratio."""
    mn = min(len(a), len(b))
    mx = max(len(a), len(b))
    if mx == 0:
        return 1.0
    matches = sum(1 for i in range(mn) if a[i] == b[i])
    return matches / mx


def prefix_match_len(a: str, b: str) -> int:
    """Length of common prefix in characters."""
    n = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            n += 1
        else:
            break
    return n


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Three-route generation ablation")
    p.add_argument("--model", type=str,
                   default="D:/Software_Development/Project/models/Llama-3.2-1B")
    p.add_argument("--max-tokens", type=int, default=50)
    p.add_argument("--prompts", type=str, nargs="*", default=None,
                   help="Override prompt list")
    p.add_argument("--output", type=str, default="eval_generation_routes.csv")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--skip-native", action="store_true", default=False,
                   help="Skip native FP16 baseline (faster for debugging)")
    args = p.parse_args()

    prompts = args.prompts if args.prompts else PROMPTS
    routes = build_routes()
    use_native = not args.skip_native

    _logger.info(f"Model: {args.model}")
    _logger.info(f"Max tokens: {args.max_tokens}")
    _logger.info(f"Prompts: {len(prompts)}")
    _logger.info(f"Routes: {len(routes)}")
    _logger.info(f"Native baseline: {use_native}")

    # ── Load tokenizer (shared) ──
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Native FP16 baseline (once per prompt) ──
    native_outputs: Dict[str, str] = {}
    if use_native:
        _logger.info("Loading native FP16 model for baseline ...")
        model_native = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model_native.eval()
        for prompt in prompts:
            _logger.info(f"Native baseline: \"{prompt[:40]}...\"")
            inputs = tokenizer(prompt, return_tensors="pt").to(model_native.device)
            with torch.no_grad():
                out = model_native.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            native_outputs[prompt] = text
            _logger.info(f"  => \"{text[len(prompt):][:60]}...\"")
        # Free native model
        del model_native
        torch.cuda.empty_cache()
        _logger.info("Native baseline complete.")

    # ── Collect results ──
    results: List[dict] = []

    # We cache model+tokenizer loads to avoid re-loading per route
    # Each route gets its own DSKVCacheModel wrapper on top
    for ri, route in enumerate(routes):
        cfg = make_config(route)
        _logger.info(f"[{ri+1}/{len(routes)}] {route.label}: "
                     f"xf={route.transform_mode} mask={route.adaptive_masking} "
                     f"crosshead={route.cross_head_error_share}")

        t0 = time.time()
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            model.eval()
            wrapper = DSKVCacheModel(model, tokenizer, cfg=cfg)

            for prompt in prompts:
                _logger.info(f"  Prompt: \"{prompt[:40]}...\"")
                try:
                    gen_text = wrapper.generate(
                        prompt,
                        max_new_tokens=args.max_tokens,
                        temperature=1.0,
                        do_sample=True,
                    )
                except Exception as e:
                    _logger.error(f"  GENERATION FAILED: {e}", exc_info=True)
                    gen_text = f"[ERROR: {e}]"

                native_text = native_outputs.get(prompt, "")
                prompt_only = prompt
                gen_new = gen_text[len(prompt_only):] if gen_text.startswith(prompt_only) else gen_text
                nat_new = native_text[len(prompt_only):] if native_text.startswith(prompt_only) else native_text

                results.append({
                    "route": route.name,
                    "label": route.label,
                    "transform_mode": route.transform_mode,
                    "adapt_mask": int(route.adaptive_masking),
                    "cross_head": int(route.cross_head_error_share),
                    "prompt": prompt_only[:40],
                    "native_output": nat_new[:200],
                    "route_output": gen_new[:200],
                    "char_match": round(char_match_ratio(gen_new, nat_new), 4) if native_text else None,
                    "prefix_match": prefix_match_len(gen_new, nat_new) if native_text else None,
                    "time_s": round(time.time() - t0, 1),
                })

            # Free model for next route
            del wrapper, model
            torch.cuda.empty_cache()

        except Exception as e:
            _logger.error(f"Route {route.label} FAILED: {e}")
            for prompt in prompts:
                results.append({
                    "route": route.name,
                    "label": route.label,
                    "transform_mode": route.transform_mode,
                    "adapt_mask": int(route.adaptive_masking),
                    "cross_head": int(route.cross_head_error_share),
                    "prompt": prompt[:40],
                    "native_output": "",
                    "route_output": f"[ROUTE ERROR: {e}]",
                    "char_match": None,
                    "prefix_match": None,
                    "time_s": round(time.time() - t0, 1),
                })

    # ── Write CSV ──
    if results:
        fieldnames = ["route", "label", "transform_mode", "adapt_mask", "cross_head",
                       "prompt", "char_match", "prefix_match", "native_output",
                       "route_output", "time_s"]
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)

        # ── Summary table ──
        print(f"\n{'='*80}")
        print(f"{'Route':<24s} {'CharMatch':>10s} {'Prefix':>8s} {'Time(s)':>8s}")
        print(f"{'-'*50}")
        from collections import defaultdict
        summary: Dict[str, List[float]] = defaultdict(list)
        for r in results:
            if r["char_match"] is not None:
                summary[r["label"]].append(r["char_match"])
        for label in sorted(summary.keys()):
            vals = summary[label]
            avg = sum(vals) / len(vals)
            print(f"{label:<24s} {avg:>10.4f}")
        print(f"\nResults saved to {args.output}")
    else:
        _logger.warning("No results generated.")

    print(f"\n[OK] eval_generation_routes complete.")


if __name__ == "__main__":
    main()