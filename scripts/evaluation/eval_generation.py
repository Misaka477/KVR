"""
DS-KVCache End-to-End Generation Evaluation
=============================================

Compares DS-KVCache output vs baseline (FP16) on a prompt.

Run::
    python scripts/eval_generation.py
    python scripts/eval_generation.py --prompt "The future of AI is" --max-tokens 50 --do-sample
"""

from __future__ import annotations

import argparse
import logging
import sys
sys.path.insert(0, ".")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rina.config import DSKVCacheConfig
from rina.model_wrapper import DSKVCacheModel

logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("eval_generation")


def main():
    p = argparse.ArgumentParser(description="DS-KVCache end-to-end generation")
    p.add_argument("--model", type=str,
                   default="D:/Software_Development/Project/models/Llama-3.2-1B")
    p.add_argument("--prompt", type=str, default="The future of AI is")
    p.add_argument("--max-tokens", type=int, default=50)
    # ── Heterogeneous N ─────────────────────────────────────────────────
    p.add_argument("--n-steps", type=int, default=None,
                   help="Unified n_steps (fallback)")
    p.add_argument("--n-steps-k", type=int, default=3)
    p.add_argument("--n-steps-v", type=int, default=5)
    p.add_argument("--tile-size", type=int, default=16)
    p.add_argument("--beta", type=float, default=0.15)
    # ── Adaptive N ──────────────────────────────────────────────────────
    p.add_argument("--adaptive-n", action="store_true", default=False,
                   help="Enable adaptive N scheduling (off for consistent compression)")
    # ── Noise shaping ───────────────────────────────────────────────────
    p.add_argument("--ns", action="store_true", default=True, dest="use_ns",
                   help="Enable noise shaping (default: on)")
    p.add_argument("--no-ns", action="store_false", dest="use_ns",
                   help="Disable noise shaping")
    # ── Differential ────────────────────────────────────────────────────
    p.add_argument("--diff", action="store_true", default=True, dest="use_diff",
                   help="Enable differential encoding (default: on)")
    p.add_argument("--no-diff", action="store_false", dest="use_diff",
                   help="Disable differential encoding")
    p.add_argument("--diff-gamma", type=float, default=0.25,
                   help="Residual shrinkage γ (default: 0.25)")
    p.add_argument("--diff-residual-nsteps", type=int, default=1)
    # ── V orthogonal transform ──────────────────────────────────────────
    p.add_argument("--v-ortho", action="store_true", default=True, dest="v_ortho",
                   help="Enable V orthogonal transform (default: on)")
    p.add_argument("--no-v-ortho", action="store_false", dest="v_ortho",
                   help="Disable V orthogonal transform")
    # ── Cross-token + Σ-Δ ───────────────────────────────────────────────
    p.add_argument("--cross-token-group", type=int, default=4,
                   help="Cross-token joint encoding group size (default: 4)")
    p.add_argument("--order2-gamma", type=float, default=0.3,
                   help="Second-order Σ-Δ coupling (default: 0.3)")
    # ── Protected layers (§8.1.8) ────────────────────────────────────────
    p.add_argument("--protected-layers", type=str, default="",
                   help="Comma-separated layer indices to keep as FP16 (e.g. '0,15')")
    # ── Weighted reconstruction ──────────────────────────────────────────
    p.add_argument("--recon-weights", action="store_true", default=True, dest="use_recon_weights",
                   help="Enable weighted reconstruction (default: on)")
    p.add_argument("--no-recon-weights", action="store_false", dest="use_recon_weights",
                   help="Disable weighted reconstruction")
    # ── Layer step pyramid (§8.1.6) ──────────────────────────────────────
    p.add_argument("--layer-step-pyramid", action="store_true", default=False,
                   help="Enable per-layer adaptive step allocation (default: off)")
    p.add_argument("--no-layer-step-pyramid", action="store_false", dest="layer_step_pyramid",
                   help="Disable per-layer step allocation")
    # ── Cross-head error share (§8.1.9) ──────────────────────────────────
    p.add_argument("--cross-head-share", action="store_true", default=True, dest="cross_head_error_share",
                   help="Enable cross-head error sharing (default: on)")
    p.add_argument("--no-cross-head-share", action="store_false", dest="cross_head_error_share",
                   help="Disable cross-head error sharing")
    # ── Beta decay ───────────────────────────────────────────────────────
    p.add_argument("--beta-decay-start", type=float, default=0.30,
                   help="Initial beta at token 1 (default: 0.30)")
    p.add_argument("--beta-decay-end", type=float, default=0.05,
                   help="Final beta after decay window (default: 0.05)")
    p.add_argument("--beta-decay-tokens", type=int, default=0,
                   help="Number of tokens over which to decay beta (0=off, default: 0)")
    # ── Temperature ─────────────────────────────────────────────────────
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--do-sample", action="store_true", default=True)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    fallback_n = args.n_steps if args.n_steps is not None else max(args.n_steps_k, args.n_steps_v)

    # Parse protected layers
    protected_layers = []
    if args.protected_layers.strip():
        protected_layers = [int(x.strip()) for x in args.protected_layers.split(",") if x.strip()]

    # Parse layer step map: if pyramid enabled, use default; otherwise empty
    layer_step_map = None  # None = use config default (pyramid)
    if not args.layer_step_pyramid:
        layer_step_map = {}  # empty = disable per-layer override, fall back to global

    # Beta decay config
    beta_decay_config = {}
    if args.beta_decay_tokens > 0:
        beta_decay_config = {
            "beta_decay_start": args.beta_decay_start,
            "beta_decay_end": args.beta_decay_end,
            "beta_decay_tokens": args.beta_decay_tokens,
        }

    cfg = DSKVCacheConfig(
        n_steps=fallback_n,
        n_steps_k=args.n_steps_k,
        n_steps_v=args.n_steps_v,
        tile_size=args.tile_size,
        beta=args.beta,
        use_noise_shaping=args.use_ns,
        proj_rank=8,
        proj_beta=0.3 if args.use_ns else 0.0,
        adaptive_eta=args.use_ns,
        adaptive_n=args.adaptive_n,
        n_upper_bound=max(args.n_steps_k, args.n_steps_v) + 2 if args.adaptive_n else 10,
        use_differential=args.use_diff,
        diff_strategy="residual", diff_residual_gamma=args.diff_gamma,
        diff_residual_n_steps=args.diff_residual_nsteps,
        v_orthogonal_transform=args.v_ortho,
        order2_gamma=args.order2_gamma,
        cross_token_group=args.cross_token_group,
        protected_layers=protected_layers,
        use_recon_weights=args.use_recon_weights,
        cross_head_error_share=args.cross_head_error_share,
        layer_step_map=layer_step_map,
        base_dtype="fp16",
        verbose=False,
    )

    _logger.info(
        f"Config: n_steps_k={cfg.get_n_steps_k()}, n_steps_v={cfg.get_n_steps_v()}, "
        f"v_ortho={cfg.v_orthogonal_transform}, diff={cfg.use_differential}, "
        f"adaptive_n={cfg.adaptive_n}, cross_token_group={cfg.cross_token_group}, "
        f"order2_gamma={cfg.order2_gamma}, recon_weights={cfg.use_recon_weights}, "
        f"protected_layers={cfg.protected_layers}"
    )

    # ── DS-KVCache generation ──────────────────────────────────────────
    _logger.info(f"Loading {args.model} with DS-KVCache ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    wrapper = DSKVCacheModel(model, tokenizer, cfg=cfg)

    _logger.info(f"Prompt: \"{args.prompt}\"")
    result = wrapper.generate(
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
    )
    _logger.info(f"DS-KVCache output: \"{result}\"")

    # ── Baseline generation (FP16, no DS) ──────────────────────────────
    _logger.info(f"Loading {args.model} baseline ...")
    model_base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer_base = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    inputs = tokenizer_base(args.prompt, return_tensors="pt").to(model_base.device)
    with torch.no_grad():
        output_base = model_base.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            pad_token_id=tokenizer_base.eos_token_id,
        )
    baseline = tokenizer_base.decode(output_base[0], skip_special_tokens=True)
    _logger.info(f"Baseline output:  \"{baseline}\"")

    # ── Compare ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Prompt:         {args.prompt}")
    print(f"Baseline (FP16): {baseline}")
    print(f"DS-KVCache:     {result}")
    if result == baseline:
        print(f"  >> EXACT MATCH")
    else:
        # Character-level divergence
        match_chars = sum(1 for a, b in zip(result, baseline) if a == b)
        total = max(len(result), len(baseline))
        sim = match_chars / total * 100 if total > 0 else 0
        print(f"  >> char-level match: {sim:.1f}%")

    # ── Memory stats ───────────────────────────────────────────────────
    wrapper.print_stats()

    print(f"\n[OK] eval_generation complete.")


if __name__ == "__main__":
    main()