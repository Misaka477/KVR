#!/usr/bin/env python
"""
Phase 3 — 推迟首步发散的 4-way 精度优化实验 (v3)
================================================
对 Llama 3.2 1B 跑 4 组配置，每组记录：
  - first_divergence_position
  - token_match_rate
  - compression_ratio
  - mean CosSim (K/V)

实验组:
    A. baseline       — 当前最优 (cross_token_group=4, order2_gamma=0.3, recon_weights, cross_head_share)
    B. pyramid_protect — baseline + protected_layers=[0,15] 锁定首尾层为FP16
    C. adaptive_combo  — baseline + layer_step_pyramid + beta_decay (0.30→0.05 over 10 tokens)
    D. full_stack      — B + C 全开

用法:
    python scripts/exp_push_divergence.py
    python scripts/exp_push_divergence.py --max-tokens 80 --output results/push_divergence.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rina.config import DSKVCacheConfig
from rina.model_wrapper import DSKVCacheModel
from scripts.auto_config import detect_gpu_info, detect_model_info, generate_optimal_config

logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("exp_push_divergence")


@torch.no_grad()
def run_single_config(
    model,
    tokenizer,
    cfg: DSKVCacheConfig,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    label: str,
) -> Dict[str, Any]:
    """Run DS-KVCache generation with a given config and return metrics."""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]

    wrapper = DSKVCacheModel(model, tokenizer, cfg=cfg)

    # Prefill
    out = wrapper.model(input_ids=input_ids, use_cache=True)
    logits_list = [out.logits[0, -1, :].cpu()]
    
    wrapper._bulk_encode_from_prefill(out.past_key_values, input_ids)
    past = wrapper._build_past_from_ds()

    generated_ids = input_ids[0].tolist()

    # Greedy argmax
    first_id = int(out.logits[0, -1, :].argmax().item())
    if first_id == tokenizer.eos_token_id:
        return {"error": "EOS on first token", "label": label}
    generated_ids.append(first_id)

    t_start = time.perf_counter()

    for step in range(1, max_new_tokens):
        last_token = torch.tensor([[generated_ids[-1]]], device=device)
        out = wrapper.model(
            input_ids=last_token,
            use_cache=True,
            past_key_values=past,
        )
        logits_list.append(out.logits[0, -1, :].cpu())

        # Pass decode_step for dynamic beta decay (§8.1.11)
        wrapper._append_incremental(out.past_key_values, new_token_idx=-1, decode_step=step - 1)
        past = wrapper._build_past_from_ds()

        next_id = int(out.logits[0, -1, :].argmax().item())
        generated_ids.append(next_id)

        if next_id == tokenizer.eos_token_id:
            break

    elapsed = time.perf_counter() - t_start
    gen_tokens = len(generated_ids) - prompt_len

    # ── Baseline comparison ──
    base_inputs = tokenizer(prompt, return_tensors="pt").to(device)
    base_out = model(input_ids=base_inputs["input_ids"], use_cache=True)
    base_past = base_out.past_key_values
    base_ids = base_inputs["input_ids"][0].tolist()
    base_first = int(base_out.logits[0, -1, :].argmax().item())
    base_ids.append(base_first)

    for s in range(1, max_new_tokens):
        last_t = torch.tensor([[base_ids[-1]]], device=device)
        base_out = model(input_ids=last_t, use_cache=True, past_key_values=base_past)
        base_past = base_out.past_key_values
        nxt = int(base_out.logits[0, -1, :].argmax().item())
        base_ids.append(nxt)
        if nxt == tokenizer.eos_token_id:
            break

    # ── Token match ──
    min_len = min(len(generated_ids), len(base_ids))
    match_count = sum(1 for i in range(min_len) if generated_ids[i] == base_ids[i])
    token_match_rate = match_count / min_len if min_len > 0 else 0.0

    first_div = None
    for i in range(min_len):
        if generated_ids[i] != base_ids[i]:
            first_div = i - prompt_len  # relative to first generated token
            break

    # ── Compression ──
    stats_list = wrapper.get_stats() if wrapper is not None else []
    total_fp16 = sum(s["fp16_memory_bytes"] for s in stats_list)
    total_ds = sum(s["ds_memory_bytes"] for s in stats_list)
    comp_ratio = total_fp16 / (total_ds + 1e-12) if total_ds > 0 else 0.0

    return {
        "label": label,
        "n_steps_k": cfg.get_n_steps_k(),
        "n_steps_v": cfg.get_n_steps_v(),
        "proj_beta": cfg.proj_beta,
        "diff_residual_gamma": cfg.diff_residual_gamma,
        "order2_gamma": cfg.order2_gamma,
        "cross_token_group": cfg.cross_token_group,
        "protected_layers": str(cfg.protected_layers),
        "layer_step_map": "on" if cfg.layer_step_map else "off",
        "beta_decay": "on" if cfg.beta_decay_tokens > 0 else "off",
        "num_tokens_generated": gen_tokens,
        "first_divergence": first_div,
        "token_match_rate": round(token_match_rate, 4),
        "compression_ratio": round(comp_ratio, 2),
        "ds_memory_mb": round(total_ds / (1024**2), 2),
        "elapsed_s": round(elapsed, 2),
    }


def main():
    p = argparse.ArgumentParser(description="4-way precision optimization experiment (v3)")
    p.add_argument("--model", type=str, default="D:/Software_Development/Project/models/Llama-3.2-1B")
    p.add_argument("--prompt", type=str, default="The future of artificial intelligence lies in")
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--output", type=str, default=None, help="CSV output path")

    # ── Common base config overrides ──
    p.add_argument("--n-steps-k", type=int, default=3)
    p.add_argument("--n-steps-v", type=int, default=5)
    p.add_argument("--beta", type=float, default=0.15)
    p.add_argument("--tile-size", type=int, default=16)
    p.add_argument("--no-ns", action="store_false", dest="use_ns", default=True)
    p.add_argument("--no-diff", action="store_false", dest="use_diff", default=True)
    p.add_argument("--diff-gamma", type=float, default=0.25)
    p.add_argument("--v-ortho", action="store_true", default=True, dest="v_ortho")
    p.add_argument("--no-v-ortho", action="store_false", dest="v_ortho")

    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ── Hardware & Model ──
    gpu_info = detect_gpu_info()
    model_info = detect_model_info(args.model)
    device = torch.device(gpu_info["recommended_device"] if torch.cuda.is_available() else "cpu")

    _logger.info(f"GPU: {gpu_info['name']} ({gpu_info['vram_gb']} GB)")
    _logger.info(f"Model: {model_info['model_type']} L={model_info['num_layers']} "
                 f"GQA={model_info['gqa_ratio']}x d_head={model_info['d_head']}")

    # ── Load model ──
    _logger.info(f"Loading model {args.model} ...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hf_model.eval()

    # ── Define 4 experimental configs (v3) ──

    def _base_config(**overrides):
        """Build base config from CLI args + overrides."""
        from rina.config import _default_layer_step_map
        kwargs = {
            "n_steps_k": args.n_steps_k,
            "n_steps_v": args.n_steps_v,
            "tile_size": args.tile_size,
            "beta": args.beta,
            "use_noise_shaping": args.use_ns,
            "proj_rank": 8,
            "proj_beta": 0.3 if args.use_ns else 0.0,
            "adaptive_eta": args.use_ns,
            "use_differential": args.use_diff,
            "diff_strategy": "residual",
            "diff_residual_gamma": args.diff_gamma,
            "diff_residual_n_steps": 2,
            "v_orthogonal_transform": args.v_ortho,
            "order2_gamma": 0.3,
            "cross_token_group": 4,
            "protected_layers": [],
            "use_recon_weights": True,
            "cross_head_error_share": True,
            "layer_step_map": None,  # None = use config default (pyramid)
            "beta_decay_start": 0.30,
            "beta_decay_end": 0.05,
            "beta_decay_tokens": 0,  # off by default, enabled in C/D
            "base_dtype": "fp16",
            "verbose": False,
        }
        kwargs.update(overrides)
        return DSKVCacheConfig(**kwargs)

    configs = []

    # A: baseline — current best config
    cfg_a = _base_config()
    configs.append(("A_baseline", cfg_a))

    # B: pyramid_protect — baseline + protected first/last layers
    cfg_b = _base_config(protected_layers=[0, 15])
    configs.append(("B_pyramid_protect", cfg_b))

    # C: adaptive_combo — baseline + layer_step_pyramid + beta_decay
    from rina.config import _default_layer_step_map
    cfg_c = _base_config(
        layer_step_map=_default_layer_step_map(),
        beta_decay_start=0.30,
        beta_decay_end=0.05,
        beta_decay_tokens=10,
    )
    configs.append(("C_adaptive_combo", cfg_c))

    # D: full_stack — B + C
    cfg_d = _base_config(
        protected_layers=[0, 15],
        layer_step_map=_default_layer_step_map(),
        beta_decay_start=0.30,
        beta_decay_end=0.05,
        beta_decay_tokens=10,
    )
    configs.append(("D_full_stack", cfg_d))

    # ── Run experiments ──
    results = []
    for label, cfg in configs:
        _logger.info(f"\n{'='*60}")
        _logger.info(f"  Running: {label}")
        _logger.info(f"  n_steps_k={cfg.get_n_steps_k()}, n_steps_v={cfg.get_n_steps_v()}, "
                     f"beta={cfg.beta}, order2_gamma={cfg.order2_gamma}, "
                     f"cross_token_group={cfg.cross_token_group}")
        _logger.info(f"  protected_layers={cfg.protected_layers}, "
                     f"layer_step_map={'on' if cfg.layer_step_map else 'off'}, "
                     f"beta_decay={'on' if cfg.beta_decay_tokens > 0 else 'off'}")
        _logger.info(f"{'='*60}")

        result = run_single_config(
            hf_model, tokenizer, cfg, args.prompt,
            args.max_tokens, device, label,
        )
        results.append(result)

        _logger.info(f"  Result: first_divergence={result.get('first_divergence')}, "
                     f"match_rate={result.get('token_match_rate', 'N/A')}, "
                     f"compression={result.get('compression_ratio', 'N/A')}x")

    # ── Summary ──
    _logger.info(f"\n{'='*70}")
    _logger.info("  SUMMARY — 4-way Push Divergence Experiment (v3)")
    _logger.info(f"{'='*70}")
    header = (f"{'Label':<22} {'nk':>3} {'nv':>3} {'o2g':>5} {'ctg':>3} "
              f"{'protect':>9} {'step_map':>8} {'b_decay':>8} "
              f"{'1st_div':>8} {'match':>8} {'comp':>5}")
    _logger.info(header)
    _logger.info("-" * len(header))
    for r in results:
        beta_str = str(r.get("beta_decay", "off"))
        _logger.info(
            f"{r['label']:<22} {r['n_steps_k']:>3} {r['n_steps_v']:>3} "
            f"{r['order2_gamma']:>5.2f} {r['cross_token_group']:>3} "
            f"{r['protected_layers']:>9} {r['layer_step_map']:>8} {beta_str:>8} "
            f"{str(r['first_divergence']):>8} {r['token_match_rate']:>8.4f} "
            f"{r['compression_ratio']:>4.1f}x"
        )

    # ── Save CSV ──
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "label", "n_steps_k", "n_steps_v", "proj_beta", "diff_residual_gamma",
            "order2_gamma", "cross_token_group", "protected_layers", "layer_step_map",
            "beta_decay", "num_tokens_generated", "first_divergence",
            "token_match_rate", "compression_ratio", "ds_memory_mb", "elapsed_s",
        ]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        _logger.info(f"\nSaved to {output_path}")

    _logger.info("\nDone.")


if __name__ == "__main__":
    main()