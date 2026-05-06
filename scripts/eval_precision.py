#!/usr/bin/env python
"""
DS-KVCache End-to-End Precision Evaluation
============================================

Comprehensive precision comparison between Baseline (raw FP16) and
DS-KVCache (RINA compression) across three levels:

Level 1 — Logit precision
  • Per-position KL divergence (raw & symmetrized)
  • Cosine similarity of logit vectors
  • Max absolute error in logits
  • Jensen-Shannon divergence

Level 2 — Token precision
  • Exact match rate (argmax)
  • Token overlap rate (greedy decode)
  • First-divergence position

Level 3 — Text output
  • Full generated text side-by-side
  • Character-level match %
  • Diff-style divergence marker

Plus:
  • Cache compression ratio & memory savings
  • Per-token latency (ms/token) for both paths

Usage:
    python scripts/eval_precision.py
    python scripts/eval_precision.py --model /path/to/model --prompt "Hello world"
    python scripts/eval_precision.py --max-tokens 100 --json
    python scripts/eval_precision.py --quality quality
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rina.config import DSKVCacheConfig
from rina.model_wrapper import DSKVCacheModel

logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("eval_precision")


# ═══════════════════════════════════════════════════════════════════════════════
# Utility: auto-config imports (from scripts/auto_config.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _import_auto_config():
    """Lazy-import the auto_config detection functions."""
    from scripts.auto_config import detect_gpu_info, detect_model_info, generate_optimal_config
    return detect_gpu_info, detect_model_info, generate_optimal_config


# ═══════════════════════════════════════════════════════════════════════════════
# Level 1: Logit precision metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_kl_div(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """KL(P || Q) in nats.  P=baseline, Q=DS."""
    log_p = F.log_softmax(logits_p.float(), dim=-1)
    p = F.softmax(logits_p.float(), dim=-1)
    q = F.softmax(logits_q.float(), dim=-1)
    return float((p * (log_p - torch.log(q + 1e-12))).sum().item())


def compute_jsd(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """Jensen-Shannon divergence (nats)."""
    log_p = F.log_softmax(logits_p.float(), dim=-1)
    log_q = F.log_softmax(logits_q.float(), dim=-1)
    m_log = torch.logsumexp(torch.stack([log_p, log_q]), dim=0) - math.log(2)
    kl_pm = float((F.softmax(logits_p.float(), dim=-1) * (log_p - m_log)).sum().item())
    kl_qm = float((F.softmax(logits_q.float(), dim=-1) * (log_q - m_log)).sum().item())
    return 0.5 * (kl_pm + kl_qm)


def compute_cos_sim(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """Cosine similarity between two logit vectors."""
    a = logits_p.float().flatten()
    b = logits_q.float().flatten()
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def compute_max_err(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """Max absolute difference between two logit vectors."""
    return float((logits_p.float() - logits_q.float()).abs().max().item())


# ═══════════════════════════════════════════════════════════════════════════════
# Token-by-token forward collection
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_baseline_logits(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[int], str, float]:
    """Vanilla FP16 generation, collecting logits at every decode step.

    Returns (logits_per_step, generated_ids, full_text, elapsed_seconds).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # Prefill
    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    logits_list = [out.logits[0, -1, :].cpu().clone()]  # first token logits
    generated_ids = input_ids[0].tolist()

    first_id = _sample_token(out.logits[0, -1, :], temperature, do_sample)
    generated_ids.append(first_id)

    t_start = time.perf_counter()

    for step in range(1, max_new_tokens):
        last_token = torch.tensor([[generated_ids[-1]]], device=device)
        out = model(input_ids=last_token, use_cache=True, past_key_values=past)
        past = out.past_key_values
        logits_list.append(out.logits[0, -1, :].cpu().clone())

        next_id = _sample_token(out.logits[0, -1, :], temperature, do_sample)
        generated_ids.append(next_id)

        if next_id == tokenizer.eos_token_id:
            break

    elapsed = time.perf_counter() - t_start
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return logits_list, generated_ids, text, elapsed


@torch.no_grad()
def collect_ds_logits(
    model,
    tokenizer,
    cfg: DSKVCacheConfig,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[int], str, float, DSKVCacheModel]:
    """DS-KVCache generation, collecting logits at every decode step.

    Returns (logits_per_step, generated_ids, full_text, elapsed_seconds, wrapper).
    """
    from rina.ds_kv_cache import DSKVCacheStore, _build_v_rotation

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    wrapper = DSKVCacheModel(model, tokenizer, cfg=cfg)

    # Prefill
    out = wrapper.model(input_ids=input_ids, use_cache=True)
    logits_list = [out.logits[0, -1, :].cpu().clone()]

    wrapper._bulk_encode_from_prefill(out.past_key_values, input_ids)
    past = wrapper._build_past_from_ds()

    generated_ids = input_ids[0].tolist()
    first_id = _sample_token(out.logits[0, -1, :], temperature, do_sample)
    generated_ids.append(first_id)

    t_start = time.perf_counter()

    for step in range(1, max_new_tokens):
        last_token = torch.tensor([[generated_ids[-1]]], device=device)
        out = wrapper.model(
            input_ids=last_token,
            use_cache=True,
            past_key_values=past,
        )
        logits_list.append(out.logits[0, -1, :].cpu().clone())

        # Append new token to DS store & rebuild past
        wrapper._append_incremental(out.past_key_values, new_token_idx=-1)
        past = wrapper._build_past_from_ds()

        next_id = _sample_token(out.logits[0, -1, :], temperature, do_sample)
        generated_ids.append(next_id)

        if next_id == tokenizer.eos_token_id:
            break

    elapsed = time.perf_counter() - t_start
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return logits_list, generated_ids, text, elapsed, wrapper


# ═══════════════════════════════════════════════════════════════════════════════
# Same-Trace mode: force DS model to consume baseline's token sequence
# This isolates compression fidelity from autoregressive drift
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_forced_ds_logits(
    model,
    tokenizer,
    cfg: DSKVCacheConfig,
    prompt: str,
    forced_token_ids: List[int],
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[int], float, DSKVCacheModel]:
    """DS-KVCache forced-decode: consume the SAME tokens as baseline.

    At each step we feed the baseline's next token (not DS's own argmax/sample),
    so both paths share the same token prefix.  Logit comparison then measures
    pure compression fidelity, not autoregressive drift.

    forced_token_ids should contain ALL tokens (prompt + generated).
    Returns (logits_per_step, generated_ids, elapsed_seconds, wrapper).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]

    wrapper = DSKVCacheModel(model, tokenizer, cfg=cfg)

    # Prefill
    out = wrapper.model(input_ids=input_ids, use_cache=True)
    logits_list = [out.logits[0, -1, :].cpu().clone()]

    wrapper._bulk_encode_from_prefill(out.past_key_values, input_ids)
    past = wrapper._build_past_from_ds()

    generated_ids = list(forced_token_ids[:prompt_len])  # prompt portion
    decode_ids = forced_token_ids[prompt_len:]  # tokens to force-feed

    t_start = time.perf_counter()

    for step, forced_id in enumerate(decode_ids):
        if step == 0:
            # First decode step: use the baseline's first generated token
            last_token = torch.tensor([[forced_id]], device=device)
        else:
            last_token = torch.tensor([[forced_id]], device=device)

        out = wrapper.model(
            input_ids=last_token,
            use_cache=True,
            past_key_values=past,
        )
        logits_list.append(out.logits[0, -1, :].cpu().clone())

        # Append new token to DS store & rebuild past
        wrapper._append_incremental(out.past_key_values, new_token_idx=-1)
        past = wrapper._build_past_from_ds()

        generated_ids.append(forced_id)

        if forced_id == tokenizer.eos_token_id:
            break

    elapsed = time.perf_counter() - t_start
    return logits_list, generated_ids, elapsed, wrapper


def _sample_token(logits: torch.Tensor, temperature: float, do_sample: bool) -> int:
    """Single-token sampler (no top-p for simplicity in precision eval)."""
    if temperature > 0 and do_sample:
        scaled = logits.float() / temperature
        probs = F.softmax(scaled, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())
    return int(logits.argmax().item())


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    model_path: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    quality: str = "balanced",
    cfg_override: Optional[Dict[str, Any]] = None,
    device_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full precision evaluation pipeline.

    Returns a dict with all metrics and metadata.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ── 0. Auto-detect optimal config ──
    detect_gpu_info, detect_model_info, generate_optimal_config = _import_auto_config()
    gpu_info = detect_gpu_info()
    model_info = detect_model_info(model_path)
    cfg = generate_optimal_config(gpu_info, model_info, quality_preference=quality)
    if cfg_override:
        for k, v in cfg_override.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    device_str = device_override or gpu_info.get("recommended_device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # ── 1. Load model (shared) ──
    _logger.info(f"Loading model {model_path} ...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hf_model.eval()

    # ── 2. Baseline forward ──
    _logger.info(f"Baseline (FP16) — prompt: \"{prompt[:60]}...\"")
    base_logits, base_ids, base_text, base_time = collect_baseline_logits(
        hf_model, tokenizer, prompt, max_new_tokens, temperature, do_sample, device,
    )
    base_tokens_generated = len(base_ids) - len(tokenizer(prompt)["input_ids"])
    _logger.info(f"  Generated {base_tokens_generated} tokens in {base_time:.3f}s")

    # ── 3. DS-KVCache forward ──
    _logger.info(f"DS-KVCache — n_steps_k={cfg.get_n_steps_k()}, n_steps_v={cfg.get_n_steps_v()}")
    ds_logits, ds_ids, ds_text, ds_time, wrapper = collect_ds_logits(
        hf_model, tokenizer, cfg, prompt, max_new_tokens, temperature, do_sample, device,
    )
    ds_tokens_generated = len(ds_ids) - len(tokenizer(prompt)["input_ids"])
    _logger.info(f"  Generated {ds_tokens_generated} tokens in {ds_time:.3f}s")

    # ── 4. Level 1: Logit metrics ──
    eval_len = min(len(base_logits), len(ds_logits))
    kl_list, jsd_list, cos_list, maxe_list = [], [], [], []
    for i in range(eval_len):
        bp = base_logits[i]
        dp = ds_logits[i]
        kl_list.append(compute_kl_div(bp, dp))
        jsd_list.append(compute_jsd(bp, dp))
        cos_list.append(compute_cos_sim(bp, dp))
        maxe_list.append(compute_max_err(bp, dp))

    def _stats(arr):
        arr = [float(x) for x in arr]
        return {
            "mean": round(sum(arr) / len(arr), 6),
            "std": round(float(torch.tensor(arr).std().item()), 6),
            "min": round(min(arr), 6),
            "max": round(max(arr), 6),
        }

    # ── 5. Level 2: Token metrics ──
    exact_match_count = sum(1 for a, b in zip(base_ids, ds_ids) if a == b)
    base_list = base_ids[1:]  # skip prompt
    ds_list = ds_ids[1:]
    min_len = min(len(base_list), len(ds_list))
    match_count = sum(1 for i in range(min_len) if base_list[i] == ds_list[i])
    token_match_rate = match_count / min_len if min_len > 0 else 0.0

    first_div_pos = None
    for i in range(min_len):
        if base_list[i] != ds_list[i]:
            first_div_pos = i
            break

    # ── 6. Level 3: Text metrics ──
    match_chars = sum(1 for a, b in zip(base_text, ds_text) if a == b)
    char_total = max(len(base_text), len(ds_text))
    char_match_pct = match_chars / char_total * 100 if char_total > 0 else 0.0

    # Diff marker
    min_text_len = min(len(base_text), len(ds_text))
    diff_marker = ""
    for i in range(min_text_len):
        diff_marker += " " if base_text[i] == ds_text[i] else "^"

    # ── 7. Compression stats ──
    stats_list = wrapper.get_stats() if wrapper is not None else []
    total_fp16 = sum(s["fp16_memory_bytes"] for s in stats_list)
    total_ds = sum(s["ds_memory_bytes"] for s in stats_list)
    comp_ratio = total_fp16 / (total_ds + 1e-12) if total_ds > 0 else 0.0

    # ── 8. Latency ──
    if base_tokens_generated > 0:
        base_us_per_token = base_time / base_tokens_generated * 1e6
    else:
        base_us_per_token = 0.0
    if ds_tokens_generated > 0:
        ds_us_per_token = ds_time / ds_tokens_generated * 1e6
    else:
        ds_us_per_token = 0.0

    # ── 9. Aggregate results ──
    results = {
        "hardware": {
            "gpu_name": gpu_info["name"],
            "vram_gb": gpu_info["vram_gb"],
            "device": str(device),
        },
        "model": {
            "path": model_path,
            "type": model_info["model_type"],
            "num_layers": model_info["num_layers"],
            "num_q_heads": model_info["num_q_heads"],
            "num_kv_heads": model_info["num_kv_heads"],
            "d_head": model_info["d_head"],
            "gqa_ratio": model_info["gqa_ratio"],
            "size_category": model_info["model_size_category"],
        },
        "config": {
            "n_steps_k": cfg.get_n_steps_k(),
            "n_steps_v": cfg.get_n_steps_v(),
            "tile_size": cfg.tile_size,
            "beta": cfg.beta,
            "noise_shaping": cfg.use_noise_shaping,
            "proj_rank": cfg.proj_rank,
            "proj_beta": cfg.proj_beta,
            "differential": cfg.use_differential,
            "diff_residual_gamma": cfg.diff_residual_gamma,
            "v_orthogonal": cfg.v_orthogonal_transform,
            "quality_preference": quality,
        },
        "logit_precision": {
            "evaluated_positions": eval_len,
            "KL_divergence": _stats(kl_list),
            "Jensen_Shannon_divergence": _stats(jsd_list),
            "Cosine_similarity": _stats(cos_list),
            "Max_absolute_error": _stats(maxe_list),
        },
        "token_precision": {
            "baseline_tokens": base_tokens_generated,
            "ds_tokens": ds_tokens_generated,
            "exact_matches": match_count,
            "token_match_rate": round(token_match_rate, 6),
            "first_divergence_position": first_div_pos,
        },
        "text_precision": {
            "char_match_pct": round(char_match_pct, 2),
            "diff_length": min_text_len,
        },
        "compression": {
            "fp16_memory_mb": round(total_fp16 / (1024**2), 2),
            "ds_memory_mb": round(total_ds / (1024**2), 2),
            "compression_ratio": round(comp_ratio, 1),
            "memory_saved_pct": round((1 - total_ds / (total_fp16 + 1e-12)) * 100, 1),
        },
        "latency": {
            "baseline_us_per_token": round(base_us_per_token, 1),
            "ds_us_per_token": round(ds_us_per_token, 1),
            "ds_overhead_pct": round((ds_us_per_token - base_us_per_token) / (base_us_per_token + 1e-12) * 100, 1),
        },
        "raw_outputs": {
            "baseline_text": base_text,
            "ds_text": ds_text,
            "diff_marker": diff_marker,
        },
    }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Display
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(results: Dict[str, Any]):
    """Pretty-print the full evaluation report."""
    HW = results["hardware"]
    MD = results["model"]
    CF = results["config"]
    LP = results["logit_precision"]
    TP = results["token_precision"]
    TX = results["text_precision"]
    CO = results["compression"]
    LA = results["latency"]
    RW = results["raw_outputs"]

    print("\n" + "═" * 75)
    print("  DS-KVCache Precision Evaluation Report")
    print("═" * 75)

    # Hardware
    print("\n── Hardware ──")
    print(f"  GPU:      {HW['gpu_name']}  ({HW['vram_gb']} GB)")
    print(f"  Device:   {HW['device']}")

    # Model
    print("\n── Model ──")
    print(f"  Type:     {MD['type']}")
    print(f"  Layers:   {MD['num_layers']}")
    print(f"  Heads:    {MD['num_q_heads']}Q / {MD['num_kv_heads']}KV  (GQA={MD['gqa_ratio']}x)")
    print(f"  d_head:   {MD['d_head']}")
    print(f"  Scale:    {MD['size_category']}")

    # Config
    print("\n── DS-KVCache Config (auto) ──")
    print(f"  n_steps_k         = {CF['n_steps_k']}")
    print(f"  n_steps_v         = {CF['n_steps_v']}")
    print(f"  β (Σ-Δ)           = {CF['beta']}")
    print(f"  noise_shaping     = {CF['noise_shaping']}  (proj_rank={CF['proj_rank']}, β={CF['proj_beta']})")
    print(f"  differential      = {CF['differential']}  (γ={CF['diff_residual_gamma']})")
    print(f"  v_orthogonal      = {CF['v_orthogonal']}")
    print(f"  quality           = {CF['quality_preference']}")

    # ── Logit Precision ──
    print("\n── Level 1: Logit Precision ({:d} positions) ──".format(LP['evaluated_positions']))
    for metric, key in [
        ("KL Divergence          ", "KL_divergence"),
        ("Jensen-Shannon Div     ", "Jensen_Shannon_divergence"),
        ("Cosine Similarity       ", "Cosine_similarity"),
        ("Max Absolute Error      ", "Max_absolute_error"),
    ]:
        s = LP[key]
        print(f"  {metric}: mean={s['mean']:.6f}  σ={s['std']:.6f}  min={s['min']:.6f}  max={s['max']:.6f}")

    # ── Token Precision ──
    print("\n── Level 2: Token Precision ──")
    print(f"  Baseline tokens: {TP['baseline_tokens']}")
    print(f"  DS tokens:       {TP['ds_tokens']}")
    print(f"  Exact matches:   {TP['exact_matches']}")
    print(f"  Token match rate: {TP['token_match_rate']*100:.2f}%")
    if TP['first_divergence_position'] is not None:
        print(f"  First divergence @ position {TP['first_divergence_position']}")
    else:
        print(f"  No divergence — IDENTICAL token sequence")

    # ── Text Precision ──
    print("\n── Level 3: Text Output ──")
    print(f"  Char match:     {TX['char_match_pct']:.1f}%")
    print()

    # Side-by-side
    base_txt = RW['baseline_text']
    ds_txt = RW['ds_text']
    diff = RW['diff_marker']
    max_width = 120
    for i in range(0, max(len(base_txt), len(ds_txt), len(diff)), max_width):
        chunk_base = base_txt[i:i+max_width]
        chunk_ds = ds_txt[i:i+max_width]
        chunk_diff = diff[i:i+max_width] if i < len(diff) else ""
        print(f"  [BASELINE] {chunk_base}")
        print(f"  [DS/KV]    {chunk_ds}")
        if chunk_diff.strip():
            print(f"  [DIFF]     {chunk_diff}")
        print()

    # ── Compression ──
    print("── Compression ──")
    print(f"  FP16 cache:  {CO['fp16_memory_mb']:.2f} MB")
    print(f"  DS cache:    {CO['ds_memory_mb']:.2f} MB")
    print(f"  Ratio:       {CO['compression_ratio']:.1f}x")
    print(f"  Memory saved: {CO['memory_saved_pct']:.1f}%")

    # ── Latency ──
    print("\n── Latency ──")
    print(f"  Baseline:  {LA['baseline_us_per_token']:.0f} µs/token")
    print(f"  DS-KVCache: {LA['ds_us_per_token']:.0f} µs/token")
    print(f"  Overhead:   +{LA['ds_overhead_pct']:.1f}%")

    print("\n" + "═" * 75)
    print("  Evaluation complete.")
    print("═" * 75 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="DS-KVCache Precision Evaluation — Baseline vs DS-KVCache"
    )
    p.add_argument(
        "--model", type=str,
        default="D:/Software_Development/Project/models/Llama-3.2-1B",
    )
    p.add_argument(
        "--prompt", type=str,
        default="The future of artificial intelligence lies in",
    )
    p.add_argument("--max-tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--do-sample", action="store_true", default=False,
                   help="Use multinomial sampling (default: greedy argmax)")
    p.add_argument(
        "--quality", type=str, default="balanced",
        choices=["quality", "balanced", "speed"],
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--json", action="store_true", default=False,
        help="Output JSON only (for scripting)"
    )
    p.add_argument("--output", type=str, default=None,
                   help="Save JSON report to file")
    args = p.parse_args()

    results = run_evaluation(
        model_path=args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        quality=args.quality,
        device_override=args.device,
    )

    if args.json:
        # Sanitize non-serializable fields
        out = json.dumps(results, indent=2, ensure_ascii=False, default=str)
        print(out)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(out, encoding="utf-8")
            _logger.info(f"Saved to {args.output}")
    else:
        print_report(results)
        if args.output:
            out = json.dumps(results, indent=2, ensure_ascii=False, default=str)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(out, encoding="utf-8")
            _logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()