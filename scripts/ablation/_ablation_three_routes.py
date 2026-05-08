"""
§A Three-Route Unified Ablation — Llama 3.2 1B
==============================================

Tests the three upgrade routes individually and combined:

  Route 3 — DCT Energy Compaction (transform_mode sweep)
  Route 1 — Adaptive Bit-Rate Masking (per-tile sensitivity boost)
  Route 2 — Cross-Head Reconstruction Residual (ε = X - X̂ bias injection)

Combined test: DCT+auto + adaptive_masking + cross_head_residual

Run::
    python _ablation_three_routes.py
    python _ablation_three_routes.py --model D:/Software_Development/Project/models/Llama-3.2-1B --layers 0,4,8,12,15
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from tabulate import tabulate

from rina.config import DSKVCacheConfig
from rina.ds_kv_cache import encode_kv_cache, decode_kvcache_store, DSKVCacheStore

logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("three_routes")


# ===========================================================================
# Utility
# ===========================================================================

def extract_kv_from_hf(model, tokenizer, text: str) -> list:
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], use_cache=True)
    kv_pairs = []
    for k_cache, v_cache in outputs.past_key_values:
        kv_pairs.append((k_cache[0], v_cache[0]))
    return kv_pairs


def compute_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
    o, r = original.float(), reconstructed.float()
    mse = F.mse_loss(r, o).item()
    signal_power = (o ** 2).mean().item()
    noise_power = ((o - r) ** 2).mean().item()
    snr_db = 10 * math.log10(signal_power / (noise_power + 1e-12))
    cos_sim = F.cosine_similarity(
        r.flatten().unsqueeze(0), o.flatten().unsqueeze(0)
    ).item()
    return {"MSE": mse, "SNR_dB": snr_db, "CosSim": cos_sim}


# ===========================================================================
# Route 2: Cross-Head Reconstruction Residual Bias
# ===========================================================================

def encode_with_cross_head_residual(
    K: torch.Tensor,        # (n_kv, seq_len, d_head)
    V: torch.Tensor,
    cfg: DSKVCacheConfig,
) -> tuple:
    """
    Route 2 — Cross-Head Reconstruction Residual Injection.

    For GQA groups (Q heads sharing one KV head):
      1. Encode K_h, V_h normally → get reconstruction K̂_h, V̂_h
      2. Compute residual ε_K = K_h - K̂_h, ε_V = V_h - V̂_h
      3. Distribute ε as bias into the V encoding of the NEXT head in the GQA group.

    This ensures quantization errors are distributed across heads
    within a GQA group rather than accumulating independently.
    """
    n_kv, seq_len, d_head = K.shape
    tile_size = cfg.tile_size

    all_k_stores = []
    all_v_stores = []
    residual_K = torch.zeros(seq_len, d_head, device=K.device, dtype=torch.float32)
    residual_V = torch.zeros(seq_len, d_head, device=V.device, dtype=torch.float32)
    gamma = getattr(cfg, 'cross_head_residual_gamma', 0.25)

    for h in range(n_kv):
        K_h = K[h].float()
        V_h = V[h].float()

        # Inject accumulated residual from prior heads
        if h > 0:
            K_h = K_h + gamma * residual_K
            V_h = V_h + gamma * residual_V

        k_store, v_store = encode_kv_cache(K_h, V_h, cfg)

        # Decode to compute reconstruction residual
        k_hat = decode_kvcache_store(k_store, tile_size, cfg.use_differential)
        v_hat = decode_kvcache_store(v_store, tile_size, cfg.use_differential)

        # Compute this head's reconstruction residual (unbiased — the raw error)
        eps_k = K[h].float() - k_hat  # use original, not injected
        eps_v = V[h].float() - v_hat

        # Decay accumulated residual and add new
        residual_K = (1 - gamma) * residual_K + gamma * eps_k
        residual_V = (1 - gamma) * residual_V + gamma * eps_v

        all_k_stores.append(k_store)
        all_v_stores.append(v_store)

    return all_k_stores, all_v_stores


def decode_cross_head_residual(
    k_stores: list,
    v_stores: list,
    K_original: torch.Tensor,
    V_original: torch.Tensor,
    tile_size: int,
    use_diff: bool,
):
    """Decode and re-inject residuals for fair evaluation."""
    n_kv, seq_len, d_head = K_original.shape
    k_hats = []
    v_hats = []
    residual_K = torch.zeros(seq_len, d_head, device=K_original.device, dtype=torch.float32)
    residual_V = torch.zeros(seq_len, d_head, device=V_original.device, dtype=torch.float32)
    gamma = 0.25  # must match encode

    for h in range(n_kv):
        k_hat_raw = decode_kvcache_store(k_stores[h], tile_size, use_diff)
        v_hat_raw = decode_kvcache_store(v_stores[h], tile_size, use_diff)

        if h > 0:
            k_hat_raw = k_hat_raw + gamma * residual_K
            v_hat_raw = v_hat_raw + gamma * residual_V

        eps_k = K_original[h].float() - k_hat_raw
        eps_v = V_original[h].float() - v_hat_raw

        residual_K = (1 - gamma) * residual_K + gamma * eps_k
        residual_V = (1 - gamma) * residual_V + gamma * eps_v

        k_hats.append(k_hat_raw)
        v_hats.append(v_hat_raw)

    return k_hats, v_hats


def encode_with_cross_head_residual_v2(
    K: torch.Tensor,        # (n_kv, seq_len, d_head)
    V: torch.Tensor,
    cfg: DSKVCacheConfig,
) -> tuple:
    """
    Route 2 v2 — Cross-Head Reconstruction Residual Injection.

    Cleaner implementation: for each KV head, encode then compute
    ε = X - X̂, and inject ε * gamma into the NEXT head's input before encoding.

    The first head gets no injection (baseline). Subsequent heads receive
    the accumulated residual from prior heads in the GQA group.
    """
    n_kv, seq_len, d_head = K.shape
    tile_size = cfg.tile_size

    all_k_stores = []
    all_v_stores = []
    # Residual accumulator (exponential moving average)
    residual_k = None  # (seq_len, d_head)
    residual_v = None
    gamma = getattr(cfg, 'cross_head_residual_gamma', 0.25)

    for h in range(n_kv):
        K_h = K[h].float().clone()
        V_h = V[h].float().clone()

        # Inject accumulated residual bias from prior heads
        if residual_k is not None:
            K_h = K_h + gamma * residual_k
            V_h = V_h + gamma * residual_v

        k_store, v_store = encode_kv_cache(K_h, V_h, cfg)

        # Decode to compute this head's reconstruction residual
        k_hat = decode_kvcache_store(k_store, tile_size, cfg.use_differential)
        v_hat = decode_kvcache_store(v_store, tile_size, cfg.use_differential)

        # Reconstruction residual — difference between original (un-injected) and decoded
        eps_k = K[h].float() - k_hat
        eps_v = V[h].float() - v_hat

        # Update EMA residual for next head
        if residual_k is None:
            residual_k = eps_k
            residual_v = eps_v
        else:
            residual_k = (1 - gamma) * residual_k + gamma * eps_k
            residual_v = (1 - gamma) * residual_v + gamma * eps_v

        all_k_stores.append(k_store)
        all_v_stores.append(v_store)

    return all_k_stores, all_v_stores


def decode_cross_head_residual_v2(
    k_stores: list,
    v_stores: list,
    K_orig: torch.Tensor,
    V_orig: torch.Tensor,
    tile_size: int,
    use_diff: bool,
):
    """Decode cross-head residual stores.

    During encode, head h received bias from head h-1's residual.
    The bias is already baked into the encoded bases — no further
    injection is needed during decode.  The metric comparison
    ``K_orig[h] - k_hat_raw`` correctly measures the combined effect
    of residual injection + 1-bit encoding error against the original
    (unbiased) ground truth.
    """
    k_hats = []
    v_hats = []
    for h in range(len(k_stores)):
        k_hats.append(decode_kvcache_store(k_stores[h], tile_size, use_diff))
        v_hats.append(decode_kvcache_store(v_stores[h], tile_size, use_diff))
    return k_hats, v_hats


# ===========================================================================
# Experiment runner
# ===========================================================================

def run_experiment(
    name: str,
    K: torch.Tensor,
    V: torch.Tensor,
    cfg: DSKVCacheConfig,
    n_kv: int,
    tile_size: int,
    use_route2: bool = False,
) -> dict:
    """Encode/decode all heads and return aggregate metrics."""
    k_cos_list, v_cos_list = [], []
    total_mem = 0
    total_time = 0.0

    if use_route2:
        t0 = time.perf_counter()
        k_stores, v_stores = encode_with_cross_head_residual_v2(K, V, cfg)
        total_time = time.perf_counter() - t0

        k_hats, v_hats = decode_cross_head_residual_v2(
            k_stores, v_stores, K, V, tile_size, cfg.use_differential
        )

        for h in range(n_kv):
            km = compute_metrics(K[h].float(), k_hats[h])
            vm = compute_metrics(V[h].float(), v_hats[h])
            k_cos_list.append(km["CosSim"])
            v_cos_list.append(vm["CosSim"])
            total_mem += k_stores[h].memory_bytes + v_stores[h].memory_bytes
    else:
        t0 = time.perf_counter()
        for h in range(n_kv):
            k_store, v_store = encode_kv_cache(K[h].float(), V[h].float(), cfg)
            k_hat = decode_kvcache_store(k_store, tile_size, cfg.use_differential)
            v_hat = decode_kvcache_store(v_store, tile_size, cfg.use_differential)

            km = compute_metrics(K[h].float(), k_hat)
            vm = compute_metrics(V[h].float(), v_hat)
            k_cos_list.append(km["CosSim"])
            v_cos_list.append(vm["CosSim"])
            total_mem += k_store.memory_bytes + v_store.memory_bytes
        total_time = time.perf_counter() - t0

    seq_len = K.shape[1]
    d_head = K.shape[2]
    fp16_bytes = seq_len * n_kv * d_head * 2 * 2  # K+V fp16
    comp_ratio = fp16_bytes / (total_mem + 1e-12)

    return {
        "name": name,
        "avg_k_cos": sum(k_cos_list) / n_kv,
        "avg_v_cos": sum(v_cos_list) / n_kv,
        "comp_ratio": comp_ratio,
        "mem_kb": total_mem / 1024,
        "time_ms": total_time * 1000,
    }


def make_base_cfg(n_steps_k: int, n_steps_v: int, tile_size: int, d_head: int) -> dict:
    """Base config — transforms are incompatible with use_differential.
    
    Roadmap 3 (DCT) note: transform pads rows to tile-alignment, changing 
    orig_shape. The differential residual path (k_enc - k_hat) then fails 
    with dimension mismatch. Therefore use_differential=False for all 
    transform experiments; Route 1 experiments can enable it separately.
    """
    return dict(
        n_steps=max(n_steps_k, n_steps_v),
        n_steps_k=n_steps_k,
        n_steps_v=n_steps_v,
        tile_size=tile_size,
        beta=0.15,
        proj_beta=0.3,
        adaptive_eta=0.10,
        use_differential=False,        # off — incompatible with DCT/DWT transforms
        diff_strategy="residual",
        diff_residual_gamma=0.25,
        diff_residual_n_steps=1,
        v_orthogonal_transform=False,  # off for clean ablation
        cross_token_group=1,           # per-token encoding
        use_fwht=False,                # no FWHT — we test DCT/DWT/Hybrid explicitly
        order2_gamma=0.0,
        verbose=False,
    )


# ===========================================================================
# Main
# ===========================================================================

def main():
    p = argparse.ArgumentParser(description="Three-Route Unified Ablation")
    p.add_argument("--model", type=str, default="D:/Software_Development/Project/models/Llama-3.2-1B")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--layers", type=str, default="all")
    p.add_argument("--n-steps-k", type=int, default=3)
    p.add_argument("--n-steps-v", type=int, default=5)
    p.add_argument("--tile-size", type=int, default=16)
    p.add_argument("--beta", type=float, default=0.15)
    p.add_argument("--output", type=str, default="ablation_three_routes.csv")
    p.add_argument("--skip-route2", action="store_true", help="Skip Route 2 (cross-head residual)")
    args = p.parse_args()

    # ── Load model ─────────────────────────────────────────────────────
    _logger.info(f"Loading {args.model} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model.eval()

    mcfg = model.config
    n_heads = getattr(mcfg, "num_attention_heads", getattr(mcfg, "n_head", None))
    n_kv = getattr(mcfg, "num_key_value_heads", getattr(mcfg, "n_kv_head", n_heads))
    d_head = getattr(mcfg, "head_dim", mcfg.hidden_size // n_heads)
    n_layers = getattr(mcfg, "num_hidden_layers", getattr(mcfg, "n_layer", None))
    gqa_ratio = n_heads // n_kv
    _logger.info(f"Architecture: {n_layers} layers, {n_heads}Q/{n_kv}KV heads, "
                 f"d_head={d_head}, GQA ratio={gqa_ratio}x")

    # ── Prompt ─────────────────────────────────────────────────────────
    prompt = (
        "The transformer architecture has revolutionized natural language processing "
        "by enabling models to process sequential data in parallel rather than "
        "sequentially. This parallelization allows for significantly faster training "
        "times and the ability to scale to much larger datasets. Attention mechanisms "
        "compute contextual representations by weighing the importance of different "
        "input tokens, creating rich embeddings that capture long-range dependencies. "
        "Modern large language models leverage these capabilities to achieve "
        "state-of-the-art performance across a wide range of natural language tasks "
        "including translation, summarization, question answering, and code generation. "
        "The key innovation lies in the self-attention mechanism which computes "
        "pairwise interactions between all positions in the input sequence. "
        "This quadratic complexity has spurred research into efficient attention "
        "variants including sparse attention, linear attention, and KV-cache "
        "compression techniques that aim to reduce memory footprint while "
        "preserving model quality. Recent advances in 1-bit quantization of "
        "KV caches have demonstrated that aggressive compression ratios of "
        "10-20x are achievable with minimal degradation in downstream task "
        "performance, opening new frontiers for efficient LLM deployment."
    )

    # ── Extract K/V ────────────────────────────────────────────────────
    _logger.info("Extracting K/V from forward pass ...")
    kv_layers = extract_kv_from_hf(model, tokenizer, prompt)
    seq_len = kv_layers[0][0].shape[1]
    _logger.info(f"Sequence length: {seq_len}")

    # ── Resolve layers ─────────────────────────────────────────────────
    if args.layers == "all":
        eval_layers = list(range(n_layers))
    else:
        eval_layers = [int(x) for x in args.layers.split(",")]

    layer_subsets = {
        "early": [l for l in eval_layers if l <= 3],
        "middle": [l for l in eval_layers if 4 <= l <= 11],
        "late": [l for l in eval_layers if l >= 12],
    }

    # ── Define experiments ─────────────────────────────────────────────
    base = make_base_cfg(args.n_steps_k, args.n_steps_v, args.tile_size, d_head)

    experiments = []

    # Baseline (no transform, no masking, no cross-head)
    experiments.append(("Baseline", DSKVCacheConfig(**base)))

    # Route 3: DCT Energy Compaction
    experiments.append(("R3_DCT", DSKVCacheConfig(**base, transform_mode="dct")))
    experiments.append(("R3_DWT", DSKVCacheConfig(**base, transform_mode="dwt")))
    # NOTE: R3_Hybrid skipped — Hybrid forward transform has a shape bug (line 344 in transforms.py)
    # where dct_low.reshape(1, M//2) expects 128 elements but gets 256. Fix requires
    # redesigning the packed format for rect tiles (16×d_head). DCT/DWT/AUTO cover all
    # meaningful ablation dimensions.
    # experiments.append(("R3_Hybrid", DSKVCacheConfig(**base, transform_mode="hybrid")))
    experiments.append(("R3_AUTO", DSKVCacheConfig(**base, transform_mode="auto")))

    # Route 1: Adaptive Masking (no transform)
    experiments.append(("R1_AdaptMask", DSKVCacheConfig(
        **base,
        adaptive_masking=True,
        mask_outlier_threshold=3.0,
        mask_n_steps_boost=1,
        mask_proj_beta_boost=0.5,
    )))
    experiments.append(("R1_AdaptMask_Strong", DSKVCacheConfig(
        **base,
        adaptive_masking=True,
        mask_outlier_threshold=2.0,
        mask_n_steps_boost=2,
        mask_proj_beta_boost=0.8,
    )))

    # Route 1 + Route 3 combined
    experiments.append(("R1+R3_AUTO+Mask", DSKVCacheConfig(
        **base,
        transform_mode="auto",
        adaptive_masking=True,
        mask_outlier_threshold=3.0,
        mask_n_steps_boost=1,
        mask_proj_beta_boost=0.5,
    )))
    experiments.append(("R1+R3_DCT+Mask", DSKVCacheConfig(
        **base,
        transform_mode="dct",
        adaptive_masking=True,
        mask_outlier_threshold=3.0,
        mask_n_steps_boost=1,
        mask_proj_beta_boost=0.5,
    )))

    # Route 2: Cross-head residual (implemented above)
    if not args.skip_route2:
        experiments.append(("R2_CrossHeadRes", DSKVCacheConfig(**base)))
        experiments.append(("R2+DCT_CrossHead", DSKVCacheConfig(**base, transform_mode="dct")))
        experiments.append(("R1+R2+R3_Full", DSKVCacheConfig(
            **base,
            transform_mode="auto",
            adaptive_masking=True,
            mask_outlier_threshold=3.0,
            mask_n_steps_boost=1,
            mask_proj_beta_boost=0.5,
        )))

    # ── Run experiments ────────────────────────────────────────────────
    all_results = []

    for exp_name, cfg in experiments:
        is_route2 = exp_name.startswith("R2_") or "R2" in exp_name.split("+")

        _logger.info(f"\n{'='*70}")
        _logger.info(f"  {exp_name}")
        if cfg.transform_mode and cfg.transform_mode != "none":
            _logger.info(f"  transform={cfg.transform_mode}")
        if cfg.adaptive_masking:
            _logger.info(f"  adaptive_masking=True, outlier_thresh={cfg.mask_outlier_threshold}, "
                         f"n_boost={cfg.mask_n_steps_boost}, η_boost={cfg.mask_proj_beta_boost}")
        if is_route2:
            _logger.info(f"  cross_head_residual_gamma={getattr(cfg, 'cross_head_residual_gamma', 0.25)}")
        _logger.info(f"{'='*70}")

        layer_results = []
        for layer_idx in eval_layers:
            K_layer, V_layer = kv_layers[layer_idx]

            result = run_experiment(
                exp_name, K_layer, V_layer, cfg,
                n_kv=n_kv, tile_size=args.tile_size,
                use_route2=is_route2,
            )

            region = ("early" if layer_idx in layer_subsets["early"]
                      else "middle" if layer_idx in layer_subsets["middle"]
                      else "late")
            result["layer"] = layer_idx
            result["region"] = region
            layer_results.append(result)

            _logger.info(
                f"  L{layer_idx:02d}[{region:6s}] | "
                f"K CosSim={result['avg_k_cos']:.4f} | "
                f"V CosSim={result['avg_v_cos']:.4f} | "
                f"Ratio={result['comp_ratio']:.1f}x | "
                f"Time={result['time_ms']:.1f}ms"
            )

        # Aggregate
        agg_k = sum(r["avg_k_cos"] for r in layer_results) / len(layer_results)
        agg_v = sum(r["avg_v_cos"] for r in layer_results) / len(layer_results)
        agg_r = sum(r["comp_ratio"] for r in layer_results) / len(layer_results)
        agg_mem = sum(r["mem_kb"] for r in layer_results) / len(layer_results)
        agg_time = sum(r["time_ms"] for r in layer_results) / len(layer_results)

        # Per-region aggregates
        region_agg = {}
        for region in ["early", "middle", "late"]:
            region_results = [r for r in layer_results if r["region"] == region]
            if region_results:
                region_agg[region] = {
                    "v_cos": sum(r["avg_v_cos"] for r in region_results) / len(region_results),
                    "k_cos": sum(r["avg_k_cos"] for r in region_results) / len(region_results),
                    "ratio": sum(r["comp_ratio"] for r in region_results) / len(region_results),
                }

        all_results.append({
            "experiment": exp_name,
            "avg_k_cos": agg_k,
            "avg_v_cos": agg_v,
            "avg_ratio": agg_r,
            "avg_mem_kb": agg_mem,
            "avg_time_ms": agg_time,
            "per_layer": layer_results,
            "per_region": region_agg,
        })

    # ── Summary table ──────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  Three-Route Unified Ablation — Summary")
    print(f"{'='*90}")
    headers = ["Experiment", "K CosSim", "V CosSim", "ΔV vs Base", "Ratio", "Mem KB", "Time ms"]
    rows = []
    baseline_v = all_results[0]["avg_v_cos"]

    for r in all_results:
        rows.append([
            r["experiment"],
            f"{r['avg_k_cos']:.4f}",
            f"{r['avg_v_cos']:.4f}",
            f"{r['avg_v_cos'] - baseline_v:+.4f}",
            f"{r['avg_ratio']:.1f}x",
            f"{r['avg_mem_kb']:.0f}",
            f"{r['avg_time_ms']:.1f}",
        ])

    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # ── Per-region breakdown ───────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  Per-Region V CosSim Breakdown")
    print(f"{'='*90}")
    region_headers = ["Experiment", "Early L0-3", "Middle L4-11", "Late L12-15"]
    region_rows = []
    for r in all_results:
        pr = r.get("per_region", {})
        region_rows.append([
            r["experiment"],
            f"{pr.get('early', {}).get('v_cos', 0):.4f}" if 'early' in pr else "-",
            f"{pr.get('middle', {}).get('v_cos', 0):.4f}" if 'middle' in pr else "-",
            f"{pr.get('late', {}).get('v_cos', 0):.4f}" if 'late' in pr else "-",
        ])
    print(tabulate(region_rows, headers=region_headers, tablefmt="grid"))

    # ── Best combinations ──────────────────────────────────────────────
    best_v = max(all_results, key=lambda r: r["avg_v_cos"])
    best_ratio = max(all_results, key=lambda r: r["avg_ratio"])
    best_pareto = max(all_results, key=lambda r: r["avg_v_cos"] * math.log(r["avg_ratio"] + 1))

    print(f"\n{'='*90}")
    print(f"  Best V CosSim:  {best_v['experiment']:<30s} V={best_v['avg_v_cos']:.4f}  (Δ={best_v['avg_v_cos']-baseline_v:+.4f})")
    print(f"  Best Ratio:     {best_ratio['experiment']:<30s} Ratio={best_ratio['avg_ratio']:.1f}x")
    print(f"  Best Pareto:    {best_pareto['experiment']:<30s} V={best_pareto['avg_v_cos']:.4f}  Ratio={best_pareto['avg_ratio']:.1f}x")
    print(f"{'='*90}")

    # ── Save CSV ───────────────────────────────────────────────────────
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "avg_k_cossim", "avg_v_cossim", "delta_v", "avg_ratio",
                     "avg_mem_kb", "avg_time_ms",
                     "early_v_cos", "middle_v_cos", "late_v_cos"])
        for r in all_results:
            pr = r.get("per_region", {})
            w.writerow([
                r["experiment"],
                f"{r['avg_k_cos']:.6f}",
                f"{r['avg_v_cos']:.6f}",
                f"{r['avg_v_cos'] - baseline_v:+.6f}",
                f"{r['avg_ratio']:.2f}",
                f"{r['avg_mem_kb']:.1f}",
                f"{r['avg_time_ms']:.1f}",
                f"{pr.get('early', {}).get('v_cos', 0):.6f}" if 'early' in pr else "",
                f"{pr.get('middle', {}).get('v_cos', 0):.6f}" if 'middle' in pr else "",
                f"{pr.get('late', {}).get('v_cos', 0):.6f}" if 'late' in pr else "",
            ])
    _logger.info(f"Results saved to {args.output}")

    # ── Generate per-layer detail CSV ──────────────────────────────────
    detail_path = args.output.replace(".csv", "_detail.csv")
    with open(detail_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "layer", "region", "k_cossim", "v_cossim", "ratio", "mem_kb", "time_ms"])
        for r in all_results:
            for lr in r.get("per_layer", []):
                w.writerow([
                    r["experiment"], lr["layer"], lr["region"],
                    f"{lr['avg_k_cos']:.6f}", f"{lr['avg_v_cos']:.6f}",
                    f"{lr['comp_ratio']:.2f}", f"{lr['mem_kb']:.1f}", f"{lr['time_ms']:.1f}",
                ])
    _logger.info(f"Per-layer details saved to {detail_path}")

    print(f"\n[OK] Three-route ablation complete.")
    print(f"  Summary: {args.output}")
    print(f"  Details: {detail_path}")


if __name__ == "__main__":
    main()