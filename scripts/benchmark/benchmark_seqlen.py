r"""
DS-KVCache Sequence Length Benchmark
======================================

Measures compression ratio & reconstruction quality vs sequence length L.

Formula: compression_ratio ∝ L / (c₀ + c₁·L) → approaches c₁⁻¹ as L → ∞.

Run::
    python scripts/benchmark_seqlen.py
    python scripts/benchmark_seqlen.py --model D:/Software_Development/Project/models/Qwen2.5-0.5B --layer 12
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rina.config import DSKVCacheConfig
from rina.ds_kv_cache import encode_kv_cache, decode_kvcache_store

logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("benchmark_seqlen")

_LOREM = (
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
    "pairwise interactions between all tokens in a sequence, enabling the model "
    "to capture complex linguistic patterns and relationships that span across "
    "entire documents. Researchers continue to push the boundaries of what is "
    "possible with transformer-based architectures by scaling up model sizes, "
    "improving training efficiency, and developing novel attention variants "
    "that reduce computational complexity while maintaining performance. "
    "Recent advances include mixture-of-experts layers, rotary position embeddings, "
    "grouped query attention, and sliding window attention mechanisms that "
    "collectively enable efficient processing of extremely long context windows "
    "exceeding hundreds of thousands of tokens. The field continues to evolve "
    "rapidly with new breakthroughs announced regularly from leading research "
    "laboratories and technology companies around the world."
)


def build_prompt_for_length(tokenizer, target_len: int) -> str:
    """Build a prompt with exactly target_len tokens by truncating lorem text."""
    ids = tokenizer(_LOREM, add_special_tokens=False)["input_ids"]
    if len(ids) >= target_len:
        ids = ids[:target_len]
    else:
        # Repeat until we have enough tokens
        repeated = ids * ((target_len // len(ids)) + 2)
        ids = repeated[:target_len]
    return tokenizer.decode(ids, skip_special_tokens=True)


def extract_kv_for_layer(model, tokenizer, text: str, layer_idx: int):
    """Run forward pass and return K, V at a specific layer."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], use_cache=True, output_hidden_states=False)
    past = outputs.past_key_values
    k_cache, v_cache = past[layer_idx]
    return k_cache[0], v_cache[0]  # remove batch dim → (n_kv_heads, seq_len, d_head)


def main():
    p = argparse.ArgumentParser(description="DS-KVCache sequence length benchmark")
    p.add_argument("--model", type=str,
                   default="D:/Software_Development/Project/models/Qwen2.5-0.5B",
                   help="Model path")
    p.add_argument("--layer", type=int, default=12, help="Layer index to benchmark")
    p.add_argument("--seq-lens", type=str, default="64,128,256,512,1024,2048",
                   help="Comma-separated sequence lengths")
    p.add_argument("--n-steps", type=int, default=5)
    p.add_argument("--tile-size", type=int, default=16)
    p.add_argument("--output-csv", type=str, default="benchmark_seqlen_results.csv",
                   help="Output CSV path")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    # ── Load model ─────────────────────────────────────────────────────
    _logger.info(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    mcfg = model.config
    n_heads = getattr(mcfg, "num_attention_heads", getattr(mcfg, "n_head", None))
    n_kv = getattr(mcfg, "num_key_value_heads", getattr(mcfg, "n_kv_head", n_heads))
    d_head = getattr(mcfg, "head_dim", mcfg.hidden_size // n_heads)
    n_layers_model = getattr(mcfg, "num_hidden_layers", getattr(mcfg, "n_layer", None))
    if args.layer >= n_layers_model:
        _logger.error(f"Layer {args.layer} out of range (0-{n_layers_model - 1})")
        sys.exit(1)

    _logger.info(
        f"Architecture: {n_layers_model} layers, {n_heads}Q/{n_kv}KV heads, d_head={d_head}"
    )
    _logger.info(f"Benchmarking layer {args.layer} at L ∈ {seq_lens}")

    # ── Config ─────────────────────────────────────────────────────────
    cfg = DSKVCacheConfig(
        n_steps=args.n_steps,
        tile_size=args.tile_size,
        beta=0.15,
        use_noise_shaping=True,
        proj_rank=min(8, d_head // 4),
        proj_beta=0.3,
        adaptive_eta=True,
        adaptive_n=True,
        n_upper_bound=args.n_steps + 5,
        use_differential=True,
        diff_strategy="momentum_shift", order2_gamma=0.0,
        base_dtype="fp16",
        verbose=False,
    )

    # ── Run benchmark ──────────────────────────────────────────────────
    rows = []
    for L in seq_lens:
        _logger.info(f"  L={L}: building prompt ...")
        prompt = build_prompt_for_length(tokenizer, L)
        _logger.info(f"  L={L}: extracting K/V (layer {args.layer}) ...")
        K, V = extract_kv_for_layer(model, tokenizer, prompt, args.layer)
        actual_len = K.shape[1]
        _logger.info(f"  L={L}: actual tokens={actual_len}")

        # Encode & decode per KV head
        k_cos_all, v_cos_all = [], []
        total_mem = 0

        for h in range(n_kv):
            k_mat = K[h].float()
            v_mat = V[h].float()
            k_store, v_store = encode_kv_cache(k_mat, v_mat, cfg)
            k_hat = decode_kvcache_store(k_store, cfg.tile_size, cfg.use_differential)
            v_hat = decode_kvcache_store(v_store, cfg.tile_size, cfg.use_differential)
            k_cos_all.append(
                F.cosine_similarity(k_hat.flatten().unsqueeze(0), k_mat.flatten().unsqueeze(0)).item()
            )
            v_cos_all.append(
                F.cosine_similarity(v_hat.flatten().unsqueeze(0), v_mat.flatten().unsqueeze(0)).item()
            )
            total_mem += k_store.memory_bytes + v_store.memory_bytes

        fp16_bytes = actual_len * n_kv * d_head * 2 * 2  # K+V
        comp_ratio = fp16_bytes / (total_mem + 1e-12)
        avg_k_cos = sum(k_cos_all) / n_kv
        avg_v_cos = sum(v_cos_all) / n_kv
        ds_mem_kb = total_mem / 1024

        rows.append({
            "L_target": L,
            "L_actual": actual_len,
            "K_CosSim": round(avg_k_cos, 4),
            "V_CosSim": round(avg_v_cos, 4),
            "Compression": round(comp_ratio, 2),
            "DS_Mem_KB": round(ds_mem_kb, 2),
        })

        _logger.info(
            f"    K CosSim={avg_k_cos:.4f}  V CosSim={avg_v_cos:.4f}  "
            f"Compress={comp_ratio:.1f}x  DS={ds_mem_kb:.1f} KB"
        )

    # ── Print table ────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print(f"  Sequence Length Benchmark — Layer {args.layer}, {n_kv} KV heads")
    print("=" * 75)
    print(f"  {'L':>5}  {'K CosSim':>9}  {'V CosSim':>9}  {'Compress':>9}  {'DS Mem':>9}")
    print("  " + "-" * 65)
    for r in rows:
        print(
            f"  {r['L_actual']:>5}  {r['K_CosSim']:>9.4f}  {r['V_CosSim']:>9.4f}  "
            f"{r['Compression']:>7.1f}x  {r['DS_Mem_KB']:>7.1f} KB"
        )
    print("=" * 75)

    # ── Asymptotic analysis ────────────────────────────────────────────
    print("\nAsymptotic Analysis:")
    print(f"  Compression ratio C(L) = L·(2·n_kv·d_head·2) / memory(L)")
    print(f"  As L → ∞, C(L) → tile_size / overhead_per_token")
    # Calculate overhead per token from linear fit
    L_vals = [r["L_actual"] for r in rows]
    C_vals = [r["Compression"] for r in rows]

    # Linear fit: memory = m·L + b → C = L·256 / (m·L + b) → asymptotic = 256/m
    # Use simple linear regression
    n = len(L_vals)
    sum_x = sum(L_vals)
    sum_y = [r["DS_Mem_KB"] * 1024 / r["L_actual"] for r in rows]  # bytes per token
    avg_bytes_per_token = sum(sum_y) / n
    asymptotic_cr = (d_head * n_kv * 2 * 2) / avg_bytes_per_token
    print(f"  Avg DS memory per token: {avg_bytes_per_token:.1f} bytes")
    print(f"  Asymptotic compression ratio: {asymptotic_cr:.1f}x")
    print(f"  Equivalent bit-width: {(avg_bytes_per_token * 8) / (d_head * n_kv * 2):.2f} bits/element")

    # ── Write CSV ──────────────────────────────────────────────────────
    output_path = args.output_csv
    fieldnames = ["L_target", "L_actual", "K_CosSim", "V_CosSim", "Compression", "DS_Mem_KB"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _logger.info(f"\nResults saved to {output_path}")
    print(f"[OK] benchmark_seqlen complete — output: {output_path}")


if __name__ == "__main__":
    main()