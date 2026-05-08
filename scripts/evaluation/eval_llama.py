r"""
DS-KVCache Real Model Evaluation (Heterogeneous K/V)
=====================================================

Evaluates DS-KVCache encoding on real attention K/V tensors from a
HuggingFace language model (Llama 3.2 1B by default: GQA 32Q/8KV, d_head=64, 16 layers).

Heterogeneous encoding:
  • K uses n_steps_k (default 3)
  • V uses n_steps_v (default 5) — V needs more protection
  • Optional V orthogonal transform (§8.1.4)
  • Two-stage residual differential encoding (§7.3)

Metrics (per layer):
  - Reconstruction MSE / SNR / CosSim (K and V separately)
  - Attention output MSE / CosSim (how DS-KVCache affects attention score)
  - Compression ratio vs FP16 cache

Run::
    python scripts/eval_llama.py
    python scripts/eval_llama.py --model D:/Software_Development/Project/models/Llama-3.2-1B --n-steps-k 3 --n-steps-v 5
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
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

logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("eval_llama")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def extract_kv_from_hf(
    model,
    tokenizer,
    text: str,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Run a single forward pass through the model and collect per-layer K/V.

    Returns list of (K, V) tensors per layer, each (n_kv_heads, seq_len, d_head).
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=False,
        )

    past = outputs.past_key_values
    kv_pairs = []
    for k_cache, v_cache in past:
        # DynamicCache stores (batch, n_kv_heads, seq_len, d_head)
        kv_pairs.append((k_cache[0], v_cache[0]))  # remove batch dim

    return kv_pairs


def compute_attention_scores(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    K_hat: torch.Tensor,
    V_hat: torch.Tensor,
) -> dict:
    """Compute attention outputs for original vs DS-KVCache K/V.

    Parameters
    ----------
    Q: (n_heads, seq_len, d_head) — query states
    K, K_hat: (n_kv_heads, seq_len, d_head) — original & DS-encoded keys
    V, V_hat: (n_kv_heads, seq_len, d_head) — original & DS-encoded values

    Returns
    -------
    dict with attn_mse, attn_cos_sim, weight_mae
    """
    n_heads, seq_len, d_head = Q.shape
    n_kv = K.shape[0]
    scale = math.sqrt(d_head)

    # GQA: repeat K/V heads to match Q heads
    n_groups = n_heads // n_kv
    if n_groups > 1:
        K_orig = K.repeat_interleave(n_groups, dim=0)
        V_orig = V.repeat_interleave(n_groups, dim=0)
        K_ds = K_hat.repeat_interleave(n_groups, dim=0)
        V_ds = V_hat.repeat_interleave(n_groups, dim=0)
    else:
        K_orig, V_orig = K, V
        K_ds, V_ds = K_hat, V_hat

    # Original attention
    attn_weights_orig = torch.softmax(
        (Q @ K_orig.transpose(-2, -1)) / scale, dim=-1
    )
    attn_out_orig = attn_weights_orig @ V_orig

    # DS-KVCache attention
    attn_weights_ds = torch.softmax(
        (Q @ K_ds.transpose(-2, -1)) / scale, dim=-1
    )
    attn_out_ds = attn_weights_ds @ V_ds

    # Metrics
    attn_mse = F.mse_loss(attn_out_ds, attn_out_orig).item()
    attn_cos = F.cosine_similarity(
        attn_out_ds.flatten().unsqueeze(0),
        attn_out_orig.flatten().unsqueeze(0),
    ).item()
    weight_mae = F.l1_loss(attn_weights_ds, attn_weights_orig).item()

    return {
        "attn_mse": attn_mse,
        "attn_cos_sim": attn_cos,
        "weight_mae": weight_mae,
    }


def compute_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
    """Return MSE, SNR, CosSim."""
    o = original.float()
    r = reconstructed.float()

    mse = F.mse_loss(r, o).item()
    signal_power = (o**2).mean().item()
    noise_power = ((o - r)**2).mean().item()
    snr_db = 10 * math.log10(signal_power / (noise_power + 1e-12))

    cos_sim = F.cosine_similarity(
        r.flatten().unsqueeze(0), o.flatten().unsqueeze(0),
    ).item()

    return {"MSE": mse, "SNR_dB": snr_db, "CosSim": cos_sim}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def main():
    p = argparse.ArgumentParser(description="DS-KVCache real model evaluation (heterogeneous K/V)")
    p.add_argument("--model", type=str,
                   default="D:/Software_Development/Project/models/Llama-3.2-1B",
                   help="Model path or HF model name")
    # ── Heterogeneous N (K vs V) ────────────────────────────────────────
    p.add_argument("--n-steps", type=int, default=None,
                   help="Unified n_steps fallback (if n-steps-k/v not set)")
    p.add_argument("--n-steps-k", type=int, default=3,
                   help="Number of 1-bit bases for Key path (default: 3)")
    p.add_argument("--n-steps-v", type=int, default=5,
                   help="Number of 1-bit bases for Value path (default: 5)")
    # ── Encoding hyperparameters ────────────────────────────────────────
    p.add_argument("--tile-size", type=int, default=16,
                   help="Tile size")
    p.add_argument("--beta", type=float, default=0.15,
                   help="Σ-Δ momentum")
    p.add_argument("--text", type=str,
                   default="The transformer architecture has revolutionized "
                           "natural language processing by enabling models to "
                           "process sequential data in parallel rather than "
                           "sequentially. This parallelization allows for "
                           "significantly faster training times and the "
                           "ability to scale to much larger datasets. "
                           "Attention mechanisms compute contextual "
                           "representations by weighing the importance of "
                           "different input tokens, creating rich embeddings "
                           "that capture long-range dependencies.",
                   help="Input text for extracting K/V")
    p.add_argument("--layers", type=str, default="all",
                   help="Comma-separated layer indices to evaluate, or 'all'")
    # ── Adaptive N ─────────────────────────────────────────────────────
    p.add_argument("--adaptive-n", action="store_true", default=False,
                   help="Enable adaptive N scheduling (off by default for consistent compression)")
    # ── Noise shaping ──────────────────────────────────────────────────
    p.add_argument("--use-ns", action="store_true", default=True,
                   help="Enable noise shaping (default: on)")
    p.add_argument("--no-ns", dest="use_ns", action="store_false",
                   help="Disable noise shaping")
    # ── Differential ───────────────────────────────────────────────────
    p.add_argument("--use-diff", action="store_true", default=True,
                   help="Enable differential encoding (default: on)")
    p.add_argument("--no-diff", dest="use_diff", action="store_false",
                   help="Disable differential encoding")
    p.add_argument("--diff-gamma", type=float, default=0.25,
                   help="Residual shrinkage factor γ for two-stage diff (default: 0.25)")
    p.add_argument("--diff-residual-nsteps", type=int, default=1,
                   help="N bases for residual stage (default: 1)")
    # ── V orthogonal transform (§8.1.4) ─────────────────────────────────
    p.add_argument("--v-ortho", action="store_true", default=True,
                   help="Enable V orthogonal transform (default: on)")
    p.add_argument("--no-v-ortho", dest="v_ortho", action="store_false",
                   help="Disable V orthogonal transform")
    # ── Misc ───────────────────────────────────────────────────────────
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    # ── Load model ─────────────────────────────────────────────────────
    _logger.info(f"Loading {args.model} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model.eval()

    # Architecture info (robust across different HF config formats)
    mcfg = model.config
    n_heads = getattr(mcfg, "num_attention_heads",
               getattr(mcfg, "n_head", None))
    n_kv = getattr(mcfg, "num_key_value_heads",
            getattr(mcfg, "n_kv_head", n_heads))
    d_head = getattr(mcfg, "head_dim", mcfg.hidden_size // n_heads)
    n_layers = getattr(mcfg, "num_hidden_layers",
                getattr(mcfg, "n_layer", None))

    _logger.info(
        f"Architecture: {n_layers} layers, {n_heads}Q/{n_kv}KV heads, d_head={d_head}"
    )

    # ── Extract K/V ────────────────────────────────────────────────────
    _logger.info("Extracting K/V from forward pass ...")
    kv_layers = extract_kv_from_hf(model, tokenizer, args.text)
    seq_len = kv_layers[0][0].shape[1]
    _logger.info(f" Sequence length: {seq_len}")

    # ── Resolve n_steps ────────────────────────────────────────────────
    fallback_n = args.n_steps if args.n_steps is not None else max(args.n_steps_k, args.n_steps_v)

    # ── Config (heterogeneous K/V, no adaptive_n for consistent compression) ──
    cfg = DSKVCacheConfig(
        n_steps=fallback_n,
        n_steps_k=args.n_steps_k,
        n_steps_v=args.n_steps_v,
        tile_size=args.tile_size,
        beta=args.beta,
        use_noise_shaping=args.use_ns,
        proj_rank=min(8, d_head // 4),
        proj_beta=0.3 if args.use_ns else 0.0,
        adaptive_eta=args.use_ns,
        adaptive_n=args.adaptive_n,
        n_upper_bound=max(args.n_steps_k, args.n_steps_v) + 2 if args.adaptive_n else 10,
        use_differential=args.use_diff,
        diff_strategy="residual", diff_residual_gamma=args.diff_gamma,
        diff_residual_n_steps=args.diff_residual_nsteps,
        v_orthogonal_transform=args.v_ortho,
        order2_gamma=0.0,
        base_dtype="fp16",
        verbose=False,
    )

    _logger.info(
        f"Config: n_steps_k={cfg.get_n_steps_k()}, n_steps_v={cfg.get_n_steps_v()}, "
        f"v_ortho={cfg.v_orthogonal_transform}, "
        f"diff_strategy={cfg.diff_strategy}, diff_gamma={cfg.diff_residual_gamma}, "
        f"adaptive_n={cfg.adaptive_n}"
    )

    # ── Evaluate layers ────────────────────────────────────────────────
    if args.layers == "all":
        eval_layers = list(range(n_layers))
    else:
        eval_layers = [int(x) for x in args.layers.split(",")]

    rows = []

    for layer_idx in eval_layers:
        if layer_idx >= n_layers:
            continue

        K, V = kv_layers[layer_idx]
        # K, V: (n_kv_heads, seq_len, d_head)

        # Per-head metrics
        k_metrics_all = []
        v_metrics_all = []
        total_mem_bytes = 0

        t0 = time.perf_counter()

        for h in range(n_kv):
            k_mat = K[h].float()   # (seq_len, d_head)
            v_mat = V[h].float()

            k_store, v_store = encode_kv_cache(k_mat, v_mat, cfg)

            k_hat = decode_kvcache_store(k_store, cfg.tile_size, cfg.use_differential)
            v_hat = decode_kvcache_store(v_store, cfg.tile_size, cfg.use_differential)

            k_metrics_all.append(compute_metrics(k_mat, k_hat))
            v_metrics_all.append(compute_metrics(v_mat, v_hat))

            total_mem_bytes += k_store.memory_bytes + v_store.memory_bytes

        elapsed = time.perf_counter() - t0

        # Aggregate
        avg_k_mse = sum(m["MSE"] for m in k_metrics_all) / n_kv
        avg_k_snr = sum(m["SNR_dB"] for m in k_metrics_all) / n_kv
        avg_k_cos = sum(m["CosSim"] for m in k_metrics_all) / n_kv
        avg_v_mse = sum(m["MSE"] for m in v_metrics_all) / n_kv
        avg_v_snr = sum(m["SNR_dB"] for m in v_metrics_all) / n_kv
        avg_v_cos = sum(m["CosSim"] for m in v_metrics_all) / n_kv

        # Compression
        fp16_bytes = seq_len * n_kv * d_head * 2 * 2  # K+V, fp16
        comp_ratio = fp16_bytes / (total_mem_bytes + 1e-12)

        rows.append([
            f"Layer {layer_idx:02d}",
            f"{avg_k_cos:.4f}",
            f"{avg_k_snr:.1f}",
            f"{avg_k_mse:.6f}",
            f"{avg_v_cos:.4f}",
            f"{avg_v_snr:.1f}",
            f"{avg_v_mse:.6f}",
            f"{comp_ratio:.1f}x",
            f"{total_mem_bytes / 1024:.1f} KB",
            f"{elapsed * 1000:.1f} ms",
        ])

    # ── Print results ──────────────────────────────────────────────────
    print("\n")
    print(tabulate(
        rows,
        headers=[
            "Layer", "K CosSim", "K SNR", "K MSE",
            "V CosSim", "V SNR", "V MSE",
            "Compress", "DS Mem", "Time",
        ],
        tablefmt="grid",
    ))

    # ── Summary ────────────────────────────────────────────────────────
    avg_k_cos_all = sum(float(r[1]) for r in rows) / len(rows)
    avg_v_cos_all = sum(float(r[4]) for r in rows) / len(rows)
    avg_comp_all = sum(float(r[7].replace("x", "")) for r in rows) / len(rows)

    print(f"\nSummary across {len(rows)} layers:")
    print(f"  Avg K CosSim:  {avg_k_cos_all:.4f}")
    print(f"  Avg V CosSim:  {avg_v_cos_all:.4f}")
    print(f"  Avg Compression: {avg_comp_all:.1f}x")

    # ── Check pass/fail ────────────────────────────────────────────────
    if avg_v_cos_all >= 0.99:
        print(f"\n[PASS] V CosSim ({avg_v_cos_all:.4f}) >= 0.99 acceptance threshold")
    else:
        print(f"\n[FAIL] V CosSim ({avg_v_cos_all:.4f}) < 0.99 acceptance threshold")

    if avg_comp_all >= 3.0:
        print(f"[PASS] Compression ({avg_comp_all:.1f}x) >= 3.0x acceptance threshold")
    else:
        print(f"[FAIL] Compression ({avg_comp_all:.1f}x) < 3.0x acceptance threshold")

    print(f"[OK] eval_llama complete.")


if __name__ == "__main__":
    main()