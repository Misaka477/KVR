"""
§A Roadmap 3 — DCT/DWT/Hybrid Transform Ablation (Llama 3.2 1B)
===============================================================

Evaluates DS-KVCache reconstruction quality across all transform modes:
  none, fwht, dct, dwt, hybrid, auto

Run::
    python _ablation_roadmap3_dct.py
    python _ablation_roadmap3_dct.py --model D:/Software_Development/Project/models/Llama-3.2-1B --layers 0,4,8,12
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
from rina.ds_kv_cache import encode_kv_cache, decode_kvcache_store

logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("ablation_roadmap3")

TRANSFORM_MODES = ["none", "fwht", "dct", "dwt", "hybrid", "auto"]


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


def main():
    p = argparse.ArgumentParser(description="Roadmap 3 — DCT/DWT/Hybrid ablation")
    p.add_argument("--model", type=str, default="D:/Software_Development/Project/models/Llama-3.2-1B")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--layers", type=str, default="all",
                   help="Comma-separated layer indices, or 'all'")
    p.add_argument("--n-steps-k", type=int, default=3)
    p.add_argument("--n-steps-v", type=int, default=5)
    p.add_argument("--tile-size", type=int, default=16)
    p.add_argument("--beta", type=float, default=0.15)
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
                           "that capture long-range dependencies.")
    p.add_argument("--modes", type=str, default="all",
                   help="Comma-separated transform modes, or 'all'")
    p.add_argument("--output", type=str, default="ablation_roadmap3.csv")
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
    _logger.info(f"Architecture: {n_layers} layers, {n_heads}Q/{n_kv}KV heads, d_head={d_head}")

    # ── Extract K/V ────────────────────────────────────────────────────
    _logger.info("Extracting K/V from forward pass ...")
    kv_layers = extract_kv_from_hf(model, tokenizer, args.text)
    seq_len = kv_layers[0][0].shape[1]
    _logger.info(f" Sequence length: {seq_len}")

    # ── Resolve layers ─────────────────────────────────────────────────
    if args.layers == "all":
        eval_layers = list(range(n_layers))
    else:
        eval_layers = [int(x) for x in args.layers.split(",")]

    # ── Resolve modes ──────────────────────────────────────────────────
    if args.modes == "all":
        modes = TRANSFORM_MODES
    else:
        modes = [m.strip() for m in args.modes.split(",")]

    # ── Run ablation ───────────────────────────────────────────────────
    all_rows = []
    base_cfg = dict(
        n_steps=max(args.n_steps_k, args.n_steps_v),
        n_steps_k=args.n_steps_k,
        n_steps_v=args.n_steps_v,
        tile_size=args.tile_size,
        beta=args.beta,
        use_noise_shaping=True,
        proj_rank=min(8, d_head // 4),
        proj_beta=0.3,
        adaptive_eta=True,
        use_differential=True,
        diff_strategy="residual",
        diff_residual_gamma=0.25,
        diff_residual_n_steps=1,
        v_orthogonal_transform=True,
        order2_gamma=0.0,
        verbose=False,
    )

    for mode in modes:
        _logger.info(f"\n{'='*60}")
        _logger.info(f"  Transform mode: {mode}")
        _logger.info(f"{'='*60}")

        mode_rows = []
        mode_k_cos, mode_v_cos, mode_ratio = [], [], []

        for layer_idx in eval_layers:
            K, V = kv_layers[layer_idx]
            layer_k_cos, layer_v_cos = [], []
            total_mem = 0
            t0 = time.perf_counter()

            for h in range(n_kv):
                cfg = DSKVCacheConfig(
                    **base_cfg,
                    transform_mode=mode,
                )

                k_store, v_store = encode_kv_cache(K[h].float(), V[h].float(), cfg)
                k_hat = decode_kvcache_store(k_store, args.tile_size, True)
                v_hat = decode_kvcache_store(v_store, args.tile_size, True)

                km = compute_metrics(K[h].float(), k_hat)
                vm = compute_metrics(V[h].float(), v_hat)
                layer_k_cos.append(km["CosSim"])
                layer_v_cos.append(vm["CosSim"])
                total_mem += k_store.memory_bytes + v_store.memory_bytes

            elapsed = time.perf_counter() - t0
            avg_k_cos = sum(layer_k_cos) / n_kv
            avg_v_cos = sum(layer_v_cos) / n_kv
            fp16_bytes = seq_len * n_kv * d_head * 2 * 2
            comp_ratio = fp16_bytes / (total_mem + 1e-12)
            mode_k_cos.append(avg_k_cos)
            mode_v_cos.append(avg_v_cos)
            mode_ratio.append(comp_ratio)

            mode_rows.append([
                f"L{layer_idx:02d}",
                f"{avg_k_cos:.4f}",
                f"{avg_v_cos:.4f}",
                f"{comp_ratio:.1f}x",
                f"{total_mem/1024:.1f}KB",
                f"{elapsed*1000:.1f}ms",
            ])
            _logger.info(
                f"  Layer {layer_idx:02d}: K-Cos={avg_k_cos:.4f} V-Cos={avg_v_cos:.4f} "
                f"Ratio={comp_ratio:.1f}x"
            )

        # Aggregate this mode
        avg_k = sum(mode_k_cos) / len(mode_k_cos)
        avg_v = sum(mode_v_cos) / len(mode_v_cos)
        avg_r = sum(mode_ratio) / len(mode_ratio)
        all_rows.append([mode, f"{avg_k:.4f}", f"{avg_v:.4f}", f"{avg_r:.1f}x"])

        # Print per-mode table
        print(f"\n  --- {mode} per-layer ---")
        print(tabulate(mode_rows, headers=["Layer", "K CosSim", "V CosSim", "Ratio", "Mem", "Time"],
                       tablefmt="grid"))

    # ── Final summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Route 3 — Transform Ablation Summary")
    print(f"{'='*60}")
    print(tabulate(all_rows, headers=["Transform", "Avg K CosSim", "Avg V CosSim", "Avg Ratio"],
                   tablefmt="grid"))

    # ── Find best ──────────────────────────────────────────────────────
    best_mode = max(all_rows, key=lambda r: float(r[2]))  # by V CosSim
    best_ratio = max(all_rows, key=lambda r: float(r[3].replace('x','')))
    _logger.info(f"\nBest reconstruction: {best_mode[0]} (V CosSim={best_mode[2]})")
    _logger.info(f"Best compression:    {best_ratio[0]} (Ratio={best_ratio[3]})")

    # ── Save ───────────────────────────────────────────────────────────
    import csv
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["transform_mode", "avg_k_cossim", "avg_v_cossim", "avg_ratio"])
        w.writerows(all_rows)
    _logger.info(f"Results saved to {args.output}")

    print(f"\n[OK] ablation_roadmap3 complete.")


if __name__ == "__main__":
    main()