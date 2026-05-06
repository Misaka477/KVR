r"""
DS-KVCache Ablation Study v2 — Heterogeneous K/V + Two-Stage Resid Diff + V Ortho
=================================================================================

Tests each component's contribution (dB / CosSim) to reconstruction quality.
Configurations:
  C₀:  Baseline (1-bit RTN, n_steps_k=2, n_steps_v=3, no noise shaping, no diff)
  C₁:  + Σ-Δ Noise Shaping only
  C₂:  + Two-stage residual differential only  (§7.3, diff_residual_gamma=0.25)
  C₃:  Full DS-KVCache (C₁ + C₂ + V orthogonal transform)

Run::
    python scripts/ablation_study.py
    python scripts/ablation_study.py --model D:/Software_Development/Project/models/Qwen2.5-0.5B --layer 12

Output: ablation_results.csv
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
_logger = logging.getLogger("ablation")

_LOREM = (
    "The transformer architecture has revolutionized natural language processing "
    "by enabling models to process sequential data in parallel rather than "
    "sequentially. This parallelization allows for significantly faster training "
    "times and the ability to scale to much larger datasets. Attention mechanisms "
    "compute contextual representations by weighing the importance of different "
    "input tokens, creating rich embeddings that capture long-range dependencies. "
    "Modern large language models leverage these capabilities to achieve "
    "state-of-the-art performance across a wide range of natural language tasks "
    "including translation, summarization, question answering, and code generation."
)


def build_prompt(tokenizer, target_len: int) -> str:
    ids = tokenizer(_LOREM, add_special_tokens=False)["input_ids"]
    if len(ids) >= target_len:
        ids = ids[:target_len]
    else:
        repeated = ids * ((target_len // len(ids)) + 2)
        ids = repeated[:target_len]
    return tokenizer.decode(ids, skip_special_tokens=True)


def make_config(
    n_steps_k: int, n_steps_v: int,
    tile_size: int, d_head: int,
    use_ns: bool, use_diff: bool, use_v_ortho: bool = False,
) -> DSKVCacheConfig:
    """Build ablation configuration.

    *use_ns* controls Σ-Δ noise shaping.
    *use_diff* controls two-stage residual differential.
    *use_v_ortho* controls V orthogonal rotation (§8.1.4).
    """
    return DSKVCacheConfig(
        n_steps=5,  # fallback only — overridden by n_steps_k / n_steps_v
        n_steps_k=n_steps_k,
        n_steps_v=n_steps_v,
        tile_size=tile_size,
        beta=0.15 if use_ns else 0.0,
        use_noise_shaping=use_ns,
        proj_rank=min(8, d_head // 4),
        proj_beta=0.3 if use_ns else 0.0,
        adaptive_eta=use_ns,
        adaptive_n=False,  # off for ablation purity
        n_upper_bound=max(n_steps_k, n_steps_v) + 5,
        use_differential=use_diff,
        diff_strategy="residual",  # two-stage residual (§7.3); ignored for 'residual' strategy
        diff_residual_gamma=0.25 if use_diff else 0.0,
        diff_residual_n_steps=1,
        v_orthogonal_transform=use_v_ortho,
        order2_gamma=0.0,
        base_dtype="fp16",
        verbose=False,
    )


def evaluate_ablation(K, V, cfg, n_kv, tile_size) -> dict:
    """Encode/decode and return K/V SNR + CosSim averaged over heads."""
    k_snr_all, v_snr_all = [], []
    k_cos_all, v_cos_all = [], []
    total_mem = 0

    for h in range(n_kv):
        k_mat = K[h].float()
        v_mat = V[h].float()
        k_store, v_store = encode_kv_cache(k_mat, v_mat, cfg)
        k_hat = decode_kvcache_store(k_store, tile_size, cfg.use_differential)
        v_hat = decode_kvcache_store(v_store, tile_size, cfg.use_differential)

        # SNR
        for orig, recon, lst in [(k_mat, k_hat, k_snr_all), (v_mat, v_hat, v_snr_all)]:
            signal_power = (orig**2).mean().item()
            noise_power = ((orig - recon)**2).mean().item()
            lst.append(10 * math.log10(signal_power / (noise_power + 1e-12)))

        # CosSim
        k_cos_all.append(
            F.cosine_similarity(k_hat.flatten().unsqueeze(0), k_mat.flatten().unsqueeze(0)).item()
        )
        v_cos_all.append(
            F.cosine_similarity(v_hat.flatten().unsqueeze(0), v_mat.flatten().unsqueeze(0)).item()
        )

        total_mem += k_store.memory_bytes + v_store.memory_bytes

    L_actual = K.shape[1]
    fp16_bytes = L_actual * n_kv * K.shape[2] * 2 * 2  # K+V fp16
    comp_ratio = fp16_bytes / (total_mem + 1e-12)

    return {
        "K_SNR_dB": round(sum(k_snr_all) / n_kv, 2),
        "V_SNR_dB": round(sum(v_snr_all) / n_kv, 2),
        "K_CosSim": round(sum(k_cos_all) / n_kv, 4),
        "V_CosSim": round(sum(v_cos_all) / n_kv, 4),
        "Compression": round(comp_ratio, 2),
        "DS_Mem_KB": round(total_mem / 1024, 2),
    }


def main():
    p = argparse.ArgumentParser(description="DS-KVCache ablation study v2")
    p.add_argument("--model", type=str,
                   default="D:/Software_Development/Project/models/Qwen2.5-0.5B",
                   help="Model path")
    p.add_argument("--layer", type=int, default=12, help="Layer index")
    p.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    p.add_argument("--n-steps-k", type=int, default=2)
    p.add_argument("--n-steps-v", type=int, default=3)
    p.add_argument("--tile-size", type=int, default=16)
    p.add_argument("--output-csv", type=str, default="ablation_results.csv")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    n_steps_k = args.n_steps_k
    n_steps_v = args.n_steps_v

    # ── Load model & extract K/V once ──────────────────────────────────
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
        _logger.error(f"Layer {args.layer} out of range")
        sys.exit(1)

    prompt = build_prompt(tokenizer, args.seq_len)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], use_cache=True)
    past = outputs.past_key_values
    K, V = past[args.layer]
    K, V = K[0], V[0]  # (n_kv_heads, seq_len, d_head)
    L_actual = K.shape[1]
    _logger.info(
        f"Layer {args.layer}, L={L_actual}, "
        f"{n_heads}Q/{n_kv}KV, d_head={d_head}, "
        f"n_steps_k={n_steps_k}, n_steps_v={n_steps_v}"
    )

    # ── Run ablations ──────────────────────────────────────────────────
    configs = {
        "C0_baseline_1bit_RTN": make_config(
            n_steps_k, n_steps_v, args.tile_size, d_head,
            use_ns=False, use_diff=False, use_v_ortho=False,
        ),
        "C1_noise_shaping_only": make_config(
            n_steps_k, n_steps_v, args.tile_size, d_head,
            use_ns=True, use_diff=False, use_v_ortho=False,
        ),
        "C2_differential_only": make_config(
            n_steps_k, n_steps_v, args.tile_size, d_head,
            use_ns=False, use_diff=True, use_v_ortho=False,
        ),
        "C3_full_DS_KVCache": make_config(
            n_steps_k, n_steps_v, args.tile_size, d_head,
            use_ns=True, use_diff=True, use_v_ortho=True,
        ),
    }

    results = {}
    for name, cfg in configs.items():
        _logger.info(f"  → {name} ...")
        results[name] = evaluate_ablation(K, V, cfg, n_kv, args.tile_size)
        _logger.info(
            f"      K SNR={results[name]['K_SNR_dB']:.1f} dB  "
            f"V SNR={results[name]['V_SNR_dB']:.1f} dB  "
            f"K CosSim={results[name]['K_CosSim']:.4f}  "
            f"V CosSim={results[name]['V_CosSim']:.4f}  "
            f"Compress={results[name]['Compression']:.1f}x"
        )

    # ── Compute deltas ─────────────────────────────────────────────────
    baseline = results["C0_baseline_1bit_RTN"]
    c1_delta_k = results["C1_noise_shaping_only"]["K_SNR_dB"] - baseline["K_SNR_dB"]
    c1_delta_v = results["C1_noise_shaping_only"]["V_SNR_dB"] - baseline["V_SNR_dB"]
    c2_delta_k = results["C2_differential_only"]["K_SNR_dB"] - baseline["K_SNR_dB"]
    c2_delta_v = results["C2_differential_only"]["V_SNR_dB"] - baseline["V_SNR_dB"]
    c3_delta_k = results["C3_full_DS_KVCache"]["K_SNR_dB"] - baseline["K_SNR_dB"]
    c3_delta_v = results["C3_full_DS_KVCache"]["V_SNR_dB"] - baseline["V_SNR_dB"]

    # ── Print summary table ────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"  Ablation Study v2 — Layer {args.layer}, L={L_actual}, "
          f"n_steps_k={n_steps_k}, n_steps_v={n_steps_v}")
    print("=" * 90)
    print(
        f"  {'Config':<32} {'K SNR':>8} {'ΔK dB':>8} {'V SNR':>8} {'ΔV dB':>8} "
        f"{'CosSim(K)':>10} {'CosSim(V)':>10} {'Compress':>10}"
    )
    print("  " + "-" * 88)

    config_order = [
        ("C0_baseline_1bit_RTN",   "C0: 1-bit RTN baseline"),
        ("C1_noise_shaping_only",   "C1: + Σ-Δ Noise Shaping"),
        ("C2_differential_only",    "C2: + 2-Stage Diff (γ=0.25)"),
        ("C3_full_DS_KVCache",      "C3: Full (NS + Diff + V Ortho)"),
    ]
    deltas_k = [0.0, c1_delta_k, c2_delta_k, c3_delta_k]
    deltas_v = [0.0, c1_delta_v, c2_delta_v, c3_delta_v]

    for (key, label), dk, dv in zip(config_order, deltas_k, deltas_v):
        r = results[key]
        print(
            f"  {label:<32} {r['K_SNR_dB']:>6.1f}  {dk:>+6.1f} dB"
            f"  {r['V_SNR_dB']:>6.1f}  {dv:>+6.1f} dB"
            f"  {r['K_CosSim']:>10.4f}"
            f"  {r['V_CosSim']:>10.4f}"
            f"  {r['Compression']:>8.1f}x"
        )

    print("=" * 90)

    # ── Key findings ───────────────────────────────────────────────────
    v_cosim_c3 = results["C3_full_DS_KVCache"]["V_CosSim"]
    print(f"\n  Key Findings:")
    print(f"    Σ-Δ Noise Shaping contribution:        ΔK={c1_delta_k:+.1f} dB  ΔV={c1_delta_v:+.1f} dB")
    print(f"    2-stage Diff (γ=0.25) contribution:    ΔK={c2_delta_k:+.1f} dB  ΔV={c2_delta_v:+.1f} dB")
    print(f"    Full DS-KVCache (NS+Diff+V Ortho):     ΔK={c3_delta_k:+.1f} dB  ΔV={c3_delta_v:+.1f} dB")

    synergy_k = c3_delta_k - (c1_delta_k + c2_delta_k)
    synergy_v = c3_delta_v - (c1_delta_v + c2_delta_v)
    print(f"    Synergy (combined - sum):              ΔK={synergy_k:+.1f} dB  ΔV={synergy_v:+.1f} dB")
    print(f"    V CosSim (C3):                         {v_cosim_c3:.4f}  "
          f"({'✅ ≥ 0.99' if v_cosim_c3 >= 0.99 else '❌ < 0.99'})")

    # ── Write CSV ──────────────────────────────────────────────────────
    output_path = args.output_csv
    fieldnames = [
        "Config", "K_SNR_dB", "V_SNR_dB", "K_CosSim", "V_CosSim",
        "Compression", "DS_Mem_KB", "Delta_K_dB", "Delta_V_dB",
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (key, label), dk, dv in zip(config_order, deltas_k, deltas_v):
            r = results[key]
            writer.writerow({
                "Config": label,
                "K_SNR_dB": r["K_SNR_dB"],
                "V_SNR_dB": r["V_SNR_dB"],
                "K_CosSim": r["K_CosSim"],
                "V_CosSim": r["V_CosSim"],
                "Compression": r["Compression"],
                "DS_Mem_KB": r["DS_Mem_KB"],
                "Delta_K_dB": dk,
                "Delta_V_dB": dv,
            })
    _logger.info(f"\nResults saved to {output_path}")
    print(f"[OK] ablation_study complete — output: {output_path}")


if __name__ == "__main__":
    main()