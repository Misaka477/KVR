"""KV Fidelity Test — encode→decode round-trip fidelity measurement.

Two modes:
  synthetic (default, CPU): random tensors → encode_kv_cache → reconstruct → metrics
  real (--model required): HuggingFace model forward pass → extract K/V → encode → metrics

Run::
    python tests/test_kv_fidelity.py --mode synthetic
    python tests/test_kv_fidelity.py --mode real --model <path>
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from rina.config import DSKVCacheConfig
from rina.ds_kv_cache import DSKVCacheStore, encode_kv_cache

SEED = 42
S_EQ_LENS = [64, 256, 1024]
D_HEADS = [64, 128]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_logger = logging.getLogger("kv_fidelity")


def compute_metrics(original: torch.Tensor, reconstructed: torch.Tensor, store: DSKVCacheStore) -> dict:
    approx = reconstructed.float()
    orig = original.float()

    mse = F.mse_loss(approx, orig).item()
    signal_power = (orig ** 2).mean().item()
    noise_power = ((orig - approx) ** 2).mean().item()
    snr_db = 10 * math.log10(signal_power / (noise_power + 1e-12))

    cos_sim = F.cosine_similarity(
        approx.flatten().unsqueeze(0),
        orig.flatten().unsqueeze(0),
    ).item()

    max_abs_error = (approx - orig).abs().max().item()

    original_bytes = original.element_size() * original.numel()
    comp_ratio = original_bytes / (store.memory_bytes + 1e-12)

    return {
        "cos_sim": float(cos_sim),
        "mse": float(mse),
        "snr_db": float(snr_db),
        "max_abs_error": float(max_abs_error),
        "compression_ratio": float(comp_ratio),
    }


def _make_config() -> DSKVCacheConfig:
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
        cross_head_error_share=False,
        transform_mode="none",
        adaptive_masking=False,
        mask_outlier_threshold=3.0,
        mask_n_steps_boost=0,
        mask_proj_beta_boost=0.0,
        use_mask_gating=False,
        base_dtype="fp16",
        verbose=False,
    )


def run_synthetic(cfg: DSKVCacheConfig) -> list[dict]:
    torch.manual_seed(SEED)
    results = []

    for seq_len in S_EQ_LENS:
        for d_head in D_HEADS:
            _logger.info(f"Synthetic: seq_len={seq_len}, d_head={d_head}")

            k = torch.randn(seq_len, d_head)
            v = torch.randn(seq_len, d_head)

            k_store, v_store = encode_kv_cache(k, v, cfg)
            k_recon = k_store.reconstruct_all(cfg.tile_size, cfg.use_differential)
            v_recon = v_store.reconstruct_all(cfg.tile_size, cfg.use_differential)

            k_metrics = compute_metrics(k, k_recon, k_store)
            v_metrics = compute_metrics(v, v_recon, v_store)

            results.append({
                "name": f"synth_seq{seq_len}_d{d_head}_K",
                "seq_len": seq_len,
                "d_head": d_head,
                "tensor_type": "K",
                **k_metrics,
            })
            results.append({
                "name": f"synth_seq{seq_len}_d{d_head}_V",
                "seq_len": seq_len,
                "d_head": d_head,
                "tensor_type": "V",
                **v_metrics,
            })

            _logger.info(f"  K: CosSim={k_metrics['cos_sim']:.6f} MSE={k_metrics['mse']:.2e} SNR={k_metrics['snr_db']:.2f}dB max_abs_err={k_metrics['max_abs_error']:.2e} CR={k_metrics['compression_ratio']:.1f}x")
            _logger.info(f"  V: CosSim={v_metrics['cos_sim']:.6f} MSE={v_metrics['mse']:.2e} SNR={v_metrics['snr_db']:.2f}dB max_abs_err={v_metrics['max_abs_error']:.2e} CR={v_metrics['compression_ratio']:.1f}x")

    return results


def _past_get_kv(past, layer_idx: int):
    from transformers.cache_utils import DynamicCache
    if isinstance(past, DynamicCache):
        k = past.key_cache[layer_idx]
        v = past.value_cache[layer_idx]
    elif isinstance(past, tuple):
        layer = past[layer_idx]
        k, v = layer[0], layer[1]
    else:
        raise TypeError(f"Unsupported past_key_values type: {type(past)}")
    return k, v


def run_real(model_path: str, cfg: DSKVCacheConfig, prompts: list[str], device: str) -> list[dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    _logger.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    n_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)

    results = []

    for prompt in prompts:
        _logger.info(f"Real mode prompt: \"{prompt[:50]}...\"")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                use_cache=True,
                past_key_values=None,
            )

        past_key_values = output.past_key_values
        seq_len = input_ids.shape[1]

        layer_results = []
        for layer_idx in range(num_layers):
            k_full, v_full = _past_get_kv(past_key_values, layer_idx)
            layer_cfg = cfg.get_layer_config(layer_idx, num_layers)
            is_protected = layer_idx in cfg.protected_layers

            for h in range(n_kv_heads):
                k_orig = k_full[0, h].float()
                v_orig = v_full[0, h].float()

                k_store, v_store = encode_kv_cache(k_orig, v_orig, layer_cfg, protected=is_protected)
                k_recon = k_store.reconstruct_all(layer_cfg.tile_size, layer_cfg.use_differential)
                v_recon = v_store.reconstruct_all(layer_cfg.tile_size, layer_cfg.use_differential)

                k_metrics = compute_metrics(k_orig, k_recon, k_store)
                v_metrics = compute_metrics(v_orig, v_recon, v_store)

                layer_results.append({
                    "layer": layer_idx,
                    "head": h,
                    "k": k_metrics,
                    "v": v_metrics,
                })

        per_layer_cos_k = [r["k"]["cos_sim"] for r in layer_results]
        per_layer_cos_v = [r["v"]["cos_sim"] for r in layer_results]
        per_layer_snr_k = [r["k"]["snr_db"] for r in layer_results]
        per_layer_snr_v = [r["v"]["snr_db"] for r in layer_results]
        per_layer_mse_k = [r["k"]["mse"] for r in layer_results]
        per_layer_mse_v = [r["v"]["mse"] for r in layer_results]
        per_layer_cr = [r["k"]["compression_ratio"] for r in layer_results]

        results.append({
            "name": f"real_{prompt[:30].replace(' ', '_')}",
            "prompt": prompt,
            "seq_len": seq_len,
            "num_layers": num_layers,
            "n_kv_heads": n_kv_heads,
            "avg_cos_sim_k": float(sum(per_layer_cos_k) / len(per_layer_cos_k)),
            "avg_cos_sim_v": float(sum(per_layer_cos_v) / len(per_layer_cos_v)),
            "avg_snr_db_k": float(sum(per_layer_snr_k) / len(per_layer_snr_k)),
            "avg_snr_db_v": float(sum(per_layer_snr_v) / len(per_layer_snr_v)),
            "avg_mse_k": float(sum(per_layer_mse_k) / len(per_layer_mse_k)),
            "avg_mse_v": float(sum(per_layer_mse_v) / len(per_layer_mse_v)),
            "avg_compression_ratio": float(sum(per_layer_cr) / len(per_layer_cr)),
            "per_layer_per_head": layer_results,
        })

        _logger.info(f"  Avg K CosSim={results[-1]['avg_cos_sim_k']:.6f} SNR={results[-1]['avg_snr_db_k']:.2f}dB")
        _logger.info(f"  Avg V CosSim={results[-1]['avg_cos_sim_v']:.6f} SNR={results[-1]['avg_snr_db_v']:.2f}dB")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    p = argparse.ArgumentParser(description="KV Fidelity Test — encode→decode round-trip measurement")
    p.add_argument("--mode", choices=["synthetic", "real"], default="synthetic",
                   help="Test mode: synthetic (random tensors, CPU) or real (HF model)")
    p.add_argument("--model", type=str, default=None,
                   help="Model path (required for real mode)")
    p.add_argument("--output", type=str, default="test_kv_fidelity_results.json",
                   help="JSON output path")
    p.add_argument("--prompts", type=str, nargs="*", default=None,
                   help="Prompts for real mode (default: single short prompt)")
    p.add_argument("--max-seq-len", type=int, default=1024,
                   help="Maximum sequence length for synthetic tests")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Torch device (default: auto-detect)")
    args = p.parse_args()

    if args.mode == "real" and args.model is None:
        p.error("--model is required for real mode")

    ctx = "test_kv_fidelity"
    prompts = args.prompts if args.prompts else ["The future of AI is"]

    cfg = _make_config()

    _logger.info(f"Mode: {args.mode}, Device: {args.device}")

    if args.mode == "synthetic":
        if args.max_seq_len:
            global S_EQ_LENS
            S_EQ_LENS = [s for s in S_EQ_LENS if s <= args.max_seq_len]
        results = run_synthetic(cfg)
    else:
        results = run_real(args.model, cfg, prompts, args.device)

    report = {
        "config": cfg.to_dict(),
        "mode": args.mode,
        "tests": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    _logger.info(f"Results saved to {output_path}")

    print(f"\n{'='*70}")
    print(f"KV Fidelity Summary ({args.mode})")
    print(f"{'='*70}")
    for r in results:
        if args.mode == "synthetic":
            print(f"  {r['name']}: CosSim={r['cos_sim']:.6f} MSE={r['mse']:.2e} "
                  f"SNR={r['snr_db']:.2f}dB MaxAbsErr={r['max_abs_error']:.2e} CR={r['compression_ratio']:.1f}x")
        else:
            print(f"  {r['name']}: K CosSim={r['avg_cos_sim_k']:.6f} SNR={r['avg_snr_db_k']:.2f}dB "
                  f"V CosSim={r['avg_cos_sim_v']:.6f} SNR={r['avg_snr_db_v']:.2f}dB "
                  f"CR={r['avg_compression_ratio']:.1f}x")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
