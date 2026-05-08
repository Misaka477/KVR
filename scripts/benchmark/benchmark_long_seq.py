r"""DS-KVCache long-context stress test — auto-config with --model argument.

Tests: 8K, 16K, 32K (if VRAM allows).  Reports VRAM, K/V CosSim, Compression.
"""
from __future__ import annotations
import torch, sys, logging, math, time, argparse
sys.path.insert(0, ".")

import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, AutoConfig

from rina.config import DSKVCacheConfig
from rina.ds_kv_cache import encode_kv_cache, decode_kvcache_store
from rina.model_adapter import HardwareProfile, ModelProfile, ModelAdapter

logging.basicConfig(level=logging.INFO, format="%(message)s")
_log = logging.getLogger("stress_test")

DEFAULT_MODEL_PATH = "D:/Software_Development/Project/models/Llama-3.2-1B"


def _make_text(target_tokens: int) -> str:
    """Build a repeating text long enough to reach *target_tokens*."""
    base = (
        "The transformer architecture has revolutionized natural language processing "
        "by enabling models to process sequential data in parallel rather than sequentially. "
        "This parallelization allows for significantly faster training times and the ability "
        "to scale to much larger datasets. "
    )
    # ~10 tokens per repeat of base, pad generously
    repeats = (target_tokens // 12) + 5
    return (base * repeats)[: target_tokens * 6]


def test_one_length(model, tokenizer, target_seq: int, cfg: DSKVCacheConfig):
    """Forward + encode a single sequence length, returning metrics."""
    text = _make_text(target_seq)
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=target_seq).to(model.device)
    actual_len = inp.input_ids.shape[1]
    if actual_len < 64:
        _log.warning("  Sequence too short (%d), skipping", actual_len)
        return None

    # Reset peak memory tracking
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    _log.info("  Forward pass seq_len=%d ...", actual_len)
    t_fwd = time.perf_counter()
    with torch.no_grad():
        out = model(**inp, use_cache=True)
    fwd_time = time.perf_counter() - t_fwd
    vr_after_fwd = torch.cuda.max_memory_allocated() / (1024 ** 2)

    past = out.past_key_values
    k0, v0 = past[0]  # layer 0: (B, n_kv_heads, T, d_head)
    T, n_kv, d_head = k0.shape[2], k0.shape[1], k0.shape[3]
    fp16_kv_bytes = T * n_kv * d_head * 2 * 2  # K+V single layer
    fp16_full_mb = fp16_kv_bytes / (1024 ** 2)

    # Encode head 0 of layer 0
    k_h = k0[0, 0].float()
    v_h = v0[0, 0].float()

    t_enc = time.perf_counter()
    try:
        k_store, v_store = encode_kv_cache(k_h, v_h, cfg)
        k_hat = decode_kvcache_store(k_store, cfg.tile_size, cfg.use_differential)
        v_hat = decode_kvcache_store(v_store, cfg.tile_size, cfg.use_differential)
    except Exception as e:
        _log.error("  encode failed: %s", e)
        return None
    enc_time = time.perf_counter() - t_enc
    vr_after_enc = torch.cuda.max_memory_allocated() / (1024 ** 2)

    k_cos = F.cosine_similarity(k_hat.flatten().unsqueeze(0), k_h.flatten().unsqueeze(0)).item()
    v_cos = F.cosine_similarity(v_hat.flatten().unsqueeze(0), v_h.flatten().unsqueeze(0)).item()
    k_mse = F.mse_loss(k_hat, k_h).item()
    v_mse = F.mse_loss(v_hat, v_h).item()
    k_snr = 10 * math.log10(max((k_h ** 2).mean().item() / max(k_mse, 1e-12), 1e-12))
    v_snr = 10 * math.log10(max((v_h ** 2).mean().item() / max(v_mse, 1e-12), 1e-12))

    ds_kb = (k_store.memory_bytes + v_store.memory_bytes) / 1024
    comp_ratio = fp16_kv_bytes / max(k_store.memory_bytes + v_store.memory_bytes, 1)
    n_tiles = k_store.n_tiles

    # Cleanup to free VRAM for next run
    del out, past, k0, v0, k_h, v_h, k_store, v_store, k_hat, v_hat, inp

    return {
        "seq_len": actual_len,
        "n_tiles": n_tiles,
        "k_cos": k_cos,
        "k_snr": k_snr,
        "v_cos": v_cos,
        "v_snr": v_snr,
        "comp_ratio": comp_ratio,
        "fp16_mb": fp16_full_mb,
        "ds_kb": ds_kb,
        "vr_fwd_mb": vr_after_fwd,
        "vr_enc_mb": vr_after_enc,
        "fwd_s": fwd_time,
        "enc_s": enc_time,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                   help="Path to HuggingFace model directory")
    p.add_argument("--seqs", type=int, nargs="+", default=[8192, 12288, 16384, 24576, 32768])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--only-seq", type=int, default=None,
                   help="Run only a single seq length for quick test")
    args = p.parse_args()

    model_path = args.model

    if not torch.cuda.is_available():
        _log.error("CUDA not available")
        return

    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    _log.info("GPU: %s (%.1f GB VRAM)", torch.cuda.get_device_name(0), total_vram)

    # Auto-detect GPU & model info, generate optimal config
    hw = HardwareProfile.detect()
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    profile = ModelProfile.from_hf_config(hf_config)
    adapter = ModelAdapter(profile, hw)
    cfg = adapter.recommend_config()

    _log.info("Auto Config: n_steps_k=%d, n_steps_v=%d, gamma=%.2f, proj_beta=%.2f, v_ortho=%s",
              cfg.get_n_steps_k(), cfg.get_n_steps_v(),
              cfg.diff_residual_gamma, cfg.proj_beta, cfg.v_orthogonal_transform)

    _log.info("Loading model from %s ...", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    base_vram = torch.cuda.memory_allocated() / (1024 ** 2)
    _log.info("Model VRAM: %.1f MB (fp16)", base_vram)

    seqs = [args.only_seq] if args.only_seq else args.seqs
    results = []
    oom_flags = []

    for seq_len in seqs:
        _log.info("─" * 60)
        _log.info("Testing seq_len=%d ...", seq_len)
        try:
            r = test_one_length(model, tokenizer, seq_len, cfg)
            if r is None:
                oom_flags.append("FAIL")
                continue
            results.append(r)
            oom_flags.append("No")
            _log.info("  OK: K=%.4f V=%.4f Comp=%.1fx VRAM=%.1f MB",
                      r["k_cos"], r["v_cos"], r["comp_ratio"], r["vr_enc_mb"])
        except torch.cuda.OutOfMemoryError:
            _log.error("  OOM at seq_len=%d!", seq_len)
            oom_flags.append("YES")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            _log.error("  Error at seq_len=%d: %s", seq_len, e)
            oom_flags.append("ERR")
            torch.cuda.empty_cache()
            continue

    # ── Report table ────────────────────────────────────────────────────
    if not results:
        _log.error("No successful runs!")
        return

    header = (
        f"{'Seq Len':>8} {'VRAM-MB':>9} {'K CosSim':>10} {'K SNR':>8} "
        f"{'V CosSim':>10} {'V SNR':>8} {'Compress':>10} {'Tiles':>6} "
        f"{'DS KB':>8} {'OOM?':>6}"
    )
    print()
    print(header)
    print("-" * 95)

    for i, r in enumerate(results):
        print(
            f"{r['seq_len']:>8} {r['vr_enc_mb']:>9.1f} "
            f"{r['k_cos']:>10.4f} {r['k_snr']:>8.1f} "
            f"{r['v_cos']:>10.4f} {r['v_snr']:>8.1f} "
            f"{r['comp_ratio']:>10.1f}x {r['n_tiles']:>6} "
            f"{r['ds_kb']:>8.1f} {oom_flags[i]:>6}"
        )

    avg_k = sum(r["k_cos"] for r in results) / len(results)
    avg_v = sum(r["v_cos"] for r in results) / len(results)
    avg_comp = sum(r["comp_ratio"] for r in results) / len(results)

    print()
    print(
        "Avg K CosSim: {:.4f} | Avg V CosSim: {:.4f} | Avg Compression: {:.1f}x".format(
            avg_k, avg_v, avg_comp
        )
    )
    print("V CosSim >= 0.99: {}".format(avg_v >= 0.99))
    print("No OOM: {}".format(all(f == "No" for f in oom_flags) and len(oom_flags) == len(seqs)))
    print("[OK] long-context stress test complete.")


if __name__ == "__main__":
    main()