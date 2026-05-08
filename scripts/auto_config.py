#!/usr/bin/env python
"""
DS-KVCache Auto-Adaptation CLI
===============================

Auto-detect GPU hardware and model architecture, then generate optimal
DSKVCacheConfig parameters via :class:`rina.model_adapter.ModelAdapter`.

Usage:
    python scripts/auto_config.py
    python scripts/auto_config.py --model D:/path/to/model
    python scripts/auto_config.py --model D:/path/to/model --calibrate
    python scripts/auto_config.py --device-preference cuda:0 --output auto_cfg.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rina.config import DSKVCacheConfig
from rina.model_adapter import HardwareProfile, ModelProfile, ModelAdapter

logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("auto_config")


# ══════════════════════════════════════════════════════════════════════════════
# Report printer (CLI-only formatting)
# ══════════════════════════════════════════════════════════════════════════════


def print_report(
    hw: HardwareProfile,
    profile: ModelProfile,
    cfg: DSKVCacheConfig,
    metrics: Optional[dict] = None,
):
    """Pretty-print the auto-detected config."""
    print("\n" + "=" * 70)
    print("  DS-KVCache 自动适配报告")
    print("=" * 70)

    print("\n  ── 硬件 ──")
    print(f"    GPU:          {hw.name}")
    print(f"    VRAM:         {hw.vram_gb:.1f} GB")
    print(f"    CC:           {hw.compute_capability}")
    print(f"    Device:       {hw.recommended_device}")

    print("\n  ── 模型架构 ──")
    print(f"    Type:         {profile.model_name}")
    print(f"    Layers:       {profile.num_layers}")
    print(f"    Q/KV heads:   {profile.num_attention_heads}/{profile.num_kv_heads}")
    print(f"    d_head:       {profile.d_head}")
    print(f"    GQA ratio:    {profile.gqa_ratio}x")
    print(f"    Size:         {profile.model_size_category}")
    print(f"    Max seq len:  {profile.max_position_embeddings}")

    print("\n  ── 推荐配置 ──")
    print(f"    n_steps_k          = {cfg.get_n_steps_k()}")
    print(f"    n_steps_v          = {cfg.get_n_steps_v()}")
    print(f"    tile_size          = {cfg.tile_size}")
    print(f"    beta (Σ-Δ)         = {cfg.beta}")
    print(f"    noise_shaping      = {cfg.use_noise_shaping}")
    print(f"    proj_rank          = {cfg.proj_rank}")
    print(f"    proj_beta          = {cfg.proj_beta}")
    print(f"    adaptive_eta       = {cfg.adaptive_eta}")
    print(f"    order2_gamma       = {cfg.order2_gamma}")
    print(f"    cross_token_group  = {cfg.cross_token_group}")
    print(f"    v_orthogonal       = {cfg.v_orthogonal_transform}")
    print(f"    differential       = {cfg.use_differential}")
    print(f"    diff_strategy      = {cfg.diff_strategy}")
    print(f"    diff_residual_gamma= {cfg.diff_residual_gamma}")
    print(f"    diff_residual_n    = {cfg.diff_residual_n_steps}")
    print(f"    adaptive_n         = {cfg.adaptive_n}")

    if metrics:
        print("\n  ── 校准结果 ──")
        if "error" in metrics:
            print(f"    ❌ 错误: {metrics['error']}")
        else:
            print(f"    K  SNR:      {metrics.get('K_SNR_dB', 'N/A')} dB")
            print(f"    V  SNR:      {metrics.get('V_SNR_dB', 'N/A')} dB")
            print(f"    K  CosSim:   {metrics.get('K_CosSim', 'N/A')}")
            print(f"    V  CosSim:   {metrics.get('V_CosSim', 'N/A')}")
            print(f"    压缩率:      {metrics.get('Compression', 'N/A')}x")
            print(f"    达标:        {'✅' if metrics.get('passed', False) else '⚠️'}")

    print("\n" + "═" * 70)
    print("  要使用此配置:")
    print("    from rina.config import DSKVCacheConfig")
    print("    cfg = DSKVCacheConfig.from_dict(config_dict)")
    print("    model = DSKVCacheModel(hf_model, tokenizer, cfg=cfg)")
    print("═" * 70 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def main():
    p = argparse.ArgumentParser(
        description="DS-KVCache 自动适配 — 检测硬件 & 模型, 生成最优配置",
    )
    p.add_argument(
        "--model", type=str,
        default="D:/Software_Development/Project/models/Llama-3.2-1B",
        help="HuggingFace 模型路径或本地路径",
    )
    p.add_argument(
        "--calibrate", action="store_true", default=False,
        help="运行快速校准 (前向传播 + DS 编码, 测量 SNR/CosSim)",
    )
    p.add_argument(
        "--auto-tune", action="store_true", default=False,
        help="校准不达标时自动调整 n_steps (需要 --calibrate)",
    )
    p.add_argument(
        "--quality", type=str, default="balanced",
        choices=["quality", "balanced", "speed"],
        help="质量偏好: quality=最大保真度, balanced=平衡, speed=最大压缩率",
    )
    p.add_argument(
        "--seq-len-hint", type=int, default=None,
        help="预期最大序列长度 (用于 adaptive_n 决策)",
    )
    p.add_argument(
        "--device-preference", type=str, default=None,
        help="设备偏好 (如 cuda:0, mps, cpu)",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="输出 JSON 配置文件路径",
    )
    p.add_argument(
        "--json", action="store_true", default=False,
        help="仅输出 JSON (静默模式, 用于脚本集成)",
    )
    p.add_argument(
        "--protected-layers", type=str, default=None,
        help="逗号分隔的保护层索引, 如 '0,15' 表示层 0 和 15 使用 FP16 无损存储。"
             "未指定时自动设置为首尾各 1 层 (0, num_layers-1)",
    )
    args = p.parse_args()

    # ── 1. Hardware detection ──
    hw = HardwareProfile.detect()
    if args.device_preference:
        hw.recommended_device = args.device_preference

    # ── 2. Model detection ──
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    profile = ModelProfile.from_hf_config(hf_config)

    if not args.json:
        _logger.info("🔍 检测硬件 ...")
        _logger.info(f"   GPU: {hw.name} ({hw.vram_gb:.1f} GB)")
        _logger.info("🔍 检测模型架构 ...")
        _logger.info(
            f"   {profile.model_name}: {profile.num_layers} layers, "
            f"{profile.num_attention_heads}Q/{profile.num_kv_heads}KV, "
            f"d_head={profile.d_head}, "
            f"GQA={profile.gqa_ratio}x, "
            f"规模={profile.model_size_category}"
        )

    # ── 2b. Protected layers ──
    if args.protected_layers is not None:
        protected_layers = sorted({int(x.strip()) for x in args.protected_layers.split(",") if x.strip()})
    else:
        if profile.num_layers > 0:
            protected_layers = sorted({0, profile.num_layers - 1})
        else:
            protected_layers = []

    # ── 3. Generate config ──
    adapter = ModelAdapter(profile, hw)
    cfg = adapter.recommend_config(quality=args.quality, max_seq_len_hint=args.seq_len_hint)
    cfg.protected_layers = protected_layers

    if not args.json:
        _logger.info("⚙️  生成推荐配置 ...")

    # ── 4. Optional calibration ──
    metrics = None
    if args.calibrate:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        if args.auto_tune:
            cfg = adapter.auto_tune(model, quality=args.quality, max_seq_len_hint=args.seq_len_hint)
        else:
            cfg = adapter.quick_calibrate(model, cfg)

    # ── 5. Output ──
    if args.json:
        output = {
            "hardware": {
                "gpu_name": hw.name,
                "vram_gb": hw.vram_gb,
                "compute_capability": list(hw.compute_capability),
                "device": hw.recommended_device,
            },
            "model": {
                "model_type": profile.model_name,
                "num_layers": profile.num_layers,
                "num_q_heads": profile.num_attention_heads,
                "num_kv_heads": profile.num_kv_heads,
                "d_head": profile.d_head,
                "gqa_ratio": profile.gqa_ratio,
                "hidden_size": profile.hidden_size,
                "model_size_category": profile.model_size_category,
            },
            "config": cfg.to_dict(),
            "config_effective": {
                "n_steps_k": cfg.get_n_steps_k(),
                "n_steps_v": cfg.get_n_steps_v(),
            },
        }
        if metrics:
            output["calibration"] = metrics

        json_str = json.dumps(output, indent=2, ensure_ascii=False, default=str)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json_str, encoding="utf-8")
            print(json_str)
        else:
            print(json_str)
    else:
        print_report(hw, profile, cfg, metrics)

        if args.output:
            output = {
                "hardware": {
                    "gpu_name": hw.name,
                    "vram_gb": hw.vram_gb,
                    "device": hw.recommended_device,
                },
                "model": {
                    "model_type": profile.model_name,
                    "num_layers": profile.num_layers,
                    "gqa_ratio": profile.gqa_ratio,
                    "d_head": profile.d_head,
                },
                "config": cfg.to_dict(),
                "config_effective": {
                    "n_steps_k": cfg.get_n_steps_k(),
                    "n_steps_v": cfg.get_n_steps_v(),
                },
            }
            if metrics:
                output["calibration"] = metrics
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(
                json.dumps(output, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            _logger.info(f"配置已保存到: {args.output}")


if __name__ == "__main__":
    main()
