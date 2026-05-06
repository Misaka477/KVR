#!/usr/bin/env python
"""
DS-KVCache Auto-Adaptation Script
===================================

自动检测用户的 GPU 设备和模型架构，然后根据硬件能力与模型特性
自动生成最优的 DSKVCacheConfig 参数配置。

功能：
  1. 硬件探测 — GPU 型号、VRAM、计算能力
  2. 模型探测 — 层数、注意力头数、GQA 比率、d_head
  3. 参数推荐 — 基于 heuristic 规则自动计算最优参数
  4. 快速校准 — 可选：用少量 token 跑一次前向传播验证 SNR / CosSim
  5. 输出 — 直接输出可用于 DSKVCacheModel 的配置字典/JSON

用法：
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
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rina.config import DSKVCacheConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("auto_config")

# ────────────────────────────────────────────────────────────────────────────
# Hardware Detection
# ────────────────────────────────────────────────────────────────────────────

def detect_gpu_info() -> Dict[str, Any]:
    """检测 GPU 硬件信息。

    Returns
    -------
    dict with keys:
        available, count, name, vram_mb, vram_gb, compute_capability,
        is_nvidia, is_amd, is_apple, recommended_device
    """
    info: Dict[str, Any] = {
        "available": False,
        "count": 0,
        "name": "CPU",
        "vram_mb": 0,
        "vram_gb": 0.0,
        "compute_capability": (0, 0),
        "is_nvidia": False,
        "is_amd": False,
        "is_apple": False,
        "recommended_device": "cpu",
    }

    if torch.cuda.is_available():
        info["available"] = True
        info["count"] = torch.cuda.device_count()
        info["is_nvidia"] = True
        props = torch.cuda.get_device_properties(0)
        info["name"] = props.name
        total_mem = getattr(props, "total_memory", getattr(props, "total_mem", 0))
        info["vram_mb"] = total_mem // (1024 * 1024)
        info["vram_gb"] = round(total_mem / (1024**3), 1)
        info["compute_capability"] = (props.major, props.minor)
        info["recommended_device"] = "cuda"
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        info["available"] = True
        info["count"] = 1
        info["is_apple"] = True
        info["name"] = "Apple MPS"
        # Apple Silicon 统一内存，保守估计可用 75%
        try:
            import subprocess
            result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            total_ram = int(result.stdout.strip())
            info["vram_mb"] = int(total_ram * 0.75) // (1024 * 1024)
            info["vram_gb"] = round(total_ram * 0.75 / (1024**3), 1)
        except Exception:
            info["vram_mb"] = 8192  # 保守默认 8GB
            info["vram_gb"] = 8.0
        info["recommended_device"] = "mps"
    else:
        # CPU fallback — 估算系统内存
        try:
            import psutil
            mem = psutil.virtual_memory()
            info["vram_mb"] = mem.available // (1024 * 1024)
            info["vram_gb"] = round(mem.available / (1024**3), 1)
        except ImportError:
            info["vram_mb"] = 4096
            info["vram_gb"] = 4.0

    return info


# ────────────────────────────────────────────────────────────────────────────
# Model Detection
# ────────────────────────────────────────────────────────────────────────────

def detect_model_info(model_path: str, tokenizer_path: Optional[str] = None) -> Dict[str, Any]:
    """检测模型架构信息。

    Parameters
    ----------
    model_path: HuggingFace 模型路径或本地路径

    Returns
    -------
    dict with keys:
        model_type, num_layers, num_q_heads, num_kv_heads, d_head,
        hidden_size, gqa_ratio, vocab_size, max_position_embeddings,
        rope_theta, intermediate_size, model_size_category, path
    """
    from transformers import AutoConfig

    info: Dict[str, Any] = {
        "path": model_path,
        "model_type": "unknown",
        "num_layers": 0,
        "num_q_heads": 0,
        "num_kv_heads": 0,
        "d_head": 64,
        "hidden_size": 0,
        "gqa_ratio": 1.0,
        "vocab_size": 0,
        "max_position_embeddings": 2048,
        "rope_theta": 10000.0,
        "intermediate_size": 0,
        "model_size_category": "unknown",
    }

    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        _logger.warning(f"无法加载模型配置: {e}")
        return info

    info["model_type"] = getattr(cfg, "model_type", "unknown")
    info["vocab_size"] = getattr(cfg, "vocab_size", 0)

    # ── 层数 ──
    n_layers = getattr(cfg, "num_hidden_layers",
                       getattr(cfg, "n_layer",
                               getattr(cfg, "num_layers", 0)))
    info["num_layers"] = n_layers

    # ── 注意力头 ──
    n_q = getattr(cfg, "num_attention_heads",
                  getattr(cfg, "n_head",
                          getattr(cfg, "num_heads", 0)))
    n_kv = getattr(cfg, "num_key_value_heads",
                   getattr(cfg, "n_kv_head", n_q))  # 未设置 GQA 则 = Q heads

    info["num_q_heads"] = n_q
    info["num_kv_heads"] = n_kv

    # ── head_dim ──
    hidden_size = getattr(cfg, "hidden_size", getattr(cfg, "d_model", 0))
    info["hidden_size"] = hidden_size
    if hasattr(cfg, "head_dim"):
        info["d_head"] = cfg.head_dim
    elif n_q > 0 and hidden_size > 0:
        info["d_head"] = hidden_size // n_q
    else:
        info["d_head"] = 64  # fallback

    # ── GQA ratio ──
    if n_kv > 0:
        info["gqa_ratio"] = round(n_q / n_kv, 2)

    # ── 位置编码 ──
    info["max_position_embeddings"] = getattr(
        cfg, "max_position_embeddings",
        getattr(cfg, "max_seq_len", 2048),
    )
    info["rope_theta"] = getattr(cfg, "rope_theta", 10000.0)

    # ── FFN 中间层 ──
    info["intermediate_size"] = getattr(cfg, "intermediate_size",
                                        getattr(cfg, "ffn_dim", 0))

    # ── 模型规模分类 ──
    total_params_est = _estimate_params(info)
    if total_params_est < 1.5e9:
        info["model_size_category"] = "small"       # < 1.5B
    elif total_params_est < 8e9:
        info["model_size_category"] = "medium"      # 1.5B – 8B
    elif total_params_est < 30e9:
        info["model_size_category"] = "large"       # 8B – 30B
    else:
        info["model_size_category"] = "xlarge"      # > 30B

    return info


def _estimate_params(model_info: Dict[str, Any]) -> float:
    """粗略估算模型参数量 (不含 embedding / LM head)。"""
    L = model_info["num_layers"]
    d = model_info["hidden_size"]
    d_ff = model_info["intermediate_size"] or (d * 8 // 3)

    # Attention: Q, K, V, O 四个投影
    attn_params = 4 * d * d
    # FFN: gate, up, down (SwiGLU 风格)
    ffn_params = 3 * d * d_ff
    # RMS Norm: 2 per layer
    norm_params = 2 * d
    per_layer = attn_params + ffn_params + norm_params
    return L * per_layer


# ────────────────────────────────────────────────────────────────────────────
# Dynamic Gamma — 自动适配残差补偿强度
# ────────────────────────────────────────────────────────────────────────────

def _compute_dynamic_gamma(
    *,
    n_kv_heads: int,
    gqa_ratio: float,
    d_head: int,
    model_size_category: str,
    quality_preference: str = "balanced",
    max_seq_len_hint: Optional[int] = None,
) -> float:
    """动态计算 diff_residual_gamma，基于多因子模型自适应。
    
    三因子模型:
      gamma = base * (1 + α_kv + α_gqa) * (1 + β_dhead) * (1 + β_seq)
    
    因子:
      α_kv:  KV head 稀缺惩罚 (n_kv_heads ≤ 4 时激活)
      α_gqa: GQA 增强修正 (gqa_ratio ≥ 8 时激活)
      β_dhead: d_head 缩放 (d_head ≥ 96 时激活)
      β_seq:  长序列保护 (max_seq_len_hint ≥ 8K 时激活)
      quality: quality→+0.04, speed→-0.04
    
    Examples
    --------
    Llama 3.2 1B  (n_kv=8, GQA=4x,  d_head=64)     → γ=0.15
    Qwen 2.5 0.5B  (n_kv=3, GQA=7x,  d_head=96)     → γ≈0.24
    Qwen 2.5 7B    (n_kv=4, GQA=7x,  d_head=128)    → γ≈0.26
    Mixtral 8×7B   (n_kv=8, GQA=4x,  d_head=128)    → γ≈0.18
    """
    
    # Step 1: base_gamma (静态底值，基于 GQA ratio)
    if gqa_ratio >= 8:
        base_gamma = 0.20
    elif gqa_ratio >= 4:
        base_gamma = 0.15       # ← C1 baseline for GQA=4
    else:  # MHA
        base_gamma = 0.10
    
    # Step 2: α_kv — KV head 稀缺惩罚
    #   n_kv_heads ≤ 2: 极端稀缺 +0.07
    #   n_kv_heads ≤ 4: 中等稀缺 +0.05
    #   n_kv_heads ≤ 8: 轻微稀缺 +0.02
    #   n_kv_heads > 8: 无惩罚
    if n_kv_heads <= 2:
        alpha_kv = 0.07
    elif n_kv_heads <= 4:
        alpha_kv = 0.05
    elif n_kv_heads <= 8:
        alpha_kv = 0.02
    else:
        alpha_kv = 0.0
    
    # Step 3: α_gqa — GQA 增强修正
    #   GQA ≥ 16: +0.05 (极端 GQA, V 头极度压缩)
    #   GQA ≥ 8:  +0.03
    #   GQA ≥ 4:  +0.01 (轻微增强)
    #   GQA < 4:  无修正 (MHA)
    if gqa_ratio >= 16:
        alpha_gqa = 0.05
    elif gqa_ratio >= 8:
        alpha_gqa = 0.03
    elif gqa_ratio >= 4:
        alpha_gqa = 0.01
    else:
        alpha_gqa = 0.0
    
    # Step 4: β_dhead — d_head 缩放因子
    #   d_head ≥ 128: ×1.25 (如 Qwen 7B d_head=128)
    #   d_head ≥ 96:  ×1.20 (如 Qwen 0.5B d_head=96)
    #   d_head < 96:  无缩放
    if d_head >= 128:
        beta_dhead = 0.25
    elif d_head >= 96:
        beta_dhead = 0.20
    else:
        beta_dhead = 0.0
    
    # Step 5: β_seq — 长序列保护因子
    #   max_seq_len_hint ≥ 16K: +0.04
    #   max_seq_len_hint ≥ 8K:  +0.02
    #   max_seq_len_hint < 8K:  无修正
    if max_seq_len_hint is not None:
        if max_seq_len_hint >= 16384:
            beta_seq = 0.04
        elif max_seq_len_hint >= 8192:
            beta_seq = 0.02
        else:
            beta_seq = 0.0
    else:
        beta_seq = 0.0
    
    # Step 6: quality 偏移
    q_delta = {"quality": 0.04, "balanced": 0.0, "speed": -0.04}
    q_offset = q_delta.get(quality_preference, 0.0)
    
    # Compute: gamma = base * (1 + α_kv + α_gqa) * (1 + β_dhead) * (1 + β_seq) + q_offset
    linear_factor = 1.0 + alpha_kv + alpha_gqa
    dhead_factor = 1.0 + beta_dhead
    seq_factor = 1.0 + beta_seq
    
    gamma = base_gamma * linear_factor * dhead_factor * seq_factor + q_offset
    
    # Clamp to [0.05, 0.40]
    gamma = max(0.05, min(0.40, gamma))
    
    return round(gamma, 4)


# ────────────────────────────────────────────────────────────────────────────
# Configuration Strategy
# ────────────────────────────────────────────────────────────────────────────

def generate_optimal_config(
    gpu_info: Dict[str, Any],
    model_info: Dict[str, Any],
    *,
    quality_preference: str = "balanced",
    max_seq_len_hint: Optional[int] = None,
    target_compression: Optional[float] = None,
) -> DSKVCacheConfig:
    """根据硬件和模型信息生成最优 DSKVCacheConfig。

    Parameters
    ----------
    gpu_info: detect_gpu_info() 的输出
    model_info: detect_model_info() 的输出
    quality_preference:
        "quality"  — 最大保真度，压缩率较低
        "balanced" — 平衡保真度和压缩率 (默认)
        "speed"    — 最大压缩率 / 内存效率
    max_seq_len_hint: 预期最大序列长度 (None = 使用模型最大长度)
    target_compression: 目标压缩率 (None = 自动)

    Returns
    -------
    DSKVCacheConfig
    """
    d_head = model_info["d_head"]
    gqa = model_info["gqa_ratio"]
    n_layers = model_info["num_layers"]
    n_kv = model_info["num_kv_heads"]
    category = model_info["model_size_category"]
    vram_gb = gpu_info["vram_gb"]
    compute_cap = gpu_info["compute_capability"]
    device = gpu_info["recommended_device"]

    # ── Tier 1: n_steps_k / n_steps_v (核心压缩/保真度平衡) ──
    #   对标 C1 配置 (n_steps_k=4, n_steps_v=5, 在 Llama 3.2 1B 上验证通过)
    #   规则: d_head 越大 → 更多 step 以保持保真度
    #         GQA 比率越大 → V 更需要保护 (更多 KV head 共享)
    #         quality_preference 决定偏移量
    if d_head <= 64:
        base_nk, base_nv = 4, 5          # ← C1 baseline for d_head=64
    elif d_head <= 96:
        base_nk, base_nv = 4, 6
    elif d_head <= 128:
        base_nk, base_nv = 5, 7
    else:  # > 128 (如 DeepSeek MLA 等)
        base_nk, base_nv = 6, 8

    # GQA 修正: GQA 越大，V 路径需要更多保护（仅对大 GQA 赋能）
    if gqa >= 8:
        base_nv += 1

    # quality 偏移
    q_offset = {"quality": 1, "balanced": 0, "speed": -1}
    offset = q_offset.get(quality_preference, 0)
    n_steps_k = max(1, base_nk + offset)
    n_steps_v = max(1, base_nv + offset)

    # ── Tier 1: tile_size ──
    #   GPU Tensor Core 最佳对齐: 16
    #   CPU (无 GPU): 8 以减少计算量
    #   旧 GPU (< SM 7.0): 保持 16 但关闭部分特性
    if not gpu_info["available"]:
        tile_size = 8
    elif compute_cap[0] < 7:
        tile_size = 16
    else:
        tile_size = 16  # Ampere+ Tensor Core 16×16

    # ── Tier 1: beta (Σ-Δ 动量) ──
    #   d_head 大 → 更大的 residual drift → 更高的 beta
    if d_head <= 64:
        beta = 0.10
    elif d_head <= 96:
        beta = 0.15
    elif d_head <= 128:
        beta = 0.20
    else:
        beta = 0.25

    if quality_preference == "quality":
        beta -= 0.05  # 更保守 = 更少 momentum overshoot
    elif quality_preference == "speed":
        beta += 0.05

    beta = max(0.0, min(0.30, beta))

    # ── Tier 2: Noise Shaping ──
    use_noise_shaping = True  # 始终开启

    # proj_rank: 基于 d_head
    proj_rank = max(4, d_head // 8)  # d_head=64→8, d_head=128→16

    # proj_beta: 基于模型规模 (对标 C1: proj_beta=0.3)
    #   小模型可承受更多噪声 → 更高的 proj_beta
    if category == "small":
        proj_beta = 0.30     # ← C1 baseline for Llama-1B
    elif category == "medium":
        proj_beta = 0.25
    elif category == "large":
        proj_beta = 0.20
    else:  # xlarge
        proj_beta = 0.15

    if quality_preference == "quality":
        proj_beta -= 0.05
    elif quality_preference == "speed":
        proj_beta += 0.05

    proj_beta = max(0.05, min(0.40, proj_beta))
    adaptive_eta = True

    # ── Tier 2: Second-order Σ-Δ (§8.1.2) ──
    #   二阶噪声整形抑制低频量化噪声，对长序列自回归生成至关重要。
    #   d_head 越大 → 量化误差累积越快 → 需要更强的二阶整形。
    if d_head <= 64:
        order2_gamma = 0.30       # ← Phase 1 default for d_head=64 (Llama 1B)
    elif d_head <= 96:
        order2_gamma = 0.25
    elif d_head <= 128:
        order2_gamma = 0.20
    else:
        order2_gamma = 0.15
    order2_c1, order2_c2 = 1.0, 0.5

    if quality_preference == "quality":
        order2_gamma = min(0.40, order2_gamma + 0.05)
    elif quality_preference == "speed":
        order2_gamma = max(0.05, order2_gamma - 0.10)

    # ── Tier 2: Cross-token joint encoding (§8.1.5) ──
    #   将相邻 token 的 KV 拼接成大矩阵后编码，量化误差跨 token 分布。
    #   长序列 (>4096) 推荐 G=4，中等序列 (>512) 推荐 G=2。
    if max_seq_len_hint is not None:
        if max_seq_len_hint >= 4096:
            cross_token_group = 4
        elif max_seq_len_hint >= 512:
            cross_token_group = 2
        else:
            cross_token_group = 1
    else:
        # 无 seq_len 提示时，根据模型最大长度推断
        max_pos = model_info.get("max_position_embeddings", 2048)
        if max_pos >= 8192:
            cross_token_group = 4
        elif max_pos >= 512:
            cross_token_group = 2
        else:
            cross_token_group = 1

    # ── Tier 2: V Orthogonal Transform ──
    #   始终开启 (Google 风格)
    v_orthogonal_transform = True

    # ── Tier 2: Temporal Noise Feedback (§8.1.5) ──
    #   长序列场景下开启时域噪声反馈 (tile-to-tile Σ-Δ loop)
    #   NVIDIA GPU (Tensor Core) + n_tokens ≥ 8K → 开启
    #   CPU / Apple MPS → 关闭 (性价比低)
    time_feedback_gamma = 0.0
    time_feedback_mode = "kv_both"
    if gpu_info["available"] and gpu_info["is_nvidia"] and vram_gb >= 4.0:
        # 仅在 8K+ 序列时有效果
        if max_seq_len_hint and max_seq_len_hint >= 8192:
            time_feedback_gamma = 0.15
        elif quality_preference == "quality":
            time_feedback_gamma = 0.12
        else:
            time_feedback_gamma = 0.10
    # quality 模式: 强化反馈耦合
    if quality_preference == "quality":
        time_feedback_gamma = min(0.20, time_feedback_gamma + 0.03)
    elif quality_preference == "speed":
        time_feedback_gamma = max(0.0, time_feedback_gamma - 0.03)

    # ── Tier 3: Differential ──
    use_differential = True
    diff_strategy = "residual"

    # diff_residual_gamma: 动态残余修正强度
    diff_residual_gamma = _compute_dynamic_gamma(
        n_kv_heads=n_kv,
        gqa_ratio=gqa,
        d_head=d_head,
        model_size_category=category,
        quality_preference=quality_preference,
        max_seq_len_hint=max_seq_len_hint,
    )
    diff_residual_n_steps = 1  # 1 步残余足够

    # ── Tier 3: Adaptive N Scheduling (§10.2.3) ──
    #   自适应 N 根据 tile 能量分配编码步数:
    #   - 高能量 tile → n_steps_base + n_steps_extra 步
    #   - 低能量 tile → n_steps_base 步
    #   启用条件: 序列 ≥ 4096 或 quality 模式，且非 speed 模式
    enable_adaptive = False
    if max_seq_len_hint is not None:
        if max_seq_len_hint >= 4096 and quality_preference != "speed":
            enable_adaptive = True
    if quality_preference == "quality":
        enable_adaptive = True
    if quality_preference == "speed":
        enable_adaptive = False
    adaptive_n = enable_adaptive

    # energy_threshold_factor: 高/低能量分界线 = factor × mean(tile_energy)
    #   低 factor → 更多 tile 被标记为高能量 → 更多 tile 获得额外步数
    #   d_head 小 → 能量变化更剧烈 → 使用更低的 factor (捕获更多 tile)
    if d_head <= 64:
        energy_threshold_factor = 0.30
    elif d_head <= 96:
        energy_threshold_factor = 0.50
    elif d_head <= 128:
        energy_threshold_factor = 0.70
    else:
        energy_threshold_factor = 0.90

    if quality_preference == "quality":
        energy_threshold_factor = max(0.15, energy_threshold_factor - 0.10)
    elif quality_preference == "speed":
        energy_threshold_factor = min(0.95, energy_threshold_factor + 0.15)

    n_upper_bound = max(n_steps_k, n_steps_v) + 4

    # ── Tier 3.5: Per-layer adaptive step allocation (§8.1.6) ──
    #   Shallow layers (0-25%): reduced steps — basic semantic detection,
    #     can tolerate higher quantization noise.
    #   Middle layers (25-75%): baseline — causal information flow.
    #   Deep layers (75-100%): increased steps — fine-grained positional
    #     and semantic details, quantization error here propagates to
    #     later tokens in auto-regressive generation.
    #   Allocating more steps to deep layers rebalances total bit budget
    #   without increasing storage: shallow savings fund deep investment.
    layer_step_map: Dict[int, tuple] = {}
    if n_layers >= 4:
        # Layer bands (proportional to model depth, not hardcoded)
        n_shallow = max(2, n_layers // 4)       # e.g. 0–3 for 16-layer
        n_middle = max(2, n_layers // 2)        # e.g. 4–11 for 16-layer
        # deep = n_layers - n_shallow - n_middle  e.g. 12–15

        for lyr in range(n_layers):
            if lyr < n_shallow:
                # Shallow: save 1 step on K, 1 step on V
                lyr_nk = max(1, n_steps_k - 1)
                lyr_nv = max(2, n_steps_v - 1)
            elif lyr < n_shallow + n_middle:
                # Middle: baseline
                lyr_nk = n_steps_k
                lyr_nv = n_steps_v
            else:
                # Deep: invest 1 extra step on K, 1 extra step on V
                lyr_nk = n_steps_k + 1
                lyr_nv = min(10, n_steps_v + 1)  # cap at 10 to bound storage
            layer_step_map[lyr] = (lyr_nk, lyr_nv)

    # ── Tier 4: Runtime ──
    incremental_buffer_size = 4
    base_dtype = "fp16"
    verbose = False

    # ── Generate config ──
    cfg = DSKVCacheConfig(
        n_steps=max(n_steps_k, n_steps_v),
        n_steps_k=n_steps_k,
        n_steps_v=n_steps_v,
        tile_size=tile_size,
        beta=beta,
        use_noise_shaping=use_noise_shaping,
        proj_rank=proj_rank,
        proj_beta=proj_beta,
        adaptive_eta=adaptive_eta,
        order2_gamma=order2_gamma,
        order2_c1=order2_c1,
        order2_c2=order2_c2,
        cross_token_group=cross_token_group,
        v_orthogonal_transform=v_orthogonal_transform,
        use_differential=use_differential,
        diff_strategy=diff_strategy,
        diff_residual_gamma=diff_residual_gamma,
        diff_residual_n_steps=diff_residual_n_steps,
        adaptive_n=adaptive_n,
        n_upper_bound=n_upper_bound,
        energy_threshold_factor=energy_threshold_factor,
        layer_step_map=layer_step_map,
        incremental_buffer_size=incremental_buffer_size,
        base_dtype=base_dtype,
        verbose=verbose,
    )

    return cfg


# ────────────────────────────────────────────────────────────────────────────
# Quick Calibration (§8.5 — 可选的真机校准)
# ────────────────────────────────────────────────────────────────────────────

def quick_calibrate(
    model_path: str,
    cfg: DSKVCacheConfig,
    *,
    calib_tokens: int = 256,
    target_snr_db: float = 25.0,
    target_cos_sim: float = 0.985,
) -> Tuple[bool, Dict[str, Any]]:
    """运行快速校准: 用少量 token 前向传播, 测量 SNR / CosSim。

    Returns
    -------
    (passed, metrics_dict)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rina.ds_kv_cache import encode_kv_cache, decode_kvcache_store

    _logger.info(f"  ⏳ 快速校准: 加载模型 (fp16) ...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
    except Exception as e:
        _logger.warning(f"  无法加载模型进行校准: {e}")
        return False, {"error": str(e)}

    # 生成 calib_tokens 的提示
    lorem = (
        "The transformer architecture processes sequences in parallel using "
        "self-attention mechanisms. Multi-head attention allows the model to "
        "jointly attend to information from different representation subspaces. "
        "Position-wise feed-forward networks then transform each position "
        "independently. "
    )
    ids = tokenizer(lorem, add_special_tokens=False)["input_ids"]
    repeated = (ids * ((calib_tokens // len(ids)) + 2))[:calib_tokens]
    prompt = tokenizer.decode(repeated, skip_special_tokens=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    _logger.info(f"  ⏳ 前向传播 ({inputs['input_ids'].shape[1]} tokens) ...")
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], use_cache=True)
    fwd_time = time.perf_counter() - t0
    _logger.info(f"  前向传播完成 ({fwd_time:.1f}s)")

    # 取中间层和最后一层 测量编码质量
    n_layers_model = model.config.num_hidden_layers
    test_layers = [0, n_layers_model // 2, n_layers_model - 1]

    k_snr_all, v_snr_all = [], []
    k_cos_all, v_cos_all = [], []
    total_mem = 0
    total_fp16 = 0

    for layer_idx in test_layers:
        k_cache, v_cache = outputs.past_key_values[layer_idx]
        K = k_cache[0]  # (n_kv_heads, T, d_head)
        V = v_cache[0]
        n_kv, T, d_head = K.shape

        for h in range(n_kv):
            k_h = K[h].float()
            v_h = V[h].float()

            k_store, v_store = encode_kv_cache(k_h, v_h, cfg)
            k_hat = decode_kvcache_store(k_store, cfg.tile_size, cfg.use_differential)
            v_hat = decode_kvcache_store(v_store, cfg.tile_size, cfg.use_differential)

            # K SNR
            sig_k = (k_h ** 2).mean().item()
            noise_k = ((k_h - k_hat) ** 2).mean().item()
            k_snr_all.append(10 * math.log10(sig_k / (noise_k + 1e-12)))
            # V SNR
            sig_v = (v_h ** 2).mean().item()
            noise_v = ((v_h - v_hat) ** 2).mean().item()
            v_snr_all.append(10 * math.log10(sig_v / (noise_v + 1e-12)))

            # CosSim
            k_cos = torch.nn.functional.cosine_similarity(
                k_hat.flatten().unsqueeze(0), k_h.flatten().unsqueeze(0)
            ).item()
            v_cos = torch.nn.functional.cosine_similarity(
                v_hat.flatten().unsqueeze(0), v_h.flatten().unsqueeze(0)
            ).item()
            k_cos_all.append(k_cos)
            v_cos_all.append(v_cos)

            total_mem += k_store.memory_bytes + v_store.memory_bytes
            total_fp16 += T * d_head * 2 * 2  # K + V fp16

    avg_k_snr = sum(k_snr_all) / len(k_snr_all)
    avg_v_snr = sum(v_snr_all) / len(v_snr_all)
    avg_k_cos = sum(k_cos_all) / len(k_cos_all)
    avg_v_cos = sum(v_cos_all) / len(v_cos_all)
    comp_ratio = total_fp16 / (total_mem + 1e-12)

    passed = (
        avg_k_snr >= target_snr_db * 0.8
        and avg_v_snr >= target_snr_db * 0.8
        and avg_k_cos >= target_cos_sim
        and avg_v_cos >= target_cos_sim
    )

    metrics = {
        "K_SNR_dB": round(avg_k_snr, 2),
        "V_SNR_dB": round(avg_v_snr, 2),
        "K_CosSim": round(avg_k_cos, 4),
        "V_CosSim": round(avg_v_cos, 4),
        "Compression": round(comp_ratio, 2),
        "DS_Mem_KB": round(total_mem / 1024, 2),
        "passed": passed,
        "fwd_time_s": round(fwd_time, 2),
    }

    _logger.info(
        f"  校准结果: K SNR={avg_k_snr:.1f} dB, V SNR={avg_v_snr:.1f} dB, "
        f"K CosSim={avg_k_cos:.4f}, V CosSim={avg_v_cos:.4f}, "
        f"压缩率={comp_ratio:.1f}x {'✅' if passed else '⚠️'}"
    )

    return passed, metrics


# ────────────────────────────────────────────────────────────────────────────
# Auto-tune: 如果不达标, 自动调整 n_steps
# ────────────────────────────────────────────────────────────────────────────

def auto_tune_if_needed(
    model_path: str,
    cfg: DSKVCacheConfig,
    metrics: Dict[str, Any],
    *,
    max_iterations: int = 3,
) -> DSKVCacheConfig:
    """如果校准不达标, 自动增加 n_steps 重试。

    Returns
    -------
    (possibly_tuned) DSKVCacheConfig
    """
    if metrics.get("passed", True):
        return cfg

    _logger.info("  🔧 自动调优: 逐步增加 n_steps 以提高保真度 ...")
    current_cfg = cfg
    for i in range(max_iterations):
        new_nk = current_cfg.get_n_steps_k() + 1
        new_nv = current_cfg.get_n_steps_v() + 1

        tuned_cfg = DSKVCacheConfig(
            n_steps=max(new_nk, new_nv),
            n_steps_k=new_nk,
            n_steps_v=new_nv,
            tile_size=current_cfg.tile_size,
            beta=current_cfg.beta,
            use_noise_shaping=current_cfg.use_noise_shaping,
            proj_rank=current_cfg.proj_rank,
            proj_beta=current_cfg.proj_beta,
            adaptive_eta=current_cfg.adaptive_eta,
            order2_gamma=current_cfg.order2_gamma,
            order2_c1=current_cfg.order2_c1,
            order2_c2=current_cfg.order2_c2,
            cross_token_group=current_cfg.cross_token_group,
            v_orthogonal_transform=current_cfg.v_orthogonal_transform,
            use_differential=current_cfg.use_differential,
            diff_strategy=current_cfg.diff_strategy,
            diff_residual_gamma=current_cfg.diff_residual_gamma,
            diff_residual_n_steps=current_cfg.diff_residual_n_steps,
            adaptive_n=current_cfg.adaptive_n,
            n_upper_bound=max(new_nk, new_nv) + 4,
            energy_threshold_factor=current_cfg.energy_threshold_factor,
            layer_step_map=current_cfg.layer_step_map,
            incremental_buffer_size=current_cfg.incremental_buffer_size,
            base_dtype=current_cfg.base_dtype,
            verbose=current_cfg.verbose,
        )

        _logger.info(
            f"    Iter {i+1}: n_steps_k={new_nk}, n_steps_v={new_nv}"
        )
        passed, new_metrics = quick_calibrate(model_path, tuned_cfg)
        if passed:
            _logger.info("  ✅ 自动调优达标!")
            return tuned_cfg
        current_cfg = tuned_cfg

    _logger.warning(
        f"  ⚠️ {max_iterations} 轮调优后仍未完全达标, 返回最后的配置"
    )
    return current_cfg


# ────────────────────────────────────────────────────────────────────────────
# Display
# ────────────────────────────────────────────────────────────────────────────

def print_report(
    gpu_info: Dict[str, Any],
    model_info: Dict[str, Any],
    cfg: DSKVCacheConfig,
    metrics: Optional[Dict[str, Any]] = None,
):
    """打印完整诊断报告。"""
    print("\n" + "═" * 70)
    print("  DS-KVCache 自动适配报告")
    print("═" * 70)

    # ── 硬件 ──
    print("\n  ── 硬件检测 ──")
    print(f"    GPU:        {gpu_info['name']}")
    print(f"    VRAM:       {gpu_info['vram_gb']:.1f} GB ({gpu_info['vram_mb']} MB)")
    if gpu_info["is_nvidia"]:
        cc = gpu_info["compute_capability"]
        print(f"    Compute:    SM {cc[0]}.{cc[1]}")
    print(f"    Device:     {gpu_info['recommended_device']}")

    # ── 模型 ──
    print("\n  ── 模型检测 ──")
    print(f"    路径:       {model_info['path']}")
    print(f"    类型:       {model_info['model_type']}")
    print(f"    层数:       {model_info['num_layers']}")
    print(f"    Q-Heads:    {model_info['num_q_heads']}")
    print(f"    KV-Heads:   {model_info['num_kv_heads']}")
    print(f"    GQA 比率:   {model_info['gqa_ratio']}x")
    print(f"    d_head:     {model_info['d_head']}")
    print(f"    hidden:     {model_info['hidden_size']}")
    print(f"    规模分类:   {model_info['model_size_category']}")

    # ── 推荐配置 ──
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
    print(f"    device             = {cfg.device}")

    # ── 校准结果 ──
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


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

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

    # ── 1. 硬件探测 ──
    gpu_info = detect_gpu_info()

    # 覆盖设备偏好
    if args.device_preference:
        gpu_info["recommended_device"] = args.device_preference

    # ── 2. 模型探测 ──
    model_info = detect_model_info(args.model)

    if not args.json:
        _logger.info("🔍 检测硬件 ...")
        _logger.info(f"   GPU: {gpu_info['name']} ({gpu_info['vram_gb']:.1f} GB)")
        _logger.info("🔍 检测模型架构 ...")
        _logger.info(
            f"   {model_info['model_type']}: {model_info['num_layers']} layers, "
            f"{model_info['num_q_heads']}Q/{model_info['num_kv_heads']}KV, "
            f"d_head={model_info['d_head']}, "
            f"GQA={model_info['gqa_ratio']}x, "
            f"规模={model_info['model_size_category']}"
        )

    # ── 2b. Protected layers ──
    if args.protected_layers is not None:
        # Explicit user-provided list: parse "0,15" → {0, 15}
        protected_layers = {int(x.strip()) for x in args.protected_layers.split(",") if x.strip()}
    else:
        # Auto: first + last layers (most critical for error propagation)
        num_layers = model_info["num_layers"]
        if num_layers > 0:
            protected_layers = {0, num_layers - 1}
        else:
            protected_layers = set()

    # ── 3. 生成配置 ──
    cfg = generate_optimal_config(
        gpu_info, model_info,
        quality_preference=args.quality,
        max_seq_len_hint=args.seq_len_hint,
    )
    cfg.protected_layers = protected_layers

    if not args.json:
        _logger.info("⚙️  生成推荐配置 ...")

    # ── 4. 可选校准 ──
    metrics = None
    if args.calibrate:
        passed, metrics = quick_calibrate(args.model, cfg)
        if args.auto_tune and not passed:
            cfg = auto_tune_if_needed(args.model, cfg, metrics)

    # ── 5. 输出 ──
    if args.json:
        output = {
            "hardware": {
                "gpu_name": gpu_info["name"],
                "vram_gb": gpu_info["vram_gb"],
                "compute_capability": list(gpu_info["compute_capability"]),
                "device": gpu_info["recommended_device"],
            },
            "model": {
                "model_type": model_info["model_type"],
                "num_layers": model_info["num_layers"],
                "num_q_heads": model_info["num_q_heads"],
                "num_kv_heads": model_info["num_kv_heads"],
                "d_head": model_info["d_head"],
                "gqa_ratio": model_info["gqa_ratio"],
                "hidden_size": model_info["hidden_size"],
                "model_size_category": model_info["model_size_category"],
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
        print_report(gpu_info, model_info, cfg, metrics)

        if args.output:
            output = {
                "hardware": {
                    "gpu_name": gpu_info["name"],
                    "vram_gb": gpu_info["vram_gb"],
                    "device": gpu_info["recommended_device"],
                },
                "model": {
                    "model_type": model_info["model_type"],
                    "num_layers": model_info["num_layers"],
                    "gqa_ratio": model_info["gqa_ratio"],
                    "d_head": model_info["d_head"],
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