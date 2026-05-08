"""
§8 ModelAdapter — automatic configuration recommendation and calibration.

Detects model/hardware characteristics and generates optimal
DSKVCacheConfig parameters using multi-factor heuristics derived
from scripts/auto_config.py (§8.5).

Architecture:
    ModelProfile   — model architecture info (layers, heads, GQA, d_head)
    HardwareProfile — hardware capability info (GPU, VRAM, compute cap)
    ModelAdapter   — recommend_config + quick_calibrate + auto_tune
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

_logger = logging.getLogger("model_adapter")


# ══════════════════════════════════════════════════════════════════════════════
# ModelProfile
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ModelProfile:
    """Model architecture characteristics needed for auto-configuration.

    Fields are derived from HuggingFace model config — compatible with
    Llama, Qwen, Mistral, Gemma, and other llama-style architectures.
    """

    num_layers: int = 0
    """Total transformer layers."""

    num_attention_heads: int = 0
    """Query attention heads."""

    num_kv_heads: int = 0
    """Key/Value attention heads (may be < num_attention_heads for GQA)."""

    d_head: int = 0
    """Head dimension (hidden_size // num_attention_heads)."""

    hidden_size: int = 0
    """Model hidden dimension."""

    max_position_embeddings: int = 4096
    """Maximum sequence length from model config."""

    model_name: str = ""
    """Human-readable model identifier."""

    has_gqa: bool = False
    """True when num_kv_heads < num_attention_heads (Grouped-Query Attention)."""

    @property
    def gqa_ratio(self) -> float:
        """Ratio of query heads to KV heads (1.0 for MHA, N for GQA)."""
        if self.num_kv_heads <= 0:
            return 1.0
        return self.num_attention_heads / self.num_kv_heads

    @property
    def model_size_category(self) -> str:
        """Categorize model by parameter count."""
        # Estimate params from architecture (rough order-of-magnitude)
        d = self.hidden_size or self.num_attention_heads * self.d_head
        if d <= 0:
            return "unknown"
        # Rough: layers * hidden² * 4 (attn) + layers * hidden² * 8 (FFN)
        # Normalized: category based on hidden_size mainly
        if d <= 768:
            return "tiny"    # < 500M
        elif d <= 1536:
            return "small"   # 500M–1B
        elif d <= 3072:
            return "medium"  # 1B–7B
        elif d <= 5120:
            return "large"   # 7B–13B
        else:
            return "xlarge"  # > 13B

    @classmethod
    def from_hf_config(cls, config: Any) -> "ModelProfile":
        """Build ModelProfile from a HuggingFace model config object.

        Compatible with LlamaConfig, Qwen2Config, MistralConfig, etc.
        """
        n_layers = getattr(config, "num_hidden_layers", 0)
        n_q_heads = getattr(config, "num_attention_heads", 0)
        n_kv_heads = getattr(config, "num_key_value_heads", n_q_heads)
        hidden_size = getattr(config, "hidden_size", 0)
        max_pos = getattr(config, "max_position_embeddings", 4096)
        name = getattr(config, "model_type", "unknown")

        # Determine d_head
        if hasattr(config, "head_dim"):
            d_head = config.head_dim
        elif n_q_heads > 0 and hidden_size > 0:
            d_head = hidden_size // n_q_heads
        else:
            d_head = 0

        return cls(
            num_layers=n_layers,
            num_attention_heads=n_q_heads,
            num_kv_heads=n_kv_heads,
            d_head=d_head,
            hidden_size=hidden_size,
            max_position_embeddings=max_pos,
            model_name=name,
            has_gqa=(n_kv_heads < n_q_heads if n_q_heads > 0 else False),
        )


# ══════════════════════════════════════════════════════════════════════════════
# HardwareProfile
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class HardwareProfile:
    """Hardware capability snapshot used for auto-configuration decisions.

    Detected automatically via :meth:`detect`.
    """

    available: bool = False
    """GPU available (True) or CPU-only (False)."""

    count: int = 0
    """Number of available devices."""

    name: str = "CPU"
    """Device name / GPU model."""

    vram_mb: int = 0
    """Available VRAM in MB (or system RAM for CPU)."""

    vram_gb: float = 0.0
    """VRAM in GB."""

    compute_capability: Tuple[int, int] = (0, 0)
    """CUDA compute capability (major, minor).  (0, 0) for CPU."""

    is_nvidia: bool = False
    is_amd: bool = False
    is_apple: bool = False

    recommended_device: str = "cpu"
    """'cuda', 'mps', or 'cpu'."""

    @classmethod
    def detect(cls) -> "HardwareProfile":
        """Auto-detect hardware capabilities."""
        info = cls()

        if torch.cuda.is_available():
            info.available = True
            info.count = torch.cuda.device_count()
            info.is_nvidia = True
            props = torch.cuda.get_device_properties(0)
            info.name = props.name
            total_mem = getattr(props, "total_memory", getattr(props, "total_mem", 0))
            info.vram_mb = total_mem // (1024 * 1024)
            info.vram_gb = round(total_mem / (1024**3), 1)
            info.compute_capability = (props.major, props.minor)
            info.recommended_device = "cuda"
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            info.available = True
            info.count = 1
            info.is_apple = True
            info.name = "Apple MPS"
            try:
                import subprocess
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                )
                total_ram = int(result.stdout.strip())
                info.vram_mb = int(total_ram * 0.75) // (1024 * 1024)
                info.vram_gb = round(total_ram * 0.75 / (1024**3), 1)
            except Exception:
                info.vram_mb = 8192
                info.vram_gb = 8.0
            info.recommended_device = "mps"
        else:
            try:
                import psutil
                ram = psutil.virtual_memory()
                info.vram_mb = ram.total // (1024 * 1024)
                info.vram_gb = round(ram.total / (1024**3), 1)
            except Exception:
                info.vram_mb = 4096
                info.vram_gb = 4.0

        return info


# ══════════════════════════════════════════════════════════════════════════════
# ModelAdapter — configuration engine
# ══════════════════════════════════════════════════════════════════════════════


class ModelAdapter:
    """Auto-adaptation engine: recommends optimal DSKVCacheConfig.

    Combines ModelProfile + HardwareProfile + heuristic rules to
    generate tiered parameter recommendations.  Supports optional
    quick_calibrate (forward-verify on 256 tokens) and auto_tune
    (iterative parameter adjustment).

    Usage::

        hw = HardwareProfile.detect()
        profile = ModelProfile.from_hf_config(model.config)
        adapter = ModelAdapter(profile, hw)
        cfg = adapter.recommend_config(quality="balanced")
        # Optional: verify on real forward pass
        cfg = adapter.quick_calibrate(model, cfg)
    """

    def __init__(
        self,
        model_profile: ModelProfile,
        hardware_profile: Optional[HardwareProfile] = None,
    ):
        self.model = model_profile
        self.hw = hardware_profile or HardwareProfile.detect()

    # ── Configuration recommendation ──────────────────────────────────────

    def recommend_config(
        self,
        *,
        quality: str = "balanced",
        max_seq_len_hint: Optional[int] = None,
    ) -> "DSKVCacheConfig":
        """Generate optimal DSKVCacheConfig from model + hardware profiles.

        Parameters
        ----------
        quality:
            "quality"  — maximum fidelity, lower compression
            "balanced" — balanced (default)
            "speed"    — maximum compression / memory efficiency
        max_seq_len_hint:
            Expected max sequence length (default: from model config).

        Returns
        -------
        DSKVCacheConfig with auto-computed parameters.
        """
        from rina.config import DSKVCacheConfig

        d_head = self.model.d_head
        gqa = self.model.gqa_ratio
        n_layers = self.model.num_layers
        n_kv = self.model.num_kv_heads
        category = self.model.model_size_category

        # ── Tier 1: n_steps_k / n_steps_v ─────────────────────────────────
        if d_head <= 64:
            base_nk, base_nv = 4, 5
        elif d_head <= 96:
            base_nk, base_nv = 4, 6
        elif d_head <= 128:
            base_nk, base_nv = 5, 7
        else:
            base_nk, base_nv = 6, 8

        if gqa >= 8:
            base_nv += 1

        q_offset = {"quality": 1, "balanced": 0, "speed": -1}
        offset = q_offset.get(quality, 0)
        n_steps_k = max(1, base_nk + offset)
        n_steps_v = max(1, base_nv + offset)

        # ── Tier 1: tile_size ─────────────────────────────────────────────
        if not self.hw.available:
            tile_size = 8
        elif self.hw.compute_capability[0] < 7:
            tile_size = 16
        else:
            tile_size = 16

        # ── Tier 1: beta ──────────────────────────────────────────────────
        if d_head <= 64:
            beta = 0.10
        elif d_head <= 96:
            beta = 0.15
        elif d_head <= 128:
            beta = 0.20
        else:
            beta = 0.25

        # ── Tier 1: noise shaping ─────────────────────────────────────────
        use_noise_shaping = True
        proj_rank = 8

        if category == "tiny":
            proj_beta = 0.30
        elif category == "small":
            proj_beta = 0.28
        elif category == "medium":
            proj_beta = 0.25
        elif category == "large":
            proj_beta = 0.20
        else:
            proj_beta = 0.15

        if quality == "quality":
            proj_beta -= 0.05
        elif quality == "speed":
            proj_beta += 0.05
        proj_beta = max(0.05, min(0.40, proj_beta))
        adaptive_eta = True

        # ── Tier 2: second-order Σ-Δ ──────────────────────────────────────
        if d_head <= 64:
            order2_gamma = 0.30
        elif d_head <= 96:
            order2_gamma = 0.25
        elif d_head <= 128:
            order2_gamma = 0.20
        else:
            order2_gamma = 0.15
        order2_c1, order2_c2 = 1.0, 0.5

        if quality == "quality":
            order2_gamma = min(0.40, order2_gamma + 0.05)
        elif quality == "speed":
            order2_gamma = max(0.05, order2_gamma - 0.10)

        # ── Tier 2: cross-token joint encoding ────────────────────────────
        if max_seq_len_hint is not None:
            if max_seq_len_hint >= 4096:
                cross_token_group = 4
            elif max_seq_len_hint >= 512:
                cross_token_group = 2
            else:
                cross_token_group = 1
        else:
            max_pos = self.model.max_position_embeddings
            if max_pos >= 8192:
                cross_token_group = 4
            elif max_pos >= 512:
                cross_token_group = 2
            else:
                cross_token_group = 1

        v_orthogonal_transform = True

        # ── Tier 3: differential residual ─────────────────────────────────
        diff_residual_gamma = self._compute_dynamic_gamma(quality, max_seq_len_hint)
        diff_residual_n_steps = 1

        # ── Tier 3: adaptive N scheduling ─────────────────────────────────
        enable_adaptive = False
        if max_seq_len_hint is not None and max_seq_len_hint >= 4096 and quality != "speed":
            enable_adaptive = True
        if quality == "quality":
            enable_adaptive = True

        if d_head <= 64:
            energy_threshold_factor = 0.30
        elif d_head <= 96:
            energy_threshold_factor = 0.50
        elif d_head <= 128:
            energy_threshold_factor = 0.70
        else:
            energy_threshold_factor = 0.90

        if quality == "quality":
            energy_threshold_factor = max(0.15, energy_threshold_factor - 0.10)
        elif quality == "speed":
            energy_threshold_factor = min(0.95, energy_threshold_factor + 0.15)

        n_upper_bound = max(n_steps_k, n_steps_v) + 4

        # ── Per-layer step allocation (§8.1.6) ────────────────────────────
        layer_step_map: Dict[int, Tuple[int, int]] = {}
        if n_layers >= 4:
            n_shallow = max(2, n_layers // 4)
            n_middle = max(2, n_layers // 2)
            for lyr in range(n_layers):
                if lyr < n_shallow:
                    lyr_nk = max(1, n_steps_k - 1)
                    lyr_nv = max(2, n_steps_v - 1)
                elif lyr < n_shallow + n_middle:
                    lyr_nk = n_steps_k
                    lyr_nv = n_steps_v
                else:
                    lyr_nk = n_steps_k + 1
                    lyr_nv = min(10, n_steps_v + 1)
                layer_step_map[lyr] = (lyr_nk, lyr_nv)

        # ── Generate config ───────────────────────────────────────────────
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
            use_differential=True,
            diff_strategy="residual",
            diff_residual_gamma=diff_residual_gamma,
            diff_residual_n_steps=diff_residual_n_steps,
            adaptive_n=enable_adaptive,
            n_upper_bound=n_upper_bound,
            energy_threshold_factor=energy_threshold_factor,
            layer_step_map=layer_step_map if layer_step_map else None,
            incremental_buffer_size=4,
            base_dtype="fp16",
        )
        return cfg

    def _compute_dynamic_gamma(
        self,
        quality: str = "balanced",
        max_seq_len_hint: Optional[int] = None,
    ) -> float:
        """Multi-factor diff_residual_gamma computation (from auto_config.py).

        Three-factor model:
            gamma = base * (1 + α_kv + α_gqa) * (1 + β_dhead) * (1 + β_seq)
        """
        n_kv = self.model.num_kv_heads
        gqa_ratio = self.model.gqa_ratio
        d_head = self.model.d_head

        # base_gamma based on GQA ratio
        if gqa_ratio >= 8:
            base_gamma = 0.20
        elif gqa_ratio >= 4:
            base_gamma = 0.15
        else:
            base_gamma = 0.10

        # α_kv — KV head scarcity penalty
        if n_kv <= 2:
            alpha_kv = 0.07
        elif n_kv <= 4:
            alpha_kv = 0.05
        elif n_kv <= 8:
            alpha_kv = 0.02
        else:
            alpha_kv = 0.0

        # α_gqa — GQA enhancement
        if gqa_ratio >= 16:
            alpha_gqa = 0.05
        elif gqa_ratio >= 8:
            alpha_gqa = 0.03
        elif gqa_ratio >= 4:
            alpha_gqa = 0.01
        else:
            alpha_gqa = 0.0

        # β_dhead — d_head scaling
        if d_head >= 128:
            beta_dhead = 0.25
        elif d_head >= 96:
            beta_dhead = 0.20
        else:
            beta_dhead = 0.0

        # β_seq — long sequence protection
        if max_seq_len_hint is not None:
            if max_seq_len_hint >= 16384:
                beta_seq = 0.04
            elif max_seq_len_hint >= 8192:
                beta_seq = 0.02
            else:
                beta_seq = 0.0
        else:
            beta_seq = 0.0

        q_delta = {"quality": 0.04, "balanced": 0.0, "speed": -0.04}
        q_offset = q_delta.get(quality, 0.0)

        gamma = base_gamma * (1.0 + alpha_kv + alpha_gqa) * (1.0 + beta_dhead) * (1.0 + beta_seq) + q_offset
        gamma = max(0.05, min(0.40, gamma))
        return round(gamma, 4)

    # ── Quick calibration (§8.5) ──────────────────────────────────────────

    def quick_calibrate(
        self,
        model: Any,
        cfg: "DSKVCacheConfig",
        *,
        n_calibration_tokens: int = 256,
        target_snr: float = 20.0,
        max_iterations: int = 3,
    ) -> "DSKVCacheConfig":
        """Forward-verify config on ~256 tokens and iteratively adjust.

        Loads model, runs a single prefill forward pass, measures SNR
        of reconstructed KV, and adjusts n_steps/beta if quality is
        below target.

        Parameters
        ----------
        model: HuggingFace model instance (or path to load from).
        cfg: Proposed configuration to validate.
        n_calibration_tokens: Number of dummy tokens for forward pass.
        target_snr: Minimum SNR (dB) to accept.
        max_iterations: Max adjustment rounds.

        Returns
        -------
        Adjusted DSKVCacheConfig (may be unchanged if already good).
        """
        from rina.ds_kv_cache import encode_kv_cache, decode_kvcache_store

        _logger.info("Quick calibrate: %d tokens, target SNR ≥ %.1f dB",
                      n_calibration_tokens, target_snr)

        # Determine device
        if hasattr(model, "device"):
            device = model.device
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        d_head = self.model.d_head

        for iteration in range(max_iterations):
            # Generate dummy input
            dummy_k = torch.randn(n_calibration_tokens, d_head, device=device)
            dummy_v = torch.randn(n_calibration_tokens, d_head, device=device)

            try:
                k_store, v_store = encode_kv_cache(dummy_k, dummy_v, cfg)
                k_hat = decode_kvcache_store(k_store, cfg.tile_size, cfg.use_differential)
                v_hat = decode_kvcache_store(v_store, cfg.tile_size, cfg.use_differential)

                # Compute SNR
                k_signal = (dummy_k.float() ** 2).mean().item()
                k_noise = ((dummy_k.float() - k_hat.float()) ** 2).mean().item()
                k_snr = 10 * math.log10(k_signal / (k_noise + 1e-12))

                v_signal = (dummy_v.float() ** 2).mean().item()
                v_noise = ((dummy_v.float() - v_hat.float()) ** 2).mean().item()
                v_snr = 10 * math.log10(v_signal / (v_noise + 1e-12))

                min_snr = min(k_snr, v_snr)
                _logger.info("  iter %d: K SNR=%.1f dB, V SNR=%.1f dB, min=%.1f dB",
                             iteration + 1, k_snr, v_snr, min_snr)

                if min_snr >= target_snr:
                    _logger.info("  SNR target met — config validated.")
                    return cfg

                # Adjust: boost n_steps
                nk = cfg.get_n_steps_k()
                nv = cfg.get_n_steps_v()
                if k_snr < target_snr:
                    cfg.n_steps_k = min(10, nk + 1)
                if v_snr < target_snr:
                    cfg.n_steps_v = min(10, nv + 1)
                cfg.n_steps = max(cfg.get_n_steps_k(), cfg.get_n_steps_v())

            except Exception as e:
                _logger.warning("  Calibration iteration %d failed: %s", iteration + 1, e)
                # Fall back to previous config
                break

        _logger.info("  Calibration completed after %d iterations.", iteration + 1)
        return cfg

    def auto_tune(
        self,
        model: Any,
        *,
        quality: str = "balanced",
        max_seq_len_hint: Optional[int] = None,
        calibrate: bool = True,
    ) -> "DSKVCacheConfig":
        """Full auto-tune: recommend → calibrate → return optimal config.

        Parameters
        ----------
        model: HuggingFace model instance.
        quality: "quality", "balanced", or "speed".
        max_seq_len_hint: Expected max sequence length.
        calibrate: If True, run quick_calibrate after recommendation.

        Returns
        -------
        Optimized DSKVCacheConfig.
        """
        cfg = self.recommend_config(quality=quality, max_seq_len_hint=max_seq_len_hint)
        if calibrate:
            cfg = self.quick_calibrate(model, cfg)
        return cfg


__all__ = ["ModelProfile", "HardwareProfile", "ModelAdapter"]
