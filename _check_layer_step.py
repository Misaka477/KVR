"""Verify per-layer step allocation flows through to encode_kv_cache."""
import sys
sys.path.insert(0, ".")

import torch
from rina.config import DSKVCacheConfig
from rina.ds_kv_cache import encode_kv_cache

# Build a config with explicit layer_step_map
cfg = DSKVCacheConfig(
    n_steps_k=4, n_steps_v=8,
    tile_size=16, beta=0.10,
    use_noise_shaping=False,
    layer_step_map={
        0: (3, 4),   # shallow: reduced
        5: (4, 5),   # middle: baseline
        12: (5, 6),  # deep: boosted
    },
)

# Simulate K/V for one head
T, d_head = 32, 64
k = torch.randn(T, d_head)
v = torch.randn(T, d_head)

for layer_idx in [0, 5, 12, 99]:
    layer_cfg = cfg.get_layer_config(layer_idx)
    try:
        k_store, v_store = encode_kv_cache(k, v, layer_cfg)
        # Infer actual n_steps from alphas shape
        k_steps_actual = k_store.alphas.shape[0] if k_store.alphas is not None else 0
        v_steps_actual = v_store.alphas.shape[0] if v_store.alphas is not None else 0
        k_pad_val = getattr(k_store, '_cross_token_pad', 'N/A')
        v_pad_val = getattr(v_store, '_cross_token_pad', 'N/A')
        print(f"Layer {layer_idx:>3}: n_steps_k={layer_cfg.get_n_steps_k()}, n_steps_v={layer_cfg.get_n_steps_v()}  "
              f"→ k_steps_actual={k_steps_actual}, v_steps_actual={v_steps_actual}, "
              f"k_pad={k_pad_val}, v_pad={v_pad_val}, "
              f"comp={T * d_head * 2 / (k_store.memory_bytes + v_store.memory_bytes):.1f}x")
    except Exception as e:
        print(f"Layer {layer_idx:>3}: ERROR — {e}")

print("\n[OK] Per-layer step verification complete.")