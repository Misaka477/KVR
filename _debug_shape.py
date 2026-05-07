"""Debug script: trace DCT encode/decode shape flow."""
import torch, sys
sys.path.insert(0, '.')
from rina.config import DSKVCacheConfig
from rina.ds_kv_cache import encode_kv_cache, decode_kvcache_store

cfg = DSKVCacheConfig()
cfg.transform_mode = 'dct'
cfg.use_fwht = False
cfg.verbose = False

N, d = 198, 64
k = torch.randn(N, d, dtype=torch.float16)
v = torch.randn(N, d, dtype=torch.float16)

print(f'Input k shape: {k.shape}')
k_store, v_store = encode_kv_cache(k, v, cfg)
print(f'  k_store.orig_shape: {k_store.orig_shape}')
print(f'  k_store._original_mat_shape: {getattr(k_store, "_original_mat_shape", None)}')
print(f'  k_store.transform_pad_rows: {k_store.transform_pad_rows}')
print(f'  k_store.transform_mode: {k_store.transform_mode}')
print(f'  k_store.bases.shape: {k_store.bases.shape}')
print(f'  k_store.alphas.shape: {k_store.alphas.shape}')

k_hat = decode_kvcache_store(k_store, cfg.tile_size, cfg.use_differential)
print(f'k_hat shape: {k_hat.shape}')
print(f'MSE: {torch.nn.functional.mse_loss(k_hat.float(), k.float()).item():.6f}')