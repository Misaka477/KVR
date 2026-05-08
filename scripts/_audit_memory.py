"""Audit DSKVCacheStore memory breakdown for L=64,128,256,512,1024."""
import torch, math, logging
from rina.config import DSKVCacheConfig
from rina.ds_kv_cache import encode_kv_cache

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

cfg = DSKVCacheConfig(
    n_steps=5, tile_size=16, beta=0.15, proj_beta=0.15,
    use_noise_shaping=True, proj_rank=5,
    verbose=True,
)

for L in [64, 128, 256, 512, 1024]:
    d_head = 64
    k = torch.randn(min(L, 1020), d_head)
    v = torch.randn(min(L, 1020), d_head)
    actual = k.shape[0]

    k_store, v_store = encode_kv_cache(k, v, cfg)

    print(f"\n{'='*60}")
    print(f"  L={actual} (target {L})")
    print(f"{'='*60}")

    for name, store in [("K", k_store), ("V", v_store)]:
        original_bytes = actual * d_head * 4  # fp32
        mem = store.memory_bytes
        print(f"\n  [{name}]  memory_bytes = {mem} B = {mem/1024:.1f} KB")
        print(f"    Original fp32      = {original_bytes} B = {original_bytes/1024:.1f} KB")
        print(f"    Compression ratio  = {original_bytes/(mem+1e-12):.1f}x")

        # Detailed breakdown
        print(f"    --- Components ---")
        for attr in ["bases", "bases_b", "alphas", "alphas_b", "raw_buffer"]:
            t = getattr(store, attr, None)
            if t is not None:
                el_bytes = t.element_size()
                nelem = t.numel()
                raw_bytes = el_bytes * nelem
                if attr in ("bases", "bases_b"):
                    # packed int32 → 32 bits of value
                    effective_bytes = (nelem * 32) // 8
                    print(f"    {attr}:  shape={tuple(t.shape)}  dtype={t.dtype}  "
                          f"raw={raw_bytes} B  packed_bits={effective_bytes} B  "
                          f"({effective_bytes/1024:.2f} KB)")
                else:
                    print(f"    {attr}:  shape={tuple(t.shape)}  dtype={t.dtype}  "
                          f"raw={raw_bytes} B  ({raw_bytes/1024:.2f} KB)")
            else:
                print(f"    {attr}:  None")

        # Check full_k_hat
        fkh = store.full_k_hat
        if fkh is not None:
            fkh_bytes = fkh.element_size() * fkh.numel()
            print(f"    full_k_hat:  shape={tuple(fkh.shape)}  dtype={fkh.dtype}  "
                  f"ram={fkh_bytes} B = {fkh_bytes/1024:.1f} KB  ⚠️ NOT COUNTED in memory_bytes")
        else:
            print(f"    full_k_hat:  None")

        # Check svd_shaper
        if store.svd_shaper is not None:
            print(f"    svd_shaper:  keys={list(store.svd_shaper.keys())[:5]}...")
        else:
            print(f"    svd_shaper:  None")