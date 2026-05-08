"""Quick 4K long-sequence DS-KVCache benchmark with heterogeneous config"""
import sys, time, math
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from rina.config import DSKVCacheConfig
from rina.ds_kv_cache import encode_kv_cache, decode_kvcache_store

from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = 'D:/Software_Development/Project/models/Llama-3.2-1B'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True,
)
model.eval()

# Generate 4096-token sequence
seq_len = 4096
text = "The quick brown fox jumps over the lazy dog. " * 500  # >4K tokens
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len).to(device)
input_ids = inputs["input_ids"]
actual_len = input_ids.shape[1]
print(f"Sequence length: {actual_len}")

# Run forward pass to get K/V
with torch.no_grad():
    outputs = model(input_ids=input_ids, use_cache=True)

past = outputs.past_key_values

# Test on one layer (layer 0)
k_cache, v_cache = past[0]  # DynamicCache: (batch, n_kv_heads, seq_len, d_head)
K = k_cache[0]  # (8, 4096, 64)
V = v_cache[0]

cfg = DSKVCacheConfig(
    n_steps_k=4,
    n_steps_v=6,
    tile_size=16,
    beta=0.15,
    use_noise_shaping=False,  # too expensive for 4K
    proj_beta=0.3,
    adaptive_eta=False,
    adaptive_n=False,          # DISABLE adaptive N — kills compression with default n_upper_bound=10
    use_differential=True,
    diff_strategy="residual",
    diff_residual_gamma=0.25,
    diff_residual_n_steps=1,
    v_orthogonal_transform=True,
    base_dtype="fp16",
    verbose=False,
    # ── Phase 1: 方案 B — 二阶 Σ-Δ ──
    order2_gamma=0.3,
    order2_c1=1.0,
    order2_c2=0.5,
    # ── Phase 1: 方案 A — 跨 token 联合编码 ──
    cross_token_group=4,
)

n_kv = K.shape[0]
print(f"Layer 0: {n_kv} KV heads, d_head=64")
print(f"Config: K={cfg.get_n_steps_k()} steps, V={cfg.get_n_steps_v()} steps, "
      f"v_ortho={cfg.v_orthogonal_transform}, diff_gamma={cfg.diff_residual_gamma}")

results = []
total_mem = 0
total_time = 0

for h in range(n_kv):
    k_h = K[h].float()
    v_h = V[h].float()

    t0 = time.perf_counter()
    k_store, v_store = encode_kv_cache(k_h, v_h, cfg)
    elapsed = time.perf_counter() - t0
    total_time += elapsed

    k_hat = decode_kvcache_store(k_store, cfg.tile_size, cfg.use_differential)
    v_hat = decode_kvcache_store(v_store, cfg.tile_size, cfg.use_differential)

    k_cos = F.cosine_similarity(k_hat.flatten().unsqueeze(0), k_h.flatten().unsqueeze(0)).item()
    v_cos = F.cosine_similarity(v_hat.flatten().unsqueeze(0), v_h.flatten().unsqueeze(0)).item()

    fp16_bytes = k_h.numel() * 2 + v_h.numel() * 2
    ds_bytes = k_store.memory_bytes + v_store.memory_bytes
    comp = fp16_bytes / (ds_bytes + 1e-12)
    total_mem += ds_bytes

    results.append((k_cos, v_cos, comp, k_store.n_tiles * 2))
    print(f"  Head {h}: K CosSim={k_cos:.5f}, V CosSim={v_cos:.5f}, "
          f"Compress={comp:.1f}x, "
          f"K_mem={k_store.memory_bytes}B, V_mem={v_store.memory_bytes}B, "
          f"tiles={k_store.n_tiles}+{v_store.n_tiles}")

fp16_total = K.numel() * 2 + V.numel() * 2
avg_k_cos = sum(r[0] for r in results) / n_kv
avg_v_cos = sum(r[1] for r in results) / n_kv
avg_comp = sum(r[2] for r in results) / n_kv

print(f"\n{'='*60}")
print(f"4K SEQUENCE SUMMARY (Layer 0, {seq_len} tokens)")
print(f"{'='*60}")
print(f"  Avg K CosSim:      {avg_k_cos:.5f}")
print(f"  Avg V CosSim:      {avg_v_cos:.5f}")
print(f"  Avg Compression:   {avg_comp:.1f}x")
print(f"  FP16 memory:       {fp16_total/1024:.1f} KB")
print(f"  DS memory:         {total_mem/1024:.1f} KB")
print(f"  Encode time:       {total_time:.2f}s ({total_time/n_kv*1000:.1f} ms/head)")
print(f"  Est. full model:   {fp16_total*16/1024:.1f} KB → {total_mem*16/1024:.1f} KB "
      f"({avg_comp:.1f}x)")