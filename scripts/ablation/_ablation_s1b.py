"""Quick ablation: C1 baseline vs A+B (cross_token_group=4 + order2_gamma=0.3)"""
import sys; sys.path.insert(0, '.')
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from rina.config import DSKVCacheConfig
from rina.ds_kv_cache import encode_kv_cache, decode_kvcache_store

print('=== ABLATION: C1 baseline vs A+B ===')
model = AutoModelForCausalLM.from_pretrained(
    'D:/Software_Development/Project/models/Llama-3.2-1B',
    torch_dtype=torch.float16, device_map='auto'
)
tok = AutoTokenizer.from_pretrained('D:/Software_Development/Project/models/Llama-3.2-1B')
if tok.pad_token is None: tok.pad_token = tok.eos_token
model.eval()

text = ('The transformer architecture has revolutionized natural language processing ' * 35)[:4000*6]
inp = tok(text, return_tensors='pt', truncation=True, max_length=4096).to(model.device)
print(f'Seq len: {inp.input_ids.shape[1]}')

with torch.no_grad():
    out = model(**inp, use_cache=True)
k0, v0 = out.past_key_values[0]
k_h = k0[0, 0].float()
v_h = v0[0, 0].float()

# C1 baseline
cfg_c1 = DSKVCacheConfig(
    n_steps_k=3, n_steps_v=5, tile_size=16, beta=0.15,
    use_noise_shaping=True, proj_rank=8, proj_beta=0.3, adaptive_eta=True,
    order2_gamma=0.0, cross_token_group=1,
    use_differential=True, diff_strategy='residual',
    diff_residual_gamma=0.25, diff_residual_n_steps=1,
    v_orthogonal_transform=True,
)
ks1, vs1 = encode_kv_cache(k_h, v_h, cfg_c1)
kh1 = decode_kvcache_store(ks1, 16, True)
vh1 = decode_kvcache_store(vs1, 16, True)

# A+B
cfg_ab = DSKVCacheConfig(
    n_steps_k=4, n_steps_v=5, tile_size=16, beta=0.10,
    use_noise_shaping=True, proj_rank=8, proj_beta=0.3, adaptive_eta=True,
    order2_gamma=0.3, cross_token_group=4,
    use_differential=True, diff_strategy='residual',
    diff_residual_gamma=0.15, diff_residual_n_steps=1,
    v_orthogonal_transform=True,
)
ks2, vs2 = encode_kv_cache(k_h, v_h, cfg_ab)
kh2 = decode_kvcache_store(ks2, 16, True)
vh2 = decode_kvcache_store(vs2, 16, True)

def metrics(kh, kref, label):
    c = F.cosine_similarity(kh.flatten().unsqueeze(0), kref.flatten().unsqueeze(0)).item()
    m = F.mse_loss(kh, kref).item()
    sig = (kref ** 2).mean().item()
    s = 10.0 * torch.log10(torch.tensor(max(sig / max(m, 1e-12), 1e-12))).item()
    print(f'  {label}: CosSim={c:.4f}  SNR={s:.1f}dB  MSE={m:.6f}')

print()
metrics(kh1, k_h, 'C1 K ')
metrics(vh1, v_h, 'C1 V ')
metrics(kh2, k_h, 'A+B K')
metrics(vh2, v_h, 'A+B V')

c1_bytes = ks1.memory_bytes + vs1.memory_bytes
ab_bytes = ks2.memory_bytes + vs2.memory_bytes
print(f'\nC1: {c1_bytes} bytes ({c1_bytes/1024:.1f} KB)')
print(f'A+B: {ab_bytes} bytes ({ab_bytes/1024:.1f} KB)')
dc_k = F.cosine_similarity(kh2.flatten().unsqueeze(0), k_h.flatten().unsqueeze(0)).item() - F.cosine_similarity(kh1.flatten().unsqueeze(0), k_h.flatten().unsqueeze(0)).item()
dc_v = F.cosine_similarity(vh2.flatten().unsqueeze(0), v_h.flatten().unsqueeze(0)).item() - F.cosine_similarity(vh1.flatten().unsqueeze(0), v_h.flatten().unsqueeze(0)).item()
print(f'Δ CosSim K: {dc_k:+.4f}  Δ CosSim V: {dc_v:+.4f}')
