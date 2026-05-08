"""Sprint 2 ablation: per-layer adaptive steps vs uniform baseline."""
import sys; sys.path.insert(0, '.')
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from rina.config import DSKVCacheConfig
from rina.ds_kv_cache import encode_kv_cache, decode_kvcache_store

print('=== SPRINT 2: Per-layer adaptive step allocation ===')
model = AutoModelForCausalLM.from_pretrained(
    'D:/Software_Development/Project/models/Llama-3.2-1B',
    torch_dtype=torch.float16, device_map='auto'
)
tok = AutoTokenizer.from_pretrained('D:/Software_Development/Project/models/Llama-3.2-1B')
if tok.pad_token is None: tok.pad_token = tok.eos_token
model.eval()
n_layers = model.config.num_hidden_layers

text = ('The transformer architecture has revolutionized natural language processing ' * 35)[:4000*6]
inp = tok(text, return_tensors='pt', truncation=True, max_length=4096).to(model.device)
print(f'Seq len: {inp.input_ids.shape[1]}, n_layers={n_layers}')

# ── Baseline: A+B uniform config ──
cfg_uniform = DSKVCacheConfig(
    n_steps_k=4, n_steps_v=5, tile_size=16, beta=0.10,
    use_noise_shaping=True, proj_rank=8, proj_beta=0.3, adaptive_eta=True,
    order2_gamma=0.3, cross_token_group=4,
    use_differential=True, diff_strategy='residual',
    diff_residual_gamma=0.15, diff_residual_n_steps=1,
    v_orthogonal_transform=True,
)

# ── Per-layer adaptive ──
# Shallow (0-5): 3/4, Mid (6-10): 4/5, Deep (11-15): 5/6
layer_step_map = {}
for l in range(n_layers):
    if l <= 5:
        layer_step_map[l] = (3, 4)
    elif l <= 10:
        layer_step_map[l] = (4, 5)
    else:
        layer_step_map[l] = (5, 6)

cfg_adaptive = DSKVCacheConfig(
    n_steps_k=4, n_steps_v=5, tile_size=16, beta=0.10,
    use_noise_shaping=True, proj_rank=8, proj_beta=0.3, adaptive_eta=True,
    order2_gamma=0.3, cross_token_group=4,
    use_differential=True, diff_strategy='residual',
    diff_residual_gamma=0.15, diff_residual_n_steps=1,
    v_orthogonal_transform=True,
    layer_step_map=layer_step_map,
)

# ── Forward pass to get K/V ──
with torch.no_grad():
    out = model(**inp, use_cache=True)

# Compare per-layer CosSim
print()
print(f"{'Layer':>6} {'Uni K':>8} {'Adp K':>8} {'Δ K':>8} {'Uni V':>8} {'Adp V':>8} {'Δ V':>8}")
print('-' * 70)

total_uni_cos_k, total_adp_cos_k = 0, 0
total_uni_cos_v, total_adp_cos_v = 0, 0
total_uni_bytes, total_adp_bytes = 0, 0

uni_layer_steps = {}
for l in range(n_layers):
    k_h = out.past_key_values[l][0][0, 0].float()
    v_h = out.past_key_values[l][1][0, 0].float()

    # Uniform encode
    ks_u, vs_u = encode_kv_cache(k_h, v_h, cfg_uniform)
    kh_u = decode_kvcache_store(ks_u, cfg_uniform.tile_size, True)
    vh_u = decode_kvcache_store(vs_u, cfg_uniform.tile_size, True)

    # Adaptive encode
    ks_a, vs_a = encode_kv_cache(k_h, v_h, cfg_adaptive.get_layer_config(l))
    kh_a = decode_kvcache_store(ks_a, cfg_adaptive.tile_size, True)
    vh_a = decode_kvcache_store(vs_a, cfg_adaptive.tile_size, True)

    c_uk = F.cosine_similarity(kh_u.flatten().unsqueeze(0), k_h.flatten().unsqueeze(0)).item()
    c_ak = F.cosine_similarity(kh_a.flatten().unsqueeze(0), k_h.flatten().unsqueeze(0)).item()
    c_uv = F.cosine_similarity(vh_u.flatten().unsqueeze(0), v_h.flatten().unsqueeze(0)).item()
    c_av = F.cosine_similarity(vh_a.flatten().unsqueeze(0), v_h.flatten().unsqueeze(0)).item()

    total_uni_cos_k += c_uk
    total_adp_cos_k += c_ak
    total_uni_cos_v += c_uv
    total_adp_cos_v += c_av

    total_uni_bytes += ks_u.memory_bytes + vs_u.memory_bytes
    total_adp_bytes += ks_a.memory_bytes + vs_a.memory_bytes

    dk = c_ak - c_uk
    dv = c_av - c_uv
    marker = ' **' if abs(dk) > 0.005 or abs(dv) > 0.005 else ''
    print(f'{l:>6} {c_uk:>8.4f} {c_ak:>8.4f} {dk:>+8.4f} {c_uv:>8.4f} {c_av:>8.4f} {dv:>+8.4f}{marker}')

print('-' * 70)
print(f"{'AVG':>6} {total_uni_cos_k/n_layers:>8.4f} {total_adp_cos_k/n_layers:>8.4f} "
      f"{(total_adp_cos_k-total_uni_cos_k)/n_layers:>+8.4f} "
      f"{total_uni_cos_v/n_layers:>8.4f} {total_adp_cos_v/n_layers:>8.4f} "
      f"{(total_adp_cos_v-total_uni_cos_v)/n_layers:>+8.4f}")

print(f'\nUniform bytes: {total_uni_bytes} ({total_uni_bytes/1024:.1f} KB)')
print(f'Adaptive bytes: {total_adp_bytes} ({total_adp_bytes/1024:.1f} KB)')
delta_pct = (total_adp_bytes - total_uni_bytes) / total_uni_bytes * 100
print(f'Storage delta: {delta_pct:+.1f}%')