"""Debug Triton score_kernel: step-by-step diff at small scale."""
import sys, torch
sys.path.insert(0, ".")
from modules.kvr_triton import score_kernel as sk
from modules.kvr_retrieval import _apply_rotary, _rotate_half

# Tiny case: n_kv=1, d=4, n_tok=1
n_kv, d, n_s, g = 1, 4, 1, 1
half = d // 2

# Create known values: K_int4 range [-8, 7], packed format
k_int4 = torch.tensor([[[-3, 7, -1, 5]]], device="cuda", dtype=torch.int8)
# Pack: (val+8) then interleave
ku = (k_int4 + 8).to(torch.uint8)  # (1, 1, 4)
kp = (ku[..., 0::2] << 4) | ku[..., 1::2]  # (1, 1, d//2) = (1, 1, 2) -> one token

# Scales: per-dim, shape (n_kv, d)
k_scales = torch.tensor([[0.5, 0.3, 0.7, 0.2]], device="cuda")  # (1, 4) for 1 KV head

# Q: simple
q = torch.randn(n_kv * g, d, device="cuda")

# Cos/sin tables (3 tokens)
pos = torch.arange(n_s, device="cuda").float()
freq = 10000 ** (-torch.arange(0, d, 2, device="cuda").float() / d)
ct = torch.cos(pos[:, None] * freq)
st = torch.sin(pos[:, None] * freq)

# --- Python: step by step ---
py_scores = torch.zeros(n_s, n_kv, device="cuda")
for tok in range(n_s):
    for kvh in range(n_kv):
        # Unpack
        ku_raw = k_int4[tok, kvh].float()  # [-3, 7, -1, 5]
        step = 2 * k_scales[kvh] / 16
        k_pre = ku_raw * step  # dequantized

        # RoPE
        c = ct[tok, :half]
        s = st[tok, :half]
        k0 = k_pre[:half]; k1 = k_pre[half:]
        k0r = k0 * c - k1 * s
        k1r = k0 * s + k1 * c
        k_post = torch.cat([k0r, k1r], dim=-1)

        # Score
        q0 = q[kvh * g, :half]
        q1 = q[kvh * g, half:]
        sc = (torch.sum(q0 * k0r) + torch.sum(q1 * k1r)).item() / (d ** 0.5)
        py_scores[tok, kvh] = sc

    print(f"Py token {tok}: k_pre={k_int4[tok, 0].tolist()}"
          f"  k_post={k_post[:4].tolist()}")
    print(f"  score={py_scores[tok, 0].item():.6f}")

# --- Triton ---
tri_scores = torch.empty(n_s, n_kv, device="cuda")
sk[(n_s, n_kv)](q, kp, k_scales, ct, st, tri_scores,
    N_KV=n_kv, G=g, D=d, HALF=half, N_STORED=n_s)

print(f"\nTriton scores: {tri_scores[:, 0].tolist()}")
print(f"Python scores: {py_scores[:, 0].tolist()}")
print(f"Max diff: {(py_scores - tri_scores).abs().max().item():.8f}")
