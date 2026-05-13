"""Compare kernel vs manual per-token diff breakdown."""
import sys, torch
sys.path.insert(0, ".")
from modules.kvr_hook import KVRHook
from modules.kvr_triton import score_kernel as sk
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval(); dev = model.device

inp = tok("The capital of France is Paris. " * 20, return_tensors="pt", truncation=True, max_length=128).to(dev)
kvrh = KVRHook(model, window_size=64, top_k=128, device=dev)
kvrh.prefill(inp["input_ids"])
ret = kvrh.retrievals[0]
d, half, n_kv, ns = ret.d_head, ret.d_head // 2, ret.n_kv, ret.n_stored

q_avg = torch.randn(8, 64, device="cuda")

# Kernel scores
tri = torch.empty(ns, n_kv, device="cuda")
sk[(ns, n_kv)](q_avg, ret.k_packed, ret.k_scales, ret.cos, ret.sin, tri,
    N_KV=n_kv, G=1, D=d, HALF=half, N_STORED=ns)

# Manual scores (just kvh=0)
man = torch.zeros(ns, device="cuda")
for t in range(ns):
    k_pre = ret._deq_k(0)[t]
    c = ret.cos[t, :half]; s = ret.sin[t, :half]
    k0 = k_pre[:half]; k1 = k_pre[half:]
    k0r = k0 * c - k1 * s
    k1r = k0 * s + k1 * c
    man[t] = (torch.sum(q_avg[0, 0::2] * k0r) + torch.sum(q_avg[0, 1::2] * k1r)).item() / (d ** 0.5)

diffs = (man - tri[:, 0]).abs()
for t in range(ns):
    if diffs[t] > 0.1:
        print(f"Token {t}: diff={diffs[t]:.4f}  man={man[t]:.4f}  tri={tri[t,0].item():.4f}")


