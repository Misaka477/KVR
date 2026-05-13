"""Debug: kernel's cos values vs Python cos values."""
import sys, torch, triton, triton.language as tl
sys.path.insert(0, ".")
from modules.kvr_hook import KVRHook
from transformers import AutoModelForCausalLM, AutoTokenizer

@triton.jit
def cos_check_kernel(cos_ptr, out_ptr, N: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr):
    tid = tl.program_id(0)
    if tid >= N: return
    c = tl.load(cos_ptr + tid * D + tl.arange(0, HALF))
    tl.store(out_ptr + tid * HALF + tl.arange(0, HALF), c)

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval(); dev = model.device

inp = tok("The capital of France is Paris. " * 20, return_tensors="pt", truncation=True, max_length=128).to(dev)
kvrh = KVRHook(model, window_size=64, top_k=128, device=dev)
kvrh.prefill(inp["input_ids"])
ret = kvrh.retrievals[0]
d, half = ret.d_head, ret.d_head // 2
ns = ret.n_stored

# Kernel cos output
out = torch.zeros(ns, half, device=dev)
cos_check_kernel[(ns,)](ret.cos, out, N=ns, D=d, HALF=half)

# Python cos
ref = ret.cos[:ns, :half]

diff = (out - ref).abs().max().item()
print(f"Cos diff: {diff:.8f}  PASS={diff < 1e-6}")
