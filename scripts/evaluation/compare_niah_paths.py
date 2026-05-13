"""Compare Python vs Triton NIAH at 1024 ctx."""
import sys, torch
sys.path.insert(0, ".")
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules.kvr_hook import KVRHook

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval(); dev = model.device

HS = "The grass is green. "
ND = "The secret password is KILO42. "
Q = "I just told you a secret password. The password is"
hs_ids = tok(HS, add_special_tokens=False)["input_ids"]
nd_ids = tok(ND, add_special_tokens=False)["input_ids"]
q_ids = tok(Q, add_special_tokens=False)["input_ids"]

for mode, desc in [(False, "Python"), (True, "Triton")]:
    from modules import kvr_retrieval as kr
    import modules.kvr_triton as kt
    orig = kr.RetrievalIndex.compute_all_scores

    if mode:
        # Triton (already enabled)
        pass
    else:
        # Force Python path
        def py_scores(self, q):
            d = self.d_head; nkv = self.n_kv; n_s = self.n_stored
            g = q.shape[0] // nkv; scores = torch.empty(n_s, nkv, device=self.device)
            q_avg = q.view(nkv, g, d).mean(dim=1)
            aidx = torch.arange(n_s, device=self.device)
            for kvh in range(nkv):
                kp = self._deq_k(kvh); kp2 = self._rotary(kp, aidx)
                scores[:, kvh] = (q_avg[kvh] @ kp2.T) / (d ** 0.5)
            return scores
        kr.RetrievalIndex.compute_all_scores = py_scores

    nr = max(0, int(1024 * 0.5 / len(hs_ids)))
    seq = []; ri = 0
    while len(seq) < 1024:
        seq.extend(nd_ids if ri == nr else hs_ids); ri += 1
    seq = seq[:1024] + q_ids
    t = torch.tensor([seq], device=dev)

    for run in range(3):
        kvrh = KVRHook(model, window_size=64, top_k=128, device=dev)
        kvrh.prefill(t); kvrh.register()
        gen_ids = []
        for step in range(12):
            cur = t if step == 0 else torch.cat([t] + gen_ids, dim=1)
            out = model(cur, use_cache=False)
            nid = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_ids.append(nid); kvrh._step += 1; kvrh._context_len += 1
        kvrh.remove()
        txt = tok.decode(torch.cat(gen_ids, dim=1)[0].cpu(), skip_special_tokens=True)
        p = "PASS" if "KILO42" in txt.upper() else "FAIL"
        print(f"{desc} run {run+1}: {p} -- {txt[:50]}")

    kr.RetrievalIndex.compute_all_scores = orig
