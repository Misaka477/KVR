"""Batch n_steps sweep — high quality config with DCT.

Shows subprocess output in real-time (no silent blocking).
"""
import subprocess
import sys
import time
import json

N_STEPS_VALUES = [8, 10, 12, 16]
MAX_TOKENS = 50
PYTHON = sys.executable
SCRIPT = "scripts/evaluation/eval_generation_fidelity.py"

results = {}
t_total = time.time()

for n in N_STEPS_VALUES:
    label = f"n={n}"
    json_out = f"eval_high_n{n}_v2.json"
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"[{time.strftime('%H:%M:%S')}] RUN n_steps={n} → {json_out}")
    print(f"{'='*60}")

    # Let subprocess output go to parent's stderr/stdout (visible)
    rc = subprocess.call(
        [PYTHON, SCRIPT,
         "--quality", "high",
         "--n-steps", str(n),
         "--measure-kv",
         "--max-tokens", str(MAX_TOKENS),
         "--json-output", json_out],
    )
    elapsed = time.time() - t0

    if rc != 0:
        print(f"[{time.strftime('%H:%M:%S')}] FAIL n={n}  rc={rc}  {elapsed:.0f}s")
        results[label] = {"error": f"exit code {rc}", "time_s": round(elapsed)}
        continue

    print(f"[{time.strftime('%H:%M:%S')}] DONE n={n}  {elapsed:.0f}s")

    try:
        with open(json_out, "r", encoding="utf-8") as f:
            data = json.load(f)
        rs = data.get("route_results", {})
        ds_keys = [k for k in rs if k != "native"]
        if ds_keys:
            s = rs[ds_keys[0]]
            kv = s.get("kv_fidelity", {})
            results[label] = {
                "n_steps": n,
                "char_match": s.get("avg_char_match"),
                "prefix_match": s.get("avg_prefix_match"),
                "repetition_score": s.get("avg_repetition_score"),
                "kv_cos_sim_k": kv.get("avg_cos_sim_k"),
                "kv_cos_sim_v": kv.get("avg_cos_sim_v"),
                "time_s": round(elapsed),
            }
            print(f"  char={s.get('avg_char_match')} pref={s.get('avg_prefix_match')} "
                  f"rep={s.get('avg_repetition_score')} "
                  f"K_CosSim={kv.get('avg_cos_sim_k','?')} V_CosSim={kv.get('avg_cos_sim_v','?')}")
        else:
            results[label] = {"error": "No DS route results", "time_s": round(elapsed)}
    except Exception as e:
        results[label] = {"error": str(e), "time_s": round(elapsed)}

# Summary
summary_path = "eval_nsteps_sweep_v2_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print("N-STEP SWEEP RESULTS")
print(f"Total: {time.time()-t_total:.0f}s")
print(f"{'Config':<10s} {'char':>8s} {'pref':>8s} {'rep':>8s} {'K_CosSim':>10s} {'V_CosSim':>10s} {'Time':>8s}")
print("-" * 68)
for n in N_STEPS_VALUES:
    r = results.get(f"n={n}", {})
    if "error" in r:
        print(f"  n={n:<3d}  ERROR: {r['error'][:60]}")
    else:
        print(f"  n={n:<3d}  {str(r.get('char_match','?')):>8s} {str(r.get('prefix_match','?')):>8s} "
              f"{str(r.get('repetition_score','?')):>8s} "
              f"{str(r.get('kv_cos_sim_k','?')):>10s} {str(r.get('kv_cos_sim_v','?')):>10s} "
              f"{str(r.get('time_s','?')):>8s}")
print(f"\n→ {summary_path}")
