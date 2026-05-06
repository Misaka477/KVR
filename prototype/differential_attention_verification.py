"""
Differential Attention-Level Verification (§8.2 / §10.5 follow-on)
==================================================================
Extends the differential noise cancellation validation from vector-level
(cf. tests/test_residual_pursuit.py::TestDifferentialCancellation) to
**attention-output level**: encodes K and V with two complementary paths,
then measures whether the differential combination reduces attention
output MSE vs single-path encoding.

Results feed directly into R.I.N.A Whitepaper §10.5 and the gap item
in §13.6 ("差分对消模型级验证").
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.residual_pursuit import (  # noqa: E402
    encode_matrix,
    decode_from_bases,
    differential_encode_decode,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
TILE_SIZE = 16


def set_seed(s: int = SEED):
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------------------
# Attention-level differential verification
# ---------------------------------------------------------------------------

def run_differential_attention_verification(
    seq_len: int = 256,
    d_head: int = 128,
    n_heads: int = 4,
    n_steps: int = 5,
    beta: float = 0.0,
    proj_beta: float = 0.0,
    proj_rank: int = 0,
) -> Dict[str, float]:
    """Verify differential cancellation at attention-output level.

    For each attention head:
      1. Encode K matrix (seq_len × d_head) with differential_encode_decode
      2. Encode V matrix (seq_len × d_head) with differential_encode_decode
      3. Compute attention output with:
         a)  Single-path encoding (Path A only)
         b)  Differential encoding  (A+B averaged)
         c)  Full-precision (ground truth)
      4. Report MSE & CosSim for both vs ground truth

    Returns
    -------
    results : dict
        Keys: single_mse, single_cos, diff_mse, diff_cos,
              mse_reduction_pct, cos_improvement, nrr_mean
    """
    set_seed()
    device = torch.device(DEVICE)

    # ---- Generate synthetic attention tensors ----
    Q = torch.randn(n_heads, seq_len, d_head, device=device) * 0.5
    K = torch.randn(n_heads, seq_len, d_head, device=device) * 0.5
    V = torch.randn(n_heads, seq_len, d_head, device=device) * 0.5

    scale = d_head ** 0.5

    # ---- Full-precision attention (ground truth) ----
    attn_scores_fp = torch.matmul(Q, K.transpose(-2, -1)) / scale
    attn_weights_fp = F.softmax(attn_scores_fp, dim=-1)
    attn_output_fp = torch.matmul(attn_weights_fp, V)

    # ---- Build projection matrix (if noise-shaping requested) ----
    proj_matrix = None
    if proj_rank > 0:
        # Use flatten tiles from K across all heads for PCA
        M = TILE_SIZE * TILE_SIZE
        # Concatenate all K tiles: flatten each head's K → pad → unfold
        all_tiles = []
        for h in range(n_heads):
            K_h = K[h]  # (seq_len, d_head)
            k_padded, _ = _pad_to_tile_multiple(K_h, TILE_SIZE)
            patches = F.unfold(
                k_padded.unsqueeze(0).unsqueeze(0),
                kernel_size=TILE_SIZE, stride=TILE_SIZE,
            ).squeeze(0).t()  # (n_tiles, M)
            all_tiles.append(patches)
        all_tiles_cat = torch.cat(all_tiles, dim=0)
        centered = all_tiles_cat - all_tiles_cat.mean(dim=0, keepdim=True)
        k = min(proj_rank, min(centered.shape) - 1)
        if k >= 1:
            _, _, V_pca = torch.pca_lowrank(centered.float(), q=k)
            proj_matrix = V_pca @ V_pca.T  # (M, M)

    # ---- Encode + reconstruct K and V per head ----
    K_single = torch.zeros_like(K)
    K_diff = torch.zeros_like(K)
    V_single = torch.zeros_like(V)
    V_diff = torch.zeros_like(V)

    nrr_list = []

    for h in range(n_heads):
        # --- K matrix ---
        K_h = K[h].cpu()  # (seq_len, d_head)
        # Differential encode
        K_diff_rec, diag_k = differential_encode_decode(
            K_h, n_steps=n_steps, tile_size=TILE_SIZE, beta=beta,
            proj_matrix=proj_matrix, proj_beta=proj_beta,
        )
        # Single encode (Path A only)
        bases_a, alphas_a, shape_a = encode_matrix(
            K_h, n_steps=n_steps, tile_size=TILE_SIZE, beta=beta,
            proj_matrix=proj_matrix, proj_beta=proj_beta,
        )
        K_a = decode_from_bases(bases_a, alphas_a, shape_a, tile_size=TILE_SIZE)

        K_diff[h] = K_diff_rec.to(device)
        K_single[h] = K_a.to(device)
        nrr_list.append(diag_k["nrr"])

        # --- V matrix ---
        V_h = V[h].cpu()
        V_diff_rec, diag_v = differential_encode_decode(
            V_h, n_steps=n_steps, tile_size=TILE_SIZE, beta=beta,
            proj_matrix=proj_matrix, proj_beta=proj_beta,
        )
        bases_va, alphas_va, shape_va = encode_matrix(
            V_h, n_steps=n_steps, tile_size=TILE_SIZE, beta=beta,
            proj_matrix=proj_matrix, proj_beta=proj_beta,
        )
        V_a = decode_from_bases(bases_va, alphas_va, shape_va, tile_size=TILE_SIZE)

        V_diff[h] = V_diff_rec.to(device)
        V_single[h] = V_a.to(device)
        nrr_list.append(diag_v["nrr"])

    # ---- Attention with single-path encoding ----
    attn_scores_s = torch.matmul(Q, K_single.transpose(-2, -1)) / scale
    attn_weights_s = F.softmax(attn_scores_s, dim=-1)
    attn_output_s = torch.matmul(attn_weights_s, V_single)

    # ---- Attention with differential encoding ----
    attn_scores_d = torch.matmul(Q, K_diff.transpose(-2, -1)) / scale
    attn_weights_d = F.softmax(attn_scores_d, dim=-1)
    attn_output_d = torch.matmul(attn_weights_d, V_diff)

    # ---- Metrics ----
    single_mse = F.mse_loss(attn_output_s, attn_output_fp).item()
    diff_mse = F.mse_loss(attn_output_d, attn_output_fp).item()

    single_cos = F.cosine_similarity(
        attn_output_s.reshape(-1).unsqueeze(0),
        attn_output_fp.reshape(-1).unsqueeze(0),
    ).item()
    diff_cos = F.cosine_similarity(
        attn_output_d.reshape(-1).unsqueeze(0),
        attn_output_fp.reshape(-1).unsqueeze(0),
    ).item()

    mse_reduction = (1 - diff_mse / max(single_mse, 1e-12)) * 100
    cos_improvement = diff_cos - single_cos
    nrr_mean = sum(nrr_list) / len(nrr_list) if nrr_list else 0.0

    return {
        "single_mse": single_mse,
        "single_cos": single_cos,
        "diff_mse": diff_mse,
        "diff_cos": diff_cos,
        "mse_reduction_pct": mse_reduction,
        "cos_improvement": cos_improvement,
        "nrr_mean": nrr_mean,
    }


# ---------------------------------------------------------------------------
# Attention Weight-level metrics (direct KV comparison)
# ---------------------------------------------------------------------------

def run_differential_kv_quality_verification(
    seq_len: int = 128,
    d_head: int = 128,
    n_heads: int = 2,
    n_steps: int = 5,
    beta: float = 0.0,
) -> Dict[str, float]:
    """Measure per-element K/V reconstruction improvement from differential.

    This is the bridge between vector-level (§10.5) and attention-level
    verification: it quantifies how much differential encoding improves
    the raw K and V element-wise before they enter the attention mechanism.
    """
    set_seed()
    device = torch.device(DEVICE)

    K = torch.randn(n_heads, seq_len, d_head, device=device) * 0.5
    V = torch.randn(n_heads, seq_len, d_head, device=device) * 0.5

    single_mse_k_list = []
    diff_mse_k_list = []
    single_mse_v_list = []
    diff_mse_v_list = []

    for h in range(n_heads):
        K_h = K[h].cpu()
        V_h = V[h].cpu()

        # K: differential
        K_diff, d_k = differential_encode_decode(
            K_h, n_steps=n_steps, tile_size=TILE_SIZE, beta=beta,
        )
        # K: single
        b_k, a_k, s_k = encode_matrix(K_h, n_steps=n_steps, tile_size=TILE_SIZE, beta=beta)
        K_single = decode_from_bases(b_k, a_k, s_k, tile_size=TILE_SIZE)

        single_mse_k_list.append(F.mse_loss(K_single, K_h).item())
        diff_mse_k_list.append(F.mse_loss(K_diff, K_h).item())

        # V: differential
        V_diff, d_v = differential_encode_decode(
            V_h, n_steps=n_steps, tile_size=TILE_SIZE, beta=beta,
        )
        # V: single
        b_v, a_v, s_v = encode_matrix(V_h, n_steps=n_steps, tile_size=TILE_SIZE, beta=beta)
        V_single = decode_from_bases(b_v, a_v, s_v, tile_size=TILE_SIZE)

        single_mse_v_list.append(F.mse_loss(V_single, V_h).item())
        diff_mse_v_list.append(F.mse_loss(V_diff, V_h).item())

    avg_single_mse_k = sum(single_mse_k_list) / len(single_mse_k_list)
    avg_diff_mse_k = sum(diff_mse_k_list) / len(diff_mse_k_list)
    avg_single_mse_v = sum(single_mse_v_list) / len(single_mse_v_list)
    avg_diff_mse_v = sum(diff_mse_v_list) / len(diff_mse_v_list)

    return {
        "k_single_mse": avg_single_mse_k,
        "k_diff_mse": avg_diff_mse_k,
        "k_mse_reduction_pct": (1 - avg_diff_mse_k / max(avg_single_mse_k, 1e-12)) * 100,
        "v_single_mse": avg_single_mse_v,
        "v_diff_mse": avg_diff_mse_v,
        "v_mse_reduction_pct": (1 - avg_diff_mse_v / max(avg_single_mse_v, 1e-12)) * 100,
    }


# ---------------------------------------------------------------------------
# N-step sweep on attention output
# ---------------------------------------------------------------------------

def run_n_step_sweep_attention(
    n_steps_list=[1, 3, 5, 7],
    seq_len: int = 128,
    d_head: int = 64,
    n_heads: int = 2,
) -> None:
    """Sweep N steps and report single vs differential attention output MSE."""
    print(f"\n{'='*70}")
    print(f"Attention-Level N-Step Sweep (Differential vs Single)")
    print(f"  seq_len={seq_len}, d_head={d_head}, n_heads={n_heads}")
    print(f"{'='*70}")
    print(f"{'N':<6} {'Single MSE':<14} {'Diff MSE':<14} {'MSE Reduction':<16} {'Diff Cos Δ':<14}")
    print("-" * 70)

    for n in n_steps_list:
        r = run_differential_attention_verification(
            seq_len=seq_len, d_head=d_head, n_heads=n_heads,
            n_steps=n, beta=0.0,
        )
        print(
            f"{n:<6} {r['single_mse']:<14.8f} {r['diff_mse']:<14.8f} "
            f"{r['mse_reduction_pct']:<15.2f}% {r['cos_improvement']:<+14.6f}"
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _pad_to_tile_multiple(w, tile_size):
    rows, cols = w.shape[-2], w.shape[-1]
    pad_r = (tile_size - rows % tile_size) % tile_size
    pad_c = (tile_size - cols % tile_size) % tile_size
    if pad_r == 0 and pad_c == 0:
        return w, (0, 0)
    return F.pad(w, (0, pad_c, 0, pad_r)), (pad_r, pad_c)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Differential Cancellation — Attention-Level Verification")
    print("=" * 70)

    # ---- Experiment 1: Core attention-level test ----
    print("\n--- Experiment 1: Core Attention-Level Differential ---")
    r = run_differential_attention_verification(
        seq_len=256, d_head=128, n_heads=4, n_steps=5, beta=0.0,
    )
    print(f"  Single-path:     MSE={r['single_mse']:.8f}  CosSim={r['single_cos']:.6f}")
    print(f"  Differential:    MSE={r['diff_mse']:.8f}  CosSim={r['diff_cos']:.6f}")
    print(f"  MSE Reduction:   {r['mse_reduction_pct']:+.2f}%")
    print(f"  CosSim Δ:        {r['cos_improvement']:+.6f}")
    print(f"  Mean NRR (K+V):  {r['nrr_mean']:+.4f}")

    # ---- Experiment 2: Direct K/V element-wise improvement ----
    print("\n--- Experiment 2: K/V Element-Wise Quality ---")
    kv = run_differential_kv_quality_verification(
        seq_len=128, d_head=128, n_heads=4, n_steps=5, beta=0.0,
    )
    print(f"  K: single MSE={kv['k_single_mse']:.8f}  diff MSE={kv['k_diff_mse']:.8f}  "
          f"reduction={kv['k_mse_reduction_pct']:+.2f}%")
    print(f"  V: single MSE={kv['v_single_mse']:.8f}  diff MSE={kv['v_diff_mse']:.8f}  "
          f"reduction={kv['v_mse_reduction_pct']:+.2f}%")

    # ---- Experiment 3: N-step sweep ----
    run_n_step_sweep_attention(n_steps_list=[1, 3, 5, 7, 10])

    # ---- Experiment 4: With momentum ----
    print("\n--- Experiment 4: Differential + Momentum (β=0.15) ---")
    r_mom = run_differential_attention_verification(
        seq_len=256, d_head=128, n_heads=4, n_steps=5, beta=0.15,
    )
    print(f"  Single-path:     MSE={r_mom['single_mse']:.8f}  CosSim={r_mom['single_cos']:.6f}")
    print(f"  Differential:    MSE={r_mom['diff_mse']:.8f}  CosSim={r_mom['diff_cos']:.6f}")
    print(f"  MSE Reduction:   {r_mom['mse_reduction_pct']:+.2f}%")
    print(f"  CosSim Δ:        {r_mom['cos_improvement']:+.6f}")

    # ---- Experiment 5: With noise shaping ----
    print("\n--- Experiment 5: Differential + Noise-Shaping (proj_rank=8, η=0.5) ---")
    r_ns = run_differential_attention_verification(
        seq_len=128, d_head=64, n_heads=2, n_steps=5, beta=0.0,
        proj_rank=8, proj_beta=0.5,
    )
    print(f"  Single-path:     MSE={r_ns['single_mse']:.8f}  CosSim={r_ns['single_cos']:.6f}")
    print(f"  Differential:    MSE={r_ns['diff_mse']:.8f}  CosSim={r_ns['diff_cos']:.6f}")
    print(f"  MSE Reduction:   {r_ns['mse_reduction_pct']:+.2f}%")
    print(f"  CosSim Δ:        {r_ns['cos_improvement']:+.6f}")

    print("\n" + "=" * 70)
    print("Verification complete. Results ready for whitepaper §10.5 update.")
    print("=" * 70)