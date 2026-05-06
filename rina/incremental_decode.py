"""
§5 Incremental Decode Pipeline
==============================

Implements the ring-buffer incremental encoding scheme (§5.1) that avoids
re-encoding the entire K/V cache on every token step.

Algorithm
  Each new token's K/V is appended to a raw FP16 ring buffer.
  When the buffer reaches ``buffer_size``, the accumulated tokens are:
    1. Encoded via Σ-Δ RBP → 1-bit bases + alphas
    2. Concatenated to the existing DSKVCacheStore
    3. The raw buffer is cleared

  On attention lookup, the current K/V is reconstructed from the store.
  The full reconstruction is cached to avoid O(N_tokens) decode per step.

References
  Whitepaper §5.1
"""

from __future__ import annotations

import math
import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from rina.config import DSKVCacheConfig
from rina.ds_kv_cache import (
    DSKVCacheStore,
    encode_kv_cache,
    decode_kvcache_store,
)
from modules.residual_pursuit import pack_bases, unpack_bases

_logger = logging.getLogger(__name__)


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    """Resolve a human-readable dtype string to torch.dtype."""
    _map = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    key = dtype_str.lower().strip()
    if key in _map:
        return _map[key]
    try:
        return getattr(torch, dtype_str)
    except AttributeError:
        _logger.warning(f"Unknown dtype '{dtype_str}', falling back to float16")
        return torch.float16


def init_incremental_store(
    d_head: int,
    cfg: DSKVCacheConfig,
) -> DSKVCacheStore:
    """Create an empty DSKVCacheStore with pre-allocated raw buffer.

    Parameters
    ----------
    d_head:
        Key/value head dimension.
    cfg:
        Pipeline configuration.

    Returns
    -------
    Empty store with allocated raw buffer.
    """
    dtype = _resolve_dtype(cfg.base_dtype)
    buffer = torch.zeros(cfg.incremental_buffer_size, d_head, dtype=dtype)
    return DSKVCacheStore(
        n_tokens=0,
        tile_size=cfg.tile_size,
        n_tiles=0,
        raw_buffer=buffer,
        buffer_full=0,
    )


def incremental_encode_step(
    new_token_vec: torch.Tensor,
    store: DSKVCacheStore,
    cfg: DSKVCacheConfig,
    is_key: bool = True,
) -> DSKVCacheStore:
    """Append one new token vector to the store, encoding batch if full.

    Parameters
    ----------
    new_token_vec:
        (d_head,) key or value vector for the new token.  FP32 or FP16.
    store:
        Existing DSKVCacheStore for this head.
    cfg:
        Pipeline configuration.
    is_key:
        If True, this vector is a Key; used for noise-shaping state.

    Returns
    -------
    Updated store (mutated in-place, returned for convenience).
    """
    if store.raw_buffer is None:
        dtype = _resolve_dtype(cfg.base_dtype)
        store.raw_buffer = torch.zeros(
            cfg.incremental_buffer_size, new_token_vec.shape[0],
            dtype=dtype,
        )

    # Append to ring buffer
    idx = store.buffer_full
    store.raw_buffer[idx, :] = new_token_vec.to(store.raw_buffer.dtype)
    store.buffer_full += 1

    # If buffer is full, encode the accumulated tokens
    if store.buffer_full >= cfg.incremental_buffer_size:
        return _flush_buffer(store, cfg, is_key)

    return store


def incremental_encode_batch(
    new_token_matrix: torch.Tensor,
    store: DSKVCacheStore,
    cfg: DSKVCacheConfig,
    is_key: bool = True,
) -> DSKVCacheStore:
    """Append a batch of new token vectors, handling partial buffer fill.

    Parameters
    ----------
    new_token_matrix:
        (num_new_tokens, d_head) key or value matrix.
    store:
        Existing DSKVCacheStore.
    cfg:
        Pipeline configuration.
    is_key:
        If True, this is a Key; used for noise-shaping state.

    Returns
    -------
    Updated store.
    """
    n_new = new_token_matrix.shape[0]
    pos = 0

    while pos < n_new:
        # Try to fill remaining buffer slots
        free = store.raw_buffer.shape[0] - store.buffer_full
        chunk = min(free, n_new - pos)

        store.raw_buffer[store.buffer_full : store.buffer_full + chunk, :] = \
            new_token_matrix[pos : pos + chunk].to(store.raw_buffer.dtype)
        store.buffer_full += chunk
        pos += chunk

        if store.buffer_full >= cfg.incremental_buffer_size:
            store = _flush_buffer(store, cfg, is_key)

    return store


def finalize_store(
    store: DSKVCacheStore,
    cfg: DSKVCacheConfig,
    is_key: bool = True,
) -> DSKVCacheStore:
    """Flush any remaining tokens in the raw buffer and cache reconstruction.

    Call this at the end of a decode phase (e.g., all tokens cached).

    Parameters
    ----------
    store:
        DSKVCacheStore that may have partial tokens in the raw buffer.
    cfg:
        Pipeline configuration.
    is_key:
        If True, this is a Key; used for noise-shaping state.

    Returns
    -------
    Finalized store with full reconstruction cached.
    """
    if store.buffer_full > 0:
        store = _flush_buffer(store, cfg, is_key)

    # Cache full reconstruction
    store.full_k_hat = decode_kvcache_store(
        store, cfg.tile_size, cfg.use_differential,
    )
    store.update_stats()
    return store


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════


def _flush_buffer(
    store: DSKVCacheStore,
    cfg: DSKVCacheConfig,
    is_key: bool,
) -> DSKVCacheStore:
    """Encode raw buffer tokens and merge into the existing store."""
    if store.buffer_full == 0:
        return store

    # Extract buffered tokens
    new_mat = store.raw_buffer[:store.buffer_full, :].float()

    # Encode only the new tokens (keeping existing noise shaping)
    svd_shaper = store.svd_shaper if cfg.use_noise_shaping else None
    new_bases, new_alphas, new_shape = _encode_single_matrix(
        new_mat, cfg, svd_shaper, is_key,
    )

    # Merge with existing bases (unpack old, merge, re-pack)
    if store.bases is not None and store.n_tokens > 0:
        old_bases = unpack_bases(store.bases)
        if store.bases_shape_M is not None and old_bases.shape[-1] > store.bases_shape_M:
            old_bases = old_bases[..., :store.bases_shape_M]
        merged_bases, merged_alphas, merged_shape = _concat_stores(
            old_bases, store.alphas, store.orig_shape,
            new_bases, new_alphas, new_shape,
            store.tile_size, cfg, is_key,
        )
        store.bases = pack_bases(merged_bases)
        store.bases_shape_M = merged_bases.shape[-1]
        store.alphas = merged_alphas
        store.orig_shape = merged_shape
    else:
        store.bases = pack_bases(new_bases)
        store.bases_shape_M = new_bases.shape[-1]
        store.alphas = new_alphas
        store.orig_shape = new_shape

    store.n_tokens += store.buffer_full
    store.n_tiles = store.bases.shape[1]

    # Differential residual bases become stale after re-encoding merged matrix.
    # They referenced old tile grids and old residual errors — invalidate them.
    if cfg.use_differential:
        store.bases_residual = None
        store.alphas_residual = None

    # Clear buffer
    store.raw_buffer.zero_()
    store.buffer_full = 0

    # Invalidate decode cache
    store.full_k_hat = None

    store.update_stats()
    return store


def _encode_single_matrix(
    mat: torch.Tensor,
    cfg: DSKVCacheConfig,
    svd_shaper: Optional[dict],
    is_key: bool,
) -> tuple:
    """Encode a small matrix (ring-buffer contents) using the same path
    as the bulk encoder, reusing noise-shaping state if available.

    Uses heterogeneous n_steps: cfg.get_n_steps_k() for Key, cfg.get_n_steps_v() for Value.
    """
    from modules.residual_pursuit import encode_matrix, adaptive_encode_matrix

    proj_matrix = svd_shaper.get("projector") if svd_shaper else None
    n_steps_effective = cfg.get_n_steps_k() if is_key else cfg.get_n_steps_v()

    if cfg.adaptive_n:
        n_extra = cfg.n_upper_bound - n_steps_effective
        if n_extra <= 0:
            n_extra = 2  # minimum headroom
        bases, alphas, _, orig_shape = adaptive_encode_matrix(
            mat,
            n_steps_base=n_steps_effective,
            n_steps_extra=n_extra,
            tile_size=cfg.tile_size,
            beta=cfg.beta,
            proj_matrix=proj_matrix,
            proj_beta=cfg.proj_beta,
            adaptive_eta=cfg.adaptive_eta,
            energy_threshold_ratio=cfg.energy_threshold_factor,
            order2_gamma=cfg.order2_gamma,
            order2_c1=cfg.order2_c1,
            order2_c2=cfg.order2_c2,
        )
        return bases, alphas, orig_shape
    else:
        return encode_matrix(
            mat,
            n_steps=n_steps_effective,
            tile_size=cfg.tile_size,
            beta=cfg.beta,
            proj_matrix=proj_matrix,
            proj_beta=cfg.proj_beta,
            adaptive_eta=cfg.adaptive_eta,
            order2_gamma=cfg.order2_gamma,
            order2_c1=cfg.order2_c1,
            order2_c2=cfg.order2_c2,
        )


def _concat_stores(
    old_bases, old_alphas, old_shape,
    new_bases, new_alphas, new_shape,
    tile_size: int,
    cfg: DSKVCacheConfig,
    is_key: bool,
):
    """Concatenate two DS-KVCache encoded stores along the token dimension.

    Because the tile grid spans both token and d_head dimensions, naive
    concatenation along the tile axis is incorrect.  Instead we decode
    both stores back to dense, stack along the token axis, and re-encode.

    Re-encoding uses the full config parameters (beta, noise shaping,
    order2, adaptive N) and heterogeneous n_steps (get_n_steps_k/v)
    to maintain quality parity with the initial encode.
    """
    from modules.residual_pursuit import decode_from_bases, encode_matrix, adaptive_encode_matrix

    # Decode old store
    mat_old = decode_from_bases(old_bases, old_alphas, old_shape, tile_size)
    # Decode new store
    mat_new = decode_from_bases(new_bases, new_alphas, new_shape, tile_size)

    # Concatenate along token axis
    mat_total = torch.cat([mat_old, mat_new], dim=0)  # (n_total, d_head)

    # Re-encode with heterogeneous n_steps for K/V
    n_steps_effective = cfg.get_n_steps_k() if is_key else cfg.get_n_steps_v()

    # Build noise-shaping projector for re-encoding
    proj_matrix = None
    if cfg.use_noise_shaping:
        from modules.residual_pursuit import _build_proj_matrix
        proj_matrix = _build_proj_matrix(mat_total, tile_size, cfg.proj_rank)

    if cfg.adaptive_n:
        n_extra = cfg.n_upper_bound - n_steps_effective
        if n_extra <= 0:
            n_extra = 2
        bases, alphas, _, orig_shape = adaptive_encode_matrix(
            mat_total,
            n_steps_base=n_steps_effective,
            n_steps_extra=n_extra,
            tile_size=tile_size,
            beta=cfg.beta,
            proj_matrix=proj_matrix,
            proj_beta=cfg.proj_beta,
            adaptive_eta=cfg.adaptive_eta,
            energy_threshold_ratio=cfg.energy_threshold_factor,
            order2_gamma=cfg.order2_gamma,
            order2_c1=cfg.order2_c1,
            order2_c2=cfg.order2_c2,
        )
        return bases, alphas, orig_shape
    else:
        bases, alphas, orig_shape = encode_matrix(
            mat_total,
            n_steps=n_steps_effective,
            tile_size=tile_size,
            beta=cfg.beta,
            proj_matrix=proj_matrix,
            proj_beta=cfg.proj_beta,
            adaptive_eta=cfg.adaptive_eta,
            order2_gamma=cfg.order2_gamma,
            order2_c1=cfg.order2_c1,
            order2_c2=cfg.order2_c2,
        )
        return bases, alphas, orig_shape
