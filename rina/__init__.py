"""RINA — Residual-Integrated Neural Architecture

DS-KVCache integration layer for HuggingFace transformers models.

Key modules
  • rina.config        — Unified configuration dataclass
  • rina.ds_kv_cache   — Core encode/decode pipeline + DSKVCacheStore
  • rina.incremental_decode — Ring-buffer incremental encoding (§5)
  • rina.model_wrapper — HuggingFace model wrapper with DS-KVCache hooks
"""

from .config import DSKVCacheConfig
from .ds_kv_cache import (
    DSKVCacheStore,
    encode_kv_cache,
    decode_kvcache_store,
)
from .incremental_decode import (
    init_incremental_store,
    incremental_encode_step,
    incremental_encode_batch,
    finalize_store,
)
from .model_wrapper import (
    DSKVCacheModel,
)

__all__ = [
    "DSKVCacheConfig",
    "DSKVCacheStore",
    "encode_kv_cache",
    "decode_kvcache_store",
    "init_incremental_store",
    "incremental_encode_step",
    "incremental_encode_batch",
    "finalize_store",
    "DSKVCacheModel",
]