"""
§3 Bit-packing utilities — compressed int32 storage for binary bases.

Pack/unpack functions that convert float (-1/+1) bases to compact
int32 bit-packed tensors (32 signs per int32 element).  Extracted
from residual_pursuit.py to a shared utility so both the core
encoding module and the DSKVCache store layer can use them.
"""

from __future__ import annotations

import torch

BITS_PER_PACK = 32


def pack_bases(bases: torch.Tensor) -> torch.Tensor:
    """Pack float (-1/+1) bases into bit-packed int32 tensor.

    Parameters
    ----------
    bases:
        ``(..., M)`` float tensor with values in ``{-1, +1}``.

    Returns
    -------
    packed:
        ``(..., M_packed)`` int32 tensor where each element encodes
        up to 32 consecutive signs (LSB = leftmost sign).
        ``M_packed = ceil(M / 32)``.  Trailing bits of the last word
        are zero-padded.
    """
    shape = bases.shape
    M = shape[-1]
    M_packed = (M + BITS_PER_PACK - 1) // BITS_PER_PACK

    pad_len = M_packed * BITS_PER_PACK - M
    if pad_len > 0:
        bases = torch.nn.functional.pad(bases, (0, pad_len), value=1.0)

    bits = (bases > 0).to(torch.uint8)

    bits = bits.reshape(*shape[:-1], M_packed, BITS_PER_PACK)

    bit_weights = 1 << torch.arange(BITS_PER_PACK, device=bases.device, dtype=torch.int32)
    packed = (bits.to(torch.int32) * bit_weights).sum(dim=-1)

    return packed.to(torch.int32)


def unpack_bases(packed: torch.Tensor) -> torch.Tensor:
    """Unpack bit-packed int32 tensor back to float (-1/+1) bases.

    Parameters
    ----------
    packed:
        ``(..., M_packed)`` int32 tensor.

    Returns
    -------
    bases:
        ``(..., M_packed * 32)`` float tensor with values in ``{-1, +1}``.
        Caller should slice ``[..., :original_M]`` if trailing padding
        was introduced during packing.
    """
    device = packed.device
    shape = packed.shape
    M_packed = shape[-1]

    bit_weights = 1 << torch.arange(BITS_PER_PACK, device=device, dtype=torch.int32)
    packed_expanded = packed.unsqueeze(-1)
    bits = (packed_expanded & bit_weights) != 0

    bases = bits.to(torch.float32) * 2.0 - 1.0

    bases = bases.reshape(*shape[:-1], M_packed * BITS_PER_PACK)

    return bases


__all__ = ["BITS_PER_PACK", "pack_bases", "unpack_bases"]
