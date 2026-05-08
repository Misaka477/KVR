"""
§6 EncodeBuffer — mutable FP16 ring buffer for incremental encoding.

Holds raw (unencoded) K/V rows until enough accumulate to form a tile.
Supports protected mode where the buffer holds ALL tokens without ever
triggering 1-bit encoding (§8.1.8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class EncodeBuffer:
    """Mutable raw buffer for incremental token accumulation.

    When protected=True, the buffer accumulates tokens indefinitely
    (no tile encoding triggers).  Otherwise, tokens are drained and
    encoded when ``buffer_full >= tile_size``.
    """

    data: Optional[torch.Tensor] = None
    """Raw FP16 matrix ``(n_buffered, d_head)``."""

    buffer_full: int = 0
    """Number of valid rows in *data*."""

    protected: bool = False
    """If True, never encode — keep all tokens as FP16 raw."""

    @property
    def d_head(self) -> int:
        """Head dimension from buffer data."""
        if self.data is not None:
            return self.data.shape[1]
        return 0

    @property
    def n_tokens(self) -> int:
        """Number of currently buffered tokens."""
        return self.buffer_full

    def is_empty(self) -> bool:
        """True when no buffered tokens."""
        return self.buffer_full == 0 or self.data is None

    def reset(self):
        """Clear the buffer without reallocating."""
        if self.data is not None:
            self.data.zero_()
        self.buffer_full = 0

    def append(self, new_vec: torch.Tensor) -> int:
        """Append one or more FP16 rows. Returns new buffer_full."""
        B = new_vec.shape[0]
        if self.data is None:
            self.data = new_vec.to(torch.float16)
            self.buffer_full = B
        else:
            self.data = torch.cat([self.data, new_vec.to(torch.float16)], dim=0)
            self.buffer_full += B
        return self.buffer_full

    def drain_head(self, n: int) -> torch.Tensor:
        """Remove and return the first *n* rows from the buffer.

        Returns FP32 tensor.  Updates buffer in-place.
        """
        result = self.data[:n].to(torch.float32)
        if n >= self.buffer_full:
            self.data = None
            self.buffer_full = 0
        else:
            self.data = self.data[n:]
            self.buffer_full -= n
        return result

    def peek_all(self) -> torch.Tensor:
        """Return a copy of all buffered data as FP32 (does not drain)."""
        if self.data is None or self.buffer_full == 0:
            return torch.empty(0, 0)
        return self.data[:self.buffer_full].to(torch.float32)


__all__ = ["EncodeBuffer"]
