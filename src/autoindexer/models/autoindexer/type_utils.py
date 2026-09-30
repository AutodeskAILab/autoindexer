from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import TypedDict, Tuple, Optional, Callable, Dict

from torch import Tensor


Marker = int | str
MarkerSet = Tuple[int, int]


class MarkerMap(TypedDict):
    edit: Marker  # edit token
    return_to_end: Marker  # return token
    eos: Marker  # end of sequence token


# Classes of the auxiliary marker head.
class MarkerType(IntEnum):
    NONE = 0
    EDIT = 1
    RETURN = 2
    EOS = 3


class BlockType(IntEnum):
    EXTEND = 1
    EDIT = 2
    COMPLETE = 3


@dataclass
class PositionBlock:
    key_pos_ids: Tensor
    query_pos_ids: Optional[Tensor] = None
    abs_pos_mask: Optional[Tensor] = None
    attn_mask: Optional[Tensor] = None
    start_idx: int = None
    end_idx: int = None
    block_type: BlockType = None


# (x: Tensor, position_ids: Tensor) -> (cos: Tensor, sin: Tensor)
RotaryEmbeddingFunc = Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]


@dataclass
class IterPositions:
    key_pos_ids: Tensor
    attn_mask: Optional[Tensor] = None
    is_edit_token: Tensor | bool = None
    # Separate from `attn_mask`: the backbone's own causal attention (see `decode()`) needs
    # to keep reading the *entire* context (locked prompt/prior chat turns included) for
    # coherent generation, so it always uses `attn_mask`. Only `AutoIndexerIndexHead`'s
    # cursor attention additionally applies this -- `attn_mask` further restricted to
    # positions `>= editable_start` (see `AutoIndexerIdParser`) -- so the cursor itself can
    # never point into locked context. `None` when no `editable_start` was set (i.e.
    # everything is editable), so the index head just falls back to `attn_mask`.
    index_attn_mask: Optional[Tensor] = None
    _rotary_pos_emb: Optional[Tuple[Tensor, Tensor]] = None

    def get_rotary_pos_emb(self, rotary_emb: RotaryEmbeddingFunc, x: Tensor) -> Tuple[Tensor, Tensor]:
        if self._rotary_pos_emb is None:
            key_pos_ids = self.key_pos_ids.unsqueeze(0) if self.key_pos_ids.ndim == 1 else self.key_pos_ids
            assert key_pos_ids.ndim == 2
            self._rotary_pos_emb = rotary_emb(x, key_pos_ids)
        return self._rotary_pos_emb


CursorIndices = Dict[int, Tuple[int, int]]
