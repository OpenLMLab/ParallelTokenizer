from typing import List, Tuple, Union

import numpy as np
import torch

from .logger import get_logger
from .utils import arange_like, get_size

logger = get_logger(__name__)


def position_ids(shards: List[Union[torch.Tensor, np.ndarray, List[int]]], matches: List[Tuple[int]]):
    seqlen = sum(
        [
            get_size(x[0])[-1] - x[1][1] - ((get_size(x[0])[-1] - x[1][0]) % get_size(x[0])[-1])
            for x in zip(shards, [matches[i : i + 2] for i in range(0, len(matches), 2)])
        ]
    )
    return arange_like(shards[0], start=0, end=seqlen)


SPECIAL_KEYS_DICT = {"position_ids": position_ids}

SPECIAL_TOKENIZERS_DICT = {}
