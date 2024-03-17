# Copyright (c) OpenLMLab. All rights reserved.
import multiprocessing as mp
import time
from functools import partial, reduce
from typing import Any, Callable, List, Sequence, Tuple, Union

import numpy as np
import torch
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer

from .logger import get_logger
from .sp_tokenizer import SentencePieceTokenizer
from .utils import chunks, flatten, match, merge, pairs, to_list

logger = get_logger(__name__)


class ParallelTokenizer:
    """
    ParallelTokenizer is designed to accelerate the tokenization process by utilizing multiprocessing.
    It splits the input text into chunks, tokenizes them in parallel, and then merges the results.

    Parameters:
    - tokenizer (Union[SentencePieceTokenizer, PreTrainedTokenizer]): The tokenizer instance that will be
      used for tokenizing text. It can be either a SentencePieceTokenizer or a PreTrainedTokenizer.
    - num_processes (int, optional): The number of parallel processes to use for tokenizing. Defaults to 4.
    - chunk_size (int, optional): The size of each chunk of text to be tokenized in parallel. Defaults to 40960.
    - overlap_length (int, optional): The length of the overlapping parts of the text chunks to ensure continuity
      in the tokenization process. Defaults to 512.
    - concat_keys (List[str], optional): The keys of the tokenized outputs to be concatenated when merging results.
      Defaults to ["input_ids", "attention_mask"].
    """

    def __init__(
        self,
        tokenizer: Union[SentencePieceTokenizer, PreTrainedTokenizer],
        num_processes: int = 4,
        chunk_size: int = 40960,
        overlap_length: int = 512,
        concat_keys: Sequence[str] = ("input_ids", "attention_mask"),
    ) -> None:
        assert callable(tokenizer), "tokenizer should be callable"
        self.tokenizer = tokenizer
        self.num_processes = num_processes
        self.chunk_size = chunk_size
        self.overlap_length = overlap_length
        self.concat_keys = concat_keys
        self.pool = mp.Pool(num_processes)  # pylint: disable=R1732

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Tokenizes the input text in parallel by splitting it into chunks, tokenizing each chunk in a separate
        process, and then merging the results.

        Parameters:
        - *args: Positional arguments passed to the tokenizer. The first argument is expected to be the
          text to tokenize.
        - **kwargs: Keyword arguments passed to the tokenizer. If 'text' is not provided as a positional
          argument, it should be passed as a keyword argument with the key 'text'.

        Returns:
        - The tokenized output, which can be a list of token ids, a dictionary, or a BatchEncoding object,
          depending on the tokenizer's output format.
        """
        if len(args) > 0:
            text = args[0]
            args = args[1:]
        else:
            text = kwargs.pop("text")
        assert isinstance(text, str), "Currently not support batch encoding. Please pass the text as a string."
        _tokenizer = partial(self.tokenizer, *args, **kwargs)
        shards = self.pool.map(
            partial(ParallelTokenizer.encode_handler, tokenizer=_tokenizer),
            chunks(text, self.chunk_size, self.overlap_length),
        )
        if isinstance(shards[0], (dict, BatchEncoding)):
            tokens_shards = [flatten(shard["input_ids"]) for shard in shards]
        else:
            tokens_shards = [flatten(shard) for shard in shards]

        matches = self.pool.map(ParallelTokenizer.match_handler, pairs(tokens_shards))
        matches = [0] + list(reduce(lambda x, y: x + y, matches)) + [0]

        if isinstance(shards[0], (dict, BatchEncoding)):
            result = shards[0].__class__()
            for key in self.concat_keys:
                result[key] = merge([shard[key] for shard in shards], matches)
            for key in shards[0].keys():
                if key not in self.concat_keys:
                    result[key] = [shard[key] for shard in shards]
        else:
            result = merge(shards, matches)
        return result

    def __getattr__(self, __name: str) -> Any:
        """
        Allows direct access to the tokenizer's attributes.

        Parameters:
        - __name (str): The attribute name to access from the tokenizer.

        Returns:
        - The value of the attribute named `__name` from the tokenizer.
        """
        return getattr(self.tokenizer, __name)

    def benchmark(self, *args: Any, **kwargs: Any) -> float:
        """
        Tests the efficiency and accuracy of the parallel tokenization process compared to the sequential process.

        Parameters:
        - *args: Positional arguments passed to the tokenizer for testing.
        - **kwargs: Keyword arguments passed to the tokenizer for testing.

        Returns:
        - A float representing the accuracy of the parallel tokenization process compared to the sequential process.
        """
        start = time.time()
        raw_result = self.tokenizer(*args, **kwargs)
        raw_time = time.time() - start

        start = time.time()
        parallel_result = self(*args, **kwargs)
        parallel_time = time.time() - start

        if isinstance(raw_result, (dict, BatchEncoding)):
            raw_tokens = to_list(flatten(raw_result["input_ids"]))
            parallel_tokens = to_list(flatten(parallel_result["input_ids"]))
        else:
            raw_tokens = to_list(flatten(raw_result))
            parallel_tokens = to_list(flatten(parallel_result))

        acc = [raw_tokens[i] - parallel_tokens[i] for i in range(min(len(raw_tokens), len(parallel_tokens)))].count(
            0
        ) / min(len(raw_tokens), len(parallel_tokens))

        logger.info(f"raw_time: {raw_time:.4f} - parallel_time: {parallel_time:.4f} - acc: {acc:.4f}")

        return acc

    @staticmethod
    def encode_handler(
        chunk: Union[str, Sequence[str]], tokenizer: Callable
    ) -> Union[torch.Tensor, np.ndarray, List[int]]:
        """
        A static method used as a handler for encoding a single chunk of text with the tokenizer.

        Parameters:
        - chunk (Union[str, Sequence[str]]): A single chunk of text or a sequence of texts to tokenize.
        - tokenizer (Callable): The tokenizer function to use for tokenizing the chunk.

        Returns:
        - The tokenized output for the chunk, which can be a torch.Tensor, np.ndarray, or List[int],
          depending on the tokenizer's output format.
        """
        return tokenizer(chunk)

    @staticmethod
    def match_handler(chunks: List[Union[torch.Tensor, np.ndarray, List[int]]]) -> Tuple[int]:
        """
        A static method used to handle matching and merging overlapping parts of tokenized chunks.

        Parameters:
        - chunks (List[Union[torch.Tensor, np.ndarray, List[int]]]): A list of tokenized outputs for consecutive chunks
          that need to be matched and merged.

        Returns:
        - A tuple of integers indicating the positions where chunks should be merged.
        """
        return match(chunks)


def convert_parallel_tokenizer(  # pylint: disable=W0102
    tokenizer: Union[SentencePieceTokenizer, PreTrainedTokenizer],
    num_processes: int = 4,
    chunk_size: int = 40960,
    overlap_length: int = 512,
    concat_keys: List[str] = ["input_ids", "attention_mask"],
) -> ParallelTokenizer:
    """
    A convenience function to create a ParallelTokenizer instance with the specified configuration.

    Parameters:
    - tokenizer (Union[SentencePieceTokenizer, PreTrainedTokenizer]): The tokenizer to be used for parallel
      tokenization.
    - num_processes (int, optional): The number of processes to use for parallel tokenization. Defaults to 4.
    - chunk_size (int, optional): The size of text chunks to be tokenized in parallel. Defaults to 40960.
    - overlap_length (int, optional): The length of overlaps between text chunks to ensure continuity. Defaults to 512.
    - concat_keys (List[str], optional): The keys of the tokenization output to be concatenated.
      Defaults to ["input_ids", "attention_mask"].

    Returns:
    - A ParallelTokenizer instance configured with the specified parameters.
    """
    return ParallelTokenizer(
        tokenizer=tokenizer,
        num_processes=num_processes,
        chunk_size=chunk_size,
        overlap_length=overlap_length,
        concat_keys=concat_keys,
    )
