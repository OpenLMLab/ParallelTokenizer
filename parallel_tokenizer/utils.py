from difflib import SequenceMatcher
from functools import reduce
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch

from .logger import get_logger

logger = get_logger(__name__)


def concat(*args: List[Union[torch.Tensor, np.ndarray, List[int]]], dim: int = -1):
    """
    Concatenates a sequence of tensors, numpy arrays, or lists.

    Parameters:
    - *args: A variable number of lists, each containing torch.Tensors,
      numpy.ndarrays, or Python lists to be concatenated.

    Returns:
    - Union[torch.Tensor, np.ndarray, List[int]]: The concatenated result
      as the same type as the input. The concatenation is performed along
      the first axis for tensors and arrays, and by appending lists for lists.

    Raises:
    - ValueError: If the type of the input is not supported.
    """
    if dim < 0:
        dim = get_ndim(args[0]) + dim
    assert dim < get_ndim(args[0]), "dim should be less than the number of dimensions of the input"
    if isinstance(args[0], torch.Tensor):
        return torch.cat(args, dim=dim)
    elif isinstance(args[0], np.ndarray):
        return np.concatenate(args, axis=dim)
    elif isinstance(args[0], List):
        if dim == 0:
            return list(reduce(lambda x, y: x + y, args))
        else:
            return [concat(*items, dim=dim - 1) for items in zip(*args)]
    else:
        raise ValueError(f"Unsupported type {type(args[0])} for concat")


def chunks(sentence: Union[str, Sequence[str]], chunk_size: int = 40960, overlap_length: int = 512):
    """
    Splits a string or a sequence of strings into chunks with a specified size and overlap.

    Parameters:
    - sentence (Union[str, Sequence[str]]): The input text to be chunked. Can be a single
      string or a sequence of strings.
    - chunk_size (int, optional): The size of each chunk. Defaults to 40960.
    - overlap_length (int, optional): The length of the overlap between adjacent chunks.
      Defaults to 512.

    Yields:
    - Iterable[Union[str, List[str]]]: An iterable of chunks, each being a string
      or a list of strings,
      depending on the input type.

    Raises:
    - ValueError: If the input type is not a string or a sequence of strings.
    """
    if isinstance(sentence, str):
        while sentence:
            yield sentence[: overlap_length + chunk_size]
            sentence = sentence[chunk_size:]
    elif isinstance(sentence, Sequence) and isinstance(sentence[0], str):
        while any(sentence):
            yield [s[: overlap_length + chunk_size] for s in sentence]
            sentence = [s[chunk_size:] for s in sentence]
    else:
        raise ValueError(f"Unsupported type {type(sentence)} for chunks")


def pairs(chunks: List[List[int]]) -> Iterable[List[List[int]]]:
    """
    Generates consecutive pairs of chunks for matching.

    Parameters:
    - chunks (List[List[int]]): A list of chunks, where each chunk is a list of integers.

    Yields:
    - Iterable[List[List[int]]]: An iterable of pairs of consecutive chunks.
    """
    for i in range(0, len(chunks) - 1):
        yield (chunks[i], chunks[i + 1])


def match(chunks: List[Union[torch.Tensor, np.ndarray, List[int]]]) -> Tuple[int]:
    """
    Finds the longest matching subsequence between the last part of the first chunk and the first
      part of the second chunk.

    Parameters:
    - chunks (List[Union[torch.Tensor, np.ndarray, List[int]]]): A list containing two chunks.
      Each chunk can be a torch.Tensor, a numpy.ndarray, or a list of integers.

    Returns:
    - Tuple[int]: A tuple containing the starting index of the match in the first chunk and the
      ending index of the match in the second chunk.

    Notes:
    - This function is primarily used for adjusting the boundaries between chunks that have been
      tokenized separately to ensure a seamless merge.
    """
    _match = SequenceMatcher(
        None, to_list(reverse(chunks[0])), to_list(reverse(chunks[-1])), autojunk=False
    ).find_longest_match(0, len(chunks[0]), 0, len(chunks[-1]))
    if _match.size == 0:
        logger.warning(
            "It is detected that the text cannot be tokenized accurately, "
            "which is often caused by tokenizer_chunk being too small. "
        )
    return _match.a, _match.b


def flatten(item: Union[torch.Tensor, np.ndarray, List]):
    """
    Flattens a nested sequence (torch.Tensor, numpy.ndarray, or List) into a single flat list.

    Parameters:
    - item (Union[torch.Tensor, np.ndarray, List]): The item to flatten, which can be either
      a torch.Tensor, a numpy.ndarray, or a nested list.

    Returns:
    - Union[torch.Tensor, np.ndarray, List[int]]: A flattened version of the input, as a
      torch.Tensor, a numpy.ndarray, or a list of integers.

    Raises:
    - ValueError: If the input type is not supported or if the flattening process fails.
    """
    if isinstance(item, (torch.Tensor, np.ndarray)):
        return item.flatten()
    elif isinstance(item, List):
        while all(isinstance(i, Sequence) for i in item):
            item = [i for sublist in item for i in sublist]
        assert not any(isinstance(i, Sequence) for i in item), "flatten failed"
        return item
    else:
        raise ValueError(f"Unsupported type {type(item)} for flatten")


def merge(
    shards: List[Union[torch.Tensor, np.ndarray, List[int]]], matches: List[Tuple[int]], merge_dim: int = -1
) -> Union[torch.Tensor, np.ndarray, List[int]]:
    """
    Merges a list of tokenized shards, adjusting for overlaps using provided match indices.

    Parameters:
    - shards (List[Union[torch.Tensor, np.ndarray, List[int]]]): The tokenized shards to merge.
      Each shard can be a torch.Tensor, a numpy.ndarray, or a list of integers.
    - matches (List[Tuple[int]]): A list of tuples indicating the overlap indices for adjacent shards.

    Returns:
    - Union[torch.Tensor, np.ndarray, List[int]]: The merged result, in the same type as the shards.

    Raises:
    - AssertionError: If the length of the shards list is not twice the length of the matches list,
      indicating a mismatch in provided data.
    """
    assert len(shards) * 2 == len(matches), "the length of shards should be twice the length of matches"
    result = map(
        lambda x: slicing(
            x[0],
            dim=merge_dim,
            start=(get_size(x[0])[-1] - x[1][0]) % get_size(x[0])[-1],
            end=get_size(x[0])[-1] - x[1][1],
        ),
        zip(shards, [matches[i : i + 2] for i in range(0, len(matches), 2)]),
    )
    result = reduce(lambda x, y: concat(x, y, dim=merge_dim), result)
    return result


def reverse(item: Union[torch.Tensor, np.ndarray, List[int]]):
    """
    Reverses the order of elements in the input item. The input item can be a PyTorch tensor, a NumPy array,
    or a list of integers. This function is designed to work with these specific types to accommodate common
    data structures used in machine learning and data processing tasks.

    Parameters:
    - item (Union[torch.Tensor, np.ndarray, List[int]]): The item whose elements are to be reversed. It can be a
      PyTorch tensor, a NumPy array, or a list of integers.

    Returns:
    - The reversed item, maintaining the original type. For a PyTorch tensor, the function returns a tensor with
      elements in reverse order. For a NumPy array, it returns an array with elements in reverse order. For a list
      of integers, it returns a list with integers in reverse order.

    Raises:
    - ValueError: If the input `item` is not a PyTorch tensor, a NumPy array, or a list of integers, the function
      raises a ValueError indicating the unsupported type.
    """
    if isinstance(item, (np.ndarray, list)):
        return item[::-1]
    elif isinstance(item, torch.Tensor):
        return torch.flip(item, [0])
    else:
        raise ValueError(f"Unsupported type {type(item)} for reverse")


def to_list(item: Union[torch.Tensor, np.ndarray, List[int]]) -> List[int]:
    """
    Converts the input item to a list of integers. The function supports input items that are
    PyTorch tensors, NumPy arrays, or already lists of integers.

    Parameters:
    - item (Union[torch.Tensor, np.ndarray, List[int]]): The item to convert to a list. It can be a
      PyTorch tensor, a NumPy array, or a list of integers.

    Returns:
    - List[int]: A list of integers derived from the input item.

    Raises:
    - ValueError: If the input `item` is not a supported type (i.e., not a PyTorch tensor, a NumPy array,
      or a list of integers), the function raises a ValueError.
    """
    if isinstance(item, torch.Tensor):
        return item.cpu().tolist()
    elif isinstance(item, np.ndarray):
        return item.tolist()
    elif isinstance(item, List):
        return item
    else:
        raise ValueError(f"Unsupported type {type(item)} for to_list")


def get_ndim(item: Union[torch.Tensor, np.ndarray, List[int]]) -> int:
    """
    Determines the number of dimensions of the input item. The function supports input items that
    are PyTorch tensors, NumPy arrays, or lists (potentially nested to represent higher dimensions).

    Parameters:
    - item (Union[torch.Tensor, np.ndarray, List[int]]): The item whose number of dimensions is to
      be determined.

    Returns:
    - int: The number of dimensions of the input item.

    Raises:
    - ValueError: If the input `item` is not a supported type (i.e., not a PyTorch tensor, a NumPy array,
      or a list), the function raises a ValueError.
    """
    if isinstance(item, (torch.Tensor, np.ndarray)):
        return item.ndim
    elif isinstance(item, List):
        ndim = 1
        while isinstance(item[0], List):
            item = item[0]
            ndim += 1
        return ndim
    else:
        raise ValueError(f"Unsupported type {type(item)} for get_ndim")


def slicing(item: Union[torch.Tensor, np.ndarray, List[int]], dim: int = -1, start: int = 0, end: int = None):
    """
    Slices the input item along a specified dimension from a start index to an end index.
      The function supports input items that are PyTorch tensors, NumPy arrays, or lists
      (potentially nested to represent higher dimensions).

    Parameters:
    - item (Union[torch.Tensor, np.ndarray, List[int]]): The item to slice.
    - dim (int, optional): The dimension along which to slice. Defaults to -1,
      which typically means the last dimension.
    - start (int, optional): The start index of the slice. Defaults to 0.
    - end (int, optional): The end index of the slice. If None, slicing goes to the end of
      the dimension. Defaults to None.

    Returns:
    - The sliced portion of the input item, maintaining the original type.

    Raises:
    - ValueError: If the input `item` is not a supported type or the specified dimension
      is invalid.
    """
    if dim < 0:
        dim = get_ndim(item) + dim
    assert dim < get_ndim(item), "dim should be less than the number of dimensions of the input"
    if isinstance(item, torch.Tensor):
        return item.narrow(dim, start, end - start)
    elif isinstance(item, np.ndarray):
        return item.take(range(start, end), axis=dim)
    elif isinstance(item, List):
        if dim == 0:
            return item[start:end]
        else:
            return [slicing(subitem, dim - 1, start, end) for subitem in item]


def get_size(item: Union[torch.Tensor, np.ndarray, List[int]]) -> Tuple[int]:
    """
    Returns the size (shape) of the input item. The function supports input items that are PyTorch tensors,
    NumPy arrays, or lists (potentially nested to represent higher dimensions).

    Parameters:
    - item (Union[torch.Tensor, np.ndarray, List[int]]): The item whose size is to be determined.

    Returns:
    - Tuple[int]: A tuple representing the size (shape) of the item. For tensors and arrays, it directly
      corresponds to their shape. For lists, it's the equivalent shape based on the nesting level and length
      of the lists.

    Raises:
    - ValueError: If the input `item` is not a supported type, the function raises a ValueError.
    """
    if isinstance(item, torch.Tensor):
        return item.size()
    elif isinstance(item, np.ndarray):
        return item.shape
    elif isinstance(item, List):
        ndim = get_ndim(item)
        shape = []
        for _ in range(ndim):
            shape.append(len(item))
            item = item[0]
        return tuple(shape)
    else:
        raise ValueError(f"Unsupported type {type(item)} for get_size")
