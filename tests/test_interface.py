import pytest
import random

import torch
from transformers import AutoTokenizer
from wonderwords import RandomWord

from parallel_tokenizer import ParallelTokenizer


TEST_MODELS = [
    "internlm/internlm2-7b",
    # "meta-llama/Llama-2-7b-chat-hf",
    "baichuan-inc/Baichuan2-7B-Chat",
    "mistralai/Mistral-7B-Instruct-v0.2",
    # "google/gemma-7b-it",
    "Qwen/Qwen1.5-7B-Chat",
    "THUDM/chatglm3-6b",
]
TEST_LENGTHS = [8192, 16384]


@pytest.mark.parametrize("model_name_or_path", TEST_MODELS)
@pytest.mark.parametrize("sentence_length", TEST_LENGTHS)
@pytest.mark.parametrize("add_special_tokens", [True, False])
@pytest.mark.parametrize("return_tensors", [None, "pt"])
@pytest.mark.parametrize("batch", [False])
def test_call(
    model_name_or_path: str,
    sentence_length: int,
    add_special_tokens: bool,
    return_tensors: str or None,
    batch: bool
):
    random.seed(1024)
    r = RandomWord()

    tokenizer_hf = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    parallel_tokenizer = ParallelTokenizer(
        tokenizer=tokenizer_hf,
        num_processes=4,
        chunk_size=4096,
        overlap_length=128,
        concat_keys=["input_ids", "attention_mask"],
    )

    if batch:
        input_text: list[str] = [" ".join([r.word() for _ in range(sentence_length)]) for _ in range(2)]
    else:
        input_text: str = " ".join([r.word() for _ in range(sentence_length)])

    ret_hf = tokenizer_hf(input_text, add_special_tokens=add_special_tokens, return_tensors=return_tensors)
    ret_parallel = parallel_tokenizer(input_text, add_special_tokens=add_special_tokens, return_tensors=return_tensors)

    for k in ret_hf:
        if isinstance(ret_hf[k], list):
            assert ret_hf[k] == ret_parallel[k], f"{k} is not equal"
        elif isinstance(ret_hf[k], torch.Tensor):
            assert ret_hf[k].equal(ret_parallel[k]), f"{k} is not equal"
        else:
            assert f"{type(ret_hf[k])} is not supported"


if __name__ == "__main__":
    test_call("THUDM/chatglm3-6b", 8192, True, "pt", False)
