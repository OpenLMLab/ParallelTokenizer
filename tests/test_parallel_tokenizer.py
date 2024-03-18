import os
from tempfile import TemporaryDirectory

import pytest

from parallel_tokenizer.parallel_tokenizer import ParallelTokenizer
from tests.utils import download_file


@pytest.mark.parametrize("sentence_length", [81920, 163840])
def test_sp_tokenizer_in_parallel(sentence_length: int):
    import math
    import random

    from wonderwords import RandomWord

    from parallel_tokenizer.sp_tokenizer import SentencePieceTokenizer

    random.seed(1024)
    r = RandomWord()

    with TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "tokenizer.model")
        download_file("https://huggingface.co/internlm/internlm2-20b/resolve/main/tokenizer.model", model_path)
        tokenizer = SentencePieceTokenizer(model_path)

    parallel_tokenizer = ParallelTokenizer(
        tokenizer=tokenizer,
        num_processes=4,
        chunk_size=40960,
        overlap_length=512,
        concat_keys=["input_ids", "attention_mask"],
    )
    sentence: str = " ".join([r.word() for _ in range(sentence_length)])
    _, _, acc = parallel_tokenizer.benchmark(sentence)
    assert math.isclose(acc, 1.0, abs_tol=1e-5)


@pytest.mark.parametrize("sentence_length", [81920])
@pytest.mark.parametrize("return_tensors", [None, "pt", "np"])
def test_hf_tokenizer_in_parallel(sentence_length: int, return_tensors: str):
    import math
    import random

    from transformers import AutoTokenizer
    from wonderwords import RandomWord

    random.seed(1024)
    r = RandomWord()

    tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-20b", trust_remote_code=True)
    parallel_tokenizer = ParallelTokenizer(
        tokenizer=tokenizer,
        num_processes=4,
        chunk_size=40960,
        overlap_length=512,
        concat_keys=["input_ids", "attention_mask"],
    )
    sentence: str = " ".join([r.word() for _ in range(sentence_length)])
    _, _, acc = parallel_tokenizer.benchmark(sentence, return_tensors=return_tensors)
    assert math.isclose(acc, 1.0, abs_tol=1e-5)
