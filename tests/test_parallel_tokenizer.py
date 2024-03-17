import pytest

from parallel_tokenizer.parallel_tokenizer import ParallelTokenizer


@pytest.mark.parametrize("sentence_length", [81920, 163840])
def test_sp_tokenizer_in_parallel(sentence_length: int):
    import math
    import random

    from wonderwords import RandomWord

    from parallel_tokenizer.sp_tokenizer import SentencePieceTokenizer

    random.seed(1024)
    r = RandomWord()

    tokenizer = SentencePieceTokenizer("./tests/assets/tokenizer.model")
    parallel_tokenizer = ParallelTokenizer(
        tokenizer=tokenizer,
        parallel_degree=4,
        chunk_size=40960,
        overlap_length=512,
        concat_keys=["input_ids", "attention_mask"],
    )
    sentence: str = " ".join([r.word() for _ in range(sentence_length)])
    acc: float = parallel_tokenizer.test(sentence)
    assert math.isclose(acc, 1.0, abs_tol=1e-5)


@pytest.mark.parametrize("sentence_length", [81920, 163840])
@pytest.mark.parametrize("return_tensors", [None, "tp", "np"])
def test_hf_tokenizer_in_parallel(sentence_length: int, return_tensors: str):
    import math
    import random

    from transformers import AutoTokenizer
    from wonderwords import RandomWord

    random.seed(1024)
    r = RandomWord()

    tokenizer = AutoTokenizer.from_pretrained("./tests/assets/", trust_remote_code=True)
    parallel_tokenizer = ParallelTokenizer(
        tokenizer=tokenizer,
        parallel_degree=4,
        chunk_size=40960,
        overlap_length=512,
        concat_keys=["input_ids", "attention_mask"],
    )
    sentence: str = " ".join([r.word() for _ in range(sentence_length)])
    acc: float = parallel_tokenizer.test(sentence, return_tensors=return_tensors)
    assert math.isclose(acc, 1.0, abs_tol=1e-5)
