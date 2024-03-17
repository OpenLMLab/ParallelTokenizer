# Copyright (c) OpenLMLab. All rights reserved.
import os
from typing import Any, List

from sentencepiece import SentencePieceProcessor

from .logger import get_logger

logger = get_logger(__name__)


class SentencePieceTokenizer:
    """Wrapper for SentencePiece tokenizer."""

    def __init__(self, model_path: str, use_logger: bool = True, use_bos: bool = True, use_eos: bool = True) -> None:
        # reload tokenizer
        if not os.path.isfile(model_path):
            raise ValueError(f"Got invalid `model_path={model_path}` for SentencePieceTokenizer.")
        self.sp_model = SentencePieceProcessor(model_file=model_path)  # pylint: disable=unexpected-keyword-arg

        self.use_bos = use_bos
        self.use_eos = use_eos

        if use_logger:
            logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

        if use_logger:
            logger.info(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")

        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size(), (
            f"The vocab size {self.sp_model.vocab_size()} of the model in {model_path} is not equal to "
            f"the piece size {self.sp_model.get_piece_size()} of the model."
        )

    def encode(self, s: str, use_bos: bool | None = None, use_eos: bool | None = None) -> List[int]:
        assert isinstance(s, str), "tokenizer expect a string, but got a " + str(type(s))
        t = self.sp_model.encode(s)
        if use_bos is None:
            use_bos = self.use_bos
        if use_bos:
            t = [self.bos_id] + t
        if use_eos is None:
            use_eos = self.use_eos
        if use_eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

    def __call__(self, s: str, use_bos: bool | None = None, use_eos: bool | None = None) -> Any:
        return self.encode(s, use_bos=use_bos, use_eos=use_eos)

    def id_to_piece(self, token_id):
        """
        This method is used by the scoring function module in the tools folder.
        """
        return self.sp_model.id_to_piece(token_id)
