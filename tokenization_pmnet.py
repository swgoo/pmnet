from transformers import PreTrainedTokenizer
from typing import Dict, List, Optional, Any


class ByteTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        pad_token="<|pad|>",
        vocab_size=384,
        **kwargs,
    ):
        self.pad_idx = 0
        self.bos_idx = 254
        self.eos_idx = 255
        self._vocab_size = vocab_size

        self.byte_to_token = [f"<byte_{i}>" for i in range(256)]
        self.token_to_byte = {t: i for i, t in enumerate(self.byte_to_token)}

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def get_vocab(self) -> Dict[str, int]:
        vocab = {t: i for i, t in enumerate(self.byte_to_token)}
        vocab.update(
            {
                self.bos_token: self.bos_idx,
                self.eos_token: self.eos_idx,
                self.pad_token: self.pad_idx,
            }
        )
        return vocab

    def _tokenize(self, text, **kwargs):
        return [self.byte_to_token[b] for b in text.encode("utf-8")]

    def _convert_token_to_id(self, token):
        if token == self.bos_token:
            return self.bos_idx
        if token == self.eos_token:
            return self.eos_idx
        if token == self.pad_token:
            return self.pad_idx
        return self.token_to_byte.get(token, self.pad_idx)

    def _convert_id_to_token(self, index):
        if index == self.bos_idx:
            return self.bos_token
        if index == self.eos_idx:
            return self.eos_token
        if index == self.pad_idx:
            return self.pad_token
        if 0 <= index < 256:
            return self.byte_to_token[index]
        return f"<unk_{index}>"

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        return [self.bos_idx] + token_ids_0 + [self.eos_idx]

    def _decode(
        self, token_ids: List[int], skip_special_tokens: bool = False, **kwargs
    ) -> str:
        clean_ids = []
        for i in token_ids:
            if skip_special_tokens and i in [self.bos_idx, self.eos_idx, self.pad_idx]:
                continue
            if 0 <= i < 256:
                clean_ids.append(i)
        return bytes(clean_ids).decode("utf-8", errors="ignore")

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> tuple:
        return ()


__all__ = [
    "ByteTokenizer",
]
