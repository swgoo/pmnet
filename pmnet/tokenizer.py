import torch
from transformers import PreTrainedTokenizer
from typing import Dict, List


class ByteTokenizer(PreTrainedTokenizer):
    def __init__(
        self, bos_token="<|bos|>", eos_token="<|eos|>", pad_token="<|pad|>", **kwargs
    ):
        self.pad_idx = 0
        self.bos_idx = 254
        self.eos_idx = 255

        super().__init__(
            bos_token=bos_token, eos_token=eos_token, pad_token=pad_token, **kwargs
        )

    @property
    def vocab_size(self) -> int:
        return 256

    def _tokenize(self, text, **kwargs):
        return list(text.encode("utf-8"))

    def _convert_token_to_id(self, token):
        return token if isinstance(token, int) else ord(token)

    def _convert_id_to_token(self, index):
        return index

    def get_vocab(self) -> Dict[str, int]:
        return {str(i): i for i in range(256)}

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        return [self.bos_idx] + token_ids_0 + [self.eos_idx]

    def encode_plus(self, text, add_special_tokens=True, **kwargs):
        ids = list(text.encode("utf-8"))
        if add_special_tokens:
            ids = [self.bos_idx] + ids + [self.eos_idx]

        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: List[str],
        add_special_tokens: bool = True,
        **kwargs,
    ):
        res = {"input_ids": [], "attention_mask": []}
        for text in batch_text_or_text_pairs:
            out = self.encode_plus(
                text, add_special_tokens=add_special_tokens, **kwargs
            )
            res["input_ids"].append(out["input_ids"])
            res["attention_mask"].append(out["attention_mask"])

        return res

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if skip_special_tokens:
            token_ids = [
                i
                for i in token_ids
                if i not in [self.bos_idx, self.eos_idx, self.pad_idx]
            ]

        return bytes(token_ids).decode("utf-8", errors="ignore")


__all__ = [
    "ByteTokenizer",
]
