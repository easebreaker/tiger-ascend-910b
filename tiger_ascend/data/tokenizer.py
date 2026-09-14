"""Minimal semantic-ID tokenizer (offline, no HF vocab download)."""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Sequence, Union

import torch


class SemanticIdTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"
    unk_token = "<unk>"

    def __init__(self, extra_tokens: Sequence[str], model_max_length: int = 256):
        ordered = [self.pad_token, self.eos_token, self.unk_token]
        for t in extra_tokens:
            if t not in ordered:
                ordered.append(t)
        self.tokens = ordered
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(self.tokens)}
        self.id_to_token: Dict[int, str] = {i: t for t, i in self.token_to_id.items()}
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
        self.unk_token_id = self.token_to_id[self.unk_token]
        self.model_max_length = model_max_length

    def __len__(self) -> int:
        return len(self.tokens)

    def tokenize(self, text: str) -> List[str]:
        found = re.findall(r"<[^>]+>", text)
        if found and "".join(found) == text:
            return found
        return [p for p in re.findall(r"<[^>]+>|[^<]+", text) if p]

    def convert_tokens_to_ids(self, tokens: Sequence[str]) -> List[int]:
        return [self.token_to_id.get(t, self.unk_token_id) for t in tokens]

    def encode(self, text: str, add_eos: bool = True) -> List[int]:
        ids = self.convert_tokens_to_ids(self.tokenize(text))
        if add_eos:
            ids = ids + [self.eos_token_id]
        return ids[: self.model_max_length]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        toks = []
        for i in ids:
            t = self.id_to_token.get(int(i), self.unk_token)
            if skip_special_tokens and t in {self.pad_token, self.eos_token, self.unk_token}:
                continue
            toks.append(t)
        return "".join(toks)

    def __call__(
        self,
        texts: Union[str, List[str]],
        return_tensors: Optional[str] = None,
        padding: str = "longest",
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_attention_mask: bool = True,
    ):
        if isinstance(texts, str):
            texts = [texts]
        max_length = max_length or self.model_max_length
        encoded = [self.encode(t, add_eos=True) for t in texts]
        if truncation:
            encoded = [e[:max_length] for e in encoded]
        max_len = max((len(e) for e in encoded), default=0)
        if padding != "longest":
            max_len = max_length
        input_ids, attention_mask = [], []
        for e in encoded:
            pad_n = max_len - len(e)
            input_ids.append(e + [self.pad_token_id] * pad_n)
            attention_mask.append([1] * len(e) + [0] * pad_n)
        out = {"input_ids": input_ids, "attention_mask": attention_mask}
        if return_tensors == "pt":
            out = {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}
        return out

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "semantic_tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"tokens": self.tokens, "model_max_length": self.model_max_length},
                f,
                indent=2,
                ensure_ascii=False,
            )

    @classmethod
    def from_pretrained(cls, path: str) -> "SemanticIdTokenizer":
        with open(os.path.join(path, "semantic_tokenizer.json"), "r", encoding="utf-8") as f:
            obj = json.load(f)
        specials = {cls.pad_token, cls.eos_token, cls.unk_token}
        extras = [t for t in obj["tokens"] if t not in specials]
        return cls(extras, model_max_length=obj.get("model_max_length", 256))
