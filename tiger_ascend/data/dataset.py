"""Dataset + constrained decoding helpers for TIGER.

Adapted from torch-rechub ``TigerSeqDataset`` / ``Trie`` utilities.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
from torch.utils.data import Dataset


class TigerSeqDataset(Dataset):
    """Map user item histories to semantic-ID string sequences."""

    def __init__(
        self,
        inters_json: Dict,
        indices_json: Dict,
        max_his_len: int,
        mode: str = "train",
        sample_num: int = 0,
    ):
        super().__init__()
        self.max_his_len = max_his_len
        self.sample_num = sample_num
        self.mode = mode
        self._new_tokens: Optional[List[str]] = None
        self._all_items = None
        self.inters = inters_json
        self.indices = indices_json
        self._remap_items()
        self.inter_data = self._process_data()

    def _remap_items(self) -> None:
        self.remapped_inters = {}
        for uid, items in self.inters.items():
            self.remapped_inters[uid] = ["".join(self.indices[str(i)]) for i in items]

    def _process_data(self):
        if self.mode == "train":
            return self._process_train_data()
        if self.mode == "valid":
            return self._process_valid_data()
        if self.mode == "test":
            return self._process_test_data()
        raise ValueError(f"unsupported mode={self.mode}")

    def _truncate_history(self, history):
        if self.max_his_len > 0:
            return history[-self.max_his_len :]
        return history

    def _process_train_data(self):
        inter_data = []
        for items in self.remapped_inters.values():
            items = items[:-2]
            for i in range(1, len(items)):
                history = self._truncate_history(items[:i])
                inter_data.append({"item": items[i], "inters": "".join(history)})
        return inter_data

    def _process_valid_data(self):
        inter_data = []
        for items in self.remapped_inters.values():
            history = self._truncate_history(items[:-2])
            inter_data.append({"item": items[-2], "inters": "".join(history)})
        return inter_data

    def _process_test_data(self):
        inter_data = []
        for items in self.remapped_inters.values():
            history = self._truncate_history(items[:-1])
            inter_data.append({"item": items[-1], "inters": "".join(history)})
        if self.sample_num > 0:
            all_idx = np.arange(len(inter_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()
        return inter_data

    def get_new_tokens(self) -> List[str]:
        if self._new_tokens is not None:
            return self._new_tokens
        tokens: Set[str] = set()
        for index in self.indices.values():
            tokens.update(index)
        self._new_tokens = sorted(tokens)
        return self._new_tokens

    def get_all_items(self, as_list: bool = False):
        if self._all_items is None:
            self._all_items = ["".join(index) for index in self.indices.values()]
        return list(self._all_items) if as_list else set(self._all_items)

    def get_collate_fn(self, tokenizer):
        def collate_fn(batch):
            input_texts = [d["input_ids"] for d in batch]
            label_texts = [d["labels"] for d in batch]
            inputs = tokenizer(
                input_texts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=getattr(tokenizer, "model_max_length", 512),
                return_attention_mask=True,
            )
            labels = tokenizer(
                label_texts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=getattr(tokenizer, "model_max_length", 512),
                return_attention_mask=True,
            )
            inputs["labels"] = labels["input_ids"].clone()
            inputs["labels"][inputs["labels"] == tokenizer.pad_token_id] = -100
            return inputs

        return collate_fn

    def __len__(self):
        return len(self.inter_data)

    def __getitem__(self, index):
        d = self.inter_data[index % len(self.inter_data)]
        return {"input_ids": d["inters"], "labels": d["item"]}


class Trie:
    """Prefix trie for constrained beam search over legal semantic IDs."""

    def __init__(self, sequences: Optional[Iterable[Sequence[int]]] = None):
        self.trie_dict = {}
        self.len = 0
        self.append_trie = None
        self.bos_token_id = None
        if sequences:
            for sequence in sequences:
                self.add(list(sequence))

    def add(self, sequence: Sequence[int]) -> None:
        Trie._add_to_trie(list(sequence), self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: Sequence[int]):
        return Trie._get_from_trie(
            list(prefix_sequence), self.trie_dict, self.append_trie, self.bos_token_id
        )

    def prefix_allowed_tokens_fn(self, bos_token_id: int = 0):
        """HF generate passes decoder ids that usually start with ``bos_token_id``."""

        def _fn(batch_id, sentence):
            seq = sentence.tolist()
            if seq and seq[0] == bos_token_id:
                seq = seq[1:]
            out = self.get(seq)
            return out if out else []

        return _fn

    @staticmethod
    def _add_to_trie(sequence, trie_dict):
        if not sequence:
            return
        if sequence[0] not in trie_dict:
            trie_dict[sequence[0]] = {}
        Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(prefix_sequence, trie_dict, append_trie, bos_token_id):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        if prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:], trie_dict[prefix_sequence[0]], append_trie, bos_token_id
            )
        if append_trie:
            return append_trie.get(prefix_sequence)
        return []


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_toy_data(
    inter_path: str,
    indice_path: str,
    num_users: int = 32,
    num_items: int = 64,
    seed: int = 42,
    codebook_size: int = 8,
) -> None:
    rng = random.Random(seed)
    index_data = {}
    for item_id in range(1, num_items + 1):
        idx = item_id - 1
        a = idx // (codebook_size * codebook_size)
        b = (idx // codebook_size) % codebook_size
        c = idx % codebook_size
        index_data[str(item_id)] = [f"<a_{a}>", f"<b_{b}>", f"<c_{c}>"]

    inter_data = {}
    for user_id in range(num_users):
        length = rng.randint(6, 12)
        start = rng.randint(1, num_items)
        inter_data[str(user_id)] = [
            (start - 1 + step) % num_items + 1 for step in range(length)
        ]

    write_json(inter_path, inter_data)
    write_json(indice_path, index_data)
