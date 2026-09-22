"""XiaoLongtaoo/TIGER-aligned numeric Semantic-ID codec + dataset.

Upstream: https://github.com/XiaoLongtaoo/TIGER (MIT)
- Level-i token id = code + i * codebook_size + 1  (pad=0 → vocab=1025)
- History: left-pad to max_len items, each item → 4 codes, flatten
- No user-hash; eval matches raw 4-code tuples under beam search
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

_CODE_RE = re.compile(r"<[a-z]_(\d+)>", re.I)

XLT_VOCAB_SIZE = 1025
XLT_PAD_ID = 0
XLT_EOS_ID = 0

# Upstream README Beauty (Ours column).
XLT_BEAUTY_METRICS = {
    "recall@5": 0.0392,
    "ndcg@5": 0.0257,
    "recall@10": 0.0594,
    "ndcg@10": 0.0321,
}


def parse_level_codes(sid_tokens: Sequence[str], codebook_size: int = 256) -> List[int]:
    out = []
    for i, tok in enumerate(sid_tokens):
        m = _CODE_RE.fullmatch(str(tok).strip())
        if not m:
            raise ValueError(f"bad semantic token: {tok!r}")
        c = int(m.group(1))
        if not (0 <= c < codebook_size):
            raise ValueError(f"code {c} out of range for codebook_size={codebook_size}")
        out.append(c + i * codebook_size + 1)
    return out


def build_item_code_maps(
    indices_json: Dict,
    codebook_size: int = 256,
) -> Tuple[Dict[int, List[int]], Dict[Tuple[int, ...], int]]:
    item_to_code: Dict[int, List[int]] = {}
    code_to_item: Dict[Tuple[int, ...], int] = {}
    for k, tokens in indices_json.items():
        iid = int(k)
        codes = parse_level_codes(tokens, codebook_size=codebook_size)
        item_to_code[iid] = codes
        code_to_item[tuple(codes)] = iid
    return item_to_code, code_to_item


def pad_or_truncate_items(sequence: Sequence[int], max_len: int, pad_item: int = -1) -> List[int]:
    seq = list(sequence)
    if len(seq) > max_len:
        return seq[-max_len:]
    return [pad_item] * (max_len - len(seq)) + seq


def items_to_flat_codes(
    items: Sequence[int],
    item_to_code: Dict[int, List[int]],
    pad_item: int = -1,
    n_levels: int = 4,
) -> List[int]:
    flat: List[int] = []
    for x in items:
        if x == pad_item:
            flat.extend([0] * n_levels)
        else:
            flat.extend(item_to_code[int(x)])
    return flat


class XltSeqDataset(Dataset):
    """Leave-one-out dataset matching XiaoLongtaoo GenRecDataset behavior."""

    def __init__(
        self,
        inters_json: Dict,
        indices_json: Dict,
        max_his_len: int = 20,
        mode: str = "train",
        codebook_size: int = 256,
    ):
        super().__init__()
        self.max_his_len = max_his_len
        self.mode = mode
        self.codebook_size = codebook_size
        self.n_levels = 4
        self.pad_item = -1  # item id 0 is valid in our Beauty dump
        self.item_to_code, self.code_to_item = build_item_code_maps(
            indices_json, codebook_size=codebook_size
        )
        self.inters = {str(u): [int(i) for i in seq] for u, seq in inters_json.items()}
        self.samples = self._build_samples()

    def _build_samples(self):
        out = []
        for _uid, items in self.inters.items():
            if len(items) < 4:
                continue
            if self.mode == "train":
                seq = items[:-2]
                for i in range(1, len(seq)):
                    out.append({"history": seq[:i], "target": seq[i]})
            elif self.mode == "valid":
                out.append({"history": items[:-2], "target": items[-2]})
            elif self.mode == "test":
                out.append({"history": items[:-1], "target": items[-1]})
            else:
                raise ValueError(f"unsupported mode={self.mode}")
        return out

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        hist = pad_or_truncate_items(s["history"], self.max_his_len, self.pad_item)
        history = items_to_flat_codes(hist, self.item_to_code, self.pad_item, self.n_levels)
        target = self.item_to_code[int(s["target"])]
        return {"history": history, "target": target}

    def get_collate_fn(self):
        def collate_fn(batch):
            histories = torch.tensor([b["history"] for b in batch], dtype=torch.long)
            targets = torch.tensor([b["target"] for b in batch], dtype=torch.long)
            attention_mask = (histories != 0).long()
            return {
                "input_ids": histories,
                "attention_mask": attention_mask,
                "labels": targets,
            }

        return collate_fn
