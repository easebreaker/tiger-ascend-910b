#!/usr/bin/env python3
"""Convert NonameUntitled/tiger Amazon index files into tiger-ascend format.

Expected inputs (per domain directory):
  inter.json
  index_rqvae.json   # or pass --index_file

Outputs:
  inter.json (filtered, users with >=4 interactions)
  semantic_ids.json  # ["<a_x>", "<b_y>", "<c_z>", "<d_w>", ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def codes_to_tokens(codes, levels):
    if len(codes) > len(levels):
        raise ValueError(f"need more level prefixes for {len(codes)} codes")
    return [f"<{levels[i]}_{int(c)}>" for i, c in enumerate(codes)]


def convert(src_dir: Path, dst_dir: Path, index_file: str, min_seq_len: int) -> None:
    levels = [chr(ord("a") + i) for i in range(26)]
    with open(src_dir / "inter.json", encoding="utf-8") as f:
        inter = json.load(f)
    with open(src_dir / index_file, encoding="utf-8") as f:
        index = json.load(f)

    semantic = {str(k): codes_to_tokens(v, levels) for k, v in index.items()}
    clean = {}
    missing = set()
    for uid, seq in inter.items():
        seq = [int(x) for x in seq]
        if len(seq) < min_seq_len:
            continue
        for item in seq:
            if str(item) not in semantic:
                missing.add(item)
        clean[str(uid)] = seq
    if missing:
        raise SystemExit(f"semantic index missing {len(missing)} items, e.g. {sorted(missing)[:10]}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    with open(dst_dir / "inter.json", "w", encoding="utf-8") as f:
        json.dump(clean, f, separators=(",", ":"))
    with open(dst_dir / "semantic_ids.json", "w", encoding="utf-8") as f:
        json.dump(semantic, f, indent=2)
    print(f"wrote {dst_dir}: users={len(clean)} items={len(semantic)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir", required=True, help="e.g. path/to/Beauty")
    p.add_argument("--dst_dir", required=True, help="e.g. data/amazon_beauty")
    p.add_argument("--index_file", default="index_rqvae.json")
    p.add_argument("--min_seq_len", type=int, default=4)
    args = p.parse_args()
    convert(Path(args.src_dir), Path(args.dst_dir), args.index_file, args.min_seq_len)


if __name__ == "__main__":
    main()
