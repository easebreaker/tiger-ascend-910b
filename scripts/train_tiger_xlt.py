#!/usr/bin/env python3
"""Train/eval TIGER aligned with XiaoLongtaoo/TIGER for 910B measurement.

Upstream: https://github.com/XiaoLongtaoo/TIGER (MIT)
Beauty README (Ours): R@5=0.0392 N@5=0.0257 R@10=0.0594 N@10=0.0321
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tiger_ascend.data.dataset import load_json  # noqa: E402
from tiger_ascend.data.xlt import (  # noqa: E402
    XLT_BEAUTY_METRICS,
    XLT_EOS_ID,
    XLT_PAD_ID,
    XLT_VOCAB_SIZE,
    XltSeqDataset,
)
from tiger_ascend.model.tiger import build_xlt_t5_config  # noqa: E402
from tiger_ascend.utils.device import (  # noqa: E402
    amp_device_type,
    dataloader_kwargs,
    move_batch_to_device,
    resolve_device,
    synchronize,
)
from tiger_ascend.utils.metrics import PAPER_BEAUTY_METRICS, format_vs_paper  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="XiaoLongtaoo/TIGER-aligned Beauty trainer")
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--device", choices=["auto", "npu", "cuda", "cpu"], default="auto")
    p.add_argument("--data_dir", default=os.path.join(ROOT, "data", "amazon_beauty"))
    p.add_argument("--output_dir", default=os.path.join(ROOT, "artifacts", "ckpt_beauty_xlt"))
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--infer_batch_size", type=int, default=96)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_his_len", type=int, default=20)
    p.add_argument("--beam_size", type=int, default=30)
    p.add_argument("--topk", type=str, default="5,10,20")
    p.add_argument("--early_stop", type=int, default=10)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers; keep 0 on Ascend NPU (workers>0 often Aborted)",
    )
    p.add_argument("--amp", action="store_true", help="optional; upstream is FP32")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--max_train_steps", type=int, default=0, help=">0 for smoke")
    p.add_argument("--skip_valid", action="store_true")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def inter_path(d):
    return os.path.join(d, "inter.json")


def indice_path(d):
    return os.path.join(d, "semantic_ids.json")


def build_model():
    cfg = build_xlt_t5_config(vocab_size=XLT_VOCAB_SIZE, dropout_rate=0.1)
    model = T5ForConditionalGeneration(cfg)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[xlt-model] params={n / 1e6:.2f}M vocab={cfg.vocab_size} "
        f"d_model={cfg.d_model} d_ff={cfg.d_ff} layers={cfg.num_layers} "
        f"heads={cfg.num_heads} d_kv={cfg.d_kv}"
    )
    return model


def _make_loader(ds, batch_size, shuffle, info, num_workers):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=ds.get_collate_fn(),
        **dataloader_kwargs(info, num_workers=num_workers),
    )


def calculate_pos_index(preds: torch.Tensor, labels: torch.Tensor, maxk: int) -> torch.Tensor:
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    bsz = preds.size(0)
    pos = torch.zeros((bsz, maxk), dtype=torch.bool)
    for i in range(bsz):
        gold = labels[i].tolist()
        for j in range(maxk):
            if preds[i, j].tolist() == gold:
                pos[i, j] = True
                break
    return pos


def recall_at_k(pos_index: torch.Tensor, k: int) -> torch.Tensor:
    return pos_index[:, :k].any(dim=1).float()


def ndcg_at_k(pos_index: torch.Tensor, k: int) -> torch.Tensor:
    ranks = torch.arange(1, pos_index.size(-1) + 1)
    dcg = 1.0 / torch.log2(ranks + 1.0)
    dcg = torch.where(pos_index, dcg, torch.zeros_like(dcg))
    return dcg[:, :k].sum(dim=1)


@torch.no_grad()
def evaluate(model, loader, topk_list, beam_size, device, info, use_amp: bool):
    model.eval()
    sums = {f"recall@{k}": 0.0 for k in topk_list}
    sums.update({f"ndcg@{k}": 0.0 for k in topk_list})
    n = 0
    for batch in loader:
        labels = batch["labels"]
        bsz = labels.size(0)
        batch_dev = move_batch_to_device(
            {k: v for k, v in batch.items() if k != "labels"},
            device,
            non_blocking=True,
        )
        gen_kwargs = dict(
            input_ids=batch_dev["input_ids"],
            attention_mask=batch_dev["attention_mask"],
            max_length=5,
            num_beams=beam_size,
            num_return_sequences=beam_size,
            eos_token_id=XLT_EOS_ID,
            pad_token_id=XLT_PAD_ID,
            decoder_start_token_id=XLT_PAD_ID,
        )
        if use_amp:
            with torch.autocast(device_type=amp_device_type(info), dtype=torch.float16):
                gen = model.generate(**gen_kwargs)
        else:
            gen = model.generate(**gen_kwargs)
        gen = gen[:, 1:]
        if gen.size(-1) < 4:
            pad = torch.zeros(gen.size(0), 4 - gen.size(-1), dtype=gen.dtype, device=gen.device)
            gen = torch.cat([gen, pad], dim=-1)
        gen = gen[:, :4].view(bsz, beam_size, 4)
        pos = calculate_pos_index(gen, labels, maxk=beam_size)
        for k in topk_list:
            sums[f"recall@{k}"] += float(recall_at_k(pos, k).sum())
            sums[f"ndcg@{k}"] += float(ndcg_at_k(pos, k).sum())
        n += bsz
    return {k: v / max(1, n) for k, v in sums.items()}, n


def _write_metrics(args, metrics, n_test, tag="eval"):
    payload = {
        "tag": tag,
        "metrics": metrics,
        "n_test": n_test,
        "beam_size": args.beam_size,
        "data_dir": args.data_dir,
        "xlt_beauty_reference": XLT_BEAUTY_METRICS,
        "paper_beauty_reference": PAPER_BEAUTY_METRICS,
        "recipe": "xlt",
        "upstream": "https://github.com/XiaoLongtaoo/TIGER",
    }
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "eval_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {path}")
    print("[vs XiaoLongtaoo Beauty]")
    for line in format_vs_paper(metrics, XLT_BEAUTY_METRICS):
        print(" ", line)
    print("[vs paper Beauty Table 1]")
    for line in format_vs_paper(metrics, PAPER_BEAUTY_METRICS):
        print(" ", line)


def train_loop(args):
    info = resolve_device(args.device)
    print(f"[device] kind={info.kind} device={info.device}")
    set_seed(args.seed)
    inters = load_json(inter_path(args.data_dir))
    indices = load_json(indice_path(args.data_dir))
    train_ds = XltSeqDataset(inters, indices, args.max_his_len, mode="train")
    valid_ds = XltSeqDataset(inters, indices, args.max_his_len, mode="valid")
    test_ds = XltSeqDataset(inters, indices, args.max_his_len, mode="test")
    print(f"[data] train={len(train_ds)} valid={len(valid_ds)} test={len(test_ds)}")

    model = build_model()
    model.to(info.device)
    use_amp = bool(args.amp and not args.no_amp and info.kind in {"cuda", "npu"})
    grad_accum = max(1, args.grad_accum)
    if args.batch_size * grad_accum < 256:
        print(
            f"[warn] eff_batch={args.batch_size * grad_accum} < 256 "
            f"(upstream default); raise --batch_size/--grad_accum if possible"
        )

    loader = _make_loader(train_ds, args.batch_size, True, info, args.num_workers)
    valid_loader = _make_loader(valid_ds, args.infer_batch_size, False, info, args.num_workers)
    test_loader = _make_loader(test_ds, args.infer_batch_size, False, info, args.num_workers)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    topk_list = [int(x) for x in args.topk.split(",") if x.strip()]
    beam = max(args.beam_size, max(topk_list))

    best_ndcg = -1.0
    patience = 0
    global_step = 0
    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, "pytorch_model.bin")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        micro = 0
        optim.zero_grad(set_to_none=True)
        t0 = time.time()
        for step_i, batch in enumerate(loader, start=1):
            batch = move_batch_to_device(batch, info.device, non_blocking=True)
            if use_amp:
                with torch.autocast(device_type=amp_device_type(info), dtype=torch.float16):
                    loss = model(**batch).loss / grad_accum
                loss.backward()
            else:
                loss = model(**batch).loss / grad_accum
                loss.backward()
            running += float(loss.detach()) * grad_accum
            micro += 1
            if step_i % grad_accum == 0:
                optim.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.log_every == 0:
                    print(f"epoch={epoch} step={global_step} loss={running / max(1, micro):.4f}")
                if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                    break
        synchronize(info)
        print(
            f"epoch={epoch}/{args.epochs} train_loss={running / max(1, micro):.4f} "
            f"sec={time.time() - t0:.1f}"
        )

        if args.skip_valid:
            torch.save(model.state_dict(), best_path)
            model.config.save_pretrained(args.output_dir)
            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                break
            continue

        val_metrics, _ = evaluate(
            model, valid_loader, topk_list, beam, info.device, info, use_amp
        )
        print("[valid] " + " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
        score = val_metrics.get("ndcg@20", val_metrics.get("ndcg@10", 0.0))
        if score > best_ndcg:
            best_ndcg = score
            patience = 0
            torch.save(model.state_dict(), best_path)
            model.config.save_pretrained(args.output_dir)
            test_metrics, n_test = evaluate(
                model, test_loader, topk_list, beam, info.device, info, use_amp
            )
            print(
                f"[test@best] n={n_test} "
                + " ".join(f"{k}={v:.4f}" for k, v in test_metrics.items())
            )
            _write_metrics(args, test_metrics, n_test, tag="test_at_best")
        else:
            patience += 1
            print(
                f"[early] no improve (best_ndcg={best_ndcg:.4f}) "
                f"patience={patience}/{args.early_stop}"
            )
            if patience >= args.early_stop:
                print("[early] stop")
                break
        if args.max_train_steps > 0 and global_step >= args.max_train_steps:
            break

    if not os.path.isfile(best_path):
        torch.save(model.state_dict(), best_path)
        model.config.save_pretrained(args.output_dir)
    print(f"saved -> {args.output_dir}")


def eval_loop(args):
    info = resolve_device(args.device)
    inters = load_json(inter_path(args.data_dir))
    indices = load_json(indice_path(args.data_dir))
    test_ds = XltSeqDataset(inters, indices, args.max_his_len, mode="test")
    model = build_model()
    state = torch.load(os.path.join(args.output_dir, "pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(state)
    model.to(info.device)
    use_amp = bool(args.amp and not args.no_amp and info.kind in {"cuda", "npu"})
    topk_list = [int(x) for x in args.topk.split(",") if x.strip()]
    beam = max(args.beam_size, max(topk_list))
    loader = _make_loader(test_ds, args.infer_batch_size, False, info, args.num_workers)
    t0 = time.time()
    metrics, n = evaluate(model, loader, topk_list, beam, info.device, info, use_amp)
    print(f"[eval] n={n} sec={time.time() - t0:.1f}")
    for k, v in metrics.items():
        print(f"{k}={v:.4f}")
    _write_metrics(args, metrics, n, tag="eval")


def main():
    args = parse_args()
    if args.mode == "train":
        train_loop(args)
    else:
        eval_loop(args)


if __name__ == "__main__":
    main()
