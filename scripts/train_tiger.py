#!/usr/bin/env python3
"""Train / eval TIGER toy workflow on CPU, CUDA, or Ascend NPU."""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import T5Config, get_linear_schedule_with_warmup

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tiger_ascend.data.dataset import TigerSeqDataset, Trie, generate_toy_data, load_json  # noqa: E402
from tiger_ascend.data.tokenizer import SemanticIdTokenizer  # noqa: E402
from tiger_ascend.model.tiger import TIGERModel, build_small_t5_config  # noqa: E402
from tiger_ascend.utils.device import (  # noqa: E402
    amp_device_type,
    dataloader_kwargs,
    resolve_device,
    synchronize,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["toy", "train", "eval", "all"], default="all")
    p.add_argument("--device", choices=["auto", "npu", "cuda", "cpu"], default="auto")
    p.add_argument("--data_dir", default="./artifacts/toy_data")
    p.add_argument("--output_dir", default="./artifacts/ckpt")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_his_len", type=int, default=20)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--eval_ks", type=str, default="1,5,10", help="Hit@K list, e.g. 1,5,10")
    p.add_argument("--print_samples", type=int, default=5, help="print N pred/gold pairs during eval")
    p.add_argument(
        "--diag_batches",
        type=int,
        default=50,
        help="max batches for train/test teacher-forcing diagnostics",
    )
    p.add_argument("--toy_users", type=int, default=32)
    p.add_argument("--toy_items", type=int, default=64)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def inter_path(data_dir: str) -> str:
    return os.path.join(data_dir, "inter.json")


def indice_path(data_dir: str) -> str:
    return os.path.join(data_dir, "semantic_ids.json")


def build_model(train_ds: TigerSeqDataset, args):
    tok = SemanticIdTokenizer(train_ds.get_new_tokens(), model_max_length=256)
    cfg = build_small_t5_config(
        vocab_size=len(tok),
        d_model=args.d_model,
        d_ff=args.d_model * 2,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    cfg.pad_token_id = tok.pad_token_id
    cfg.eos_token_id = tok.eos_token_id
    cfg.decoder_start_token_id = tok.pad_token_id
    model = TIGERModel(cfg)
    model.set_hyper(args.temperature)
    return tok, model


def _validate_json_alignment(inters, indices):
    missing = sorted(
        {
            str(item)
            for seq in inters.values()
            for item in seq
            if str(item) not in indices
        }
    )
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"semantic_ids.json missing {len(missing)} item ids referenced by inter.json "
            f"(e.g. {preview}). Keys must match."
        )


def train_loop(args):
    info = resolve_device(args.device)
    print(f"[device] kind={info.kind} device={info.device} count={info.device_count}")
    inters = load_json(inter_path(args.data_dir))
    indices = load_json(indice_path(args.data_dir))
    _validate_json_alignment(inters, indices)
    train_ds = TigerSeqDataset(inters, indices, args.max_his_len, mode="train")
    valid_ds = TigerSeqDataset(inters, indices, args.max_his_len, mode="valid")
    if len(train_ds) == 0:
        raise RuntimeError(
            "train dataset is empty. Each user needs >=4 interactions "
            "(leave-one-out needs history for train/valid/test)."
        )
    tok, model = build_model(train_ds, args)
    model.to(info.device)

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_ds.get_collate_fn(tok),
        **dataloader_kwargs(info),
    )
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, len(loader) * args.epochs)
    sched = get_linear_schedule_with_warmup(optim, int(0.05 * total_steps), total_steps)

    model.train()
    for epoch in range(args.epochs):
        running, steps = 0.0, 0
        for batch in loader:
            batch = {k: v.to(info.device) for k, v in batch.items()}
            optim.zero_grad(set_to_none=True)
            if args.amp and info.kind in {"cuda", "npu"}:
                with torch.autocast(device_type=amp_device_type(info), dtype=torch.float16):
                    loss = model(**batch).loss
                loss.backward()
            else:
                loss = model(**batch).loss
                loss.backward()
            optim.step()
            sched.step()
            running += float(loss.detach().cpu())
            steps += 1
        synchronize(info)
        print(
            f"epoch={epoch + 1} train_loss={running / max(1, steps):.4f} "
            f"valid_size={len(valid_ds)}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    tok.save_pretrained(args.output_dir)
    model.config.save_pretrained(args.output_dir)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
    print(f"saved checkpoint -> {args.output_dir}")


def _ce_acc_lower_bound(token_acc: float) -> float:
    """Min possible mean CE if wrong tokens have p_gold < 0.5 (argmax wrong)."""
    import math

    return max(0.0, (1.0 - token_acc) * math.log(2.0))


@torch.no_grad()
def _teacher_forcing_stats(
    model,
    loader,
    device,
    tok,
    split_name: str,
    max_batches: int = 50,
    print_ok: int = 2,
):
    """Teacher-forcing CE / token-acc / per-position acc on one split.

    Low train CE with low test TF acc ⇒ overfitting / LOO hardness, not a CE bug.
    CE below ``_ce_acc_lower_bound(token_acc)`` on the *same* split ⇒ metric bug.
    """
    token_correct = token_total = 0
    seq_correct = seq_total = 0
    loss_sum = 0.0
    steps = 0
    pos_correct: list[int] = []
    pos_total: list[int] = []
    shown = 0
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]
        out = model(**batch)
        loss_sum += float(out.loss.detach().cpu())
        steps += 1
        pred = out.logits.argmax(dim=-1)
        mask = labels != -100
        token_correct += int((pred[mask] == labels[mask]).sum().item())
        token_total += int(mask.sum().item())
        for i in range(labels.size(0)):
            positions = mask[i].nonzero(as_tuple=True)[0]
            if positions.numel() == 0:
                continue
            seq_total += 1
            gold_toks = labels[i][positions]
            pred_toks = pred[i][positions]
            if bool(torch.equal(pred_toks, gold_toks)):
                seq_correct += 1
                if shown < print_ok:
                    gold = tok.decode(gold_toks.tolist())
                    hyp = tok.decode(pred_toks.tolist())
                    print(f"[tf-ok/{split_name}] gold={gold} tf_pred={hyp}")
                    shown += 1
            for pi, pos in enumerate(positions.tolist()):
                while pi >= len(pos_correct):
                    pos_correct.append(0)
                    pos_total.append(0)
                pos_total[pi] += 1
                pos_correct[pi] += int(pred[i, pos].item() == labels[i, pos].item())

    token_acc = token_correct / max(1, token_total)
    ce = loss_sum / max(1, steps)
    lb = _ce_acc_lower_bound(token_acc)
    print(
        f"[diag/{split_name}] teacher_forcing "
        f"ce={ce:.4f} token_acc={token_acc:.4f} "
        f"seq_exact={seq_correct}/{seq_total}="
        f"{seq_correct / max(1, seq_total):.4f} "
        f"ce_lb(acc)≈{lb:.4f}"
    )
    if steps and ce + 1e-4 < lb:
        print(
            f"[diag/{split_name}] WARNING: ce < lower-bound from token_acc "
            f"on the SAME split — check loss/acc alignment."
        )
    pos_parts = []
    for i, (c, t) in enumerate(zip(pos_correct, pos_total)):
        if t:
            pos_parts.append(f"p{i}={c / t:.3f}")
    if pos_parts:
        print(f"[diag/{split_name}] per_pos_acc " + " ".join(pos_parts))
    return {"ce": ce, "token_acc": token_acc, "seq_exact": seq_correct / max(1, seq_total)}


@torch.no_grad()
def eval_loop(args):
    info = resolve_device(args.device)
    inters = load_json(inter_path(args.data_dir))
    indices = load_json(indice_path(args.data_dir))
    train_ds = TigerSeqDataset(inters, indices, args.max_his_len, mode="train")
    test_ds = TigerSeqDataset(inters, indices, args.max_his_len, mode="test")
    tok = SemanticIdTokenizer.from_pretrained(args.output_dir)
    cfg = T5Config.from_pretrained(args.output_dir)
    model = TIGERModel(cfg)
    state = torch.load(os.path.join(args.output_dir, "pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(state)
    model.to(info.device)
    model.eval()

    candidate_ids = [tok.encode(sid, add_eos=True) for sid in test_ds.get_all_items(as_list=True)]
    prefix_fn = Trie(candidate_ids).prefix_allowed_tokens_fn(bos_token_id=tok.pad_token_id)
    n_items = len(candidate_ids)
    print(
        f"[diag] n_train={len(train_ds)} n_test={len(test_ds)} "
        f"n_items_in_trie={n_items} random_hit@10≈{10 / max(1, n_items):.4f}"
    )
    print(
        "[diag] leave-one-out: train predicts items[:-2] windows; "
        "test target is items[-1] (never a train target for that user)."
    )

    ks = sorted({int(x) for x in args.eval_ks.split(",") if x.strip()})
    topk = max(ks) if ks else 1
    num_return = max(topk, args.num_beams)
    num_beams = max(args.num_beams, num_return)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=train_ds.get_collate_fn(tok),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_ds.get_collate_fn(tok),
    )
    # Compare splits: train loss≈train TF ce; test TF acc<<train ⇒ overfit / LOO, not CE bug.
    train_stats = _teacher_forcing_stats(
        model, train_loader, info.device, tok, "train", max_batches=args.diag_batches
    )
    test_stats = _teacher_forcing_stats(
        model, test_loader, info.device, tok, "test", max_batches=args.diag_batches
    )
    if train_stats["token_acc"] - test_stats["token_acc"] > 0.25:
        print(
            "[diag] large train/test TF gap → model memorizes train next-SID; "
            "Hit@K on held-out last item can stay near random. Code path OK."
        )
    if test_stats["token_acc"] < 0.55:
        print(
            "[diag] test TF token_acc is modest; autoregressive Hit@K will be much "
            "lower (errors compound over a/b/c/d)."
        )

    hits = {k: 0 for k in ks}
    total = 0
    printed = 0
    for batch in test_loader:
        labels = batch.pop("labels")
        bsz = labels.size(0)
        batch = {k: v.to(info.device) for k, v in batch.items()}
        gen = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=8,
            num_beams=num_beams,
            num_return_sequences=num_return,
            prefix_allowed_tokens_fn=prefix_fn,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        # generate returns [bsz * num_return, seq]; usually starts with decoder_start (pad)
        gen = gen.view(bsz, num_return, -1)
        for i in range(bsz):
            gold_ids = [x for x in labels[i].tolist() if x != -100]
            gold = tok.decode(gold_ids)
            preds = [tok.decode(gen[i, j].tolist()) for j in range(num_return)]
            for k in ks:
                hits[k] += int(gold in preds[:k])
            if printed < args.print_samples:
                print(f"[sample {printed}] gold={gold}")
                print(f"             pred@1={preds[0]}")
                if num_return > 1:
                    print(f"             pred@beam={preds[: min(3, num_return)]}")
                printed += 1
            total += 1

    for k in ks:
        print(f"hit@{k}={hits[k]}/{total} = {hits[k] / max(1, total):.4f}")
    # keep old name for top-1 exact match
    if 1 in hits:
        print(f"exact_match(hit@1)={hits[1]}/{total} = {hits[1] / max(1, total):.4f}")


def main():
    args = parse_args()
    if args.mode in {"toy", "all"}:
        generate_toy_data(
            inter_path(args.data_dir),
            indice_path(args.data_dir),
            num_users=args.toy_users,
            num_items=args.toy_items,
        )
        print(f"toy data -> {args.data_dir}")
    if args.mode in {"train", "all"}:
        train_loop(args)
    if args.mode in {"eval", "all"}:
        eval_loop(args)


if __name__ == "__main__":
    main()
