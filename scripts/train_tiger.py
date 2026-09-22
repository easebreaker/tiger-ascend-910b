#!/usr/bin/env python3
"""Train / eval TIGER on CPU, CUDA, or Ascend NPU.

Supports a paper-faithful Beauty recipe (``--recipe paper``) for 910B measurement.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from itertools import cycle

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import T5Config, get_linear_schedule_with_warmup

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tiger_ascend.data.dataset import TigerSeqDataset, Trie, generate_toy_data, load_json  # noqa: E402
from tiger_ascend.data.tokenizer import SemanticIdTokenizer  # noqa: E402
from tiger_ascend.model.tiger import (  # noqa: E402
    TIGERModel,
    build_paper_t5_config,
    build_small_t5_config,
)
from tiger_ascend.utils.device import (  # noqa: E402
    amp_device_type,
    dataloader_kwargs,
    move_batch_to_device,
    resolve_device,
    synchronize,
)
from tiger_ascend.utils.metrics import (  # noqa: E402
    PAPER_BEAUTY_METRICS,
    format_vs_paper,
    ndcg_at_k,
    rank_of_gold,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["toy", "train", "eval", "all"], default="all")
    p.add_argument("--device", choices=["auto", "npu", "cuda", "cpu"], default="auto")
    p.add_argument("--data_dir", default="./artifacts/toy_data")
    p.add_argument("--output_dir", default="./artifacts/ckpt")
    p.add_argument(
        "--recipe",
        choices=["smoke", "paper"],
        default="smoke",
        help="smoke=small defaults; paper=TIGER Beauty recipe for measurement",
    )
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="optimizer updates; paper Beauty=200000 (overrides epochs when set)",
    )
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument(
        "--grad_accum",
        type=int,
        default=None,
        help="micro-batches per update (eff_batch = batch_size * grad_accum)",
    )
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--warmup_steps", type=int, default=None, help="paper: 10000")
    p.add_argument(
        "--lr_schedule",
        choices=["linear", "inv_sqrt"],
        default=None,
        help="paper: constant lr for warmup then inverse-sqrt decay",
    )
    p.add_argument("--max_his_len", type=int, default=None)
    p.add_argument(
        "--user_hash_size",
        type=int,
        default=None,
        help="paper: 2000 hashed user-id tokens prepended to history; 0 disables",
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--num_beams", type=int, default=None)
    p.add_argument("--eval_ks", type=str, default=None, help="e.g. 5,10 for paper table")
    p.add_argument("--print_samples", type=int, default=5)
    p.add_argument("--diag_batches", type=int, default=50)
    p.add_argument("--skip_tf_diag", action="store_true", help="skip teacher-forcing diag (faster eval)")
    p.add_argument("--log_every", type=int, default=None)
    p.add_argument("--save_every", type=int, default=None, help="checkpoint every N updates")
    p.add_argument("--toy_users", type=int, default=32)
    p.add_argument("--toy_items", type=int, default=64)
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--d_ff", type=int, default=None, help="FFN width")
    p.add_argument("--d_kv", type=int, default=None, help="per-head kv dim")
    p.add_argument("--num_layers", type=int, default=None)
    p.add_argument("--num_heads", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--amp", action="store_true", default=None)
    p.add_argument("--no_amp", action="store_true", help="disable AMP even for paper recipe")
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument(
        "--paper_size",
        action="store_true",
        help="deprecated alias: switch to --recipe paper architecture defaults",
    )
    return p.parse_args()


def _fill(args, name, value):
    if getattr(args, name) is None:
        setattr(args, name, value)


def apply_recipe(args):
    """Apply smoke or paper defaults; explicit CLI values win."""
    if args.paper_size and args.recipe == "smoke":
        args.recipe = "paper"

    if args.recipe == "paper":
        # Architecture: 4 layers, 6 heads × dim 64 ⇒ d_model=384, d_ff=1024 (~13–15M)
        _fill(args, "d_model", 384)
        _fill(args, "d_ff", 1024)
        _fill(args, "d_kv", 64)
        _fill(args, "num_layers", 4)
        _fill(args, "num_heads", 6)
        _fill(args, "dropout", 0.1)
        _fill(args, "max_his_len", 20)
        _fill(args, "user_hash_size", 2000)
        _fill(args, "max_steps", 200_000)
        _fill(args, "lr", 0.01)
        _fill(args, "warmup_steps", 10_000)
        _fill(args, "lr_schedule", "inv_sqrt")
        _fill(args, "batch_size", 64)
        _fill(args, "grad_accum", 4)
        _fill(args, "num_workers", 4)
        _fill(args, "num_beams", 20)
        _fill(args, "eval_ks", "5,10")
        _fill(args, "save_every", 20_000)
        _fill(args, "log_every", 200)
        if args.amp is None:
            args.amp = not args.no_amp
        if args.data_dir in {"./artifacts/toy_data", "artifacts/toy_data"}:
            args.data_dir = os.path.join(ROOT, "data", "amazon_beauty")
        if args.output_dir in {"./artifacts/ckpt", "artifacts/ckpt"}:
            args.output_dir = os.path.join(ROOT, "artifacts", "ckpt_beauty_paper")
        if not args.skip_tf_diag:
            # default on for paper measurement speed; user can omit by not using recipe
            args.skip_tf_diag = True
        print(
            "[recipe=paper] Beauty: steps=%d warmup=%d lr=%s schedule=%s "
            "eff_batch=%d amp=%s user_hash=%d beams=%d data=%s"
            % (
                args.max_steps,
                args.warmup_steps,
                args.lr,
                args.lr_schedule,
                args.batch_size * args.grad_accum,
                args.amp,
                args.user_hash_size,
                args.num_beams,
                args.data_dir,
            )
        )
    else:
        _fill(args, "d_model", 128)
        _fill(args, "d_ff", 0)  # resolved in build_model
        _fill(args, "d_kv", 0)
        _fill(args, "num_layers", 2)
        _fill(args, "num_heads", 4)
        _fill(args, "dropout", 0.1)
        _fill(args, "max_his_len", 20)
        _fill(args, "user_hash_size", 0)
        _fill(args, "max_steps", 0)
        _fill(args, "lr", 1e-3)
        _fill(args, "warmup_steps", 0)
        _fill(args, "lr_schedule", "linear")
        _fill(args, "batch_size", 8)
        _fill(args, "grad_accum", 1)
        _fill(args, "num_workers", 0)
        _fill(args, "num_beams", 4)
        _fill(args, "eval_ks", "1,5,10")
        _fill(args, "save_every", 0)
        _fill(args, "log_every", 200)
        if args.amp is None:
            args.amp = False
    if args.no_amp:
        args.amp = False


def inter_path(data_dir: str) -> str:
    return os.path.join(data_dir, "inter.json")


def indice_path(data_dir: str) -> str:
    return os.path.join(data_dir, "semantic_ids.json")


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


def _make_dataset(inters, indices, args, mode: str):
    return TigerSeqDataset(
        inters,
        indices,
        args.max_his_len,
        mode=mode,
        user_hash_size=args.user_hash_size,
    )


def build_model(train_ds: TigerSeqDataset, args):
    # History can be user_tok + 20*4 SID tokens + eos ≈ 82; keep headroom.
    tok = SemanticIdTokenizer(train_ds.get_new_tokens(), model_max_length=256)
    if args.recipe == "paper" or (
        args.d_model == 384 and args.num_layers == 4 and args.d_ff == 1024
    ):
        cfg = build_paper_t5_config(vocab_size=len(tok), dropout_rate=args.dropout)
    else:
        d_kv = args.d_kv or max(16, args.d_model // max(1, args.num_heads))
        d_ff = args.d_ff or (args.d_model * 2)
        cfg = build_small_t5_config(
            vocab_size=len(tok),
            d_model=args.d_model,
            d_ff=d_ff,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_kv=d_kv,
            dropout_rate=args.dropout,
        )
    cfg.pad_token_id = tok.pad_token_id
    cfg.eos_token_id = tok.eos_token_id
    cfg.decoder_start_token_id = tok.pad_token_id
    model = TIGERModel(cfg)
    model.set_hyper(args.temperature)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[model] params={n_params / 1e6:.2f}M vocab={len(tok)} "
        f"d_model={cfg.d_model} d_ff={cfg.d_ff} layers={cfg.num_layers} "
        f"heads={cfg.num_heads} d_kv={cfg.d_kv} user_hash={args.user_hash_size}"
    )
    return tok, model


def _make_grad_scaler(info, enabled: bool):
    if not enabled or info.kind not in {"cuda", "npu"}:
        return None
    try:
        return torch.amp.GradScaler(device=info.kind, enabled=True)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=info.kind == "cuda")


def _build_scheduler(optim, args, total_updates: int):
    warmup = max(0, int(args.warmup_steps))
    if args.lr_schedule == "inv_sqrt":
        # Paper: lr=0.01 for first 10k steps, then inverse-square-root decay.
        def lr_lambda(step: int):
            # LambdaLR calls with step starting at 0 after each optim.step().
            s = step + 1
            if s <= max(1, warmup):
                return 1.0
            return math.sqrt(warmup) / math.sqrt(s)

        return LambdaLR(optim, lr_lambda)
    return get_linear_schedule_with_warmup(optim, warmup, max(1, total_updates))


def _save_ckpt(tok, model, output_dir: str, step=None):
    os.makedirs(output_dir, exist_ok=True)
    tok.save_pretrained(output_dir)
    model.config.save_pretrained(output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    meta = {"step": step}
    with open(os.path.join(output_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    tag = f" step={step}" if step is not None else ""
    print(f"saved checkpoint -> {output_dir}{tag}")


def train_loop(args):
    info = resolve_device(args.device)
    print(f"[device] kind={info.kind} device={info.device} count={info.device_count}")
    inters = load_json(inter_path(args.data_dir))
    indices = load_json(indice_path(args.data_dir))
    _validate_json_alignment(inters, indices)
    train_ds = _make_dataset(inters, indices, args, "train")
    valid_ds = _make_dataset(inters, indices, args, "valid")
    if len(train_ds) == 0:
        raise RuntimeError(
            "train dataset is empty. Each user needs >=4 interactions "
            "(leave-one-out needs history for train/valid/test)."
        )
    tok, model = build_model(train_ds, args)
    model.to(info.device)

    use_amp = bool(args.amp and info.kind in {"cuda", "npu"})
    scaler = _make_grad_scaler(info, use_amp)
    grad_accum = max(1, int(args.grad_accum))
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_ds.get_collate_fn(tok),
        **dataloader_kwargs(info, num_workers=args.num_workers),
    )

    if args.max_steps > 0:
        total_updates = args.max_steps
    else:
        updates_per_epoch = max(1, (len(loader) + grad_accum - 1) // grad_accum)
        total_updates = max(1, updates_per_epoch * args.epochs)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = _build_scheduler(optim, args, total_updates)
    print(
        f"[train] samples={len(train_ds)} batches/epoch={len(loader)} "
        f"batch_size={args.batch_size} grad_accum={grad_accum} "
        f"eff_batch={args.batch_size * grad_accum} max_updates={total_updates} "
        f"amp={use_amp} num_workers={args.num_workers} schedule={args.lr_schedule}"
    )

    model.train()
    non_blocking = info.kind in {"cuda", "npu"}
    data_iter = cycle(loader)
    optim.zero_grad(set_to_none=True)
    running_t = torch.zeros((), device=info.device)
    micro = 0
    updates = 0
    t0 = time.time()

    while updates < total_updates:
        batch = move_batch_to_device(next(data_iter), info.device, non_blocking=non_blocking)
        if use_amp:
            with torch.autocast(device_type=amp_device_type(info), dtype=torch.float16):
                loss = model(**batch).loss / grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        else:
            loss = model(**batch).loss / grad_accum
            loss.backward()
        running_t = running_t + loss.detach()
        micro += 1

        if micro % grad_accum == 0:
            if scaler is not None:
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            updates += 1

            if updates % args.log_every == 0 or updates == 1 or updates == total_updates:
                synchronize(info)
                avg = float(running_t) / max(1, micro) * grad_accum
                lr_now = sched.get_last_lr()[0]
                sps = updates / max(1e-6, time.time() - t0)
                print(
                    f"step={updates}/{total_updates} train_loss={avg:.4f} "
                    f"lr={lr_now:.6g} updates/s={sps:.2f} valid_size={len(valid_ds)}"
                )
                running_t = torch.zeros((), device=info.device)
                micro = 0

            if args.save_every > 0 and updates % args.save_every == 0:
                _save_ckpt(tok, model, args.output_dir, step=updates)

    synchronize(info)
    _save_ckpt(tok, model, args.output_dir, step=updates)


def _ce_acc_lower_bound(token_acc: float) -> float:
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
        batch = move_batch_to_device(batch, device, non_blocking=True)
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
    train_ds = _make_dataset(inters, indices, args, "train")
    test_ds = _make_dataset(inters, indices, args, "test")
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

    ks = sorted({int(x) for x in args.eval_ks.split(",") if x.strip()})
    topk = max(ks) if ks else 1
    num_return = max(topk, args.num_beams)
    num_beams = max(args.num_beams, num_return)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=train_ds.get_collate_fn(tok),
        **dataloader_kwargs(info, num_workers=args.num_workers),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_ds.get_collate_fn(tok),
        **dataloader_kwargs(info, num_workers=args.num_workers),
    )

    if not args.skip_tf_diag:
        train_stats = _teacher_forcing_stats(
            model, train_loader, info.device, tok, "train", max_batches=args.diag_batches
        )
        test_stats = _teacher_forcing_stats(
            model, test_loader, info.device, tok, "test", max_batches=args.diag_batches
        )
        if train_stats["token_acc"] - test_stats["token_acc"] > 0.25:
            print("[diag] large train/test TF gap → overfitting / LOO hardness likely.")

    hits = {k: 0 for k in ks}
    ndcg_sums = {k: 0.0 for k in ks}
    total = 0
    printed = 0
    t0 = time.time()
    for batch in test_loader:
        labels = batch.pop("labels")
        bsz = labels.size(0)
        batch = move_batch_to_device(batch, info.device, non_blocking=True)
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
        gen = gen.view(bsz, num_return, -1)
        for i in range(bsz):
            gold_ids = [x for x in labels[i].tolist() if x != -100]
            gold = tok.decode(gold_ids)
            preds = [tok.decode(gen[i, j].tolist()) for j in range(num_return)]
            for k in ks:
                r = rank_of_gold(gold, preds[:k])
                hits[k] += int(r is not None)
                ndcg_sums[k] += ndcg_at_k(r, k)
            if printed < args.print_samples:
                print(f"[sample {printed}] gold={gold}")
                print(f"             pred@1={preds[0]}")
                if num_return > 1:
                    print(f"             pred@beam={preds[: min(3, num_return)]}")
                printed += 1
            total += 1

    metrics = {}
    for k in ks:
        recall = hits[k] / max(1, total)
        ndcg = ndcg_sums[k] / max(1, total)
        metrics[f"recall@{k}"] = recall
        metrics[f"ndcg@{k}"] = ndcg
        metrics[f"hit@{k}"] = recall
        print(f"recall@{k}(=hit@{k})={hits[k]}/{total} = {recall:.4f}")
        print(f"ndcg@{k}={ndcg:.4f}")
    if 1 in hits:
        print(f"exact_match(hit@1)={hits[1]}/{total} = {hits[1] / max(1, total):.4f}")

    print(f"[eval] users={total} elapsed_s={time.time() - t0:.1f} beams={num_beams}")
    if args.recipe == "paper" or "beauty" in os.path.basename(os.path.abspath(args.data_dir)):
        print("[vs paper Beauty Table 1]")
        for line in format_vs_paper(metrics, PAPER_BEAUTY_METRICS):
            print(" ", line)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "eval_metrics.json")
    payload = {
        "metrics": metrics,
        "n_test": total,
        "n_items": n_items,
        "num_beams": num_beams,
        "data_dir": args.data_dir,
        "paper_beauty_reference": PAPER_BEAUTY_METRICS,
        "recipe": args.recipe,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out_path}")


def main():
    args = parse_args()
    apply_recipe(args)
    if args.mode in {"toy", "all"}:
        if args.recipe == "paper":
            raise SystemExit(
                "Refusing --mode toy/all with --recipe paper (would overwrite Beauty data). "
                "Use --mode train or --mode eval."
            )
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
