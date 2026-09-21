import math
import os
import sys

import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from tiger_ascend.data.dataset import TigerSeqDataset, Trie, generate_toy_data, load_json
from tiger_ascend.data.tokenizer import SemanticIdTokenizer
from tiger_ascend.model.tiger import TIGERModel, build_small_t5_config
from tiger_ascend.utils.device import resolve_device


def test_cpu_forward_backward(tmp_path):
    inter = tmp_path / "inter.json"
    indice = tmp_path / "semantic_ids.json"
    generate_toy_data(str(inter), str(indice), num_users=8, num_items=16)
    inters = load_json(str(inter))
    indices = load_json(str(indice))
    ds = TigerSeqDataset(inters, indices, max_his_len=10, mode="train")
    tok = SemanticIdTokenizer(ds.get_new_tokens())
    cfg = build_small_t5_config(vocab_size=len(tok), d_model=64, d_ff=128, num_layers=1, num_heads=2)
    cfg.pad_token_id = tok.pad_token_id
    cfg.eos_token_id = tok.eos_token_id
    cfg.decoder_start_token_id = tok.pad_token_id
    model = TIGERModel(cfg)
    model.set_hyper(1.0)
    info = resolve_device("cpu")
    model.to(info.device)
    batch = ds.get_collate_fn(tok)([ds[0], ds[1]])
    batch = {k: v.to(info.device) for k, v in batch.items()}
    out = model(**batch)
    assert out.loss is not None
    out.loss.backward()
    assert torch.isfinite(out.loss.detach().cpu()).item()


def test_leave_one_out_targets_disjoint(tmp_path):
    inter = tmp_path / "inter.json"
    indice = tmp_path / "semantic_ids.json"
    generate_toy_data(str(inter), str(indice), num_users=16, num_items=32)
    inters = load_json(str(inter))
    indices = load_json(str(indice))
    train_ds = TigerSeqDataset(inters, indices, max_his_len=20, mode="train")
    test_ds = TigerSeqDataset(inters, indices, max_his_len=20, mode="test")
    # Per-user: test target is items[-1], never used as a train target (items[:-2] windows).
    for uid, items in train_ds.remapped_inters.items():
        train_targets = set(items[1:-2]) if len(items) > 3 else set()
        assert items[-1] not in train_targets
    assert len(test_ds) > 0
    assert len(train_ds) > 0


def test_ce_and_token_acc_same_split_consistent(tmp_path):
    """On one split, mean CE cannot be below (1-acc)*log(2)."""
    inter = tmp_path / "inter.json"
    indice = tmp_path / "semantic_ids.json"
    generate_toy_data(str(inter), str(indice), num_users=12, num_items=24)
    inters = load_json(str(inter))
    indices = load_json(str(indice))
    ds = TigerSeqDataset(inters, indices, max_his_len=10, mode="train")
    tok = SemanticIdTokenizer(ds.get_new_tokens())
    cfg = build_small_t5_config(vocab_size=len(tok), d_model=64, d_ff=128, num_layers=1, num_heads=2)
    cfg.pad_token_id = tok.pad_token_id
    cfg.eos_token_id = tok.eos_token_id
    cfg.decoder_start_token_id = tok.pad_token_id
    model = TIGERModel(cfg)
    model.set_hyper(1.0)
    loader = DataLoader(ds, batch_size=4, collate_fn=ds.get_collate_fn(tok))
    model.eval()
    tok_c = tok_t = 0
    loss_sum = 0.0
    steps = 0
    with torch.no_grad():
        for batch in loader:
            out = model(**batch)
            labels = batch["labels"]
            pred = out.logits.argmax(dim=-1)
            mask = labels != -100
            tok_c += int((pred[mask] == labels[mask]).sum().item())
            tok_t += int(mask.sum().item())
            loss_sum += float(out.loss)
            steps += 1
    acc = tok_c / max(1, tok_t)
    ce = loss_sum / max(1, steps)
    assert ce + 1e-4 >= (1.0 - acc) * math.log(2.0)


def test_generate_decode_matches_label_format(tmp_path):
    inter = tmp_path / "inter.json"
    indice = tmp_path / "semantic_ids.json"
    generate_toy_data(str(inter), str(indice), num_users=8, num_items=16)
    inters = load_json(str(inter))
    indices = load_json(str(indice))
    ds = TigerSeqDataset(inters, indices, max_his_len=10, mode="test")
    tok = SemanticIdTokenizer(ds.get_new_tokens())
    cfg = build_small_t5_config(vocab_size=len(tok), d_model=64, d_ff=128, num_layers=1, num_heads=2)
    cfg.pad_token_id = tok.pad_token_id
    cfg.eos_token_id = tok.eos_token_id
    cfg.decoder_start_token_id = tok.pad_token_id
    model = TIGERModel(cfg)
    model.eval()
    batch = ds.get_collate_fn(tok)([ds[0]])
    labels = batch.pop("labels")
    cands = [tok.encode(s, add_eos=True) for s in ds.get_all_items(as_list=True)]
    prefix = Trie(cands).prefix_allowed_tokens_fn(bos_token_id=tok.pad_token_id)
    with torch.no_grad():
        gen = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=8,
            num_beams=4,
            num_return_sequences=4,
            prefix_allowed_tokens_fn=prefix,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    gold = tok.decode([x for x in labels[0].tolist() if x != -100])
    preds = [tok.decode(g.tolist()) for g in gen]
    assert gold.startswith("<") and ">" in gold
    assert all(p.startswith("<") for p in preds)
    assert all(p in set(ds.get_all_items(as_list=True)) for p in preds)
