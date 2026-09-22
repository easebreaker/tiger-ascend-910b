import math
import os
import sys

import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from tiger_ascend.data.dataset import TigerSeqDataset, Trie, generate_toy_data, hash_user_token, load_json
from tiger_ascend.data.tokenizer import SemanticIdTokenizer
from tiger_ascend.model.tiger import TIGERModel, build_paper_t5_config, build_small_t5_config
from tiger_ascend.utils.device import resolve_device
from tiger_ascend.utils.metrics import PAPER_BEAUTY_METRICS, ndcg_at_k, summarize_ranking


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
    for uid, items in train_ds.remapped_inters.items():
        train_targets = set(items[1:-2]) if len(items) > 3 else set()
        assert items[-1] not in train_targets
    assert len(test_ds) > 0
    assert len(train_ds) > 0


def test_user_hash_prefix(tmp_path):
    inter = tmp_path / "inter.json"
    indice = tmp_path / "semantic_ids.json"
    generate_toy_data(str(inter), str(indice), num_users=4, num_items=16)
    inters = load_json(str(inter))
    indices = load_json(str(indice))
    ds = TigerSeqDataset(inters, indices, max_his_len=10, mode="train", user_hash_size=2000)
    sample = ds[0]["input_ids"]
    assert sample.startswith("<u_")
    assert any(t.startswith("<u_") for t in ds.get_new_tokens())
    assert hash_user_token("7", 2000).startswith("<u_")


def test_ce_and_token_acc_same_split_consistent(tmp_path):
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


def test_paper_config_near_13m():
    extras = [f"<a_{i}>" for i in range(256)] + [f"<u_{i}>" for i in range(2000)]
    tok = SemanticIdTokenizer(extras)
    cfg = build_paper_t5_config(len(tok))
    model = TIGERModel(cfg)
    n = sum(p.numel() for p in model.parameters())
    assert 10e6 < n < 20e6
    assert cfg.d_model == 384 and cfg.num_layers == 4 and cfg.num_heads == 6


def test_ndcg_and_paper_reference():
    assert abs(ndcg_at_k(0, 5) - 1.0) < 1e-9
    assert ndcg_at_k(None, 5) == 0.0
    metrics = summarize_ranking(
        ["a", "b"],
        [["a", "x", "y"], ["z", "b", "y"]],
        ks=[1, 2],
    )
    assert abs(metrics["recall@1"] - 0.5) < 1e-9
    assert abs(metrics["recall@2"] - 1.0) < 1e-9
    assert "recall@5" in PAPER_BEAUTY_METRICS


def test_xlt_codec_and_dataset():
    from tiger_ascend.data.xlt import XltSeqDataset, parse_level_codes

    codes = parse_level_codes(["<a_1>", "<b_2>", "<c_3>", "<d_4>"])
    assert codes == [1 + 1, 2 + 256 + 1, 3 + 512 + 1, 4 + 768 + 1]
    inter = {"0": [0, 1, 2, 3, 4], "1": [1, 2, 3, 4, 5, 6]}
    indices = {
        str(i): [f"<a_{i % 8}>", f"<b_{(i + 1) % 8}>", f"<c_{(i + 2) % 8}>", f"<d_{(i + 3) % 8}>"]
        for i in range(8)
    }
    ds = XltSeqDataset(inter, indices, max_his_len=20, mode="train", codebook_size=256)
    assert len(ds) > 0
    row = ds[0]
    assert len(row["history"]) == 80
    assert len(row["target"]) == 4
    batch = ds.get_collate_fn()([ds[0], ds[1]])
    assert batch["input_ids"].shape[1] == 80
    assert batch["labels"].shape[1] == 4


def test_xlt_config_size():
    from transformers import T5ForConditionalGeneration

    from tiger_ascend.model.tiger import build_xlt_t5_config

    cfg = build_xlt_t5_config(1025, layout="xlt")
    model = T5ForConditionalGeneration(cfg)
    n = sum(p.numel() for p in model.parameters())
    assert cfg.d_model == 128 and cfg.d_ff == 1024 and cfg.eos_token_id == 0
    assert 3e6 < n < 8e6
    cfg2 = build_xlt_t5_config(1025, layout="npu_safe")
    assert cfg2.d_model == 128 and cfg2.num_heads * cfg2.d_kv == cfg2.d_model
    cfg3 = build_xlt_t5_config(1025, layout="npu_large")
    assert cfg3.d_model == 384 and cfg3.num_heads * cfg3.d_kv == cfg3.d_model


def test_move_module_to_device_safe_cpu_and_shared():
    """T5 reuses shared Embedding; safe move must keep tying and land on device."""
    from transformers import T5ForConditionalGeneration

    from tiger_ascend.model.tiger import build_xlt_t5_config
    from tiger_ascend.utils.device import move_module_to_device_safe

    cfg = build_xlt_t5_config(128, layout="npu_safe", use_cache=False)
    model = T5ForConditionalGeneration(cfg)
    assert model.shared is model.encoder.embed_tokens
    assert model.shared is model.decoder.embed_tokens
    move_module_to_device_safe(model, "cpu", log=False, sync_each=False, strategy="copy_")
    assert model.shared is model.encoder.embed_tokens
    assert next(model.parameters()).device.type == "cpu"
    all_p = list(model.named_parameters(remove_duplicate=False))
    uniq_p = list(model.named_parameters(remove_duplicate=True))
    assert len(all_p) > len(uniq_p)
    move_module_to_device_safe(model, "cpu", log=False, sync_each=False, strategy="to")
    assert next(model.parameters()).device.type == "cpu"