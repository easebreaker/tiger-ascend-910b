import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from tiger_ascend.data.dataset import TigerSeqDataset, generate_toy_data, load_json
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
