"""Builds Conditional D3PM models and loads their checkpoints for DPO."""
import torch

from conditionD3PM.model import D3PM
from conditionD3PM.noise import AbsorbingMaskScheduler


def build_d3pm_model(cfg, device, ckpt_path: str, eval_mode: bool = True):
    mcfg = cfg.model
    model = D3PM(
        vocab_size=mcfg.vocab_size,
        mask_token_id=mcfg.mask_token_id,
        seq_len=mcfg.seq_len,
        d_model=mcfg.model_dim,
        num_layers=mcfg.num_layers,
        num_heads=mcfg.num_heads,
        dropout=mcfg.dropout,
        use_global=mcfg.use_global,
        use_full_local=mcfg.use_full_local,
        use_motif_local=mcfg.use_motif_local,
        global_mode=mcfg.global_mode,
        local_mode=mcfg.local_mode,
        fusion_mode=mcfg.fusion_mode,
        cross_heads=mcfg.cross_heads,
        cross_dim=mcfg.cross_dim,
        cross_layers=mcfg.cross_layers,
        gate_channel=mcfg.gate_channel,
        motif_span_left=mcfg.motif_span_left,
        motif_span_right=mcfg.motif_span_right,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    if eval_mode:
        model.eval()
    return model


def build_scheduler(cfg, device):
    return AbsorbingMaskScheduler(
        num_steps=cfg.model.num_steps,
        mask_token_id=cfg.model.mask_token_id,
        device=device,
    )
