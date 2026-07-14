"""Implements reverse sampling for the conditional discrete diffusion model."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def ancestral_sample_cond(
    model,
    scheduler,
    seq_len,
    num_steps,
    batch_size=8,
    device=None,
    cond_global=None,
    cond_full_local=None,
    cond_motif_local=None,
    temp: float = 1.0,
):


    device = device or next(model.parameters()).device
    B = batch_size
    mask_id = model.mask_token_id

    printed_gate_debug = False


    x = torch.full(
        (B, seq_len),
        mask_id,
        dtype=torch.long,
        device=device,
    )


    for t in reversed(range(1, num_steps)):
        tt = torch.full((B,), t, dtype=torch.long, device=device)

        logits = model.predict_logits(
            x,
            tt,
            cond_global=cond_global,
            cond_full_local=cond_full_local,
            cond_motif_local=cond_motif_local,
        )


        if not printed_gate_debug:
            gf = getattr(model, "gate_fusion", None)
            last_gate = None
            if gf is not None:
                last_gate = getattr(gf, "last_gate", None)
            if last_gate is not None:
                g_flat = last_gate.float().reshape(-1).detach().cpu()
                mean = g_flat.mean().item()
                std = g_flat.std().item()
                gmin = g_flat.min().item()
                gmax = g_flat.max().item()
                p01 = (g_flat < 0.01).float().mean().item()
                p99 = (g_flat > 0.99).float().mean().item()


                printed_gate_debug = True


        logits = logits.clone()
        logits[..., mask_id] = -1e9

        probs = F.softmax(logits / temp, dim=-1)

        x0_hat = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            1
        ).view(B, seq_len)


        update_mask = scheduler.sample_update_mask(x, t)
        update_mask = update_mask & (x == mask_id)

        x_new = x.clone()
        x_new[update_mask] = x0_hat[update_mask]
        x = x_new


    residual_mask = (x == mask_id)
    if residual_mask.any():
        tt = torch.zeros((B,), dtype=torch.long, device=device)
        logits = model.predict_logits(
            x,
            tt,
            cond_global=cond_global,
            cond_full_local=cond_full_local,
            cond_motif_local=cond_motif_local,
        )

        logits = logits.clone()
        logits[..., mask_id] = -1e9

        fill_tokens = torch.argmax(logits, dim=-1)
        x[residual_mask] = fill_tokens[residual_mask]


    return x
