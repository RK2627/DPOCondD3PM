"""Defines loss functions for conditional discrete diffusion training."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def _check_logits_targets(logits, targets):

    B, L, C = logits.shape
    if targets.shape != (B, L):
        raise ValueError(f"targets shape {targets.shape} != (B,L) {(B,L)}")
    if targets.min().item() < 0 or targets.max().item() >= C:
        raise ValueError(f"targets out of range: min={targets.min().item()}, max={targets.max().item()}, C={C}")


def masked_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    xt: torch.Tensor,
    mask_id: int,
    aux_weight: float = 1.0,
):


    _check_logits_targets(logits, targets)

    B, L, C = logits.shape
    device = logits.device


    mask_pos = (xt == mask_id)

    if mask_pos.sum() == 0:
        loss_main = F.cross_entropy(
            logits.reshape(-1, C), targets.reshape(-1), reduction="mean"
        )
    else:
        loss_main = F.cross_entropy(
            logits[mask_pos], targets[mask_pos], reduction="mean"
        )


    loss_aux = F.cross_entropy(
        logits.reshape(-1, C), targets.reshape(-1), reduction="mean"
    )

    loss = loss_main + aux_weight * loss_aux
    return loss, {"main": float(loss_main.item()), "aux": float(loss_aux.item())}
