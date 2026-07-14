"""Configures trainable model parameters and the optimizer for DPO."""
import torch


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_condition_only_trainable(model, verbose=True):
    freeze_all(model)
    keywords = [
        "global_proj", "gamma", "beta",
        "local_proj", "local_add_proj",
        "cross_blocks", "global_cond_proj", "full_local_cond_proj", "motif_local_cond_proj",
        "gate_fusion",
    ]
    enabled = []
    for name, p in model.named_parameters():
        lower = name.lower()
        if any(k.lower() in lower for k in keywords):
            p.requires_grad = True
            enabled.append(name)
    if verbose:
        print("[Trainable Setup] strategy = condition_only")
        for n in enabled:
            print(f"  [T] {n}")
        print(f"[Trainable Setup] total trainable params = {count_trainable_params(model)}")
    return enabled


def setup_full_model_trainable(model, verbose=True):
    unfreeze_all(model)
    if verbose:
        print("[Trainable Setup] strategy = full_model")
        print(f"[Trainable Setup] total trainable params = {count_trainable_params(model)}")


def build_optimizer(model, lr=1e-4, weight_decay=0.0, betas=(0.9, 0.999)):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters are available; the optimizer cannot be built")
    return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)


def setup_trainable(model, strategy="full_model", verbose=True):
    if strategy == "condition_only":
        return setup_condition_only_trainable(model, verbose=verbose)
    if strategy == "full_model":
        return setup_full_model_trainable(model, verbose=verbose)
    raise ValueError(f"Unknown strategy={strategy}")
