"""Trains Conditional D3PM with aligned condition features."""
import os
import time
import numpy as np
import torch

from conditionD3PM.config import TrainConfig
from conditionD3PM.data import build_dataloader, DNALineDataset
from conditionD3PM.model import D3PM
from conditionD3PM.noise import AbsorbingMaskScheduler, UniformMaskScheduler
from conditionD3PM.losses import masked_ce_loss
from conditionD3PM.utils import set_seed, select_device


def build_scheduler(cfg: TrainConfig, device: torch.device):
    if cfg.noise_type == "absorbing":
        sch = AbsorbingMaskScheduler(
            num_steps=cfg.num_steps,
            mask_token_id=cfg.mask_token_id,
            device=device,
        )
    else:
        sch = UniformMaskScheduler(
            num_steps=cfg.num_steps,
            mask_token_id=cfg.mask_token_id,
            device=device,
        )
    return sch


def build_model(cfg: TrainConfig, device: torch.device):
    model = D3PM(
        vocab_size=cfg.vocab_size,
        mask_token_id=cfg.mask_token_id,
        seq_len=cfg.seq_len,
        d_model=cfg.model_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,

        use_global=cfg.use_global,
        use_full_local=cfg.use_full_local,
        use_motif_local=cfg.use_motif_local,

        global_mode=cfg.global_mode,
        local_mode=cfg.local_mode,


        fusion_mode=cfg.fusion_mode,
        cross_heads=cfg.cross_heads,
        cross_dim=cfg.cross_dim,
        cross_layers=cfg.cross_layers,
        gate_channel=cfg.gate_channel,
        motif_span_left=cfg.motif_span_left,
        motif_span_right=cfg.motif_span_right,
    ).to(device)
    return model


class EarlyStopping:
    def __init__(self, patience=20, save_path=None, min_delta=1e-4, ema_alpha=0.1):


        self.patience = patience
        self.save_path = save_path
        self.min_delta = float(min_delta)
        self.ema_alpha = float(ema_alpha)

        self.best = float("inf")
        self.counter = 0
        self.early_stop = False

        self._ema = None

    def step(self, loss, epoch, model):

        loss = float(loss)
        if self.ema_alpha > 0:
            if self._ema is None:
                self._ema = loss
            else:
                self._ema = self.ema_alpha * loss + (1.0 - self.ema_alpha) * self._ema
            monitor = self._ema
        else:
            monitor = loss


        improved = (self.best - monitor) > self.min_delta

        if improved:
            self.best = monitor
            self.counter = 0
            if self.save_path:
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                torch.save(model.state_dict(), self.save_path.replace(".pth", "_best.pth"))
            print(f"[BEST] epoch={epoch} monitor={monitor:.6f} raw_loss={loss:.6f}")
        else:
            self.counter += 1
            print(f"[NO IMPROVE] epoch={epoch} monitor={monitor:.6f} best={self.best:.6f} "
                  f"delta={(self.best-monitor):.6f} counter={self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"[EARLY STOP] patience={self.patience} best_monitor={self.best:.6f} "
                      f"(min_delta={self.min_delta}, ema_alpha={self.ema_alpha})")


def _load_feature_tensor(path: str, num_samples: int, name: str):
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"[{name}] feature file does not exist: {path}")

    feats = torch.load(path)
    if isinstance(feats, np.ndarray):
        feats = torch.from_numpy(feats)
    feats = feats.float()

    if feats.size(0) != num_samples:
        raise ValueError(
            f"[{name}] feature first dimension does not match sample count: "
            f"features={feats.size(0)}, data={num_samples}"
        )

    print(f"[LOAD] {name} features from {path}, shape={tuple(feats.shape)}")
    return feats


def build_feature_bank(cfg: TrainConfig, num_samples: int):


    global_feats = None
    full_local_feats = None
    motif_local_feats = None


    if cfg.use_global:
        if cfg.global_feature_path is None:
            raise ValueError("use_global=True but global_feature_path is empty")
        global_feats = _load_feature_tensor(
            cfg.global_feature_path, num_samples, name="global"
        )


    if cfg.use_full_local:
        if cfg.full_local_feature_path is None:
            raise ValueError("use_full_local=True but full_local_feature_path is empty")
        full_local_feats = _load_feature_tensor(
            cfg.full_local_feature_path, num_samples, name="full_local"
        )


    if cfg.use_motif_local:
        if cfg.motif_local_feature_path is None:
            raise ValueError("use_motif_local=True but motif_local_feature_path is empty")
        motif_local_feats = _load_feature_tensor(
            cfg.motif_local_feature_path, num_samples, name="motif_local"
        )

    return global_feats, full_local_feats, motif_local_feats


def train():
    cfg = TrainConfig()


    set_seed(cfg.seed)
    device = select_device(cfg.gpu_id)

    assert os.path.exists(cfg.data_path), f"Data file does not exist: {cfg.data_path}"
    print("========== TrainConfig ==========")
    print(cfg)
    print("=================================\n")


    train_loader = build_dataloader(
        cfg.data_path,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    num_samples = len(train_loader.dataset)
    print(f"[DATA] num_samples={num_samples}")


    if not (cfg.use_global or cfg.use_full_local or cfg.use_motif_local):
        global_feats = None
        full_local_feats = None
        motif_local_feats = None
    else:
        global_feats, full_local_feats, motif_local_feats = build_feature_bank(cfg, num_samples)


    model = build_model(cfg, device)
    scheduler = build_scheduler(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    stopper = EarlyStopping(
        patience=cfg.early_patience,
        save_path=cfg.save_path,
        min_delta=cfg.early_min_delta,
        ema_alpha=cfg.early_ema_alpha,
    )
    save_every = 5

    total_start = time.time()


    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        steps = 0
        ep_start = time.time()

        for batch in train_loader:

            x0, idx = batch
            x0 = x0.long().to(device)
            idx = idx.long()

            B = x0.size(0)
            t = torch.randint(0, cfg.num_steps, (B,), device=device)
            x_t = scheduler.q_sample(x0, t)

            cond_global = None
            cond_full_local = None
            cond_motif_local = None
            if global_feats is not None:
                cond_global = global_feats[idx].to(device)
            if full_local_feats is not None:
                cond_full_local = full_local_feats[idx].to(device)
            if motif_local_feats is not None:
                cond_motif_local = motif_local_feats[idx].to(device)


            logits = model.predict_logits(
                x_t,
                t,
                cond_global=cond_global,
                cond_full_local=cond_full_local,
                cond_motif_local=cond_motif_local,
            )
            loss, _ = masked_ce_loss(
                logits=logits,
                targets=x0,
                xt=x_t,
                mask_id=cfg.mask_token_id,
                aux_weight=cfg.aux_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            steps += 1

        avg_loss = running_loss / max(1, steps)
        dur = time.time() - ep_start
        print(f"[Epoch {epoch:03d}/{cfg.epochs}] loss={avg_loss:.4f} time={dur:.1f}s")

        stopper.step(avg_loss, epoch, model)

        if epoch % save_every == 0:
            torch.save(model.state_dict(), cfg.save_path)
            print(f"[CHECKPOINT] epoch={epoch:03d} -> {cfg.save_path}")

        if stopper.early_stop:
            break

        torch.cuda.empty_cache()

    torch.save(model.state_dict(), cfg.save_path)
    print(f"[FINAL CHECKPOINT] -> {cfg.save_path}")

    total_time = time.time() - total_start
    h, rem = divmod(total_time, 3600)
    m, s = divmod(rem, 60)
    print(f"\nTotal training time: {int(h)}h {int(m)}m {int(s)}s")


if __name__ == "__main__":
    train()
