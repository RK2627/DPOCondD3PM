"""Trains the diffusion policy with offline preference and anchor losses."""
import json
import os
from itertools import cycle

import torch
import torch.nn.functional as F

from conditionD3PM.losses import masked_ce_loss
from conditionD3PM.utils import set_seed, select_device
from DPO.config import OfflinePreferenceConfig
from DPO.feature_pool import TopFeaturePoolBank, TrainFeatureBank
from DPO.model_utils import build_d3pm_model, build_scheduler
from DPO.offline_preference_data import build_pref_dataloader, build_real_anchor_dataloader
from DPO.trainable_setup import setup_trainable, build_optimizer


class OfflinePreferenceTrainer:
    def __init__(self, cfg: OfflinePreferenceConfig):
        self.cfg = cfg
        set_seed(cfg.runtime.seed)
        self.device = select_device(cfg.runtime.gpu_id)
        self.model = build_d3pm_model(cfg, self.device, cfg.path.base_ckpt_path, eval_mode=False)
        print(f"[OfflinePreferenceTrainer] loaded base ckpt: {cfg.path.base_ckpt_path}")
        setup_trainable(self.model, strategy=cfg.train.train_strategy, verbose=True)
        self.optimizer = build_optimizer(self.model, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        self.scheduler = build_scheduler(cfg, self.device)
        self.pref_loader, self.real_loader = self._build_loaders()
        os.makedirs(cfg.path.train_out_dir, exist_ok=True)
        self.history = []

    def _build_loaders(self):
        top_bank = TopFeaturePoolBank(
            global_path=self.cfg.path.top_global_pool_path if self.cfg.model.use_global else None,
            full_local_path=self.cfg.path.top_full_local_pool_path if self.cfg.model.use_full_local else None,
            motif_local_path=self.cfg.path.top_motif_local_pool_path if self.cfg.model.use_motif_local else None,
            device=None,
        )
        pref_loader = build_pref_dataloader(
            pair_csv=self.cfg.path.pair_csv,
            top_feature_bank=top_bank,
            seq_len=self.cfg.model.seq_len,
            batch_size=self.cfg.train.pref_batch_size,
            shuffle=True,
            num_workers=self.cfg.runtime.num_workers,
        )
        if not self.cfg.train.use_real_anchor:
            return pref_loader, None

        train_bank = TrainFeatureBank(
            global_path=self.cfg.path.train_global_feature_path if self.cfg.model.use_global else None,
            full_local_path=self.cfg.path.train_full_local_feature_path if self.cfg.model.use_full_local else None,
            motif_local_path=self.cfg.path.train_motif_local_feature_path if self.cfg.model.use_motif_local else None,
            device=None,
        )
        real_loader = build_real_anchor_dataloader(
            data_path=self.cfg.path.real_train_csv,
            seq_len=self.cfg.model.seq_len,
            train_feature_bank=train_bank,
            batch_size=self.cfg.train.real_batch_size,
            shuffle=True,
            num_workers=self.cfg.runtime.num_workers,
        )
        return pref_loader, real_loader

    def _to_device(self, batch):
        out = {}
        for k, v in batch.items():
            if v is None:
                out[k] = None
            elif torch.is_tensor(v):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    def _per_sample_diffusion_loss(self, x0, cond_global, cond_full_local, cond_motif_local, t):
        xt = self.scheduler.q_sample(x0, t)
        logits = self.model.predict_logits(
            xt,
            t,
            cond_global=cond_global,
            cond_full_local=cond_full_local,
            cond_motif_local=cond_motif_local,
        )

        B, L, C = logits.shape
        flat_ce = F.cross_entropy(
            logits.reshape(-1, C),
            x0.reshape(-1),
            reduction="none",
        ).reshape(B, L)

        mask_pos = (xt == self.cfg.model.mask_token_id)
        mask_count = mask_pos.sum(dim=1)
        full_mean = flat_ce.mean(dim=1)

        main_sum = (flat_ce * mask_pos.float()).sum(dim=1)
        main_mean = torch.where(
            mask_count > 0,
            main_sum / mask_count.clamp_min(1).float(),
            full_mean,
        )
        total = main_mean + self.cfg.train.aux_weight * full_mean
        return total, {
            "main_mean": float(main_mean.mean().detach().item()),
            "aux_mean": float(full_mean.mean().detach().item()),
        }

    def _pref_pair_loss(self, batch):
        pref_x0 = batch["preferred_x0"]
        rej_x0 = batch["rejected_x0"]
        cond_global = batch["cond_global"]
        cond_full_local = batch["cond_full_local"]
        cond_motif_local = batch["cond_motif_local"]
        B = pref_x0.size(0)

        pref_losses = []
        rej_losses = []
        for _ in range(self.cfg.train.pref_mc_times):
            t = torch.randint(0, self.cfg.model.num_steps, (B,), device=self.device)
            pref_loss_i, _ = self._per_sample_diffusion_loss(pref_x0, cond_global, cond_full_local, cond_motif_local, t)
            rej_loss_i, _ = self._per_sample_diffusion_loss(rej_x0, cond_global, cond_full_local, cond_motif_local, t)
            pref_losses.append(pref_loss_i)
            rej_losses.append(rej_loss_i)

        loss_pref_vec = torch.stack(pref_losses, dim=0).mean(dim=0)
        loss_rej_vec = torch.stack(rej_losses, dim=0).mean(dim=0)
        margin_vec = loss_rej_vec - loss_pref_vec
        pref_loss = -F.logsigmoid(margin_vec / max(self.cfg.train.pref_temperature, 1e-6)).mean()
        stats = {
            "loss_pref": float(loss_pref_vec.mean().detach().item()),
            "loss_rej": float(loss_rej_vec.mean().detach().item()),
            "margin": float(margin_vec.mean().detach().item()),
        }
        return pref_loss, stats

    def _real_anchor_loss(self, batch):
        x0 = batch["x0"]
        B = x0.size(0)
        t = torch.randint(0, self.cfg.model.num_steps, (B,), device=self.device)
        xt = self.scheduler.q_sample(x0, t)
        logits = self.model.predict_logits(
            xt,
            t,
            cond_global=batch["cond_global"],
            cond_full_local=batch["cond_full_local"],
            cond_motif_local=batch["cond_motif_local"],
        )
        real_loss, parts = masked_ce_loss(
            logits=logits,
            targets=x0,
            xt=xt,
            mask_id=self.cfg.model.mask_token_id,
            aux_weight=self.cfg.train.aux_weight,
        )
        return real_loss, parts

    def train(self):
        best_metric = -1e18
        global_step = 0
        real_iter = cycle(self.real_loader) if self.real_loader is not None else None

        for epoch in range(1, self.cfg.train.epochs_per_round + 1):
            self.model.train()
            epoch_records = []

            for step, pref_batch in enumerate(self.pref_loader, start=1):
                pref_batch = self._to_device(pref_batch)
                self.optimizer.zero_grad(set_to_none=True)

                pref_loss, pref_stats = self._pref_pair_loss(pref_batch)
                total_loss = pref_loss
                real_stats = None

                if self.cfg.train.use_real_anchor and real_iter is not None:
                    real_batch = self._to_device(next(real_iter))
                    real_loss, real_parts = self._real_anchor_loss(real_batch)
                    total_loss = total_loss + self.cfg.train.real_loss_coef * real_loss
                    real_stats = {
                        "real_loss": float(real_loss.detach().item()),
                        "real_main": real_parts["main"],
                        "real_aux": real_parts["aux"],
                    }

                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
                self.optimizer.step()
                global_step += 1

                record = {
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                    "pref_loss": float(pref_loss.detach().item()),
                    "total_loss": float(total_loss.detach().item()),
                    "grad_norm": float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm),
                    **pref_stats,
                }
                if real_stats is not None:
                    record.update(real_stats)
                epoch_records.append(record)

                if step % self.cfg.train.print_every == 0:
                    print(
                        f"[Epoch {epoch} Step {step}] "
                        f"total={record['total_loss']:.4f} "
                        f"pref={record['pref_loss']:.4f} "
                        f"margin={record['margin']:.4f}"
                    )

            epoch_metric = sum(r["margin"] for r in epoch_records) / max(len(epoch_records), 1)
            epoch_total = sum(r["total_loss"] for r in epoch_records) / max(len(epoch_records), 1)
            summary = {
                "epoch": epoch,
                "epoch_mean_margin": epoch_metric,
                "epoch_mean_total_loss": epoch_total,
                "num_steps": len(epoch_records),
                "last_step": epoch_records[-1] if epoch_records else {},
            }
            self.history.append(summary)
            print(f"[Epoch {epoch}] mean_margin={epoch_metric:.6f} mean_total_loss={epoch_total:.6f}")

            if epoch_metric > best_metric:
                best_metric = epoch_metric
                torch.save(self.model.state_dict(), self.cfg.path.best_ckpt_path)
                print(f"[Best] epoch={epoch}, mean_margin={epoch_metric:.6f} -> {self.cfg.path.best_ckpt_path}")

            if self.cfg.train.save_every_epoch:
                torch.save(self.model.state_dict(), os.path.join(self.cfg.path.train_out_dir, f"epoch_{epoch}.pth"))

        torch.save(self.model.state_dict(), self.cfg.path.last_ckpt_path)
        with open(self.cfg.path.history_json_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"[TrainDone] last -> {self.cfg.path.last_ckpt_path}")
        print(f"[TrainDone] history -> {self.cfg.path.history_json_path}")
        return self.history
