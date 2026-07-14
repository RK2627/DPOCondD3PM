"""Samples sequences from an offline preference-trained DPO checkpoint."""
import math
import os
import time

from conditionD3PM.sampler_cond_d3pm import ancestral_sample_cond
from conditionD3PM.utils import set_seed, select_device
from DPO.config import OfflinePreferenceConfig
from DPO.feature_pool import TopFeaturePoolBank
from DPO.model_utils import build_d3pm_model, build_scheduler

ID2DNA = {0: "A", 1: "C", 2: "G", 3: "T"}


def ids_to_dna(x):
    seqs = []
    for row in x.detach().cpu().long():
        seqs.append("".join(ID2DNA.get(int(v), "A") for v in row.tolist()))
    return seqs


def sample_model(cfg: OfflinePreferenceConfig, ckpt_path: str, out_txt: str):
    set_seed(cfg.runtime.seed)
    device = select_device(cfg.runtime.gpu_id)

    print("=" * 80)
    print("[Sample] Start sampling")
    print(f"[Sample] seed                : {cfg.runtime.seed}")
    print(f"[Sample] device              : {device}")
    print(f"[Sample] ckpt_path           : {ckpt_path}")
    print(f"[Sample] output_txt          : {out_txt}")
    print("=" * 80)

    model = build_d3pm_model(cfg, device, ckpt_path, eval_mode=True)
    scheduler = build_scheduler(cfg, device)

    pool_bank = TopFeaturePoolBank(
        global_path=cfg.path.top_global_pool_path if cfg.model.use_global else None,
        full_local_path=cfg.path.top_full_local_pool_path if cfg.model.use_full_local else None,
        motif_local_path=cfg.path.top_motif_local_pool_path if cfg.model.use_motif_local else None,
        device=device,
    )

    cond_ids = pool_bank.sample_condition_indices(
        cfg.eval_sample.num_conditions,
        unique=False,
        seed=cfg.runtime.seed,
    )

    expanded = cond_ids.repeat_interleave(cfg.eval_sample.samples_per_condition)

    total_conditions = cfg.eval_sample.num_conditions
    samples_per_condition = cfg.eval_sample.samples_per_condition
    batch_size = cfg.eval_sample.batch_size
    total_samples = expanded.numel()
    total_rounds = math.ceil(total_samples / batch_size) if total_samples > 0 else 0

    print("[Sample] Sampling plan")
    print(f"[Sample] num_conditions      : {total_conditions}")
    print(f"[Sample] samples/condition   : {samples_per_condition}")
    print(f"[Sample] total_samples       : {total_samples}")
    print(f"[Sample] batch_size          : {batch_size}")
    print(f"[Sample] total_rounds        : {total_rounds}")
    print("-" * 80)

    ptr = 0
    round_idx = 0
    start_time = time.time()

    out_dir = os.path.dirname(out_txt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_txt, "w") as f:
        while ptr < total_samples:
            round_idx += 1
            cur = expanded[ptr: ptr + batch_size]
            cur_bs = cur.numel()

            print(
                f"[Sample] Round {round_idx}/{total_rounds} | "
                f"current_batch={cur_bs} | "
                f"done_before={ptr}/{total_samples}"
            )

            conds = pool_bank.get_by_indices(cur)

            x = ancestral_sample_cond(
                model=model,
                scheduler=scheduler,
                seq_len=cfg.model.seq_len,
                num_steps=cfg.model.num_steps,
                batch_size=cur_bs,
                device=device,
                cond_global=conds["cond_global"],
                cond_full_local=conds["cond_full_local"],
                cond_motif_local=conds["cond_motif_local"],
                temp=cfg.model.sample_temp,
            )

            seqs = ids_to_dna(x)
            for s in seqs:
                f.write(s + "\n")

            ptr += cur_bs
            percent = 100.0 * ptr / total_samples if total_samples > 0 else 100.0
            elapsed = time.time() - start_time

            print(
                f"[Sample] Round {round_idx}/{total_rounds} finished | "
                f"generated_now={cur_bs} | "
                f"done_after={ptr}/{total_samples} | "
                f"progress={percent:.2f}% | "
                f"elapsed={elapsed:.1f}s"
            )
            print("-" * 80)

    total_time = time.time() - start_time
    print("=" * 80)
    print(f"[Sample] Finished: total={total_samples}")
    print(f"[Sample] Saved to: {out_txt}")
    print(f"[Sample] Total time: {total_time:.1f}s")
    print("=" * 80)


def main():
    cfg = OfflinePreferenceConfig()
    ckpt = cfg.path.best_ckpt_path if cfg.eval_sample.use_best_ckpt else cfg.path.last_ckpt_path
    sample_model(cfg, ckpt, cfg.path.sample_out_txt)


if __name__ == "__main__":
    main()
