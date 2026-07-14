"""Generates and scores offline candidates for preference-pair construction."""
import csv
import math
import os

from conditionD3PM.sampler_cond_d3pm import ancestral_sample_cond
from conditionD3PM.utils import set_seed, select_device
from DPO.config import OfflinePreferenceConfig
from DPO.feature_pool import TopFeaturePoolBank
from DPO.model_utils import build_d3pm_model, build_scheduler
from DPO.predictor_utils import build_predictor_scorer, sanitize_dna

ID2DNA = {0: "A", 1: "C", 2: "G", 3: "T"}


def ids_to_dna(x):
    seqs = []
    for row in x.detach().cpu().long():
        seqs.append("".join(ID2DNA.get(int(v), "A") for v in row.tolist()))
    return seqs


def build_candidates(cfg: OfflinePreferenceConfig):
    set_seed(cfg.runtime.seed)
    device = select_device(cfg.runtime.gpu_id)
    os.makedirs(os.path.dirname(cfg.path.candidate_csv), exist_ok=True)

    print("=" * 80)
    print("[CandidateBuild] start building offline candidates")
    print(f"[CandidateBuild] device                : {device}")
    print(f"[CandidateBuild] seed                  : {cfg.runtime.seed}")
    print(f"[CandidateBuild] predictor_type        : {cfg.candidate.predictor_type}")
    print(f"[CandidateBuild] num_conditions        : {cfg.candidate.num_conditions}")
    print(f"[CandidateBuild] samples_per_condition : {cfg.candidate.samples_per_condition}")
    print(f"[CandidateBuild] sample_batch_size     : {cfg.candidate.sample_batch_size}")
    print(f"[CandidateBuild] predictor_batch_size  : {cfg.candidate.predictor_batch_size}")
    print(f"[CandidateBuild] unique_condition      : {cfg.candidate.unique_condition}")
    print(f"[CandidateBuild] output_csv            : {cfg.path.candidate_csv}")
    print("=" * 80)

    pool_bank = TopFeaturePoolBank(
        global_path=cfg.path.top_global_pool_path if cfg.model.use_global else None,
        full_local_path=cfg.path.top_full_local_pool_path if cfg.model.use_full_local else None,
        motif_local_path=cfg.path.top_motif_local_pool_path if cfg.model.use_motif_local else None,
        device=device,
    )

    condition_ids = pool_bank.sample_condition_indices(
        num_conditions=cfg.candidate.num_conditions,
        unique=cfg.candidate.unique_condition,
        seed=cfg.runtime.seed,
    )
    print(f"[CandidateBuild] actual_num_conditions : {len(condition_ids)}")

    model = build_d3pm_model(cfg, device, cfg.path.base_ckpt_path, eval_mode=True)
    print(f"[CandidateBuild] loaded base ckpt: {cfg.path.base_ckpt_path}")

    scheduler = build_scheduler(cfg, device)

    scorer = build_predictor_scorer(
        predictor_type=cfg.candidate.predictor_type,
        ecoli_ckpt_path=cfg.path.predictor_ecoli_ckpt_path,
        actino_dir=cfg.path.predictor_actino_dir,
        device=device,
    )

    rows = []
    sample_id = 0

    rep = cfg.candidate.samples_per_condition
    cond_batch = max(cfg.candidate.sample_batch_size // max(rep, 1), 1)
    total_condition_num = len(condition_ids)
    total_rounds = math.ceil(total_condition_num / cond_batch)
    expected_total_samples = total_condition_num * rep

    print("-" * 80)
    print(f"[CandidateBuild] usable_conditions     : {total_condition_num}")
    print(f"[CandidateBuild] conditions_per_round  : {cond_batch}")
    print(f"[CandidateBuild] total_rounds          : {total_rounds}")
    print(f"[CandidateBuild] expected_total_samples: {expected_total_samples}")
    print("-" * 80)

    for round_idx, st in enumerate(range(0, total_condition_num, cond_batch), start=1):
        ed = min(st + cond_batch, total_condition_num)
        cur_cond_ids = condition_ids[st:ed]

        expanded_ids = cur_cond_ids.repeat_interleave(rep)
        conds = pool_bank.get_by_indices(expanded_ids)

        cur_condition_num = cur_cond_ids.numel()
        cur_sample_num = expanded_ids.numel()

        print(
            f"[CandidateBuild] round {round_idx}/{total_rounds} | "
            f"conditions {st + 1}-{ed}/{total_condition_num} | "
            f"cur_conditions={cur_condition_num} | "
            f"samples_per_condition={rep} | "
            f"cur_samples={cur_sample_num} | "
            f"accumulated_before={len(rows)}"
        )

        x = ancestral_sample_cond(
            model=model,
            scheduler=scheduler,
            seq_len=cfg.model.seq_len,
            num_steps=cfg.model.num_steps,
            batch_size=cur_sample_num,
            device=device,
            cond_global=conds["cond_global"],
            cond_full_local=conds["cond_full_local"],
            cond_motif_local=conds["cond_motif_local"],
            temp=cfg.model.sample_temp,
        )

        seqs = [sanitize_dna(s, seq_len=cfg.model.seq_len) for s in ids_to_dna(x)]
        scores = scorer.score(seqs, batch_size=cfg.candidate.predictor_batch_size).tolist()

        for i, seq in enumerate(seqs):
            rows.append(
                {
                    "condition_id": int(expanded_ids[i].item()),
                    "sample_id": int(sample_id),
                    "sequence": seq,
                    "predictor_score": float(scores[i]),
                }
            )
            sample_id += 1

        print(
            f"[CandidateBuild] round {round_idx}/{total_rounds} finished | "
            f"new_rows={cur_sample_num} | accumulated_after={len(rows)}"
        )

        if device.type == "cuda":
            import torch
            torch.cuda.empty_cache()

    with open(cfg.path.candidate_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["condition_id", "sample_id", "sequence", "predictor_score"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 80)
    print(f"[CandidateBuild] finished all rounds")
    print(f"[CandidateBuild] total_rows_saved : {len(rows)}")
    print(f"[CandidateBuild] saved_to         : {cfg.path.candidate_csv}")
    print("=" * 80)


if __name__ == "__main__":
    build_candidates(OfflinePreferenceConfig())
