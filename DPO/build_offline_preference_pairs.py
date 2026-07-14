"""Builds preferred and rejected pairs from scored offline candidates."""
import pandas as pd

from DPO.config import OfflinePreferenceConfig


def build_pairs(cfg: OfflinePreferenceConfig):
    df = pd.read_csv(cfg.path.candidate_csv)

    print("=" * 80)
    print("[PairBuild] start building offline preference pairs")
    print(f"[PairBuild] candidate_csv           : {cfg.path.candidate_csv}")
    print(f"[PairBuild] total_candidates        : {len(df)}")


    GAP_THRESHOLD = cfg.pair.min_score_gap
    TOP_K = cfg.pair.top_k
    BOTTOM_K = cfg.pair.bottom_k

    print(f"[PairBuild] GAP_THRESHOLD           : {GAP_THRESHOLD}")
    print(f"[PairBuild] TOP_K                   : {TOP_K}")
    print(f"[PairBuild] BOTTOM_K                : {BOTTOM_K}")
    print("=" * 80)

    required_cols = {"condition_id", "sample_id", "sequence", "predictor_score"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"[PairBuild] candidate csv missing columns: {sorted(missing_cols)}")

    pairs = []
    total_conditions = df["condition_id"].nunique()

    kept_conditions = 0
    dropped_pairs_gap = 0
    total_raw_pairs = 0

    grouped = df.groupby("condition_id")

    for idx, (cond_id, group) in enumerate(grouped, start=1):
        group = group.sort_values("predictor_score", ascending=False).reset_index(drop=True)

        top = group.head(TOP_K)
        bottom = group.tail(BOTTOM_K)

        valid_pairs_this_cond = 0
        raw_pairs_this_cond = 0

        top_best = float(top["predictor_score"].max()) if len(top) > 0 else None
        bottom_best = float(bottom["predictor_score"].min()) if len(bottom) > 0 else None

        for _, t in top.iterrows():
            for _, b in bottom.iterrows():
                raw_pairs_this_cond += 1
                total_raw_pairs += 1

                gap = float(t["predictor_score"]) - float(b["predictor_score"])

                if gap < GAP_THRESHOLD:
                    dropped_pairs_gap += 1
                    continue

                pairs.append(
                    {
                        "condition_id": int(cond_id),
                        "preferred_sample_id": int(t["sample_id"]),
                        "rejected_sample_id": int(b["sample_id"]),
                        "preferred_seq": str(t["sequence"]),
                        "rejected_seq": str(b["sequence"]),
                        "preferred_score": float(t["predictor_score"]),
                        "rejected_score": float(b["predictor_score"]),
                        "score_gap": float(gap),
                    }
                )
                valid_pairs_this_cond += 1

        if valid_pairs_this_cond > 0:
            kept_conditions += 1

        print(
            f"[PairBuild] condition {idx}/{total_conditions} | "
            f"condition_id={cond_id} | "
            f"group_size={len(group)} | "
            f"top_best={top_best:.4f} | "
            f"bottom_best={bottom_best:.4f} | "
            f"raw_pairs={raw_pairs_this_cond} | "
            f"kept_pairs={valid_pairs_this_cond}"
        )

    pair_df = pd.DataFrame(pairs)

    pair_df.to_csv(cfg.path.pair_csv, index=False)

    print("=" * 80)
    print("[PairBuild] finished")
    print(f"[PairBuild] total_conditions         : {total_conditions}")
    print(f"[PairBuild] kept_conditions          : {kept_conditions}")
    print(f"[PairBuild] total_raw_pairs          : {total_raw_pairs}")
    print(f"[PairBuild] dropped_pairs_gap        : {dropped_pairs_gap}")
    print(f"[PairBuild] final_pairs              : {len(pair_df)}")
    print(f"[PairBuild] saved_to                 : {cfg.path.pair_csv}")
    print("=" * 80)


if __name__ == "__main__":
    build_pairs(OfflinePreferenceConfig())
