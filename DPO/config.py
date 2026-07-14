"""Defines configuration dataclasses for offline preference optimization."""
from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class PathConfig:

    base_ckpt_path: str = "./conditionD3PM/checkpoints/conditional_d3pm_best.pth"


    real_train_csv: str = "./DPO/data/real_train.csv"


    top_global_pool_path: Optional[str] = "./DPO/features/global_top5.pth"
    top_full_local_pool_path: Optional[str] = "./DPO/features/full_local_top5.pth"
    top_motif_local_pool_path: Optional[str] = "./DPO/features/motif_local_top5.pth"


    train_global_feature_path: Optional[str] = "./DPO/features/global.pth"
    train_full_local_feature_path: Optional[str] = "./DPO/features/full_local.pth"
    train_motif_local_feature_path: Optional[str] = "./DPO/features/motif_local.pth"


    offline_dir: str = "./DPO/offline_preference"
    candidate_csv: str = "./DPO/offline_preference/candidates.csv"
    pair_csv: str = "./DPO/offline_preference/pairs.csv"


    train_out_dir: str = "./DPO/checkpoints/full_model"
    best_ckpt_path: str = "./DPO/checkpoints/full_model/best.pth"
    last_ckpt_path: str = "./DPO/checkpoints/full_model/last.pth"
    history_json_path: str = "./DPO/checkpoints/full_model/history.json"


    sample_out_txt: str = "./DPO/outputs/dpo_samples.txt"


    predictor_ecoli_ckpt_path: str = "./Predictor/ecoli_predictor/checkpoints/LSTMModel.pth"
    predictor_actino_dir: str = "./Predictor/actinobacteria_predictor/Actinobacteria"


@dataclass
class ModelConfig:
    vocab_size: int = 4
    mask_token_id: int = 4
    seq_len: int = 165
    num_steps: int = 200
    model_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1

    use_global: bool = True
    use_full_local: bool = True
    use_motif_local: bool = True
    global_mode: str = "concat"
    local_mode: str = "concat"
    fusion_mode: str = "gate"
    cross_heads: int = 8
    cross_dim: int = 256
    cross_layers: int = 1
    gate_channel: bool = True
    motif_span_left: int = 35
    motif_span_right: int = 7
    sample_temp: float = 1.0


@dataclass
class RuntimeConfig:
    gpu_id: int = 0
    seed: int = 42
    num_workers: int = 0

    @property
    def device(self):
        return torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")


@dataclass
class CandidateConfig:
    predictor_type: str = "predictor_ecoli"
    num_conditions: int = 500
    samples_per_condition: int = 30
    sample_batch_size: int = 256
    predictor_batch_size: int = 128
    unique_condition: bool = True
    save_every_n_batches: int = 10


@dataclass
class PairConfig:
    top_k: int = 2
    bottom_k: int = 2
    min_score_gap: float = 2.0


@dataclass
class TrainConfig:
    train_strategy: str = "full_model"
    epochs_per_round: int = 20
    pref_batch_size: int = 32
    real_batch_size: int = 64
    lr: float = 3e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    pref_temperature: float = 1.0
    real_loss_coef: float = 0.3
    use_real_anchor: bool = True
    pref_mc_times: int = 2
    aux_weight: float = 1.0
    print_every: int = 20
    save_every_epoch: bool = True


@dataclass
class EvalSampleConfig:
    num_conditions: int = 500
    samples_per_condition: int = 6
    batch_size: int = 256
    use_best_ckpt: bool = True


@dataclass
class OfflinePreferenceConfig:
    path: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    candidate: CandidateConfig = field(default_factory=CandidateConfig)
    pair: PairConfig = field(default_factory=PairConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval_sample: EvalSampleConfig = field(default_factory=EvalSampleConfig)
