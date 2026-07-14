"""Defines configuration dataclasses for Conditional D3PM training and sampling."""
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class TrainConfig:

    data_path: str = "./conditionD3PM/data/ecoli_exp.csv"
    save_path: str = "./conditionD3PM/checkpoints/ecoli/condd3pm_full_concat_best.pth"


    vocab_size: int = 4
    mask_token_id: int = 4
    seq_len: int = 165


    batch_size: int = 128
    epochs: int = 100
    lr: float = 3e-4
    num_steps: int = 200
    model_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    gpu_id: int = 0
    seed: int = 42
    num_workers: int = 0


    early_patience: int = 10
    early_min_delta: float = 3e-4
    early_ema_alpha: float = 0.05


    aux_weight: float = 1.0
    loss_schedule: str = "cosine"
    noise_type: str = "absorbing"


    use_global: bool = False
    use_full_local: bool = True
    use_motif_local: bool = False


    global_mode: str = "concat"


    local_mode: str = "concat"


    fusion_mode: str = "auto"


    cross_heads: int = 8
    cross_dim: int = 256
    cross_layers: int = 1


    gate_channel: bool = True


    motif_span_left: int = 35
    motif_span_right: int = 7


    global_feature_path: Optional[str] = (
        "./conditionD3PM/features/ecoli/global.pth"
    )
    full_local_feature_path: Optional[str] = (
        "./conditionD3PM/features/ecoli/full_local.pth"
    )
    motif_local_feature_path: Optional[str] = (
        "./conditionD3PM/features/ecoli/motif_local.pth"
    )


    @property
    def device(self):
        return torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")


@dataclass
class SampleConfig:
    ckpt_path: str = ("./conditionD3PM/checkpoints/ecoli/condd3pm_full_concat_best.pth")
    out_path: str = ("./conditionD3PM/outputs/ecoli/condd3pm_full_concat_best.txt")

    seq_len: int = 165
    num_steps: int = 200
    num_samples: int = 3000
    batch_size: int = 256
    temp: float = 1.0
    gpu_id: int = 0
    seed: int = 42


    use_global: bool = False
    use_full_local: bool = True
    use_motif_local: bool = False

    global_mode: str = "concat"
    local_mode: str = "concat"
    fusion_mode: str = "auto"


    cross_heads: int = 8
    cross_dim: int = 256
    cross_layers: int = 1
    gate_channel: bool = True


    motif_span_left: int = 35
    motif_span_right: int = 7


    sample_global_path: Optional[str] = (
        "./conditionD3PM/features/ecoli/global_top5.pth"
    )
    sample_full_local_path: Optional[str] = (
        "./conditionD3PM/features/ecoli/full_local_top5.pth"
    )
    sample_motif_local_path: Optional[str] = (
        "./conditionD3PM/features/ecoli/motif_local_top5.pth"
    )


    @property
    def device(self):
        return torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
