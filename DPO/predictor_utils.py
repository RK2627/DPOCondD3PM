"""Loads E. coli and Actinobacteria predictors for candidate scoring."""
from typing import List

import torch

from Predictor.ecoli_predictor.predictor_models import LSTMModel
from Predictor.actinobacteria_predictor.predictor import load_activity_predictor, predict_activity_batch

DNA2ID = {"A": 0, "C": 1, "G": 2, "T": 3}
VALID_DNA = set(DNA2ID.keys())


def sanitize_dna(seq: str, seq_len: int = 165, replace_invalid_with: str = "A") -> str:
    seq = str(seq).strip().upper()
    chars = [ch if ch in VALID_DNA else replace_invalid_with for ch in seq]
    seq = "".join(chars)
    if len(seq) < seq_len:
        seq = seq + replace_invalid_with * (seq_len - len(seq))
    elif len(seq) > seq_len:
        seq = seq[:seq_len]
    return seq


class EcoliPredictorScorer:
    def __init__(self, ckpt_path: str, seq_len: int = 165, hidden_size: int = 256, output_size: int = 1,
                 dropout_rate: float = 0.2, lambda_l2: float = 1e-3, device=None):
        self.seq_len = seq_len
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMModel(
            input_size=4,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout_rate=dropout_rate,
            lambda_l2=lambda_l2,
        ).to(self.device)
        state = torch.load(ckpt_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        print(f"[EcoliPredictorScorer] loaded ckpt={ckpt_path}")

    def _onehot(self, seq: str) -> torch.Tensor:
        seq = sanitize_dna(seq, seq_len=self.seq_len)
        x = torch.zeros(self.seq_len, 4, dtype=torch.float32)
        for i, ch in enumerate(seq):
            x[i, DNA2ID[ch]] = 1.0
        return x

    @torch.no_grad()
    def score(self, seqs: List[str], batch_size: int = 128) -> torch.Tensor:
        mats = [self._onehot(s) for s in seqs]
        outs = []
        for st in range(0, len(mats), batch_size):
            x = torch.stack(mats[st: st + batch_size], dim=0).to(self.device)
            y = self.model(x).squeeze(-1).float()
            outs.append(y.detach().cpu())
        return torch.cat(outs, dim=0)


class ActinoPredictorScorer:
    def __init__(self, predictor_dir: str, device=None, max_length: int = 200):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.model, self.tokenizer, self.pred_device = load_activity_predictor(predictor_dir, device=str(self.device))
        print(f"[ActinoPredictorScorer] loaded dir={predictor_dir}")

    @torch.no_grad()
    def score(self, seqs: List[str], batch_size: int = 32) -> torch.Tensor:
        seqs = [sanitize_dna(s, seq_len=len(s), replace_invalid_with="A") for s in seqs]
        values = predict_activity_batch(
            self.model,
            self.tokenizer,
            self.pred_device,
            seqs,
            max_length=self.max_length,
            batch_size=batch_size,
        )
        return torch.tensor(values, dtype=torch.float32)


def build_predictor_scorer(predictor_type: str, ecoli_ckpt_path: str, actino_dir: str, device=None):
    predictor_type = predictor_type.lower()
    if predictor_type == "predictor_ecoli":
        return EcoliPredictorScorer(ckpt_path=ecoli_ckpt_path, device=device)
    if predictor_type == "predictor_actino":
        return ActinoPredictorScorer(predictor_dir=actino_dir, device=device)
    raise ValueError(f"Unknown predictor_type={predictor_type}")
