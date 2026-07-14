"""Loads the Actinobacteria predictor and scores input sequences."""


import os
import re
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


def clean_dna_seq(s: str) -> str:
    s = str(s).upper().replace("U", "T")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^ACGTN]", "N", s)
    return s


def prepare_load_kwargs(name_or_path: str):
    is_local = os.path.isdir(name_or_path)
    if is_local:
        os.environ["HF_HUB_OFFLINE"] = "1"
        return dict(trust_remote_code=True, local_files_only=True)
    return dict(trust_remote_code=True)


class FrozenEncoderCNNRegressor(nn.Module):
    def __init__(
        self,
        encoder,
        hidden_size: int,
        pooling: str = "mean_last4",
        head_hidden: int = 512,
        head_dropout: float = 0.05,
        cnn_kernels=(3, 5, 7, 11),
        cnn_channels=128,
        cnn_pool: str = "maxmean",
        lse_tau: float = 1.0,
        label_transform: str = "none",
        label_mu: float = 0.0,
        label_sd: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.label_transform = label_transform
        self.label_mu = float(label_mu)
        self.label_sd = float(label_sd)

        self.cnn_kernels = [int(k) for k in cnn_kernels]
        self.cnn_channels = int(cnn_channels)
        self.cnn_pool = str(cnn_pool)
        self.lse_tau = float(lse_tau)

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_size, self.cnn_channels, int(k), padding=int(k) // 2),
                nn.GELU(),
                nn.GroupNorm(num_groups=8, num_channels=self.cnn_channels),
            )
            for k in self.cnn_kernels
        ])

        base_dim = self.cnn_channels * len(self.cnn_kernels)
        feat_dim = base_dim * 2 if self.cnn_pool == "maxmean" else base_dim

        self.reg_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def _encode_hidden(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=(self.pooling == "mean_last4"),
            return_dict=True,
        )
        if self.pooling == "mean_last4":
            H = torch.stack(out.hidden_states[-4:], dim=0).mean(dim=0)
        else:
            H = out.last_hidden_state
        return H

    def _cnn_feats(self, H, attention_mask):
        x = H.transpose(1, 2)
        m = attention_mask.unsqueeze(1).to(dtype=x.dtype)
        feats = []

        for conv in self.convs:
            y = conv(x)
            y_masked = y.masked_fill(m == 0, -1e9)
            y_max = torch.amax(y_masked, dim=-1)

            if self.cnn_pool == "maxmean":
                denom = m.sum(dim=-1).clamp(min=1.0)
                y_mean = (y * m).sum(dim=-1) / denom
                feats.append(torch.cat([y_max, y_mean], dim=-1))
            elif self.cnn_pool == "max":
                feats.append(y_max)
            else:
                tau = max(self.lse_tau, 1e-6)
                y_lse = torch.logsumexp(y_masked / tau, dim=-1) * tau
                feats.append(y_lse)

        return torch.cat(feats, dim=-1)

    def denorm_predictions(self, preds):
        if self.label_transform == "zscore":
            return preds * self.label_sd + self.label_mu
        return preds

    @torch.no_grad()
    def predict_activity(self, input_ids=None, attention_mask=None):
        H = self._encode_hidden(input_ids, attention_mask)
        f = self._cnn_feats(H, attention_mask)
        logits = self.reg_head(f).view(-1)
        return self.denorm_predictions(logits)


def load_activity_predictor(pred_dir: str, device: str = None):


    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    meta_path = os.path.join(pred_dir, "model_meta.json")
    weight_path = os.path.join(pred_dir, "pytorch_model.bin")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing file: {meta_path}")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Missing file: {weight_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    encoder_path = meta["model_name"]
    encoder = AutoModel.from_pretrained(encoder_path, **prepare_load_kwargs(encoder_path))
    hidden = int(getattr(encoder.config, "hidden_size", 768))
    kernels = [int(x) for x in str(meta["cnn_kernels"]).split(",") if x.strip()]

    model = FrozenEncoderCNNRegressor(
        encoder=encoder,
        hidden_size=hidden,
        pooling=meta.get("pooling", "mean_last4"),
        head_hidden=int(meta.get("head_hidden", 512)),
        head_dropout=float(meta.get("head_dropout", 0.05)),
        cnn_kernels=kernels,
        cnn_channels=int(meta.get("cnn_channels", 128)),
        cnn_pool=meta.get("cnn_pool", "maxmean"),
        lse_tau=float(meta.get("lse_tau", 1.0)),
        label_transform=meta.get("label_transform", "none"),
        label_mu=float(meta.get("label_mu", 0.0)),
        label_sd=float(meta.get("label_sd", 1.0)),
    )

    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)


    try:
        tokenizer = AutoTokenizer.from_pretrained(
            pred_dir, trust_remote_code=True, local_files_only=True
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            encoder_path, **prepare_load_kwargs(encoder_path)
        )

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token

    return model, tokenizer, device


@torch.no_grad()
def predict_activity_batch(
    model,
    tokenizer,
    device,
    sequences,
    max_length: int = 200,
    batch_size: int = 32,
):


    sequences = [clean_dna_seq(s) for s in sequences]
    preds = []

    for st in range(0, len(sequences), batch_size):
        batch = sequences[st: st + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        y = model.predict_activity(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )
        preds.extend(y.detach().cpu().numpy().tolist())

    return preds


@torch.no_grad()
def predict_activity_one(
    model,
    tokenizer,
    device,
    sequence: str,
    max_length: int = 200,
):
    return predict_activity_batch(
        model=model,
        tokenizer=tokenizer,
        device=device,
        sequences=[sequence],
        max_length=max_length,
        batch_size=1,
    )[0]
