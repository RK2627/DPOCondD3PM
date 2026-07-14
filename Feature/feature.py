
"""Trains the promoter feature encoder and extracts or evaluates conditional features."""

import os
import math
import argparse
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


def _safe_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return 0.0
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _rankdata_average(x: np.ndarray) -> np.ndarray:

    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return x.copy()

    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i

        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1

        avg_rank = 0.5 * ((i + 1) + (j + 1))
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    return ranks


def _safe_spearmanr(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return 0.0

    r_true = _rankdata_average(y_true)
    r_pred = _rankdata_average(y_pred)
    return _safe_pearsonr(r_true, r_pred)


def load_and_split_csv(csv_path: str, valid_ratio: float = 0.1, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path, header=None)
    df.columns = ["sequence", "expression"]
    df["expression"] = df["expression"].astype(float)

    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_valid = int(len(df) * valid_ratio)

    valid_idx = idx[:n_valid]
    train_idx = idx[n_valid:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    valid_df = df.iloc[valid_idx].reset_index(drop=True)
    return train_df, valid_df


def one_hot_encode(seq: str, L: int) -> np.ndarray:
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    x = np.zeros((4, L), dtype=np.float32)
    for i, ch in enumerate(seq[:L]):
        if ch in mapping:
            x[mapping[ch], i] = 1.0
    return x


class SeqExprDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mode: str = "train", seq_len: int = 165):
        super().__init__()
        self.df = df
        self.mode = mode
        self.seq_len = seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        seq = self.df.loc[idx, "sequence"]
        x = one_hot_encode(seq, self.seq_len)
        x = torch.from_numpy(x)

        if self.mode == "train":
            y = float(self.df.loc[idx, "expression"])
            y = torch.tensor(y, dtype=torch.float32)
            return x, y
        else:
            return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, mode: str = "none"):
        super().__init__()
        self.mode = mode
        if mode == "learned":
            self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.normal_(self.pe, std=0.02)
        elif mode == "sin":
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)
        else:
            self.pe = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.mode == "none":
            return x
        B, L, D = x.shape
        if self.mode == "sin":
            return x + self.pe[:, :L, :]
        else:
            return x + self.pe[:, :L, :]


class FeatureEncoder(nn.Module):


    def __init__(
        self,
        embed_dim: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
        pos_enc_mode: str = "none",
        max_len: int = 500,
    ):
        super().__init__()
        assert pos_enc_mode in ["none", "sin", "learned"]

        self.embed_dim = embed_dim
        self.pos_enc_mode = pos_enc_mode

        self.cnn = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(embed_dim)

        if pos_enc_mode != "none":
            self.pos_encoder = PositionalEncoding(
                d_model=embed_dim, max_len=max_len, mode=pos_enc_mode
            )
        else:
            self.pos_encoder = None

    def encode_backbone(self, x: torch.Tensor) -> torch.Tensor:

        h = self.cnn(x)
        h = h.permute(0, 2, 1)
        if self.pos_encoder is not None:
            h = self.pos_encoder(h)
        attn_out, _ = self.attn(h, h, h)
        attn_out = self.ln(attn_out)
        return attn_out

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "global",
    ) -> torch.Tensor:

        h_full = self.encode_backbone(x)
        B, L, D = h_full.shape

        if mode == "global":
            return h_full.mean(dim=1)

        elif mode == "full_local":
            return h_full

        elif mode == "motif_local":
            assert L >= 35, f"seq_len={L} < 35, cannot extract the [-35,-7] interval."
            start_idx = L - 35
            end_idx = L - 7 + 1
            assert 0 <= start_idx < end_idx <= L, (
                f"motif index error: L={L}, start={start_idx}, end={end_idx}"
            )
            return h_full[:, start_idx:end_idx, :]

        else:
            raise ValueError(f"Unknown mode={mode}, expected global/full_local/motif_local")


class FeaturePredictor(nn.Module):

    def __init__(self, encoder: FeatureEncoder):
        super().__init__()
        self.encoder = encoder
        D = encoder.embed_dim
        self.regressor = nn.Sequential(
            nn.Linear(D, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x, mode="global")
        out = self.regressor(feat).squeeze(-1)
        return out


def train_encoder(
    csv_path: str,
    save_ckpt: str,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
    valid_ratio: float = 0.1,
    pos_enc: str = "learned",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    train_df, valid_df = load_and_split_csv(csv_path, valid_ratio=valid_ratio)
    seq_len = int(
        max(
            train_df["sequence"].map(len).max(),
            valid_df["sequence"].map(len).max(),
        )
    )
    print(f"[INFO] seq_len (max) = {seq_len}")

    train_ds = SeqExprDataset(train_df, mode="train", seq_len=seq_len)
    valid_ds = SeqExprDataset(valid_df, mode="train", seq_len=seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    encoder = FeatureEncoder(
        embed_dim=256,
        n_heads=8,
        dropout=0.1,
        pos_enc_mode=pos_enc,
        max_len=seq_len,
    ).to(device)

    model = FeaturePredictor(encoder).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()


    best_spearman = -1e9
    best_pearson = -1e9

    for ep in range(1, epochs + 1):

        model.train()
        tr_losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        tr_loss = float(np.mean(tr_losses)) if len(tr_losses) else float("inf")


        model.eval()
        va_losses = []
        y_true_list = []
        y_pred_list = []

        with torch.no_grad():
            for x, y in valid_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)

                loss = loss_fn(pred, y)
                va_losses.append(loss.item())

                y_true_list.append(y.detach().cpu().numpy())
                y_pred_list.append(pred.detach().cpu().numpy())

        val_loss = float(np.mean(va_losses)) if len(va_losses) else float("inf")

        y_true = np.concatenate(y_true_list, axis=0) if len(y_true_list) else np.array([], dtype=np.float64)
        y_pred = np.concatenate(y_pred_list, axis=0) if len(y_pred_list) else np.array([], dtype=np.float64)

        val_pearson = _safe_pearsonr(y_true, y_pred)
        val_spearman = _safe_spearmanr(y_true, y_pred)

        print(
            f"[Epoch {ep:03d}] "
            f"train_mse={tr_loss:.6f}  "
            f"valid_mse={val_loss:.6f}  "
            f"valid_pearson={val_pearson:.4f}  "
            f"valid_spearman={val_spearman:.4f}"
        )


        better = False
        if val_spearman > best_spearman + 1e-12:
            better = True
        elif abs(val_spearman - best_spearman) <= 1e-12 and val_pearson > best_pearson + 1e-12:
            better = True

        if better:
            best_spearman = val_spearman
            best_pearson = val_pearson
            os.makedirs(os.path.dirname(save_ckpt), exist_ok=True)
            torch.save(encoder.state_dict(), save_ckpt)
            print(
                f"[CKPT] Saved best encoder to {save_ckpt} "
                f"(spearman={best_spearman:.4f}, pearson={best_pearson:.4f})"
            )


def extract_features(
    csv_path: str,
    ckpt_path: str,
    out_path: str,
    feature_type: str = "global",
    batch_size: int = 128,
    pos_enc: str = "learned",
):

    assert feature_type in ["global", "full_local", "motif_local"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    print(f"[Extract] type={feature_type}")

    df = pd.read_csv(csv_path, header=None)
    df.columns = ["sequence", "expression"]
    df["expression"] = df["expression"].astype(float)
    seq_len = int(df["sequence"].map(len).max())
    print(f"[INFO] seq_len (max) = {seq_len} (extract)")

    ds = SeqExprDataset(df, mode="extract", seq_len=seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    encoder = FeatureEncoder(
        embed_dim=256,
        n_heads=8,
        dropout=0.0,
        pos_enc_mode=pos_enc,
        max_len=seq_len,
    ).to(device)
    encoder.load_state_dict(torch.load(ckpt_path, map_location=device))
    encoder.eval()
    print(f"[INFO] Loaded encoder ckpt: {ckpt_path}")

    all_feats = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            feats = encoder(x, mode=feature_type)
            all_feats.append(feats.cpu())
    feats_all = torch.cat(all_feats, dim=0)
    print(f"[INFO] features shape = {tuple(feats_all.shape)}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(feats_all, out_path)
    print(f"[SAVE] features saved to {out_path}")


def evaluate_cluster(
    csv_path: str,
    feature_path: str,
    feature_type: str = "global",
    out_prefix: str = "umap",
):

    assert feature_type in ["global", "full_local", "motif_local"]


    feats = torch.load(feature_path)
    if isinstance(feats, np.ndarray):
        feats = torch.from_numpy(feats)
    feats = feats.float()
    print(f"[Eval] Loaded features from {feature_path}, shape={tuple(feats.shape)}")

    if feature_type in ["full_local", "motif_local"]:

        N, Lr, D = feats.shape
        feats = feats.view(N, Lr * D)
        print(f"[Eval] Flatten {feature_type} features -> shape={tuple(feats.shape)}")

    X = feats.numpy()


    df = pd.read_csv(csv_path, header=None)
    df.columns = ["sequence", "expression"]
    y = df["expression"].astype(float).values
    assert len(y) == X.shape[0], "features and CSV sample counts do not match"


    q10, q40, q90 = np.quantile(y, [0.1, 0.4, 0.9])
    labels = np.zeros_like(y, dtype=int)
    labels[y > q10] = 1
    labels[y > q40] = 2
    labels[y > q90] = 3

    label_names = [
        f"0-10% (<= {q10:.2f})",
        f"10-40% ({q10:.2f}-{q40:.2f})",
        f"40-90% ({q40:.2f}-{q90:.2f})",
        f"90-100% (> {q90:.2f})",
    ]
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:red"]


    if HAS_UMAP:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.1,
            metric="euclidean",
            random_state=42,
        )
        coords = reducer.fit_transform(X)
        method_name = "umap"
    else:
        print("[WARN] umap-learn is unavailable; using PCA(2D) instead.")
        pca2 = PCA(n_components=2, random_state=42)
        coords = pca2.fit_transform(X)
        method_name = "pca2"


    plt.figure(figsize=(6, 5))
    draw_order = [3, 2, 1, 0]
    for lab_idx in draw_order:
        mask = labels == lab_idx
        if not np.any(mask):
            continue
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=10,
            alpha=0.7,
            color=colors[lab_idx],
            label=label_names[lab_idx],
        )
    plt.xlabel(f"{method_name.upper()}-1")
    plt.ylabel(f"{method_name.upper()}-2")

    handles = []
    for i in range(4):
        handles.append(
            plt.Line2D(
                [], [],
                marker="o",
                linestyle="None",
                markersize=6,
                markerfacecolor=colors[i],
                markeredgecolor=colors[i],
                alpha=0.7,
                label=label_names[i],
            )
        )
    plt.legend(handles=handles, markerscale=1.0, fontsize=8)
    plt.title(f"{feature_type} embedding ({method_name})")

    out_png = f"{out_prefix}_{feature_type}_{method_name}.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[SAVE] scatter saved to {out_png}")


    pca2 = PCA(n_components=2, random_state=42)
    coords_pca = pca2.fit_transform(X)

    plt.figure(figsize=(6, 5))
    for lab_idx in draw_order:
        mask = labels == lab_idx
        if not np.any(mask):
            continue
        plt.scatter(
            coords_pca[mask, 0],
            coords_pca[mask, 1],
            s=10,
            alpha=0.7,
            color=colors[lab_idx],
            label=label_names[lab_idx],
        )
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")

    handles = []
    for i in range(4):
        handles.append(
            plt.Line2D(
                [], [],
                marker="o",
                linestyle="None",
                markersize=6,
                markerfacecolor=colors[i],
                markeredgecolor=colors[i],
                alpha=0.7,
                label=label_names[i],
            )
        )
    plt.legend(handles=handles, markerscale=1.0, fontsize=8)
    plt.title(f"{feature_type} embedding (PCA2)")

    out_png2 = f"{out_prefix}_{feature_type}_pca2.png"
    plt.tight_layout()
    plt.savefig(out_png2, dpi=300)
    plt.close()
    print(f"[SAVE] scatter saved to {out_png2}")


    pc1 = coords_pca[:, 0]
    corr = np.corrcoef(pc1, y)[0, 1]
    print(f"[Eval] Corr(PC1, expression) = {corr:.4f}")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--mode",
        type=str,
        choices=["train", "extract", "evaluate"],
        default="extract",
    )

    p.add_argument(
        "--csv",
        type=str,
        default="./data/ecoli/ecoli_exp.csv",

        help="Input CSV with two headerless columns: sequence and expression",
    )


    p.add_argument(
        "--save_ckpt",
        type=str,

        default="./Feature/checkpoints/feature_encoder_ecoli.pth",
        help="Path for the encoder checkpoint saved after training",
    )
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--valid_ratio", type=float, default=0.1)
    p.add_argument(
        "--pos_enc",
        type=str,
        choices=["none", "sin", "learned"],
        default="learned",
    )


    p.add_argument(
        "--ckpt",
        type=str,
        default="./Feature/checkpoints/feature_encoder_ecoli.pth",

        help="Encoder checkpoint path used for feature extraction",
    )
    p.add_argument(
        "--out_pth",
        type=str,
        default="./Feature/outputs/ecoli_global.pth",

        help="Output .pth path for extracted features",
    )
    p.add_argument(
        "--feature_type",
        type=str,
        choices=["global", "full_local", "motif_local"],
        default="global",
        help="Feature type to extract: global / full_local / motif_local",
    )
    p.add_argument("--extract_bs", type=int, default=128)


    p.add_argument(
        "--feat_pth",
        type=str,
        default="./Feature/outputs/ecoli_global.pth",
        help="Feature file path used in evaluate mode",
    )
    p.add_argument(
        "--eval_feature_type",
        type=str,
        choices=["global", "full_local", "motif_local"],
        default="global",
        help="Feature type used in evaluate mode: global / full_local / motif_local",
    )
    p.add_argument(
        "--out_prefix",
        type=str,
        default="./Feature/outputs/embedding",
        help="Output prefix for clustering figures",
    )

    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "train":
        train_encoder(
            csv_path=args.csv,
            save_ckpt=args.save_ckpt,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            valid_ratio=args.valid_ratio,
            pos_enc=args.pos_enc,
        )

    elif args.mode == "extract":
        extract_features(
            csv_path=args.csv,
            ckpt_path=args.ckpt,
            out_path=args.out_pth,
            feature_type=args.feature_type,
            batch_size=args.extract_bs,
            pos_enc=args.pos_enc,
        )

    elif args.mode == "evaluate":
        evaluate_cluster(
            csv_path=args.csv,
            feature_path=args.feat_pth,
            feature_type=args.eval_feature_type,
            out_prefix=args.out_prefix,
        )


if __name__ == "__main__":
    main()
