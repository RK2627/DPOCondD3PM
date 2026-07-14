"""Defines datasets for offline preference pairs and real anchor sequences."""
import csv

import torch
from torch.utils.data import Dataset, DataLoader

from conditionD3PM.data import DNALineDataset
from DPO.feature_pool import TopFeaturePoolBank, TrainFeatureBank

DNA_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def encode_dna(seq: str, seq_len: int) -> torch.Tensor:
    seq = str(seq).strip().upper()
    ids = [DNA_MAP.get(ch, 0) for ch in seq]
    if len(ids) >= seq_len:
        ids = ids[:seq_len]
    else:
        ids = ids + [0] * (seq_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


class PreferencePairDataset(Dataset):
    def __init__(self, pair_csv: str, top_feature_bank: TopFeaturePoolBank, seq_len: int):
        self.rows = []
        self.seq_len = seq_len
        self.bank = top_feature_bank
        with open(pair_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(
                    {
                        "condition_id": int(row["condition_id"]),
                        "preferred_seq": row["preferred_seq"],
                        "rejected_seq": row["rejected_seq"],
                        "preferred_score": float(row["preferred_score"]),
                        "rejected_score": float(row["rejected_score"]),
                        "score_gap": float(row["score_gap"]),
                    }
                )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        cond_id = torch.tensor(row["condition_id"], dtype=torch.long)
        cond = self.bank.get_by_indices(cond_id.view(1))
        return {
            "condition_id": cond_id,
            "preferred_x0": encode_dna(row["preferred_seq"], self.seq_len),
            "rejected_x0": encode_dna(row["rejected_seq"], self.seq_len),
            "preferred_score": torch.tensor(row["preferred_score"], dtype=torch.float32),
            "rejected_score": torch.tensor(row["rejected_score"], dtype=torch.float32),
            "score_gap": torch.tensor(row["score_gap"], dtype=torch.float32),
            "cond_global": None if cond["cond_global"] is None else cond["cond_global"].squeeze(0),
            "cond_full_local": None if cond["cond_full_local"] is None else cond["cond_full_local"].squeeze(0),
            "cond_motif_local": None if cond["cond_motif_local"] is None else cond["cond_motif_local"].squeeze(0),
        }


class RealAnchorDataset(Dataset):
    def __init__(self, data_path: str, seq_len: int, train_feature_bank: TrainFeatureBank):
        self.seq_ds = DNALineDataset(data_path, seq_len=seq_len)
        self.bank = train_feature_bank

    def __len__(self):
        return len(self.seq_ds)

    def __getitem__(self, idx):
        x0, real_idx = self.seq_ds[idx]
        idx_tensor = torch.tensor(real_idx, dtype=torch.long)
        cond = self.bank.get_by_indices(idx_tensor.view(1))
        return {
            "x0": x0,
            "real_idx": idx_tensor,
            "cond_global": None if cond["cond_global"] is None else cond["cond_global"].squeeze(0),
            "cond_full_local": None if cond["cond_full_local"] is None else cond["cond_full_local"].squeeze(0),
            "cond_motif_local": None if cond["cond_motif_local"] is None else cond["cond_motif_local"].squeeze(0),
        }


def _stack_or_none(values):
    if values[0] is None:
        return None
    return torch.stack(values, dim=0)


def pref_collate_fn(batch):
    out = {}
    for key in batch[0].keys():
        vals = [b[key] for b in batch]
        out[key] = _stack_or_none(vals) if torch.is_tensor(vals[0]) or vals[0] is None else vals
    return out


def real_collate_fn(batch):
    out = {}
    for key in batch[0].keys():
        vals = [b[key] for b in batch]
        out[key] = _stack_or_none(vals)
    return out


def build_pref_dataloader(pair_csv: str, top_feature_bank: TopFeaturePoolBank, seq_len: int, batch_size: int, shuffle: bool, num_workers: int):
    ds = PreferencePairDataset(pair_csv=pair_csv, top_feature_bank=top_feature_bank, seq_len=seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=pref_collate_fn, drop_last=True)


def build_real_anchor_dataloader(data_path: str, seq_len: int, train_feature_bank: TrainFeatureBank, batch_size: int, shuffle: bool, num_workers: int):
    ds = RealAnchorDataset(data_path=data_path, seq_len=seq_len, train_feature_bank=train_feature_bank)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=real_collate_fn, drop_last=True)
