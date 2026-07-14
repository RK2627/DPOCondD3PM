"""Loads and encodes promoter sequences for Conditional D3PM."""
import torch
from torch.utils.data import Dataset, DataLoader

DNA_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


class DNALineDataset(Dataset):


    def __init__(self, path: str, seq_len: int):
        self.seq_len = seq_len
        self.data = []

        with open(path, "r") as f:
            for line in f:
                s = line.strip().split(",")[0].upper()
                if not s:
                    continue
                ids = [DNA_MAP.get(ch, 0) for ch in s]
                if len(ids) >= seq_len:
                    ids = ids[:seq_len]
                else:
                    ids = ids + [0] * (seq_len - len(ids))
                self.data.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):


        return self.data[idx], idx


def build_dataloader(path, seq_len, batch_size, shuffle=True, num_workers=2):


    ds = DNALineDataset(path, seq_len)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )
