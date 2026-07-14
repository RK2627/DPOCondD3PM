"""Loads and samples conditional feature pools for DPO workflows."""
from dataclasses import dataclass
from typing import Optional, Dict

import torch

from conditionD3PM.utils import load_feature_pth


@dataclass
class FeaturePoolSet:
    global_pool: Optional[torch.Tensor]
    full_local_pool: Optional[torch.Tensor]
    motif_local_pool: Optional[torch.Tensor]

    @property
    def size(self) -> int:
        pools = [
            p.size(0)
            for p in [self.global_pool, self.full_local_pool, self.motif_local_pool]
            if p is not None
        ]
        if not pools:
            raise ValueError("No condition feature pool is enabled")
        return min(pools)


class TopFeaturePoolBank:
    def __init__(self, global_path=None, full_local_path=None, motif_local_path=None, device=None):
        self.device = device
        self.pools = FeaturePoolSet(
            global_pool=load_feature_pth(global_path, "top_global_pool") if global_path else None,
            full_local_pool=load_feature_pth(full_local_path, "top_full_local_pool") if full_local_path else None,
            motif_local_pool=load_feature_pth(motif_local_path, "top_motif_local_pool") if motif_local_path else None,
        )
        print(f"[TopFeaturePoolBank] usable_size={self.pools.size}")

    def get_by_indices(self, indices: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        out = {
            "cond_global": self.pools.global_pool[indices] if self.pools.global_pool is not None else None,
            "cond_full_local": self.pools.full_local_pool[indices] if self.pools.full_local_pool is not None else None,
            "cond_motif_local": self.pools.motif_local_pool[indices] if self.pools.motif_local_pool is not None else None,
        }
        if self.device is not None:
            for k, v in out.items():
                if v is not None:
                    out[k] = v.to(self.device)
        return out

    def sample_condition_indices(self, num_conditions: int, unique: bool = True,
                                 seed: Optional[int] = None) -> torch.Tensor:
        n = self.pools.size
        g = None
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(seed)

        actual_num = num_conditions
        if unique:
            actual_num = min(num_conditions, n)
            if actual_num < num_conditions:
                print(
                    f"[TopFeaturePoolBank] requested num_conditions={num_conditions}, "
                    f"but usable_size={n}. Auto-adjust to {actual_num}."
                )
            idx = torch.randperm(n, generator=g)[:actual_num]
        else:
            idx = torch.randint(0, n, (num_conditions,), generator=g)

        return idx.long()


class TrainFeatureBank:
    def __init__(self, global_path=None, full_local_path=None, motif_local_path=None, device=None):
        self.device = device
        self.global_feats = load_feature_pth(global_path, "train_global_feats") if global_path else None
        self.full_local_feats = load_feature_pth(full_local_path, "train_full_local_feats") if full_local_path else None
        self.motif_local_feats = load_feature_pth(motif_local_path, "train_motif_local_feats") if motif_local_path else None

    def get_by_indices(self, indices: torch.Tensor):
        out = {
            "cond_global": self.global_feats[indices] if self.global_feats is not None else None,
            "cond_full_local": self.full_local_feats[indices] if self.full_local_feats is not None else None,
            "cond_motif_local": self.motif_local_feats[indices] if self.motif_local_feats is not None else None,
        }
        if self.device is not None:
            for k, v in out.items():
                if v is not None:
                    out[k] = v.to(self.device)
        return out
