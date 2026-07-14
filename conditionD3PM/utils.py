"""Provides shared device, seed, and feature-loading helpers."""
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(gpu_id: int = 0):
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"[Device] {device}")
    return device


def load_feature_pth(path: str, name="feature"):

    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"[{name}] feature file does not exist: {path}")

    feats = torch.load(path)
    if isinstance(feats, np.ndarray):
        feats = torch.from_numpy(feats)
    feats = feats.float()
    print(f"[LOAD] {name}: {path}, shape={tuple(feats.shape)}")
    return feats


def get_motif_indices(seq_len: int, motif_span_left: int = 35, motif_span_right: int = 7):


    start = seq_len - motif_span_left
    end = seq_len - motif_span_right + 1
    return start, end


def get_region_indices(seq_len: int):

    return get_motif_indices(seq_len, motif_span_left=35, motif_span_right=7)


def slice_motif_local(motif_feats: torch.Tensor, seq_len: int):


    return motif_feats


def slice_region_local(local_feats: torch.Tensor, seq_len: int):

    return slice_motif_local(local_feats, seq_len)


def expand_global_to_seq(global_feat: torch.Tensor, seq_len: int):

    return global_feat.unsqueeze(1).expand(-1, seq_len, -1)


def film_from_global(global_feat: torch.Tensor, gamma_fc, beta_fc):

    gamma = gamma_fc(global_feat)
    beta = beta_fc(global_feat)
    return gamma, beta


def select_random_cond(global_pool, full_local_pool, motif_local_pool, batch_size):


    cond_global = None
    cond_full_local = None
    cond_motif_local = None

    if global_pool is not None:
        idx = torch.randint(0, global_pool.size(0), (batch_size,))
        cond_global = global_pool[idx]

    if full_local_pool is not None:
        idx = torch.randint(0, full_local_pool.size(0), (batch_size,))
        cond_full_local = full_local_pool[idx]

    if motif_local_pool is not None:
        idx = torch.randint(0, motif_local_pool.size(0), (batch_size,))
        cond_motif_local = motif_local_pool[idx]

    return cond_global, cond_full_local, cond_motif_local
