"""Creates compact top-5% feature pools from activity-ordered tensors."""

import os
import torch

def extract_top5(input_pth, output_pth):

    feats = torch.load(input_pth, map_location="cpu")
    if not torch.is_tensor(feats):
        feats = torch.tensor(feats)

    N = feats.size(0)
    top5 = int(N * 0.05)

    print(f"[INFO] Total features: {N}; first 5%: {top5}")

    top_feats = feats[:top5].detach().clone().contiguous()

    output_dir = os.path.dirname(output_pth)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(top_feats, output_pth)

    print(f"[SAVE] Saved the first 5% of features to: {output_pth}")


if __name__ == "__main__":

    input_pth = "./conditionD3PM/features/actinobacteria/full_local.pth"
    output_pth = "./Feature/outputs/full_local_top5.pth"

    extract_top5(input_pth, output_pth)
