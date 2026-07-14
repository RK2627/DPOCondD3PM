"""Runs Conditional D3PM sampling from configured checkpoints and feature pools."""
import os
import torch

from conditionD3PM.config import SampleConfig, TrainConfig
from conditionD3PM.model import D3PM
from conditionD3PM.noise import AbsorbingMaskScheduler
from conditionD3PM.sampler_cond_d3pm import ancestral_sample_cond
from conditionD3PM.utils import select_device, load_feature_pth, set_seed

ID2DNA = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}


def decode_seq(x):

    return "".join(ID2DNA[int(i)] for i in x)


def sample_cond_d3pm():

    scfg = SampleConfig()
    tcfg = TrainConfig()
    set_seed(scfg.seed)

    device = select_device(scfg.gpu_id)

    print("\n========== SampleConfig ==========")
    print(scfg)
    print("==================================\n")


    model = D3PM(
        vocab_size=tcfg.vocab_size,
        mask_token_id=tcfg.mask_token_id,
        seq_len=scfg.seq_len,
        d_model=tcfg.model_dim,
        num_layers=tcfg.num_layers,
        num_heads=tcfg.num_heads,
        dropout=tcfg.dropout,

        use_global=scfg.use_global,

        use_full_local=scfg.use_full_local,
        use_motif_local=scfg.use_motif_local,

        global_mode=scfg.global_mode,
        local_mode=scfg.local_mode,


        fusion_mode=scfg.fusion_mode,
        cross_heads=scfg.cross_heads,
        cross_dim=scfg.cross_dim,
        cross_layers=scfg.cross_layers,
        gate_channel=scfg.gate_channel,
        motif_span_left=scfg.motif_span_left,
        motif_span_right=scfg.motif_span_right,
    ).to(device)

    print(f"[LOAD] Loading checkpoint: {scfg.ckpt_path}")
    model.load_state_dict(torch.load(scfg.ckpt_path, map_location=device))
    model.eval()


    global_pool = None
    full_local_pool = None
    motif_local_pool = None

    if scfg.use_global:
        global_pool = load_feature_pth(scfg.sample_global_path, "global_pool")
    if scfg.use_full_local:
        full_local_pool = load_feature_pth(scfg.sample_full_local_path, "full_local_pool")
    if scfg.use_motif_local:
        motif_local_pool = load_feature_pth(scfg.sample_motif_local_path, "motif_local_pool")


    scheduler = AbsorbingMaskScheduler(
        num_steps=scfg.num_steps,
        mask_token_id=tcfg.mask_token_id,
        device=device,
    )

    total, bs = scfg.num_samples, scfg.batch_size
    num_batches = (total + bs - 1) // bs
    print(f"[INFO] Generating {total} sequences in {num_batches} batches...\n")

    os.makedirs(os.path.dirname(scfg.out_path), exist_ok=True)


    with open(scfg.out_path, "w") as f:
        for i in range(num_batches):
            cur_bs = min(bs, total - i * bs)


            cond_global = None
            cond_full_local = None
            cond_motif_local = None

            pools = []
            if global_pool is not None:
                pools.append(global_pool.size(0))
            if full_local_pool is not None:
                pools.append(full_local_pool.size(0))
            if motif_local_pool is not None:
                pools.append(motif_local_pool.size(0))

            if len(pools) > 0:

                N = min(pools)
                idx = torch.randint(0, N, (cur_bs,))
                if global_pool is not None:
                    cond_global = global_pool[idx].to(device)
                if full_local_pool is not None:
                    cond_full_local = full_local_pool[idx].to(device)
                if motif_local_pool is not None:
                    cond_motif_local = motif_local_pool[idx].to(device)


            x = ancestral_sample_cond(
                model,
                scheduler,
                seq_len=scfg.seq_len,
                num_steps=scfg.num_steps,
                batch_size=cur_bs,
                device=device,
                cond_global=cond_global,
                cond_full_local=cond_full_local,
                cond_motif_local=cond_motif_local,
                temp=scfg.temp,
            )


            for j in range(cur_bs):
                f.write(decode_seq(x[j].cpu().tolist()) + "\n")

            print(f"[Batch {i+1}/{num_batches}] done.")
            torch.cuda.empty_cache()

    print(f"\n[FINISH] Samples saved to: {scfg.out_path}")


if __name__ == "__main__":
    sample_cond_d3pm()
