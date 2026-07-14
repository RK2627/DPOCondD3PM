# DPOCondD3PM

## Overview

DPOCondD3PM generates 165-bp promoter sequences with Conditional D3PM and
improves candidate preferences through full-model offline DPO. Species-specific
feature encoders produce global, full-local, and motif-local conditions, while
the corresponding activity predictors score generated sequences.

## Workflow

1. Extract the three condition features with the supplied feature encoder.
2. Build the first-5% high-activity feature pools when required.
3. Train or sample Conditional D3PM with the selected species configuration.
4. Score candidates, construct preference pairs, and run full-model DPO.
5. Sample promoter sequences from a DPO checkpoint.

## Repository structure

```text
data/             E. coli and Actinobacteria datasets
Feature/          pretrained feature encoders and feature extraction
conditionD3PM/    conditional diffusion training and sampling
DPO/              candidate generation, preference pairs, and full-model DPO
Predictor/        E. coli and Actinobacteria activity predictors
```

## Environment

```bash
conda create -n dpocondd3pm python=3.10
conda activate dpocondd3pm
pip install -r requirements.txt
```

Run all commands from the repository root. Git LFS is required for the bundled
model checkpoints and predictor weights.

## Usage

### 1. Feature extraction

The repository provides the encoder checkpoints under `Feature/ecoli/` and
`Feature/actinobacteria/`. Extracted feature tensors are generated locally and
are not included in the repository.

E. coli:

```bash
python -m Feature.feature --mode extract --csv data/ecoli/ecoli_exp.csv --ckpt Feature/ecoli/feature_encoder_ecoli.pth --feature_type global --out_pth Feature/ecoli/global.pth
python -m Feature.feature --mode extract --csv data/ecoli/ecoli_exp.csv --ckpt Feature/ecoli/feature_encoder_ecoli.pth --feature_type full_local --out_pth Feature/ecoli/full_local.pth
python -m Feature.feature --mode extract --csv data/ecoli/ecoli_exp.csv --ckpt Feature/ecoli/feature_encoder_ecoli.pth --feature_type motif_local --out_pth Feature/ecoli/motif_local.pth
```

Actinobacteria:

```bash
python -m Feature.feature --mode extract --csv data/actinomycetes/ac_exp.csv --ckpt Feature/actinobacteria/feature_encoder_ac.pth --feature_type global --out_pth Feature/actinobacteria/global.pth
python -m Feature.feature --mode extract --csv data/actinomycetes/ac_exp.csv --ckpt Feature/actinobacteria/feature_encoder_ac.pth --feature_type full_local --out_pth Feature/actinobacteria/full_local.pth
python -m Feature.feature --mode extract --csv data/actinomycetes/ac_exp.csv --ckpt Feature/actinobacteria/feature_encoder_ac.pth --feature_type motif_local --out_pth Feature/actinobacteria/motif_local.pth
```

Input: the activity-ordered sequence/activity CSV and the matching encoder
checkpoint. Output: `global.pth`, `full_local.pth`, and `motif_local.pth` in the
selected `Feature/<species>/` directory.

When a first-5% high-activity pool is required, run `extract_top5` for each
feature tensor. For example:

```bash
python -c "from Feature.extract_top5 import extract_top5; extract_top5('Feature/ecoli/global.pth', 'Feature/ecoli/global_top5.pth')"
```

Repeat with `full_local` and `motif_local`, and use the corresponding species
directory. This operation assumes that the input data and extracted features
are ordered from highest to lowest activity.

### 2. Activity predictors

E. coli:

```bash
python -m Predictor.ecoli_predictor.predictor_train
```

Actinobacteria:

```bash
python -m Predictor.actinobacteria_predictor.Predictor_train --help
```

Configure the data and checkpoint paths in the corresponding predictor entry.
Predictor input is a sequence set; output is a trained checkpoint or predicted
activity score, depending on the selected operation.

### 3. Conditional D3PM

Before running, edit `conditionD3PM/config.py` so the data, feature, checkpoint,
and output paths refer to one species. Point enabled feature paths to the files
generated under `Feature/ecoli/` or `Feature/actinobacteria/`.

```bash
python -m conditionD3PM.train_cond_d3pm
python -m conditionD3PM.sample_cond_d3pm
```

Training produces a Conditional D3PM checkpoint. Sampling produces generated
165-bp promoter sequences.

### 4. Full-model offline DPO

Edit `DPO/config.py` so the data, full and first-5% feature paths, Conditional
D3PM base checkpoint, predictor, and output paths all refer to the same species.
The feature paths may point directly to the tensors generated under `Feature/`.

```bash
python -m DPO.build_offline_candidates
python -m DPO.build_offline_preference_pairs
python -m DPO.train_full_model_offline_preference
python -m DPO.sample_offline_preference
```

These steps generate scored candidates, preference pairs, full-model DPO
checkpoints, and final promoter samples.

## Third-party attribution

See `THIRD_PARTY_NOTICES.md` for the ProkBERT and ProkBERT-mini components used
by the Actinobacteria predictor.
