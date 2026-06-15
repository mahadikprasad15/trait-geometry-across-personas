# Hugging Face Artifact Sync Runbook

Use this after each Vast stage completes locally. The experiment runners should
write local artifacts first; HF sync is a separate upload step.

## Repo Setup

Default configured dataset repo:

```text
prasadmahadik/trait-geometry-across-personas
```

Config:

```text
configs/storage/hf_sync.yaml
```

Install dependency on Vast:

```bash
pip install huggingface_hub pyyaml
```

Set your token:

```bash
export HF_TOKEN=...
```

Do not commit tokens.

If the repo does not exist yet, either create it on the HF website as a dataset
repo, or add `--create-repo` to the first upload command.

## Dry Run First

Always inspect the selected files before upload:

```bash
python3 scripts/artifacts/sync_to_hf.py \
  --config configs/storage/hf_sync.yaml \
  --local-path artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/generation/20260614T000000Z-test \
  --commit-message "Dry run sync generation smoke artifacts" \
  --dry-run
```

Dry runs write a local sync manifest under:

```text
artifacts/sync_manifests/
```

## Sync Configs and Prompt Grids

Run this before or after the first generation run so the HF repo has the exact
inputs used for the run:

```bash
python3 scripts/artifacts/sync_to_hf.py \
  --config configs/storage/hf_sync.yaml \
  --local-path configs \
  --commit-message "Sync experiment configs"

python3 scripts/artifacts/sync_to_hf.py \
  --config configs/storage/hf_sync.yaml \
  --local-path data/prompts \
  --commit-message "Sync prompt grids"
```

## Sync Generation Artifacts

```bash
python3 scripts/artifacts/sync_to_hf.py \
  --config configs/storage/hf_sync.yaml \
  --local-path artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/generation/20260614T000000Z-test \
  --commit-message "Sync generation smoke artifacts"
```

Uploads include metadata, checkpoints, inputs, and `results/generations.jsonl`.
Logs are excluded by default.

## Sync Activation Artifacts

Default activation sync uploads metadata, checkpoints, inputs, and
`results/activation_index.jsonl`. Full activation `.pt` tensors are excluded by
default because they can become large.

```bash
python3 scripts/artifacts/sync_to_hf.py \
  --config configs/storage/hf_sync.yaml \
  --local-path artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/activations/20260614T000000Z-test \
  --commit-message "Sync activation smoke index"
```

For tiny smoke runs, include activation tensors:

```bash
python3 scripts/artifacts/sync_to_hf.py \
  --config configs/storage/hf_sync.yaml \
  --local-path artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/activations/20260614T000000Z-test \
  --commit-message "Sync activation smoke tensors" \
  --include-activations
```

## Sync Analysis Artifacts

After vectors, rulers, scalars, and gates:

```bash
python3 scripts/artifacts/sync_to_hf.py \
  --config configs/storage/hf_sync.yaml \
  --local-path artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/analysis/20260614T000000Z-test \
  --commit-message "Sync analysis smoke artifacts"
```

Uploads include:

```text
results/vectors/**
results/rulers/**
results/scalars/**
results/gates/**
meta/**
checkpoints/**
inputs/**
```

## Override Repo Name

If you choose a different HF repo:

```bash
python3 scripts/artifacts/sync_to_hf.py \
  --config configs/storage/hf_sync.yaml \
  --repo-id yourname/your-dataset-repo \
  --local-path artifacts/runs/... \
  --commit-message "Sync artifacts"
```

## Policy Notes

- HF sync never runs model generation or activation extraction.
- Upload failures should not change local experiment artifacts.
- Dry-run manifests are useful for checking accidental large uploads.
- Full activation tensor upload is opt-in with `--include-activations`.
