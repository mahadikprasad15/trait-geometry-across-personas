# Vast Activation Runbook

Use this after a generation run has produced:

```text
results/generations.jsonl
```

## First Target

Model:

```text
meta-llama/Llama-3.2-1B-Instruct
```

Layer policy:

```text
layer 8 only
readout: response_token_mean
```

Reason:

Llama 3.2 1B has 16 layers in current TransformerLens metadata. Layer 8 is the single middle-layer smoke-run target using zero-based layer ids.

## Environment Setup

Install:

```bash
pip install torch transformers transformer_lens accelerate huggingface_hub pyyaml
```

If the model requires gated access:

```bash
export HF_TOKEN=...
```

Do not commit tokens.

## Dry Run

```bash
python3 scripts/activations/cache_activations.py \
  --generations-jsonl artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/generation/20260614T000000Z-test/results/generations.jsonl \
  --model-config configs/models/llama_3_2_1b_instruct.yaml \
  --run-root artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/activations/20260614T000000Z-test \
  --dry-run
```

## Tiny Activation Test

```bash
python3 scripts/activations/cache_activations.py \
  --generations-jsonl artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/generation/20260614T000000Z-test/results/generations.jsonl \
  --model-config configs/models/llama_3_2_1b_instruct.yaml \
  --run-root artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/activations/20260614T000000Z-test \
  --limit 8 \
  --save-every 2
```

Expected outputs:

```text
meta/activation_manifest.json
meta/activation_status.json
checkpoints/activation_progress.json
results/activation_index.jsonl
results/activations/<prompt_id>.pt
logs/activation_cache.log
```

## Full Activation Run

After the tiny test succeeds:

```bash
python3 scripts/activations/cache_activations.py \
  --generations-jsonl artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/generation/20260614T000000Z-full/results/generations.jsonl \
  --model-config configs/models/llama_3_2_1b_instruct.yaml \
  --run-root artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/activations/20260614T000000Z-full \
  --save-every 10
```

## Resume Behavior

Use the same `--run-root`. The runner reads:

```text
results/activation_index.jsonl
checkpoints/activation_progress.json
```

and skips already cached `prompt_id`s whose activation artifact path exists.

## Sync After Activation

After local activation caching completes, dry-run HF sync first:

```bash
python3 scripts/artifacts/sync_to_hf.py \
  --config configs/storage/hf_sync.yaml \
  --local-path artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/activations/20260614T000000Z-test \
  --commit-message "Dry run sync activation smoke artifacts" \
  --dry-run
```

Default upload excludes activation `.pt` tensors and uploads the activation index
plus metadata/checkpoints. For tiny smoke runs, include tensors:

```bash
python3 scripts/artifacts/sync_to_hf.py \
  --config configs/storage/hf_sync.yaml \
  --local-path artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/activations/20260614T000000Z-test \
  --commit-message "Sync activation smoke artifacts with tensors" \
  --include-activations
```

See `docs/design/hf_sync_runbook.md` for the full sync policy.
