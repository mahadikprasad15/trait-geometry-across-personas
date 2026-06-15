# Vast Generation Runbook

Use this when running prompt generation on a model-enabled Vast instance.

## First Target

```text
meta-llama/Llama-3.2-1B-Instruct
```

Config:

```text
configs/models/llama_3_2_1b_instruct.yaml
```

Prompt grid:

```text
data/prompts/warmth_coldness_smoke_v001.jsonl
```

Balanced prompt grid for primary-ruler smoke testing:

```text
data/prompts/warmth_coldness_balanced_smoke_v001.jsonl
```

## Environment Setup

Install the model stack in the Vast environment:

```bash
pip install torch transformers accelerate huggingface_hub pyyaml
```

If the Llama model requires gated access, set:

```bash
export HF_TOKEN=...
```

Do not commit tokens.

## Dry Run

Use this to verify paths and artifact writing without loading the model:

```bash
python3 scripts/generation/run_generation.py \
  --prompt-jsonl data/prompts/warmth_coldness_smoke_v001.jsonl \
  --model-config configs/models/llama_3_2_1b_instruct.yaml \
  --run-root artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/generation/20260614T000000Z-test \
  --limit 8 \
  --dry-run
```

## Tiny Execution Test

Run a tiny generation first:

```bash
python3 scripts/generation/run_generation.py \
  --prompt-jsonl data/prompts/warmth_coldness_smoke_v001.jsonl \
  --model-config configs/models/llama_3_2_1b_instruct.yaml \
  --run-root artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/generation/20260614T000000Z-test \
  --limit 8 \
  --save-every 2
```

Expected outputs:

```text
meta/run_manifest.json
meta/status.json
checkpoints/progress.json
results/generations.jsonl
logs/run.log
```

## Balanced Primary-Ruler Smoke Run

Use this after the first mechanical `--limit 8` test. It covers all four
primary roles and all four smoke conditions with 16 records total, so the
`primary_roles_mean` ruler can be built.

```bash
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-balanced-smoke"
export BASE="artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles"

python3 scripts/generation/run_generation.py \
  --prompt-jsonl data/prompts/warmth_coldness_balanced_smoke_v001.jsonl \
  --model-config configs/models/llama_3_2_1b_instruct.yaml \
  --run-root "$BASE/generation/$RUN_ID" \
  --save-every 4
```

## Full 1B Smoke Run

After the tiny test succeeds, run without `--limit`:

```bash
python3 scripts/generation/run_generation.py \
  --prompt-jsonl data/prompts/warmth_coldness_smoke_v001.jsonl \
  --model-config configs/models/llama_3_2_1b_instruct.yaml \
  --run-root artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/generation/20260614T000000Z-full \
  --save-every 10
```

## Additional Trait Grids

Available full 480-record prompt grids:

```text
data/prompts/warmth_coldness_smoke_v001.jsonl
data/prompts/sincerity_manipulativeness_smoke_v001.jsonl
```

Template for a full-trait generation run:

```bash
export TRAIT_AXIS="sincerity_manipulativeness"
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${TRAIT_AXIS}-full"
export BASE="artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/${TRAIT_AXIS}/primary_roles"

python3 scripts/generation/run_generation.py \
  --prompt-jsonl "data/prompts/${TRAIT_AXIS}_smoke_v001.jsonl" \
  --model-config configs/models/llama_3_2_1b_instruct.yaml \
  --run-root "$BASE/generation/$RUN_ID" \
  --save-every 10
```

## Resume Behavior

The runner reads existing `results/generations.jsonl` and `checkpoints/progress.json`, then skips completed `prompt_id`s.

If the run fails halfway:

```bash
python3 scripts/generation/run_generation.py \
  --prompt-jsonl data/prompts/warmth_coldness_smoke_v001.jsonl \
  --model-config configs/models/llama_3_2_1b_instruct.yaml \
  --run-root <same-run-root> \
  --save-every 10
```

Use the same run root to resume.

## Sync After Generation

After local generation completes, dry-run HF sync first:

```bash
python3 scripts/artifacts/sync_to_hf.py \
  --config configs/storage/hf_sync.yaml \
  --local-path artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/generation/20260614T000000Z-test \
  --commit-message "Dry run sync generation smoke artifacts" \
  --dry-run
```

Then upload:

```bash
python3 scripts/artifacts/sync_to_hf.py \
  --config configs/storage/hf_sync.yaml \
  --local-path artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/generation/20260614T000000Z-test \
  --commit-message "Sync generation smoke artifacts"
```

See `docs/design/hf_sync_runbook.md` for repo setup and upload policy.
