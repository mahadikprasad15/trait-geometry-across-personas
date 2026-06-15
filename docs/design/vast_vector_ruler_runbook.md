# Vast Vector and Ruler Runbook

Use this after activation caching has produced:

```text
results/activation_index.jsonl
results/activations/*.pt
```

## Build Vectors

Layer policy for the first 1B smoke run:

```text
layer 8 only
```

Dry-run summary:

```bash
python3 scripts/analysis/build_vectors.py \
  --activation-index artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/activations/20260614T000000Z-full/results/activation_index.jsonl \
  --output-dir artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/analysis/20260614T000000Z-full/results/vectors \
  --layers 8 \
  --dry-run
```

Actual vector build:

```bash
python3 scripts/analysis/build_vectors.py \
  --activation-index artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/activations/20260614T000000Z-full/results/activation_index.jsonl \
  --output-dir artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/analysis/20260614T000000Z-full/results/vectors \
  --trait-axis-id warmth_coldness \
  --layers 8
```

Expected outputs:

```text
results/vectors/role_condition_means.pt
results/vectors/role_trait_vectors.pt
results/vectors/vector_index.json
meta/vector_manifest.json
```

## Build Primary-Role Ruler

The first ruler uses primary roles and the role `axis_vector`:

```text
axis_vector(role) = mean_positive(role) - mean_negative(role)
```

Dry run:

```bash
python3 scripts/analysis/build_rulers.py \
  --role-trait-vectors artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/analysis/20260614T000000Z-full/results/vectors/role_trait_vectors.pt \
  --experiment-config configs/experiments/pilot_v0.yaml \
  --output-dir artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/analysis/20260614T000000Z-full/results/rulers \
  --layer 8 \
  --trait-axis-id warmth_coldness \
  --dry-run
```

Actual ruler build:

```bash
python3 scripts/analysis/build_rulers.py \
  --role-trait-vectors artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/analysis/20260614T000000Z-full/results/vectors/role_trait_vectors.pt \
  --experiment-config configs/experiments/pilot_v0.yaml \
  --output-dir artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/analysis/20260614T000000Z-full/results/rulers \
  --layer 8 \
  --trait-axis-id warmth_coldness
```

Expected outputs:

```text
results/rulers/warmth_coldness_layer8_primary_roles_mean_axis_vector.pt
results/rulers/ruler_index.json
meta/ruler_manifest.json
```

The ruler manifest records:

- selected roles,
- layer,
- vector type,
- raw norm,
- unit norm,
- pairwise role-role cosines.

## Build Role-Free Ruler

First run generation and activation caching for:

```text
data/prompts/warmth_coldness_role_free_v001.jsonl
```

Then build role-free vectors from that role-free activation index, and run:

```bash
python3 scripts/analysis/build_rulers.py \
  --role-trait-vectors artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/role_free/analysis/20260614T000000Z-full/results/vectors/role_trait_vectors.pt \
  --experiment-config configs/experiments/pilot_v0.yaml \
  --output-dir artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/role_free/analysis/20260614T000000Z-full/results/rulers \
  --layer 8 \
  --method role_free_mean
```

This builds a lower-circularity generic ruler from the no-persona `role_free` prompt grid.

## Build Scalar Decomposition

Use this after vectors and a ruler exist. The scalar table projects each role's
offset, positive shift, negative shift, and role-specific axis onto the selected
unit ruler.

Dry run:

```bash
python3 scripts/analysis/build_scalar_decomposition.py \
  --role-condition-means artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/analysis/20260614T000000Z-full/results/vectors/role_condition_means.pt \
  --role-trait-vectors artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/analysis/20260614T000000Z-full/results/vectors/role_trait_vectors.pt \
  --ruler artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/analysis/20260614T000000Z-full/results/rulers/warmth_coldness_layer8_primary_roles_mean_axis_vector.pt \
  --output-dir artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/analysis/20260614T000000Z-full/results/scalars \
  --layer 8 \
  --dry-run
```

Actual scalar build:

```bash
python3 scripts/analysis/build_scalar_decomposition.py \
  --role-condition-means artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/analysis/20260614T000000Z-full/results/vectors/role_condition_means.pt \
  --role-trait-vectors artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/analysis/20260614T000000Z-full/results/vectors/role_trait_vectors.pt \
  --ruler artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/analysis/20260614T000000Z-full/results/rulers/warmth_coldness_layer8_primary_roles_mean_axis_vector.pt \
  --output-dir artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/analysis/20260614T000000Z-full/results/scalars \
  --layer 8
```

Expected outputs:

```text
results/scalars/scalar_decomposition.json
results/scalars/scalar_decomposition.csv
meta/scalar_decomposition_manifest.json
```

Important columns:

- `offset_scalar`: role-neutral baseline projected onto the ruler.
- `positive_shift_scalar`: positive elicitation minus same-role neutral, projected onto the ruler.
- `negative_shift_scalar`: negative elicitation minus same-role neutral, projected onto the ruler.
- `axis_projection_scalar`: positive mean minus negative mean, projected onto the ruler.
- `axis_alignment_cosine`: cosine between this role's axis vector and the unit ruler.
- `mention_shift_scalar`: mention-control shift from neutral, projected onto the ruler if present.

## Run Salience Gate

Use this after scalar decomposition exists. This gate checks whether the scalar
signals look trait-like rather than lexical/prompt-word-like.

Dry run:

```bash
python3 scripts/analysis/run_salience_gate.py \
  --scalar-decomposition artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/analysis/20260614T000000Z-full/results/scalars/scalar_decomposition.json \
  --output-dir artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/analysis/20260614T000000Z-full/results/gates \
  --dry-run
```

Actual gate run:

```bash
python3 scripts/analysis/run_salience_gate.py \
  --scalar-decomposition artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/analysis/20260614T000000Z-full/results/scalars/scalar_decomposition.json \
  --output-dir artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/analysis/20260614T000000Z-full/results/gates
```

Expected outputs:

```text
results/gates/salience_gate.json
results/gates/salience_gate.csv
meta/salience_gate_manifest.json
```

Default pilot thresholds:

```text
min_axis_alignment = 0.2
max_mention_to_shift_ratio = 0.5
warn_if_fail_fraction_at_least = 0.5
```

The gate checks:

- positive elicitation shifts positively along the ruler,
- negative elicitation shifts negatively along the ruler,
- mention-only controls are not too large relative to the larger positive/negative elicitation shift,
- each role axis is at least weakly aligned with the ruler.

## Sync After Analysis

After vectors, rulers, scalars, and gates complete locally, dry-run HF sync:

```bash
python3 scripts/artifacts/sync_to_hf.py \
  --config configs/storage/hf_sync.yaml \
  --local-path artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/analysis/20260614T000000Z-test \
  --commit-message "Dry run sync analysis smoke artifacts" \
  --dry-run
```

Then upload:

```bash
python3 scripts/artifacts/sync_to_hf.py \
  --config configs/storage/hf_sync.yaml \
  --local-path artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/warmth_coldness/primary_roles/analysis/20260614T000000Z-test \
  --commit-message "Sync analysis smoke artifacts"
```
