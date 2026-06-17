# Geometry Analyzer Runbook

Use this after vector and ruler artifacts exist locally. The analyzer does not
run the model and does not recompute activations.

## Purpose

`GeometryAnalyzer` checks whether trait directions are shared across personas or
fragmented by role. The scalar decomposition says how much each role projects on
the benchmark ruler; geometry analysis checks whether the underlying role
vectors point in similar directions.

## Inputs

Per trait:

```text
artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/<trait>/primary_roles/analysis/<run_id>/results/vectors/role_trait_vectors.pt
artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/<trait>/primary_roles/analysis/<run_id>/results/rulers/<trait>_layer8_primary_roles_mean_axis_vector.pt
```

## Dry Run

Use this first to check resolved paths:

```bash
python3 scripts/analysis/run_geometry.py \
  --base-root artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct \
  --output-dir artifacts/reports/five_trait_geometry_v0 \
  --dry-run
```

## Full Five-Trait Run

```bash
python3 scripts/analysis/run_geometry.py \
  --base-root artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct \
  --output-dir artifacts/reports/five_trait_geometry_v0 \
  --layer 8
```

Use explicit run ids if more than one analysis run exists:

```bash
python3 scripts/analysis/run_geometry.py \
  --base-root artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct \
  --output-dir artifacts/reports/five_trait_geometry_v0 \
  --layer 8 \
  --run-id warmth_coldness=20260615T113926Z-warmth_coldness-full \
  --run-id sincerity_manipulativeness=20260615T122554Z-sincerity_manipulativeness-full \
  --run-id caution_recklessness=20260615T124623Z-caution_recklessness-full \
  --run-id curiosity_closed_mindedness=20260615T125133Z-curiosity_closed_mindedness-full \
  --run-id skepticism_gullibility=20260615T125703Z-skepticism_gullibility-full
```

## Outputs

```text
artifacts/reports/five_trait_geometry_v0/geometry_summary.json
artifacts/reports/five_trait_geometry_v0/role_pair_cosines.csv
artifacts/reports/five_trait_geometry_v0/role_ruler_alignment.csv
artifacts/reports/five_trait_geometry_v0/ruler_cosines.csv
artifacts/reports/five_trait_geometry_v0/same_role_cross_trait_cosines.csv
artifacts/reports/five_trait_geometry_v0/pca_summary.csv
artifacts/reports/five_trait_geometry_v0/geometry_summary.md
artifacts/reports/five_trait_geometry_v0/geometry_manifest.json
```

## How to Read It

- `role_pair_cosines.csv`: within each trait, whether role-specific vectors
  point in similar directions.
- `role_ruler_alignment.csv`: whether each role vector aligns with the pooled
  benchmark ruler.
- `ruler_cosines.csv`: whether different trait rulers are distinct or collapsed.
- `same_role_cross_trait_cosines.csv`: whether one role uses similar directions
  for different traits.
- `pca_summary.csv`: whether the vectors are low-dimensional or spread across
  many dimensions.
