# Reporting Runbook

Use this after the five-trait pilot artifacts have been downloaded or synced
locally.

## Multi-Trait Scalar/Gate Summary

The first summary script reads existing scalar and salience-gate outputs. It
does not recompute activations, vectors, rulers, or scalar projections.

Inputs per trait:

```text
artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/<trait>/primary_roles/analysis/<run_id>/results/scalars/scalar_decomposition.json
artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/<trait>/primary_roles/analysis/<run_id>/results/gates/salience_gate.json
```

Run:

```bash
python3 scripts/reporting/summarize_traits.py \
  --base-root artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct \
  --output-dir artifacts/reports/five_trait_pilot_v0
```

Expected outputs:

```text
artifacts/reports/five_trait_pilot_v0/multi_trait_summary.json
artifacts/reports/five_trait_pilot_v0/trait_summary.csv
artifacts/reports/five_trait_pilot_v0/role_scalar_gate_summary.csv
artifacts/reports/five_trait_pilot_v0/multi_trait_summary.md
```

If multiple analysis runs exist for a trait, pass explicit run IDs:

```bash
python3 scripts/reporting/summarize_traits.py \
  --base-root artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct \
  --output-dir artifacts/reports/five_trait_pilot_v0 \
  --run-id warmth_coldness=20260615T113926Z-warmth_coldness-full \
  --run-id sincerity_manipulativeness=20260615T122554Z-sincerity_manipulativeness-full \
  --run-id caution_recklessness=20260615T124623Z-caution_recklessness-full \
  --run-id curiosity_closed_mindedness=20260615T125133Z-curiosity_closed_mindedness-full \
  --run-id skepticism_gullibility=20260615T125703Z-skepticism_gullibility-full
```

## Scalar-Behavior Summary

Use this after behavior metrics exist. This report joins three artifact streams:

```text
scalar_decomposition.json
salience_gate.json
behavior_metrics.json
```

Run:

```bash
python3 scripts/reporting/summarize_scalar_behavior.py \
  --base-root artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct \
  --output-dir artifacts/reports/five_trait_behavior_v0
```

Expected outputs:

```text
artifacts/reports/five_trait_behavior_v0/scalar_behavior_summary.json
artifacts/reports/five_trait_behavior_v0/scalar_behavior_trait_summary.csv
artifacts/reports/five_trait_behavior_v0/scalar_behavior_role_summary.csv
artifacts/reports/five_trait_behavior_v0/scalar_behavior_summary.md
```

If analysis and behavior run ids differ, pass them separately:

```bash
python3 scripts/reporting/summarize_scalar_behavior.py \
  --base-root artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct \
  --output-dir artifacts/reports/five_trait_behavior_v0 \
  --analysis-run-id warmth_coldness=20260615T113926Z-warmth_coldness-full \
  --behavior-run-id warmth_coldness=20260615T113926Z-warmth_coldness-full
```

## Integrated Report

Use this after any of the summary artifacts exist. By default the report builder
is tolerant: missing sections are recorded as missing instead of failing the
run. This lets the same command produce an activation-only report now and a
full activation/geometry/behavior report later.

Run after scalar/gate and geometry summaries:

```bash
python3 scripts/reporting/build_report.py \
  --output-dir artifacts/reports/integrated_pilot_v0
```

Expected default inputs:

```text
artifacts/reports/five_trait_pilot_v0/multi_trait_summary.json
artifacts/reports/five_trait_geometry_v0/geometry_summary.json
artifacts/reports/five_trait_behavior_v0/scalar_behavior_summary.json
```

Expected outputs:

```text
artifacts/reports/integrated_pilot_v0/integrated_report.json
artifacts/reports/integrated_pilot_v0/integrated_report.md
artifacts/reports/integrated_pilot_v0/integrated_report_manifest.json
```

Use `--strict` when all enabled sections should already exist:

```bash
python3 scripts/reporting/build_report.py \
  --output-dir artifacts/reports/integrated_pilot_v0 \
  --strict
```

Use explicit paths if you choose different report directories:

```bash
python3 scripts/reporting/build_report.py \
  --output-dir artifacts/reports/integrated_pilot_v0 \
  --scalar-gate-summary artifacts/reports/five_trait_pilot_v0/multi_trait_summary.json \
  --geometry-summary artifacts/reports/five_trait_geometry_v0/geometry_summary.json \
  --scalar-behavior-summary artifacts/reports/five_trait_behavior_v0/scalar_behavior_summary.json
```
