# Behavior Judging Runbook

Use this after a generation run has produced `results/generations.jsonl`.

Behavior judging scores generated completions. It does not load the generation
model, cache activations, or compute representation vectors.

## Dry Run

```bash
export TRAIT_AXIS="warmth_coldness"
export RUN_ID="20260615T113926Z-warmth_coldness-full"
export BASE="artifacts/runs/trait_geometry_pilot_v0/llama_3_2_1b_instruct/${TRAIT_AXIS}/primary_roles"

python3 scripts/analysis/run_trait_judge.py \
  --generations-jsonl "$BASE/generation/$RUN_ID/results/generations.jsonl" \
  --trait-config "configs/traits/${TRAIT_AXIS}.yaml" \
  --rubric-config configs/judges/trait_behavior_rubric_v001.yaml \
  --judge-config configs/models/judge_openai_gpt_4_1_mini.yaml \
  --run-root "$BASE/judging/$RUN_ID" \
  --limit 8 \
  --dry-run
```

Expected dry-run outputs:

```text
meta/judge_manifest.json
meta/judge_status.json
checkpoints/judge_progress.json
inputs/judge_preview.json
logs/judge.log
```

## Execute Small Pilot

Set:

```bash
export OPENAI_API_KEY=...
```

Then run a small first pass:

```bash
python3 scripts/analysis/run_trait_judge.py \
  --generations-jsonl "$BASE/generation/$RUN_ID/results/generations.jsonl" \
  --trait-config "configs/traits/${TRAIT_AXIS}.yaml" \
  --rubric-config configs/judges/trait_behavior_rubric_v001.yaml \
  --judge-config configs/models/judge_openai_gpt_4_1_mini.yaml \
  --run-root "$BASE/judging/$RUN_ID" \
  --limit 32 \
  --save-every 8
```

Inspect:

```bash
wc -l "$BASE/judging/$RUN_ID/results/judgments.jsonl"
cat "$BASE/judging/$RUN_ID/meta/judge_status.json"
```

## Full Trait Run

After the first judged rows look sane, rerun without `--limit` using the same
run root:

```bash
python3 scripts/analysis/run_trait_judge.py \
  --generations-jsonl "$BASE/generation/$RUN_ID/results/generations.jsonl" \
  --trait-config "configs/traits/${TRAIT_AXIS}.yaml" \
  --rubric-config configs/judges/trait_behavior_rubric_v001.yaml \
  --judge-config configs/models/judge_openai_gpt_4_1_mini.yaml \
  --run-root "$BASE/judging/$RUN_ID" \
  --save-every 10
```

The runner resumes by `prompt_id`, using:

```text
results/judgments.jsonl
checkpoints/judge_progress.json
```

## Output Fields

Each judgment row includes:

```text
positive_pole_score
negative_pole_score
role_adherence_score
coherence_score
prompt_following_score
trait_word_discussion_score
positive_evidence
negative_evidence
role_adherence_evidence
rationale
```

These rows feed the future `BehaviorMetricsBuilder`.

## Build Behavior Metrics

After `judgments.jsonl` exists, aggregate the raw judgment rows:

```bash
python3 scripts/analysis/build_behavior_metrics.py \
  --judgments-jsonl "$BASE/judging/$RUN_ID/results/judgments.jsonl" \
  --run-root "$BASE/behavior/$RUN_ID"
```

Expected outputs:

```text
behavior/<run_id>/results/behavior_metrics.json
behavior/<run_id>/results/behavior_metrics.csv
behavior/<run_id>/meta/behavior_metrics_manifest.json
```

The builder computes:

- condition means by trait and role,
- positive-pole behavior shift,
- negative-pole behavior shift,
- mention-control behavior shifts,
- matched-pair shifts when matched neutral ids are available,
- role adherence, coherence, and prompt-following quality means.
