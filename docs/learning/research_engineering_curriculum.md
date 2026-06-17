# Research Engineering Curriculum Notes

This document is the running learning record for the project. Each implementation step should add concepts, patterns, and reusable lessons that can transfer to future repos.

## Learning Goal

Build the project in a way that teaches reusable research-engineering patterns:

- Python project structure,
- typed data objects,
- config-driven experiments,
- durable artifacts,
- resumable execution,
- PyTorch tensor handling,
- TransformerLens activation extraction,
- probe and vector workflows,
- structured evaluation,
- report generation.

## Reusable Pattern: Config to Artifact to Audit

Most stages should follow this shape:

```text
config
  -> builder/runner/analyzer
  -> artifact files
  -> manifest/status
  -> read-back validation
  -> report
```

This pattern prevents one-off notebook state from becoming the only source of truth.

## Component Types

### Builder

Constructs a durable artifact from inputs.

Examples:

- `PromptGridBuilder`
- `ActivationCacheBuilder`
- `RulerBuilder`
- `VectorBuilder`
- `ReportBuilder`

Learning focus:

- dataclass configs,
- path planning,
- deterministic outputs,
- validation before write,
- manifest writing.

### Runner

Executes an expensive or external process.

Examples:

- `GenerationRunner`
- `TraitJudgeRunner`
- `SteeringRunner`

Learning focus:

- batching,
- progress checkpoints,
- resume/skip behavior,
- logging,
- error recovery,
- rate or memory constraints.

### Analyzer

Consumes artifacts and produces metrics.

Examples:

- `ScalarDecompositionBuilder`
- `GeometryAnalyzer`
- `SaturationAnalyzer`
- `CompositionalityAnalyzer`

Learning focus:

- tensor shapes,
- grouping by metadata,
- numerical validation,
- plotting,
- interpretation boundaries.

### Gate

Makes a proceed/pivot decision.

Examples:

- salience gate,
- ruler convergence gate,
- sanity steering gate,
- register-artifact gate.

Learning focus:

- explicit thresholds,
- decision records,
- negative results,
- avoiding silent patching.

## Core Engineering Concepts to Learn

### Dataclass Configs

Use dataclasses or structured config objects to keep scripts parameterized.

Example concepts:

- required vs optional fields,
- default values,
- validation in `__post_init__`,
- serializing config into manifests.

### Pathlib and Artifact Roots

Use `pathlib.Path` for all paths. Keep outputs inside `artifacts/runs/...`.

Reusable idea:

```text
resolve paths once from config, then pass path objects to stages
```

### JSON, JSONL, YAML

Use:

- YAML for human-authored configs,
- JSON for structured summaries and manifests,
- JSONL for per-record prompt/generation/judgment data.

### Manifest and Status Files

Every durable run needs:

- immutable `run_manifest.json`,
- mutable `status.json`,
- resumable `progress.json`.

Learning focus:

- difference between manifest and status,
- why status changes but manifest should not,
- how progress supports resume.

### Tensor Shapes

Track shapes explicitly.

Examples:

```text
tokens: [batch, seq]
residual activations: [batch, seq, d_model]
pooled activations: [batch, d_model]
role vectors: [d_model]
layerwise vectors: [layer, d_model]
```

Each analyzer should document expected tensor shapes.

### TransformerLens Patterns

Concepts to learn:

- loading a model,
- tokenization,
- forward hooks or cached activations,
- residual stream naming,
- layer selection,
- position/span readout,
- memory-aware batching.

### Probe Patterns

Concepts to learn:

- train/test splits,
- role-held-out evaluation,
- logistic probe direction,
- probe direction vs diff-in-means direction,
- per-role transfer matrix.

### Planned Execution

For larger runs, execution should be planned before execution:

```text
build work units
write inputs snapshot
run units with checkpoints
aggregate completed units
mark status complete
```

This keeps large experiment matrices debuggable.

### Subprocess Patterns

Use subprocess orchestration only when needed, for example when launching separate model-generation jobs.

Learning focus:

- command construction from configs,
- logging stdout/stderr,
- exit-code checks,
- not hiding failures,
- writing run logs.

## Project-Specific Concept Ladder

### Level 1: Prompt and Metadata

Learn:

- prompt schemas,
- matched pairs,
- scenario ids,
- trait-word leakage checks,
- JSONL records.

Build:

- `PromptGridBuilder`.

### Level 2: Persistence and Resume

Learn:

- artifact paths,
- manifest/status/progress,
- read-back validation.

Build:

- run directory creation,
- status updates,
- checkpoint saves.

### Level 3: Activations and Tensors

Learn:

- activation cache shape,
- layer and position extraction,
- mean pooling,
- token spans.

Build:

- `ActivationCacheBuilder`,
- `PositionReadoutExtractor`.

### Level 4: Vectors and Rulers

Learn:

- diff-in-means,
- centering,
- unit normalization,
- projection,
- cosine similarity.

Build:

- `VectorBuilder`,
- `RulerBuilder`,
- scalar projection functions.

### Level 5: Validation Gates

Learn:

- salience control,
- convergence test,
- causal sanity steering,
- register artifact control.

Build:

- `ValidationGateRunner`,
- decision records.

### Level 6: Behavior and Judging

Learn:

- structured judge rubrics,
- behavior baselines,
- behavior shifts,
- rating audits.

Build:

- `TraitJudgeRunner`,
- `BehaviorMetricsBuilder`.

### Level 7: Geometry

Learn:

- pairwise cosine matrix,
- PCA,
- residual PCA,
- probe-vs-vector comparison.

Build:

- `GeometryAnalyzer`.

### Level 8: Steering

Learn:

- activation interventions,
- alpha schedules,
- dose-response curves,
- quality-constrained steering.

Build:

- `SteeringRunner`,
- `DoseResponseBuilder`.

## Running Notes Template

Use this template when adding a new learning entry:

```text
## <Date> - <Concept or Component>

Concept:

Where it appears in this repo:

Why it matters:

Reusable pattern:

Implementation notes:

Common failure modes:

Related previous pattern:
```

## 2026-06-16 - Five-Trait Pilot Research-Engineering Lessons

Concept:

The first complete vertical slice used a chain of small, artifact-first
components rather than one large experiment script:

```text
PromptGridBuilder
  -> GenerationRunner
  -> ActivationCacheBuilder
  -> VectorBuilder
  -> RulerBuilder
  -> ScalarDecompositionBuilder
  -> SalienceGateRunner
  -> HfArtifactSyncRunner
  -> MultiTraitSummaryBuilder
```

Each component has a narrow contract: read structured inputs, produce durable
outputs, and write enough metadata for resume, audit, and downstream use.

Where it appears in this repo:

- `scripts/prompts/build_prompt_grid.py`
- `scripts/generation/run_generation.py`
- `scripts/activations/cache_activations.py`
- `scripts/analysis/build_vectors.py`
- `scripts/analysis/build_rulers.py`
- `scripts/analysis/build_scalar_decomposition.py`
- `scripts/analysis/run_salience_gate.py`
- `scripts/artifacts/sync_to_hf.py`
- `scripts/reporting/summarize_traits.py`

Why it matters:

The experiment became debuggable because every stage wrote inspectable files.
When a later stage failed, we could locate the exact missing artifact or bad
metadata rather than rerunning the whole model pipeline.

Reusable pattern:

```text
load config/artifacts
build work items
check dependencies
create run dirs
load completed ids from outputs/checkpoints
process remaining work
append per-record outputs
write progress/status
write final manifest/status
```

Implementation notes:

- CLI wrappers under `scripts/` are thin entrypoints. Most logic lives in
  importable package modules under `src/trait_geometry/`.
- YAML is for human-authored configuration and prompt specs.
- JSONL is for per-record data such as prompts, generations, activations, and
  future judgments.
- JSON is for manifests, summaries, status files, and reports.
- `.pt` files hold tensor payloads when JSON would lose tensor type/shape.
- Each downstream stage should preserve upstream metadata that may be needed
  later, especially `trait_axis_id`, `role_id`, `condition`, `scenario_id`,
  layer, method, and readout policy.

Common failure modes:

- Hardcoding a smoke-run trait id caused multi-trait rulers to be written with
  the wrong filename. Fix: carry `trait_axis_id` in vector/ruler payloads and
  expose `--trait-axis-id` for older artifacts.
- Shell variable typos can silently create wrong paths, such as a `RUN_ID`
  glued to the word `export`. Fix: echo variables and check expected files
  before running expensive stages.
- Local docs can become stale after remote Vast runs. Fix: update the tracker
  after every successful run batch and sync.

Related previous pattern:

This generalizes the earlier `config -> artifact -> audit` pattern into a full
experiment chain with resumable model execution, tensor artifacts, validation
gates, and reporting.

### Builder, Runner, Analyzer, Gate In This Repo

Builder:

Creates a reusable artifact from structured inputs. Examples:

- `PromptGridBuilder`: configs/specs -> prompt JSONL/manifest.
- `VectorBuilder`: activation index + `.pt` tensors -> condition means and
  role trait vectors.
- `RulerBuilder`: role vectors -> unit ruler.
- `MultiTraitSummaryBuilder`: scalar/gate JSONs -> summary tables.

Runner:

Executes expensive or external work and must support resume. Examples:

- `GenerationRunner`: model inference over prompt records.
- `ActivationCacheBuilder`: TransformerLens forward/cache over generated text.
- `TraitJudgeRunner`: judge-model calls over generated completions.
- Future `ConstantSteeringRunner`: intervention runs over prompts.

Analyzer:

Consumes artifacts and computes derived measurements. Examples:

- `ScalarDecompositionBuilder`: projects offsets/shifts/axes onto a ruler.
- `GeometryAnalyzer`: computes role-role and trait-trait geometry.
- `BehaviorMetricsBuilder`: converts judgment rows into behavior baselines and
  shifts.

Gate:

Turns measurements into a proceed/pivot signal. Example:

- `SalienceGateRunner`: checks direction signs, mention-control size, and axis
  alignment. A failing gate is a finding, not something to hide.

### Helper Function Anatomy

Most modules use the same helper categories:

- `load_*`: read YAML, JSON, JSONL, or `.pt` payloads.
- `check_*_dependency`: fail early with useful install guidance.
- `make_run_dirs`: create canonical `inputs/`, `results/`, `checkpoints/`,
  `logs/`, and `meta/` directories.
- `completed_*_from_*`: discover already-finished records for resume.
- `build_*_records` or `build_work_items`: convert raw input rows into typed
  work units.
- `write_*_artifacts`: write final JSON/CSV/PT files and manifests.
- `main`: parse CLI args and orchestrate the helper graph.

The reusable lesson is that helpers should expose the pipeline shape. A reader
should be able to scan function names and understand the stage lifecycle.

### Resumability Pattern

Expensive runners use two resume sources:

```text
results file/index
checkpoint progress file
```

For generation:

- Output: `results/generations.jsonl`
- Progress: `checkpoints/progress.json`
- Resume key: `prompt_id`

For activation caching:

- Output: `results/activation_index.jsonl`
- Progress: `checkpoints/activation_progress.json`
- Resume key: `prompt_id`
- Extra check: the activation `.pt` path must exist before a prompt counts as
  complete.

This avoids duplicate work after interruptions and keeps partial Vast runs
recoverable.

### Manifest vs Status vs Progress

Manifest:

- mostly immutable,
- records provenance and intended inputs,
- answers "what was this run supposed to be?"

Status:

- mutable,
- records running/completed/failed state,
- answers "where did the run end?"

Progress:

- mutable checkpoint,
- records completed work units,
- answers "what can be skipped on resume?"

This separation is useful because a failed run can still have a valid manifest
and partial progress.

### Transformers Generation Batching

Generation batching used left padding.

Reason:

Decoder-only generation appends new tokens after the padded input width. With
left padding, all prompts in the batch share one input width, and decoding the
completion is simply:

```text
generated_tokens = output_ids[input_width:]
```

Important details:

- Set `tokenizer.pad_token = tokenizer.eos_token` if no pad token exists.
- Set `tokenizer.padding_side = "left"` for generation.
- Keep `batch_size` in the effective generation config and manifest.
- If a batch fails, record all prompt ids in that failed batch.

### TransformerLens Activation Batching

Activation caching used right padding.

Reason:

We are not generating; we are reading residual-stream activations over existing
full text. With right padding, each row's real response span remains:

```text
prompt_len:full_len
```

Padding appears after the real text and does not shift the response span.

Important details:

- Compute `prompt_len` and `full_len` per item before batching.
- Run `model.run_with_cache` on the padded batch.
- For each row, pool only `cache[name][row_idx, prompt_len:full_len, :]`.
- Preserve one `.pt` artifact and one index row per prompt even when the model
  forward pass is batched. This keeps downstream scripts unchanged.

### Tensor Shape Contracts

Generation:

```text
input_ids: [batch, seq]
outputs: [batch, seq + new_tokens]
```

Activation cache:

```text
full_tokens_batch: [batch, seq]
cache[blocks.L.hook_resid_post]: [batch, seq, d_model]
response span: [response_tokens, d_model]
pooled response activation: [d_model]
```

Vectors:

```text
condition means: role x condition x [d_model]
positive_shift: positive_mean - neutral_mean
negative_shift: negative_mean - neutral_mean
axis_vector: positive_mean - negative_mean
offset_vector: role_neutral - global_neutral_mean
```

Rulers:

```text
selected role axes: roles x [d_model]
raw ruler: mean(selected role axes)
unit ruler: raw ruler / ||raw ruler||
```

Scalars:

```text
scalar = vector dot unit_ruler
axis_alignment = cosine(role_axis, unit_ruler)
```

### Metadata Propagation Lesson

The multi-trait run exposed a classic research-engineering bug: an early script
used the experiment config's smoke-run trait id for all rulers. That worked for
warmth/coldness and failed silently for other traits until scalar decomposition
looked for the trait-specific filename.

The durable fix:

- activation index rows include `trait_axis_id`,
- vector payloads include `trait_axis_id`,
- ruler builder resolves trait id from payload first,
- CLI supports `--trait-axis-id` for older artifacts,
- downstream filenames use the resolved trait id.

Reusable rule:

Any field that determines artifact identity should travel inside the artifact,
not only live in an external config.

### HF Sync and Artifact Policy

Two sync modes were useful:

- Lightweight sync: generations, indexes, manifests, scalar/gate outputs.
- Raw activation sync: includes heavy `.pt` activation tensors.

The lightweight sync is enough for reporting over existing scalar/gate outputs.
Raw tensors are needed for later probe training, new pooling policies, different
layers, residual PCA, or prompt-level geometry.

The sync manifest is itself an artifact. It records file count, repo paths, and
the HF commit URL, making later audits possible after the Vast instance is gone.

### Reporting Builder Pattern

`MultiTraitSummaryBuilder` is not a model runner. It is an interpretation-layer
builder:

```text
scalar_decomposition.json + salience_gate.json
  -> role-level joined rows
  -> trait-level aggregate rows
  -> JSON/CSV/Markdown report
```

It teaches a different RE pattern: do not recompute measurements when the needed
artifacts already exist. Build a thin aggregation layer that makes the existing
artifacts inspectable.

### Next Scripts and What They Teach

`TraitJudgeRunner`

- Type: runner.
- Purpose: run judge-model evaluation over generated completions.
- Inputs: generations JSONL, trait rubric config, judge model config.
- Outputs: judgment JSONL, manifest/status/progress.
- RE lesson: external API/model judging, rubric versioning, resumable per-record
  evaluation, and auditability of subjective ratings.

Implementation pattern:

- Keep the rubric config separate from the judge model config. The rubric says
  what the score means; the model config says who is doing the scoring.
- Build one work item per generated completion.
- Preserve `prompt_id`, `trait_axis_id`, `role_id`, `condition`, and
  `scenario_id` in every judgment row.
- Use structured JSON output so downstream behavior metrics do not parse prose.
- Resume by reading existing `judgments.jsonl` plus `judge_progress.json`.
- Start with `--dry-run`, then a small `--limit` pilot, then full judging.

Common failure modes:

- Judge prompt drift: changing rubric wording changes score meaning. Fix with
  `rubric_id` and a versioned config path in every manifest.
- Hidden API cost: judging is per completion and can be expensive. Fix with
  `--limit`, checkpoints, and progress files.
- JSON fragility: prose outputs break aggregation. Fix with structured-output
  schemas and validation before appending JSONL rows.

`BehaviorMetricsBuilder`

- Type: analyzer.
- Purpose: convert judgment rows into baseline and elicitation-shift summaries.
- Inputs: judgment JSONL plus prompt metadata.
- Outputs: behavior summary CSV/JSON.
- RE lesson: separating raw evaluations from aggregate metrics, matching
  positive/negative/neutral conditions, and preserving role-level baselines.

Implementation pattern:

- Validate all judgment rows before aggregating.
- Compute simple condition means by `trait_axis_id`, `role_id`, and
  `condition`.
- Compute shifts against the same role's neutral baseline.
- Also compute matched-pair shifts when `matched_neutral_id` is present in
  metadata. This controls for scenario and role-instruction variant.
- Keep quality metrics separate from trait metrics:
  `role_adherence_score`, `coherence_score`, and `prompt_following_score` are
  not trait scores, but they tell us whether behavior measurements are
  trustworthy.

Reusable lesson:

Raw evaluator rows are not yet experiment results. The analyzer should turn
them into matched, role-aware, condition-aware summaries before we compare them
to activation scalars.

`ScalarBehaviorSummaryBuilder`

- Type: reporting builder.
- Purpose: join activation scalar summaries, salience gates, and behavioral
  judge metrics into one inspection surface.
- Inputs: `scalar_decomposition.json`, `salience_gate.json`, and
  `behavior_metrics.json` for each trait.
- Outputs: joined role CSV, trait aggregate CSV, structured JSON, and Markdown.
- RE lesson: once several pipelines exist, the useful next object is often a
  joiner, not another model runner. The joiner makes cross-pipeline agreement
  inspectable without recomputing expensive artifacts.

Implementation pattern:

- Resolve analysis and behavior run ids separately. They may not be the same
  run, because judging can happen after generation/activation analysis.
- Validate role coverage across all joined artifacts. A missing role should
  fail loudly, otherwise the report can silently compare different populations.
- Keep role-level rows and trait-level rows separate. Role rows are for finding
  persona-specific mismatches; trait rows are for deciding which trait axis is
  worth deeper geometry or steering work.
- Preserve gate decisions next to behavior shifts. A strong behavior shift with
  a failed salience gate means behavior and activation scalar measurement are
  disagreeing, not that one is automatically correct.

Reusable lesson:

Reporting scripts are still research-engineering components. They should have
clear inputs, schemas, validation, and structured outputs, because downstream
interpretation depends on them just as much as it depends on model runners.

`ReportBuilder`

- Type: reporting builder.
- Purpose: combine already-built summaries into one human-readable and
  machine-readable report.
- Inputs: scalar/gate summary JSON, geometry summary JSON, and scalar-behavior
  summary JSON.
- Outputs: integrated Markdown report, integrated JSON report, and manifest.
- RE lesson: final reports should not recompute expensive measurements. They
  should assemble stable summary artifacts, record missing sections explicitly,
  and be rerunnable as new sections become available.

Implementation pattern:

- Keep section summaries separate from the integrated report. This preserves
  modularity: scalar, geometry, and behavior can each be rerun independently.
- Default to tolerant mode so early reports can be generated before behavior
  judging exists.
- Add `--strict` for the final stage, where missing enabled sections should be
  treated as an error.
- Emit both Markdown and JSON. Markdown is for reading; JSON is for later paper
  tables, plots, or automated comparisons.
- Record every consumed input path in the manifest.

Reusable lesson:

An integrated report is a product artifact, not a scratch notebook. It should
make current evidence inspectable while clearly separating completed sections
from pending sections.

`GeometryAnalyzer`

- Type: analyzer.
- Purpose: inspect shared vs role-specific geometry.
- Inputs: role vectors, rulers, scalar summaries, optionally raw activations.
- Outputs: role-role cosine matrices, cross-trait ruler cosines, PCA/residual
  summaries, optional plots.
- RE lesson: tensor-matrix analysis, dimensionality reduction, and distinguishing
  scalar agreement from geometric agreement.

Implemented v0 structure:

- `ArtifactResolver`: resolves `role_trait_vectors.pt` and the matching
  `<trait>_layer8_primary_roles_mean_axis_vector.pt` ruler for each trait.
- `VectorLoader`: loads tensor payloads and extracts the layer-specific role
  vectors.
- `TraitGeometryBuilder`: computes role-role cosine tables within each trait
  for `axis_vector`, `positive_shift`, `negative_shift`, and `offset_vector`.
- `RoleRulerAlignmentBuilder`: computes each role vector's cosine and
  projection against the benchmark ruler.
- `CrossTraitGeometryBuilder`: computes ruler-ruler cosines and same-role
  cross-trait cosines.
- `PCASummaryBuilder`: stacks vectors and uses SVD to estimate explained
  variance and how many components explain 80/90/95 percent of variation.
- `ReportWriter`: writes CSV, JSON, Markdown, and a manifest.

Interpretation pattern:

- High role-ruler alignment plus high role-role cosine means a shared trait
  direction is plausible.
- High scalar projection but low role-role cosine means scalar agreement may be
  hiding role-specific geometry.
- High ruler-ruler cosine across different traits means the chosen axes may not
  be well separated in this model/layer.
- Low PC count for high explained variance means the role/trait geometry is
  compressed into a low-dimensional subspace.

Implementation caveat:

The v0 analyzer compiles and supports dry-run path resolution locally, but full
tensor execution needs a working PyTorch install. The local Mac environment had
a broken torch dynamic-library install, so runtime verification should happen on
Vast or another working torch environment.

`RoleFreeGridBuilder` and `RoleFreeRulerPipeline`

- Type: builder plus reuse of existing runners.
- Purpose: create lower-circularity rulers without a named persona.
- Inputs: trait configs and role-free prompt specs.
- Outputs: role-free prompt grids, generations, activations, vectors, rulers,
  and scalar comparisons.
- RE lesson: reusing the same pipeline with a new role scope rather than writing
  one-off scripts.

`HeldoutTransferRunner`

- Type: analyzer/gate.
- Purpose: test whether primary-role rulers transfer to mediator/strategist.
- Inputs: heldout vectors and primary rulers.
- Outputs: heldout scalar/gate summaries.
- RE lesson: train/test split thinking for representation experiments.

`ProbeComparisonRunner`

- Type: runner/analyzer.
- Purpose: train trait probes and compare probe directions to diff-in-means
  rulers.
- Inputs: raw activations, labels, role splits.
- Outputs: probe metrics, transfer matrix, direction cosines.
- RE lesson: supervised probe workflow, heldout evaluation, and distinguishing
  probe efficacy from representation geometry.

`ConstantSteeringRunner`

- Type: runner.
- Purpose: apply trait directions as interventions and observe behavioral
  effects.
- Inputs: model, ruler, prompts, alpha schedule.
- Outputs: steered generations, status/progress, steering manifest.
- RE lesson: causal intervention runs, dose schedules, quality checks, and
  separating correlation from causality.

`SaturationAnalyzer`

- Type: analyzer.
- Purpose: check whether roles with high offsets have smaller shifts or
  steering effects.
- Inputs: scalar summaries, behavior metrics, steering outputs.
- Outputs: offset-vs-shift tables and plots.
- RE lesson: interpreting ceilings, dose response, and confounds in activation
  steering.

## Initial Learning Commitments

- Every major component gets a short explanation before implementation.
- Every run writes a manifest, status file, and progress file.
- Every important tensor operation documents expected shapes.
- Every analysis report states what is source-verified, runtime-verified, or only planned.
- Negative validation results are preserved as findings, not patched away silently.
