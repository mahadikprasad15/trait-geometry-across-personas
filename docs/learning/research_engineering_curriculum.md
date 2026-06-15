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

## Initial Learning Commitments

- Every major component gets a short explanation before implementation.
- Every run writes a manifest, status file, and progress file.
- Every important tensor operation documents expected shapes.
- Every analysis report states what is source-verified, runtime-verified, or only planned.
- Negative validation results are preserved as findings, not patched away silently.

