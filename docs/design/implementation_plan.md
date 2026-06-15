# Trait Geometry Across Personas: Implementation Plan

This plan turns the experiment design into a staged research-engineering build. The first implementation target is a small vertical slice, not the full experiment matrix.

## Project Goal

Measure whether behavioral traits in LLM activations are represented as:

- a shared direction across personas,
- fragmented role-specific directions,
- or one shared direction with persona-specific offsets and shifts.

The first pilot uses five bounded trait axes rather than broad personality domains:

- warmth vs coldness,
- sincerity vs manipulativeness,
- caution vs recklessness,
- curiosity vs closed-mindedness,
- skepticism vs gullibility.

These axes cover prosocial social affect, alignment-conflicting social intent, cognitive risk posture, epistemic exploration, and epistemic trust. The first smoke-run should execute one axis end-to-end before expanding to all five.

## Core Objects

`Trait`
: Behavioral axis under study, such as warmth/coldness or caution/recklessness.

`Persona`
: Role conditioning used in prompts. Pilot roles are split into primary extraction roles, held-out transfer roles, and later stress-test roles.

`Prompt`
: Structured input with metadata for trait, persona, condition, scenario id, and matched neutral id.

`ActivationRecord`
: Cached model activation at a specified layer and token/span readout.

`BenchmarkRuler`
: Unit-normalized trait direction `b_T` used for scalar projections.

`RoleTraitVector`
: Per-role contrast vector `v_T,P = E[h_trait,P] - E[h_neutral,P]`.

`ScalarDecomposition`
: Offset, shift, and other-response displacement for each trait/persona.

`BehaviorRating`
: Structured LLM-judge rating for generated behavior.

`RunManifest`
: Immutable metadata for a run: config, model, dataset, layers, artifact paths, git commit if available, and timestamps.

`RunStatus`
: Mutable lifecycle state: running, completed, failed, paused, and last completed unit.

## Canonical Repo Layout

```text
configs/
  experiments/
  models/
  traits/
  personas/

data/
  prompts/
  raw/
  processed/

artifacts/
  runs/
    <experiment_name>/
      <model_name>/
        <dataset_name>/
          <probe_set>/
            <variant>/
              <run_id>/
                inputs/
                checkpoints/
                results/
                logs/
                meta/

src/
  trait_geometry/
    prompts/
    generation/
    activations/
    vectors/
    rulers/
    validation/
    behavior/
    geometry/
    steering/
    reporting/

docs/
  design/
  learning/
  sources/
```

All experiment outputs go under `artifacts/runs/...`. Source data and prompt specs live under `data/` or `configs/`. No ad-hoc result paths.

## Main Pipeline

```text
configs
  -> PromptGridBuilder
  -> GenerationRunner
  -> ActivationCacheBuilder
  -> VectorBuilder
  -> RulerBuilder
  -> ValidationGateRunner
  -> ScalarDecompositionBuilder
  -> BehaviorEvaluationRunner
  -> GeometryAnalyzer
  -> SteeringRunner
  -> ReportBuilder
```

Each stage must have:

- explicit config input,
- structured output artifacts,
- manifest/status updates,
- resume or skip behavior,
- validation checks.

## Stage 0: Design Lock

Purpose: convert the experiment design into buildable specs.

Deliverables:

- `configs/traits/warmth_coldness.yaml`
- `configs/traits/sincerity_manipulativeness.yaml`
- `configs/traits/caution_recklessness.yaml`
- `configs/traits/curiosity_closed_mindedness.yaml`
- `configs/traits/skepticism_gullibility.yaml`
- `configs/personas/core_roles.yaml`
- prompt condition schema
- run manifest schema
- artifact path policy

Decisions to make:

- exact first model target,
- first activation layer or layer sweep,
- first judge model/rubric,
- whether the first vertical slice uses only scenario-induced prompts or includes instruction prompts immediately,
- which single axis is used for the first smoke-run before expanding to all five.

Default smoke-run axis: warmth vs coldness. It is expected to be the cleanest shared-direction candidate and is less likely to be blocked by safety-policy behavior than manipulativeness or recklessness.

## Stage 1: Minimal Prompt Grid

Build the smallest prompt set that can support the salience gate and scalar decomposition.

Initial scope:

- five pilot axes configured, one axis executed first as a smoke-run,
- four primary roles for initial extraction: counselor, tutor, debugger, journalist,
- two held-out roles for transfer checks: mediator, strategist,
- at least six scenarios per persona,
- conditions: `present_trait`, `present_neutral`, `mention_without_possession`,
- optional early additions: `instruction_based`, `other_trait`, `other_neutral`.

Component:

```text
PromptGridBuilder
```

Artifacts:

```text
data/prompts/pilot_traits_core_v001.jsonl
data/prompts/pilot_traits_core_v001_manifest.json
```

Checks:

- every trait prompt has a matched neutral prompt,
- scenario ids are balanced across personas,
- scenario-induced prompts do not contain leaked trait words unless explicitly allowed,
- condition labels are schema-valid.

## Stage 2: Minimal Generation and Activation Cache

Run completions and cache activations for the minimal prompt grid.

Components:

```text
GenerationRunner
ActivationCacheBuilder
```

Required run files:

```text
meta/run_manifest.json
meta/status.json
checkpoints/progress.json
results/generations.jsonl
results/activations.<format>
logs/run.log
```

Resume behavior:

- skip completed prompt ids,
- write checkpoint after fixed record intervals,
- never overwrite completed runs unless explicitly forced.

## Stage 3: Ruler Construction and Validation Funnel

Build candidate benchmark rulers and validate them before any full-scale experiment.

Candidate rulers:

- role-free ruler,
- pooled leave-one-role-out ruler,
- external ruler if a compatible source exists.

Components:

```text
VectorBuilder
RulerBuilder
ValidationGateRunner
ValidationReportBuilder
```

Validation gates:

1. salience gate,
2. ruler convergence,
3. early sanity steering,
4. register-artifact control.

Proceed only if the chosen ruler distinguishes trait expression from trait mention. If mention-without-possession prompts project high, pivot to the finding that the ruler/probe detects topic salience rather than trait possession.

## Stage 4: First Scalar and Behavior Report

Compute offset and shift for the first trait/persona matrix and attach behavior ratings.

Components:

```text
ScalarDecompositionBuilder
BehaviorGenerationRunner
TraitJudgeRunner
BehaviorMetricsBuilder
```

Core metrics:

- standing offset,
- scenario-induced shift,
- instruction-induced shift if included,
- behavioral baseline,
- behavioral shift.

Report:

```text
artifacts/runs/.../results/scalar_decomposition.json
artifacts/runs/.../results/behavior_summary.json
artifacts/runs/.../reports/first_scalar_behavior_report.md
```

## Stage 5: Geometry Diagnostics

Run Layer 2 checks after the scalar pipeline works.

Components:

```text
GeometryAnalyzer
ProbeComparisonRunner
```

Analyses:

- pairwise role-role cosine matrix,
- PCA on stacked role vectors,
- PCA on individual centered examples,
- residual PCA after removing `b_T`,
- logistic probe vs diff-in-means direction.

This stage decides whether scalar results should be interpreted as a shared direction or one coordinate of a higher-dimensional role-specific object.

Held-out role checks should test whether directions built from the primary extraction roles transfer to mediator and strategist without rebuilding the ruler on those held-out roles.

## Stage 6: Present/Other Readout

Add transcript prompts and position-resolved activation extraction.

Components:

```text
SpeakerSpanAnnotator
PositionReadoutExtractor
PresentOtherAnalyzer
```

Requirements:

- prompt metadata must preserve speaker spans,
- readout position must be explicit,
- matched `other_trait` and `other_neutral` conditions are mandatory.

## Stage 7: Saturation and Steering

Only run after scalar decomposition and behavioral evaluation are stable.

Components:

```text
SaturationAnalyzer
ConstantSteeringRunner
DoseResponseBuilder
```

Analyses:

- offset vs shift correlation,
- constant alpha steering across roles,
- dose-response curves,
- quality-constrained steering range.

Judge dimensions:

- target trait rating,
- persona integrity,
- coherence,
- task usefulness,
- off-target trait movement.

## Stage 8: Base vs Instruct

Repeat the validated pipeline on base and instruct variants.

Questions:

- does the ruler exist in base?
- does instruction tuning sharpen the direction?
- does instruction tuning move persona offsets?
- does alignment compress or homogenize persona baselines?

## Stage 9: Safety Translation

Convert results into deployable interpretations.

Outputs:

- persona-specific probe threshold correction table,
- role-specific steering dose calibration curve,
- character-training QA checklist,
- probe false-alarm analysis for trait mention vs trait possession.

## Role Sets

The first pilot uses roles selected from the Assistant Axis role inventory.

Primary extraction roles:

```text
counselor
tutor
debugger
journalist
```

Held-out transfer roles:

```text
mediator
strategist
```

Later robustness and stress-test roles:

```text
critic
doctor
lawyer
spy
caregiver
skeptic
```

Use primary roles to build the first trait directions and scalar decomposition. Use held-out roles to test transfer/generalization. Use stress-test roles only after the core pipeline is stable, because doctor/lawyer introduce domain-safety caution, spy introduces alignment-conflicting persona behavior, and critic/skeptic can amplify register or stance confounds.

## First Vertical Slice

The first implementation milestone should be:

```text
warmth/coldness smoke-run first
then all five pilot axes:
  warmth/coldness
  sincerity/manipulativeness
  caution/recklessness
  curiosity/closed-mindedness
  skepticism/gullibility
primary roles:
  counselor
  tutor
  debugger
  journalist
held-out roles for later transfer:
  mediator
  strategist
six scenarios each
scenario-induced present_trait/present_neutral
mention_without_possession
one model
one or small number of layers
activation cache
one benchmark ruler
salience gate
offset/shift table
basic behavior judge
first report
```

This is the smallest useful end-to-end path. Expand only after it runs and produces auditable artifacts.
