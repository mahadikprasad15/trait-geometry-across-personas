# Source Research Plan

This document lists the source research needed before locking prompts, datasets, models, and comparison claims.

## Research Questions

1. Which datasets or papers can support warmth/coldness contrast construction?
2. Which datasets or papers can support sincerity/manipulativeness contrast construction?
3. Which datasets or papers can support caution/recklessness, curiosity/closed-mindedness, and skepticism/gullibility?
4. Which persona prompt sets are reusable or adaptable?
5. Which papers should guide trait-vector, persona-vector, emotion-vector, and steering methods?
6. What is the current practical support for Llama 3.2 with TransformerLens or equivalent activation extraction?

## Dataset Categories

### Warmth and Coldness

Look for:

- agreeableness trait prompts,
- empathy or warmth evaluation prompts,
- toxicity/harshness benchmarks,
- supportive vs detached response pairs,
- persona-conditioned social tone prompts.

Selection criteria:

- clear positive/negative labels,
- compatible with generation or classification,
- can be adapted into scenario-induced prompts,
- avoids trivial lexical leakage where possible,
- license allows local use.

### Sincerity and Manipulativeness

Look for:

- honesty-humility trait prompts,
- sycophancy/flattery datasets,
- deception or manipulation prompt sets,
- villain-roleplay trait lexicons,
- self-serving vs transparent response scenarios.

Important distinction:

```text
sincerity != politeness
manipulativeness != overt aggression
```

The prompt/judge design must separate genuine transparency from agreeable tone, and subtle manipulation from shallow hostility.

### Caution and Recklessness

Look for:

- risk-aware decision prompts,
- safety-critical advice settings,
- medical/legal/finance/debugging caution scenarios,
- conscientiousness or deliberation trait prompts.

Purpose:

Caution is a cognitive-action trait. It should be expressible across roles without being purely moral or purely stylistic.

### Curiosity and Closed-Mindedness

Look for:

- openness trait prompts,
- information-seeking behavior prompts,
- exploration vs premature-conclusion scenarios,
- question-asking and alternative-hypothesis tasks.

Purpose:

Curiosity is an epistemic/exploratory trait that should transfer across teaching, coding, therapy, and management contexts.

### Skepticism and Gullibility

Look for:

- epistemic trust and belief-update prompts,
- misinformation or unreliable-user scenarios,
- critical-thinking benchmarks,
- over-trust, gullibility, and verification-seeking settings.

Purpose:

Skepticism is especially useful for present/other experiments because it can be elicited when another speaker is unreliable, deceptive, or overconfident.

### Persona Sources

Look for:

- existing persona prompt libraries,
- role descriptions from persona-vector papers,
- role-play benchmark prompts,
- assistant persona/evaluation datasets.

Primary extraction roles:

- counselor,
- tutor,
- debugger,
- journalist.

Held-out transfer roles:

- mediator,
- strategist.

Later robustness and stress-test roles:

- critic,
- doctor,
- lawyer,
- spy,
- caregiver,
- skeptic.

Criteria:

- roles should be distinct in style and domain,
- roles should plausibly vary in trait offsets,
- roles should support matched neutral and trait scenarios.

## Paper Categories

### Emotion Vectors

Use for:

- universalist prior,
- present/other distinction,
- concept-vs-state caution,
- scenario/story-based elicitation.

Questions to extract:

- how vectors were built,
- how causal steering was validated,
- how present/other representations were distinguished,
- what readout positions were used.

### Persona Vectors

Use for:

- persona direction construction,
- external ruler candidates,
- compositionality regression context,
- relation to character-training or assistant-axis work.

Questions to extract:

- how persona vectors were extracted,
- which traits/personas overlap with this project,
- whether directions are model/layer specific,
- how behavior was evaluated.

### Activation Steering and Task Vectors

Use for:

- alpha schedules,
- causal liveness checks,
- dose-response curves,
- off-target and quality degradation metrics.

Questions to extract:

- what intervention site is used,
- how steering strength is chosen,
- how incoherence/persona drift is measured,
- whether steering transfers across contexts.

### Probe Transfer

Use for:

- role-held-out probe evaluation,
- linearly readable vs causally steerable distinction,
- probe threshold calibration.

Questions to extract:

- how splits are constructed,
- whether probes detect topic mention or possession,
- how thresholds are calibrated across domains.

## Prompt Research

For each trait, collect candidate scenario templates:

```text
present_trait
present_neutral
other_trait
other_neutral
mention_without_possession
instruction_based
```

For every template, record:

- trait,
- role,
- scenario family,
- condition,
- whether trait word appears,
- matched neutral id,
- source or author,
- expected behavior.

## Model and Tooling Research

Questions:

- Is Llama 3.2 supported directly by TransformerLens?
- If not, what is the best equivalent activation-extraction path?
- Which layer names correspond to residual stream sites?
- What GPU/CPU memory constraints apply locally?
- Can we run small vertical-slice tests locally, or do we need remote execution?

Fallback paths:

- TransformerLens if supported,
- Hugging Face hooks if TransformerLens support is incomplete,
- smaller model for smoke tests,
- cached fake activations for unit tests.

## Research Memo Template

Use one memo per source:

```text
## <Source Name>

Link:

Type:
paper | dataset | prompt set | codebase | documentation

Summary:

What we can reuse:

Risks or mismatch:

License/access:

How it maps to this repo:

Decision:
use | maybe | reject
```

## Initial Research Order

1. Confirm model/tooling path for activation extraction.
2. Find sources for the five pilot axes: warmth/coldness, sincerity/manipulativeness, caution/recklessness, curiosity/closed-mindedness, skepticism/gullibility.
3. Find persona-vector and emotion-vector papers.
4. Find persona prompt sources.
5. Decide what to author ourselves versus reuse.

Do not lock the full prompt grid until the source memo identifies which prompts are reused, adapted, or newly authored.
