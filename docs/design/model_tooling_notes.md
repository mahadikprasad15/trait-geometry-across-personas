# Model and Tooling Notes

## Current Local Environment

Checked on 2026-06-14:

```text
Python: 3.9.6
torch: missing
transformers: missing
transformer_lens: missing
PyYAML: installed
```

This means the repo can currently build configs and prompt grids, but cannot yet run model generation or activation extraction locally.

## First Smoke-Run Model

Default:

```text
meta-llama/Llama-3.2-1B-Instruct
```

Reason:

- It matches the project target family.
- It is smaller than 3B/8B models, so it is better for pipeline validation.
- It lets us test prompt rendering, generation artifacts, dependency handling, and activation-cache interfaces before scaling.

## Tooling Plan

Generation:

```text
Hugging Face transformers
```

Activation extraction:

```text
Prefer TransformerLens if installed version supports the model.
Fallback to Hugging Face hooks if needed.
```

Current source check:

- Current TransformerLens source includes explicit config branches for Llama 3.2 1B and 3B text models in `loading_from_pretrained.py`.
- This makes TransformerLens plausible for the activation-cache path, but we still need to verify with the installed package version once dependencies are available.

## Dependency Policy

Do not hardcode credentials or model paths.

Expected later setup:

```text
torch
transformers
transformer_lens
accelerate
huggingface_hub
```

Meta Llama models may require:

```text
HF_TOKEN
accepted Hugging Face model license/access
```

## Runner Implication

`GenerationRunner` should:

- read `configs/models/llama_3_2_1b_instruct.yaml`,
- check dependencies before running,
- fail with a clear setup message if packages are missing,
- write run manifest/status/progress when execution begins,
- support a small `--limit` for the first smoke test.

`ActivationCacheBuilder` should:

- stay blocked until model dependencies are installed,
- support TransformerLens and HF-hook backends,
- record backend, layer list, readout policy, and tensor shapes in its manifest.

