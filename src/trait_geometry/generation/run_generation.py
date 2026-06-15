from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GenerationWorkItem:
    prompt_id: str
    full_prompt: str
    role_id: str
    condition: str
    scenario_id: str
    metadata: dict[str, Any]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def load_prompt_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_no}: {exc}") from exc
            if limit is not None and len(records) >= limit:
                break
    return records


def check_generation_dependencies() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for module_name in ["torch", "transformers", "huggingface_hub"]:
        try:
            module = __import__(module_name)
            results[module_name] = {
                "installed": True,
                "version": getattr(module, "__version__", "installed"),
            }
        except Exception as exc:
            results[module_name] = {
                "installed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    results["ready"] = all(item["installed"] for item in results.values())
    return results


def progress_iter(items: list[Any], description: str, unit: str = "prompt"):
    try:
        from tqdm.auto import tqdm

        return tqdm(items, desc=description, unit=unit)
    except Exception:
        return items


def batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def build_work_items(prompt_records: list[dict[str, Any]]) -> list[GenerationWorkItem]:
    items: list[GenerationWorkItem] = []
    for record in prompt_records:
        items.append(
            GenerationWorkItem(
                prompt_id=record["prompt_id"],
                full_prompt=record["full_prompt"],
                role_id=record["role_id"],
                condition=record["condition"],
                scenario_id=record["scenario_id"],
                metadata={
                    "trait_axis_id": record["trait_axis_id"],
                    "role_instruction_variant_id": record["role_instruction_variant_id"],
                    "matched_neutral_id": record["matched_neutral_id"],
                    "matched_positive_id": record["matched_positive_id"],
                    "matched_negative_id": record["matched_negative_id"],
                    "trait_word_present": record["trait_word_present"],
                    "readout_policy": record["readout_policy"],
                },
            )
        )
    return items


def make_run_dirs(run_root: Path) -> dict[str, Path]:
    paths = {
        "inputs": run_root / "inputs",
        "checkpoints": run_root / "checkpoints",
        "results": run_root / "results",
        "logs": run_root / "logs",
        "meta": run_root / "meta",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_log(path: Path, message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{timestamp} {message}\n")


def read_progress(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def completed_prompt_ids_from_results(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in existing generations file line {line_no}: {exc}") from exc
            prompt_id = record.get("prompt_id")
            if prompt_id:
                completed.add(str(prompt_id))
    return completed


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def model_config_with_generation_overrides(model_config: dict[str, Any], batch_size: int) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    effective_config = dict(model_config)
    effective_config["generation"] = dict(model_config["generation"])
    effective_config["generation"]["batch_size"] = batch_size
    return effective_config


def write_dry_run_artifacts(
    run_root: Path,
    prompt_jsonl: Path,
    model_config_path: Path,
    model_config: dict[str, Any],
    work_items: list[GenerationWorkItem],
    dependency_status: dict[str, Any],
) -> None:
    paths = make_run_dirs(run_root)
    now = datetime.now(timezone.utc).isoformat()

    manifest = {
        "schema_version": "0.1",
        "runner": "GenerationRunner",
        "mode": "dry_run",
        "created_at_utc": now,
        "prompt_jsonl": str(prompt_jsonl),
        "model_config": str(model_config_path),
        "model_id": model_config["model_id"],
        "huggingface_model_name": model_config["huggingface_model_name"],
        "generation_config": model_config["generation"],
        "planned_records": len(work_items),
        "run_root": str(run_root),
    }
    status = {
        "status": "dry_run_complete",
        "updated_at_utc": now,
        "completed_records": 0,
        "planned_records": len(work_items),
        "dependencies": dependency_status,
    }
    progress = {
        "cursor": 0,
        "completed_prompt_ids": [],
        "planned_prompt_ids": [item.prompt_id for item in work_items],
    }
    preview = {
        "planned_records": len(work_items),
        "first_items": [asdict(item) for item in work_items[:5]],
    }

    write_json(paths["meta"] / "run_manifest.json", manifest)
    write_json(paths["meta"] / "status.json", status)
    write_json(paths["checkpoints"] / "progress.json", progress)
    write_json(paths["inputs"] / "generation_preview.json", preview)
    append_log(paths["logs"] / "run.log", "dry run completed; no model generation executed")


def write_initial_run_artifacts(
    run_root: Path,
    prompt_jsonl: Path,
    model_config_path: Path,
    model_config: dict[str, Any],
    work_items: list[GenerationWorkItem],
    dependency_status: dict[str, Any],
) -> dict[str, Path]:
    paths = make_run_dirs(run_root)
    now = datetime.now(timezone.utc).isoformat()
    manifest_path = paths["meta"] / "run_manifest.json"
    status_path = paths["meta"] / "status.json"
    progress_path = paths["checkpoints"] / "progress.json"

    if not manifest_path.exists():
        write_json(
            manifest_path,
            {
                "schema_version": "0.1",
                "runner": "GenerationRunner",
                "mode": "execute",
                "created_at_utc": now,
                "prompt_jsonl": str(prompt_jsonl),
                "model_config": str(model_config_path),
                "model_id": model_config["model_id"],
                "huggingface_model_name": model_config["huggingface_model_name"],
                "generation_config": model_config["generation"],
                "planned_records": len(work_items),
                "run_root": str(run_root),
                "output_generations": str(paths["results"] / "generations.jsonl"),
            },
        )

    write_json(
        status_path,
        {
            "status": "running",
            "updated_at_utc": now,
            "completed_records": 0,
            "planned_records": len(work_items),
            "dependencies": dependency_status,
        },
    )
    if not progress_path.exists():
        write_json(
            progress_path,
            {
                "cursor": 0,
                "completed_prompt_ids": [],
                "planned_prompt_ids": [item.prompt_id for item in work_items],
            },
        )
    append_log(paths["logs"] / "run.log", "generation run initialized")
    return paths


def load_transformers_model(model_config: dict[str, Any]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = model_config["huggingface_model_name"]
    token = os.environ.get("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device_map = "auto" if torch.cuda.is_available() else None
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    if device_map is None:
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return tokenizer, model


def generated_text_from_output(tokenizer: Any, output_ids: Any, prompt_length: int) -> str:
    generated_ids = output_ids[prompt_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def generate_one(
    tokenizer: Any,
    model: Any,
    item: GenerationWorkItem,
    generation_config: dict[str, Any],
) -> dict[str, Any]:
    import torch

    encoded = tokenizer(item.full_prompt, return_tensors="pt")
    model_device = next(model.parameters()).device
    encoded = {key: value.to(model_device) for key, value in encoded.items()}
    prompt_length = int(encoded["input_ids"].shape[-1])

    do_sample = bool(generation_config.get("do_sample", False))
    temperature = float(generation_config.get("temperature", 0.0))
    generate_kwargs = {
        "max_new_tokens": int(generation_config.get("max_new_tokens", 192)),
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature

    with torch.no_grad():
        output = model.generate(**encoded, **generate_kwargs)
    completion = generated_text_from_output(tokenizer, output[0], prompt_length)
    return {
        "prompt_id": item.prompt_id,
        "role_id": item.role_id,
        "condition": item.condition,
        "scenario_id": item.scenario_id,
        "metadata": item.metadata,
        "full_prompt": item.full_prompt,
        "completion": completion,
        "generation_config": generation_config,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def generate_batch(
    tokenizer: Any,
    model: Any,
    items: list[GenerationWorkItem],
    generation_config: dict[str, Any],
) -> list[dict[str, Any]]:
    import torch

    if not items:
        return []

    encoded = tokenizer([item.full_prompt for item in items], return_tensors="pt", padding=True)
    model_device = next(model.parameters()).device
    encoded = {key: value.to(model_device) for key, value in encoded.items()}
    input_width = int(encoded["input_ids"].shape[-1])

    do_sample = bool(generation_config.get("do_sample", False))
    temperature = float(generation_config.get("temperature", 0.0))
    generate_kwargs = {
        "max_new_tokens": int(generation_config.get("max_new_tokens", 192)),
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(**encoded, **generate_kwargs)

    generated_at = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []
    if len(outputs) != len(items):
        raise RuntimeError(f"model returned {len(outputs)} outputs for {len(items)} prompts")
    for item, output_ids in zip(items, outputs):
        completion = generated_text_from_output(tokenizer, output_ids, input_width)
        records.append(
            {
                "prompt_id": item.prompt_id,
                "role_id": item.role_id,
                "condition": item.condition,
                "scenario_id": item.scenario_id,
                "metadata": item.metadata,
                "full_prompt": item.full_prompt,
                "completion": completion,
                "generation_config": generation_config,
                "generated_at_utc": generated_at,
            }
        )
    return records


def run_generation(
    run_root: Path,
    prompt_jsonl: Path,
    model_config_path: Path,
    model_config: dict[str, Any],
    work_items: list[GenerationWorkItem],
    dependency_status: dict[str, Any],
    save_every: int,
    batch_size: int,
) -> dict[str, Any]:
    model_config = model_config_with_generation_overrides(model_config, batch_size)
    paths = write_initial_run_artifacts(
        run_root=run_root,
        prompt_jsonl=prompt_jsonl,
        model_config_path=model_config_path,
        model_config=model_config,
        work_items=work_items,
        dependency_status=dependency_status,
    )
    results_path = paths["results"] / "generations.jsonl"
    progress_path = paths["checkpoints"] / "progress.json"
    status_path = paths["meta"] / "status.json"
    log_path = paths["logs"] / "run.log"

    completed = completed_prompt_ids_from_results(results_path)
    progress = read_progress(progress_path) or {}
    completed.update(str(prompt_id) for prompt_id in progress.get("completed_prompt_ids", []))
    remaining = [item for item in work_items if item.prompt_id not in completed]

    generation_config = model_config["generation"]

    append_log(log_path, f"loading model {model_config['huggingface_model_name']}")
    tokenizer, model = load_transformers_model(model_config)
    append_log(log_path, f"model loaded; remaining records={len(remaining)} batch_size={batch_size}")

    generated_count = 0
    last_checkpoint_count = 0
    for batch in progress_iter(batched(remaining, batch_size), "generating", unit="batch"):
        try:
            records = generate_batch(tokenizer, model, batch, generation_config)
            append_jsonl(results_path, records)
            completed.update(item.prompt_id for item in batch)
            generated_count += len(batch)
        except Exception as exc:
            failed_ids = [item.prompt_id for item in batch]
            write_json(
                status_path,
                {
                    "status": "failed",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "completed_records": len(completed),
                    "planned_records": len(work_items),
                    "error": f"{type(exc).__name__}: {exc}",
                    "failed_prompt_ids": failed_ids,
                },
            )
            append_log(log_path, f"failed on prompt_ids={failed_ids}: {type(exc).__name__}: {exc}")
            raise

        if generated_count - last_checkpoint_count >= save_every:
            write_json(
                progress_path,
                {
                    "cursor": len(completed),
                    "completed_prompt_ids": sorted(completed),
                    "planned_prompt_ids": [work_item.prompt_id for work_item in work_items],
                },
            )
            write_json(
                status_path,
                {
                    "status": "running",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "completed_records": len(completed),
                    "planned_records": len(work_items),
                },
            )
            append_log(log_path, f"checkpoint saved completed={len(completed)}")
            last_checkpoint_count = generated_count

    write_json(
        progress_path,
        {
            "cursor": len(completed),
            "completed_prompt_ids": sorted(completed),
            "planned_prompt_ids": [work_item.prompt_id for work_item in work_items],
        },
    )
    write_json(
        status_path,
        {
            "status": "completed",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_records": len(completed),
            "planned_records": len(work_items),
            "output_generations": str(results_path),
        },
    )
    append_log(log_path, f"generation completed completed={len(completed)}")
    return {
        "status": "completed",
        "planned_records": len(work_items),
        "completed_records": len(completed),
        "generated_this_run": generated_count,
        "batch_size": batch_size,
        "output_generations": str(results_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare or run model generations for a prompt grid.")
    parser.add_argument("--prompt-jsonl", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Generation batch size. Defaults to generation.batch_size in the model config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write manifests/previews and dependency status without loading a model.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    model_config = load_yaml(args.model_config)
    prompt_records = load_prompt_jsonl(args.prompt_jsonl, limit=args.limit)
    work_items = build_work_items(prompt_records)
    dependency_status = check_generation_dependencies()
    batch_size = int(args.batch_size or model_config.get("generation", {}).get("batch_size", 1))
    model_config = model_config_with_generation_overrides(model_config, batch_size)

    if args.dry_run:
        write_dry_run_artifacts(
            run_root=args.run_root,
            prompt_jsonl=args.prompt_jsonl,
            model_config_path=args.model_config,
            model_config=model_config,
            work_items=work_items,
            dependency_status=dependency_status,
        )
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "run_root": str(args.run_root),
                    "planned_records": len(work_items),
                    "dependencies_ready": dependency_status["ready"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if not dependency_status["ready"]:
        print(
            json.dumps(
                {
                    "error": "generation dependencies are missing",
                    "dependencies": dependency_status,
                    "next_step": "Install torch, transformers, and huggingface_hub in the execution environment, then rerun or use --dry-run.",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    result = run_generation(
        run_root=args.run_root,
        prompt_jsonl=args.prompt_jsonl,
        model_config_path=args.model_config,
        model_config=model_config,
        work_items=work_items,
        dependency_status=dependency_status,
        save_every=args.save_every,
        batch_size=batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
