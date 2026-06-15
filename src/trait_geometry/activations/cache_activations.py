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
class ActivationWorkItem:
    prompt_id: str
    full_prompt: str
    completion: str
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


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_no} in {path}: {exc}") from exc
            if limit is not None and len(records) >= limit:
                break
    return records


def check_activation_dependencies(backend: str) -> dict[str, Any]:
    modules = ["torch"]
    if backend == "transformer_lens":
        modules.append("transformer_lens")
    elif backend == "transformers_hooks":
        modules.append("transformers")
    else:
        raise ValueError(f"unsupported backend {backend!r}")

    results: dict[str, Any] = {}
    for module_name in modules:
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


def make_run_dirs(run_root: Path) -> dict[str, Path]:
    paths = {
        "inputs": run_root / "inputs",
        "checkpoints": run_root / "checkpoints",
        "results": run_root / "results",
        "activations": run_root / "results" / "activations",
        "logs": run_root / "logs",
        "meta": run_root / "meta",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def append_log(path: Path, message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{timestamp} {message}\n")


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def completed_prompt_ids_from_index(path: Path) -> set[str]:
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
                raise ValueError(f"invalid JSON in activation index line {line_no}: {exc}") from exc
            prompt_id = record.get("prompt_id")
            artifact_path = record.get("activation_path")
            if prompt_id and artifact_path and Path(artifact_path).exists():
                completed.add(str(prompt_id))
    return completed


def build_work_items(generation_records: list[dict[str, Any]]) -> list[ActivationWorkItem]:
    items: list[ActivationWorkItem] = []
    for record in generation_records:
        completion = record.get("completion")
        if completion is None:
            raise ValueError(f"generation record {record.get('prompt_id')} lacks completion")
        items.append(
            ActivationWorkItem(
                prompt_id=record["prompt_id"],
                full_prompt=record["full_prompt"],
                completion=str(completion),
                role_id=record["role_id"],
                condition=record["condition"],
                scenario_id=record["scenario_id"],
                metadata=record.get("metadata", {}),
            )
        )
    return items


def artifact_name(prompt_id: str) -> str:
    safe = prompt_id.replace("/", "_").replace(":", "_")
    return f"{safe}.pt"


def write_dry_run_artifacts(
    run_root: Path,
    generations_jsonl: Path,
    model_config_path: Path,
    model_config: dict[str, Any],
    work_items: list[ActivationWorkItem],
    layers: list[int],
    backend: str,
    dependency_status: dict[str, Any],
) -> None:
    paths = make_run_dirs(run_root)
    now = datetime.now(timezone.utc).isoformat()
    write_json(
        paths["meta"] / "activation_manifest.json",
        {
            "schema_version": "0.1",
            "runner": "ActivationCacheBuilder",
            "mode": "dry_run",
            "created_at_utc": now,
            "generations_jsonl": str(generations_jsonl),
            "model_config": str(model_config_path),
            "model_id": model_config["model_id"],
            "huggingface_model_name": model_config["huggingface_model_name"],
            "backend": backend,
            "layers": layers,
            "readout_policy": model_config["activation_extraction"]["first_layer_policy"]["readout_policy"],
            "planned_records": len(work_items),
            "run_root": str(run_root),
        },
    )
    write_json(
        paths["meta"] / "activation_status.json",
        {
            "status": "dry_run_complete",
            "updated_at_utc": now,
            "completed_records": 0,
            "planned_records": len(work_items),
            "dependencies": dependency_status,
        },
    )
    write_json(
        paths["checkpoints"] / "activation_progress.json",
        {
            "cursor": 0,
            "completed_prompt_ids": [],
            "planned_prompt_ids": [item.prompt_id for item in work_items],
        },
    )
    write_json(
        paths["inputs"] / "activation_preview.json",
        {
            "planned_records": len(work_items),
            "layers": layers,
            "backend": backend,
            "first_items": [asdict(item) for item in work_items[:5]],
        },
    )
    append_log(paths["logs"] / "activation_cache.log", "dry run completed; no activations cached")


def load_transformer_lens_model(model_config: dict[str, Any]):
    import torch
    from transformer_lens import HookedTransformer

    model_name = model_config["huggingface_model_name"]
    token = os.environ.get("HF_TOKEN") or None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_attempts = []
    if token:
        load_attempts.extend([{"token": token}, {"use_auth_token": token}])
    load_attempts.append({})

    last_type_error: TypeError | None = None
    for kwargs in load_attempts:
        try:
            model = HookedTransformer.from_pretrained(model_name, device=device, **kwargs)
            model.eval()
            return model
        except TypeError as exc:
            message = str(exc)
            if "unexpected keyword argument" not in message:
                raise
            last_type_error = exc
            continue
    if last_type_error is not None:
        raise last_type_error
    raise RuntimeError(f"failed to load TransformerLens model {model_name}")


def cache_one_transformer_lens(
    model: Any,
    item: ActivationWorkItem,
    layers: list[int],
    activation_path: Path,
) -> dict[str, Any]:
    import torch

    full_text = item.full_prompt + item.completion
    prompt_tokens = model.to_tokens(item.full_prompt, prepend_bos=True)
    full_tokens = model.to_tokens(full_text, prepend_bos=True)
    prompt_len = int(prompt_tokens.shape[-1])
    full_len = int(full_tokens.shape[-1])
    if full_len <= prompt_len:
        raise ValueError(f"no response tokens detected for prompt_id={item.prompt_id}")

    names = [f"blocks.{layer}.hook_resid_post" for layer in layers]
    with torch.no_grad():
        _, cache = model.run_with_cache(
            full_tokens,
            names_filter=lambda name: name in set(names),
            remove_batch_dim=False,
        )

    pooled: dict[int, Any] = {}
    for layer, name in zip(layers, names):
        resid = cache[name][0, prompt_len:full_len, :].detach().cpu()
        pooled[layer] = resid.mean(dim=0)

    payload = {
        "prompt_id": item.prompt_id,
        "role_id": item.role_id,
        "condition": item.condition,
        "scenario_id": item.scenario_id,
        "layers": layers,
        "readout_policy": "response_token_mean",
        "prompt_token_count": prompt_len,
        "full_token_count": full_len,
        "response_token_count": full_len - prompt_len,
        "metadata": item.metadata,
        "pooled_resid_post": pooled,
    }
    activation_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, activation_path)
    return {
        "prompt_id": item.prompt_id,
        "activation_path": str(activation_path),
        "layers": layers,
        "readout_policy": "response_token_mean",
        "prompt_token_count": prompt_len,
        "full_token_count": full_len,
        "response_token_count": full_len - prompt_len,
        "role_id": item.role_id,
        "condition": item.condition,
        "scenario_id": item.scenario_id,
    }


def write_initial_artifacts(
    run_root: Path,
    generations_jsonl: Path,
    model_config_path: Path,
    model_config: dict[str, Any],
    work_items: list[ActivationWorkItem],
    layers: list[int],
    backend: str,
    dependency_status: dict[str, Any],
) -> dict[str, Path]:
    paths = make_run_dirs(run_root)
    now = datetime.now(timezone.utc).isoformat()
    manifest_path = paths["meta"] / "activation_manifest.json"
    if not manifest_path.exists():
        write_json(
            manifest_path,
            {
                "schema_version": "0.1",
                "runner": "ActivationCacheBuilder",
                "mode": "execute",
                "created_at_utc": now,
                "generations_jsonl": str(generations_jsonl),
                "model_config": str(model_config_path),
                "model_id": model_config["model_id"],
                "huggingface_model_name": model_config["huggingface_model_name"],
                "backend": backend,
                "layers": layers,
                "readout_policy": model_config["activation_extraction"]["first_layer_policy"]["readout_policy"],
                "planned_records": len(work_items),
                "run_root": str(run_root),
                "activation_index": str(paths["results"] / "activation_index.jsonl"),
            },
        )
    write_json(
        paths["meta"] / "activation_status.json",
        {
            "status": "running",
            "updated_at_utc": now,
            "completed_records": 0,
            "planned_records": len(work_items),
            "dependencies": dependency_status,
        },
    )
    progress_path = paths["checkpoints"] / "activation_progress.json"
    if not progress_path.exists():
        write_json(
            progress_path,
            {
                "cursor": 0,
                "completed_prompt_ids": [],
                "planned_prompt_ids": [item.prompt_id for item in work_items],
            },
        )
    append_log(paths["logs"] / "activation_cache.log", "activation cache run initialized")
    return paths


def run_activation_cache(
    run_root: Path,
    generations_jsonl: Path,
    model_config_path: Path,
    model_config: dict[str, Any],
    work_items: list[ActivationWorkItem],
    layers: list[int],
    backend: str,
    dependency_status: dict[str, Any],
    save_every: int,
) -> dict[str, Any]:
    if backend != "transformer_lens":
        raise NotImplementedError("Only transformer_lens backend is implemented")

    paths = write_initial_artifacts(
        run_root,
        generations_jsonl,
        model_config_path,
        model_config,
        work_items,
        layers,
        backend,
        dependency_status,
    )
    index_path = paths["results"] / "activation_index.jsonl"
    progress_path = paths["checkpoints"] / "activation_progress.json"
    status_path = paths["meta"] / "activation_status.json"
    log_path = paths["logs"] / "activation_cache.log"

    completed = completed_prompt_ids_from_index(index_path)
    progress = read_json(progress_path) or {}
    completed.update(str(prompt_id) for prompt_id in progress.get("completed_prompt_ids", []))
    remaining = [item for item in work_items if item.prompt_id not in completed]

    append_log(log_path, f"loading TransformerLens model {model_config['huggingface_model_name']}")
    model = load_transformer_lens_model(model_config)
    append_log(log_path, f"model loaded; remaining records={len(remaining)}")

    cached_this_run = 0
    for item in remaining:
        try:
            act_path = paths["activations"] / artifact_name(item.prompt_id)
            index_record = cache_one_transformer_lens(model, item, layers, act_path)
            append_jsonl(index_path, [index_record])
            completed.add(item.prompt_id)
            cached_this_run += 1
        except Exception as exc:
            write_json(
                status_path,
                {
                    "status": "failed",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "completed_records": len(completed),
                    "planned_records": len(work_items),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            append_log(log_path, f"failed on prompt_id={item.prompt_id}: {type(exc).__name__}: {exc}")
            raise

        if cached_this_run % save_every == 0:
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
            "activation_index": str(index_path),
        },
    )
    append_log(log_path, f"activation caching completed completed={len(completed)}")
    return {
        "status": "completed",
        "planned_records": len(work_items),
        "completed_records": len(completed),
        "cached_this_run": cached_this_run,
        "activation_index": str(index_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache residual-stream activations from generated completions.")
    parser.add_argument("--generations-jsonl", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--backend", choices=["transformer_lens", "transformers_hooks"], default="transformer_lens")
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    model_config = load_yaml(args.model_config)
    layer_policy = model_config["activation_extraction"]["first_layer_policy"]
    layers = args.layers if args.layers is not None else [int(layer) for layer in layer_policy["layers"]]
    generation_records = load_jsonl(args.generations_jsonl, limit=args.limit)
    work_items = build_work_items(generation_records)
    dependency_status = check_activation_dependencies(args.backend)

    if args.dry_run:
        write_dry_run_artifacts(
            run_root=args.run_root,
            generations_jsonl=args.generations_jsonl,
            model_config_path=args.model_config,
            model_config=model_config,
            work_items=work_items,
            layers=layers,
            backend=args.backend,
            dependency_status=dependency_status,
        )
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "run_root": str(args.run_root),
                    "planned_records": len(work_items),
                    "layers": layers,
                    "backend": args.backend,
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
                    "error": "activation dependencies are missing",
                    "backend": args.backend,
                    "dependencies": dependency_status,
                    "next_step": "Install torch and transformer_lens in the execution environment, then rerun or use --dry-run.",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    result = run_activation_cache(
        run_root=args.run_root,
        generations_jsonl=args.generations_jsonl,
        model_config_path=args.model_config,
        model_config=model_config,
        work_items=work_items,
        layers=layers,
        backend=args.backend,
        dependency_status=dependency_status,
        save_every=args.save_every,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
