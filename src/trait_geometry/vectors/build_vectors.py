from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_CONDITIONS = ("present_positive", "present_negative", "present_neutral")


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check_torch_dependency() -> dict[str, Any]:
    try:
        import torch

        return {"ready": True, "torch": {"installed": True, "version": torch.__version__}}
    except Exception as exc:
        return {
            "ready": False,
            "torch": {"installed": False, "error": f"{type(exc).__name__}: {exc}"},
        }


def summarize_index(index_records: list[dict[str, Any]]) -> dict[str, Any]:
    condition_counts = Counter(str(record["condition"]) for record in index_records)
    role_counts = Counter(str(record["role_id"]) for record in index_records)
    layers = sorted({int(layer) for record in index_records for layer in record["layers"]})
    missing_paths = [
        record["activation_path"]
        for record in index_records
        if not Path(record["activation_path"]).exists()
    ]
    return {
        "records": len(index_records),
        "condition_counts": dict(sorted(condition_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "layers": layers,
        "missing_activation_paths": missing_paths,
    }


def validate_condition_coverage(index_records: list[dict[str, Any]]) -> dict[str, Any]:
    by_role: dict[str, set[str]] = defaultdict(set)
    for record in index_records:
        condition = str(record["condition"])
        if condition == "mention_without_possession":
            continue
        by_role[str(record["role_id"])].add(condition)

    missing: dict[str, list[str]] = {}
    for role_id, conditions in by_role.items():
        absent = [condition for condition in REQUIRED_CONDITIONS if condition not in conditions]
        if absent:
            missing[role_id] = absent
    return {
        "required_conditions": list(REQUIRED_CONDITIONS),
        "missing_by_role": missing,
        "passed": not missing,
    }


def mean_tensor(tensors: list[Any]):
    import torch

    if not tensors:
        raise ValueError("cannot average an empty tensor list")
    return torch.stack(tensors, dim=0).mean(dim=0)


def load_activation_vector(activation_path: Path, layer: int):
    import torch

    payload = torch.load(activation_path, map_location="cpu")
    pooled = payload["pooled_resid_post"]
    if layer in pooled:
        return pooled[layer]
    layer_key = str(layer)
    if layer_key in pooled:
        return pooled[layer_key]
    raise ValueError(f"layer {layer} missing from {activation_path}")


def compute_vectors(index_records: list[dict[str, Any]], layers: list[int]) -> dict[str, Any]:
    grouped: dict[int, dict[str, dict[str, list[Any]]]] = {
        layer: defaultdict(lambda: defaultdict(list)) for layer in layers
    }
    counts: dict[int, dict[str, dict[str, int]]] = {
        layer: defaultdict(lambda: defaultdict(int)) for layer in layers
    }

    for record in index_records:
        condition = str(record["condition"])
        role_id = str(record["role_id"])
        activation_path = Path(record["activation_path"])
        for layer in layers:
            vector = load_activation_vector(activation_path, layer)
            grouped[layer][role_id][condition].append(vector)
            counts[layer][role_id][condition] += 1

    role_condition_means: dict[int, dict[str, dict[str, Any]]] = {}
    role_trait_vectors: dict[int, dict[str, dict[str, Any]]] = {}
    global_neutral_means: dict[int, Any] = {}

    for layer in layers:
        role_condition_means[layer] = {}
        neutral_means = []
        for role_id, condition_map in grouped[layer].items():
            missing = [condition for condition in REQUIRED_CONDITIONS if condition not in condition_map]
            if missing:
                raise ValueError(f"role {role_id} layer {layer} missing conditions {missing}")
            role_condition_means[layer][role_id] = {
                condition: mean_tensor(tensors)
                for condition, tensors in condition_map.items()
            }
            neutral_means.append(role_condition_means[layer][role_id]["present_neutral"])

        global_neutral_means[layer] = mean_tensor(neutral_means)
        role_trait_vectors[layer] = {}
        for role_id, means in role_condition_means[layer].items():
            positive = means["present_positive"]
            negative = means["present_negative"]
            neutral = means["present_neutral"]
            role_trait_vectors[layer][role_id] = {
                "positive_shift": positive - neutral,
                "negative_shift": negative - neutral,
                "axis_vector": positive - negative,
                "offset_vector": neutral - global_neutral_means[layer],
            }

    return {
        "role_condition_means": role_condition_means,
        "role_trait_vectors": role_trait_vectors,
        "global_neutral_means": global_neutral_means,
        "counts": counts,
    }


def tensor_counts_to_plain(counts: dict[int, dict[str, dict[str, int]]]) -> dict[str, Any]:
    return {
        str(layer): {
            role_id: dict(sorted(condition_counts.items()))
            for role_id, condition_counts in sorted(role_counts.items())
        }
        for layer, role_counts in sorted(counts.items())
    }


def write_vector_artifacts(
    output_dir: Path,
    activation_index: Path,
    layers: list[int],
    vector_payload: dict[str, Any],
    summary: dict[str, Any],
    coverage: dict[str, Any],
) -> dict[str, str]:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    means_path = output_dir / "role_condition_means.pt"
    vectors_path = output_dir / "role_trait_vectors.pt"
    index_path = output_dir / "vector_index.json"
    manifest_path = output_dir.parent.parent / "meta" / "vector_manifest.json"

    torch.save(
        {
            "layers": layers,
            "role_condition_means": vector_payload["role_condition_means"],
            "global_neutral_means": vector_payload["global_neutral_means"],
        },
        means_path,
    )
    torch.save(
        {
            "layers": layers,
            "role_trait_vectors": vector_payload["role_trait_vectors"],
            "counts": vector_payload["counts"],
        },
        vectors_path,
    )
    plain_counts = tensor_counts_to_plain(vector_payload["counts"])
    write_json(
        index_path,
        {
            "schema_version": "0.1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "activation_index": str(activation_index),
            "layers": layers,
            "role_condition_means": str(means_path),
            "role_trait_vectors": str(vectors_path),
            "counts": plain_counts,
        },
    )
    write_json(
        manifest_path,
        {
            "schema_version": "0.1",
            "builder": "VectorBuilder",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "activation_index": str(activation_index),
            "output_dir": str(output_dir),
            "layers": layers,
            "input_summary": summary,
            "condition_coverage": coverage,
            "counts": plain_counts,
            "artifacts": {
                "role_condition_means": str(means_path),
                "role_trait_vectors": str(vectors_path),
                "vector_index": str(index_path),
            },
        },
    )
    return {
        "role_condition_means": str(means_path),
        "role_trait_vectors": str(vectors_path),
        "vector_index": str(index_path),
        "vector_manifest": str(manifest_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build role-level vectors from cached activations.")
    parser.add_argument("--activation-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[8])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    index_records = load_jsonl(args.activation_index, limit=args.limit)
    summary = summarize_index(index_records)
    coverage = validate_condition_coverage(index_records)
    dependency_status = check_torch_dependency()

    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "activation_index": str(args.activation_index),
                    "output_dir": str(args.output_dir),
                    "layers": args.layers,
                    "summary": summary,
                    "condition_coverage": coverage,
                    "dependencies": dependency_status,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if coverage["passed"] and not summary["missing_activation_paths"] else 1

    if not dependency_status["ready"]:
        print(
            json.dumps(
                {
                    "error": "vector dependencies are missing",
                    "dependencies": dependency_status,
                    "next_step": "Install torch in the execution environment, then rerun or use --dry-run.",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    if summary["missing_activation_paths"]:
        print(
            json.dumps(
                {
                    "error": "activation paths are missing",
                    "missing_activation_paths": summary["missing_activation_paths"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    if not coverage["passed"]:
        print(json.dumps({"error": "condition coverage failed", "coverage": coverage}, indent=2))
        return 2

    vector_payload = compute_vectors(index_records, args.layers)
    artifacts = write_vector_artifacts(
        output_dir=args.output_dir,
        activation_index=args.activation_index,
        layers=args.layers,
        vector_payload=vector_payload,
        summary=summary,
        coverage=coverage,
    )
    print(json.dumps({"status": "completed", "artifacts": artifacts}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
