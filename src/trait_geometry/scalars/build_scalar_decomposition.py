from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def check_input_paths(paths: dict[str, Path]) -> dict[str, Any]:
    missing = {name: str(path) for name, path in paths.items() if not path.exists()}
    return {"passed": not missing, "missing": missing}


def load_torch_payload(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a dictionary payload")
    return payload


def get_layer_map(payload: dict[str, Any], key: str, layer: int) -> dict[str, Any]:
    by_layer = payload[key]
    layer_payload = by_layer.get(layer) or by_layer.get(str(layer))
    if layer_payload is None:
        raise ValueError(f"layer {layer} missing from payload key {key!r}")
    if not isinstance(layer_payload, dict):
        raise ValueError(f"payload key {key!r} layer {layer} must be a mapping")
    return layer_payload


def l2_norm(vector: Any) -> float:
    return float(vector.norm())


def dot_projection(vector: Any, unit_ruler: Any) -> float:
    return float(vector.dot(unit_ruler))


def cosine_alignment(vector: Any, unit_ruler: Any) -> float | None:
    import torch

    if float(vector.norm()) == 0.0:
        return None
    return float(torch.nn.functional.cosine_similarity(vector, unit_ruler, dim=0))


def scalar_or_none(vector: Any | None, unit_ruler: Any) -> float | None:
    if vector is None:
        return None
    return dot_projection(vector, unit_ruler)


def norm_or_none(vector: Any | None) -> float | None:
    if vector is None:
        return None
    return l2_norm(vector)


def load_ruler(path: Path) -> dict[str, Any]:
    payload = load_torch_payload(path)
    required = ["trait_axis_id", "layer", "method", "vector_type", "unit_ruler"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"ruler payload missing keys: {missing}")
    return payload


def compute_role_row(
    role_id: str,
    condition_means: dict[str, Any],
    trait_vectors: dict[str, Any],
    unit_ruler: Any,
    layer: int,
    ruler_payload: dict[str, Any],
) -> dict[str, Any]:
    positive_shift = trait_vectors["positive_shift"]
    negative_shift = trait_vectors["negative_shift"]
    axis_vector = trait_vectors["axis_vector"]
    offset_vector = trait_vectors["offset_vector"]

    neutral_mean = condition_means["present_neutral"]
    positive_mean = condition_means["present_positive"]
    negative_mean = condition_means["present_negative"]
    mention_mean = condition_means.get("mention_without_possession")
    mention_shift = mention_mean - neutral_mean if mention_mean is not None else None

    return {
        "trait_axis_id": str(ruler_payload["trait_axis_id"]),
        "role_id": role_id,
        "layer": layer,
        "ruler_method": str(ruler_payload["method"]),
        "ruler_vector_type": str(ruler_payload["vector_type"]),
        "offset_scalar": dot_projection(offset_vector, unit_ruler),
        "positive_shift_scalar": dot_projection(positive_shift, unit_ruler),
        "negative_shift_scalar": dot_projection(negative_shift, unit_ruler),
        "axis_projection_scalar": dot_projection(axis_vector, unit_ruler),
        "axis_alignment_cosine": cosine_alignment(axis_vector, unit_ruler),
        "neutral_projection": dot_projection(neutral_mean, unit_ruler),
        "positive_projection": dot_projection(positive_mean, unit_ruler),
        "negative_projection": dot_projection(negative_mean, unit_ruler),
        "mention_projection": scalar_or_none(mention_mean, unit_ruler),
        "mention_shift_scalar": scalar_or_none(mention_shift, unit_ruler),
        "offset_norm": l2_norm(offset_vector),
        "positive_shift_norm": l2_norm(positive_shift),
        "negative_shift_norm": l2_norm(negative_shift),
        "axis_norm": l2_norm(axis_vector),
        "mention_shift_norm": norm_or_none(mention_shift),
    }


def build_scalar_rows(
    role_condition_payload: dict[str, Any],
    role_trait_payload: dict[str, Any],
    ruler_payload: dict[str, Any],
    layer: int,
    roles: list[str] | None,
) -> list[dict[str, Any]]:
    if int(ruler_payload["layer"]) != layer:
        raise ValueError(f"ruler layer {ruler_payload['layer']} does not match requested layer {layer}")

    condition_by_role = get_layer_map(role_condition_payload, "role_condition_means", layer)
    vectors_by_role = get_layer_map(role_trait_payload, "role_trait_vectors", layer)
    unit_ruler = ruler_payload["unit_ruler"]

    selected_roles = roles or sorted(vectors_by_role)
    rows = []
    missing = []
    for role_id in selected_roles:
        if role_id not in condition_by_role or role_id not in vectors_by_role:
            missing.append(role_id)
            continue
        rows.append(
            compute_role_row(
                role_id=role_id,
                condition_means=condition_by_role[role_id],
                trait_vectors=vectors_by_role[role_id],
                unit_ruler=unit_ruler,
                layer=layer,
                ruler_payload=ruler_payload,
            )
        )
    if missing:
        raise ValueError(f"selected roles missing from scalar inputs: {missing}")
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0, "roles": []}

    numeric_keys = [
        "offset_scalar",
        "positive_shift_scalar",
        "negative_shift_scalar",
        "axis_projection_scalar",
        "axis_alignment_cosine",
        "mention_shift_scalar",
    ]
    summary: dict[str, Any] = {"rows": len(rows), "roles": [row["role_id"] for row in rows]}
    for key in numeric_keys:
        values = [row[key] for row in rows if row.get(key) is not None]
        if values:
            summary[key] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trait_axis_id",
        "role_id",
        "layer",
        "ruler_method",
        "ruler_vector_type",
        "offset_scalar",
        "positive_shift_scalar",
        "negative_shift_scalar",
        "axis_projection_scalar",
        "axis_alignment_cosine",
        "neutral_projection",
        "positive_projection",
        "negative_projection",
        "mention_projection",
        "mention_shift_scalar",
        "offset_norm",
        "positive_shift_norm",
        "negative_shift_norm",
        "axis_norm",
        "mention_shift_norm",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_scalar_artifacts(
    output_dir: Path,
    role_condition_means_path: Path,
    role_trait_vectors_path: Path,
    ruler_path: Path,
    layer: int,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "scalar_decomposition.json"
    csv_path = output_dir / "scalar_decomposition.csv"
    manifest_path = output_dir.parent.parent / "meta" / "scalar_decomposition_manifest.json"

    write_json(
        json_path,
        {
            "schema_version": "0.1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "layer": layer,
            "summary": summary,
            "rows": rows,
        },
    )
    write_csv(csv_path, rows)
    write_json(
        manifest_path,
        {
            "schema_version": "0.1",
            "builder": "ScalarDecompositionBuilder",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "role_condition_means": str(role_condition_means_path),
                "role_trait_vectors": str(role_trait_vectors_path),
                "ruler": str(ruler_path),
            },
            "output_dir": str(output_dir),
            "layer": layer,
            "summary": summary,
            "artifacts": {
                "scalar_decomposition_json": str(json_path),
                "scalar_decomposition_csv": str(csv_path),
                "scalar_decomposition_manifest": str(manifest_path),
            },
        },
    )
    return {
        "scalar_decomposition_json": str(json_path),
        "scalar_decomposition_csv": str(csv_path),
        "scalar_decomposition_manifest": str(manifest_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project role trait vectors onto a benchmark ruler.")
    parser.add_argument("--role-condition-means", type=Path, required=True)
    parser.add_argument("--role-trait-vectors", type=Path, required=True)
    parser.add_argument("--ruler", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--roles", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    input_paths = {
        "role_condition_means": args.role_condition_means,
        "role_trait_vectors": args.role_trait_vectors,
        "ruler": args.ruler,
    }
    path_status = check_input_paths(input_paths)
    dependency_status = check_torch_dependency()

    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "summary": {
                        "role_condition_means": str(args.role_condition_means),
                        "role_trait_vectors": str(args.role_trait_vectors),
                        "ruler": str(args.ruler),
                        "output_dir": str(args.output_dir),
                        "layer": args.layer,
                        "roles": args.roles or "all",
                    },
                    "input_paths": path_status,
                    "dependencies": dependency_status,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if path_status["passed"] else 1

    if not dependency_status["ready"]:
        print(
            json.dumps(
                {
                    "error": "scalar decomposition dependencies are missing",
                    "dependencies": dependency_status,
                    "next_step": "Install torch in the execution environment, then rerun or use --dry-run.",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    if not path_status["passed"]:
        print(json.dumps({"error": "scalar inputs are missing", "input_paths": path_status}, indent=2))
        return 2

    role_condition_payload = load_torch_payload(args.role_condition_means)
    role_trait_payload = load_torch_payload(args.role_trait_vectors)
    ruler_payload = load_ruler(args.ruler)
    rows = build_scalar_rows(
        role_condition_payload=role_condition_payload,
        role_trait_payload=role_trait_payload,
        ruler_payload=ruler_payload,
        layer=args.layer,
        roles=args.roles,
    )
    summary = summarize_rows(rows)
    artifacts = write_scalar_artifacts(
        output_dir=args.output_dir,
        role_condition_means_path=args.role_condition_means,
        role_trait_vectors_path=args.role_trait_vectors,
        ruler_path=args.ruler,
        layer=args.layer,
        rows=rows,
        summary=summary,
    )
    print(json.dumps({"status": "completed", "summary": summary, "artifacts": artifacts}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
