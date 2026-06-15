from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


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


def l2_normalize(vector: Any):
    norm = vector.norm()
    if float(norm) == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return vector / norm


def cosine(a: Any, b: Any) -> float:
    import torch

    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def load_role_trait_vectors(path: Path) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu")


def resolve_trait_axis_id(
    role_trait_payload: dict[str, Any] | None,
    experiment_config: dict[str, Any],
    explicit_trait_axis_id: str | None,
) -> str:
    if explicit_trait_axis_id:
        return explicit_trait_axis_id
    if role_trait_payload and role_trait_payload.get("trait_axis_id"):
        return str(role_trait_payload["trait_axis_id"])
    return str(experiment_config["smoke_run"]["trait_axis_id"])


def select_vectors(
    role_trait_payload: dict[str, Any],
    layer: int,
    roles: list[str],
    vector_type: str,
) -> dict[str, Any]:
    vectors_by_layer = role_trait_payload["role_trait_vectors"]
    layer_vectors = vectors_by_layer.get(layer) or vectors_by_layer.get(str(layer))
    if layer_vectors is None:
        raise ValueError(f"layer {layer} missing from role trait vectors")

    selected = {}
    missing = []
    for role in roles:
        role_vectors = layer_vectors.get(role)
        if role_vectors is None:
            missing.append(role)
            continue
        if vector_type not in role_vectors:
            raise ValueError(f"vector type {vector_type!r} missing for role {role!r}")
        selected[role] = role_vectors[vector_type]
    if missing:
        raise ValueError(f"selected roles missing from vectors: {missing}")
    return selected


def build_ruler(
    role_trait_payload: dict[str, Any],
    layer: int,
    roles: list[str],
    vector_type: str,
) -> dict[str, Any]:
    import torch

    selected = select_vectors(role_trait_payload, layer, roles, vector_type)
    stacked = torch.stack([selected[role] for role in roles], dim=0)
    raw_ruler = stacked.mean(dim=0)
    unit_ruler = l2_normalize(raw_ruler)

    pairwise_cosines: dict[str, float] = {}
    for i, role_a in enumerate(roles):
        for role_b in roles[i + 1 :]:
            pairwise_cosines[f"{role_a}__{role_b}"] = cosine(selected[role_a], selected[role_b])

    return {
        "layer": layer,
        "roles": roles,
        "vector_type": vector_type,
        "raw_ruler": raw_ruler,
        "unit_ruler": unit_ruler,
        "raw_norm": float(raw_ruler.norm()),
        "unit_norm": float(unit_ruler.norm()),
        "pairwise_cosines": pairwise_cosines,
    }


def write_ruler_artifacts(
    output_dir: Path,
    role_trait_vectors_path: Path,
    experiment_config_path: Path,
    trait_axis_id: str,
    method: str,
    ruler_payload: dict[str, Any],
) -> dict[str, str]:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_name = (
        f"{trait_axis_id}_layer{ruler_payload['layer']}_{method}_{ruler_payload['vector_type']}.pt"
    )
    ruler_path = output_dir / artifact_name
    index_path = output_dir / "ruler_index.json"
    manifest_path = output_dir.parent.parent / "meta" / "ruler_manifest.json"

    torch.save(
        {
            "trait_axis_id": trait_axis_id,
            "layer": ruler_payload["layer"],
            "roles": ruler_payload["roles"],
            "method": method,
            "vector_type": ruler_payload["vector_type"],
            "raw_ruler": ruler_payload["raw_ruler"],
            "unit_ruler": ruler_payload["unit_ruler"],
            "raw_norm": ruler_payload["raw_norm"],
            "unit_norm": ruler_payload["unit_norm"],
        },
        ruler_path,
    )

    index_payload = {
        "schema_version": "0.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "trait_axis_id": trait_axis_id,
        "layer": ruler_payload["layer"],
        "method": method,
        "vector_type": ruler_payload["vector_type"],
        "roles": ruler_payload["roles"],
        "ruler_path": str(ruler_path),
        "raw_norm": ruler_payload["raw_norm"],
        "unit_norm": ruler_payload["unit_norm"],
        "pairwise_cosines": ruler_payload["pairwise_cosines"],
    }
    write_json(index_path, index_payload)
    write_json(
        manifest_path,
        {
            "schema_version": "0.1",
            "builder": "RulerBuilder",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "role_trait_vectors": str(role_trait_vectors_path),
            "experiment_config": str(experiment_config_path),
            "output_dir": str(output_dir),
            "ruler": index_payload,
        },
    )
    return {
        "ruler": str(ruler_path),
        "ruler_index": str(index_path),
        "ruler_manifest": str(manifest_path),
    }


def dry_run_summary(
    experiment_config: dict[str, Any],
    layer: int,
    roles: list[str] | None,
    vector_type: str,
    role_trait_vectors_path: Path,
    method: str,
    trait_axis_id: str,
) -> dict[str, Any]:
    chosen_roles = resolve_roles(experiment_config, method, roles)
    return {
        "role_trait_vectors": str(role_trait_vectors_path),
        "trait_axis_id": trait_axis_id,
        "layer": layer,
        "roles": chosen_roles,
        "vector_type": vector_type,
        "method": method,
    }


def resolve_roles(
    experiment_config: dict[str, Any],
    method: str,
    explicit_roles: list[str] | None,
) -> list[str]:
    if explicit_roles:
        return explicit_roles
    if method == "primary_roles_mean":
        return list(experiment_config["role_sets"]["primary"])
    if method == "role_free_mean":
        return ["role_free"]
    raise ValueError(
        f"method {method!r} requires explicit --roles or one of: primary_roles_mean, role_free_mean"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a unit benchmark ruler from role vectors.")
    parser.add_argument("--role-trait-vectors", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--roles", nargs="+", default=None)
    parser.add_argument("--vector-type", default="axis_vector")
    parser.add_argument("--trait-axis-id", default=None)
    parser.add_argument("--method", choices=["primary_roles_mean", "role_free_mean"], default="primary_roles_mean")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    experiment_config = load_yaml(args.experiment_config)
    dependency_status = check_torch_dependency()
    role_trait_payload = None
    if not args.dry_run and dependency_status["ready"]:
        role_trait_payload = load_role_trait_vectors(args.role_trait_vectors)
    trait_axis_id = resolve_trait_axis_id(role_trait_payload, experiment_config, args.trait_axis_id)
    summary = dry_run_summary(
        experiment_config=experiment_config,
        layer=args.layer,
        roles=args.roles,
        vector_type=args.vector_type,
        role_trait_vectors_path=args.role_trait_vectors,
        method=args.method,
        trait_axis_id=trait_axis_id,
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "summary": summary,
                    "dependencies": dependency_status,
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
                    "error": "ruler dependencies are missing",
                    "dependencies": dependency_status,
                    "next_step": "Install torch in the execution environment, then rerun or use --dry-run.",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    roles = resolve_roles(experiment_config, args.method, args.roles)
    payload = role_trait_payload or load_role_trait_vectors(args.role_trait_vectors)
    trait_axis_id = resolve_trait_axis_id(payload, experiment_config, args.trait_axis_id)
    ruler_payload = build_ruler(payload, args.layer, roles, args.vector_type)
    artifacts = write_ruler_artifacts(
        output_dir=args.output_dir,
        role_trait_vectors_path=args.role_trait_vectors,
        experiment_config_path=args.experiment_config,
        trait_axis_id=trait_axis_id,
        method=args.method,
        ruler_payload=ruler_payload,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "artifacts": artifacts,
                "unit_norm": ruler_payload["unit_norm"],
                "pairwise_cosines": ruler_payload["pairwise_cosines"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
