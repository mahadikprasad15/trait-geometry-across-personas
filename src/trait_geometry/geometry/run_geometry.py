from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_TRAITS = [
    "warmth_coldness",
    "sincerity_manipulativeness",
    "caution_recklessness",
    "curiosity_closed_mindedness",
    "skepticism_gullibility",
]

DEFAULT_VECTOR_TYPES = ["axis_vector", "positive_shift", "negative_shift", "offset_vector"]

ROLE_PAIR_FIELDS = [
    "trait_axis_id",
    "analysis_run_id",
    "layer",
    "vector_type",
    "role_a",
    "role_b",
    "cosine",
]

ROLE_RULER_FIELDS = [
    "trait_axis_id",
    "analysis_run_id",
    "layer",
    "role_id",
    "vector_type",
    "cosine_to_ruler",
    "projection_on_ruler",
    "vector_norm",
    "ruler_method",
]

RULER_COSINE_FIELDS = [
    "trait_a",
    "trait_b",
    "analysis_run_id_a",
    "analysis_run_id_b",
    "layer",
    "ruler_method_a",
    "ruler_method_b",
    "cosine",
]

SAME_ROLE_CROSS_TRAIT_FIELDS = [
    "role_id",
    "trait_a",
    "trait_b",
    "analysis_run_id_a",
    "analysis_run_id_b",
    "layer",
    "vector_type",
    "cosine",
]

PCA_FIELDS = [
    "scope",
    "trait_axis_id",
    "vector_type",
    "rows",
    "dimensions",
    "pc1_explained_variance",
    "pc2_explained_variance",
    "pc3_explained_variance",
    "pcs_for_80pct",
    "pcs_for_90pct",
    "pcs_for_95pct",
]


def check_torch_dependency() -> dict[str, Any]:
    try:
        import torch

        return {"ready": True, "torch": {"installed": True, "version": torch.__version__}}
    except Exception as exc:
        return {
            "ready": False,
            "torch": {"installed": False, "error": f"{type(exc).__name__}: {exc}"},
        }


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{field: row.get(field) for field in fieldnames} for row in rows])


def parse_run_id_args(values: list[str]) -> dict[str, str]:
    run_ids: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"run id overrides must look like trait_axis_id=run_id, got {value!r}")
        trait_axis_id, run_id = value.split("=", 1)
        if not trait_axis_id or not run_id:
            raise ValueError(f"invalid run id override {value!r}")
        run_ids[trait_axis_id] = run_id
    return run_ids


def latest_run(root: Path) -> str:
    if not root.exists():
        raise FileNotFoundError(f"run root does not exist: {root}")
    candidates = sorted(path for path in root.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"no runs found under {root}")
    return candidates[-1].name


def resolve_run_id(root: Path, trait_axis_id: str, explicit_run_ids: dict[str, str]) -> str:
    return explicit_run_ids.get(trait_axis_id) or latest_run(root)


def artifact_paths(
    base_root: Path,
    trait_axis_id: str,
    role_scope: str,
    analysis_run_id: str,
    layer: int,
    ruler_method: str,
    ruler_vector_type: str,
) -> dict[str, Path]:
    analysis_root = base_root / trait_axis_id / role_scope / "analysis" / analysis_run_id
    return {
        "role_trait_vectors": analysis_root / "results" / "vectors" / "role_trait_vectors.pt",
        "ruler": (
            analysis_root
            / "results"
            / "rulers"
            / f"{trait_axis_id}_layer{layer}_{ruler_method}_{ruler_vector_type}.pt"
        ),
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


def layer_map(payload: dict[str, Any], key: str, layer: int) -> dict[str, Any]:
    by_layer = payload.get(key)
    if not isinstance(by_layer, dict):
        raise ValueError(f"payload key {key!r} must be a mapping")
    result = by_layer.get(layer) or by_layer.get(str(layer))
    if not isinstance(result, dict):
        raise ValueError(f"layer {layer} missing from payload key {key!r}")
    return result


def cosine(a: Any, b: Any) -> float | None:
    import torch

    if float(a.norm()) == 0.0 or float(b.norm()) == 0.0:
        return None
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def projection(vector: Any, unit_vector: Any) -> float:
    return float(vector.dot(unit_vector))


def vector_norm(vector: Any) -> float:
    return float(vector.norm())


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def build_role_pair_rows(
    trait_axis_id: str,
    analysis_run_id: str,
    layer: int,
    vectors_by_role: dict[str, dict[str, Any]],
    vector_types: list[str],
    roles: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for vector_type in vector_types:
        for i, role_a in enumerate(roles):
            for role_b in roles[i + 1 :]:
                rows.append(
                    {
                        "trait_axis_id": trait_axis_id,
                        "analysis_run_id": analysis_run_id,
                        "layer": layer,
                        "vector_type": vector_type,
                        "role_a": role_a,
                        "role_b": role_b,
                        "cosine": cosine(
                            vectors_by_role[role_a][vector_type],
                            vectors_by_role[role_b][vector_type],
                        ),
                    }
                )
    return rows


def build_role_ruler_rows(
    trait_axis_id: str,
    analysis_run_id: str,
    layer: int,
    vectors_by_role: dict[str, dict[str, Any]],
    ruler_payload: dict[str, Any],
    roles: list[str],
    vector_type: str,
) -> list[dict[str, Any]]:
    unit_ruler = ruler_payload["unit_ruler"]
    rows = []
    for role_id in roles:
        vector = vectors_by_role[role_id][vector_type]
        rows.append(
            {
                "trait_axis_id": trait_axis_id,
                "analysis_run_id": analysis_run_id,
                "layer": layer,
                "role_id": role_id,
                "vector_type": vector_type,
                "cosine_to_ruler": cosine(vector, unit_ruler),
                "projection_on_ruler": projection(vector, unit_ruler),
                "vector_norm": vector_norm(vector),
                "ruler_method": ruler_payload.get("method"),
            }
        )
    return rows


def build_ruler_cosine_rows(
    trait_payloads: dict[str, dict[str, Any]],
    layer: int,
) -> list[dict[str, Any]]:
    traits = sorted(trait_payloads)
    rows = []
    for i, trait_a in enumerate(traits):
        for trait_b in traits[i + 1 :]:
            payload_a = trait_payloads[trait_a]
            payload_b = trait_payloads[trait_b]
            ruler_a = payload_a["ruler_payload"]
            ruler_b = payload_b["ruler_payload"]
            rows.append(
                {
                    "trait_a": trait_a,
                    "trait_b": trait_b,
                    "analysis_run_id_a": payload_a["analysis_run_id"],
                    "analysis_run_id_b": payload_b["analysis_run_id"],
                    "layer": layer,
                    "ruler_method_a": ruler_a.get("method"),
                    "ruler_method_b": ruler_b.get("method"),
                    "cosine": cosine(ruler_a["unit_ruler"], ruler_b["unit_ruler"]),
                }
            )
    return rows


def build_same_role_cross_trait_rows(
    trait_payloads: dict[str, dict[str, Any]],
    layer: int,
    vector_types: list[str],
) -> list[dict[str, Any]]:
    traits = sorted(trait_payloads)
    common_roles = sorted(
        set.intersection(
            *[
                set(payload["vectors_by_role"].keys())
                for payload in trait_payloads.values()
            ]
        )
    )
    rows = []
    for role_id in common_roles:
        for vector_type in vector_types:
            for i, trait_a in enumerate(traits):
                for trait_b in traits[i + 1 :]:
                    payload_a = trait_payloads[trait_a]
                    payload_b = trait_payloads[trait_b]
                    rows.append(
                        {
                            "role_id": role_id,
                            "trait_a": trait_a,
                            "trait_b": trait_b,
                            "analysis_run_id_a": payload_a["analysis_run_id"],
                            "analysis_run_id_b": payload_b["analysis_run_id"],
                            "layer": layer,
                            "vector_type": vector_type,
                            "cosine": cosine(
                                payload_a["vectors_by_role"][role_id][vector_type],
                                payload_b["vectors_by_role"][role_id][vector_type],
                            ),
                        }
                    )
    return rows


def pca_summary(scope: str, trait_axis_id: str | None, vector_type: str, vectors: list[Any]) -> dict[str, Any]:
    import torch

    if not vectors:
        return {
            "scope": scope,
            "trait_axis_id": trait_axis_id,
            "vector_type": vector_type,
            "rows": 0,
            "dimensions": 0,
        }
    matrix = torch.stack(vectors, dim=0).float()
    rows, dimensions = matrix.shape
    if rows < 2:
        return {
            "scope": scope,
            "trait_axis_id": trait_axis_id,
            "vector_type": vector_type,
            "rows": rows,
            "dimensions": dimensions,
            "pc1_explained_variance": None,
            "pc2_explained_variance": None,
            "pc3_explained_variance": None,
            "pcs_for_80pct": None,
            "pcs_for_90pct": None,
            "pcs_for_95pct": None,
        }

    centered = matrix - matrix.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    variances = singular_values.pow(2)
    total = float(variances.sum())
    ratios = (variances / total).tolist() if total > 0.0 else []

    def ratio_at(index: int) -> float | None:
        if index >= len(ratios):
            return None
        return float(ratios[index])

    def pcs_for(threshold: float) -> int | None:
        if not ratios:
            return None
        cumulative = 0.0
        for index, ratio in enumerate(ratios, start=1):
            cumulative += float(ratio)
            if cumulative >= threshold:
                return index
        return len(ratios)

    return {
        "scope": scope,
        "trait_axis_id": trait_axis_id,
        "vector_type": vector_type,
        "rows": rows,
        "dimensions": dimensions,
        "pc1_explained_variance": ratio_at(0),
        "pc2_explained_variance": ratio_at(1),
        "pc3_explained_variance": ratio_at(2),
        "pcs_for_80pct": pcs_for(0.80),
        "pcs_for_90pct": pcs_for(0.90),
        "pcs_for_95pct": pcs_for(0.95),
    }


def build_pca_rows(
    trait_payloads: dict[str, dict[str, Any]],
    vector_types: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for vector_type in vector_types:
        all_vectors = []
        for trait_axis_id, payload in sorted(trait_payloads.items()):
            trait_vectors = [
                role_vectors[vector_type]
                for role_vectors in payload["vectors_by_role"].values()
            ]
            all_vectors.extend(trait_vectors)
            rows.append(pca_summary("within_trait_roles", trait_axis_id, vector_type, trait_vectors))
        rows.append(pca_summary("all_trait_role_vectors", None, vector_type, all_vectors))
    return rows


def load_trait_payloads(
    base_root: Path,
    traits: list[str],
    role_scope: str,
    explicit_run_ids: dict[str, str],
    layer: int,
    roles: list[str] | None,
    vector_types: list[str],
    ruler_method: str,
    ruler_vector_type: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    payloads = {}
    resolved_inputs = []
    for trait_axis_id in traits:
        analysis_root = base_root / trait_axis_id / role_scope / "analysis"
        analysis_run_id = resolve_run_id(analysis_root, trait_axis_id, explicit_run_ids)
        paths = artifact_paths(
            base_root=base_root,
            trait_axis_id=trait_axis_id,
            role_scope=role_scope,
            analysis_run_id=analysis_run_id,
            layer=layer,
            ruler_method=ruler_method,
            ruler_vector_type=ruler_vector_type,
        )
        input_check = check_input_paths(paths)
        if not input_check["passed"]:
            raise FileNotFoundError(
                f"geometry inputs are missing for {trait_axis_id}: {input_check['missing']}"
            )
        role_trait_payload = load_torch_payload(paths["role_trait_vectors"])
        ruler_payload = load_torch_payload(paths["ruler"])
        vectors_by_role = layer_map(role_trait_payload, "role_trait_vectors", layer)
        selected_roles = roles or sorted(vectors_by_role)
        missing_roles = [role for role in selected_roles if role not in vectors_by_role]
        if missing_roles:
            raise ValueError(f"selected roles missing for {trait_axis_id}: {missing_roles}")
        for role_id in selected_roles:
            missing_vector_types = [
                vector_type
                for vector_type in vector_types
                if vector_type not in vectors_by_role[role_id]
            ]
            if missing_vector_types:
                raise ValueError(
                    f"role {role_id} in {trait_axis_id} missing vector types {missing_vector_types}"
                )
        payloads[trait_axis_id] = {
            "analysis_run_id": analysis_run_id,
            "role_trait_payload": role_trait_payload,
            "ruler_payload": ruler_payload,
            "vectors_by_role": {role: vectors_by_role[role] for role in selected_roles},
            "roles": selected_roles,
            "paths": {name: str(path) for name, path in paths.items()},
        }
        resolved_inputs.append(
            {
                "trait_axis_id": trait_axis_id,
                "analysis_run_id": analysis_run_id,
                "role_trait_vectors": str(paths["role_trait_vectors"]),
                "ruler": str(paths["ruler"]),
                "roles": selected_roles,
            }
        )
    return payloads, resolved_inputs


def summarize_geometry(
    role_pair_rows: list[dict[str, Any]],
    role_ruler_rows: list[dict[str, Any]],
    ruler_cosine_rows: list[dict[str, Any]],
    same_role_rows: list[dict[str, Any]],
    pca_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    axis_role_pairs = [
        row for row in role_pair_rows if row.get("vector_type") == "axis_vector"
    ]
    axis_same_role = [
        row for row in same_role_rows if row.get("vector_type") == "axis_vector"
    ]
    return {
        "role_pair_cosine_axis_mean": mean(numeric_values(axis_role_pairs, "cosine")),
        "role_pair_cosine_axis_min": min(numeric_values(axis_role_pairs, "cosine") or [0.0]),
        "role_pair_cosine_axis_max": max(numeric_values(axis_role_pairs, "cosine") or [0.0]),
        "role_ruler_alignment_mean": mean(numeric_values(role_ruler_rows, "cosine_to_ruler")),
        "ruler_cosine_mean": mean(numeric_values(ruler_cosine_rows, "cosine")),
        "same_role_cross_trait_axis_mean": mean(numeric_values(axis_same_role, "cosine")),
        "rows": {
            "role_pair_cosines": len(role_pair_rows),
            "role_ruler_alignment": len(role_ruler_rows),
            "ruler_cosines": len(ruler_cosine_rows),
            "same_role_cross_trait_cosines": len(same_role_rows),
            "pca_summary": len(pca_rows),
        },
    }


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    header = "| " + " | ".join(fields) + " |"
    divider = "| " + " | ".join("---" for _ in fields) + " |"
    body = []
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append("" if value is None else str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body]) + "\n"


def write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        "# Geometry Summary\n\n"
        f"Created at: {payload['created_at_utc']}\n\n"
        f"Base root: `{payload['base_root']}`\n\n"
        f"Layer: `{payload['layer']}`\n\n"
        "## Summary\n\n"
        + markdown_table([payload["summary"]], list(payload["summary"].keys()))
        + "\n## Ruler Cosines\n\n"
        + markdown_table(payload["ruler_cosines"], RULER_COSINE_FIELDS)
        + "\n## PCA Summary\n\n"
        + markdown_table(payload["pca_summary"], PCA_FIELDS)
        + "\n## Role-Ruler Alignment\n\n"
        + markdown_table(payload["role_ruler_alignment"], ROLE_RULER_FIELDS)
    )
    path.write_text(text, encoding="utf-8")


def build_geometry_summary(
    base_root: Path,
    traits: list[str],
    role_scope: str,
    explicit_run_ids: dict[str, str],
    layer: int,
    roles: list[str] | None,
    vector_types: list[str],
    ruler_method: str,
    ruler_vector_type: str,
) -> dict[str, Any]:
    trait_payloads, resolved_inputs = load_trait_payloads(
        base_root=base_root,
        traits=traits,
        role_scope=role_scope,
        explicit_run_ids=explicit_run_ids,
        layer=layer,
        roles=roles,
        vector_types=vector_types,
        ruler_method=ruler_method,
        ruler_vector_type=ruler_vector_type,
    )

    role_pair_rows = []
    role_ruler_rows = []
    for trait_axis_id, payload in sorted(trait_payloads.items()):
        role_pair_rows.extend(
            build_role_pair_rows(
                trait_axis_id=trait_axis_id,
                analysis_run_id=payload["analysis_run_id"],
                layer=layer,
                vectors_by_role=payload["vectors_by_role"],
                vector_types=vector_types,
                roles=payload["roles"],
            )
        )
        role_ruler_rows.extend(
            build_role_ruler_rows(
                trait_axis_id=trait_axis_id,
                analysis_run_id=payload["analysis_run_id"],
                layer=layer,
                vectors_by_role=payload["vectors_by_role"],
                ruler_payload=payload["ruler_payload"],
                roles=payload["roles"],
                vector_type=ruler_vector_type,
            )
        )

    ruler_cosine_rows = build_ruler_cosine_rows(trait_payloads, layer)
    same_role_rows = build_same_role_cross_trait_rows(trait_payloads, layer, vector_types)
    pca_rows = build_pca_rows(trait_payloads, vector_types)
    summary = summarize_geometry(
        role_pair_rows=role_pair_rows,
        role_ruler_rows=role_ruler_rows,
        ruler_cosine_rows=ruler_cosine_rows,
        same_role_rows=same_role_rows,
        pca_rows=pca_rows,
    )

    return {
        "schema_version": "0.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_root": str(base_root),
        "role_scope": role_scope,
        "traits": traits,
        "layer": layer,
        "roles": roles,
        "vector_types": vector_types,
        "ruler_method": ruler_method,
        "ruler_vector_type": ruler_vector_type,
        "resolved_inputs": resolved_inputs,
        "summary": summary,
        "role_pair_cosines": role_pair_rows,
        "role_ruler_alignment": role_ruler_rows,
        "ruler_cosines": ruler_cosine_rows,
        "same_role_cross_trait_cosines": same_role_rows,
        "pca_summary": pca_rows,
    }


def write_geometry_artifacts(output_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "geometry_summary.json"
    role_pair_csv = output_dir / "role_pair_cosines.csv"
    role_ruler_csv = output_dir / "role_ruler_alignment.csv"
    ruler_cosines_csv = output_dir / "ruler_cosines.csv"
    same_role_csv = output_dir / "same_role_cross_trait_cosines.csv"
    pca_csv = output_dir / "pca_summary.csv"
    markdown_report = output_dir / "geometry_summary.md"
    manifest_path = output_dir / "geometry_manifest.json"

    write_json(summary_json, payload)
    write_csv(role_pair_csv, payload["role_pair_cosines"], ROLE_PAIR_FIELDS)
    write_csv(role_ruler_csv, payload["role_ruler_alignment"], ROLE_RULER_FIELDS)
    write_csv(ruler_cosines_csv, payload["ruler_cosines"], RULER_COSINE_FIELDS)
    write_csv(same_role_csv, payload["same_role_cross_trait_cosines"], SAME_ROLE_CROSS_TRAIT_FIELDS)
    write_csv(pca_csv, payload["pca_summary"], PCA_FIELDS)
    write_markdown_report(markdown_report, payload)
    write_json(
        manifest_path,
        {
            "schema_version": "0.1",
            "builder": "GeometryAnalyzer",
            "created_at_utc": payload["created_at_utc"],
            "base_root": payload["base_root"],
            "output_dir": str(output_dir),
            "resolved_inputs": payload["resolved_inputs"],
            "artifacts": {
                "summary_json": str(summary_json),
                "role_pair_cosines_csv": str(role_pair_csv),
                "role_ruler_alignment_csv": str(role_ruler_csv),
                "ruler_cosines_csv": str(ruler_cosines_csv),
                "same_role_cross_trait_cosines_csv": str(same_role_csv),
                "pca_summary_csv": str(pca_csv),
                "markdown_report": str(markdown_report),
            },
        },
    )
    return {
        "summary_json": str(summary_json),
        "role_pair_cosines_csv": str(role_pair_csv),
        "role_ruler_alignment_csv": str(role_ruler_csv),
        "ruler_cosines_csv": str(ruler_cosines_csv),
        "same_role_cross_trait_cosines_csv": str(same_role_csv),
        "pca_summary_csv": str(pca_csv),
        "markdown_report": str(markdown_report),
        "manifest": str(manifest_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze role and trait geometry from vector/ruler artifacts.")
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--traits", nargs="+", default=DEFAULT_TRAITS)
    parser.add_argument("--role-scope", default="primary_roles")
    parser.add_argument("--run-id", action="append", default=[], help="Optional run id override: trait_axis_id=run_id.")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--roles", nargs="+", default=None)
    parser.add_argument("--vector-types", nargs="+", default=DEFAULT_VECTOR_TYPES)
    parser.add_argument("--ruler-method", default="primary_roles_mean")
    parser.add_argument("--ruler-vector-type", default="axis_vector")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    explicit_run_ids = parse_run_id_args(args.run_id)
    if args.dry_run:
        resolved = []
        for trait_axis_id in args.traits:
            analysis_root = args.base_root / trait_axis_id / args.role_scope / "analysis"
            analysis_run_id = resolve_run_id(analysis_root, trait_axis_id, explicit_run_ids)
            paths = artifact_paths(
                base_root=args.base_root,
                trait_axis_id=trait_axis_id,
                role_scope=args.role_scope,
                analysis_run_id=analysis_run_id,
                layer=args.layer,
                ruler_method=args.ruler_method,
                ruler_vector_type=args.ruler_vector_type,
            )
            resolved.append(
                {
                    "trait_axis_id": trait_axis_id,
                    "analysis_run_id": analysis_run_id,
                    "input_paths": {name: str(path) for name, path in paths.items()},
                    "input_check": check_input_paths(paths),
                }
            )
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "dependencies": check_torch_dependency(),
                    "resolved_inputs": resolved,
                    "output_dir": str(args.output_dir),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    dependency_status = check_torch_dependency()
    if not dependency_status["ready"]:
        print(
            json.dumps(
                {
                    "error": "geometry dependencies are missing",
                    "dependencies": dependency_status,
                    "next_step": "Install torch in the execution environment, then rerun.",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    payload = build_geometry_summary(
        base_root=args.base_root,
        traits=args.traits,
        role_scope=args.role_scope,
        explicit_run_ids=explicit_run_ids,
        layer=args.layer,
        roles=args.roles,
        vector_types=args.vector_types,
        ruler_method=args.ruler_method,
        ruler_vector_type=args.ruler_vector_type,
    )
    artifacts = write_geometry_artifacts(args.output_dir, payload)
    print(
        json.dumps(
            {
                "status": "completed",
                "traits": len(args.traits),
                "summary": payload["summary"],
                "artifacts": artifacts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
