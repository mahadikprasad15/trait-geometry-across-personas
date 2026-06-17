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

ROLE_ROW_FIELDS = [
    "trait_axis_id",
    "analysis_run_id",
    "behavior_run_id",
    "role_id",
    "positive_shift_scalar",
    "positive_behavior_shift",
    "positive_behavior_matched_shift",
    "negative_shift_scalar",
    "negative_behavior_shift",
    "negative_behavior_matched_shift",
    "axis_alignment_cosine",
    "mention_shift_scalar",
    "mention_to_elicitation_ratio",
    "mention_positive_behavior_shift",
    "mention_negative_behavior_shift",
    "gate_overall",
    "role_adherence_mean",
    "coherence_mean",
    "prompt_following_mean",
    "low_quality_flag",
]

TRAIT_ROW_FIELDS = [
    "trait_axis_id",
    "analysis_run_id",
    "behavior_run_id",
    "roles",
    "gate_overall",
    "low_quality_roles",
    "positive_shift_scalar_mean",
    "positive_behavior_shift_mean",
    "positive_behavior_matched_shift_mean",
    "negative_shift_scalar_mean",
    "negative_behavior_shift_mean",
    "negative_behavior_matched_shift_mean",
    "axis_alignment_mean",
    "mention_to_elicitation_ratio_max",
]


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
    behavior_run_id: str,
) -> dict[str, Path]:
    trait_root = base_root / trait_axis_id / role_scope
    analysis_root = trait_root / "analysis" / analysis_run_id
    behavior_root = trait_root / "behavior" / behavior_run_id
    return {
        "scalar_decomposition": analysis_root / "results" / "scalars" / "scalar_decomposition.json",
        "salience_gate": analysis_root / "results" / "gates" / "salience_gate.json",
        "behavior_metrics": behavior_root / "results" / "behavior_metrics.json",
    }


def rows_by_role(payload: dict[str, Any], artifact_name: str) -> dict[str, dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{artifact_name} must contain list key 'rows'")
    return {str(row["role_id"]): row for row in rows}


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def combine_rows(
    trait_axis_id: str,
    analysis_run_id: str,
    behavior_run_id: str,
    scalar_payload: dict[str, Any],
    gate_payload: dict[str, Any],
    behavior_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    scalar_by_role = rows_by_role(scalar_payload, "scalar_decomposition")
    gate_by_role = rows_by_role(gate_payload, "salience_gate")
    behavior_by_role = rows_by_role(behavior_payload, "behavior_metrics")
    scalar_roles = set(scalar_by_role)
    gate_roles = set(gate_by_role)
    behavior_roles = set(behavior_by_role)
    all_roles = scalar_roles | gate_roles | behavior_roles
    missing = {
        "missing_scalar": sorted(all_roles - scalar_roles),
        "missing_gate": sorted(all_roles - gate_roles),
        "missing_behavior": sorted(all_roles - behavior_roles),
    }
    if any(missing.values()):
        raise ValueError(f"role coverage mismatch for {trait_axis_id}: {missing}")
    roles = sorted(all_roles)

    combined = []
    for role_id in roles:
        scalar_row = scalar_by_role[role_id]
        gate_row = gate_by_role[role_id]
        behavior_row = behavior_by_role[role_id]
        combined.append(
            {
                "trait_axis_id": trait_axis_id,
                "analysis_run_id": analysis_run_id,
                "behavior_run_id": behavior_run_id,
                "role_id": role_id,
                "positive_shift_scalar": scalar_row.get("positive_shift_scalar"),
                "positive_behavior_shift": behavior_row.get("positive_behavior_shift"),
                "positive_behavior_matched_shift": behavior_row.get("positive_behavior_matched_shift"),
                "negative_shift_scalar": scalar_row.get("negative_shift_scalar"),
                "negative_behavior_shift": behavior_row.get("negative_behavior_shift"),
                "negative_behavior_matched_shift": behavior_row.get("negative_behavior_matched_shift"),
                "axis_alignment_cosine": scalar_row.get("axis_alignment_cosine"),
                "mention_shift_scalar": scalar_row.get("mention_shift_scalar"),
                "mention_to_elicitation_ratio": gate_row.get("mention_to_elicitation_ratio"),
                "mention_positive_behavior_shift": behavior_row.get("mention_positive_behavior_shift"),
                "mention_negative_behavior_shift": behavior_row.get("mention_negative_behavior_shift"),
                "gate_overall": gate_row.get("decision") or gate_row.get("overall"),
                "role_adherence_mean": behavior_row.get("role_adherence_mean"),
                "coherence_mean": behavior_row.get("coherence_mean"),
                "prompt_following_mean": behavior_row.get("prompt_following_mean"),
                "low_quality_flag": behavior_row.get("low_quality_flag"),
            }
        )
    return combined


def summarize_trait(
    trait_axis_id: str,
    analysis_run_id: str,
    behavior_run_id: str,
    role_rows: list[dict[str, Any]],
    gate_payload: dict[str, Any],
) -> dict[str, Any]:
    gate_summary = gate_payload.get("summary", {})
    return {
        "trait_axis_id": trait_axis_id,
        "analysis_run_id": analysis_run_id,
        "behavior_run_id": behavior_run_id,
        "roles": len(role_rows),
        "gate_overall": gate_summary.get("overall") if isinstance(gate_summary, dict) else None,
        "low_quality_roles": sum(1 for row in role_rows if row.get("low_quality_flag")),
        "positive_shift_scalar_mean": mean(numeric_values(role_rows, "positive_shift_scalar")),
        "positive_behavior_shift_mean": mean(numeric_values(role_rows, "positive_behavior_shift")),
        "positive_behavior_matched_shift_mean": mean(
            numeric_values(role_rows, "positive_behavior_matched_shift")
        ),
        "negative_shift_scalar_mean": mean(numeric_values(role_rows, "negative_shift_scalar")),
        "negative_behavior_shift_mean": mean(numeric_values(role_rows, "negative_behavior_shift")),
        "negative_behavior_matched_shift_mean": mean(
            numeric_values(role_rows, "negative_behavior_matched_shift")
        ),
        "axis_alignment_mean": mean(numeric_values(role_rows, "axis_alignment_cosine")),
        "mention_to_elicitation_ratio_max": max(
            numeric_values(role_rows, "mention_to_elicitation_ratio") or [0.0]
        ),
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


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    role_fields = [
        "trait_axis_id",
        "role_id",
        "positive_shift_scalar",
        "positive_behavior_matched_shift",
        "negative_shift_scalar",
        "negative_behavior_matched_shift",
        "axis_alignment_cosine",
        "gate_overall",
    ]
    text = (
        "# Scalar-Behavior Summary\n\n"
        f"Created at: {summary['created_at_utc']}\n\n"
        f"Base root: `{summary['base_root']}`\n\n"
        f"Role scope: `{summary['role_scope']}`\n\n"
        "## Trait Summary\n\n"
        + markdown_table(summary["trait_summary"], TRAIT_ROW_FIELDS)
        + "\n## Role Rows\n\n"
        + markdown_table(summary["role_rows"], role_fields)
    )
    path.write_text(text, encoding="utf-8")


def build_summary(
    base_root: Path,
    traits: list[str],
    role_scope: str,
    explicit_analysis_run_ids: dict[str, str],
    explicit_behavior_run_ids: dict[str, str],
) -> dict[str, Any]:
    trait_rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    inputs: dict[str, Any] = {}

    for trait_axis_id in traits:
        trait_root = base_root / trait_axis_id / role_scope
        analysis_run_id = resolve_run_id(
            trait_root / "analysis", trait_axis_id, explicit_analysis_run_ids
        )
        behavior_run_id = resolve_run_id(
            trait_root / "behavior", trait_axis_id, explicit_behavior_run_ids
        )
        paths = artifact_paths(base_root, trait_axis_id, role_scope, analysis_run_id, behavior_run_id)
        missing = {name: str(path) for name, path in paths.items() if not path.exists()}
        if missing:
            raise FileNotFoundError(f"missing artifacts for {trait_axis_id}: {missing}")

        scalar_payload = read_json(paths["scalar_decomposition"])
        gate_payload = read_json(paths["salience_gate"])
        behavior_payload = read_json(paths["behavior_metrics"])
        combined = combine_rows(
            trait_axis_id=trait_axis_id,
            analysis_run_id=analysis_run_id,
            behavior_run_id=behavior_run_id,
            scalar_payload=scalar_payload,
            gate_payload=gate_payload,
            behavior_payload=behavior_payload,
        )
        role_rows.extend(combined)
        trait_rows.append(summarize_trait(trait_axis_id, analysis_run_id, behavior_run_id, combined, gate_payload))
        inputs[trait_axis_id] = {
            "analysis_run_id": analysis_run_id,
            "behavior_run_id": behavior_run_id,
            "scalar_decomposition": str(paths["scalar_decomposition"]),
            "salience_gate": str(paths["salience_gate"]),
            "behavior_metrics": str(paths["behavior_metrics"]),
        }

    return {
        "schema_version": "0.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_root": str(base_root),
        "role_scope": role_scope,
        "inputs": inputs,
        "trait_summary": trait_rows,
        "role_rows": role_rows,
    }


def write_summary_artifacts(output_dir: Path, summary: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "scalar_behavior_summary.json"
    trait_csv_path = output_dir / "scalar_behavior_trait_summary.csv"
    role_csv_path = output_dir / "scalar_behavior_role_summary.csv"
    markdown_path = output_dir / "scalar_behavior_summary.md"
    write_json(json_path, summary)
    write_csv(trait_csv_path, summary["trait_summary"], TRAIT_ROW_FIELDS)
    write_csv(role_csv_path, summary["role_rows"], ROLE_ROW_FIELDS)
    write_markdown_report(markdown_path, summary)
    return {
        "summary_json": str(json_path),
        "trait_summary_csv": str(trait_csv_path),
        "role_summary_csv": str(role_csv_path),
        "markdown_report": str(markdown_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Join scalar, gate, and behavior summaries across traits.")
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--traits", nargs="+", default=DEFAULT_TRAITS)
    parser.add_argument("--role-scope", default="primary_roles")
    parser.add_argument(
        "--analysis-run-id",
        action="append",
        default=[],
        help="Optional analysis run id override: trait_axis_id=run_id.",
    )
    parser.add_argument(
        "--behavior-run-id",
        action="append",
        default=[],
        help="Optional behavior run id override: trait_axis_id=run_id.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = build_summary(
        base_root=args.base_root,
        traits=list(args.traits),
        role_scope=args.role_scope,
        explicit_analysis_run_ids=parse_run_id_args(args.analysis_run_id),
        explicit_behavior_run_ids=parse_run_id_args(args.behavior_run_id),
    )
    artifacts = write_summary_artifacts(args.output_dir, summary)
    print(
        json.dumps(
            {
                "status": "completed",
                "traits": len(summary["trait_summary"]),
                "role_rows": len(summary["role_rows"]),
                "artifacts": artifacts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
