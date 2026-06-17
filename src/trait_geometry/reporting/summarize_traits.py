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
    "run_id",
    "role_id",
    "layer",
    "ruler_method",
    "offset_scalar",
    "positive_shift_scalar",
    "negative_shift_scalar",
    "axis_projection_scalar",
    "axis_alignment_cosine",
    "mention_shift_scalar",
    "mention_to_elicitation_ratio",
    "positive_direction_status",
    "negative_direction_status",
    "mention_control_status",
    "axis_alignment_status",
    "gate_overall",
]

TRAIT_ROW_FIELDS = [
    "trait_axis_id",
    "run_id",
    "roles",
    "gate_overall",
    "gate_pass",
    "gate_warn",
    "gate_fail",
    "positive_shift_mean",
    "negative_shift_mean",
    "axis_alignment_mean",
    "axis_alignment_min",
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
            raise ValueError(f"--run-id values must look like trait_axis_id=run_id, got {value!r}")
        trait_axis_id, run_id = value.split("=", 1)
        if not trait_axis_id or not run_id:
            raise ValueError(f"invalid --run-id value {value!r}")
        run_ids[trait_axis_id] = run_id
    return run_ids


def latest_analysis_run(base_root: Path, trait_axis_id: str, role_scope: str) -> str:
    analysis_root = base_root / trait_axis_id / role_scope / "analysis"
    if not analysis_root.exists():
        raise FileNotFoundError(f"analysis root does not exist: {analysis_root}")
    candidates = sorted(path for path in analysis_root.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"no analysis runs found under {analysis_root}")
    return candidates[-1].name


def resolve_run_id(
    base_root: Path,
    trait_axis_id: str,
    role_scope: str,
    explicit_run_ids: dict[str, str],
) -> str:
    return explicit_run_ids.get(trait_axis_id) or latest_analysis_run(base_root, trait_axis_id, role_scope)


def artifact_paths(base_root: Path, trait_axis_id: str, role_scope: str, run_id: str) -> dict[str, Path]:
    analysis_root = base_root / trait_axis_id / role_scope / "analysis" / run_id
    return {
        "analysis_root": analysis_root,
        "scalar_decomposition": analysis_root / "results" / "scalars" / "scalar_decomposition.json",
        "salience_gate": analysis_root / "results" / "gates" / "salience_gate.json",
    }


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def gate_rows_by_role(gate_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = gate_payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("salience gate payload must contain list key 'rows'")
    return {str(row["role_id"]): row for row in rows}


def scalar_rows(scalar_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = scalar_payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("scalar payload must contain list key 'rows'")
    return rows


def combine_role_rows(
    trait_axis_id: str,
    run_id: str,
    scalar_payload: dict[str, Any],
    gate_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    gate_by_role = gate_rows_by_role(gate_payload)
    combined = []
    for scalar_row in scalar_rows(scalar_payload):
        role_id = str(scalar_row["role_id"])
        gate_row = gate_by_role.get(role_id, {})
        combined.append(
            {
                "trait_axis_id": trait_axis_id,
                "run_id": run_id,
                "role_id": role_id,
                "layer": scalar_row.get("layer"),
                "ruler_method": scalar_row.get("ruler_method"),
                "offset_scalar": scalar_row.get("offset_scalar"),
                "positive_shift_scalar": scalar_row.get("positive_shift_scalar"),
                "negative_shift_scalar": scalar_row.get("negative_shift_scalar"),
                "axis_projection_scalar": scalar_row.get("axis_projection_scalar"),
                "axis_alignment_cosine": scalar_row.get("axis_alignment_cosine"),
                "mention_shift_scalar": scalar_row.get("mention_shift_scalar"),
                "mention_to_elicitation_ratio": gate_row.get("mention_to_elicitation_ratio"),
                "positive_direction_status": gate_row.get("positive_direction_status"),
                "negative_direction_status": gate_row.get("negative_direction_status"),
                "mention_control_status": gate_row.get("mention_control_status"),
                "axis_alignment_status": gate_row.get("axis_alignment_status"),
                "gate_overall": gate_row.get("overall"),
            }
        )
    return combined


def summarize_trait_rows(
    trait_axis_id: str,
    run_id: str,
    role_rows: list[dict[str, Any]],
    gate_payload: dict[str, Any],
) -> dict[str, Any]:
    gate_summary = gate_payload.get("summary", {})
    counts = gate_summary.get("counts", {}) if isinstance(gate_summary, dict) else {}
    alignments = numeric_values(role_rows, "axis_alignment_cosine")
    mention_ratios = numeric_values(role_rows, "mention_to_elicitation_ratio")
    return {
        "trait_axis_id": trait_axis_id,
        "run_id": run_id,
        "roles": len(role_rows),
        "gate_overall": gate_summary.get("overall") if isinstance(gate_summary, dict) else None,
        "gate_pass": counts.get("pass", 0),
        "gate_warn": counts.get("warn", 0),
        "gate_fail": counts.get("fail", 0),
        "positive_shift_mean": mean(numeric_values(role_rows, "positive_shift_scalar")),
        "negative_shift_mean": mean(numeric_values(role_rows, "negative_shift_scalar")),
        "axis_alignment_mean": mean(alignments),
        "axis_alignment_min": min(alignments) if alignments else None,
        "mention_to_elicitation_ratio_max": max(mention_ratios) if mention_ratios else None,
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


def write_markdown_report(
    path: Path,
    base_root: Path,
    role_scope: str,
    trait_rows: list[dict[str, Any]],
    role_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    role_fields = [
        "trait_axis_id",
        "role_id",
        "positive_shift_scalar",
        "negative_shift_scalar",
        "axis_alignment_cosine",
        "mention_to_elicitation_ratio",
        "gate_overall",
    ]
    text = (
        "# Five-Trait Pilot Summary\n\n"
        f"Created at: {datetime.now(timezone.utc).isoformat()}\n\n"
        f"Base root: `{base_root}`\n\n"
        f"Role scope: `{role_scope}`\n\n"
        "## Trait Summary\n\n"
        + markdown_table(trait_rows, TRAIT_ROW_FIELDS)
        + "\n## Role Rows\n\n"
        + markdown_table(role_rows, role_fields)
    )
    path.write_text(text, encoding="utf-8")


def build_summary(
    base_root: Path,
    traits: list[str],
    role_scope: str,
    explicit_run_ids: dict[str, str],
) -> dict[str, Any]:
    trait_rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    inputs: dict[str, Any] = {}

    for trait_axis_id in traits:
        run_id = resolve_run_id(base_root, trait_axis_id, role_scope, explicit_run_ids)
        paths = artifact_paths(base_root, trait_axis_id, role_scope, run_id)
        missing = {
            name: str(path)
            for name, path in paths.items()
            if name != "analysis_root" and not path.exists()
        }
        if missing:
            raise FileNotFoundError(f"missing artifacts for {trait_axis_id}: {missing}")

        scalar_payload = read_json(paths["scalar_decomposition"])
        gate_payload = read_json(paths["salience_gate"])
        combined_rows = combine_role_rows(trait_axis_id, run_id, scalar_payload, gate_payload)
        role_rows.extend(combined_rows)
        trait_rows.append(summarize_trait_rows(trait_axis_id, run_id, combined_rows, gate_payload))
        inputs[trait_axis_id] = {
            "run_id": run_id,
            "scalar_decomposition": str(paths["scalar_decomposition"]),
            "salience_gate": str(paths["salience_gate"]),
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
    json_path = output_dir / "multi_trait_summary.json"
    trait_csv_path = output_dir / "trait_summary.csv"
    role_csv_path = output_dir / "role_scalar_gate_summary.csv"
    markdown_path = output_dir / "multi_trait_summary.md"

    trait_rows = summary["trait_summary"]
    role_rows = summary["role_rows"]
    write_json(json_path, summary)
    write_csv(trait_csv_path, trait_rows, TRAIT_ROW_FIELDS)
    write_csv(role_csv_path, role_rows, ROLE_ROW_FIELDS)
    write_markdown_report(
        markdown_path,
        base_root=Path(summary["base_root"]),
        role_scope=str(summary["role_scope"]),
        trait_rows=trait_rows,
        role_rows=role_rows,
    )
    return {
        "summary_json": str(json_path),
        "trait_summary_csv": str(trait_csv_path),
        "role_summary_csv": str(role_csv_path),
        "markdown_report": str(markdown_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize scalar and salience-gate outputs across traits.")
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--traits", nargs="+", default=DEFAULT_TRAITS)
    parser.add_argument("--role-scope", default="primary_roles")
    parser.add_argument(
        "--run-id",
        action="append",
        default=[],
        help="Optional trait-specific run id override, formatted as trait_axis_id=run_id.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    run_ids = parse_run_id_args(args.run_id)
    summary = build_summary(
        base_root=args.base_root,
        traits=list(args.traits),
        role_scope=args.role_scope,
        explicit_run_ids=run_ids,
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
