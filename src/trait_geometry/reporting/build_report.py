from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SCALAR_GATE_SUMMARY = Path("artifacts/reports/five_trait_pilot_v0/multi_trait_summary.json")
DEFAULT_GEOMETRY_SUMMARY = Path("artifacts/reports/five_trait_geometry_v0/geometry_summary.json")
DEFAULT_SCALAR_BEHAVIOR_SUMMARY = Path(
    "artifacts/reports/five_trait_behavior_v0/scalar_behavior_summary.json"
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_section(path: Path | None, required: bool, label: str) -> dict[str, Any]:
    if path is None:
        return {
            "label": label,
            "status": "skipped",
            "path": None,
            "payload": None,
            "message": "No input path was provided.",
        }
    if not path.exists():
        if required:
            raise FileNotFoundError(f"required {label} summary is missing: {path}")
        return {
            "label": label,
            "status": "missing",
            "path": str(path),
            "payload": None,
            "message": f"Summary not found at {path}.",
        }
    return {
        "label": label,
        "status": "available",
        "path": str(path),
        "payload": read_json(path),
        "message": None,
    }


def number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return str(value)


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    header = "| " + " | ".join(fields) + " |"
    divider = "| " + " | ".join("---" for _ in fields) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(fmt(row.get(field)) for field in fields) + " |")
    return "\n".join([header, divider, *body]) + "\n"


def top_n_by_abs(rows: list[dict[str, Any]], key: str, n: int = 8) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: abs(number(row.get(key)) or 0.0),
        reverse=True,
    )[:n]


def lowest_n(rows: list[dict[str, Any]], key: str, n: int = 8) -> list[dict[str, Any]]:
    valid = [row for row in rows if number(row.get(key)) is not None]
    return sorted(valid, key=lambda row: number(row.get(key)) or 0.0)[:n]


def highest_n(rows: list[dict[str, Any]], key: str, n: int = 8) -> list[dict[str, Any]]:
    valid = [row for row in rows if number(row.get(key)) is not None]
    return sorted(valid, key=lambda row: number(row.get(key)) or 0.0, reverse=True)[:n]


def scalar_gate_digest(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {"available": False}
    trait_rows = list(payload.get("trait_summary") or [])
    role_rows = list(payload.get("role_rows") or [])
    failed_traits = [
        row["trait_axis_id"]
        for row in trait_rows
        if str(row.get("gate_overall")) == "fail"
    ]
    warned_traits = [
        row["trait_axis_id"]
        for row in trait_rows
        if str(row.get("gate_overall")) == "warn"
    ]
    return {
        "available": True,
        "traits": len(trait_rows),
        "role_rows": len(role_rows),
        "failed_traits": failed_traits,
        "warned_traits": warned_traits,
        "trait_summary": trait_rows,
        "largest_positive_shifts": highest_n(role_rows, "positive_shift_scalar"),
        "largest_negative_shifts": lowest_n(role_rows, "negative_shift_scalar"),
        "lowest_axis_alignment": lowest_n(role_rows, "axis_alignment_cosine"),
        "highest_mention_ratio": highest_n(role_rows, "mention_to_elicitation_ratio"),
    }


def geometry_digest(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {"available": False}
    summary = dict(payload.get("summary") or {})
    role_pair_rows = list(payload.get("role_pair_cosines") or [])
    ruler_rows = list(payload.get("ruler_cosines") or [])
    pca_rows = list(payload.get("pca_summary") or [])
    return {
        "available": True,
        "summary": summary,
        "lowest_role_pair_axis_cosines": lowest_n(
            [row for row in role_pair_rows if row.get("vector_type") == "axis_vector"],
            "cosine",
        ),
        "highest_ruler_cosines": highest_n(ruler_rows, "cosine"),
        "lowest_ruler_cosines": lowest_n(ruler_rows, "cosine"),
        "pca_summary": pca_rows,
    }


def scalar_behavior_digest(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {"available": False}
    trait_rows = list(payload.get("trait_summary") or [])
    role_rows = list(payload.get("role_rows") or [])
    return {
        "available": True,
        "traits": len(trait_rows),
        "role_rows": len(role_rows),
        "trait_summary": trait_rows,
        "largest_positive_behavior_shifts": highest_n(
            role_rows, "positive_behavior_matched_shift"
        ),
        "largest_negative_behavior_shifts": lowest_n(
            role_rows, "negative_behavior_matched_shift"
        ),
        "lowest_quality_rows": lowest_n(role_rows, "coherence_mean"),
    }


def build_interpretation_flags(
    scalar_gate: dict[str, Any],
    geometry: dict[str, Any],
    scalar_behavior: dict[str, Any],
) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    if scalar_gate.get("available"):
        if scalar_gate.get("failed_traits"):
            flags.append(
                {
                    "level": "warning",
                    "topic": "salience_gate",
                    "message": "At least one trait failed the scalar salience gate; inspect mention controls and axis alignment before treating scalar shifts as reliable.",
                }
            )
        if scalar_gate.get("warned_traits"):
            flags.append(
                {
                    "level": "note",
                    "topic": "salience_gate",
                    "message": "Some traits produced salience warnings; these should be interpreted before steering or probe comparisons.",
                }
            )
    if geometry.get("available"):
        summary = geometry.get("summary") or {}
        role_pair_mean = number(summary.get("role_pair_cosine_axis_mean"))
        if role_pair_mean is not None and role_pair_mean < 0.3:
            flags.append(
                {
                    "level": "warning",
                    "topic": "geometry",
                    "message": "Mean role-pair axis cosine is low, suggesting fragmented role-specific trait geometry.",
                }
            )
        ruler_mean = number(summary.get("ruler_cosine_mean"))
        if ruler_mean is not None and ruler_mean > 0.6:
            flags.append(
                {
                    "level": "note",
                    "topic": "geometry",
                    "message": "Mean cross-trait ruler cosine is high, suggesting some trait axes may share a broad direction.",
                }
            )
    if not scalar_behavior.get("available"):
        flags.append(
            {
                "level": "pending",
                "topic": "behavior",
                "message": "Behavior metrics are not present yet, so activation findings are not behavior-validated in this report.",
            }
        )
    return flags


def build_report_payload(
    scalar_gate_path: Path | None,
    geometry_path: Path | None,
    scalar_behavior_path: Path | None,
    strict: bool,
) -> dict[str, Any]:
    sections = {
        "scalar_gate": load_section(
            scalar_gate_path, required=strict, label="scalar_gate"
        ),
        "geometry": load_section(geometry_path, required=strict, label="geometry"),
        "scalar_behavior": load_section(
            scalar_behavior_path, required=strict, label="scalar_behavior"
        ),
    }
    scalar_gate = scalar_gate_digest(sections["scalar_gate"]["payload"])
    geometry = geometry_digest(sections["geometry"]["payload"])
    scalar_behavior = scalar_behavior_digest(sections["scalar_behavior"]["payload"])
    return {
        "schema_version": "0.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "strict": strict,
        "inputs": {
            name: {
                "status": section["status"],
                "path": section["path"],
                "message": section["message"],
            }
            for name, section in sections.items()
        },
        "digests": {
            "scalar_gate": scalar_gate,
            "geometry": geometry,
            "scalar_behavior": scalar_behavior,
        },
        "interpretation_flags": build_interpretation_flags(
            scalar_gate=scalar_gate,
            geometry=geometry,
            scalar_behavior=scalar_behavior,
        ),
    }


def section_status_lines(payload: dict[str, Any]) -> str:
    rows = [
        {"section": name, **info}
        for name, info in payload["inputs"].items()
    ]
    return markdown_table(rows, ["section", "status", "path", "message"])


def write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scalar_gate = payload["digests"]["scalar_gate"]
    geometry = payload["digests"]["geometry"]
    scalar_behavior = payload["digests"]["scalar_behavior"]

    parts = [
        "# Trait Geometry Pilot Report",
        "",
        f"Created at: {payload['created_at_utc']}",
        "",
        "## Artifact Status",
        "",
        section_status_lines(payload),
        "",
        "## Interpretation Flags",
        "",
        markdown_table(payload["interpretation_flags"], ["level", "topic", "message"]),
    ]

    if scalar_gate.get("available"):
        parts.extend(
            [
                "",
                "## Scalar And Salience",
                "",
                markdown_table(
                    scalar_gate["trait_summary"],
                    [
                        "trait_axis_id",
                        "gate_overall",
                        "positive_shift_mean",
                        "negative_shift_mean",
                        "axis_alignment_mean",
                        "mention_to_elicitation_ratio_max",
                    ],
                ),
                "",
                "### Lowest Axis Alignment Rows",
                "",
                markdown_table(
                    scalar_gate["lowest_axis_alignment"],
                    [
                        "trait_axis_id",
                        "role_id",
                        "axis_alignment_cosine",
                        "positive_shift_scalar",
                        "negative_shift_scalar",
                        "gate_overall",
                    ],
                ),
            ]
        )

    if geometry.get("available"):
        parts.extend(
            [
                "",
                "## Geometry",
                "",
                markdown_table([geometry["summary"]], list(geometry["summary"].keys())),
                "",
                "### Lowest Role-Pair Axis Cosines",
                "",
                markdown_table(
                    geometry["lowest_role_pair_axis_cosines"],
                    ["trait_axis_id", "role_a", "role_b", "vector_type", "cosine"],
                ),
                "",
                "### Highest Cross-Trait Ruler Cosines",
                "",
                markdown_table(
                    geometry["highest_ruler_cosines"],
                    ["trait_a", "trait_b", "cosine"],
                ),
                "",
                "### PCA Summary",
                "",
                markdown_table(
                    geometry["pca_summary"],
                    [
                        "scope",
                        "trait_axis_id",
                        "vector_type",
                        "rows",
                        "pc1_explained_variance",
                        "pcs_for_90pct",
                    ],
                ),
            ]
        )

    if scalar_behavior.get("available"):
        parts.extend(
            [
                "",
                "## Scalar-Behavior",
                "",
                markdown_table(
                    scalar_behavior["trait_summary"],
                    [
                        "trait_axis_id",
                        "gate_overall",
                        "positive_shift_scalar_mean",
                        "positive_behavior_matched_shift_mean",
                        "negative_shift_scalar_mean",
                        "negative_behavior_matched_shift_mean",
                        "axis_alignment_mean",
                    ],
                ),
                "",
                "### Largest Positive Behavior Shifts",
                "",
                markdown_table(
                    scalar_behavior["largest_positive_behavior_shifts"],
                    [
                        "trait_axis_id",
                        "role_id",
                        "positive_shift_scalar",
                        "positive_behavior_matched_shift",
                        "gate_overall",
                    ],
                ),
            ]
        )

    path.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")


def write_report_artifacts(output_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_json = output_dir / "integrated_report.json"
    report_md = output_dir / "integrated_report.md"
    manifest = output_dir / "integrated_report_manifest.json"
    write_json(report_json, payload)
    write_markdown_report(report_md, payload)
    write_json(
        manifest,
        {
            "schema_version": "0.1",
            "builder": "ReportBuilder",
            "created_at_utc": payload["created_at_utc"],
            "output_dir": str(output_dir),
            "inputs": payload["inputs"],
            "artifacts": {
                "report_json": str(report_json),
                "report_markdown": str(report_md),
            },
        },
    )
    return {
        "report_json": str(report_json),
        "report_markdown": str(report_md),
        "manifest": str(manifest),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an integrated pilot report from summary artifacts.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scalar-gate-summary", type=Path, default=DEFAULT_SCALAR_GATE_SUMMARY)
    parser.add_argument("--geometry-summary", type=Path, default=DEFAULT_GEOMETRY_SUMMARY)
    parser.add_argument("--scalar-behavior-summary", type=Path, default=DEFAULT_SCALAR_BEHAVIOR_SUMMARY)
    parser.add_argument("--no-scalar-gate", action="store_true")
    parser.add_argument("--no-geometry", action="store_true")
    parser.add_argument("--no-scalar-behavior", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Fail if any enabled input summary is missing.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    payload = build_report_payload(
        scalar_gate_path=None if args.no_scalar_gate else args.scalar_gate_summary,
        geometry_path=None if args.no_geometry else args.geometry_summary,
        scalar_behavior_path=None if args.no_scalar_behavior else args.scalar_behavior_summary,
        strict=args.strict,
    )
    artifacts = write_report_artifacts(args.output_dir, payload)
    available = [
        name
        for name, section in payload["inputs"].items()
        if section["status"] == "available"
    ]
    missing = [
        name
        for name, section in payload["inputs"].items()
        if section["status"] == "missing"
    ]
    print(
        json.dumps(
            {
                "status": "completed",
                "available_sections": available,
                "missing_sections": missing,
                "artifacts": artifacts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
