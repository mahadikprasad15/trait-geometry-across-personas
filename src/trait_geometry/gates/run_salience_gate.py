from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PASS = "pass"
WARN = "warn"
FAIL = "fail"
SKIP = "skip"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_scalar_decomposition(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    if "rows" not in payload or not isinstance(payload["rows"], list):
        raise ValueError(f"{path} must contain a list at key 'rows'")
    return payload


def check_input_paths(paths: dict[str, Path]) -> dict[str, Any]:
    missing = {name: str(path) for name, path in paths.items() if not path.exists()}
    return {"passed": not missing, "missing": missing}


def safe_abs_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if abs(denominator) == 0.0:
        return None
    return abs(numerator) / abs(denominator)


def bool_status(value: bool | None, skipped: bool = False) -> str:
    if skipped:
        return SKIP
    if value is None:
        return WARN
    return PASS if value else FAIL


def row_overall_status(check_statuses: list[str]) -> str:
    active_statuses = [status for status in check_statuses if status != SKIP]
    if FAIL in active_statuses:
        return FAIL
    if WARN in active_statuses:
        return WARN
    return PASS


def evaluate_role_row(
    row: dict[str, Any],
    min_axis_alignment: float,
    max_mention_to_shift_ratio: float,
    require_positive_shift_sign: bool,
    require_negative_shift_sign: bool,
) -> dict[str, Any]:
    positive_shift = row.get("positive_shift_scalar")
    negative_shift = row.get("negative_shift_scalar")
    mention_shift = row.get("mention_shift_scalar")
    axis_alignment = row.get("axis_alignment_cosine")

    positive_direction_pass = None
    if require_positive_shift_sign:
        positive_direction_pass = positive_shift is not None and positive_shift > 0.0

    negative_direction_pass = None
    if require_negative_shift_sign:
        negative_direction_pass = negative_shift is not None and negative_shift < 0.0

    positive_mention_ratio = safe_abs_ratio(mention_shift, positive_shift)
    negative_mention_ratio = safe_abs_ratio(mention_shift, negative_shift)
    max_observed_mention_ratio_values = [
        ratio for ratio in [positive_mention_ratio, negative_mention_ratio] if ratio is not None
    ]
    max_observed_mention_ratio = (
        max(max_observed_mention_ratio_values) if max_observed_mention_ratio_values else None
    )

    mention_control_pass = None
    if max_observed_mention_ratio is not None:
        mention_control_pass = max_observed_mention_ratio <= max_mention_to_shift_ratio

    axis_alignment_pass = None
    if axis_alignment is not None:
        axis_alignment_pass = axis_alignment >= min_axis_alignment

    statuses = [
        bool_status(positive_direction_pass, skipped=not require_positive_shift_sign),
        bool_status(negative_direction_pass, skipped=not require_negative_shift_sign),
        bool_status(mention_control_pass),
        bool_status(axis_alignment_pass),
    ]
    overall = row_overall_status(statuses)

    return {
        "trait_axis_id": row.get("trait_axis_id"),
        "role_id": row.get("role_id"),
        "layer": row.get("layer"),
        "ruler_method": row.get("ruler_method"),
        "positive_shift_scalar": positive_shift,
        "negative_shift_scalar": negative_shift,
        "mention_shift_scalar": mention_shift,
        "axis_alignment_cosine": axis_alignment,
        "positive_mention_ratio": positive_mention_ratio,
        "negative_mention_ratio": negative_mention_ratio,
        "max_observed_mention_ratio": max_observed_mention_ratio,
        "positive_direction_pass": positive_direction_pass,
        "negative_direction_pass": negative_direction_pass,
        "mention_control_pass": mention_control_pass,
        "axis_alignment_pass": axis_alignment_pass,
        "positive_direction_status": bool_status(
            positive_direction_pass, skipped=not require_positive_shift_sign
        ),
        "negative_direction_status": bool_status(
            negative_direction_pass, skipped=not require_negative_shift_sign
        ),
        "mention_control_status": bool_status(mention_control_pass),
        "axis_alignment_status": bool_status(axis_alignment_pass),
        "overall": overall,
    }


def evaluate_rows(
    scalar_rows: list[dict[str, Any]],
    min_axis_alignment: float,
    max_mention_to_shift_ratio: float,
    require_positive_shift_sign: bool,
    require_negative_shift_sign: bool,
) -> list[dict[str, Any]]:
    return [
        evaluate_role_row(
            row=row,
            min_axis_alignment=min_axis_alignment,
            max_mention_to_shift_ratio=max_mention_to_shift_ratio,
            require_positive_shift_sign=require_positive_shift_sign,
            require_negative_shift_sign=require_negative_shift_sign,
        )
        for row in scalar_rows
    ]


def summarize_gate_rows(rows: list[dict[str, Any]], warn_if_fail_fraction_at_least: float) -> dict[str, Any]:
    counts = {PASS: 0, WARN: 0, FAIL: 0}
    for row in rows:
        counts[str(row["overall"])] += 1
    total = len(rows)
    fail_fraction = counts[FAIL] / total if total else 0.0

    if total == 0:
        overall = FAIL
    elif counts[FAIL] == 0 and counts[WARN] == 0:
        overall = PASS
    elif fail_fraction >= warn_if_fail_fraction_at_least:
        overall = FAIL
    else:
        overall = WARN

    return {
        "overall": overall,
        "rows": total,
        "counts": counts,
        "fail_fraction": fail_fraction,
        "roles_failed": [row["role_id"] for row in rows if row["overall"] == FAIL],
        "roles_warned": [row["role_id"] for row in rows if row["overall"] == WARN],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trait_axis_id",
        "role_id",
        "layer",
        "ruler_method",
        "positive_shift_scalar",
        "negative_shift_scalar",
        "mention_shift_scalar",
        "axis_alignment_cosine",
        "positive_mention_ratio",
        "negative_mention_ratio",
        "max_observed_mention_ratio",
        "positive_direction_pass",
        "negative_direction_pass",
        "mention_control_pass",
        "axis_alignment_pass",
        "positive_direction_status",
        "negative_direction_status",
        "mention_control_status",
        "axis_alignment_status",
        "overall",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_gate_artifacts(
    output_dir: Path,
    scalar_decomposition_path: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "salience_gate.json"
    csv_path = output_dir / "salience_gate.csv"
    manifest_path = output_dir.parent.parent / "meta" / "salience_gate_manifest.json"

    write_json(
        json_path,
        {
            "schema_version": "0.1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "thresholds": thresholds,
            "rows": rows,
        },
    )
    write_csv(csv_path, rows)
    write_json(
        manifest_path,
        {
            "schema_version": "0.1",
            "runner": "SalienceGateRunner",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {"scalar_decomposition": str(scalar_decomposition_path)},
            "output_dir": str(output_dir),
            "summary": summary,
            "thresholds": thresholds,
            "artifacts": {
                "salience_gate_json": str(json_path),
                "salience_gate_csv": str(csv_path),
                "salience_gate_manifest": str(manifest_path),
            },
        },
    )
    return {
        "salience_gate_json": str(json_path),
        "salience_gate_csv": str(csv_path),
        "salience_gate_manifest": str(manifest_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run scalar salience gates for a trait ruler.")
    parser.add_argument("--scalar-decomposition", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-axis-alignment", type=float, default=0.2)
    parser.add_argument("--max-mention-to-shift-ratio", type=float, default=0.5)
    parser.add_argument("--warn-if-fail-fraction-at-least", type=float, default=0.5)
    parser.add_argument("--no-require-positive-shift-sign", action="store_true")
    parser.add_argument("--no-require-negative-shift-sign", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    input_paths = {"scalar_decomposition": args.scalar_decomposition}
    path_status = check_input_paths(input_paths)
    thresholds = {
        "min_axis_alignment": args.min_axis_alignment,
        "max_mention_to_shift_ratio": args.max_mention_to_shift_ratio,
        "warn_if_fail_fraction_at_least": args.warn_if_fail_fraction_at_least,
        "require_positive_shift_sign": not args.no_require_positive_shift_sign,
        "require_negative_shift_sign": not args.no_require_negative_shift_sign,
    }

    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "summary": {
                        "scalar_decomposition": str(args.scalar_decomposition),
                        "output_dir": str(args.output_dir),
                    },
                    "input_paths": path_status,
                    "thresholds": thresholds,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if path_status["passed"] else 1

    if not path_status["passed"]:
        print(json.dumps({"error": "salience gate inputs are missing", "input_paths": path_status}, indent=2))
        return 2

    scalar_payload = load_scalar_decomposition(args.scalar_decomposition)
    rows = evaluate_rows(
        scalar_rows=scalar_payload["rows"],
        min_axis_alignment=args.min_axis_alignment,
        max_mention_to_shift_ratio=args.max_mention_to_shift_ratio,
        require_positive_shift_sign=not args.no_require_positive_shift_sign,
        require_negative_shift_sign=not args.no_require_negative_shift_sign,
    )
    summary = summarize_gate_rows(rows, args.warn_if_fail_fraction_at_least)
    artifacts = write_gate_artifacts(
        output_dir=args.output_dir,
        scalar_decomposition_path=args.scalar_decomposition,
        rows=rows,
        summary=summary,
        thresholds=thresholds,
    )
    print(json.dumps({"status": "completed", "summary": summary, "artifacts": artifacts}, indent=2, sort_keys=True))
    return 0 if summary["overall"] in {PASS, WARN} else 1


if __name__ == "__main__":
    raise SystemExit(main())
