from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCORE_FIELDS = [
    "positive_pole_score",
    "negative_pole_score",
    "role_adherence_score",
    "coherence_score",
    "prompt_following_score",
    "trait_word_discussion_score",
]

CONDITIONS = [
    "present_positive",
    "present_negative",
    "present_neutral",
    "mention_without_possession",
]

ROLE_ROW_FIELDS = [
    "trait_axis_id",
    "role_id",
    "records",
    "present_positive_records",
    "present_negative_records",
    "present_neutral_records",
    "mention_without_possession_records",
    "positive_pole_neutral_mean",
    "positive_pole_positive_mean",
    "positive_behavior_shift",
    "positive_behavior_matched_shift",
    "negative_pole_neutral_mean",
    "negative_pole_negative_mean",
    "negative_behavior_shift",
    "negative_behavior_matched_shift",
    "positive_pole_mention_mean",
    "negative_pole_mention_mean",
    "mention_positive_behavior_shift",
    "mention_negative_behavior_shift",
    "trait_word_discussion_neutral_mean",
    "trait_word_discussion_mention_mean",
    "mention_trait_word_discussion_shift",
    "role_adherence_mean",
    "coherence_mean",
    "prompt_following_mean",
    "low_quality_flag",
]


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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{field: row.get(field) for field in fieldnames} for row in rows])


def make_run_dirs(run_root: Path) -> dict[str, Path]:
    paths = {
        "results": run_root / "results",
        "meta": run_root / "meta",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def validate_judgment_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing_required: list[str] = []
    invalid_scores: list[str] = []
    required = ["prompt_id", "trait_axis_id", "role_id", "condition", *SCORE_FIELDS]
    for index, row in enumerate(rows):
        row_id = str(row.get("prompt_id", f"row_{index}"))
        for field in required:
            if field not in row:
                missing_required.append(f"{row_id}:{field}")
        for field in SCORE_FIELDS:
            value = row.get(field)
            if not isinstance(value, int) or not 1 <= value <= 5:
                invalid_scores.append(f"{row_id}:{field}={value!r}")
    return {
        "passed": not missing_required and not invalid_scores,
        "missing_required": missing_required,
        "invalid_scores": invalid_scores,
        "rows": len(rows),
    }


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def score_mean(rows: list[dict[str, Any]], field: str) -> float | None:
    return mean([float(row[field]) for row in rows if row.get(field) is not None])


def grouped_by(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key)) for key in keys)].append(row)
    return dict(grouped)


def rows_by_prompt_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["prompt_id"]): row for row in rows}


def condition_rows(rows: list[dict[str, Any]], condition: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("condition") == condition]


def shift(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def matched_shift_for_condition(
    rows: list[dict[str, Any]],
    all_rows_by_prompt_id: dict[str, dict[str, Any]],
    condition: str,
    score_field: str,
) -> float | None:
    diffs: list[float] = []
    for row in condition_rows(rows, condition):
        metadata = row.get("metadata") or {}
        neutral_id = metadata.get("matched_neutral_id")
        if not neutral_id:
            continue
        neutral_row = all_rows_by_prompt_id.get(str(neutral_id))
        if neutral_row is None:
            continue
        diffs.append(float(row[score_field]) - float(neutral_row[score_field]))
    return mean(diffs)


def summarize_role_rows(
    rows: list[dict[str, Any]],
    all_rows_by_prompt_id: dict[str, dict[str, Any]],
    min_quality_mean: float,
) -> dict[str, Any]:
    trait_axis_id = str(rows[0]["trait_axis_id"])
    role_id = str(rows[0]["role_id"])
    by_condition = {condition: condition_rows(rows, condition) for condition in CONDITIONS}

    positive_neutral = score_mean(by_condition["present_neutral"], "positive_pole_score")
    positive_positive = score_mean(by_condition["present_positive"], "positive_pole_score")
    negative_neutral = score_mean(by_condition["present_neutral"], "negative_pole_score")
    negative_negative = score_mean(by_condition["present_negative"], "negative_pole_score")
    positive_mention = score_mean(by_condition["mention_without_possession"], "positive_pole_score")
    negative_mention = score_mean(by_condition["mention_without_possession"], "negative_pole_score")
    trait_word_neutral = score_mean(by_condition["present_neutral"], "trait_word_discussion_score")
    trait_word_mention = score_mean(by_condition["mention_without_possession"], "trait_word_discussion_score")

    role_adherence_mean = score_mean(rows, "role_adherence_score")
    coherence_mean = score_mean(rows, "coherence_score")
    prompt_following_mean = score_mean(rows, "prompt_following_score")
    quality_values = [
        value
        for value in [role_adherence_mean, coherence_mean, prompt_following_mean]
        if value is not None
    ]
    low_quality_flag = bool(quality_values and min(quality_values) < min_quality_mean)

    return {
        "trait_axis_id": trait_axis_id,
        "role_id": role_id,
        "records": len(rows),
        "present_positive_records": len(by_condition["present_positive"]),
        "present_negative_records": len(by_condition["present_negative"]),
        "present_neutral_records": len(by_condition["present_neutral"]),
        "mention_without_possession_records": len(by_condition["mention_without_possession"]),
        "positive_pole_neutral_mean": positive_neutral,
        "positive_pole_positive_mean": positive_positive,
        "positive_behavior_shift": shift(positive_positive, positive_neutral),
        "positive_behavior_matched_shift": matched_shift_for_condition(
            rows, all_rows_by_prompt_id, "present_positive", "positive_pole_score"
        ),
        "negative_pole_neutral_mean": negative_neutral,
        "negative_pole_negative_mean": negative_negative,
        "negative_behavior_shift": shift(negative_negative, negative_neutral),
        "negative_behavior_matched_shift": matched_shift_for_condition(
            rows, all_rows_by_prompt_id, "present_negative", "negative_pole_score"
        ),
        "positive_pole_mention_mean": positive_mention,
        "negative_pole_mention_mean": negative_mention,
        "mention_positive_behavior_shift": shift(positive_mention, positive_neutral),
        "mention_negative_behavior_shift": shift(negative_mention, negative_neutral),
        "trait_word_discussion_neutral_mean": trait_word_neutral,
        "trait_word_discussion_mention_mean": trait_word_mention,
        "mention_trait_word_discussion_shift": shift(trait_word_mention, trait_word_neutral),
        "role_adherence_mean": role_adherence_mean,
        "coherence_mean": coherence_mean,
        "prompt_following_mean": prompt_following_mean,
        "low_quality_flag": low_quality_flag,
    }


def summarize_metric(rows: list[dict[str, Any]], key: str) -> dict[str, float] | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def build_behavior_metrics(
    judgment_rows: list[dict[str, Any]],
    min_quality_mean: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validation = validate_judgment_rows(judgment_rows)
    if not validation["passed"]:
        raise ValueError(f"judgment validation failed: {validation}")

    by_prompt_id = rows_by_prompt_id(judgment_rows)
    grouped = grouped_by(judgment_rows, ["trait_axis_id", "role_id"])
    role_rows = [
        summarize_role_rows(rows, by_prompt_id, min_quality_mean)
        for _, rows in sorted(grouped.items())
    ]
    summary = {
        "rows": len(role_rows),
        "judgment_rows": len(judgment_rows),
        "traits": sorted({row["trait_axis_id"] for row in role_rows}),
        "roles": sorted({row["role_id"] for row in role_rows}),
        "low_quality_roles": [
            {"trait_axis_id": row["trait_axis_id"], "role_id": row["role_id"]}
            for row in role_rows
            if row["low_quality_flag"]
        ],
        "positive_behavior_shift": summarize_metric(role_rows, "positive_behavior_shift"),
        "negative_behavior_shift": summarize_metric(role_rows, "negative_behavior_shift"),
        "positive_behavior_matched_shift": summarize_metric(role_rows, "positive_behavior_matched_shift"),
        "negative_behavior_matched_shift": summarize_metric(role_rows, "negative_behavior_matched_shift"),
        "role_adherence_mean": summarize_metric(role_rows, "role_adherence_mean"),
        "coherence_mean": summarize_metric(role_rows, "coherence_mean"),
        "prompt_following_mean": summarize_metric(role_rows, "prompt_following_mean"),
    }
    return role_rows, summary


def write_behavior_artifacts(
    run_root: Path,
    judgments_jsonl: Path,
    role_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    validation: dict[str, Any],
    min_quality_mean: float,
) -> dict[str, str]:
    paths = make_run_dirs(run_root)
    json_path = paths["results"] / "behavior_metrics.json"
    csv_path = paths["results"] / "behavior_metrics.csv"
    manifest_path = paths["meta"] / "behavior_metrics_manifest.json"
    created_at = datetime.now(timezone.utc).isoformat()

    write_json(
        json_path,
        {
            "schema_version": "0.1",
            "created_at_utc": created_at,
            "judgments_jsonl": str(judgments_jsonl),
            "summary": summary,
            "rows": role_rows,
        },
    )
    write_csv(csv_path, role_rows, ROLE_ROW_FIELDS)
    write_json(
        manifest_path,
        {
            "schema_version": "0.1",
            "builder": "BehaviorMetricsBuilder",
            "created_at_utc": created_at,
            "judgments_jsonl": str(judgments_jsonl),
            "run_root": str(run_root),
            "min_quality_mean": min_quality_mean,
            "validation": validation,
            "summary": summary,
            "artifacts": {
                "behavior_metrics_json": str(json_path),
                "behavior_metrics_csv": str(csv_path),
            },
        },
    )
    return {
        "behavior_metrics_json": str(json_path),
        "behavior_metrics_csv": str(csv_path),
        "behavior_metrics_manifest": str(manifest_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build behavior metrics from trait-judge JSONL rows.")
    parser.add_argument("--judgments-jsonl", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-quality-mean", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    judgment_rows = load_jsonl(args.judgments_jsonl, limit=args.limit)
    validation = validate_judgment_rows(judgment_rows)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "judgments_jsonl": str(args.judgments_jsonl),
                    "run_root": str(args.run_root),
                    "validation": validation,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if validation["passed"] else 1
    if not validation["passed"]:
        print(json.dumps({"error": "judgment validation failed", "validation": validation}, indent=2))
        return 2

    role_rows, summary = build_behavior_metrics(judgment_rows, args.min_quality_mean)
    artifacts = write_behavior_artifacts(
        run_root=args.run_root,
        judgments_jsonl=args.judgments_jsonl,
        role_rows=role_rows,
        summary=summary,
        validation=validation,
        min_quality_mean=args.min_quality_mean,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "rows": len(role_rows),
                "judgment_rows": len(judgment_rows),
                "artifacts": artifacts,
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
