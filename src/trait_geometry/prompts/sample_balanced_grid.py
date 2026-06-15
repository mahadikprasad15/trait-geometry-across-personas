from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CONDITIONS = (
    "present_positive",
    "present_negative",
    "present_neutral",
    "mention_without_possession",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def count_field(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record[field]) for record in records).items()))


def complete_scenario_records(
    records: list[dict[str, Any]],
    role_id: str,
    variant_id: str,
    conditions: tuple[str, ...],
    scenario_rank: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        if record["role_id"] != role_id:
            continue
        if record["role_instruction_variant_id"] != variant_id:
            continue
        condition = str(record["condition"])
        if condition not in conditions:
            continue
        grouped[str(record["scenario_id"])][condition] = record

    complete = [
        scenario_id
        for scenario_id, condition_map in grouped.items()
        if all(condition in condition_map for condition in conditions)
    ]
    complete = sorted(complete)
    if len(complete) <= scenario_rank:
        raise ValueError(
            f"role={role_id!r} variant={variant_id!r} has only {len(complete)} "
            f"complete scenarios; cannot select scenario_rank={scenario_rank}"
        )

    scenario_id = complete[scenario_rank]
    return [grouped[scenario_id][condition] for condition in conditions]


def validate_balanced_records(records: list[dict[str, Any]], expected_count: int) -> dict[str, Any]:
    ids = [record["prompt_id"] for record in records]
    duplicate_ids = sorted(prompt_id for prompt_id, count in Counter(ids).items() if count > 1)
    id_set = set(ids)
    missing_links = []
    for record in records:
        for field in ["matched_neutral_id", "matched_positive_id", "matched_negative_id"]:
            linked_id = record.get(field)
            if linked_id is not None and linked_id not in id_set:
                missing_links.append({"prompt_id": record["prompt_id"], "field": field, "missing": linked_id})
    return {
        "passed": not duplicate_ids and not missing_links and len(records) == expected_count,
        "record_count": len(records),
        "expected_count": expected_count,
        "duplicate_prompt_ids": duplicate_ids,
        "missing_matched_links": missing_links,
    }


def build_balanced_records(
    input_records: list[dict[str, Any]],
    roles: list[str],
    variant_id: str,
    conditions: tuple[str, ...],
    scenario_rank: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for role_id in roles:
        selected.extend(
            complete_scenario_records(
                records=input_records,
                role_id=role_id,
                variant_id=variant_id,
                conditions=conditions,
                scenario_rank=scenario_rank,
            )
        )
    return selected


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample a small balanced prompt grid from a larger grid.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--roles", nargs="+", required=True)
    parser.add_argument("--variant", default="iv01")
    parser.add_argument("--scenario-rank", type=int, default=0)
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    conditions = tuple(str(condition) for condition in args.conditions)
    input_records = load_jsonl(args.input_jsonl)
    selected = build_balanced_records(
        input_records=input_records,
        roles=args.roles,
        variant_id=args.variant,
        conditions=conditions,
        scenario_rank=args.scenario_rank,
    )
    validation = validate_balanced_records(
        records=selected,
        expected_count=len(args.roles) * len(conditions),
    )
    if not validation["passed"]:
        raise ValueError(f"balanced sample validation failed: {validation}")

    write_jsonl(args.output_jsonl, selected)
    write_json(
        args.manifest,
        {
            "schema_version": "0.1",
            "builder": "BalancedPromptGridSampler",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_jsonl": str(args.input_jsonl),
            "output_jsonl": str(args.output_jsonl),
            "roles": args.roles,
            "variant": args.variant,
            "scenario_rank": args.scenario_rank,
            "conditions": list(conditions),
            "counts": {
                "records": len(selected),
                "role_id": count_field(selected, "role_id"),
                "condition": count_field(selected, "condition"),
                "role_instruction_variant_id": count_field(selected, "role_instruction_variant_id"),
            },
            "validation": validation,
        },
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "output_jsonl": str(args.output_jsonl),
                "manifest": str(args.manifest),
                "records": len(selected),
                "validation": validation,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
