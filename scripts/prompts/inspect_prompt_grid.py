#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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
                raise ValueError(f"invalid JSON on line {line_no}: {exc}") from exc
    return records


def print_counts(records: list[dict[str, Any]]) -> None:
    print("Counts")
    print("------")
    print(f"records: {len(records)}")
    for field in ["trait_axis_id", "role_id", "condition", "role_instruction_variant_id"]:
        counts = Counter(str(record[field]) for record in records)
        print(f"{field}:")
        for key, count in sorted(counts.items()):
            print(f"  {key}: {count}")
    print()


def grouped_sample(
    records: list[dict[str, Any]],
    role: str,
    scenario_id: str | None,
    variant: str,
) -> list[dict[str, Any]]:
    by_role = [record for record in records if record["role_id"] == role]
    if scenario_id is None:
        scenario_id = sorted({record["scenario_id"] for record in by_role})[0]
    sample = [
        record
        for record in by_role
        if record["scenario_id"] == scenario_id
        and record["role_instruction_variant_id"] == variant
    ]
    condition_order = {
        "present_positive": 0,
        "present_negative": 1,
        "present_neutral": 2,
        "mention_without_possession": 3,
        "instruction_positive": 0,
        "instruction_negative": 1,
        "instruction_neutral": 2,
    }
    return sorted(sample, key=lambda record: condition_order.get(record["condition"], 99))


def print_record(record: dict[str, Any]) -> None:
    print(f"[{record['condition']}] {record['prompt_id']}")
    print(f"polarity: {record['polarity']}")
    print(f"matched_neutral_id: {record['matched_neutral_id']}")
    print(f"matched_positive_id: {record['matched_positive_id']}")
    print(f"matched_negative_id: {record['matched_negative_id']}")
    print(f"trait_word_present: {record['trait_word_present']}")
    print("role_instruction:")
    print(f"  {record['role_instruction']}")
    print("prompt_text:")
    print(f"  {record['prompt_text']}")
    print()


def inspect_records(
    records: list[dict[str, Any]],
    roles: list[str] | None,
    scenario_id: str | None,
    variant: str,
) -> None:
    if roles is None:
        roles = sorted({record["role_id"] for record in records})
    for role in roles:
        print("=" * 88)
        print(f"Role: {role}")
        sample = grouped_sample(records, role, scenario_id, variant)
        if not sample:
            print(f"No records found for role={role!r}, scenario={scenario_id!r}, variant={variant!r}")
            continue
        print(f"Scenario: {sample[0]['scenario_id']}")
        topic = sample[0]["source"].get("topic") or sample[0]["source"].get("question_category")
        print(f"Topic: {topic}")
        print("-" * 88)
        for record in sample:
            print_record(record)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect samples from an expanded prompt grid.")
    parser.add_argument("--prompt-jsonl", type=Path, required=True)
    parser.add_argument(
        "--role",
        action="append",
        dest="roles",
        help="Role to inspect. May be passed multiple times. Defaults to all roles.",
    )
    parser.add_argument(
        "--scenario-id",
        help="Scenario id to inspect. Defaults to the first scenario for each role.",
    )
    parser.add_argument(
        "--variant",
        default="iv01",
        help="Role instruction variant id to inspect.",
    )
    parser.add_argument(
        "--no-counts",
        action="store_true",
        help="Skip count summary.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    records = load_jsonl(args.prompt_jsonl)
    if not args.no_counts:
        print_counts(records)
    inspect_records(records, args.roles, args.scenario_id, args.variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
