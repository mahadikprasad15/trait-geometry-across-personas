from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


CONDITION_POLARITY = {
    "present_positive": "positive",
    "present_negative": "negative",
    "present_neutral": "neutral",
    "mention_without_possession": None,
    "instruction_positive": "positive",
    "instruction_negative": "negative",
    "other_positive": "positive",
    "other_negative": "negative",
    "other_neutral": "neutral",
}

SCENARIO_INDUCED_CONDITIONS = {
    "present_positive",
    "present_negative",
    "present_neutral",
    "other_positive",
    "other_negative",
    "other_neutral",
}


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    trait_axis_id: str
    role_id: str
    role_set: str
    role_instruction_variant_id: str
    role_instruction: str
    scenario_id: str
    condition: str
    polarity: str | None
    prompt_text: str
    full_prompt: str
    matched_neutral_id: str | None
    matched_positive_id: str | None
    matched_negative_id: str | None
    trait_word_present: bool
    source: dict[str, Any]
    readout_policy: str
    speaker_spans: dict[str, Any] | None
    safety_notes: list[str]
    validation_tags: list[str]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def variant_id(index: int) -> str:
    return f"iv{index:02d}"


def stable_prompt_id(
    trait_axis_id: str,
    role_id: str,
    scenario_id: str,
    condition: str,
    instruction_variant_id: str,
) -> str:
    return "__".join(
        [trait_axis_id, role_id, scenario_id, condition, instruction_variant_id]
    )


def render_full_prompt(role_instruction: str, prompt_text: str) -> str:
    return (
        "Role instruction:\n"
        f"{normalize_text(role_instruction)}\n\n"
        "Scenario:\n"
        f"{normalize_text(prompt_text)}"
    )


def leakage_pattern(trait_config: dict[str, Any]) -> re.Pattern[str]:
    terms = {
        str(trait_config["trait_axis_id"]),
        str(trait_config["display_name"]),
        str(trait_config["positive_pole"]["id"]),
        str(trait_config["positive_pole"]["label"]),
        str(trait_config["negative_pole"]["id"]),
        str(trait_config["negative_pole"]["label"]),
    }
    terms.update(str(term) for term in trait_config.get("construction", {}).get("lexical_leakage_terms", []))
    # Include common split forms for hyphenated or underscored labels.
    expanded = set()
    for term in terms:
        expanded.add(term)
        expanded.update(re.split(r"[-_\s]+", term))
    escaped = [re.escape(term) for term in expanded if term]
    return re.compile(r"\b(" + "|".join(sorted(escaped, key=len, reverse=True)) + r")\b", re.I)


def trait_word_present(text: str, pattern: re.Pattern[str]) -> bool:
    return pattern.search(text) is not None


def role_set_for(role_id: str, roles_config: dict[str, Any]) -> str:
    try:
        return str(roles_config["roles"][role_id]["set"])
    except KeyError as exc:
        raise ValueError(f"role {role_id!r} is missing from role config") from exc


def selected_instruction_variants(
    role_id: str,
    roles_config: dict[str, Any],
    selection: str,
) -> list[tuple[str, str]]:
    variants = roles_config["roles"][role_id]["instruction_variants"]
    if selection == "all":
        selected = variants
    elif selection == "first":
        selected = variants[:1]
    else:
        raise ValueError(
            f"unsupported role instruction variant selection {selection!r}; "
            "use 'all' or 'first'"
        )
    return [(variant_id(i), str(text)) for i, text in enumerate(selected, start=1)]


def validate_spec_roles(prompt_spec: dict[str, Any], roles_config: dict[str, Any]) -> None:
    role_ids = set(roles_config["roles"])
    missing = [role for role in prompt_spec["expansion"]["roles"] if role not in role_ids]
    if missing:
        raise ValueError(f"prompt spec references unknown roles: {missing}")

    scenario_roles = set(prompt_spec["scenarios"])
    expansion_roles = set(prompt_spec["expansion"]["roles"])
    if scenario_roles != expansion_roles:
        raise ValueError(
            "scenario role keys must match expansion roles: "
            f"scenario_roles={sorted(scenario_roles)} expansion_roles={sorted(expansion_roles)}"
        )


def validate_spec_conditions(prompt_spec: dict[str, Any], allowed_conditions: set[str]) -> None:
    conditions = prompt_spec["expansion"]["conditions"]
    unknown = [condition for condition in conditions if condition not in allowed_conditions]
    if unknown:
        raise ValueError(f"unknown conditions in prompt spec: {unknown}")

    missing: list[tuple[str, str, str]] = []
    for role_id, scenarios in prompt_spec["scenarios"].items():
        for scenario in scenarios:
            scenario_id = scenario.get("scenario_id", "<missing>")
            for condition in conditions:
                if condition not in scenario:
                    missing.append((role_id, scenario_id, condition))
    if missing:
        raise ValueError(f"missing condition text: {missing}")


def matched_id(
    trait_axis_id: str,
    role_id: str,
    scenario_id: str,
    condition: str,
    instruction_variant_id: str,
) -> str:
    return stable_prompt_id(
        trait_axis_id,
        role_id,
        scenario_id,
        condition,
        instruction_variant_id,
    )


def build_prompt_records(
    roles_config: dict[str, Any],
    trait_config: dict[str, Any],
    prompt_schema: dict[str, Any],
    prompt_spec: dict[str, Any],
) -> tuple[list[PromptRecord], dict[str, Any]]:
    validate_spec_roles(prompt_spec, roles_config)
    allowed_conditions = set(prompt_schema["required_fields"]["condition"]["allowed"])
    validate_spec_conditions(prompt_spec, allowed_conditions)

    trait_axis_id = str(prompt_spec["trait_axis_id"])
    if trait_axis_id != trait_config["trait_axis_id"]:
        raise ValueError(
            f"trait mismatch: spec={trait_axis_id!r} config={trait_config['trait_axis_id']!r}"
        )

    instruction_selection = str(prompt_spec["expansion"]["role_instruction_variants"])
    readout_policy = str(prompt_spec["expansion"]["readout_policy"])
    pattern = leakage_pattern(trait_config)
    records: list[PromptRecord] = []
    scenario_leaks: list[str] = []

    for role_id in prompt_spec["expansion"]["roles"]:
        role_set = role_set_for(role_id, roles_config)
        variants = selected_instruction_variants(role_id, roles_config, instruction_selection)
        for scenario in prompt_spec["scenarios"][role_id]:
            scenario_id = str(scenario["scenario_id"])
            for instruction_variant_id, role_instruction in variants:
                positive_id = (
                    matched_id(
                        trait_axis_id,
                        role_id,
                        scenario_id,
                        "present_positive",
                        instruction_variant_id,
                    )
                    if "present_positive" in prompt_spec["expansion"]["conditions"]
                    else None
                )
                negative_id = (
                    matched_id(
                        trait_axis_id,
                        role_id,
                        scenario_id,
                        "present_negative",
                        instruction_variant_id,
                    )
                    if "present_negative" in prompt_spec["expansion"]["conditions"]
                    else None
                )
                neutral_id = (
                    matched_id(
                        trait_axis_id,
                        role_id,
                        scenario_id,
                        "present_neutral",
                        instruction_variant_id,
                    )
                    if "present_neutral" in prompt_spec["expansion"]["conditions"]
                    else None
                )

                for condition in prompt_spec["expansion"]["conditions"]:
                    prompt_text = normalize_text(str(scenario[condition]))
                    prompt_id = stable_prompt_id(
                        trait_axis_id,
                        role_id,
                        scenario_id,
                        condition,
                        instruction_variant_id,
                    )
                    has_trait_word = trait_word_present(prompt_text, pattern)
                    validation_tags = []
                    if has_trait_word:
                        validation_tags.append("trait_word_present")
                    if condition in SCENARIO_INDUCED_CONDITIONS and has_trait_word:
                        scenario_leaks.append(prompt_id)

                    polarity = CONDITION_POLARITY[condition]
                    if condition in {
                        "present_positive",
                        "present_negative",
                        "other_positive",
                        "other_negative",
                    }:
                        matched_neutral_id = neutral_id
                    else:
                        matched_neutral_id = None

                    matched_positive_id = positive_id if condition == "present_neutral" else None
                    matched_negative_id = negative_id if condition == "present_neutral" else None

                    records.append(
                        PromptRecord(
                            prompt_id=prompt_id,
                            trait_axis_id=trait_axis_id,
                            role_id=role_id,
                            role_set=role_set,
                            role_instruction_variant_id=instruction_variant_id,
                            role_instruction=normalize_text(role_instruction),
                            scenario_id=scenario_id,
                            condition=condition,
                            polarity=polarity,
                            prompt_text=prompt_text,
                            full_prompt=render_full_prompt(role_instruction, prompt_text),
                            matched_neutral_id=matched_neutral_id,
                            matched_positive_id=matched_positive_id,
                            matched_negative_id=matched_negative_id,
                            trait_word_present=has_trait_word,
                            source={
                                "prompt_spec_id": prompt_spec["prompt_spec_id"],
                                "prompt_spec_author": prompt_spec["source"].get("author"),
                                "role_source": prompt_spec["source"].get("role_source"),
                                "trait_source": prompt_spec["source"].get("trait_source"),
                                "topic": scenario.get("topic"),
                            },
                            readout_policy=readout_policy,
                            speaker_spans=None,
                            safety_notes=[],
                            validation_tags=validation_tags,
                        )
                    )

    validation = validate_records(records, prompt_spec, scenario_leaks)
    return records, validation


def validate_records(
    records: list[PromptRecord],
    prompt_spec: dict[str, Any],
    scenario_leaks: list[str],
) -> dict[str, Any]:
    prompt_ids = [record.prompt_id for record in records]
    unique_prompt_ids = len(prompt_ids) == len(set(prompt_ids))
    record_by_id = {record.prompt_id: record for record in records}

    unresolved_neutral_links = [
        record.prompt_id
        for record in records
        if record.matched_neutral_id and record.matched_neutral_id not in record_by_id
    ]
    unresolved_positive_links = [
        record.prompt_id
        for record in records
        if record.matched_positive_id and record.matched_positive_id not in record_by_id
    ]
    unresolved_negative_links = [
        record.prompt_id
        for record in records
        if record.matched_negative_id and record.matched_negative_id not in record_by_id
    ]
    mention_records = [
        record.prompt_id
        for record in records
        if record.condition == "mention_without_possession"
    ]
    condition_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    for record in records:
        condition_counts[record.condition] = condition_counts.get(record.condition, 0) + 1
        role_counts[record.role_id] = role_counts.get(record.role_id, 0) + 1

    passed = (
        unique_prompt_ids
        and not unresolved_neutral_links
        and not unresolved_positive_links
        and not unresolved_negative_links
        and not scenario_leaks
    )
    return {
        "passed": passed,
        "unique_prompt_ids": unique_prompt_ids,
        "unresolved_neutral_links": unresolved_neutral_links,
        "unresolved_positive_links": unresolved_positive_links,
        "unresolved_negative_links": unresolved_negative_links,
        "scenario_induced_trait_word_leaks": scenario_leaks,
        "mention_without_possession_records": len(mention_records),
        "condition_counts": condition_counts,
        "role_counts": role_counts,
        "expected_conditions": prompt_spec["expansion"]["conditions"],
    }


def write_jsonl(path: Path, records: list[PromptRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")


def write_manifest(
    path: Path,
    records: list[PromptRecord],
    validation: dict[str, Any],
    source_paths: dict[str, Path],
    output_jsonl: Path,
    prompt_spec: dict[str, Any],
) -> None:
    role_ids = sorted({record.role_id for record in records})
    scenario_ids = sorted({record.scenario_id for record in records})
    conditions = prompt_spec["expansion"]["conditions"]
    manifest = {
        "schema_version": "0.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder": "PromptGridBuilder",
        "prompt_spec_id": prompt_spec["prompt_spec_id"],
        "trait_axis_id": prompt_spec["trait_axis_id"],
        "roles": role_ids,
        "conditions": conditions,
        "readout_policy": prompt_spec["expansion"]["readout_policy"],
        "counts": {
            "records": len(records),
            "roles": len(role_ids),
            "scenarios": len(scenario_ids),
            "conditions": len(conditions),
            "instruction_variants": len(
                sorted({record.role_instruction_variant_id for record in records})
            ),
        },
        "output_jsonl": str(output_jsonl),
        "output_jsonl_sha256": file_sha256(output_jsonl),
        "source_files": {
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for name, path in source_paths.items()
        },
        "validation": validation,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an expanded prompt grid JSONL.")
    parser.add_argument("--roles-config", type=Path, required=True)
    parser.add_argument("--trait-config", type=Path, required=True)
    parser.add_argument("--prompt-schema", type=Path, required=True)
    parser.add_argument("--prompt-spec", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    roles_config = load_yaml(args.roles_config)
    trait_config = load_yaml(args.trait_config)
    prompt_schema = load_yaml(args.prompt_schema)
    prompt_spec = load_yaml(args.prompt_spec)

    records, validation = build_prompt_records(
        roles_config=roles_config,
        trait_config=trait_config,
        prompt_schema=prompt_schema,
        prompt_spec=prompt_spec,
    )
    write_jsonl(args.output_jsonl, records)
    write_manifest(
        args.output_manifest,
        records,
        validation,
        {
            "roles_config": args.roles_config,
            "trait_config": args.trait_config,
            "prompt_schema": args.prompt_schema,
            "prompt_spec": args.prompt_spec,
        },
        args.output_jsonl,
        prompt_spec,
    )

    print(
        json.dumps(
            {
                "output_jsonl": str(args.output_jsonl),
                "output_manifest": str(args.output_manifest),
                "records": len(records),
                "validation_passed": validation["passed"],
                "validation": validation,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
