from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from trait_geometry.prompts.build_prompt_grid import (
    PromptRecord,
    file_sha256,
    normalize_text,
    role_set_for,
    stable_prompt_id,
    trait_word_present,
    variant_id,
)


CONDITION_POLARITY = {
    "instruction_positive": "positive",
    "instruction_negative": "negative",
    "instruction_neutral": "neutral",
}


@dataclass(frozen=True)
class AssistantAxisQuestion:
    id: int
    category: str
    question: str


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[PromptRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")


def hash_records(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def select_role_variants(
    role_id: str,
    roles_config: dict[str, Any],
    first_n: int,
) -> list[tuple[str, str]]:
    variants = roles_config["roles"][role_id]["instruction_variants"]
    if first_n < 1:
        raise ValueError("role_instruction_variants.first_n must be >= 1")
    selected = variants[:first_n]
    if len(selected) < first_n:
        raise ValueError(f"role {role_id!r} only has {len(selected)} instruction variants")
    return [(variant_id(index), normalize_text(str(text))) for index, text in enumerate(selected, start=1)]


def load_questions(config: dict[str, Any]) -> list[AssistantAxisQuestion]:
    questions = []
    for item in config["questions"]["selected"]:
        questions.append(
            AssistantAxisQuestion(
                id=int(item["id"]),
                category=str(item["category"]),
                question=normalize_text(str(item["question"])),
            )
        )
    ids = [question.id for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("question ids must be unique")
    return questions


def validate_config(config: dict[str, Any], roles_config: dict[str, Any]) -> None:
    roles = config["roles"]
    missing_roles = [role for role in roles if role not in roles_config["roles"]]
    if missing_roles:
        raise ValueError(f"roles missing from role config: {missing_roles}")

    conditions = config["conditions"]
    if conditions != ["instruction_positive", "instruction_negative", "instruction_neutral"]:
        raise ValueError(
            "AssistantAxisGridBuilder currently expects conditions "
            "['instruction_positive', 'instruction_negative', 'instruction_neutral']"
        )

    trait_ids = [trait["trait_axis_id"] for trait in config["trait_axes"]]
    if len(trait_ids) != len(set(trait_ids)):
        raise ValueError("trait_axis_id values must be unique")

    required_poles = ["positive_pole", "negative_pole"]
    for trait in config["trait_axes"]:
        for pole in required_poles:
            if pole not in trait:
                raise ValueError(f"trait {trait.get('trait_axis_id')} missing {pole}")
            if "id" not in trait[pole] or "definition" not in trait[pole]:
                raise ValueError(f"trait {trait.get('trait_axis_id')} {pole} needs id and definition")


def instruction_text_for_condition(
    condition: str,
    trait: dict[str, Any],
    role_id: str,
) -> str:
    positive = trait["positive_pole"]
    negative = trait["negative_pole"]
    neutral_target = str(
        trait.get(
            "neutral_target",
            "Answer naturally in the assigned role without emphasizing either trait pole.",
        )
    )
    if condition == "instruction_positive":
        return (
            f"While staying in the {role_id} role, respond in a distinctly "
            f"{positive['label']} way. Trait definition: {positive['definition']}"
        )
    if condition == "instruction_negative":
        return (
            f"While staying in the {role_id} role, respond in a distinctly "
            f"{negative['label']} way. Trait definition: {negative['definition']}"
        )
    if condition == "instruction_neutral":
        return f"While staying in the {role_id} role, {neutral_target}"
    raise ValueError(f"unknown instruction condition: {condition}")


def render_full_prompt(role_instruction: str, trait_instruction: str, question: str) -> str:
    return (
        "Role instruction:\n"
        f"{normalize_text(role_instruction)}\n\n"
        "Trait instruction:\n"
        f"{normalize_text(trait_instruction)}\n\n"
        "Question:\n"
        f"{normalize_text(question)}"
    )


def leakage_terms_for_trait(trait: dict[str, Any]) -> list[str]:
    terms = [
        trait["trait_axis_id"],
        trait["positive_pole"]["id"],
        trait["positive_pole"]["label"],
        trait["negative_pole"]["id"],
        trait["negative_pole"]["label"],
    ]
    terms.extend(trait.get("lexical_leakage_terms", []))
    return [str(term).lower() for term in terms]


def question_has_trait_word(question: str, terms: list[str]) -> bool:
    lowered = question.lower()
    return any(term and term in lowered for term in terms)


def build_trait_records(
    config: dict[str, Any],
    roles_config: dict[str, Any],
    trait: dict[str, Any],
    questions: list[AssistantAxisQuestion],
) -> tuple[list[PromptRecord], dict[str, Any]]:
    trait_axis_id = str(trait["trait_axis_id"])
    roles = list(config["roles"])
    conditions = list(config["conditions"])
    variant_count = int(config["role_instruction_variants"]["first_n"])
    readout_policy = str(config["readout_policy"])
    records: list[PromptRecord] = []
    question_trait_leaks: list[dict[str, Any]] = []
    leakage_terms = leakage_terms_for_trait(trait)

    for role_id in roles:
        role_set = role_set_for(role_id, roles_config)
        variants = select_role_variants(role_id, roles_config, variant_count)
        for question in questions:
            scenario_id = f"aaq{question.id:03d}"
            for variant, role_instruction in variants:
                positive_id = stable_prompt_id(
                    trait_axis_id, role_id, scenario_id, "instruction_positive", variant
                )
                negative_id = stable_prompt_id(
                    trait_axis_id, role_id, scenario_id, "instruction_negative", variant
                )
                neutral_id = stable_prompt_id(
                    trait_axis_id, role_id, scenario_id, "instruction_neutral", variant
                )
                if question_has_trait_word(question.question, leakage_terms):
                    question_trait_leaks.append(
                        {
                            "trait_axis_id": trait_axis_id,
                            "question_id": question.id,
                            "question": question.question,
                        }
                    )

                for condition in conditions:
                    prompt_id = stable_prompt_id(
                        trait_axis_id, role_id, scenario_id, condition, variant
                    )
                    trait_instruction = instruction_text_for_condition(condition, trait, role_id)
                    prompt_text = f"{trait_instruction}\n\n{question.question}"
                    has_trait_word = trait_word_present(prompt_text, _literal_pattern(leakage_terms))
                    records.append(
                        PromptRecord(
                            prompt_id=prompt_id,
                            trait_axis_id=trait_axis_id,
                            role_id=role_id,
                            role_set=role_set,
                            role_instruction_variant_id=variant,
                            role_instruction=role_instruction,
                            scenario_id=scenario_id,
                            condition=condition,
                            polarity=CONDITION_POLARITY[condition],
                            prompt_text=normalize_text(prompt_text),
                            full_prompt=render_full_prompt(
                                role_instruction=role_instruction,
                                trait_instruction=trait_instruction,
                                question=question.question,
                            ),
                            matched_neutral_id=neutral_id
                            if condition in {"instruction_positive", "instruction_negative"}
                            else None,
                            matched_positive_id=positive_id if condition == "instruction_neutral" else None,
                            matched_negative_id=negative_id if condition == "instruction_neutral" else None,
                            trait_word_present=has_trait_word,
                            source={
                                "experiment_id": config["experiment_id"],
                                "source_name": config["source"]["name"],
                                "source_repository": config["source"]["repository"],
                                "role_source_url": roles_config["roles"][role_id].get("source_url"),
                                "trait_source_url": config["source"]["trait_list_url"],
                                "question_source_url": config["source"]["questions_url"],
                                "question_id": question.id,
                                "question_category": question.category,
                                "trait_positive_source_key": trait["positive_pole"]["id"],
                                "trait_negative_source_key": trait["negative_pole"]["id"],
                                "elicitation_mode": "explicit_trait_instruction",
                            },
                            readout_policy=readout_policy,
                            speaker_spans=None,
                            safety_notes=[],
                            validation_tags=["explicit_trait_instruction"],
                        )
                    )

    validation = validate_records(records, expected_count=len(roles) * len(questions) * variant_count * len(conditions))
    validation["question_trait_word_leaks"] = question_trait_leaks
    return records, validation


def _literal_pattern(terms: list[str]):
    import re

    escaped = [re.escape(term) for term in sorted(set(terms), key=len, reverse=True) if term]
    if not escaped:
        return re.compile(r"a\Ab")
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.I)


def validate_records(records: list[PromptRecord], expected_count: int) -> dict[str, Any]:
    ids = [record.prompt_id for record in records]
    id_set = set(ids)
    duplicate_ids = sorted(prompt_id for prompt_id in id_set if ids.count(prompt_id) > 1)
    missing_links = []
    for record in records:
        for field in ["matched_neutral_id", "matched_positive_id", "matched_negative_id"]:
            value = getattr(record, field)
            if value and value not in id_set:
                missing_links.append({"prompt_id": record.prompt_id, "field": field, "missing": value})
    return {
        "passed": len(records) == expected_count and not duplicate_ids and not missing_links,
        "record_count": len(records),
        "expected_count": expected_count,
        "duplicate_prompt_ids": duplicate_ids,
        "missing_matched_links": missing_links,
    }


def count_field(records: list[PromptRecord], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = str(getattr(record, field))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def write_outputs(
    config_path: Path,
    roles_config_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    all_outputs: list[dict[str, Any]],
) -> Path:
    manifest_path = output_dir / f"{config['experiment_id']}_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": "0.1",
            "builder": "AssistantAxisGridBuilder",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_id": config["experiment_id"],
            "source": config["source"],
            "config": {
                "path": str(config_path),
                "sha256": file_sha256(config_path),
            },
            "roles_config": {
                "path": str(roles_config_path),
                "sha256": file_sha256(roles_config_path),
            },
            "roles": config["roles"],
            "conditions": config["conditions"],
            "role_instruction_variants": config["role_instruction_variants"],
            "questions": {
                "count": len(config["questions"]["selected"]),
                "categories": sorted({item["category"] for item in config["questions"]["selected"]}),
                "selected_ids": [item["id"] for item in config["questions"]["selected"]],
                "selected_sha256": hash_records(config["questions"]["selected"]),
            },
            "trait_outputs": all_outputs,
        },
    )
    return manifest_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Assistant Axis explicit trait-instruction prompt grids.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--roles-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_yaml(args.config)
    roles_config = load_yaml(args.roles_config)
    validate_config(config, roles_config)
    questions = load_questions(config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_outputs = []
    for trait in config["trait_axes"]:
        records, validation = build_trait_records(config, roles_config, trait, questions)
        if not validation["passed"]:
            raise ValueError(f"prompt validation failed for {trait['trait_axis_id']}: {validation}")
        output_jsonl = args.output_dir / f"{trait['trait_axis_id']}_{config['prompt_set_id']}.jsonl"
        output_manifest = args.output_dir / f"{trait['trait_axis_id']}_{config['prompt_set_id']}_manifest.json"
        write_jsonl(output_jsonl, records)
        trait_output = {
            "trait_axis_id": trait["trait_axis_id"],
            "output_jsonl": str(output_jsonl),
            "output_jsonl_sha256": file_sha256(output_jsonl),
            "output_manifest": str(output_manifest),
            "counts": {
                "records": len(records),
                "roles": count_field(records, "role_id"),
                "conditions": count_field(records, "condition"),
                "role_instruction_variant_id": count_field(records, "role_instruction_variant_id"),
            },
            "validation": validation,
        }
        write_json(
            output_manifest,
            {
                "schema_version": "0.1",
                "builder": "AssistantAxisGridBuilder",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                **trait_output,
            },
        )
        all_outputs.append(trait_output)

    aggregate_manifest = write_outputs(
        config_path=args.config,
        roles_config_path=args.roles_config,
        output_dir=args.output_dir,
        config=config,
        all_outputs=all_outputs,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "experiment_id": config["experiment_id"],
                "prompt_set_id": config["prompt_set_id"],
                "traits": len(all_outputs),
                "records_total": sum(output["counts"]["records"] for output in all_outputs),
                "output_dir": str(args.output_dir),
                "manifest": str(aggregate_manifest),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
