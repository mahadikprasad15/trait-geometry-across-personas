from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SCORE_FIELDS = [
    "positive_pole_score",
    "negative_pole_score",
    "role_adherence_score",
    "coherence_score",
    "prompt_following_score",
    "trait_word_discussion_score",
]


@dataclass(frozen=True)
class JudgeWorkItem:
    prompt_id: str
    trait_axis_id: str
    role_id: str
    condition: str
    scenario_id: str
    full_prompt: str
    completion: str
    metadata: dict[str, Any]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


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


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def append_log(path: Path, message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{timestamp} {message}\n")


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def progress_iter(items: list[Any], description: str):
    try:
        from tqdm.auto import tqdm

        return tqdm(items, desc=description, unit="judgment")
    except Exception:
        return items


def make_run_dirs(run_root: Path) -> dict[str, Path]:
    paths = {
        "inputs": run_root / "inputs",
        "checkpoints": run_root / "checkpoints",
        "results": run_root / "results",
        "logs": run_root / "logs",
        "meta": run_root / "meta",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def completed_prompt_ids_from_judgments(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in existing judgments line {line_no}: {exc}") from exc
            if record.get("prompt_id"):
                completed.add(str(record["prompt_id"]))
    return completed


def build_work_items(generation_records: list[dict[str, Any]]) -> list[JudgeWorkItem]:
    items: list[JudgeWorkItem] = []
    for record in generation_records:
        metadata = record.get("metadata") or {}
        trait_axis_id = record.get("trait_axis_id") or metadata.get("trait_axis_id")
        if not trait_axis_id:
            raise ValueError(f"generation record {record.get('prompt_id')} lacks trait_axis_id")
        completion = record.get("completion")
        if completion is None:
            raise ValueError(f"generation record {record.get('prompt_id')} lacks completion")
        items.append(
            JudgeWorkItem(
                prompt_id=str(record["prompt_id"]),
                trait_axis_id=str(trait_axis_id),
                role_id=str(record["role_id"]),
                condition=str(record["condition"]),
                scenario_id=str(record["scenario_id"]),
                full_prompt=str(record["full_prompt"]),
                completion=str(completion),
                metadata=metadata,
            )
        )
    return items


def format_markers(markers: list[Any]) -> str:
    return "\n".join(f"- {marker}" for marker in markers)


def render_judge_messages(
    item: JudgeWorkItem,
    trait_config: dict[str, Any],
    rubric_config: dict[str, Any],
) -> list[dict[str, str]]:
    template = rubric_config["judge_prompt"]["user_template"]
    user_content = template.format(
        trait_axis_id=item.trait_axis_id,
        positive_label=trait_config["positive_pole"]["label"],
        positive_definition=trait_config["positive_pole"]["definition"],
        positive_markers=format_markers(trait_config["positive_pole"].get("behavioral_markers", [])),
        negative_label=trait_config["negative_pole"]["label"],
        negative_definition=trait_config["negative_pole"]["definition"],
        negative_markers=format_markers(trait_config["negative_pole"].get("behavioral_markers", [])),
        role_id=item.role_id,
        condition=item.condition,
        scenario_id=item.scenario_id,
        full_prompt=item.full_prompt,
        completion=item.completion,
    )
    return [
        {"role": "system", "content": rubric_config["judge_prompt"]["system"]},
        {"role": "user", "content": user_content},
    ]


def judgment_json_schema() -> dict[str, Any]:
    properties: dict[str, Any] = {
        field: {"type": "integer", "minimum": 1, "maximum": 5}
        for field in SCORE_FIELDS
    }
    properties.update(
        {
            "positive_evidence": {"type": "string"},
            "negative_evidence": {"type": "string"},
            "role_adherence_evidence": {"type": "string"},
            "rationale": {"type": "string"},
        }
    )
    return {
        "type": "object",
        "properties": properties,
        "required": [
            *SCORE_FIELDS,
            "positive_evidence",
            "negative_evidence",
            "role_adherence_evidence",
            "rationale",
        ],
        "additionalProperties": False,
    }


def responses_text_format() -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": "trait_behavior_judgment",
            "strict": True,
            "schema": judgment_json_schema(),
        }
    }


def check_judge_dependencies(backend: str) -> dict[str, Any]:
    if backend != "openai_responses":
        return {"ready": True}
    try:
        import openai

        return {"ready": True, "openai": {"installed": True, "version": getattr(openai, "__version__", "installed")}}
    except Exception as exc:
        return {
            "ready": False,
            "openai": {"installed": False, "error": f"{type(exc).__name__}: {exc}"},
        }


def extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)
    if hasattr(response, "model_dump"):
        payload = response.model_dump()
    elif isinstance(response, dict):
        payload = response
    else:
        raise ValueError("could not extract text from OpenAI response")
    texts: list[str] = []
    for output_item in payload.get("output", []):
        for content in output_item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                texts.append(str(content["text"]))
    if not texts:
        raise ValueError("OpenAI response did not contain output text")
    return "\n".join(texts)


def validate_judgment_payload(payload: dict[str, Any]) -> dict[str, Any]:
    for field in SCORE_FIELDS:
        value = payload.get(field)
        if not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"judgment field {field!r} must be an integer from 1 to 5")
    for field in ["positive_evidence", "negative_evidence", "role_adherence_evidence", "rationale"]:
        if not isinstance(payload.get(field), str):
            raise ValueError(f"judgment field {field!r} must be a string")
    return payload


def call_openai_responses(
    item: JudgeWorkItem,
    trait_config: dict[str, Any],
    rubric_config: dict[str, Any],
    judge_config: dict[str, Any],
    model_override: str | None,
) -> dict[str, Any]:
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for --backend openai_responses")
    client = OpenAI()
    model = model_override or judge_config["openai_model"]
    generation = judge_config.get("generation", {})
    response = client.responses.create(
        model=model,
        input=render_judge_messages(item, trait_config, rubric_config),
        temperature=float(generation.get("temperature", 0.0)),
        max_output_tokens=int(generation.get("max_output_tokens", 500)),
        text=responses_text_format(),
    )
    text = extract_response_text(response)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"judge returned invalid JSON for prompt_id={item.prompt_id}: {text}") from exc
    return validate_judgment_payload(payload)


def build_judgment_record(
    item: JudgeWorkItem,
    judgment_payload: dict[str, Any],
    rubric_config: dict[str, Any],
    judge_config: dict[str, Any],
    backend: str,
    model_override: str | None,
) -> dict[str, Any]:
    return {
        "prompt_id": item.prompt_id,
        "trait_axis_id": item.trait_axis_id,
        "role_id": item.role_id,
        "condition": item.condition,
        "scenario_id": item.scenario_id,
        "metadata": item.metadata,
        "judge_backend": backend,
        "judge_model": model_override or judge_config.get("openai_model"),
        "rubric_id": rubric_config["rubric_id"],
        "judged_at_utc": datetime.now(timezone.utc).isoformat(),
        **judgment_payload,
    }


def write_dry_run_artifacts(
    run_root: Path,
    generations_jsonl: Path,
    trait_config_path: Path,
    rubric_config_path: Path,
    judge_config_path: Path,
    backend: str,
    work_items: list[JudgeWorkItem],
    dependency_status: dict[str, Any],
) -> None:
    paths = make_run_dirs(run_root)
    now = datetime.now(timezone.utc).isoformat()
    write_json(
        paths["meta"] / "judge_manifest.json",
        {
            "schema_version": "0.1",
            "runner": "TraitJudgeRunner",
            "mode": "dry_run",
            "created_at_utc": now,
            "generations_jsonl": str(generations_jsonl),
            "trait_config": str(trait_config_path),
            "rubric_config": str(rubric_config_path),
            "judge_config": str(judge_config_path),
            "backend": backend,
            "planned_records": len(work_items),
            "output_judgments": str(paths["results"] / "judgments.jsonl"),
        },
    )
    write_json(
        paths["meta"] / "judge_status.json",
        {
            "status": "dry_run_complete",
            "updated_at_utc": now,
            "completed_records": 0,
            "planned_records": len(work_items),
            "dependencies": dependency_status,
        },
    )
    write_json(
        paths["checkpoints"] / "judge_progress.json",
        {
            "cursor": 0,
            "completed_prompt_ids": [],
            "planned_prompt_ids": [item.prompt_id for item in work_items],
        },
    )
    write_json(
        paths["inputs"] / "judge_preview.json",
        {
            "planned_records": len(work_items),
            "first_items": [asdict(item) for item in work_items[:5]],
        },
    )
    append_log(paths["logs"] / "judge.log", "dry run completed; no judge calls executed")


def write_initial_artifacts(
    run_root: Path,
    generations_jsonl: Path,
    trait_config_path: Path,
    rubric_config_path: Path,
    judge_config_path: Path,
    backend: str,
    work_items: list[JudgeWorkItem],
    dependency_status: dict[str, Any],
) -> dict[str, Path]:
    paths = make_run_dirs(run_root)
    now = datetime.now(timezone.utc).isoformat()
    manifest_path = paths["meta"] / "judge_manifest.json"
    if not manifest_path.exists():
        write_json(
            manifest_path,
            {
                "schema_version": "0.1",
                "runner": "TraitJudgeRunner",
                "mode": "execute",
                "created_at_utc": now,
                "generations_jsonl": str(generations_jsonl),
                "trait_config": str(trait_config_path),
                "rubric_config": str(rubric_config_path),
                "judge_config": str(judge_config_path),
                "backend": backend,
                "planned_records": len(work_items),
                "output_judgments": str(paths["results"] / "judgments.jsonl"),
            },
        )
    write_json(
        paths["meta"] / "judge_status.json",
        {
            "status": "running",
            "updated_at_utc": now,
            "completed_records": 0,
            "planned_records": len(work_items),
            "dependencies": dependency_status,
        },
    )
    progress_path = paths["checkpoints"] / "judge_progress.json"
    if not progress_path.exists():
        write_json(
            progress_path,
            {
                "cursor": 0,
                "completed_prompt_ids": [],
                "planned_prompt_ids": [item.prompt_id for item in work_items],
            },
        )
    append_log(paths["logs"] / "judge.log", "trait judge run initialized")
    return paths


def run_trait_judge(
    run_root: Path,
    generations_jsonl: Path,
    trait_config_path: Path,
    rubric_config_path: Path,
    judge_config_path: Path,
    trait_config: dict[str, Any],
    rubric_config: dict[str, Any],
    judge_config: dict[str, Any],
    backend: str,
    work_items: list[JudgeWorkItem],
    dependency_status: dict[str, Any],
    save_every: int,
    model_override: str | None,
) -> dict[str, Any]:
    if backend != "openai_responses":
        raise ValueError(f"unsupported judge backend {backend!r}")
    paths = write_initial_artifacts(
        run_root=run_root,
        generations_jsonl=generations_jsonl,
        trait_config_path=trait_config_path,
        rubric_config_path=rubric_config_path,
        judge_config_path=judge_config_path,
        backend=backend,
        work_items=work_items,
        dependency_status=dependency_status,
    )
    judgments_path = paths["results"] / "judgments.jsonl"
    progress_path = paths["checkpoints"] / "judge_progress.json"
    status_path = paths["meta"] / "judge_status.json"
    log_path = paths["logs"] / "judge.log"

    completed = completed_prompt_ids_from_judgments(judgments_path)
    progress = read_json(progress_path) or {}
    completed.update(str(prompt_id) for prompt_id in progress.get("completed_prompt_ids", []))
    remaining = [item for item in work_items if item.prompt_id not in completed]

    judged_this_run = 0
    for item in progress_iter(remaining, "judging completions"):
        try:
            judgment_payload = call_openai_responses(
                item=item,
                trait_config=trait_config,
                rubric_config=rubric_config,
                judge_config=judge_config,
                model_override=model_override,
            )
            record = build_judgment_record(
                item=item,
                judgment_payload=judgment_payload,
                rubric_config=rubric_config,
                judge_config=judge_config,
                backend=backend,
                model_override=model_override,
            )
            append_jsonl(judgments_path, [record])
            completed.add(item.prompt_id)
            judged_this_run += 1
        except Exception as exc:
            write_json(
                status_path,
                {
                    "status": "failed",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "completed_records": len(completed),
                    "planned_records": len(work_items),
                    "error": f"{type(exc).__name__}: {exc}",
                    "failed_prompt_id": item.prompt_id,
                },
            )
            append_log(log_path, f"failed on prompt_id={item.prompt_id}: {type(exc).__name__}: {exc}")
            raise

        if judged_this_run % save_every == 0:
            write_json(
                progress_path,
                {
                    "cursor": len(completed),
                    "completed_prompt_ids": sorted(completed),
                    "planned_prompt_ids": [work_item.prompt_id for work_item in work_items],
                },
            )
            write_json(
                status_path,
                {
                    "status": "running",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "completed_records": len(completed),
                    "planned_records": len(work_items),
                },
            )
            append_log(log_path, f"checkpoint saved completed={len(completed)}")

    write_json(
        progress_path,
        {
            "cursor": len(completed),
            "completed_prompt_ids": sorted(completed),
            "planned_prompt_ids": [work_item.prompt_id for work_item in work_items],
        },
    )
    write_json(
        status_path,
        {
            "status": "completed",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_records": len(completed),
            "planned_records": len(work_items),
            "output_judgments": str(judgments_path),
        },
    )
    append_log(log_path, f"trait judging completed completed={len(completed)}")
    return {
        "status": "completed",
        "planned_records": len(work_items),
        "completed_records": len(completed),
        "judged_this_run": judged_this_run,
        "output_judgments": str(judgments_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run behavioral trait judging over generated completions.")
    parser.add_argument("--generations-jsonl", type=Path, required=True)
    parser.add_argument("--trait-config", type=Path, required=True)
    parser.add_argument("--rubric-config", type=Path, required=True)
    parser.add_argument("--judge-config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--backend", choices=["openai_responses"], default="openai_responses")
    parser.add_argument("--model", default=None, help="Optional judge model override.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    generation_records = load_jsonl(args.generations_jsonl, limit=args.limit)
    work_items = build_work_items(generation_records)
    trait_config = load_yaml(args.trait_config)
    rubric_config = load_yaml(args.rubric_config)
    judge_config = load_yaml(args.judge_config)
    dependency_status = check_judge_dependencies(args.backend)

    if args.dry_run:
        write_dry_run_artifacts(
            run_root=args.run_root,
            generations_jsonl=args.generations_jsonl,
            trait_config_path=args.trait_config,
            rubric_config_path=args.rubric_config,
            judge_config_path=args.judge_config,
            backend=args.backend,
            work_items=work_items,
            dependency_status=dependency_status,
        )
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "planned_records": len(work_items),
                    "run_root": str(args.run_root),
                    "dependencies_ready": dependency_status["ready"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if not dependency_status["ready"]:
        print(
            json.dumps(
                {
                    "error": "judge dependencies are missing",
                    "backend": args.backend,
                    "dependencies": dependency_status,
                    "next_step": "Install openai and set OPENAI_API_KEY, then rerun or use --dry-run.",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    result = run_trait_judge(
        run_root=args.run_root,
        generations_jsonl=args.generations_jsonl,
        trait_config_path=args.trait_config,
        rubric_config_path=args.rubric_config,
        judge_config_path=args.judge_config,
        trait_config=trait_config,
        rubric_config=rubric_config,
        judge_config=judge_config,
        backend=args.backend,
        work_items=work_items,
        dependency_status=dependency_status,
        save_every=args.save_every,
        model_override=args.model,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
