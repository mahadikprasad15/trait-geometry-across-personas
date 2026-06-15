from __future__ import annotations

import argparse
import fnmatch
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SelectedFile:
    local_path: str
    repo_path: str
    size_bytes: int


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def repo_root_from_cwd() -> Path:
    return Path.cwd().resolve()


def to_posix(path: Path) -> str:
    return path.as_posix()


def relpath(path: Path, root: Path) -> str:
    return to_posix(path.resolve().relative_to(root.resolve()))


def normalize_repo_path(path: str | None) -> str:
    if not path:
        return ""
    return path.strip("/")


def pattern_matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def load_policy(config: dict[str, Any], include_activations: bool, ignore_policy: bool) -> tuple[list[str], list[str]]:
    if ignore_policy:
        return ["**"], [".git/**", "**/.DS_Store"]

    upload_policy = config.get("upload_policy", {})
    include = list(upload_policy.get("include", []))
    exclude = list(upload_policy.get("exclude", []))
    if include_activations:
        activation_policy = config.get("activation_upload_policy", {})
        include.extend(activation_policy.get("include_when_flagged", []))
        exclude = [
            pattern
            for pattern in exclude
            if pattern not in set(activation_policy.get("include_when_flagged", []))
        ]
    return include or ["**"], exclude


def iter_local_files(local_path: Path) -> list[Path]:
    if local_path.is_file():
        return [local_path]
    if not local_path.exists():
        raise FileNotFoundError(f"local path does not exist: {local_path}")
    return sorted(path for path in local_path.rglob("*") if path.is_file())


def selected_repo_path(file_path: Path, local_path: Path, repo_path_prefix: str, repo_root: Path) -> str:
    if local_path.is_file():
        if repo_path_prefix and not repo_path_prefix.endswith("/"):
            return repo_path_prefix
        return "/".join(part for part in [repo_path_prefix, file_path.name] if part)
    relative_to_local = to_posix(file_path.resolve().relative_to(local_path.resolve()))
    return "/".join(part for part in [repo_path_prefix, relative_to_local] if part)


def select_files(
    local_path: Path,
    repo_path_prefix: str,
    repo_root: Path,
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> list[SelectedFile]:
    selected: list[SelectedFile] = []
    for file_path in iter_local_files(local_path):
        repo_relative = relpath(file_path, repo_root)
        upload_path = selected_repo_path(file_path, local_path, repo_path_prefix, repo_root)
        include_match = pattern_matches(repo_relative, include_patterns) or pattern_matches(
            upload_path, include_patterns
        )
        exclude_match = pattern_matches(repo_relative, exclude_patterns) or pattern_matches(
            upload_path, exclude_patterns
        )
        if not include_match or exclude_match:
            continue
        selected.append(
            SelectedFile(
                local_path=str(file_path),
                repo_path=upload_path,
                size_bytes=file_path.stat().st_size,
            )
        )
    return selected


def check_hf_dependency() -> dict[str, Any]:
    try:
        import huggingface_hub

        return {
            "ready": True,
            "huggingface_hub": {
                "installed": True,
                "version": getattr(huggingface_hub, "__version__", "installed"),
            },
        }
    except Exception as exc:
        return {
            "ready": False,
            "huggingface_hub": {"installed": False, "error": f"{type(exc).__name__}: {exc}"},
        }


def write_sync_manifest(
    manifest_dir: Path,
    mode: str,
    repo_id: str,
    repo_type: str,
    revision: str,
    local_path: Path,
    repo_path: str,
    selected_files: list[SelectedFile],
    commit_message: str,
    status: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    now = datetime.now(timezone.utc)
    manifest_path = manifest_dir / f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-hf-sync.json"
    payload = {
        "schema_version": "0.1",
        "runner": "HfArtifactSyncRunner",
        "mode": mode,
        "status": status,
        "created_at_utc": now.isoformat(),
        "repo_id": repo_id,
        "repo_type": repo_type,
        "revision": revision,
        "local_path": str(local_path),
        "repo_path": repo_path,
        "commit_message": commit_message,
        "file_count": len(selected_files),
        "total_bytes": sum(item.size_bytes for item in selected_files),
        "files": [asdict(item) for item in selected_files],
    }
    if extra:
        payload["extra"] = extra
    write_json(manifest_path, payload)
    return manifest_path


def upload_selected_files(
    selected_files: list[SelectedFile],
    repo_id: str,
    repo_type: str,
    revision: str,
    commit_message: str,
    token: str | None,
    create_repo: bool,
) -> str:
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    if create_repo:
        api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    operations = [
        CommitOperationAdd(path_in_repo=item.repo_path, path_or_fileobj=item.local_path)
        for item in selected_files
    ]
    commit_info = api.create_commit(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        operations=operations,
        commit_message=commit_message,
    )
    return str(getattr(commit_info, "commit_url", commit_info))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync selected local artifacts to a Hugging Face dataset repo.")
    parser.add_argument("--config", type=Path, default=Path("configs/storage/hf_sync.yaml"))
    parser.add_argument("--local-path", type=Path, required=True)
    parser.add_argument("--repo-path", default=None)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--repo-type", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--commit-message", required=True)
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--include-activations", action="store_true")
    parser.add_argument("--ignore-policy", action="store_true")
    parser.add_argument("--create-repo", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_yaml(args.config)
    repo_root = repo_root_from_cwd()
    local_path = args.local_path.resolve()
    repo_path = normalize_repo_path(args.repo_path)
    if args.repo_path is None:
        try:
            repo_path = relpath(local_path, repo_root)
        except ValueError:
            repo_path = local_path.name

    repo_id = args.repo_id or str(config["repo_id"])
    repo_type = args.repo_type or str(config.get("repo_type", "dataset"))
    revision = args.revision or str(config.get("revision", "main"))
    manifest_dir = args.manifest_dir or Path(config.get("sync_metadata", {}).get("manifest_dir", "artifacts/sync_manifests"))
    include_patterns, exclude_patterns = load_policy(
        config=config,
        include_activations=args.include_activations,
        ignore_policy=args.ignore_policy,
    )

    selected_files = select_files(
        local_path=local_path,
        repo_path_prefix=repo_path,
        repo_root=repo_root,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )
    dependency_status = check_hf_dependency()

    if args.dry_run:
        manifest_path = write_sync_manifest(
            manifest_dir=manifest_dir,
            mode="dry_run",
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            local_path=local_path,
            repo_path=repo_path,
            selected_files=selected_files,
            commit_message=args.commit_message,
            status="dry_run_complete",
            extra={
                "dependencies": dependency_status,
                "include_patterns": include_patterns,
                "exclude_patterns": exclude_patterns,
            },
        )
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "revision": revision,
                    "file_count": len(selected_files),
                    "total_bytes": sum(item.size_bytes for item in selected_files),
                    "manifest": str(manifest_path),
                    "first_files": [asdict(item) for item in selected_files[:10]],
                    "dependencies": dependency_status,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if not selected_files:
        print(json.dumps({"error": "no files selected for upload", "local_path": str(local_path)}, indent=2))
        return 2
    if not dependency_status["ready"]:
        print(
            json.dumps(
                {
                    "error": "HF sync dependencies are missing",
                    "dependencies": dependency_status,
                    "next_step": "Install huggingface_hub in the execution environment, then rerun.",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    token = os.environ.get("HF_TOKEN") or None
    require_token = bool(config.get("sync_metadata", {}).get("require_hf_token_for_upload", True))
    if require_token and not token:
        print(
            json.dumps(
                {
                    "error": "HF_TOKEN is required for upload",
                    "next_step": "Set HF_TOKEN in the Vast environment and rerun.",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    commit_url = upload_selected_files(
        selected_files=selected_files,
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        commit_message=args.commit_message,
        token=token,
        create_repo=args.create_repo,
    )
    manifest_path = write_sync_manifest(
        manifest_dir=manifest_dir,
        mode="upload",
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_path=local_path,
        repo_path=repo_path,
        selected_files=selected_files,
        commit_message=args.commit_message,
        status="uploaded",
        extra={"commit_url": commit_url},
    )
    print(
        json.dumps(
            {
                "status": "uploaded",
                "repo_id": repo_id,
                "file_count": len(selected_files),
                "total_bytes": sum(item.size_bytes for item in selected_files),
                "commit_url": commit_url,
                "manifest": str(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
