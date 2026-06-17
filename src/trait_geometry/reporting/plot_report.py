from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SCALAR_GATE_SUMMARY = Path("artifacts/reports/five_trait_pilot_v0/multi_trait_summary.json")
DEFAULT_GEOMETRY_SUMMARY = Path("artifacts/reports/five_trait_geometry_v0/geometry_summary.json")

TRAIT_LABELS = {
    "warmth_coldness": "warmth",
    "sincerity_manipulativeness": "sincerity",
    "caution_recklessness": "caution",
    "curiosity_closed_mindedness": "curiosity",
    "skepticism_gullibility": "skepticism",
}

ROLE_ORDER = ["counselor", "tutor", "debugger", "journalist"]
TRAIT_ORDER = [
    "warmth_coldness",
    "sincerity_manipulativeness",
    "caution_recklessness",
    "curiosity_closed_mindedness",
    "skepticism_gullibility",
]


def load_plot_deps():
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        return plt, np
    except Exception as exc:
        raise RuntimeError(
            "Plot dependencies are missing. Install with: pip install matplotlib numpy"
        ) from exc


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def trait_label(trait_axis_id: str) -> str:
    return TRAIT_LABELS.get(trait_axis_id, trait_axis_id)


def ordered_unique(values: list[str], preferred: list[str]) -> list[str]:
    present = set(values)
    ordered = [value for value in preferred if value in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def matrix_from_role_rows(
    rows: list[dict[str, Any]],
    value_key: str,
) -> tuple[list[str], list[str], list[list[float | None]]]:
    roles = ordered_unique([str(row["role_id"]) for row in rows], ROLE_ORDER)
    traits = ordered_unique([str(row["trait_axis_id"]) for row in rows], TRAIT_ORDER)
    by_key = {
        (str(row["role_id"]), str(row["trait_axis_id"])): number(row.get(value_key))
        for row in rows
    }
    matrix = [[by_key.get((role, trait)) for trait in traits] for role in roles]
    return roles, traits, matrix


def matrix_to_numpy(matrix: list[list[float | None]]):
    _, np = load_plot_deps()
    return np.array(
        [[float("nan") if value is None else float(value) for value in row] for row in matrix],
        dtype=float,
    )


def diverging_limits(values) -> tuple[float, float]:
    _, np = load_plot_deps()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    vmax = max(abs(float(finite.min())), abs(float(finite.max())))
    if vmax == 0.0:
        vmax = 1.0
    return -vmax, vmax


def save_heatmap(
    path: Path,
    matrix: list[list[float | None]],
    x_labels: list[str],
    y_labels: list[str],
    title: str,
    colorbar_label: str,
    center_zero: bool = True,
) -> None:
    plt, np = load_plot_deps()
    data = matrix_to_numpy(matrix)
    if center_zero:
        vmin, vmax = diverging_limits(data)
        cmap = "RdBu_r"
    else:
        finite = data[np.isfinite(data)]
        vmin = float(finite.min()) if finite.size else 0.0
        vmax = float(finite.max()) if finite.size else 1.0
        cmap = "viridis"

    height = max(3.2, 0.55 * len(y_labels) + 1.4)
    width = max(5.8, 0.95 * len(x_labels) + 2.0)
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(x_labels)), [trait_label(label) for label in x_labels], rotation=35, ha="right")
    ax.set_yticks(range(len(y_labels)), y_labels)
    ax.set_title(title, fontsize=12, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([i - 0.5 for i in range(1, len(x_labels))], minor=True)
    ax.set_yticks([i - 0.5 for i in range(1, len(y_labels))], minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for y_idx, row in enumerate(data):
        for x_idx, value in enumerate(row):
            if not np.isfinite(value):
                continue
            text_color = "#ffffff" if abs(float(value)) > 0.65 * max(abs(vmin), abs(vmax)) else "#222222"
            ax.text(x_idx, y_idx, f"{value:.2f}", ha="center", va="center", fontsize=8, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label, rotation=90)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_ruler_matrix(
    rows: list[dict[str, Any]],
) -> tuple[list[str], list[list[float | None]]]:
    traits = ordered_unique(
        [str(row["trait_a"]) for row in rows] + [str(row["trait_b"]) for row in rows],
        TRAIT_ORDER,
    )
    values: dict[tuple[str, str], float] = {(trait, trait): 1.0 for trait in traits}
    for row in rows:
        trait_a = str(row["trait_a"])
        trait_b = str(row["trait_b"])
        value = number(row.get("cosine"))
        if value is None:
            continue
        values[(trait_a, trait_b)] = value
        values[(trait_b, trait_a)] = value
    matrix = [[values.get((trait_a, trait_b)) for trait_b in traits] for trait_a in traits]
    return traits, matrix


def save_role_pair_bar(path: Path, rows: list[dict[str, Any]]) -> None:
    plt, np = load_plot_deps()
    axis_rows = [row for row in rows if row.get("vector_type") == "axis_vector"]
    axis_rows = sorted(
        axis_rows,
        key=lambda row: (str(row.get("trait_axis_id")), number(row.get("cosine")) or 0.0),
    )
    labels = [
        f"{trait_label(str(row['trait_axis_id']))}\n{row['role_a']}|{row['role_b']}"
        for row in axis_rows
    ]
    values = [number(row.get("cosine")) or 0.0 for row in axis_rows]
    colors = ["#2b6cb0" if value >= 0 else "#c93a2e" for value in values]
    fig_width = max(8.0, 0.35 * len(values))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")
    ax.bar(range(len(values)), values, color=colors, alpha=0.85)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("cosine")
    ax.set_title("Role-pair axis-vector cosines", fontsize=12, pad=12)
    ax.set_xticks(range(len(values)), labels, rotation=70, ha="right", fontsize=7)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_markdown(path: Path, artifacts: dict[str, str]) -> None:
    lines = [
        "# Pilot Plot Pack",
        "",
        "Generated plots:",
        "",
    ]
    for name, artifact_path in artifacts.items():
        if not artifact_path.endswith(".png"):
            continue
        lines.append(f"- `{name}`: `{artifact_path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_plots(
    scalar_gate_summary: Path,
    geometry_summary: Path,
    output_dir: Path,
) -> dict[str, str]:
    scalar_payload = read_json(scalar_gate_summary)
    geometry_payload = read_json(geometry_summary)
    role_rows = list(scalar_payload.get("role_rows") or [])
    if not role_rows:
        raise ValueError(f"{scalar_gate_summary} does not contain role_rows")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}

    roles, traits, positive_matrix = matrix_from_role_rows(role_rows, "positive_shift_scalar")
    positive_path = output_dir / "scalar_positive_shift_heatmap.png"
    save_heatmap(
        positive_path,
        positive_matrix,
        traits,
        roles,
        "Positive elicitation shifts by role and trait",
        "projection on ruler",
    )
    artifacts["scalar_positive_shift_heatmap"] = str(positive_path)

    _, _, negative_matrix = matrix_from_role_rows(role_rows, "negative_shift_scalar")
    negative_path = output_dir / "scalar_negative_shift_heatmap.png"
    save_heatmap(
        negative_path,
        negative_matrix,
        traits,
        roles,
        "Negative elicitation shifts by role and trait",
        "projection on ruler",
    )
    artifacts["scalar_negative_shift_heatmap"] = str(negative_path)

    _, _, alignment_matrix = matrix_from_role_rows(role_rows, "axis_alignment_cosine")
    alignment_path = output_dir / "axis_alignment_heatmap.png"
    save_heatmap(
        alignment_path,
        alignment_matrix,
        traits,
        roles,
        "Role axis-vector alignment with pooled ruler",
        "cosine",
    )
    artifacts["axis_alignment_heatmap"] = str(alignment_path)

    ruler_rows = list(geometry_payload.get("ruler_cosines") or [])
    if ruler_rows:
        ruler_traits, ruler_matrix = build_ruler_matrix(ruler_rows)
        ruler_path = output_dir / "cross_trait_ruler_cosines.png"
        save_heatmap(
            ruler_path,
            ruler_matrix,
            ruler_traits,
            [trait_label(trait) for trait in ruler_traits],
            "Cross-trait ruler cosine matrix",
            "cosine",
        )
        artifacts["cross_trait_ruler_cosines"] = str(ruler_path)

    role_pair_rows = list(geometry_payload.get("role_pair_cosines") or [])
    if role_pair_rows:
        role_pair_path = output_dir / "role_pair_axis_cosines.png"
        save_role_pair_bar(role_pair_path, role_pair_rows)
        artifacts["role_pair_axis_cosines"] = str(role_pair_path)

    markdown_path = output_dir / "plot_pack.md"
    write_markdown(markdown_path, artifacts)
    artifacts["markdown"] = str(markdown_path)

    manifest_path = output_dir / "plot_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": "0.1",
            "builder": "PilotPlotBuilder",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "scalar_gate_summary": str(scalar_gate_summary),
                "geometry_summary": str(geometry_summary),
            },
            "output_dir": str(output_dir),
            "artifacts": artifacts,
        },
    )
    artifacts["manifest"] = str(manifest_path)
    return artifacts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build scalar and geometry plots for the pilot report.")
    parser.add_argument("--scalar-gate-summary", type=Path, default=DEFAULT_SCALAR_GATE_SUMMARY)
    parser.add_argument("--geometry-summary", type=Path, default=DEFAULT_GEOMETRY_SUMMARY)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    artifacts = build_plots(
        scalar_gate_summary=args.scalar_gate_summary,
        geometry_summary=args.geometry_summary,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "artifacts": artifacts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
