from __future__ import annotations

import argparse
import csv
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
    "assertive_deferential": "assertive_deferential",
    "calm_anxious": "calm_anxious",
    "cautious_adventurous": "cautious_adventurous",
    "diplomacy_bluntness": "diplomacy_bluntness",
    "empathy_detachment": "empathy_detachment",
    "skeptical_naive": "skeptical_naive",
}

ROLE_ORDER = ["counselor", "tutor", "debugger", "journalist", "doctor", "strategist"]
TRAIT_ORDER = [
    "warmth_coldness",
    "sincerity_manipulativeness",
    "caution_recklessness",
    "curiosity_closed_mindedness",
    "skepticism_gullibility",
    "assertive_deferential",
    "calm_anxious",
    "cautious_adventurous",
    "diplomacy_bluntness",
    "empathy_detachment",
    "skeptical_naive",
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


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def pca_rows_from_payload(geometry_payload: dict[str, Any], geometry_summary: Path) -> list[dict[str, Any]]:
    rows = list(geometry_payload.get("pca_summary") or [])
    if rows:
        return rows
    pca_csv = geometry_summary.parent / "pca_summary.csv"
    if pca_csv.exists():
        return read_csv(pca_csv)
    return []


def save_pca_variance_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    plt, np = load_plot_deps()
    all_rows = [
        row
        for row in rows
        if row.get("scope") == "all_trait_role_vectors"
    ]
    if not all_rows:
        return
    vector_types = [str(row["vector_type"]) for row in all_rows]
    pc1 = [number(row.get("pc1_explained_variance")) or 0.0 for row in all_rows]
    pc2 = [number(row.get("pc2_explained_variance")) or 0.0 for row in all_rows]
    pc3 = [number(row.get("pc3_explained_variance")) or 0.0 for row in all_rows]

    x = np.arange(len(vector_types))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(7.0, 1.3 * len(vector_types)), 4.4))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")
    ax.bar(x - width, pc1, width, label="PC1", color="#376795", alpha=0.9)
    ax.bar(x, pc2, width, label="PC2", color="#72a1c9", alpha=0.9)
    ax.bar(x + width, pc3, width, label="PC3", color="#d8b365", alpha=0.9)
    ax.set_xticks(x, vector_types, rotation=25, ha="right")
    ax.set_ylabel("explained variance ratio")
    ax.set_ylim(0, max(0.7, max(pc1 + pc2 + pc3) * 1.15))
    ax.set_title("PCA variance explained across all trait-role vectors", fontsize=12, pad=12)
    ax.legend(frameon=False, ncols=3)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_pca_complexity_heatmap(path: Path, rows: list[dict[str, Any]], value_key: str, title: str) -> None:
    within_rows = [
        row
        for row in rows
        if row.get("scope") == "within_trait_roles" and row.get("trait_axis_id")
    ]
    if not within_rows:
        return
    traits = ordered_unique([str(row["trait_axis_id"]) for row in within_rows], TRAIT_ORDER)
    vector_types = ordered_unique([str(row["vector_type"]) for row in within_rows], [])
    by_key = {
        (str(row["trait_axis_id"]), str(row["vector_type"])): number(row.get(value_key))
        for row in within_rows
    }
    matrix = [[by_key.get((trait, vector_type)) for vector_type in vector_types] for trait in traits]
    save_heatmap(
        path=path,
        matrix=matrix,
        x_labels=vector_types,
        y_labels=[trait_label(trait) for trait in traits],
        title=title,
        colorbar_label=value_key,
        center_zero=False,
    )


def load_torch_payload(path: Path) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PCA scatter plots require torch. Install with: pip install torch") from exc

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a dictionary payload")
    return payload


def get_layer_map(payload: dict[str, Any], key: str, layer: int) -> dict[str, Any]:
    by_layer = payload[key]
    result = by_layer.get(layer) or by_layer.get(str(layer))
    if not isinstance(result, dict):
        raise ValueError(f"layer {layer} missing from payload key {key!r}")
    return result


def pca_coordinates(vectors: list[Any]) -> tuple[Any, list[float]]:
    import torch

    matrix = torch.stack(vectors, dim=0).float()
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    coords = centered @ vh[:2].T
    variances = singular_values.pow(2)
    total = float(variances.sum())
    ratios = (variances / total).tolist() if total > 0.0 else []
    return coords, [float(value) for value in ratios[:2]]


def collect_vectors_for_pca_scatter(
    geometry_payload: dict[str, Any],
    vector_type: str,
) -> tuple[list[dict[str, str]], list[Any]]:
    layer = int(geometry_payload.get("layer", 8))
    points: list[dict[str, str]] = []
    vectors: list[Any] = []
    for resolved in geometry_payload.get("resolved_inputs") or []:
        trait_axis_id = str(resolved["trait_axis_id"])
        payload = load_torch_payload(Path(str(resolved["role_trait_vectors"])))
        vectors_by_role = get_layer_map(payload, "role_trait_vectors", layer)
        roles = resolved.get("roles") or sorted(vectors_by_role)
        for role_id in roles:
            if role_id not in vectors_by_role:
                continue
            role_vectors = vectors_by_role[role_id]
            if vector_type not in role_vectors:
                continue
            points.append({"trait_axis_id": trait_axis_id, "role_id": str(role_id)})
            vectors.append(role_vectors[vector_type])
    return points, vectors


def save_pca_scatter(path: Path, geometry_payload: dict[str, Any], vector_type: str) -> None:
    plt, np = load_plot_deps()
    points, vectors = collect_vectors_for_pca_scatter(geometry_payload, vector_type)
    if len(vectors) < 3:
        return
    coords, ratios = pca_coordinates(vectors)
    coords_np = coords.detach().cpu().numpy()
    traits = ordered_unique([point["trait_axis_id"] for point in points], TRAIT_ORDER)
    roles = ordered_unique([point["role_id"] for point in points], ROLE_ORDER)
    cmap = plt.get_cmap("tab10")
    trait_colors = {trait: cmap(index % 10) for index, trait in enumerate(traits)}
    role_markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    role_to_marker = {role: role_markers[index % len(role_markers)] for index, role in enumerate(roles)}

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")
    for idx, point in enumerate(points):
        ax.scatter(
            coords_np[idx, 0],
            coords_np[idx, 1],
            color=trait_colors[point["trait_axis_id"]],
            marker=role_to_marker[point["role_id"]],
            s=80,
            alpha=0.85,
            edgecolor="#ffffff",
            linewidth=0.7,
        )
    ax.axhline(0, color="#bbbbbb", linewidth=0.8)
    ax.axvline(0, color="#bbbbbb", linewidth=0.8)
    pc1_label = f"PC1 ({ratios[0]:.1%})" if len(ratios) > 0 else "PC1"
    pc2_label = f"PC2 ({ratios[1]:.1%})" if len(ratios) > 1 else "PC2"
    ax.set_xlabel(pc1_label)
    ax.set_ylabel(pc2_label)
    ax.set_title(f"PCA scatter for {vector_type}", fontsize=12, pad=12)
    ax.grid(color="#dddddd", linewidth=0.8, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)

    trait_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=trait_colors[trait], label=trait_label(trait), markersize=7)
        for trait in traits
    ]
    role_handles = [
        plt.Line2D([0], [0], marker=role_to_marker[role], linestyle="", color="#555555", label=role, markersize=7)
        for role in roles
    ]
    first_legend = ax.legend(handles=trait_handles, title="trait", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    ax.add_artist(first_legend)
    ax.legend(handles=role_handles, title="role", loc="lower left", bbox_to_anchor=(1.02, 0.0), frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


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

    pca_rows = pca_rows_from_payload(geometry_payload, geometry_summary)
    if pca_rows:
        pca_variance_path = output_dir / "pca_all_vector_variance.png"
        save_pca_variance_plot(pca_variance_path, pca_rows)
        if pca_variance_path.exists():
            artifacts["pca_all_vector_variance"] = str(pca_variance_path)

        pca_pc1_path = output_dir / "pca_within_trait_pc1_heatmap.png"
        save_pca_complexity_heatmap(
            pca_pc1_path,
            pca_rows,
            value_key="pc1_explained_variance",
            title="Within-trait role PCA: PC1 explained variance",
        )
        if pca_pc1_path.exists():
            artifacts["pca_within_trait_pc1_heatmap"] = str(pca_pc1_path)

        pca_pcs90_path = output_dir / "pca_within_trait_pcs90_heatmap.png"
        save_pca_complexity_heatmap(
            pca_pcs90_path,
            pca_rows,
            value_key="pcs_for_90pct",
            title="Within-trait role PCA: PCs needed for 90% variance",
        )
        if pca_pcs90_path.exists():
            artifacts["pca_within_trait_pcs90_heatmap"] = str(pca_pcs90_path)

    for vector_type in ["axis_vector", "positive_shift", "negative_shift", "offset_vector"]:
        try:
            pca_scatter_path = output_dir / f"pca_scatter_{vector_type}.png"
            save_pca_scatter(pca_scatter_path, geometry_payload, vector_type)
            if pca_scatter_path.exists():
                artifacts[f"pca_scatter_{vector_type}"] = str(pca_scatter_path)
        except RuntimeError:
            # Keep the plot pack usable in lightweight environments without torch.
            continue

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
