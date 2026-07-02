"""Simple BedSpread screening pipeline.

Readable workflow:
1) Build per-peak node sets from sgm + peaks_df.
2) Merge similar node sets by symmetric-difference bp until convergence.
3) Classify nodes and merged sets with node-length weighted voting.
4) Plot BedSpread with a set-class annotation track.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from intervaltree import Interval, IntervalTree

from bedspread.core import SparseGraphMatrix


TARGET_CLASSES = [
    "Path unique",
    "Pangenome conserved",
    "Pangenome variable",
    "Structural Variant conserved",
    "Structural Variant variable",
    "Unclassified",
]


def _build_node_interval_trees(sgm: SparseGraphMatrix) -> Dict[str, IntervalTree]:
    """Build per-path interval trees mapping local offsets to node metadata."""
    node_trees: Dict[str, IntervalTree] = {}
    path_orientations = getattr(sgm, "path_to_node_orientations", {})

    for path_name, node_ids in sgm.path_to_nodes.items():
        tree = IntervalTree()
        orientations = path_orientations.get(path_name, [])
        offset = 0

        for i, node_id in enumerate(node_ids):
            node_len = int(sgm.node_to_layout[node_id]["length"])
            is_reverse = bool(orientations[i]) if i < len(orientations) else False
            if node_len > 0:
                tree.add(
                    Interval(
                        offset,
                        offset + node_len,
                        {"node_id": int(node_id), "is_reverse": is_reverse},
                    )
                )
            offset += node_len

        node_trees[path_name] = tree

    return node_trees


def _classify_from_masks(region_mask: np.ndarray, peak_mask: np.ndarray) -> str:
    """Assign class from per-path region/peak masks."""
    n_total = int(len(region_mask))
    n_region = int(region_mask.sum())
    n_peak_total = int(peak_mask.sum())
    n_peak_in_region = int((region_mask & peak_mask).sum())

    if n_region == 0 or n_peak_total == 0:
        return "Unclassified"
    if n_peak_total == 1:
        return "Path unique"
    if n_region == n_total and n_peak_total == n_total:
        return "Pangenome conserved"
    if n_region == n_total and 0 < n_peak_total < n_total:
        return "Pangenome variable"
    if n_region < n_total and n_peak_in_region == n_region and n_region > 0:
        return "Structural Variant conserved"
    if n_region < n_total and 0 < n_peak_in_region < n_region:
        return "Structural Variant variable"
    return "Unclassified"


def _diff_bp(nodes_a: set, nodes_b: set, node_len_bp: Dict[int, int]) -> int:
    """Symmetric-difference bp between two node sets."""
    diff_nodes = nodes_a.symmetric_difference(nodes_b)
    return int(sum(int(node_len_bp.get(n, 0)) for n in diff_nodes))


def build_peak_nodeset_df(
    sgm: SparseGraphMatrix,
    peaks_df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build one row per peak with overlapping node IDs.

    Required peaks_df columns:
      - path
      - local_start
      - local_end
    """
    required = ["path", "local_start", "local_end"]
    missing = [c for c in required if c not in peaks_df.columns]
    if missing:
        raise ValueError(f"peaks_df missing required columns: {missing}")

    node_trees = _build_node_interval_trees(sgm)
    node_len_bp = {int(n): int(v["length"]) for n, v in sgm.node_to_layout.items()}

    peaks = peaks_df.copy().reset_index(drop=False).rename(columns={"index": "peak_id"})
    optional_cols = [c for c in ["chrom", "score", "signal", "qval"] if c in peaks.columns]

    rows: List[dict] = []
    skipped_no_tree = 0

    for _, row in peaks.iterrows():
        peak_id = int(row["peak_id"])
        origin_path = str(row["path"])
        local_start = int(row["local_start"])
        local_end = int(row["local_end"])

        if origin_path not in node_trees:
            skipped_no_tree += 1
            node_ids: List[int] = []
        else:
            overlaps = node_trees[origin_path].overlap(local_start, local_end)
            node_ids = sorted({int(iv.data["node_id"]) for iv in overlaps})

        total_len = int(sum(node_len_bp.get(n, 0) for n in node_ids))

        out = {
            "peak_id": peak_id,
            "origin_path": origin_path,
            "local_start": local_start,
            "local_end": local_end,
            "peak_nodeset": node_ids,
            "node_count": int(len(node_ids)),
            "node_total_len_bp": total_len,
        }
        for col in optional_cols:
            out[col] = row[col]

        rows.append(out)

    peak_nodeset_df = pd.DataFrame(rows)

    if verbose:
        print("[simple] build_peak_nodeset_df complete:")
        print(f"  input peaks: {len(peaks_df)}")
        print(f"  output rows: {len(peak_nodeset_df)}")
        print(f"  peaks with missing origin path in graph: {skipped_no_tree}")

    return peak_nodeset_df


def merge_peak_nodesets_by_diff_bp(
    peak_nodeset_df: pd.DataFrame,
    sgm: SparseGraphMatrix,
    max_diff_bp: int = 5,
    max_iterations: int = 100,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Merge peak node sets if symmetric-difference bp is < max_diff_bp.

    Repeats merge passes until convergence.
    """
    required = ["peak_id", "peak_nodeset"]
    missing = [c for c in required if c not in peak_nodeset_df.columns]
    if missing:
        raise ValueError(f"peak_nodeset_df missing required columns: {missing}")

    node_len_bp = {int(n): int(v["length"]) for n, v in sgm.node_to_layout.items()}

    current_sets: List[dict] = []
    for _, row in peak_nodeset_df.iterrows():
        peak_id = int(row["peak_id"])
        nodes = set(int(n) for n in row["peak_nodeset"])
        current_sets.append({"nodes": nodes, "member_peak_ids": {peak_id}})

    iteration = 0
    total_merges = 0

    while iteration < max_iterations:
        iteration += 1
        merged_in_iteration = 0
        changed = True

        while changed:
            changed = False
            n_sets = len(current_sets)
            for i in range(n_sets):
                if i >= len(current_sets):
                    break
                for j in range(i + 1, len(current_sets)):
                    diff_bp = _diff_bp(current_sets[i]["nodes"], current_sets[j]["nodes"], node_len_bp)
                    if diff_bp < max_diff_bp:
                        current_sets[i]["nodes"].update(current_sets[j]["nodes"])
                        current_sets[i]["member_peak_ids"].update(current_sets[j]["member_peak_ids"])
                        del current_sets[j]
                        merged_in_iteration += 1
                        total_merges += 1
                        changed = True
                        break
                if changed:
                    break

        if verbose:
            print(
                f"[simple] merge iteration {iteration}: "
                f"merged={merged_in_iteration}, sets_now={len(current_sets)}"
            )

        if merged_in_iteration == 0:
            break

    merged_rows: List[dict] = []
    map_rows: List[dict] = []

    for idx, s in enumerate(current_sets, start=1):
        set_id = f"mset_{idx:06d}"
        merged_nodes = sorted(int(n) for n in s["nodes"])
        member_peaks = sorted(int(p) for p in s["member_peak_ids"])

        set_total_len_bp = int(sum(node_len_bp.get(n, 0) for n in merged_nodes))

        merged_rows.append(
            {
                "set_id": set_id,
                "merged_nodeset": merged_nodes,
                "n_nodes": int(len(merged_nodes)),
                "set_total_len_bp": set_total_len_bp,
                "member_peak_ids": member_peaks,
                "n_member_peaks": int(len(member_peaks)),
            }
        )

        for peak_id in member_peaks:
            map_rows.append({"peak_id": int(peak_id), "set_id": set_id})

    merged_peak_nodeset_df = pd.DataFrame(merged_rows)
    peak_to_set_map_df = pd.DataFrame(map_rows)

    if verbose:
        print("[simple] merge_peak_nodesets_by_diff_bp complete:")
        print(f"  initial sets: {len(peak_nodeset_df)}")
        print(f"  merged sets: {len(merged_peak_nodeset_df)}")
        print(f"  total merges: {total_merges}")
        print(f"  iterations used: {iteration}")

    return merged_peak_nodeset_df, peak_to_set_map_df


def classify_merged_peak_nodesets(
    merged_peak_nodeset_df: pd.DataFrame,
    sgm: SparseGraphMatrix,
    signal_threshold: float = 0.0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Classify nodes and merged sets with node-length weighting."""
    required = ["set_id", "merged_nodeset"]
    missing = [c for c in required if c not in merged_peak_nodeset_df.columns]
    if missing:
        raise ValueError(f"merged_peak_nodeset_df missing required columns: {missing}")

    n_total_paths = int(len(sgm.path_names))

    node_label_rows: List[dict] = []
    set_label_rows: List[dict] = []

    class_precedence = [
        "Path unique",
        "Pangenome conserved",
        "Pangenome variable",
        "Structural Variant conserved",
        "Structural Variant variable",
        "Unclassified",
    ]

    for _, row in merged_peak_nodeset_df.iterrows():
        set_id = str(row["set_id"])
        nodes = [int(n) for n in row["merged_nodeset"]]

        class_bp = {cls: 0 for cls in TARGET_CLASSES}
        set_total_len = 0

        for node_id in nodes:
            if node_id not in sgm.node_to_idx:
                continue

            node_idx = int(sgm.node_to_idx[node_id])
            vals = sgm.matrix[node_idx, :].toarray().ravel()

            region_mask = vals > 0
            peak_mask = np.maximum(0.0, vals - 1.0) > signal_threshold

            node_class = _classify_from_masks(region_mask, peak_mask)
            node_len = int(sgm.node_to_layout[node_id]["length"])

            class_bp[node_class] += node_len
            set_total_len += node_len

            node_label_rows.append(
                {
                    "set_id": set_id,
                    "node_id": int(node_id),
                    "node_len_bp": node_len,
                    "node_class": node_class,
                    "n_paths_total": n_total_paths,
                    "n_paths_region": int(region_mask.sum()),
                    "n_paths_peak_total": int(peak_mask.sum()),
                    "n_paths_peak_in_region": int((region_mask & peak_mask).sum()),
                }
            )

        def _vote_key(cls: str) -> Tuple[int, int]:
            return int(class_bp[cls]), -class_precedence.index(cls)

        set_class = max(class_precedence, key=_vote_key)
        winner_bp = int(class_bp[set_class])
        class_weight_fraction = float(winner_bp / set_total_len) if set_total_len > 0 else float("nan")

        out = {
            "set_id": set_id,
            "set_class": set_class,
            "class_weight_fraction": class_weight_fraction,
            "set_total_len_bp": int(set_total_len),
            "n_nodes": int(len(nodes)),
        }
        for cls in TARGET_CLASSES:
            col = cls.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
            out[f"bp_{col}"] = int(class_bp[cls])

        set_label_rows.append(out)

    node_label_df = pd.DataFrame(node_label_rows)
    set_label_df = pd.DataFrame(set_label_rows)

    if verbose:
        print("[simple] classify_merged_peak_nodesets complete:")
        print(f"  sets classified: {len(set_label_df)}")
        print(f"  node labels: {len(node_label_df)}")

    return node_label_df, set_label_df


def plot_bedspread_with_set_labels(
    sgm: SparseGraphMatrix,
    merged_peak_nodeset_df: pd.DataFrame,
    set_label_df: pd.DataFrame,
    paths: Optional[List[str]] = None,
    bp_resolution: int = 2048,
    figsize: Tuple[float, float] = (16, 4),
    title: str = "BedSpread + Merged Peak Nodeset Classes",
    class_colors: Optional[Dict[str, str]] = None,
):
    """Plot BedSpread heatmap with a second track colored by merged set class."""
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Patch

    if paths is None:
        paths = list(sgm.path_names)

    if class_colors is None:
        class_colors = {
            "Path unique": "#7b2cbf",
            "Pangenome conserved": "#2ecc71",
            "Pangenome variable": "#3498db",
            "Structural Variant conserved": "#e74c3c",
            "Structural Variant variable": "#f39c12",
            "Unclassified": "#ffffff",
        }

    required_merged = ["set_id", "merged_nodeset", "set_total_len_bp"]
    missing_merged = [c for c in required_merged if c not in merged_peak_nodeset_df.columns]
    if missing_merged:
        raise ValueError(f"merged_peak_nodeset_df missing required columns: {missing_merged}")

    required_labels = ["set_id", "set_class", "class_weight_fraction", "set_total_len_bp"]
    missing_labels = [c for c in required_labels if c not in set_label_df.columns]
    if missing_labels:
        raise ValueError(f"set_label_df missing required columns: {missing_labels}")

    label_map = set_label_df.set_index("set_id").to_dict("index")
    node_candidates: Dict[int, List[Tuple[str, float, int]]] = defaultdict(list)

    for _, row in merged_peak_nodeset_df.iterrows():
        set_id = str(row["set_id"])
        info = label_map.get(set_id)
        if info is None:
            continue
        cls = str(info["set_class"])
        frac = float(info["class_weight_fraction"]) if pd.notna(info["class_weight_fraction"]) else -1.0
        total_len = int(info["set_total_len_bp"])

        for node_id in row["merged_nodeset"]:
            node_candidates[int(node_id)].append((cls, frac, total_len))

    node_to_class: Dict[int, str] = {}
    for node_id, candidates in node_candidates.items():
        best = sorted(candidates, key=lambda x: (x[1], x[2]), reverse=True)[0]
        node_to_class[node_id] = best[0]

    path_indices = [sgm.path_to_idx[p] for p in paths if p in sgm.path_to_idx]
    path_names = [p for p in paths if p in sgm.path_to_idx]

    data = sgm.matrix[:, path_indices].toarray().T
    vmax = float(np.nanmax(data)) if data.size > 0 else 1.0

    total_bp = int(max(v["x_end"] for v in sgm.node_to_layout.values()))
    raster = np.zeros((len(path_names), bp_resolution), dtype=np.float32)

    for i, node_id in enumerate(sgm.node_ids):
        layout = sgm.node_to_layout[node_id]
        cs = int(layout["x_start"] * bp_resolution / total_bp)
        ce = int(layout["x_end"] * bp_resolution / total_bp)
        ce = max(cs + 1, ce)
        ce = min(ce, bp_resolution)
        if cs >= bp_resolution:
            continue
        node_vals = data[:, i]
        np.maximum(raster[:, cs:ce], node_vals[:, np.newaxis], out=raster[:, cs:ce])

    grey_frac = 0.15
    positions = [
        0.0, 0.06, grey_frac, grey_frac + 0.01,
        grey_frac + (1 - grey_frac) * 0.33,
        grey_frac + (1 - grey_frac) * 0.67,
        1.0,
    ]
    colors_cmap = ["white", "white", "#d4d4d4", "#ffb6c1", "#ff6699", "#cc0000", "#800000"]
    cmap = LinearSegmentedColormap.from_list("bedspread", list(zip(positions, colors_cmap)), N=256)

    class TwoStageNorm(Normalize):
        def __call__(self, value, clip=None):
            v = np.asarray(value, dtype=float)
            out = np.where(
                v <= 1.0,
                v * grey_frac,
                grey_frac + (v - 1.0) / max(vmax - 1.0, 1e-9) * (1.0 - grey_frac),
            )
            return np.ma.masked_array(np.clip(out, 0, 1))

        def inverse(self, value):
            y = np.asarray(value, dtype=float)
            out = np.where(
                y <= grey_frac,
                y / max(grey_frac, 1e-9),
                1.0 + (y - grey_frac) / max(1.0 - grey_frac, 1e-9) * max(vmax - 1.0, 1e-9),
            )
            return out

    norm = TwoStageNorm(vmin=0, vmax=vmax) if vmax > 1 else Normalize(vmin=0, vmax=1)

    fig, (ax_main, ax_track) = plt.subplots(
        2, 1, figsize=figsize,
        gridspec_kw={"height_ratios": [3, 0.5]},
        sharex=True,
    )

    im = ax_main.imshow(
        raster, aspect="auto", cmap=cmap, norm=norm,
        interpolation="nearest", origin="upper",
        extent=(0, total_bp, len(path_names), 0),
    )
    ax_main.set_yticks(range(len(path_names)))
    ax_main.set_yticklabels(path_names, fontsize=8)
    ax_main.set_ylabel("Paths", fontsize=9)
    ax_main.set_title(title, fontsize=11, fontweight="bold")

    track_raster = np.zeros((1, bp_resolution, 3), dtype=np.uint8)
    for node_id in sgm.node_ids:
        cls = node_to_class.get(int(node_id), "Unclassified")
        rgb = tuple(int(x * 255) for x in mcolors.to_rgb(class_colors.get(cls, "#7f8c8d")))
        layout = sgm.node_to_layout[node_id]
        cs = int(layout["x_start"] * bp_resolution / total_bp)
        ce = int(layout["x_end"] * bp_resolution / total_bp)
        ce = max(cs + 1, ce)
        ce = min(ce, bp_resolution)
        if cs >= bp_resolution:
            continue
        track_raster[0, cs:ce, :] = rgb

    ax_track.imshow(
        track_raster, aspect="auto", origin="upper",
        interpolation="nearest", extent=(0, total_bp, 1, 0),
    )
    ax_track.set_yticks([])
    ax_track.set_ylabel("Class", fontsize=9)
    ax_track.spines[["top", "right", "left"]].set_visible(False)
    ax_track.tick_params(left=False)

    tick_bp = np.linspace(0, total_bp, 6).astype(int)
    ax_track.set_xticks(tick_bp)
    ax_track.set_xticklabels([f"{v:,}" for v in tick_bp], fontsize=8)
    ax_track.set_xlabel("Graph coordinate (bp)", fontsize=9)

    ax_main.set_xlim(0, total_bp)
    ax_track.set_xlim(0, total_bp)

    if vmax > 1:
        fig.subplots_adjust(right=0.88)
        cax = fig.add_axes([0.895, 0.16, 0.012, 0.72])
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.set_ticks([0, 1, 200, 400, 600, 800, 1000, 1200] if vmax >= 1200 else [0, 1, vmax])
        cbar.set_label("Signal intensity", fontsize=9, rotation=270, labelpad=15)

    legend_elements = [
        Patch(facecolor=class_colors["Path unique"], edgecolor="black", label="Path unique"),
        Patch(facecolor=class_colors["Pangenome conserved"], edgecolor="black", label="Pangenome conserved"),
        Patch(facecolor=class_colors["Pangenome variable"], edgecolor="black", label="Pangenome variable"),
        Patch(facecolor=class_colors["Structural Variant conserved"], edgecolor="black", label="SV conserved"),
        Patch(facecolor=class_colors["Structural Variant variable"], edgecolor="black", label="SV variable"),
        Patch(facecolor=class_colors["Unclassified"], edgecolor="black", label="Unclassified"),
    ]
    fig.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(0.95, 0.95), fontsize=9)

    if vmax <= 1:
        plt.tight_layout()
    return fig, (ax_main, ax_track)


def classify_peak_nodesets(
    peak_nodeset_df: pd.DataFrame,
    sgm: SparseGraphMatrix,
    signal_threshold: float = 0.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Classify each raw peak independently by node-length weighted voting.

    For each peak's node set, checks per-node which paths carry the region
    (matrix value > 0) and which carry a peak signal (matrix value > 1 + threshold),
    then votes by bp to assign one of the five consensus classes.

    This classification is entirely independent of the nodeset merging step —
    every input peak gets its own class based solely on its own overlapping nodes.

    Parameters
    ----------
    peak_nodeset_df : pd.DataFrame
        Output of build_peak_nodeset_df. Must have columns:
        ``peak_id``, ``peak_nodeset``, and optionally ``origin_path``, ``score``.
    sgm : SparseGraphMatrix
        The sparse graph matrix with signal values.
    signal_threshold : float
        Minimum signal above 1.0 to count a path as having a peak on a node.
        Default 0.0 means any value > 1 counts.
    verbose : bool
        Print summary on completion.

    Returns
    -------
    pd.DataFrame
        One row per peak with columns:
        ``peak_id``, ``peak_class``, ``class_weight_fraction``,
        ``node_total_len_bp``, ``n_nodes``, and per-class bp breakdown columns.
    """
    required = ["peak_id", "peak_nodeset"]
    missing = [c for c in required if c not in peak_nodeset_df.columns]
    if missing:
        raise ValueError(f"peak_nodeset_df missing required columns: {missing}")

    rows: List[dict] = []

    for _, row in peak_nodeset_df.iterrows():
        peak_id = int(row["peak_id"])
        nodes = [int(n) for n in row["peak_nodeset"]]

        class_bp = {cls: 0 for cls in TARGET_CLASSES}
        total_len = 0

        for node_id in nodes:
            if node_id not in sgm.node_to_idx:
                continue

            node_idx = int(sgm.node_to_idx[node_id])
            vals = sgm.matrix[node_idx, :].toarray().ravel()

            region_mask = vals > 0
            peak_mask = np.maximum(0.0, vals - 1.0) > signal_threshold

            node_class = _classify_from_masks(region_mask, peak_mask)
            node_len = int(sgm.node_to_layout[node_id]["length"])

            class_bp[node_class] += node_len
            total_len += node_len

        class_precedence = [
            "Path unique",
            "Pangenome conserved",
            "Pangenome variable",
            "Structural Variant conserved",
            "Structural Variant variable",
            "Unclassified",
        ]

        def _vote_key(cls: str) -> Tuple[int, int]:
            return int(class_bp[cls]), -class_precedence.index(cls)

        peak_class = max(class_precedence, key=_vote_key)
        winner_bp = int(class_bp[peak_class])
        class_weight_fraction = float(winner_bp / total_len) if total_len > 0 else float("nan")

        out: dict = {
            "peak_id": peak_id,
            "peak_class": peak_class,
            "class_weight_fraction": class_weight_fraction,
            "node_total_len_bp": total_len,
            "n_nodes": len(nodes),
        }
        for col in ["origin_path", "chrom", "local_start", "local_end", "score"]:
            if col in peak_nodeset_df.columns:
                out[col] = row[col]
        for cls in TARGET_CLASSES:
            col = cls.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
            out[f"bp_{col}"] = int(class_bp[cls])

        rows.append(out)

    peak_class_df = pd.DataFrame(rows)

    if verbose:
        print("[simple] classify_peak_nodesets complete:")
        print(f"  peaks classified: {len(peak_class_df)}")
        if not peak_class_df.empty:
            print(peak_class_df["peak_class"].value_counts().to_string())

    return peak_class_df


__all__ = [
    "build_peak_nodeset_df",
    "merge_peak_nodesets_by_diff_bp",
    "classify_merged_peak_nodesets",
    "classify_peak_nodesets",
    "plot_bedspread_with_set_labels",
]
