"""Interactive Plotly rendering for BedSpread, designed for use inside marimo.

Reuses the existing data layer (``SparseGraphMatrix`` from ``bedspread.core``).
The classic ``bedspread_plot_fast`` rasterizes the graph into an
``(n_paths x bp_resolution)`` image and draws it with a single matplotlib
``imshow``. This module produces the same raster but renders it as a Plotly
``go.Heatmap`` so the plot gains native zoom / pan / hover, and exposes helpers
to turn a selected genomic window into nodes and per-path sequences.

Public API
----------
rasterize(sgm, paths=None, signal_column_value=True, bp_resolution=4096,
          x0_bp=None, x1_bp=None) -> RasterResult
    Build the raster + per-column node map (optionally for a zoomed bp window).
bedspread_plotly(sgm, ...) -> plotly.graph_objects.Figure
    Interactive heatmap (zoom/pan/hover) with the BedSpread color palette.
nodes_in_bp_window(sgm, x0_bp, x1_bp) -> list[int]
    Node IDs whose layout span overlaps [x0_bp, x1_bp).
extract_sequences(g, sgm, node_ids, paths=None) -> (nodes_df, path_seq_df)
    Per-node sequences + per-path concatenated sequence over the selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Color palette — mirrors bedspread_plot_fast (white -> grey for presence,
# pink -> dark red for signal), with a "two-stage" mapping that keeps grey
# (presence, value == 1) visible regardless of how large the max signal is.
# ---------------------------------------------------------------------------
GREY_FRAC = 0.15
_PALETTE = [
    (0.0, "white"),
    (GREY_FRAC, "#d4d4d4"),
    (GREY_FRAC + 0.01, "#ffb6c1"),
    (GREY_FRAC + (1 - GREY_FRAC) * 0.33, "#ff6699"),
    (GREY_FRAC + (1 - GREY_FRAC) * 0.67, "#cc0000"),
    (1.0, "#800000"),
]
PLOTLY_COLORSCALE = [[pos, color] for pos, color in _PALETTE]


def _two_stage_norm(values: np.ndarray, vmax: float) -> np.ndarray:
    """Compress 0..1 into [0, GREY_FRAC]; stretch 1..vmax into [GREY_FRAC, 1]."""
    v = np.asarray(values, dtype=float)
    if vmax <= 1:
        return np.clip(v * GREY_FRAC, 0, 1)
    out = np.where(
        v <= 1.0,
        v * GREY_FRAC,
        GREY_FRAC + (v - 1.0) / (vmax - 1.0) * (1.0 - GREY_FRAC),
    )
    return np.clip(out, 0, 1)


def _binary_norm(values: np.ndarray) -> np.ndarray:
    """Flatten signal into 3 levels: absent -> 0, present -> GREY_FRAC, peak -> 1."""
    v = np.asarray(values, dtype=float)
    return np.where(v > 1.0, 1.0, np.where(v >= 1.0, GREY_FRAC, 0.0))


@dataclass
class RasterResult:
    raster_raw: np.ndarray      # (n_paths, W) raw matrix values (0 / 1 / 1+signal)
    z_display: np.ndarray       # (n_paths, W) two-stage-normalized into [0, 1]
    node_at_col: np.ndarray     # (W,) node_id occupying each column (-1 if none)
    x_centers: np.ndarray       # (W,) bp coordinate at each column center
    paths: List[str]            # path names, top-to-bottom row order
    vmax: float
    x0_bp: float
    x1_bp: float


def _resolve_paths(sgm, paths: Optional[List[str]]) -> List[str]:
    if paths is None:
        return list(sgm.path_names)
    return [p for p in paths if p in sgm.path_to_idx]


def rasterize(
    sgm,
    paths: Optional[List[str]] = None,
    bp_resolution: int = 4096,
    x0_bp: Optional[float] = None,
    x1_bp: Optional[float] = None,
    min_signal: float = 0.0,
    binarize: bool = False,
) -> RasterResult:
    """Rasterize the graph into an (n_paths x bp_resolution) image.

    If ``x0_bp``/``x1_bp`` are given, only that genomic window is rasterized at
    full ``bp_resolution`` — this is how zoom re-rendering stays crisp.
    """
    paths = _resolve_paths(sgm, paths)
    path_idx = [sgm.path_to_idx[p] for p in paths]
    n_paths = len(paths)

    total_bp = max(v["x_end"] for v in sgm.node_to_layout.values())
    x0 = 0.0 if x0_bp is None else max(0.0, float(x0_bp))
    x1 = float(total_bp) if x1_bp is None else min(float(total_bp), float(x1_bp))
    if x1 <= x0:
        x0, x1 = 0.0, float(total_bp)
    span = x1 - x0

    W = int(bp_resolution)
    raster = np.zeros((n_paths, W), dtype=np.float32)
    node_at_col = np.full(W, -1, dtype=np.int64)

    # dense submatrix for the selected paths (nodes x n_paths)
    data = sgm.matrix[:, path_idx].toarray()  # (n_nodes, n_paths)

    for i, nid in enumerate(sgm.node_ids):
        layout = sgm.node_to_layout[nid]
        # map node bp span into the [x0, x1) window -> columns
        cs = int((layout["x_start"] - x0) / span * W)
        ce = int((layout["x_end"] - x0) / span * W)
        if ce <= 0 or cs >= W:
            continue
        cs = max(cs, 0)
        ce = min(max(ce, cs + 1), W)
        vals = data[i, :]                       # (n_paths,)
        np.maximum(raster[:, cs:ce], vals[:, None], out=raster[:, cs:ce])
        node_at_col[cs:ce] = nid                # last writer wins (fine for hover)

    if min_signal > 0:
        raster = np.where((raster > 1.0) & ((raster - 1.0) < min_signal), 1.0, raster)

    vmax = float(raster.max()) if raster.size else 1.0
    z_display = _binary_norm(raster) if binarize else _two_stage_norm(raster, vmax)
    x_centers = x0 + (np.arange(W) + 0.5) / W * span

    return RasterResult(
        raster_raw=raster,
        z_display=z_display,
        node_at_col=node_at_col,
        x_centers=x_centers,
        paths=paths,
        vmax=vmax,
        x0_bp=x0,
        x1_bp=x1,
    )


def bedspread_plotly(
    sgm,
    paths: Optional[List[str]] = None,
    bp_resolution: int = 4096,
    x0_bp: Optional[float] = None,
    x1_bp: Optional[float] = None,
    min_signal: float = 0.0,
    binarize: bool = False,
    highlight_nodes: Optional[set] = None,
    title: str = "BedSpread",
    height: Optional[int] = None,
) -> go.Figure:
    """Interactive BedSpread heatmap (rows = paths, x = graph bp, color = signal).

    Drag to box-zoom (genomic window), hover for node/signal, double-click to
    reset. ``binarize`` flattens the signal gradient to 3 flat colors (white /
    grey / red) instead of shading red by signal strength. ``highlight_nodes``
    outlines columns belonging to the given node IDs.
    """
    rr = rasterize(sgm, paths=paths, bp_resolution=bp_resolution,
                   x0_bp=x0_bp, x1_bp=x1_bp, min_signal=min_signal, binarize=binarize)
    n_paths = len(rr.paths)

    # customdata: node id + raw signal per cell, for the hover tooltip
    node_row = rr.node_at_col[None, :].repeat(n_paths, axis=0)
    customdata = np.dstack([node_row, rr.raster_raw])  # (n_paths, W, 2)

    short_paths = [p.split(":")[0] for p in rr.paths]
    hovertemplate = (
        "path: %{y}<br>"
        "bp: %{x:,.0f}<br>"
        "node: %{customdata[0]}<br>"
        "value: %{customdata[1]:.2f}"
        "<extra></extra>"
    )

    heat = go.Heatmap(
        z=rr.z_display,
        x=rr.x_centers,
        y=short_paths,
        customdata=customdata,
        colorscale=PLOTLY_COLORSCALE,
        zmin=0.0,
        zmax=1.0,
        showscale=False,
        hovertemplate=hovertemplate,
        xgap=0,
        ygap=2,
    )
    fig = go.Figure(data=[heat])

    # outline highlighted nodes as thin vertical bands
    if highlight_nodes:
        for nid in highlight_nodes:
            lay = sgm.node_to_layout.get(int(nid))
            if not lay:
                continue
            if lay["x_end"] <= rr.x0_bp or lay["x_start"] >= rr.x1_bp:
                continue
            fig.add_vrect(
                x0=lay["x_start"], x1=lay["x_end"],
                line_width=0, fillcolor="#7b2cbf", opacity=0.25, layer="above",
            )

    if height is None:
        height = max(140, 40 + n_paths * 26)
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=40, b=30),
        dragmode="zoom",
        xaxis=dict(title="graph position (bp)", showgrid=False,
                   range=[rr.x0_bp, rr.x1_bp], constrain="domain"),
        yaxis=dict(showgrid=False, autorange="reversed"),
        plot_bgcolor="white",
    )
    return fig


def nodes_in_bp_window(sgm, x0_bp: float, x1_bp: float) -> List[int]:
    """Return node IDs whose layout span overlaps [x0_bp, x1_bp)."""
    x0, x1 = float(min(x0_bp, x1_bp)), float(max(x0_bp, x1_bp))
    out = []
    for nid, lay in sgm.node_to_layout.items():
        if lay["x_end"] > x0 and lay["x_start"] < x1:
            out.append(int(nid))
    out.sort(key=lambda n: sgm.node_to_layout[n]["x_start"])
    return out


def extract_sequences(
    g,
    sgm,
    node_ids: List[int],
    paths: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-node sequences + per-path concatenated sequence over the selection.

    Returns
    -------
    nodes_df : one row per selected node (node_id, x_start, x_end, length,
               sequence, and presence/value per path).
    path_seq_df : one row per path, the sequence formed by concatenating the
               selected nodes the path actually traverses, in path order.
    """
    paths = _resolve_paths(sgm, paths)
    sel = sorted(
        (int(n) for n in node_ids if int(n) in sgm.node_to_idx),
        key=lambda n: sgm.node_to_layout[n]["x_start"],
    )

    # node-level table
    rows = []
    node_seq: Dict[int, str] = {}
    for nid in sel:
        h = g.get_handle(nid, False)
        seq = g.get_sequence(h)
        node_seq[nid] = seq
        lay = sgm.node_to_layout[nid]
        row = {
            "node_id": nid,
            "x_start": lay["x_start"],
            "x_end": lay["x_end"],
            "length": lay["length"],
            "sequence": seq,
        }
        ridx = sgm.node_to_idx[nid]
        for p in paths:
            row[p.split(":")[0]] = float(sgm.matrix[ridx, sgm.path_to_idx[p]])
        rows.append(row)
    nodes_df = pd.DataFrame(rows)

    # per-path concatenated sequence over the selection (in path traversal order)
    sel_set = set(sel)
    seq_rows = []
    for p in paths:
        ordered = sgm.path_to_nodes.get(p, [])
        orients = sgm.path_to_node_orientations.get(p, [None] * len(ordered))
        pieces, used = [], 0
        for nid, is_rev in zip(ordered, orients):
            if nid not in sel_set:
                continue
            s = node_seq.get(nid)
            if s is None:
                h = g.get_handle(nid, False)
                s = g.get_sequence(h)
                node_seq[nid] = s
            if is_rev:
                s = _revcomp(s)
            pieces.append(s)
            used += 1
        joined = "".join(pieces)
        seq_rows.append({
            "path": p,
            "n_selected_nodes": used,
            "length_bp": len(joined),
            "sequence": joined,
        })
    path_seq_df = pd.DataFrame(seq_rows)
    return nodes_df, path_seq_df


_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


def save_upload(name: str, data: bytes, dest_dir: str = "/tmp/bs_uploads") -> str:
    """Write uploaded file bytes to disk and return the path (for odgi/ingest)."""
    import os
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, os.path.basename(name))
    with open(path, "wb") as fh:
        fh.write(data)
    return path
