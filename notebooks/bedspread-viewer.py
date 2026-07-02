# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "https://broadinstitute.github.io/bedspread/bedspread-0.1.1-py3-none-any.whl",
#   "pandas",
#   "plotly",
#   "scipy",
#   "numpy",
#   "intervaltree",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def bs_setup():
    import pandas as pd
    import plotly.graph_objects as go
    from bedspread import load_sgm_viewer, ingest_peak_bed_list, overlay_peaks
    import bedspread.interactive as bi
    import bedspread.screening as bss

    return bi, bss, go, ingest_peak_bed_list, load_sgm_viewer, overlay_peaks, pd


@app.cell
def bs_md(mo):
    mo.md("""
    # BedSpread Viewer

    Upload a pre-computed graph (`.npz` from **bedspread-prep**) and one or more `.bed` peak files.
    """)
    return


@app.cell
def bs_upload_npz(mo):
    bs_npz_file = mo.ui.file(filetypes=[".npz"], label="Upload graph .npz (from bedspread-prep)")
    bs_npz_file
    return (bs_npz_file,)


@app.cell
def bs_load_npz(bs_npz_file, load_sgm_viewer, mo):
    import tempfile, pathlib
    mo.stop(not bs_npz_file.value, mo.md("Upload a `.npz` graph file above."))
    _f = bs_npz_file.value[0]
    _tmp = pathlib.Path(tempfile.mkdtemp()) / _f.name
    _tmp.write_bytes(_f.contents)
    bs_sgm_base = load_sgm_viewer(str(_tmp))
    bs_path_names = bs_sgm_base.path_names
    mo.md(f"Loaded **{_f.name}** — **{len(bs_path_names)} paths**, {bs_sgm_base.shape[0]:,} nodes.")
    return bs_sgm_base, bs_path_names


@app.cell
def bs_has_header(mo):
    bs_has_header = mo.ui.checkbox(label="BED files have a header row", value=False)
    bs_has_header
    return (bs_has_header,)


@app.cell
def bs_upload_beds(mo):
    bs_bed_files = mo.ui.file(
        filetypes=[".bed", ".narrowPeak", ".broadPeak"],
        multiple=True,
        label="Upload peak BED files",
    )
    bs_bed_files
    return (bs_bed_files,)


@app.cell
def bs_ingest(bs_bed_files, bs_has_header, bs_path_names, ingest_peak_bed_list, mo, pd):
    import tempfile, pathlib
    if not bs_bed_files.value:
        bs_peaks_df = pd.DataFrame()
        _msg = mo.callout(mo.md("Upload `.bed` peak files above to overlay peaks on the graph."), kind="info")
    else:
        _dfs = []
        _report = {"total_peaks": 0, "total_valid": 0}
        for _f in bs_bed_files.value:
            _p_bed = pathlib.Path(tempfile.mkdtemp()) / _f.name
            _p_bed.write_bytes(_f.contents)
            _df, _rep = ingest_peak_bed_list(
                [str(_p_bed)], bs_path_names,
                col_names=["chrom", "start", "end", "score"],
                has_header=bs_has_header.value, verbose=False,
            )
            if len(_df):
                _df["source_bed"] = _f.name
                _dfs.append(_df)
            _report["total_peaks"] += _rep["total_peaks"]
            _report["total_valid"] += _rep["total_valid"]
        bs_peaks_df = pd.concat(_dfs, ignore_index=True) if _dfs else pd.DataFrame()
        _msg = mo.md(
            f"Ingested **{len(bs_peaks_df)}** valid peaks from **{len(bs_bed_files.value)}** BED files "
            f"({_report['total_valid']}/{_report['total_peaks']} rows)."
        )
    _msg
    return (bs_peaks_df,)


@app.cell(hide_code=True)
def bs_signal_col(mo):
    bs_signal_col = mo.ui.dropdown(
        ["score", "signal", "qval"], value="score", label="signal column",
        allow_select_none=False,
    )
    return (bs_signal_col,)


@app.cell(hide_code=True)
def bs_peak_sets(bs_bed_files, mo):
    bs_peak_sets = mo.ui.multiselect(
        options=[f.name for f in bs_bed_files.value] if bs_bed_files.value else [],
        value=[f.name for f in bs_bed_files.value] if bs_bed_files.value else [],
        label="peak sets",
    )
    return (bs_peak_sets,)


@app.cell(hide_code=True)
def bs_build(bs_peak_sets, bs_peaks_df, bs_sgm_base, bs_signal_col, overlay_peaks):
    bs_peaks_used = bs_peaks_df
    if "source_bed" in bs_peaks_df.columns:
        bs_peaks_used = bs_peaks_df[bs_peaks_df["source_bed"].isin(bs_peak_sets.value)]

    bs_sgm = overlay_peaks(
        bs_sgm_base, peaks_df=bs_peaks_used,
        signal_column=bs_signal_col.value, verbose=False,
    )
    bs_build_status = (
        f"**Sparse matrix:** {bs_sgm.shape[0]:,} nodes × {bs_sgm.shape[1]} paths · "
        f"{(bs_sgm.matrix > 1).nnz:,} peak-signal cells "
        f"({len(bs_peaks_used):,} peaks from {len(bs_peak_sets.value)} BED files)"
    )
    return bs_build_status, bs_peaks_used, bs_sgm


@app.cell(hide_code=True)
def bs_nodesets(bs_peaks_used, bs_sgm, bss, pd):
    bs_class_colors = {
        "Path unique": "#7b2cbf",
        "Pangenome conserved": "#2ecc71",
        "Pangenome variable": "#3498db",
        "Structural Variant conserved": "#e74c3c",
        "Structural Variant variable": "#f39c12",
        "Unclassified": "#ffffff",
    }
    _NODESET_COLS = ["peak_id", "origin_path", "local_start", "local_end",
                     "peak_nodeset", "node_count", "node_total_len_bp"]
    _SPANS_COLS = ["peak_id", "origin_path", "source_bed", "bp0", "bp1"]

    if len(bs_peaks_used):
        bs_peak_nodeset_df = bss.build_peak_nodeset_df(bs_sgm, bs_peaks_used, verbose=False)
    else:
        bs_peak_nodeset_df = pd.DataFrame(columns=_NODESET_COLS)

    _lay = bs_sgm.node_to_layout
    _bed_by_pid = bs_peaks_used["source_bed"].to_dict() if "source_bed" in bs_peaks_used.columns else {}
    _rows = []
    for _, _r in bs_peak_nodeset_df.iterrows():
        _nodes = [int(n) for n in _r["peak_nodeset"] if int(n) in _lay]
        if not _nodes:
            continue
        _rows.append({
            "peak_id": int(_r["peak_id"]),
            "origin_path": _r["origin_path"],
            "source_bed": _bed_by_pid.get(int(_r["peak_id"])),
            "bp0": float(min(_lay[n]["x_start"] for n in _nodes)),
            "bp1": float(max(_lay[n]["x_end"] for n in _nodes)),
        })
    bs_peak_spans_all = pd.DataFrame(_rows) if _rows else pd.DataFrame(columns=_SPANS_COLS)
    bs_nodeset_status = f"**{len(bs_peak_nodeset_df):,} peak nodesets** built (fixed across path selection)"

    return bs_class_colors, bs_nodeset_status, bs_peak_nodeset_df, bs_peak_spans_all


@app.cell(hide_code=True)
def bs_display_controls(bs_sgm, mo):
    bs_paths_sel = mo.ui.multiselect(
        options=list(bs_sgm.path_names), value=list(bs_sgm.path_names), label="paths",
    )
    bs_res = mo.ui.dropdown(["1024", "2048", "4096", "8192"], value="4096", label="resolution")
    _vmax = float(max(2.0, bs_sgm.matrix.max()))
    bs_min_signal = mo.ui.slider(0.0, _vmax - 1.0, value=0.0, label="min signal")
    return bs_min_signal, bs_paths_sel, bs_res


@app.cell(hide_code=True)
def bs_track_defs(bs_paths_sel, bs_peaks_used, bs_sgm):
    def _track_defs(peaks, sel_paths):
        _has = "source_bed" in peaks.columns
        _defs = []
        for _p in sel_paths:
            _beds = sorted(peaks[peaks["path"] == _p]["source_bed"].dropna().unique()) if _has else []
            _short = _p.split(":")[0]
            if len(_beds) <= 1:
                _b = _beds[0] if _beds else None
                _defs.append((_p, _b, f"{_short} : {_b}" if _b else _short))
            else:
                for _b in _beds:
                    _defs.append((_p, _b, f"{_short} : {_b}"))
        return _defs
    _sel_paths0 = [p for p in bs_paths_sel.value if p in bs_sgm.path_to_idx]
    bs_base_track_defs = _track_defs(bs_peaks_used, _sel_paths0)
    return (bs_base_track_defs,)


@app.cell(hide_code=True)
def bs_track_order(bs_base_track_defs, mo):
    bs_track_order = mo.ui.multiselect(
        options=[d[2] for d in bs_base_track_defs],
        value=[],
        label="track order (top→bottom; unselected tracks follow)",
    )
    return (bs_track_order,)


@app.cell(hide_code=True)
def bs_classify(bs_paths_sel, bs_peak_nodeset_df, bs_peak_spans_all, bs_sgm, bss, pd):
    _sel_paths = [p for p in bs_paths_sel.value if p in bs_sgm.path_to_idx]

    class _PathSubsetSGM:
        """Lightweight sgm view: same nodes/rows, only the selected path columns."""

    _sub = _PathSubsetSGM()
    _sub.matrix = bs_sgm.matrix[:, [bs_sgm.path_to_idx[p] for p in _sel_paths]].tocsr()
    _sub.node_to_idx = bs_sgm.node_to_idx
    _sub.node_to_layout = bs_sgm.node_to_layout
    _sub.node_ids = bs_sgm.node_ids
    _sub.path_names = _sel_paths
    _sub.path_to_idx = {p: i for i, p in enumerate(_sel_paths)}

    _pnd_sel = bs_peak_nodeset_df[bs_peak_nodeset_df["origin_path"].isin(_sel_paths)]
    if len(_pnd_sel):
        bs_peak_class_df = bss.classify_peak_nodesets(_pnd_sel, _sub, verbose=False)
    else:
        bs_peak_class_df = pd.DataFrame(columns=["peak_id", "peak_class"])

    bs_peak_spans = bs_peak_spans_all.merge(
        bs_peak_class_df[["peak_id", "peak_class"]], on="peak_id", how="inner"
    )

    _counts = bs_peak_class_df["peak_class"].value_counts() if len(bs_peak_class_df) else {}
    bs_class_counts = [(c, int(_counts[c])) for c in bss.TARGET_CLASSES if c in getattr(_counts, "index", [])]
    bs_classify_status = (
        f"**{len(bs_peak_class_df):,} peaks** classified over "
        f"**{len(_sel_paths)}/{len(bs_sgm.path_names)}** selected paths · "
        + " · ".join(f"{c}: **{v}**" for c, v in bs_class_counts)
    )

    return bs_class_counts, bs_classify_status, bs_peak_class_df, bs_peak_spans


@app.cell(hide_code=True)
def bs_tracks(
    bs_base_track_defs,
    bs_build_status,
    bs_classify_status,
    bs_nodeset_status,
    bs_peak_class_df,
    bs_peaks_used,
    bs_sgm,
    bs_signal_col,
    bs_track_order,
    mo,
):
    from scipy.sparse import lil_matrix as _lil
    from intervaltree import IntervalTree as _IT

    def _build_track_sgm(base, peaks, defs, sigcol):
        _tracks = list(defs)
        _mat = _lil((len(base.node_ids), len(_tracks)), dtype="float32")
        _nidx, _lay2 = base.node_to_idx, base.node_to_layout
        for _j, (_p, _b, _l) in enumerate(_tracks):
            _nodes = base.path_to_nodes.get(_p, [])
            for _nid in _nodes:
                _mat[_nidx[_nid], _j] = 1.0
            if _b is None:
                continue
            _sub = peaks[(peaks["path"] == _p) & (peaks["source_bed"] == _b)]
            _tree = _IT()
            for _, _pk in _sub.iterrows():
                _s, _e = int(_pk["local_start"]), int(_pk["local_end"])
                if _e > _s:
                    _tree.addi(_s, _e, float(_pk[sigcol]))
            _off = 0
            for _nid in _nodes:
                _nlen = int(_lay2[_nid]["length"])
                _ov = _tree.overlap(_off, _off + _nlen)
                if _ov:
                    _mat[_nidx[_nid], _j] = max(_mat[_nidx[_nid], _j], 1.0 + max(_iv.data for _iv in _ov))
                _off += _nlen

        class _TrackSGM:
            pass
        _t = _TrackSGM()
        _labels = [x[2] for x in _tracks]
        _t.matrix = _mat.tocsr()
        _t.node_ids = base.node_ids
        _t.node_to_idx = base.node_to_idx
        _t.node_to_layout = base.node_to_layout
        _t.path_names = _labels
        _t.path_to_idx = {_l: _i for _i, _l in enumerate(_labels)}
        _t.path_to_nodes = {_l: base.path_to_nodes[x[0]] for _l, x in zip(_labels, _tracks)}
        _t.path_to_node_orientations = {_l: base.path_to_node_orientations[x[0]] for _l, x in zip(_labels, _tracks)}
        return _t, _tracks

    _by_label = {d[2]: d for d in bs_base_track_defs}
    _chosen = [_by_label[_l] for _l in bs_track_order.value if _l in _by_label]
    _chosen_labels = {d[2] for d in _chosen}
    _ordered_defs = _chosen + [d for d in bs_base_track_defs if d[2] not in _chosen_labels]

    bs_track_sgm, _tr = _build_track_sgm(bs_sgm, bs_peaks_used, _ordered_defs, bs_signal_col.value)
    bs_tracks = [{"path": p, "source_bed": b, "label": l} for (p, b, l) in _tr]
    bs_track_row_of = {(t["path"], t["source_bed"]): i for i, t in enumerate(bs_tracks)}
    _paths_shown = {t["path"] for t in bs_tracks}
    _nsplit = sum(1 for _p in _paths_shown if sum(1 for t in bs_tracks if t["path"] == _p) > 1)

    _tracks_line = (
        f"**{len(bs_tracks)} display tracks** from {len(_paths_shown)} paths"
        + (f" · {_nsplit} path(s) split across multiple BEDs" if _nsplit else " · 1 BED per path (no splits)")
    )
    _summary = (
        f"{len(bs_tracks)} tracks · {len(bs_peak_class_df)} peaks · "
        f"{len(_paths_shown)}/{len(bs_sgm.path_names)} paths · signal: {bs_signal_col.value}"
    )
    mo.accordion({
        _summary: mo.md("\n\n".join([bs_build_status, bs_nodeset_status, bs_classify_status, _tracks_line]))
    })

    return bs_track_row_of, bs_track_sgm, bs_tracks


@app.cell
def bs_plot(
    bi,
    bs_class_bar,
    bs_class_colors,
    bs_min_signal,
    bs_paths_sel,
    bs_peak_spans,
    bs_res,
    bs_search_ivals,
    bs_search_label,
    bs_sgm,
    bs_track_row_of,
    bs_track_sgm,
    bs_tracks,
    go,
    mo,
    pd,
):
    from plotly.subplots import make_subplots

    _paths = [p for p in bs_paths_sel.value if p in bs_sgm.path_to_idx]
    _W = int(bs_res.value)

    _main = bi.bedspread_plotly(bs_track_sgm, bp_resolution=_W, min_signal=bs_min_signal.value)
    _heat = _main.data[0]
    _heat.y = [t["label"] for t in bs_tracks]

    _rr = bi.rasterize(bs_sgm, paths=_paths, bp_resolution=_W, min_signal=bs_min_signal.value)
    _node_cons = (_rr.raster_raw > 0).mean(axis=0)
    _rt = bi.rasterize(bs_track_sgm, bp_resolution=_W, min_signal=bs_min_signal.value)
    _peak_cons = (_rt.raster_raw > 1).mean(axis=0)

    _fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.80, 0.10, 0.10], vertical_spacing=0.02,
    )
    _fig.add_trace(_heat, row=1, col=1)
    _fig.add_trace(go.Heatmap(
        z=[_node_cons], x=_rr.x_centers, colorscale="Blues", zmin=0, zmax=1,
        showscale=False,
        hovertemplate="bp %{x:,.0f}<br>node in %{z:.0%} of paths<extra></extra>",
    ), row=2, col=1)
    _fig.add_trace(go.Heatmap(
        z=[_peak_cons], x=_rt.x_centers, colorscale="Reds", zmin=0, zmax=1,
        showscale=False,
        hovertemplate="bp %{x:,.0f}<br>peak in %{z:.0%} of tracks<extra></extra>",
    ), row=3, col=1)

    _fig.update_yaxes(autorange="reversed", showgrid=False, row=1, col=1)
    _fig.update_yaxes(showticklabels=False, ticks="", title_text="node", row=2, col=1)
    _fig.update_yaxes(showticklabels=False, ticks="", title_text="peak", row=3, col=1)
    _fig.update_xaxes(range=[_rr.x0_bp, _rr.x1_bp], showgrid=False)
    _fig.update_xaxes(title_text="graph position (bp)", row=3, col=1)
    _fig.update_layout(
        title="BedSpread — node conservation over paths · peak conservation over tracks",
        height=max(220, 40 + len(bs_tracks) * 26) + 120,
        margin=dict(l=10, r=10, t=40, b=30),
        dragmode="select",
        plot_bgcolor="white",
    )

    _pts = getattr(bs_class_bar, "points", None) or []
    if not _pts and isinstance(bs_class_bar.value, dict):
        _pts = bs_class_bar.value.get("points", []) or []
    _sel_classes = {p.get("x") for p in _pts if isinstance(p, dict) and p.get("x")}

    _n_hi = 0
    if _sel_classes and len(bs_peak_spans):
        for _, _s in bs_peak_spans[bs_peak_spans["peak_class"].isin(_sel_classes)].iterrows():
            _sb = _s["source_bed"] if pd.notna(_s["source_bed"]) else None
            _i = bs_track_row_of.get((_s["origin_path"], _sb))
            if _i is None:
                continue
            _c = bs_class_colors.get(_s["peak_class"], "#333333")
            _fig.add_shape(
                type="rect", x0=_s["bp0"], x1=_s["bp1"], y0=_i - 0.5, y1=_i + 0.5,
                line=dict(color=_c, width=2), fillcolor=_c, opacity=0.35, layer="above",
                row=1, col=1,
            )
            _n_hi += 1

    _search_bands = []
    if bs_search_ivals:
        _tol = max(1.0, (_rr.x1_bp - _rr.x0_bp) / _W)
        _search_bands = [list(bs_search_ivals[0])]
        for _a, _b in bs_search_ivals[1:]:
            if _a <= _search_bands[-1][1] + _tol:
                _search_bands[-1][1] = max(_search_bands[-1][1], _b)
            else:
                _search_bands.append([_a, _b])
        for _bp0, _bp1 in _search_bands:
            _fig.add_shape(
                type="rect", xref="x", yref="paper", x0=_bp0, x1=_bp1, y0=0, y1=1,
                line=dict(width=0), fillcolor="#f1c40f", opacity=0.30, layer="above",
            )

    bs_plot = mo.ui.plotly(
        _fig,
        config={
            "displaylogo": False,
            "modeBarButtons": [
                ["zoom2d", "pan2d", "select2d", "lasso2d"],
                ["zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
                ["toImage"],
            ],
        },
    )
    _caption = (
        mo.md(f"Highlighting **{_n_hi}** peaks in: " + ", ".join(sorted(_sel_classes)))
        if _sel_classes else
        mo.md("_Select class bar(s) below to highlight peaks._")
    )
    _search_cap = (
        mo.md(f"Region: **{bs_search_label}** → {len(_search_bands)} band(s)")
        if bs_search_ivals else None
    )
    mo.vstack([c for c in (_caption, _search_cap, bs_plot) if c is not None])
    return (bs_plot,)


@app.cell(hide_code=True)
def bs_controls_row(
    bs_min_signal,
    bs_paths_sel,
    bs_peak_sets,
    bs_res,
    bs_signal_col,
    bs_track_order,
    mo,
):
    mo.hstack(
        [bs_signal_col, bs_peak_sets, bs_paths_sel, bs_res, bs_min_signal, bs_track_order],
        justify="start",
        gap=2,
    )
    return


@app.cell
def bs_search_controls(bs_sgm, mo):
    _path_opts = list(bs_sgm.path_names)
    bs_search_form = (
        mo.md("""
        {path}

        {start}  {end}
        """)
        .batch(
            path=mo.ui.dropdown(
                options=_path_opts,
                value=_path_opts[0] if _path_opts else None,
                allow_select_none=False,
                label="path",
            ),
            start=mo.ui.text(placeholder="start bp", label="start"),
            end=mo.ui.text(placeholder="end bp", label="end"),
        )
        .form(
            submit_button_label="Search",
            show_clear_button=True,
            clear_button_label="Clear",
        )
    )
    mo.vstack([
        mo.md("### Pangenomic region search"),
        bs_search_form,
    ])
    return (bs_search_form,)


@app.cell(hide_code=True)
def bs_search(bs_search_form, bs_sgm, mo):
    def _path_offset(_name):
        if ":" in _name:
            _tail = _name.rsplit(":", 1)[1]
            if "-" in _tail:
                try:
                    return int(_tail.split("-", 1)[0])
                except ValueError:
                    return 0
        return 0

    def _resolve_search():
        _v = bs_search_form.value
        if not _v:
            return [], [], None, None
        _p = _v.get("path")
        if not _p or _p not in bs_sgm.path_to_nodes:
            return [], [], None, None
        _s_raw = (_v.get("start") or "").strip()
        _e_raw = (_v.get("end") or "").strip()
        if not _s_raw and not _e_raw:
            return [], [], None, None
        _label = f"{_p} {_s_raw or '·'}–{_e_raw or '·'}"
        try:
            _base = _path_offset(_p)
            _s = (int(_s_raw) - _base) if _s_raw else 0
            _e = (int(_e_raw) - _base) if _e_raw else 10**18
        except ValueError:
            return [], [], "⚠️ start/end must be integers.", _label
        if _e <= _s:
            return [], [], "⚠️ end must be greater than start.", _label
        _lay = bs_sgm.node_to_layout
        _cum, _ivals, _nodes = 0, [], []
        for _n in bs_sgm.path_to_nodes[_p]:
            _n = int(_n)
            _L = _lay.get(_n, {}).get("length", 0)
            if _cum + _L > _s and _cum < _e and _n in _lay:
                _ivals.append((_lay[_n]["x_start"], _lay[_n]["x_end"]))
                _nodes.append(_n)
            _cum += _L
        if not _ivals:
            return [], [], "No nodes found in that interval for this path.", _label
        _ivals.sort()
        _cov = sum(_b - _a for _a, _b in _ivals)
        return _ivals, _nodes, f"\U0001f50e {_label}: {len(_nodes):,} nodes · {_cov:,} bp across graph", _label

    bs_search_ivals, bs_search_nodes, bs_search_msg, bs_search_label = _resolve_search()
    mo.md(bs_search_msg) if bs_search_msg else mo.md("_Pick a path + start/end and click **Search** to highlight a region._")
    return bs_search_ivals, bs_search_label, bs_search_nodes


@app.cell
def bs_selection(
    bi,
    bs_paths_sel,
    bs_peak_class_df,
    bs_peak_nodeset_df,
    bs_peaks_used,
    bs_plot,
    bs_search_label,
    bs_search_nodes,
    bs_sgm,
    mo,
    pd,
):
    _rng = bs_plot.ranges or {}
    _xkey = "x" if "x" in _rng else next((k for k in _rng if k.startswith("x")), None)
    _pts = bs_plot.points or []

    if _xkey is not None:
        _x = _rng[_xkey]
        _x0, _x1 = float(_x[0]), float(_x[1])
        bs_sel_nodes = bi.nodes_in_bp_window(bs_sgm, _x0, _x1)
        _title = f"Selected bp {_x0:,.0f}–{_x1:,.0f}"
    elif _pts:
        _xs = [p["x"] for p in _pts if isinstance(p, dict) and "x" in p]
        if _xs:
            _x0, _x1 = float(min(_xs)), float(max(_xs))
            bs_sel_nodes = bi.nodes_in_bp_window(bs_sgm, _x0, _x1)
            _title = f"Selected bp {_x0:,.0f}–{_x1:,.0f}"
        else:
            bs_sel_nodes = None
    elif bs_search_nodes:
        bs_sel_nodes = bs_search_nodes
        _title = f"Search region: {bs_search_label}"
    else:
        bs_sel_nodes = None

    mo.stop(
        bs_sel_nodes is None,
        mo.md(
            "\U0001f532 **Box- or lasso-select** a region on the plot, or enter a path + "
            "start/end and click **Search** above, to inspect nodes, peaks & sequences."
        ),
    )

    try:
        _nodes_df, bs_pathseq_df = bi.extract_sequences(
            None, bs_sgm, bs_sel_nodes, paths=bs_paths_sel.value,
        )
    except Exception:
        _nodes_df = pd.DataFrame({"node_id": bs_sel_nodes})
        bs_pathseq_df = pd.DataFrame(
            {"note": ["Sequence extraction requires odgi (not available in viewer)"]}
        )

    _sel_set = set(bs_sel_nodes)
    _cls_by_pid = dict(zip(bs_peak_class_df["peak_id"], bs_peak_class_df["peak_class"]))
    _hit_ids = [
        int(_r["peak_id"])
        for _, _r in bs_peak_nodeset_df.iterrows()
        if _r["peak_id"] in _cls_by_pid
        and _sel_set.intersection(int(_n) for _n in _r["peak_nodeset"])
    ]
    _meta = bs_peaks_used.reset_index().rename(columns={"index": "peak_id"})
    _cols = [c for c in ["peak_id", "chrom", "genomic_start", "genomic_end",
                         "source_bed", "score", "signal"] if c in _meta.columns]
    bs_sel_peaks_df = _meta[_meta["peak_id"].isin(_hit_ids)][_cols].copy()
    bs_sel_peaks_df["peak_category"] = bs_sel_peaks_df["peak_id"].map(_cls_by_pid)
    _sort = [c for c in ["chrom", "genomic_start"] if c in bs_sel_peaks_df.columns]
    if _sort:
        bs_sel_peaks_df = bs_sel_peaks_df.sort_values(_sort).reset_index(drop=True)

    mo.vstack([
        mo.md(f"### {_title} — {len(bs_sel_nodes)} nodes · {len(bs_sel_peaks_df)} peaks"),
        mo.ui.table(bs_sel_peaks_df, label="peaks in selection (coords · source · category)"),
        mo.ui.table(bs_pathseq_df, label="per-path sequence over selection"),
    ])
    return


@app.cell
def bs_class_bar(bs_class_colors, bs_class_counts, go, mo):
    if not bs_class_counts:
        bs_class_bar = mo.ui.plotly(go.Figure())
    else:
        _bar_fig = go.Figure(go.Bar(
            x=[c for c, _ in bs_class_counts],
            y=[v for _, v in bs_class_counts],
            marker_color=[bs_class_colors.get(c, "#888") for c, _ in bs_class_counts],
        ))
        _bar_fig.update_layout(
            title="Peak classification (click bar to highlight on plot)",
            height=280, margin=dict(l=10, r=10, t=40, b=80),
            xaxis_tickangle=-30, plot_bgcolor="white",
            dragmode="select",
        )
        bs_class_bar = mo.ui.plotly(_bar_fig, config={"displaylogo": False})
    bs_class_bar
    return (bs_class_bar,)


if __name__ == "__main__":
    app.run()
