# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "bedspread",
#   "pandas",
# ]
# ///
# Note: odgi is a C++ extension that must be installed in the environment
# (e.g., via bioconda: conda install -c bioconda odgi). It is NOT pip-installable.

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    import sys, subprocess
    # odgi must be installed in the environment (not pip-installable)
    try:
        import odgi  # noqa: F401
        odgi_ok = True
    except ImportError:
        odgi_ok = False
    return (odgi_ok,)


@app.cell(hide_code=True)
def _():
    from bedspread import list_paths, build_sparse_matrix, save_sgm_viewer
    import pandas as pd
    return (build_sparse_matrix, list_paths, pd, save_sgm_viewer)


@app.cell
def _(mo, odgi_ok):
    mo.callout(
        mo.md("**odgi not found.** Install it via bioconda: `conda install -c bioconda odgi`"),
        kind="danger",
    ) if not odgi_ok else mo.callout(
        mo.md("odgi ready."),
        kind="success",
    )
    return ()


@app.cell
def _(mo):
    graph_file = mo.ui.file(filetypes=[".og"], label="Upload pangenome graph (.og)")
    graph_file
    return (graph_file,)


@app.cell(hide_code=True)
def _(graph_file, list_paths, mo, odgi_ok):
    import tempfile, pathlib, os
    mo.stop(not graph_file.value, mo.md("⬆️ Upload an `.og` graph file above."))
    mo.stop(not odgi_ok, mo.md("⚠️ odgi not available — cannot load graph."))

    _f = graph_file.value[0]
    _tmp = pathlib.Path(tempfile.mkdtemp()) / _f.name
    _tmp.write_bytes(_f.contents)

    import odgi as _odgi
    graph = _odgi.graph()
    graph.load(str(_tmp))
    path_names = list_paths(graph, verbose=False)
    mo.md(f"Loaded **{_f.name}** — **{len(path_names)} paths**, {graph.get_node_count():,} nodes.")
    return (graph, path_names, pathlib, tempfile)


@app.cell(hide_code=True)
def _(build_sparse_matrix, graph, graph_file, mo, path_names):
    mo.stop(not graph_file.value)
    sgm_base = build_sparse_matrix(
        graph, path_names=path_names,
        peaks_df=None, use_signal=False, verbose=True,
    )
    mo.md(
        f"Built presence matrix: **{sgm_base.shape[0]:,} nodes × {sgm_base.shape[1]} paths** · "
        f"{sgm_base.matrix.nnz:,} non-zero cells."
    )
    return (sgm_base,)


@app.cell
def _(graph_file, mo, pathlib, save_sgm_viewer, sgm_base, tempfile):
    import io
    mo.stop(not graph_file.value)

    _stem = pathlib.Path(graph_file.value[0].name).stem
    _buf = io.BytesIO()
    # save_sgm_viewer saves to a file path; use a temp file then read back
    _out = pathlib.Path(tempfile.mkdtemp()) / f"{_stem}_viewer.npz"
    save_sgm_viewer(sgm_base, str(_out))

    mo.vstack([
        mo.md(f"Saved **{_out.name}** ({_out.stat().st_size / 1024**2:.1f} MB). Download below:"),
        mo.download(data=_out.read_bytes(), filename=_out.name, mimetype="application/octet-stream"),
    ])
    return (io,)


@app.cell
def _(mo):
    mo.md("""
## Next steps

1. Download the `.npz` file above.
2. Open **bedspread-viewer** (GitHub Pages or local `marimo run`).
3. Upload the `.npz` + your `.bed` peak files.

The `.npz` encodes the pangenome graph topology (node layout, path traversals)
without the original `.og` binary — no odgi needed in the viewer.
""")
    return ()


if __name__ == "__main__":
    app.run()
