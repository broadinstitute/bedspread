#!/usr/bin/env python3
"""Convert an odgi pangenome graph (.og) to a BedSpread viewer file (.npz).

Usage
-----
    python bedspread-prep.py graph.og
    python bedspread-prep.py graph.og -o output_viewer.npz

The output .npz encodes node layout, path traversals, and presence/absence
as a sparse matrix — no odgi required to open it in bedspread-viewer.
"""

import argparse
import sys
from pathlib import Path

# Allow running directly from a cloned repo without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Convert .og pangenome graph → BedSpread viewer .npz",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("og_file", help="Input pangenome graph (.og)")
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: <og_file stem>_viewer.npz)",
    )
    args = parser.parse_args()

    og_path = Path(args.og_file)
    if not og_path.exists():
        sys.exit(f"Error: file not found: {og_path}")

    out_path = Path(args.output) if args.output else og_path.with_name(og_path.stem + "_viewer.npz")

    try:
        import odgi
    except ImportError:
        sys.exit(
            "Error: odgi is not installed.\n"
            "Install via bioconda:  conda install -c bioconda odgi"
        )

    try:
        from bedspread import list_paths, build_sparse_matrix, save_sgm_viewer
    except ImportError:
        sys.exit(
            "Error: could not import bedspread.\n"
            "Run this script from the cloned repo root, or:\n"
            "  pip install -e /path/to/bedspread_pkg"
        )

    print(f"Loading {og_path.name} ...", flush=True)
    g = odgi.graph()
    g.load(str(og_path))
    path_names = list_paths(g, verbose=False)
    print(f"  {len(path_names)} paths, {g.get_node_count():,} nodes")

    print("Building sparse matrix ...", flush=True)
    sgm = build_sparse_matrix(
        g, path_names,
        peaks_df=None, use_signal=False, verbose=True,
    )

    print(f"Saving {out_path} ...", flush=True)
    save_sgm_viewer(sgm, str(out_path))
    size_mb = out_path.stat().st_size / 1024 ** 2
    print(f"Done. {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
