"""
BedSpread core — graph loading, peak ingestion, and sparse matrix construction.

Requires odgi to be loaded before importing this module. In your notebook:

    import ctypes, os, sys
    libdir = os.path.abspath("lib")
    sys.path.insert(0, libdir)
    ctypes.CDLL("/opt/homebrew/opt/libomp/lib/libomp.dylib", mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(os.path.join(libdir, "libodgi.dylib"), mode=ctypes.RTLD_GLOBAL)
    import odgi

    from bedspread import list_paths, ingest_peak_bed_list, build_sparse_matrix
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

from tqdm.auto import tqdm
from intervaltree import IntervalTree, Interval


# ============================================================================
# GRAPH HELPER FUNCTIONS
# ============================================================================

def list_paths(g, max_show=50, verbose=True):
    """List all path names in an ODGI graph.

    Args:
        g: odgi.graph object
        max_show: Maximum paths to print (0 for none, -1 for all)
        verbose: Print output

    Returns:
        List of path name strings
    """
    names = []

    def _collect(ph):
        names.append(g.get_path_name(ph))
        return True

    g.for_each_path_handle(_collect)

    if verbose and max_show != 0:
        print(f"Found {len(names)} paths")
        show_count = len(names) if max_show == -1 else min(max_show, len(names))
        for n in names[:show_count]:
            print(" -", n)
        if max_show > 0 and len(names) > max_show:
            print(f" ... and {len(names) - max_show} more")

    return names


def clean_path_name(path_name: str) -> str:
    """Remove coordinate suffix from path name (e.g., 'MOU#1#OK649234.1:123-456' -> 'MOU#1#OK649234.1')."""
    return path_name.split(':')[0]


def extract_path_offset(path_name: str) -> Optional[Tuple[int, int]]:
    """Extract offset coordinates from path name.

    Args:
        path_name: Path name like 'MOU#1#OK649234.1:4109765-4191507'

    Returns:
        Tuple of (start_offset, end_offset) or None if no offset found
    """
    if ':' not in path_name:
        return None

    suffix = path_name.split(':')[1]
    if '-' not in suffix:
        return None

    try:
        start, end = suffix.split('-')
        return int(start.replace(",", "")), int(end.replace(",", ""))
    except (ValueError, IndexError):
        return None


# ============================================================================
# BED FILE PARSING WITH VALIDATION
# ============================================================================

def detect_bed_format(filepath: str) -> Dict:
    """Auto-detect BED file format and column count.

    Returns:
        Dict with 'n_cols', 'has_header', 'format' (bed3/bed6/narrowPeak/etc)
    """
    with open(filepath, 'r') as f:
        # Skip comment lines
        for line in f:
            if not line.startswith('#') and line.strip():
                n_cols = len(line.strip().split('\t'))
                break

    format_map = {
        3: 'bed3',
        4: 'bed4',
        5: 'bed5',
        6: 'bed6',
        10: 'narrowPeak',
        12: 'bed12'
    }

    return {
        'n_cols': n_cols,
        'format': format_map.get(n_cols, f'bed{n_cols}')
    }


def parse_regionpeak(regionpeak_file: str,
                     graph_paths: List[str],
                     col_names: Optional[List[str]] = None,
                     has_header: bool = False,
                     verbose: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """Parse BED/narrowPeak file with robust validation and coordinate adjustment.

    Args:
        regionpeak_file: Path to BED file
        graph_paths: List of valid path names from the graph
        col_names: Custom column names (must start with chrom, start, end).
                   If None, auto-detected from column count.
        has_header: If True, skip the first non-comment row as a header
        verbose: Print detailed warnings

    Returns:
        Tuple of (DataFrame with adjusted coordinates, validation_report dict)
    """
    # Detect format
    format_info = detect_bed_format(regionpeak_file)
    n_cols = format_info['n_cols']

    # Define column names
    if col_names is not None:
        cols = col_names
    elif n_cols >= 10:  # narrowPeak
        cols = ["chrom", "start", "end", "name", "score", "strand",
                "signal", "pval", "qval", "peak"][:n_cols]
    elif n_cols >= 6:  # bed6+
        cols = ["chrom", "start", "end", "name", "score", "strand"][:n_cols]
    else:  # bed3-5
        cols = ["chrom", "start", "end", "name", "score"][:n_cols]

    # Load file (don't use comment='#' - it strips # from data like MOU#1#OK649234.1!)
    df = pd.read_csv(regionpeak_file, sep="\t", header=None, names=cols,
                     skiprows=1 if has_header else 0)

    if verbose:
        print(f"Loaded {len(df)} peaks from {regionpeak_file} (format: {format_info['format']})")

    # Create path mapping (cleaned name -> full name with coords)
    # Only use paths that have coordinate suffix (contain ':')
    path_map = {clean_path_name(p): p for p in graph_paths if ':' in p}

    n_total = len(df)
    n_skipped = 0
    adjusted_rows = []

    for idx, row in df.iterrows():
        chrom = str(row["chrom"])

        # Validate and convert coordinates
        try:
            start = int(row["start"])
            end = int(row["end"])
        except (ValueError, TypeError):
            n_skipped += 1
            if verbose:
                print(f"[WARN] Row {idx}: Invalid numeric coordinates")
            continue

        if start >= end:
            n_skipped += 1
            continue

        # Find matching path
        if chrom not in path_map:
            n_skipped += 1
            continue

        path_name = path_map[chrom]
        offset_info = extract_path_offset(path_name)
        if offset_info is None:
            n_skipped += 1
            continue

        offset_start, offset_end = offset_info
        local_start = start - offset_start
        local_end = end - offset_start

        if local_start < 0 or local_end < 0 or end > offset_end:
            n_skipped += 1
            continue

        # Extract scores with safe conversion
        score = pd.to_numeric(row.get("score", 0), errors='coerce')
        signal = pd.to_numeric(row.get("signal", score), errors='coerce')
        qval = pd.to_numeric(row.get("qval", 0), errors='coerce')

        score = 0 if pd.isna(score) else float(score)
        signal = 0 if pd.isna(signal) else float(signal)
        qval = 0 if pd.isna(qval) else float(qval)

        adjusted_rows.append({
            "path": path_name,
            "chrom": chrom,
            "genomic_start": start,
            "genomic_end": end,
            "local_start": local_start,
            "local_end": local_end,
            "score": score,
            "signal": signal,
            "qval": qval,
            "peak_group": f"peaks_{chrom}"
        })

    adjusted_df = pd.DataFrame(adjusted_rows)
    n_valid = len(adjusted_rows)

    if verbose:
        print(f"  Valid peaks: {n_valid}/{n_total} ({n_skipped} skipped)")

    return adjusted_df, {'total_peaks': n_total, 'valid_peaks': n_valid}


def ingest_peak_bed_list(bedfiles: List[str],
                        graph_paths: List[str],
                        col_names: Optional[List[str]] = None,
                        has_header: bool = False,
                        verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """Ingest and merge multiple BED files with validation.

    Args:
        bedfiles: List of BED file paths
        graph_paths: List of valid paths from ODGI graph
        col_names: Custom column names (forwarded to parse_regionpeak)
        has_header: If True, skip header row (forwarded to parse_regionpeak)
        verbose: Print per-file details

    Returns:
        Tuple of (merged DataFrame, combined validation report)
    """
    dfs = []
    combined_report = {'files': {}, 'total_valid': 0, 'total_peaks': 0}

    pbar = tqdm(bedfiles, desc="Parsing BED files", disable=not verbose)
    for bed in pbar:
        pbar.set_postfix_str(Path(bed).name)
        try:
            df, report = parse_regionpeak(bed, graph_paths, col_names=col_names,
                                          has_header=has_header, verbose=False)
            if not df.empty:
                dfs.append(df)
                combined_report['files'][bed] = report
                combined_report['total_valid'] += report['valid_peaks']
                combined_report['total_peaks'] += report['total_peaks']
        except Exception as e:
            if verbose:
                print(f"[ERROR] Failed to parse {bed}: {e}")
            combined_report['files'][bed] = {'error': str(e)}

    if not dfs:
        if verbose:
            print("[WARNING] No valid BED files ingested")
            print(f"Loaded 0 valid peaks")
        return pd.DataFrame(), combined_report

    merged = pd.concat(dfs, ignore_index=True)

    if verbose:
        print(f"\nMerged {len(bedfiles)} files: {len(merged)} total peaks")
        print(f"Success rate: {combined_report['total_valid']}/{combined_report['total_peaks']} "
              f"({100*combined_report['total_valid']/max(1,combined_report['total_peaks']):.1f}%)")

    return merged, combined_report


# ============================================================================
# SPARSE GRAPH MATRIX CLASS
# ============================================================================

@dataclass
class SparseGraphMatrix:
    """Memory-efficient sparse matrix representation of pangenome graph with peak data.

    Attributes:
        matrix: scipy.sparse.csr_matrix (nodes × paths)
        node_ids: List of node IDs (row indices)
        path_names: List of path names (column indices)
        node_to_layout: Dict mapping node_id -> {x_start, x_end, length}
        node_to_genomic: Dict mapping (node_id, path_name) -> {chrom, start, end}
        path_to_nodes: Dict mapping path_name -> [ordered node_ids]
        interval_trees: Dict mapping path_name -> IntervalTree for peak queries
    """
    matrix: csr_matrix
    node_ids: List[int]
    path_names: List[str]
    node_to_layout: Dict[int, Dict[str, int]] = field(default_factory=dict)
    node_to_genomic: Dict[Tuple[int, str], Dict[str, Union[str, int]]] = field(default_factory=dict)
    path_to_nodes: Dict[str, List[int]] = field(default_factory=dict)
    path_to_node_orientations: Dict[str, List[bool]] = field(default_factory=dict)
    interval_trees: Dict[str, IntervalTree] = field(default_factory=dict)
    graph_node_order: List[int] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Create index mappings for O(1) lookups."""
        self.node_to_idx = {nid: i for i, nid in enumerate(self.node_ids)}
        self.path_to_idx = {path: i for i, path in enumerate(self.path_names)}

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def n_nodes(self):
        return len(self.node_ids)

    @property
    def n_paths(self):
        return len(self.path_names)

    @property
    def memory_mb(self):
        """Estimate memory usage in MB."""
        matrix_bytes = self.matrix.data.nbytes + self.matrix.indices.nbytes + self.matrix.indptr.nbytes
        return matrix_bytes / 1024**2

    @property
    def sparsity(self):
        """Fraction of non-zero elements."""
        return self.matrix.nnz / (self.n_nodes * self.n_paths)

    def get_value(self, node_id: int, path_name: str) -> float:
        """Get value for specific node-path combination."""
        if node_id not in self.node_to_idx or path_name not in self.path_to_idx:
            return 0.0
        return self.matrix[self.node_to_idx[node_id], self.path_to_idx[path_name]]

    def get_node_row(self, node_id: int) -> np.ndarray:
        """Get all path values for a node."""
        if node_id not in self.node_to_idx:
            return np.zeros(self.n_paths)
        return self.matrix[self.node_to_idx[node_id], :].toarray().flatten()

    def get_path_column(self, path_name: str) -> np.ndarray:
        """Get all node values for a path."""
        if path_name not in self.path_to_idx:
            return np.zeros(self.n_nodes)
        return self.matrix[:, self.path_to_idx[path_name]].toarray().flatten()

    def get_node_layout(self) -> List[Tuple[int, int]]:
        """Get node layout as list of (x_start, x_end) tuples in node order.

        Returns:
            List of (x_start, x_end) for each node, indexed by node position in node_ids
        """
        return [(self.node_to_layout[nid]['x_start'], self.node_to_layout[nid]['x_end'])
                for nid in self.node_ids]

    def summary(self):
        """Print summary statistics."""
        print(f"SparseGraphMatrix Summary:")
        print(f"  Shape: {self.n_nodes:,} nodes × {self.n_paths} paths")
        print(f"  Non-zero elements: {self.matrix.nnz:,} ({self.sparsity*100:.2f}% filled)")
        print(f"  Memory usage: {self.memory_mb:.1f} MB")
        print(f"  Equivalent dense memory: {self.n_nodes * self.n_paths * 8 / 1024**2:.1f} MB")
        print(f"  Memory savings: {(1 - self.memory_mb / (self.n_nodes * self.n_paths * 8 / 1024**2)) * 100:.1f}%")


# ============================================================================
# GRAPH LAYOUT AND COORDINATE MAPPING
# ============================================================================

def build_node_layout(g, verbose: bool = True) -> Tuple[Dict[int, Dict], List[int]]:
    """Build node layout dictionary with graph coordinates.

    Uses for_each_handle order which matches odgi viz's position_map.

    Args:
        g: odgi.graph object
        verbose: Show progress

    Returns:
        Tuple of:
          - Dict mapping node_id -> {x_start, x_end, length}
          - List of node IDs in graph handle order (= odgi viz x-axis order)
    """
    node_layout = {}
    offset = 0

    handle_order = []
    def collect(h):
        handle_order.append(g.get_id(h))
        return True
    g.for_each_handle(collect, False)

    pbar = tqdm(handle_order, desc="Building node layout", disable=not verbose)
    for node_id in pbar:
        h = g.get_handle(node_id, False)
        seq = g.get_sequence(h)
        node_len = len(seq)

        node_layout[node_id] = {
            'x_start': offset,
            'x_end': offset + node_len,
            'length': node_len
        }
        offset += node_len

    if verbose:
        print(f"Built layout for {len(node_layout):,} nodes, total length: {offset:,} bp")

    return node_layout, handle_order


def build_path_mappings(g, path_names: List[str],
                       node_layout: Dict[int, Dict],
                       verbose: bool = True) -> Tuple[Dict, Dict, Dict]:
    """Build bidirectional mappings between paths and nodes.

    Args:
        g: odgi.graph object
        path_names: List of path names to process
        node_layout: Node layout dictionary from build_node_layout
        verbose: Show progress

    Returns:
        Tuple of (path_to_nodes, node_to_genomic, path_to_node_orientations).
        path_to_node_orientations maps path_name -> List[bool] where True means
        that step traverses the node on the reverse strand.
    """
    path_to_nodes = {}
    node_to_genomic = {}
    path_to_node_orientations = {}

    pbar = tqdm(path_names, desc="Building path mappings", disable=not verbose)
    for path_name in pbar:
        try:
            ph = g.get_path_handle(path_name)
        except RuntimeError:
            if verbose:
                print(f"[WARN] Path '{path_name}' not found in graph")
            continue

        nodes = []
        orientations = []  # True = reverse strand at this step
        offset = 0
        step = g.path_begin(ph)

        # Extract genomic coordinates from path name
        chrom = clean_path_name(path_name)
        genomic_offset = extract_path_offset(path_name)
        genomic_start = genomic_offset[0] if genomic_offset else 0

        while True:
            h = g.get_handle_of_step(step)
            node_id = g.get_id(h)
            node_len = len(g.get_sequence(h))
            is_reverse = g.get_is_reverse(h)

            nodes.append(node_id)
            orientations.append(is_reverse)

            # Store genomic coordinates for this node in this path
            node_to_genomic[(node_id, path_name)] = {
                'chrom': chrom,
                'start': genomic_start + offset,
                'end': genomic_start + offset + node_len
            }

            offset += node_len

            if not g.has_next_step(step):
                break
            step = g.get_next_step(step)

        path_to_nodes[path_name] = nodes
        path_to_node_orientations[path_name] = orientations

    if verbose:
        print(f"Built mappings for {len(path_to_nodes)} paths")
        print(f"Stored genomic coords for {len(node_to_genomic):,} (node, path) pairs")

    return path_to_nodes, node_to_genomic, path_to_node_orientations


# ============================================================================
# PATH LINEARITY ASSESSMENT (PRE-FLIGHT CHECK)
# ============================================================================

def assess_path_linearity(g,
                          path_names: List[str],
                          inversion_threshold: float = 0.1,
                          monotonicity_threshold: float = 0.2,
                          verbose: bool = True) -> pd.DataFrame:
    """Assess how linearly each path traverses the graph.

    Two statistics are computed per path:

    1. **Inversion rate** — fraction of steps where the handle is on the reverse
       strand.  A reverse step means the path walks that node's complement, so the
       cumulative offset counter drifts away from the reference coordinate encoded
       in the path name.  Values above *inversion_threshold* trigger a flag.

    2. **Monotonicity violation rate** — fraction of consecutive step pairs where
       the node ID decreases (i.e. the path moves "leftward" in the compacted
       graph).  In an odgi graph sorted with ``odgi sort``, node IDs correlate
       with linear order, so frequent non-monotonic transitions indicate
       structural rearrangements that break the offset assumption.  Values above
       *monotonicity_threshold* trigger a flag.

    A path is flagged if *either* statistic exceeds its threshold.  Flagged paths
    should be excluded from ``build_sparse_matrix()`` or interpreted with caution
    because peak-to-node signal assignment will be inaccurate.

    Args:
        g: odgi.graph object
        path_names: List of path names to assess
        inversion_threshold: Flag paths with inversion_rate >= this value (default 0.1)
        monotonicity_threshold: Flag paths with monotonicity_violation_rate >= this value (default 0.2)
        verbose: Show progress bar

    Returns:
        DataFrame with columns:
            path, n_steps, n_reverse_steps, inversion_rate,
            n_monotonicity_violations, monotonicity_violation_rate,
            flagged, flag_reasons
    """
    records = []

    pbar = tqdm(path_names, desc="Assessing path linearity", disable=not verbose)
    for path_name in pbar:
        try:
            ph = g.get_path_handle(path_name)
        except RuntimeError:
            if verbose:
                print(f"[WARN] Path '{path_name}' not found — skipping")
            continue

        n_steps = 0
        n_reverse = 0
        n_mono_violations = 0
        prev_node_id = None

        step = g.path_begin(ph)
        while True:
            h = g.get_handle_of_step(step)
            node_id = g.get_id(h)

            # Inversion: handle is on reverse strand
            if g.get_is_reverse(h):
                n_reverse += 1

            # Monotonicity: node ID went backwards relative to previous step
            if prev_node_id is not None and node_id < prev_node_id:
                n_mono_violations += 1

            prev_node_id = node_id
            n_steps += 1

            if not g.has_next_step(step):
                break
            step = g.get_next_step(step)

        inversion_rate = n_reverse / n_steps if n_steps > 0 else 0.0
        mono_violation_rate = n_mono_violations / (n_steps - 1) if n_steps > 1 else 0.0

        flag_reasons = []
        if inversion_rate >= inversion_threshold:
            flag_reasons.append(f"inversion_rate={inversion_rate:.3f}>={inversion_threshold}")
        if mono_violation_rate >= monotonicity_threshold:
            flag_reasons.append(f"monotonicity_violation_rate={mono_violation_rate:.3f}>={monotonicity_threshold}")

        records.append({
            'path': path_name,
            'n_steps': n_steps,
            'n_reverse_steps': n_reverse,
            'inversion_rate': round(inversion_rate, 4),
            'n_monotonicity_violations': n_mono_violations,
            'monotonicity_violation_rate': round(mono_violation_rate, 4),
            'flagged': len(flag_reasons) > 0,
            'flag_reasons': '; '.join(flag_reasons) if flag_reasons else '',
        })

    df = pd.DataFrame(records)

    if verbose and not df.empty:
        n_flagged = df['flagged'].sum()
        print(f"\nLinearity assessment complete: {len(df)} paths, {n_flagged} flagged")
        if n_flagged > 0:
            print(f"Flagged paths (inversion >= {inversion_threshold} or "
                  f"monotonicity violation >= {monotonicity_threshold}):")
            for _, row in df[df['flagged']].iterrows():
                print(f"  {row['path']}: {row['flag_reasons']}")

    return df


# ============================================================================
# INTERVAL TREE CONSTRUCTION FOR PEAK QUERIES
# ============================================================================

def build_interval_trees(peaks_df: pd.DataFrame,
                        path_names: List[str],
                        verbose: bool = True) -> Dict[str, IntervalTree]:
    """Build interval trees for efficient peak overlap queries.

    Args:
        peaks_df: DataFrame from parse_regionpeak with local_start, local_end columns
        path_names: List of path names
        verbose: Show progress

    Returns:
        Dict mapping path_name -> IntervalTree
    """
    interval_trees = {}

    pbar = tqdm(path_names, desc="Building interval trees", disable=not verbose)
    for path_name in pbar:
        tree = IntervalTree()
        path_peaks = peaks_df[peaks_df['path'] == path_name]

        for _, row in path_peaks.iterrows():
            # Store score/signal as interval data
            tree.add(Interval(row['local_start'], row['local_end'], {
                'score': row['score'],
                'signal': row['signal'],
                'qval': row['qval']
            }))

        interval_trees[path_name] = tree

    if verbose:
        total_intervals = sum(len(tree) for tree in interval_trees.values())
        print(f"Built {len(interval_trees)} interval trees with {total_intervals:,} total intervals")

    return interval_trees


# ============================================================================
# SPARSE MATRIX CONSTRUCTION
# ============================================================================

def build_sparse_matrix(g,
                       path_names: List[str],
                       peaks_df: Optional[pd.DataFrame] = None,
                       use_signal: bool = True,
                       signal_column: str = 'score',
                       verbose: bool = True) -> SparseGraphMatrix:
    """Build sparse matrix representation of graph with peak signals.

    Args:
        g: odgi.graph object
        path_names: List of paths to include
        peaks_df: Optional DataFrame with peak data (from ingest_peak_bed_list)
        use_signal: Whether to overlay peak signals
        signal_column: Which column to use for signal ('score', 'signal', 'qval')
        verbose: Show progress and memory usage

    Returns:
        SparseGraphMatrix object
    """
    # Step 1: Build node layout
    node_layout, graph_node_order = build_node_layout(g, verbose=verbose)
    node_ids = graph_node_order
    n_nodes = len(node_ids)
    n_paths = len(path_names)

    if verbose:
        print(f"\nMatrix dimensions: {n_nodes:,} nodes × {n_paths} paths")

    # Step 2: Build path mappings
    path_to_nodes, node_to_genomic, path_to_node_orientations = build_path_mappings(
        g, path_names, node_layout, verbose=verbose
    )

    # Step 3: Initialize sparse matrix (LIL format for efficient construction)
    if verbose:
        print("\nInitializing sparse matrix...")
    matrix = lil_matrix((n_nodes, n_paths), dtype=np.float32)

    # Create index mappings
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    path_to_idx = {path: i for i, path in enumerate(path_names)}

    # Step 4: Fill presence (1.0 where path traverses node)
    if verbose:
        print("Filling path presence...")

    pbar = tqdm(path_names, desc="Path presence", disable=not verbose)
    for path_name in pbar:
        if path_name not in path_to_nodes:
            continue

        path_idx = path_to_idx[path_name]
        for node_id in path_to_nodes[path_name]:
            node_idx = node_to_idx[node_id]
            matrix[node_idx, path_idx] = 1.0

    # Step 5: Overlay peak signals if provided
    interval_trees = {}
    if peaks_df is not None and use_signal and not peaks_df.empty:
        if verbose:
            print(f"\nOverlaying peak signals (using {signal_column} column)...")

        # Build interval trees for efficient queries
        interval_trees = build_interval_trees(peaks_df, path_names, verbose=verbose)

        # For each path, query overlapping peaks and add signal
        pbar = tqdm(path_names, desc="Signal overlay", disable=not verbose)
        for path_name in pbar:
            if path_name not in path_to_nodes or path_name not in interval_trees:
                continue

            tree = interval_trees[path_name]
            if len(tree) == 0:
                continue

            path_idx = path_to_idx[path_name]
            path_offset = 0

            # Walk through nodes in this path
            for node_id in path_to_nodes[path_name]:
                node_idx = node_to_idx[node_id]
                node_len = node_layout[node_id]['length']

                # Query overlapping peaks
                overlaps = tree.overlap(path_offset, path_offset + node_len)

                if overlaps:
                    peak_signal = max(interval.data[signal_column] for interval in overlaps)
                    matrix[node_idx, path_idx] = max(matrix[node_idx, path_idx], 1.0 + peak_signal)

                path_offset += node_len

    # Step 6: Convert to CSR for efficient operations
    if verbose:
        print("\nConverting to CSR format...")
    matrix_csr = matrix.tocsr()

    sgm = SparseGraphMatrix(
        matrix=matrix_csr,
        node_ids=node_ids,
        path_names=path_names,
        node_to_layout=node_layout,
        node_to_genomic=node_to_genomic,
        path_to_nodes=path_to_nodes,
        path_to_node_orientations=path_to_node_orientations,
        interval_trees=interval_trees,
        graph_node_order=graph_node_order,
        metadata={
            'signal_column': signal_column,
            'use_signal': use_signal,
            'n_peaks': len(peaks_df) if peaks_df is not None else 0
        }
    )

    if verbose:
        print("\n" + "="*60)
        sgm.summary()
        print("="*60)

    return sgm


# ============================================================================
# QUERY ENGINE WITH DUAL COORDINATES
# ============================================================================

def query_nodes(sgm: SparseGraphMatrix,
               conditions: Dict[str, Dict[str, Union[str, float]]],
               return_coords: str = 'both',
               max_results: Optional[int] = None) -> pd.DataFrame:
    """Query nodes based on path-specific score conditions.

    Args:
        sgm: SparseGraphMatrix object
        conditions: Dict of {path_name: {'op': '>', 'value': 5.0}}
                   Supported operators: '>', '<', '==', '>=', '<=', '!='
        return_coords: 'graph', 'genomic', or 'both'
        max_results: Maximum results to return (None for all)

    Returns:
        DataFrame with matching nodes and their coordinates

    Example:
        # Find nodes where pathA has score > 5 and pathB has score == 0
        results = query_nodes(sgm, {
            'pathA': {'op': '>', 'value': 5},
            'pathB': {'op': '==', 'value': 0}
        })
    """
    # Start with all nodes
    mask = np.ones(sgm.n_nodes, dtype=bool)

    # Apply each condition
    for path_name, condition in conditions.items():
        if path_name not in sgm.path_to_idx:
            print(f"[WARN] Path '{path_name}' not found in matrix")
            continue

        op = condition['op']
        value = condition['value']

        # Get column for this path
        path_values = sgm.get_path_column(path_name)

        # Apply operator
        if op == '>':
            mask &= path_values > value
        elif op == '<':
            mask &= path_values < value
        elif op == '==':
            mask &= path_values == value
        elif op == '>=':
            mask &= path_values >= value
        elif op == '<=':
            mask &= path_values <= value
        elif op == '!=':
            mask &= path_values != value
        else:
            print(f"[WARN] Unknown operator '{op}'")
            continue

    # Get matching node indices
    matching_indices = np.where(mask)[0]

    if max_results is not None and len(matching_indices) > max_results:
        print(f"[INFO] Truncating {len(matching_indices)} results to {max_results}")
        matching_indices = matching_indices[:max_results]

    # Build results — vectorized where possible
    node_ids = np.array(sgm.node_ids)[matching_indices]
    result = pd.DataFrame({'node_id': node_ids})

    if return_coords in ['graph', 'both']:
        layouts = [sgm.node_to_layout[nid] for nid in node_ids]
        result['graph_x_start'] = [l['x_start'] for l in layouts]
        result['graph_x_end'] = [l['x_end'] for l in layouts]
        result['node_length'] = [l['length'] for l in layouts]

    cond_paths = [p for p in conditions if p in sgm.path_to_idx]

    if return_coords in ['genomic', 'both']:
        for path_name in cond_paths:
            chroms, starts, ends = [], [], []
            for nid in node_ids:
                gen = sgm.node_to_genomic.get((nid, path_name))
                if gen:
                    chroms.append(gen['chrom'])
                    starts.append(gen['start'])
                    ends.append(gen['end'])
                else:
                    chroms.append(None)
                    starts.append(None)
                    ends.append(None)
            result[f'{path_name}_chrom'] = chroms
            result[f'{path_name}_start'] = starts
            result[f'{path_name}_end'] = ends

    for path_name in cond_paths:
        col = sgm.matrix[:, sgm.path_to_idx[path_name]].toarray().flatten()
        result[f'{path_name}_score'] = col[matching_indices]

    print(f"Found {len(result)} nodes matching query")
    return result


def export_query_bed(query_results: pd.DataFrame,
                    filename: str,
                    coord_system: str = 'genomic',
                    path_name: Optional[str] = None):
    """Export query results to BED format.

    Args:
        query_results: DataFrame from query_nodes()
        filename: Output BED file path
        coord_system: 'graph' or 'genomic'
        path_name: For genomic coords, which path to use (required if coord_system='genomic')
    """
    if coord_system == 'graph':
        bed_df = query_results[['node_id', 'graph_x_start', 'graph_x_end']].copy()
        bed_df.columns = ['name', 'start', 'end']
        bed_df['chrom'] = 'graph'
        bed_df = bed_df[['chrom', 'start', 'end', 'name']]
    elif coord_system == 'genomic':
        if path_name is None:
            raise ValueError("path_name required for genomic coordinates")

        chrom_col = f'{path_name}_chrom'
        start_col = f'{path_name}_start'
        end_col = f'{path_name}_end'

        if chrom_col not in query_results.columns:
            raise ValueError(f"Path '{path_name}' not in query results")

        bed_df = query_results[[chrom_col, start_col, end_col, 'node_id']].copy()
        bed_df.columns = ['chrom', 'start', 'end', 'name']
        # Remove rows with None values
        bed_df = bed_df.dropna()
    else:
        raise ValueError(f"Invalid coord_system: {coord_system}")

    bed_df.to_csv(filename, sep='\t', header=False, index=False)
    print(f"Exported {len(bed_df)} entries to {filename}")


# ============================================================================
# VISUALIZATION
# ============================================================================

def bedspread_plot_fast(sgm: SparseGraphMatrix,
                       paths: Optional[List[str]] = None,
                       figsize: Tuple[int, int] = (15, 2),
                       num_breaks: int = 6,
                       show_colorbar: bool = True,
                       dpi: int = 100,
                       bp_resolution: int = 4096,
                       highlight_peaks: Optional[pd.DataFrame] = None,
                       highlight_color: str = '#7b2d8b'):
    """Fast bp-proportional heatmap using a pre-rasterized pixel buffer.

    Each pixel column represents an equal-width bp slice of the graph, so wide
    backbone nodes appear wide and short bubble nodes appear narrow — matching
    the visual output of odgi viz.  Internally pre-renders a
    (n_paths × bp_resolution) numpy array and calls imshow once.

    Color mapping (identical to bedspread_plot_static):
      - 0    → white  (node absent)
      - 1.0  → grey   (node present, no peak signal)
      - >1.0 → pink→red gradient (present with signal)

    When highlight_peaks is provided (DataFrame from extract_peak_nodes with
    'node_id' column), the nodes belonging to those peaks are rendered in
    purple instead of the normal red gradient.

    When multiple nodes fall within the same pixel column their values are
    merged by taking the maximum (preserves signal, never inflates it).
    """
    if paths is None:
        paths = sgm.path_names

    path_indices = [sgm.path_to_idx[p] for p in paths if p in sgm.path_to_idx]
    n_paths_shown = len(path_indices)

    # data: shape (n_paths_shown, n_nodes)
    data = sgm.matrix[:, path_indices].toarray().T

    vmax = float(np.nanmax(data)) if data.size > 0 else 1.0

    # Colormap — identical palette to bedspread_plot_static
    if vmax <= 1:
        cmap = LinearSegmentedColormap.from_list(
        "bedspread", ["white", "#d4d4d4"], N=256)
        norm = Normalize(vmin=0, vmax=1)
    else:
        # Use a TWO-STAGE norm: 0→1 maps to [0, GREY_FRAC], 1→vmax maps to [GREY_FRAC, 1]
        # This keeps grey always visible regardless of how large vmax is.
        GREY_FRAC = 0.15
        positions = [0.0, GREY_FRAC, GREY_FRAC + 0.01,
                    GREY_FRAC + (1 - GREY_FRAC) * 0.33,
                    GREY_FRAC + (1 - GREY_FRAC) * 0.67,
                    1.0]
        colors = ["white", "#d4d4d4", "#ffb6c1", "#ff6699", "#cc0000", "#800000"]
        cmap = LinearSegmentedColormap.from_list(
            "bedspread", list(zip(positions, colors)), N=256)

        # Custom norm: compress 0–1 into [0, GREY_FRAC], stretch 1–vmax into [GREY_FRAC, 1]
        class TwoStageNorm(Normalize):
            def __call__(self, value, clip=None):
                v = np.asarray(value, dtype=float)
                out = np.where(
                    v <= 1.0,
                    v * GREY_FRAC,
                    GREY_FRAC + (v - 1.0) / (vmax - 1.0) * (1.0 - GREY_FRAC)
                )
                return np.ma.masked_array(np.clip(out, 0, 1))

        norm = TwoStageNorm()

    # --- Build bp-proportional raster (n_paths_shown × bp_resolution) ---
    total_bp = max(v['x_end'] for v in sgm.node_to_layout.values())
    raster = np.zeros((n_paths_shown, bp_resolution), dtype=np.float32)

    # Also track which pixel columns belong to highlighted peaks
    highlight_node_set = set()
    if highlight_peaks is not None and len(highlight_peaks) > 0:
        highlight_node_set = set(highlight_peaks['node_id'].unique())
    highlight_mask = np.zeros((n_paths_shown, bp_resolution), dtype=bool)

    for i, nid in enumerate(sgm.node_ids):
        layout = sgm.node_to_layout[nid]
        cs = int(layout['x_start'] * bp_resolution / total_bp)
        ce = int(layout['x_end']   * bp_resolution / total_bp)
        ce = max(cs + 1, ce)         # guarantee at least 1 pixel wide
        ce = min(ce, bp_resolution)  # clamp to buffer width
        if cs >= bp_resolution:
            continue
        node_vals = data[:, i]       # shape (n_paths_shown,)
        np.maximum(raster[:, cs:ce], node_vals[:, np.newaxis], out=raster[:, cs:ce])

        # Mark highlighted pixels where the node is present in each path
        if nid in highlight_node_set:
            for row_idx in range(n_paths_shown):
                if node_vals[row_idx] > 0:
                    highlight_mask[row_idx, cs:ce] = True

    # --- Figure ---
    fig_height = max(2, min(figsize[1], n_paths_shown * 0.25))
    fig, ax = plt.subplots(figsize=(figsize[0], fig_height), dpi=dpi)

    # --- Render with highlight_peaks colored purple ---
    if highlight_node_set:
        import matplotlib.colors as mcolors
        normed = norm(raster)
        rgba = cmap(normed)  # shape: (n_paths, bp_resolution, 4)

        hl_rgb = np.array(mcolors.to_rgb(highlight_color))
        mask_rows, mask_cols = np.where(highlight_mask)
        for r, c in zip(mask_rows, mask_cols):
            val = raster[r, c]
            if val > 1:
                intensity = min(1.0, (val - 1.0) / (vmax - 1.0)) if vmax > 1 else 0.5
                alpha = 0.4 + 0.6 * intensity
            else:
                alpha = 0.4
            rgba[r, c, :3] = (1 - alpha) * rgba[r, c, :3] + alpha * hl_rgb

        im = ax.imshow(rgba, aspect='auto', interpolation='nearest', origin='upper')
    else:
        im = ax.imshow(raster, aspect='auto', cmap=cmap, norm=norm,
                       interpolation='nearest', origin='upper')

    # --- Axes ---
    ax.set_yticks(range(n_paths_shown))
    ax.set_yticklabels(paths, fontsize=8)

    n_xticks = 6
    tick_px = np.linspace(0, bp_resolution - 1, n_xticks)
    tick_bp = (tick_px / bp_resolution * total_bp).astype(int)
    ax.set_xticks(tick_px)
    ax.set_xticklabels([f"{v:,}" for v in tick_bp], fontsize=8)
    ax.set_xlabel("Graph coordinate (bp)", fontsize=9)

    title = "BedSpread Visualization"
    if highlight_node_set:
        n_peaks_hl = highlight_peaks['peak_idx'].nunique() if 'peak_idx' in highlight_peaks.columns else len(highlight_peaks)
        title += f" ({n_peaks_hl} highlighted peaks)"
    ax.set_title(title, fontsize=10, pad=10)

    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, orientation="vertical",
                            fraction=0.02, pad=0.02)
        cbar.set_label("Signal intensity", fontsize=9, rotation=270, labelpad=15)
        if vmax > 1:
            tick_values = [0, 1] + list(np.linspace(2, vmax, min(num_breaks - 1, 5)))
            cbar.set_ticks(tick_values)
            cbar.set_ticklabels([f"{v:.1f}" for v in tick_values], fontsize=8)

    plt.tight_layout()
    return fig, ax


def export_static_plot(sgm: SparseGraphMatrix,
                      filename: str,
                      format: str = 'png',
                      paths: Optional[List[str]] = None,
                      dpi: int = 300,
                      **kwargs):
    """Render a bedspread_plot_fast figure and save it to disk.

    Args:
        sgm: SparseGraphMatrix object
        filename: Output file path
        format: Image format ('png', 'pdf', 'svg', etc.)
        paths: List of paths to display (None for all)
        dpi: Resolution
        **kwargs: Forwarded to bedspread_plot_fast()
    """
    fig, ax = bedspread_plot_fast(sgm, paths=paths, dpi=dpi, **kwargs)
    fig.savefig(filename, format=format, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {format.upper()} to {filename}")


# ============================================================================
# I/O FUNCTIONS FOR PERSISTENCE
# ============================================================================

def save_sparse_matrix(sgm: SparseGraphMatrix, prefix: str):
    """Save SparseGraphMatrix to disk.

    Creates two files:
    - {prefix}_matrix.npz: Sparse matrix data
    - {prefix}_metadata.pkl: All metadata dicts

    Args:
        sgm: SparseGraphMatrix object
        prefix: File path prefix
    """
    # Save matrix
    matrix_file = f"{prefix}_matrix.npz"
    save_npz(matrix_file, sgm.matrix)

    # Save metadata
    metadata_file = f"{prefix}_metadata.pkl"
    metadata = {
        'node_ids': sgm.node_ids,
        'path_names': sgm.path_names,
        'node_to_layout': sgm.node_to_layout,
        'node_to_genomic': sgm.node_to_genomic,
        'path_to_nodes': sgm.path_to_nodes,
        'path_to_node_orientations': sgm.path_to_node_orientations,
        'interval_trees': sgm.interval_trees,
        'graph_node_order': sgm.graph_node_order,
        'metadata': sgm.metadata
    }

    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)

    total_size = os.path.getsize(matrix_file) + os.path.getsize(metadata_file)
    print(f"Saved SparseGraphMatrix to {prefix}_* ({total_size / 1024**2:.1f} MB total)")


def load_sparse_matrix(prefix: str) -> SparseGraphMatrix:
    """Load SparseGraphMatrix from disk.

    Args:
        prefix: File path prefix used in save_sparse_matrix()

    Returns:
        SparseGraphMatrix object
    """
    # Load matrix
    matrix_file = f"{prefix}_matrix.npz"
    matrix = load_npz(matrix_file)

    # Load metadata
    metadata_file = f"{prefix}_metadata.pkl"
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)

    sgm = SparseGraphMatrix(
        matrix=matrix,
        node_ids=metadata['node_ids'],
        path_names=metadata['path_names'],
        node_to_layout=metadata['node_to_layout'],
        node_to_genomic=metadata['node_to_genomic'],
        path_to_nodes=metadata['path_to_nodes'],
        path_to_node_orientations=metadata.get('path_to_node_orientations', {}),
        interval_trees=metadata['interval_trees'],
        graph_node_order=metadata.get('graph_node_order', metadata['node_ids']),
        metadata=metadata['metadata']
    )

    print(f"Loaded SparseGraphMatrix from {prefix}_*")
    sgm.summary()
    return sgm
