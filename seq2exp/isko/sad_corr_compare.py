#!/usr/bin/env python
"""
sad_corr_compare.py

Correlate Borzoi SAD metrics with STARRseq fold enrichment across tissues.
Generates correlation tables and heatmaps.
"""

import argparse
import os
import sys
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import json

# Set plot parameters for scientific quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Correlate SAD metrics with STARRseq fold enrichment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--h5',
        required=True,
        help='Input SAD HDF5 file from Borzoi'
    )
    
    parser.add_argument(
        '--csv',
        required=True,
        help='STARRseq CSV file with fold_enrichment column'
    )
    
    parser.add_argument(
        '--tissues-json',
        default=None,
        help='Optional JSON file mapping tissue indices to names (23tissues_full_dict.json). If not provided, uses target_ids from H5 file.'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        default='sad_corr_plots',
        help='Output directory for plots and tables'
    )
    
    parser.add_argument(
        '--metrics',
        nargs='+',
        default=None,
        help='Metrics to correlate (default: all available in H5 file). Available: SAD, SADlog, logSAD, sqrtSAD, SAX, D1, logD1, sqrtD1, D2, logD2, sqrtD2, JS, logJS'
    )
    
    parser.add_argument(
        '--min-samples',
        type=int,
        default=10,
        help='Minimum number of matched samples required for correlation'
    )
    
    parser.add_argument(
        '-t',
        '--fold-enrichment-threshold',
        type=float,
        default=None,
        help='Minimum fold enrichment threshold to filter regions (default: None, no filtering)'
    )
    
    return parser.parse_args()


def load_sad_data(h5_path, metrics=None):
    """Load SAD metrics from HDF5 file.
    
    Args:
        h5_path: Path to HDF5 file
        metrics: List of metrics to load, or None for all
    
    Returns:
        Dictionary with metric data and metadata
    """
    print(f"\nLoading SAD data from {h5_path}")
    h5_file = h5py.File(h5_path, 'r')
    
    # Get all available metric datasets (exclude metadata and percentile datasets)
    exclude_keys = ['snp', 'chr', 'pos', 'target_ids', 'target_labels', 'ref_allele', 'alt_allele', 'percentiles', 'REF', 'ALT']
    all_keys = list(h5_file.keys())
    
    # Find metric datasets (exclude _pct percentile datasets and metadata)
    available_metrics = [k for k in all_keys 
                        if k not in exclude_keys 
                        and not k.endswith('_pct')
                        and len(h5_file[k].shape) == 2]  # Metrics are 2D (variants x tissues)
    
    if metrics is None:
        metrics = available_metrics
    else:
        # Validate requested metrics
        metrics = [m for m in metrics if m in available_metrics]
        if len(metrics) == 0:
            print(f"WARNING: None of the requested metrics found in H5 file")
            print(f"Available metrics: {', '.join(available_metrics)}")
            metrics = available_metrics
    
    print(f"Available metrics in H5: {', '.join(sorted(available_metrics))}")
    print(f"Loading metrics: {', '.join(metrics)}")
    
    # Load data
    sad_data = {}
    for metric in metrics:
        sad_data[metric] = h5_file[metric][:]
        # Check if metric is all zeros or constant
        unique_vals = len(np.unique(sad_data[metric]))
        if unique_vals == 1:
            print(f"  {metric}: shape={sad_data[metric].shape} (WARNING: constant values, will skip)")
        else:
            print(f"  {metric}: shape={sad_data[metric].shape}")
    
    # Load metadata
    sad_data['snp'] = h5_file['snp'][:]
    sad_data['chr'] = h5_file['chr'][:]
    sad_data['pos'] = h5_file['pos'][:]
    sad_data['target_ids'] = h5_file['target_ids'][:]
    
    h5_file.close()
    
    return sad_data, metrics


def load_starrseq_data(csv_path):
    """Load STARRseq data from CSV file.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        DataFrame with STARRseq data
    """
    print(f"\nLoading STARRseq data from {csv_path}")
    
    # Try reading with different encodings
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(csv_path)
    
    # Clean column names
    df.columns = df.columns.str.strip().str.lstrip('\ufeff')
    
    print(f"Loaded {len(df)} regions")
    print(f"Columns: {', '.join(df.columns)}")
    
    # Check for required columns
    if 'fold_enrichment' not in df.columns:
        raise ValueError("CSV must have 'fold_enrichment' column")
    
    # Ensure we have chr, start, end columns
    chr_col = None
    start_col = None
    end_col = None
    
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ['chr', 'chrom', 'chromosome']:
            chr_col = col
        elif col_lower in ['start', 'chromstart']:
            start_col = col
        elif col_lower in ['end', 'chromend']:
            end_col = col
    
    if not all([chr_col, start_col, end_col]):
        raise ValueError(f"CSV must have chr/start/end columns. Found: {df.columns.tolist()}")
    
    # Standardize column names
    df['chr'] = df[chr_col].astype(str).str.replace('chr', '', regex=False)
    df['start'] = df[start_col].astype(int)
    df['end'] = df[end_col].astype(int)
    
    return df


def load_tissue_mapping(target_ids, json_path=None):
    """Load tissue index to name mapping.
    
    Args:
        target_ids: Array of target IDs from H5 file
        json_path: Optional path to JSON file for mapping
    
    Returns:
        Dictionary mapping tissue index to name
    """
    if json_path and os.path.exists(json_path):
        print(f"\nLoading tissue mapping from {json_path}")
        with open(json_path, 'r') as f:
            tissue_dict = json.load(f)
        
        # Create index to name mapping
        tissue_mapping = {}
        for tissue_name, tissue_info in tissue_dict.items():
            if 'index' in tissue_info:
                tissue_mapping[tissue_info['index']] = tissue_name
        
        print(f"Loaded {len(tissue_mapping)} tissue mappings from JSON")
    else:
        # Use target_ids directly
        print(f"\nUsing target IDs from H5 file as tissue names")
        tissue_mapping = {}
        for idx, tid in enumerate(target_ids):
            tid_str = tid.decode('utf-8') if isinstance(tid, bytes) else str(tid)
            tissue_mapping[idx] = tid_str
        
        print(f"Loaded {len(tissue_mapping)} tissue mappings from H5")
    
    return tissue_mapping


def align_sad_starrseq(sad_data, starrseq_df, fold_enrichment_threshold=None):
    """Align SAD and STARRseq data by genomic coordinates.
    
    Args:
        sad_data: Dictionary with SAD metrics
        starrseq_df: DataFrame with STARRseq data
        fold_enrichment_threshold: Minimum fold enrichment to include (None = no filtering)
    
    Returns:
        Tuple of (matched_indices, fold_enrichments)
    """
    if fold_enrichment_threshold is not None:
        print(f"\nAligning SAD and STARRseq data (filtering fold_enrichment > {fold_enrichment_threshold})...")
        # Filter STARRseq data before matching
        original_len = len(starrseq_df)
        starrseq_df = starrseq_df[starrseq_df['fold_enrichment'] > fold_enrichment_threshold].copy()
        print(f"Filtered from {original_len} to {len(starrseq_df)} regions")
    else:
        print("\nAligning SAD and STARRseq data (no fold enrichment filtering)...")
    
    # Create coordinate lookup from SAD data
    sad_coords = {}
    for i in range(len(sad_data['snp'])):
        snp_id = sad_data['snp'][i].decode('utf-8') if isinstance(sad_data['snp'][i], bytes) else str(sad_data['snp'][i])
        chr_val = sad_data['chr'][i].decode('utf-8') if isinstance(sad_data['chr'][i], bytes) else str(sad_data['chr'][i])
        pos_val = int(sad_data['pos'][i])
        
        # Parse region from snp_id (format: chr:start-end)
        if ':' in snp_id and '-' in snp_id:
            parts = snp_id.split(':')
            chr_str = parts[0].replace('chr', '')
            pos_parts = parts[1].split('-')
            start = int(pos_parts[0])
            end = int(pos_parts[1])
            
            # Try multiple coordinate keys for fuzzy matching
            for s_offset in [-1, 0, 1]:
                for e_offset in [-1, 0, 1]:
                    key = f"{chr_str}_{start + s_offset}_{end + e_offset}"
                    if key not in sad_coords:
                        sad_coords[key] = i
    
    print(f"Created {len(sad_coords)} coordinate lookup entries from {len(sad_data['snp'])} SAD regions")
    
    # Match with STARRseq data
    matched_indices = []
    fold_enrichments = []
    
    for idx, row in starrseq_df.iterrows():
        coord_key = f"{row['chr']}_{row['start']}_{row['end']}"
        
        if coord_key in sad_coords:
            sad_idx = sad_coords[coord_key]
            matched_indices.append(sad_idx)
            fold_enrichments.append(row['fold_enrichment'])
    
    print(f"Matched {len(matched_indices)} regions")
    print(f"Success rate: {100.0 * len(matched_indices) / len(sad_data['snp']):.2f}%")
    
    if len(matched_indices) == 0:
        print("\nERROR: No matches found!")
        print(f"  SAD coord examples: {list(sad_coords.keys())[:3]}")
        print(f"  CSV coord examples: {starrseq_df['chr'].astype(str) + '_' + starrseq_df['start'].astype(str) + '_' + starrseq_df['end'].astype(str)}")[:3]
    
    return np.array(matched_indices), np.array(fold_enrichments)


def compute_correlations(sad_data, metrics, matched_indices, fold_enrichments, tissue_mapping, min_samples=10):
    """Compute correlations between SAD metrics and fold enrichment.
    
    Args:
        sad_data: Dictionary with SAD metrics
        metrics: List of metrics to correlate
        matched_indices: Indices of matched SAD regions
        fold_enrichments: Fold enrichment values
        tissue_mapping: Dictionary mapping tissue index to name
        min_samples: Minimum samples for correlation
    
    Returns:
        DataFrame with correlation results
    """
    print("\nComputing correlations...")
    
    results = []
    skipped_metrics = []
    
    for metric in metrics:
        print(f"  Processing {metric}...")
        metric_data = sad_data[metric][matched_indices]  # Shape: (n_matched, n_tissues)
        
        # Check if metric is constant (all same values)
        if len(np.unique(metric_data)) == 1:
            print(f"    Skipping {metric}: all values are constant")
            skipped_metrics.append(metric)
            continue
        
        # Correlate each tissue
        tissue_count = 0
        for tissue_idx in range(metric_data.shape[1]):
            tissue_values = metric_data[:, tissue_idx]
            
            # Check if this tissue has constant values
            if len(np.unique(tissue_values)) == 1:
                continue
            
            # Remove NaN/Inf
            valid_mask = np.isfinite(tissue_values) & np.isfinite(fold_enrichments)
            tissue_valid = tissue_values[valid_mask]
            fe_valid = fold_enrichments[valid_mask]
            
            if len(tissue_valid) < min_samples:
                continue
            
            # Compute correlations
            try:
                pearson_r, pearson_p = pearsonr(tissue_valid, fe_valid)
                spearman_r, spearman_p = spearmanr(tissue_valid, fe_valid)
                
                # Check if correlation is valid (not NaN)
                if not np.isfinite(pearson_r) or not np.isfinite(spearman_r):
                    continue
                    
            except:
                continue
            
            # Get tissue name
            tissue_name = tissue_mapping.get(tissue_idx, f"Tissue_{tissue_idx}")
            
            results.append({
                'metric': metric,
                'tissue_idx': tissue_idx,
                'tissue_name': tissue_name,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'n_samples': len(tissue_valid)
            })
            tissue_count += 1
        
        print(f"    Computed {tissue_count} tissue correlations for {metric}")
    
    results_df = pd.DataFrame(results)
    
    if len(skipped_metrics) > 0:
        print(f"\nSkipped constant metrics: {', '.join(skipped_metrics)}")
    
    print(f"Computed {len(results_df)} total correlations across {len(metrics) - len(skipped_metrics)} metrics")
    
    return results_df


def visualize_correlations(results_df, metrics, output_dir):
    """Create correlation visualizations.
    
    Args:
        results_df: DataFrame with correlation results
        metrics: List of metrics
        output_dir: Output directory
    """
    print("\nGenerating visualizations...")
    
    # 1. Save correlation table
    table_path = os.path.join(output_dir, 'correlation_table.csv')
    results_df.to_csv(table_path, index=False)
    print(f"  Saved {table_path}")
    
    # 2. Create heatmaps for both Pearson and Spearman correlations
    print("  Creating correlation heatmaps...")
    
    # Get unique tissues sorted by index
    tissue_names = sorted(results_df['tissue_name'].unique(), 
                         key=lambda x: int(results_df[results_df['tissue_name']==x]['tissue_idx'].iloc[0]))
    
    # Prepare heatmap data matrices: tissues x metrics
    pearson_data = np.zeros((len(tissue_names), len(metrics)))
    spearman_data = np.zeros((len(tissue_names), len(metrics)))
    
    for i, tissue in enumerate(tissue_names):
        for j, metric in enumerate(metrics):
            mask = (results_df['tissue_name'] == tissue) & (results_df['metric'] == metric)
            if mask.any():
                pearson_data[i, j] = results_df[mask]['pearson_r'].iloc[0]
                spearman_data[i, j] = results_df[mask]['spearman_r'].iloc[0]
    
    # Function to create a single heatmap
    def create_correlation_heatmap(data, corr_type, output_path):
        # Create figure
        fig = plt.figure(figsize=(max(8, len(metrics) * 1.2), 28))
        
        # Create axes with specific positioning
        ax = fig.add_axes([0.25, 0.1, 0.55, 0.82])
        cax = fig.add_axes([0.82, 0.1, 0.03, 0.82])
        
        # Determine color scale
        vmax = max(abs(data.min()), abs(data.max()))
        vmax = max(vmax, 0.05)  # minimum range
        
        # Create heatmap
        im = ax.imshow(data, cmap='RdBu_r', aspect='auto', 
                       vmin=-vmax, vmax=vmax, interpolation='nearest')
        
        # Set ticks and labels
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, fontsize=13, fontweight='bold', rotation=0)
        ax.set_yticks(range(len(tissue_names)))
        ax.set_yticklabels([t.replace('_', ' ') for t in tissue_names], fontsize=9.5)
        
        # Move x-axis to top
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        
        # Add gridlines
        for i in range(len(tissue_names) + 1):
            ax.axhline(i - 0.5, color='white', linewidth=1.5, alpha=0.8)
        for j in range(len(metrics) + 1):
            ax.axvline(j - 0.5, color='white', linewidth=1.5, alpha=0.8)
        
        # Add text annotations with correlation values
        for i in range(len(tissue_names)):
            for j in range(len(metrics)):
                r_val = data[i, j]
                
                # Choose text color for contrast
                text_color = 'white' if abs(r_val) > vmax * 0.5 else 'black'
                
                # Display correlation value
                ax.text(j, i, f'{r_val:.3f}', 
                       ha='center', va='center', fontsize=8,
                       color=text_color, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(f'{corr_type} r', fontsize=12, fontweight='bold', rotation=270, labelpad=20)
        cbar.ax.tick_params(labelsize=10)
        
        # Title
        fig.text(0.525, 0.97, f'{corr_type} Correlation: SAD Metrics vs STARRseq Fold Enrichment', 
                ha='center', fontsize=14, fontweight='bold')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
    
    # Create Pearson heatmap
    pearson_path = os.path.join(output_dir, 'correlation_heatmap_pearson.png')
    create_correlation_heatmap(pearson_data, 'Pearson', pearson_path)
    print(f"  Saved {pearson_path}")
    
    # Create Spearman heatmap
    spearman_path = os.path.join(output_dir, 'correlation_heatmap_spearman.png')
    create_correlation_heatmap(spearman_data, 'Spearman', spearman_path)
    print(f"  Saved {spearman_path}")
    
    # 3. Summary statistics table
    print("  Creating summary table...")
    
    summary_data = []
    for metric in metrics:
        metric_df = results_df[results_df['metric'] == metric]
        if len(metric_df) > 0:
            summary_data.append({
                'Metric': metric,
                'Mean Pearson r': f"{metric_df['pearson_r'].mean():.3f}",
                'Max Pearson r': f"{metric_df['pearson_r'].max():.3f}",
                'Mean Spearman r': f"{metric_df['spearman_r'].mean():.3f}",
                'Max Spearman r': f"{metric_df['spearman_r'].max():.3f}",
                'Significant (p<0.05)': f"{(metric_df['pearson_p'] < 0.05).sum()}/{len(metric_df)}",
                'Mean n_samples': f"{metric_df['n_samples'].mean():.0f}"
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Plot table
    fig, ax = plt.subplots(figsize=(12, max(4, len(metrics) * 0.5)))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_df.values,
                    colLabels=summary_df.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.10, 0.14, 0.13, 0.14, 0.13, 0.18, 0.13])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
    
    plt.title('SAD Metrics Correlation Summary', fontsize=14, fontweight='bold', pad=20)
    
    summary_path = os.path.join(output_dir, 'correlation_summary.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {summary_path}")


def main():
    args = parse_args()
    
    # Validate inputs
    if not os.path.exists(args.h5):
        print(f"ERROR: H5 file not found: {args.h5}")
        sys.exit(1)
    if not os.path.exists(args.csv):
        print(f"ERROR: CSV file not found: {args.csv}")
        sys.exit(1)
    if args.tissues_json and not os.path.exists(args.tissues_json):
        print(f"WARNING: Tissues JSON not found: {args.tissues_json}")
        print("Will use target_ids from H5 file instead")
        args.tissues_json = None
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("SAD Correlation Analysis")
    print("="*70)
    print(f"SAD H5:       {args.h5}")
    print(f"STARRseq CSV: {args.csv}")
    print(f"Tissues JSON: {args.tissues_json}")
    print(f"Output dir:   {args.output_dir}")
    print("="*70)
    
    # Load data
    sad_data, metrics = load_sad_data(args.h5, args.metrics)
    starrseq_df = load_starrseq_data(args.csv)
    tissue_mapping = load_tissue_mapping(sad_data['target_ids'], args.tissues_json)
    
    # Align data
    matched_indices, fold_enrichments = align_sad_starrseq(
        sad_data, starrseq_df, args.fold_enrichment_threshold
    )
    
    if len(matched_indices) < args.min_samples:
        print(f"\nERROR: Insufficient matched samples ({len(matched_indices)} < {args.min_samples})")
        sys.exit(1)
    
    # Compute correlations
    results_df = compute_correlations(
        sad_data, metrics, matched_indices, fold_enrichments, 
        tissue_mapping, args.min_samples
    )
    
    # Visualize
    visualize_correlations(results_df, metrics, args.output_dir)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
    print(f"Results saved to: {args.output_dir}")
    print("  - correlation_table.csv")
    print("  - correlation_heatmap_pearson.png")
    print("  - correlation_heatmap_spearman.png")
    print("  - correlation_summary.png")
    print("="*70)


if __name__ == "__main__":
    main()
