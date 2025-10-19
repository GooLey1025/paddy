#!/usr/bin/env python
"""
igs_corr_compare.py

Correlate Integrated Gradients (IGs) metrics with STARRseq fold enrichment across tissues.
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

# Set plot parameters
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
        description='Correlate Integrated Gradients with STARRseq fold enrichment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--h5',
        required=True,
        help='Input IGs HDF5 file from paddy_igs_region.py'
    )
    
    parser.add_argument(
        '--tissues-json',
        default=None,
        help='Optional JSON file mapping tissue indices to names'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        default='igs_corr_plots',
        help='Output directory for plots and tables'
    )
    
    parser.add_argument(
        '--metrics',
        nargs='+',
        default=['sum_abs', 'max_abs', 'mean_abs', 'l2_norm'],
        help='Metrics to compute from IGs: sum_abs, max_abs, mean_abs, l2_norm'
    )
    
    parser.add_argument(
        '--min-samples',
        type=int,
        default=10,
        help='Minimum number of samples required for correlation'
    )
    
    parser.add_argument(
        '--fold-enrichment-threshold',
        type=float,
        default=None,
        help='Minimum fold enrichment threshold to filter regions (default: None)'
    )
    
    return parser.parse_args()


def load_igs_data(h5_path):
    """Load IGs data from HDF5 file.
    
    Args:
        h5_path: Path to HDF5 file
    
    Returns:
        Dictionary with IGs data and metadata
    """
    print(f"\nLoading IGs data from {h5_path}")
    h5_file = h5py.File(h5_path, 'r')
    
    igs_data = {}
    igs_data['igs'] = h5_file['igs'][:]  # Shape: (n_regions, max_len, 4, n_tissues)
    igs_data['region_chr'] = h5_file['region_chr'][:]
    igs_data['region_start'] = h5_file['region_start'][:]
    igs_data['region_end'] = h5_file['region_end'][:]
    igs_data['region_length'] = h5_file['region_length'][:]
    igs_data['gene_id'] = h5_file['gene_id'][:]
    igs_data['fold_enrichment'] = h5_file['fold_enrichment'][:]
    igs_data['distance_to_atg'] = h5_file['distance_to_atg'][:]
    igs_data['tissue_indices'] = h5_file['tissue_indices'][:]
    
    print(f"Loaded {len(igs_data['region_chr'])} regions")
    print(f"IGs shape: {igs_data['igs'].shape}")
    print(f"Number of tissues: {len(igs_data['tissue_indices'])}")
    
    h5_file.close()
    
    return igs_data


def load_tissue_mapping(tissue_indices, json_path=None):
    """Load tissue index to name mapping.
    
    Args:
        tissue_indices: Array of tissue indices
        json_path: Optional path to JSON file for mapping
    
    Returns:
        Dictionary mapping tissue index to name
    """
    if json_path and os.path.exists(json_path):
        print(f"\nLoading tissue mapping from {json_path}")
        with open(json_path, 'r') as f:
            tissue_dict = json.load(f)
        
        tissue_mapping = {}
        for tissue_name, tissue_info in tissue_dict.items():
            if 'index' in tissue_info:
                tissue_mapping[tissue_info['index']] = tissue_name
        
        print(f"Loaded {len(tissue_mapping)} tissue mappings from JSON")
    else:
        print(f"\nUsing tissue indices as names")
        tissue_mapping = {}
        for idx in tissue_indices:
            tissue_mapping[int(idx)] = f"Tissue_{idx}"
        
        print(f"Created {len(tissue_mapping)} tissue mappings")
    
    return tissue_mapping


def compute_ig_metrics(igs_data, metrics, fold_enrichment_threshold=None):
    """Compute summary metrics from integrated gradients.
    
    Args:
        igs_data: Dictionary with IGs data
        metrics: List of metrics to compute
        fold_enrichment_threshold: Optional threshold to filter regions
    
    Returns:
        Dictionary with computed metrics (n_regions, n_tissues) for each metric
    """
    print("\nComputing IG metrics...")
    
    igs = igs_data['igs']  # (n_regions, max_len, 4, n_tissues)
    region_lengths = igs_data['region_length']
    fold_enrichments = igs_data['fold_enrichment']
    
    # Filter by fold enrichment if threshold provided
    if fold_enrichment_threshold is not None:
        valid_mask = fold_enrichments > fold_enrichment_threshold
        print(f"Filtering by fold_enrichment > {fold_enrichment_threshold}")
        print(f"  Before: {len(fold_enrichments)} regions")
        print(f"  After: {valid_mask.sum()} regions")
        
        igs = igs[valid_mask]
        region_lengths = region_lengths[valid_mask]
        fold_enrichments = fold_enrichments[valid_mask]
    else:
        valid_mask = np.ones(len(fold_enrichments), dtype=bool)
    
    n_regions = igs.shape[0]
    n_tissues = igs.shape[3]
    
    computed_metrics = {}
    
    for metric in metrics:
        print(f"  Computing {metric}...")
        metric_values = np.zeros((n_regions, n_tissues))
        
        for ri in range(n_regions):
            region_len = region_lengths[ri]
            region_igs = igs[ri, :region_len, :, :]  # (region_len, 4, n_tissues)
            
            if metric == 'sum_abs':
                # Sum of absolute values across sequence and nucleotides
                metric_values[ri, :] = np.sum(np.abs(region_igs), axis=(0, 1))
            
            elif metric == 'max_abs':
                # Maximum absolute value across sequence and nucleotides
                metric_values[ri, :] = np.max(np.abs(region_igs), axis=(0, 1))
            
            elif metric == 'mean_abs':
                # Mean absolute value across sequence and nucleotides
                metric_values[ri, :] = np.mean(np.abs(region_igs), axis=(0, 1))
            
            elif metric == 'l2_norm':
                # L2 norm across sequence and nucleotides
                metric_values[ri, :] = np.sqrt(np.sum(region_igs**2, axis=(0, 1)))
            
            else:
                raise ValueError(f"Unknown metric: {metric}")
        
        computed_metrics[metric] = metric_values
    
    # Store filtered fold enrichments and mask
    computed_metrics['fold_enrichment'] = fold_enrichments
    computed_metrics['valid_mask'] = valid_mask
    
    return computed_metrics


def compute_correlations(computed_metrics, metrics, tissue_mapping, tissue_indices, min_samples=10):
    """Compute correlations between IG metrics and fold enrichment.
    
    Args:
        computed_metrics: Dictionary with computed metrics
        metrics: List of metrics
        tissue_mapping: Dictionary mapping tissue index to name
        tissue_indices: Array of tissue indices
        min_samples: Minimum samples for correlation
    
    Returns:
        DataFrame with correlation results
    """
    print("\nComputing correlations...")
    
    fold_enrichments = computed_metrics['fold_enrichment']
    results = []
    
    for metric in metrics:
        if metric not in computed_metrics:
            print(f"  Skipping {metric}: not computed")
            continue
        
        print(f"  Processing {metric}...")
        metric_data = computed_metrics[metric]  # (n_regions, n_tissues)
        
        tissue_count = 0
        for ti, tissue_idx in enumerate(tissue_indices):
            tissue_values = metric_data[:, ti]
            
            # Remove NaN/Inf
            valid_mask = np.isfinite(tissue_values) & np.isfinite(fold_enrichments)
            tissue_valid = tissue_values[valid_mask]
            fe_valid = fold_enrichments[valid_mask]
            
            if len(tissue_valid) < min_samples:
                continue
            
            # Check for constant values
            if len(np.unique(tissue_valid)) == 1:
                continue
            
            # Compute correlations
            try:
                pearson_r, pearson_p = pearsonr(tissue_valid, fe_valid)
                spearman_r, spearman_p = spearmanr(tissue_valid, fe_valid)
                
                if not np.isfinite(pearson_r) or not np.isfinite(spearman_r):
                    continue
                
            except:
                continue
            
            tissue_name = tissue_mapping.get(int(tissue_idx), f"Tissue_{tissue_idx}")
            
            results.append({
                'metric': metric,
                'tissue_idx': int(tissue_idx),
                'tissue_name': tissue_name,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'n_samples': len(tissue_valid)
            })
            tissue_count += 1
        
        print(f"    Computed {tissue_count} tissue correlations")
    
    results_df = pd.DataFrame(results)
    print(f"Total correlations: {len(results_df)}")
    
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
    
    # 2. Create heatmaps
    print("  Creating correlation heatmaps...")
    
    # Get unique tissues sorted by index
    tissue_names = sorted(results_df['tissue_name'].unique(),
                         key=lambda x: int(results_df[results_df['tissue_name']==x]['tissue_idx'].iloc[0]))
    
    # Filter metrics that actually have results
    available_metrics = [m for m in metrics if m in results_df['metric'].unique()]
    
    if len(available_metrics) == 0:
        print("  WARNING: No metrics available for visualization")
        return
    
    # Prepare heatmap data matrices
    pearson_data = np.zeros((len(tissue_names), len(available_metrics)))
    spearman_data = np.zeros((len(tissue_names), len(available_metrics)))
    
    for i, tissue in enumerate(tissue_names):
        for j, metric in enumerate(available_metrics):
            mask = (results_df['tissue_name'] == tissue) & (results_df['metric'] == metric)
            if mask.any():
                pearson_data[i, j] = results_df[mask]['pearson_r'].iloc[0]
                spearman_data[i, j] = results_df[mask]['spearman_r'].iloc[0]
    
    # Function to create heatmap
    def create_correlation_heatmap(data, corr_type, output_path):
        fig = plt.figure(figsize=(max(8, len(available_metrics) * 1.5), 28))
        
        ax = fig.add_axes([0.25, 0.1, 0.55, 0.82])
        cax = fig.add_axes([0.82, 0.1, 0.03, 0.82])
        
        vmax = max(abs(data.min()), abs(data.max()))
        vmax = max(vmax, 0.05)
        
        im = ax.imshow(data, cmap='RdBu_r', aspect='auto',
                      vmin=-vmax, vmax=vmax, interpolation='nearest')
        
        ax.set_xticks(range(len(available_metrics)))
        ax.set_xticklabels(available_metrics, fontsize=13, fontweight='bold', rotation=0)
        ax.set_yticks(range(len(tissue_names)))
        ax.set_yticklabels([t.replace('_', ' ') for t in tissue_names], fontsize=9.5)
        
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        
        # Add gridlines
        for i in range(len(tissue_names) + 1):
            ax.axhline(i - 0.5, color='white', linewidth=1.5, alpha=0.8)
        for j in range(len(available_metrics) + 1):
            ax.axvline(j - 0.5, color='white', linewidth=1.5, alpha=0.8)
        
        # Add text annotations
        for i in range(len(tissue_names)):
            for j in range(len(available_metrics)):
                r_val = data[i, j]
                text_color = 'white' if abs(r_val) > vmax * 0.5 else 'black'
                ax.text(j, i, f'{r_val:.3f}',
                       ha='center', va='center', fontsize=8,
                       color=text_color, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(f'{corr_type} r', fontsize=12, fontweight='bold',
                      rotation=270, labelpad=20)
        cbar.ax.tick_params(labelsize=10)
        
        # Title
        fig.text(0.525, 0.97,
                f'{corr_type} Correlation: IG Metrics vs STARRseq Fold Enrichment',
                ha='center', fontsize=14, fontweight='bold')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
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
    for metric in available_metrics:
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
    fig, ax = plt.subplots(figsize=(12, max(4, len(available_metrics) * 0.5)))
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
    
    plt.title('IG Metrics Correlation Summary', fontsize=14, fontweight='bold', pad=20)
    
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
    
    if args.tissues_json and not os.path.exists(args.tissues_json):
        print(f"WARNING: Tissues JSON not found: {args.tissues_json}")
        args.tissues_json = None
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("Integrated Gradients Correlation Analysis")
    print("="*70)
    print(f"IGs H5:       {args.h5}")
    print(f"Tissues JSON: {args.tissues_json}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Metrics:      {', '.join(args.metrics)}")
    print("="*70)
    
    # Load data
    igs_data = load_igs_data(args.h5)
    tissue_mapping = load_tissue_mapping(igs_data['tissue_indices'], args.tissues_json)
    
    # Compute metrics
    computed_metrics = compute_ig_metrics(
        igs_data, args.metrics, args.fold_enrichment_threshold
    )
    
    if len(computed_metrics['fold_enrichment']) < args.min_samples:
        print(f"\nERROR: Insufficient samples ({len(computed_metrics['fold_enrichment'])} < {args.min_samples})")
        sys.exit(1)
    
    # Compute correlations
    results_df = compute_correlations(
        computed_metrics, args.metrics, tissue_mapping,
        igs_data['tissue_indices'], args.min_samples
    )
    
    if len(results_df) == 0:
        print("\nERROR: No correlations computed")
        sys.exit(1)
    
    # Visualize
    visualize_correlations(results_df, args.metrics, args.output_dir)
    
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

