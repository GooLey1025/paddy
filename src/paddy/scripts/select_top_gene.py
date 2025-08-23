#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Select top differentially expressed genes per tissue",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "expression_file",
        help="Input expression file (all_34samples_trainset.ATG_UD16K.exp)"
    )
    
    parser.add_argument(
        "-o", "--out_dir",
        default="diff_expr",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "-n", "--num_genes",
        type=int,
        default=1000,
        help="Number of top differentially expressed genes to select per tissue"
    )
    
    parser.add_argument(
        "--pseudo_count",
        type=float,
        default=0.01,
        help="Pseudo count to add to expression values to avoid zeros"
    )
    
    parser.add_argument(
        "--clip_percentile",
        type=float,
        default=99.0,
        help="Percentile to clip expression values (to avoid extreme outliers)"
    )
    
    parser.add_argument(
        "--tissue_names",
        nargs="+",
        default=None,
        help="List of tissue names (if not provided, will use column indices)"
    )
    
    parser.add_argument(
        "--tissue_indices",
        nargs="+",
        type=str,
        default=None,
        help="List of tissue indices to generate top gene files for (0-based). Can be space-separated or comma-separated. If specified, generates top gene files only for these tissues"
    )
    
    return parser.parse_args()

def load_expression_data(file_path):
    """Load expression data from file."""
    print(f"Loading expression data from {file_path}...")
    
    # Read the expression file
    # Format: sample_id gene_id chr gene_name strand expr1 expr2 ... expr23
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 28:  # 5 metadata columns + 23 expression columns
                sample_id = parts[0]
                gene_id = parts[1]
                chr_name = parts[2]
                gene_name = parts[3]
                strand = parts[4]
                expression_values = [float(x) for x in parts[5:]]  # 23 expression values (all remaining columns)
                
                data.append({
                    'sample_id': sample_id,
                    'gene_id': gene_id,
                    'chr': chr_name,
                    'gene_name': gene_name,
                    'strand': strand,
                    'expression': expression_values
                })
    
    print(f"Loaded {len(data)} genes")
    return data

def preprocess_expression_data(data, pseudo_count=0.01, clip_percentile=99.0) -> list[dict]:
    """Preprocess expression data by adding pseudo counts and clipping outliers."""
    print("Preprocessing expression data...")
    
    # Extract expression values as numpy array
    expr_matrix = np.array([row['expression'] for row in data])
    
    # Add pseudo count to avoid zeros
    expr_matrix += pseudo_count
    
    # Clip outliers per tissue (column)
    for col in range(expr_matrix.shape[1]):
        percentile_value = np.percentile(expr_matrix[:, col], clip_percentile)
        expr_matrix[:, col] = np.clip(expr_matrix[:, col], 0, percentile_value)
    
    # Update the data with processed expression values
    for i, row in enumerate(data):
        row['expression'] = expr_matrix[i].tolist()
    
    print(f"Added pseudo count of {pseudo_count}")
    print(f"Clipped values at {clip_percentile}th percentile")
    
    return data

def compute_tissue_specific_genes(data, num_genes=1000, tissue_names=None) -> dict[str, pd.DataFrame]:
    """Compute tissue-specific differentially expressed genes."""
    print("Computing tissue-specific differentially expressed genes...")
    
    # Convert to numpy array for computation
    expr_matrix = np.array([row['expression'] for row in data])
    num_tissues = expr_matrix.shape[1]
    
    # Generate tissue names if not provided (0-based indexing)
    if tissue_names is None:
        tissue_names = [f"tissue_{i}" for i in range(num_tissues)]
    
    print(f"Analyzing {num_tissues} tissues: {tissue_names}")
    
    tissue_specific_genes = {}
    
    for tissue_idx in range(num_tissues):
        tissue_name = tissue_names[tissue_idx]
        print(f"Processing {tissue_name}...")
        
        # Get expression for current tissue
        tissue_expr = expr_matrix[:, tissue_idx]
        
        # Get expression for all other tissues
        other_tissues_expr = np.delete(expr_matrix, tissue_idx, axis=1)
        other_tissues_mean = np.mean(other_tissues_expr, axis=1)
        
        # Compute log fold change
        # log2(tissue_expr / other_tissues_mean)
        log_fc = np.log2(tissue_expr / (other_tissues_mean + 1e-10))
        
        # Create DataFrame with gene info and log fold change
        gene_df = pd.DataFrame({
            'sample_id': [row['sample_id'] for row in data],
            'gene_id': [row['gene_id'] for row in data],
            'chr': [row['chr'] for row in data],
            'gene_name': [row['gene_name'] for row in data],
            'strand': [row['strand'] for row in data],
            'tissue_expr': tissue_expr,
            'other_tissues_mean': other_tissues_mean,
            'log_fold_change': log_fc
        })
        
        # Sort by log fold change (descending) and select top genes
        gene_df_sorted = gene_df.sort_values('log_fold_change', ascending=False)
        top_genes = gene_df_sorted.head(num_genes)
        
        tissue_specific_genes[tissue_name] = top_genes
        
        print(f"  Selected {len(top_genes)} genes for {tissue_name}")
        print(f"  Log fold change range: {top_genes['log_fold_change'].min():.3f} to {top_genes['log_fold_change'].max():.3f}")
    
    return tissue_specific_genes

def generate_top_genes_for_specified_tissues(data, tissue_indices, tissue_names=None, num_genes=1000) -> dict[str, pd.DataFrame]:
    """Generate top gene files for specified tissue indices with top gene selection."""
    print(f"Generating top gene files for specified tissue indices: {tissue_indices}...")
    
    # Convert to numpy array for computation
    expr_matrix = np.array([row['expression'] for row in data])
    all_num_tissues = expr_matrix.shape[1]
      
    # Generate tissue names if not provided (0-based indexing)
    if tissue_names is None:
        tissue_names = [f"tissue_{i}" for i in range(all_num_tissues)]

    valid_indices: list[int] = [idx for idx in tissue_indices if 0 <= idx < all_num_tissues]
    print(f"Valid indices: {valid_indices}")
    
    
    tissue_files = {}
    
    for tissue_idx in valid_indices:
        tissue_name = tissue_names[tissue_idx]
        print(f"Processing {tissue_name} (index {tissue_idx})...")
        
        # Get expression for current tissue
        curr_tissue_expr = expr_matrix[:, tissue_idx]
        
        # Get expression for all other tissues
        other_tissues_expr = expr_matrix[:, valid_indices]
        other_tissues_mean = np.mean(other_tissues_expr, axis=1)
        
        # Compute log fold change
        log_fc = np.log2(curr_tissue_expr / (other_tissues_mean + 1e-10))
        
        # Create DataFrame with gene info and log fold change
        gene_df = pd.DataFrame({
            'sample_id': [row['sample_id'] for row in data],
            'gene_id': [row['gene_id'] for row in data],
            'chr': [row['chr'] for row in data],
            'gene_name': [row['gene_name'] for row in data],
            'strand': [row['strand'] for row in data],
            'tissue_expr': curr_tissue_expr,
            'other_tissues_mean': other_tissues_mean,
            'log_fold_change': log_fc
        })
        
        # Sort by log fold change (descending) and select top genes
        gene_df_sorted = gene_df.sort_values('log_fold_change', ascending=False)
        top_genes = gene_df_sorted.head(num_genes)
        
        tissue_files[tissue_name] = top_genes
        
        print(f"  Selected {len(top_genes)} top genes for {tissue_name}")
        print(f"  Log fold change range: {top_genes['log_fold_change'].min():.3f} to {top_genes['log_fold_change'].max():.3f}")
    
    return tissue_files

def save_results(tissue_data, out_dir, is_direct_mode=False) -> pd.DataFrame:
    """Save results to output directory."""
    print(f"Saving results to {out_dir}...")
    
    os.makedirs(out_dir, exist_ok=True)
    combined_genes = []

    # Save individual tissue files
    for tissue_name, genes_df in tissue_data.items():
        output_file = os.path.join(out_dir, f"{tissue_name}_top_genes.tsv")
        genes_df.to_csv(output_file, sep='\t', index=False)
        print(f"  Saved {len(genes_df)} genes to {output_file}")

        tissue_id = int(tissue_name.split('_')[1]) if tissue_name.startswith('tissue_') else tissue_name
        genes_df['tissue'] = tissue_name
        genes_df['tissue_id'] = tissue_id
        combined_genes.append(genes_df)
  
    if is_direct_mode:
        combined_file = os.path.join(out_dir, "selected_tissues_top_genes.tsv")
    else:
        combined_file = os.path.join(out_dir, "all_tissue_specific_genes.tsv")
    combined_df = pd.concat(combined_genes, ignore_index=True)
    combined_df.to_csv(combined_file, sep='\t', index=False)
    print(f"  Saved {len(combined_df)} total genes to {combined_file}")
    
    # Save summary statistics as TSV file
    if is_direct_mode:
        summary_file = os.path.join(out_dir, "selected_tissues_summary.tsv")
    else:
        summary_file = os.path.join(out_dir, "summary_statistics.tsv")
    
    # Create summary data
    summary_data = []
    for tissue_name, genes_df in tissue_data.items():
        # Extract tissue_id from tissue_name (e.g., "tissue_0" -> 0)
        tissue_id = int(tissue_name.split('_')[1]) if tissue_name.startswith('tissue_') else tissue_name
        summary_data.append({
            'tissue': tissue_name,
            'tissue_id': tissue_id,
            'num_genes': len(genes_df),
            'min_log_fold_change': genes_df['log_fold_change'].min(),
            'max_log_fold_change': genes_df['log_fold_change'].max(),
            'mean_log_fold_change': genes_df['log_fold_change'].mean(),
            'median_log_fold_change': genes_df['log_fold_change'].median(),
            'std_log_fold_change': genes_df['log_fold_change'].std(),
            'q25_log_fold_change': genes_df['log_fold_change'].quantile(0.25),
            'q75_log_fold_change': genes_df['log_fold_change'].quantile(0.75)
        })
    
    # Convert to DataFrame and sort by mean log fold change (descending)
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('mean_log_fold_change', ascending=False)
    
    # Save as TSV
    summary_df.to_csv(summary_file, sep='\t', index=False, float_format='%.4f')
    
    print(f"  Saved summary statistics to {summary_file}")
    
    # Display different messages based on mode
    if is_direct_mode:
        print(f"\nSelected {len(summary_df)} tissues summary (sorted by mean log fold change):")
        print("-" * 70)
        for i, (_, row) in enumerate(summary_df.iterrows(), 1):
            print(f"{i}. {row['tissue']}: {row['mean_log_fold_change']:.4f} (±{row['std_log_fold_change']:.4f})")
    else:
        print("\nTop 5 tissues by mean log fold change:")
        print("-" * 60)
        top_5 = summary_df.head(5)
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            print(f"{i}. {row['tissue']}: {row['mean_log_fold_change']:.4f} (±{row['std_log_fold_change']:.4f})")
    
    return summary_df

def main():
    """Main function."""
    args = parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.expression_file):
        print(f"Error: Input file {args.expression_file} not found")
        sys.exit(1)
    
    # Load expression data
    data = load_expression_data(args.expression_file)
    
    # Preprocess data
    data = preprocess_expression_data(data, args.pseudo_count, args.clip_percentile)
    
    # Check if we're in direct tissue mode or top gene selection mode
    if args.tissue_indices is not None:
        print(f"Selected tissues mode: Generating top gene files for specified tissue indices: {args.tissue_indices}")
        # Parse comma-separated tissue indices into a list of integers
        tissue_indices_list = []
        for item in args.tissue_indices:
            if ',' in item:
                tissue_indices_list.extend([int(i) for i in item.split(',')])
            else:
                tissue_indices_list.append(int(item))
        
        # Generate top gene files for specified tissues
        tissue_data = generate_top_genes_for_specified_tissues(data, tissue_indices_list, args.tissue_names, args.num_genes)
        
        if not tissue_data:
            print("Error: No valid tissue data generated")
            sys.exit(1)
        
        # Save results in specified tissue mode
        save_results(tissue_data, args.out_dir, is_direct_mode=True)
        
        print("\nTop gene selection for selected tissues completed successfully!")
        print(f"Results saved to: {args.out_dir}")
        print(f"Summary TSV file: {args.out_dir}/selected_tissues_summary.tsv")
        
    else:
        print("All tissues mode: Computing tissue-specific differentially expressed genes for all tissues")
        # Compute tissue-specific genes for all tissues (original functionality)
        tissue_data = compute_tissue_specific_genes(
            data, args.num_genes, args.tissue_names
        )
        
        # Save results in all tissues mode
        summary_df = save_results(tissue_data, args.out_dir, is_direct_mode=False)
        
        print("\nTissue-specific gene selection for all tissues completed successfully!")
        print(f"Results saved to: {args.out_dir}")
        print(f"Summary TSV file: {args.out_dir}/summary_statistics.tsv")

if __name__ == "__main__":
    main()
