#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
import sys
import pandas as pd
import subprocess
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate GFF files for top tissue-specific genes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--tissue_indices",
        required=True,
        help="Comma-separated tissue indices (e.g., 19,1,23,20,21)"
    )
    
    parser.add_argument(
        "--data_dir",
        default="diff_expr",
        help="Directory containing tissue-specific gene files"
    )
    
    parser.add_argument(
        "--gtf",
        required=True,
        help="Reference GTF/GFF3 file (e.g., Rice_MSUv7.gff3)"
    )
    
    parser.add_argument(
        "-o", "--out_dir",
        default="selected_genes_gff",
        help="Output directory for GFF files"
    )
    
    parser.add_argument(
        "--gene_id_column",
        default="gene_id",
        help="Column name containing gene IDs in the TSV files"
    )
    
    return parser.parse_args()

def load_tissue_genes(data_dir, tissue_indices, gene_id_column):
    """Load gene lists for specified tissues."""
    print("Loading tissue-specific gene lists...")
    
    tissue_genes = {}
    all_genes = set()
    
    for tissue_idx in tissue_indices:
        tissue_file = os.path.join(data_dir, f"tissue_{tissue_idx}_top_genes.tsv")
        
        if not os.path.exists(tissue_file):
            print(f"Warning: File not found: {tissue_file}")
            continue
        
        try:
            # Load the TSV file
            df = pd.read_csv(tissue_file, sep='\t')
            
            if gene_id_column not in df.columns:
                print(f"Error: Column '{gene_id_column}' not found in {tissue_file}")
                print(f"Available columns: {list(df.columns)}")
                continue
            
            # Extract gene IDs
            gene_ids = df[gene_id_column].tolist()
            tissue_genes[tissue_idx] = gene_ids
            all_genes.update(gene_ids)
            
            print(f"  Loaded {len(gene_ids)} genes for tissue_{tissue_idx}")
            
        except Exception as e:
            print(f"Error loading {tissue_file}: {e}")
            continue
    
    print(f"Total unique genes across all tissues: {len(all_genes)}")
    return tissue_genes, all_genes

def extract_genes_from_gtf(gtf_file, gene_ids, output_file):
    """Extract specific genes from GTF file using grep."""
    print(f"Extracting {len(gene_ids)} genes from {gtf_file}...")
    
    if not gene_ids:
        print("No genes to extract")
        return False
    
    try:
        # Create a temporary file with gene IDs for grep
        temp_gene_file = output_file + ".temp_genes"
        with open(temp_gene_file, 'w') as f:
            for gene_id in gene_ids:
                f.write(f"{gene_id}\n")
        
        # Use grep to extract matching lines
        with open(output_file, 'w') as out_f:
            # Use grep with -f flag to search for multiple patterns from file
            cmd = ['grep', '-f', temp_gene_file, gtf_file]
            result = subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE, text=True)
            
            if result.returncode != 0:
                print(f"Warning: grep command failed or found no matches")
                print(f"Error: {result.stderr}")
        
        # Clean up temporary file
        os.remove(temp_gene_file)
        
        # Check if output file has content
        if os.path.getsize(output_file) > 0:
            # Count lines in output
            with open(output_file, 'r') as f:
                line_count = sum(1 for line in f)
            print(f"  Extracted {line_count} lines to {output_file}")
            return True
        else:
            print(f"  No matching genes found in GTF file")
            return False
            
    except Exception as e:
        print(f"Error extracting genes: {e}")
        return False

def create_tissue_gff_files(tissue_genes, gtf_file, out_dir):
    """Create individual GFF files for each tissue."""
    print("Creating tissue-specific GFF files...")
    
    os.makedirs(out_dir, exist_ok=True)
    created_files = []
    
    for tissue_idx, gene_ids in tissue_genes.items():
        output_file = os.path.join(out_dir, f"tissue_{tissue_idx}_top_genes.gff3")
        
        if extract_genes_from_gtf(gtf_file, gene_ids, output_file):
            created_files.append(output_file)
            print(f"  Created: {output_file}")
        else:
            print(f"  Failed to create: {output_file}")
    
    return created_files

def create_combined_gff_file(all_genes, gtf_file, out_dir):
    """Create a combined GFF file with all selected genes."""
    print("Creating combined GFF file...")
    
    output_file = os.path.join(out_dir, "all_selected_genes.gff3")
    
    if extract_genes_from_gtf(gtf_file, list(all_genes), output_file):
        print(f"  Created combined file: {output_file}")
        return output_file
    else:
        print(f"  Failed to create combined file")
        return None

def generate_summary_report(tissue_genes, all_genes, created_files, out_dir):
    """Generate a summary report of the extraction process."""
    summary_file = os.path.join(out_dir, "extraction_summary.txt")
    
    with open(summary_file, 'w') as f:
        f.write("Tissue-Specific Gene GFF Extraction Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total tissues processed: {len(tissue_genes)}\n")
        f.write(f"Total unique genes: {len(all_genes)}\n")
        f.write(f"GFF files created: {len(created_files)}\n\n")
        
        f.write("Tissue details:\n")
        for tissue_idx, gene_ids in tissue_genes.items():
            f.write(f"  Tissue {tissue_idx}: {len(gene_ids)} genes\n")
        
        f.write("\nCreated files:\n")
        for file_path in created_files:
            f.write(f"  {file_path}\n")
    
    print(f"Summary report saved to: {summary_file}")

def main():
    """Main function."""
    args = parse_args()
    
    # Parse tissue indices
    try:
        tissue_indices = [int(x.strip()) for x in args.tissue_indices.split(',')]
        print(f"Processing tissues: {tissue_indices}")
    except ValueError:
        print("Error: Invalid tissue indices. Please provide comma-separated integers.")
        sys.exit(1)
    
    # Check if input files exist
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.gtf):
        print(f"Error: GTF file not found: {args.gtf}")
        sys.exit(1)
    
    # Load tissue-specific gene lists
    tissue_genes, all_genes = load_tissue_genes(
        args.data_dir, tissue_indices, args.gene_id_column
    )
    
    if not tissue_genes:
        print("Error: No tissue gene files found or loaded successfully")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Create tissue-specific GFF files
    created_files = create_tissue_gff_files(tissue_genes, args.gtf, args.out_dir)
    
    # Create combined GFF file
    combined_file = create_combined_gff_file(all_genes, args.gtf, args.out_dir)
    created_files.append(combined_file)
    
    # Generate summary report
    generate_summary_report(tissue_genes, all_genes, created_files, args.out_dir)
    
    print(f"\nGFF extraction completed successfully!")
    print(f"Output directory: {args.out_dir}")

if __name__ == "__main__":
    main()
