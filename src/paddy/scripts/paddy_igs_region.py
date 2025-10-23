#!/usr/bin/env python
"""
paddy_igs_region.py

Compute Integrated Gradients for specific genomic regions of interest.
For each region in CSV, finds the nearest gene within 16kb flanking region,
extracts 32kb sequence centered on ATG, computes integrated gradients,
and saves only the gradients for the specific region of interest.
"""

from __future__ import print_function
from optparse import OptionParser
import gc
import yaml
import os
import sys
import h5py
import numpy as np
import pandas as pd
import pysam
import tensorflow as tf
from collections import defaultdict

from paddy import dna as dna_io
from paddy import gene as pgene
from paddy import seqnn


def parse_args():
    """Parse command line arguments."""
    usage = "usage: %prog [options] <params_file> <regions_csv> <genome_fasta> <genes_gff>"
    parser = OptionParser(usage)
    
    parser.add_option(
        "-o",
        dest="out_file",
        default="igs_region_out.h5",
        help="Output h5 file [Default: %default]",
    )
    parser.add_option(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Ensemble forward and reverse complement predictions [Default: %default]",
    )
    parser.add_option(
        "--model_path",
        dest="model_path",
        default=None,
        type="str",
        help="Direct path to model file (e.g., model_best.h5) [Default: %default]",
    )
    parser.add_option(
        "--head",
        dest="head_i",
        default=0,
        type="int",
        help="Model head index [Default: %default]",
    )
    parser.add_option(
        "--shifts",
        dest="shifts",
        default="0",
        type="str",
        help="Ensemble prediction shifts [Default: %default]",
    )
    parser.add_option(
        "--batch_size",
        dest="batch_size",
        default=8,
        type="int",
        help="Batch size for gradient computation [Default: %default]",
    )
    parser.add_option(
        "--num_steps",
        dest="num_steps",
        default=50,
        type="int",
        help="Number of interpolation steps for Integrated Gradients. Linearly increase the running time. [Default: %default]",
    )
    parser.add_option(
        "--baseline",
        dest="baseline_type",
        default="zero",
        type="str",
        help="Baseline type: 'zero', 'uniform', or 'gc' [Default: %default]",
    )
    parser.add_option(
        "--gc_content",
        dest="gc_content",
        default=0.435408,
        type="float",
        help="GC content for 'gc' baseline (rice default: 0.435408) [Default: %default]",
    )
    parser.add_option(
        "--track_scale",
        dest="track_scale",
        default=1.0,
        type="float",
        help="Target transform scale [Default: %default]",
    )
    parser.add_option(
        "--no_untransform",
        dest="no_untransform",
        default=False,
        action="store_true",
        help="Skip untransformation of predictions [Default: %default]",
    )
    parser.add_option(
        "--tissue_indices",
        dest="tissue_indices",
        default=None,
        type="str",
        help="Comma-separated tissue indices (e.g., '0,1,2'). If None, compute all [Default: %default]",
    )
    parser.add_option(
        "--tissue_slab",
        dest="tissue_slab",
        default=1,
        type="int",
        help="Number of tissues to process at once [Default: %default]",
    )
    parser.add_option(
        "--max_flanking",
        dest="max_flanking",
        default=16384,
        type="int",
        help="Maximum flanking distance (bp) to search for nearest gene. Usually, it is your seq_len/2. [Default: %default]",
    )
    parser.add_option(
        "--regions_csv",
        dest="regions_csv",
        default=None,
        type="str",
        help="CSV file containing regions of interest. Must have columns header: chr, start, end. (Required)",
    )
    
    return parser.parse_args()


def load_regions_csv(csv_path, genome_open):
    """Load regions of interest from CSV file and filter out regions that are too close to the edges of the genome.
    
    Args:
        csv_path: Path to CSV file
        genome_open: pysam.Fastafile object
    
    Required columns: chr, start, end
    Optional columns: fold_enrichment (and others)
    
    Returns:
        Tuple of (DataFrame with regions, has_fold_enrichment)
    """
    print(f"\nLoading regions from {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(csv_path)
    
    # Clean column names
    df.columns = df.columns.str.strip().str.lstrip('\ufeff')
    
    print(f"Loaded {len(df)} regions")
    print(f"Columns: {', '.join(df.columns)}")
    
    # Check for required columns
    required_cols = ['chr', 'start', 'end']
    for col in required_cols:
        if col not in df.columns:
            # Try case-insensitive match
            found = False
            for df_col in df.columns:
                if df_col.lower() == col.lower():
                    df.rename(columns={df_col: col}, inplace=True)
                    found = True
                    break
            if not found:
                raise ValueError(f"CSV must have '{col}' column")
    
    # Check for optional fold_enrichment column
    has_fold_enrichment = False
    if 'fold_enrichment' in df.columns:
        has_fold_enrichment = True
        print("  Found 'fold_enrichment' column - will be saved to output")
    else:
        # Try case-insensitive match
        for df_col in df.columns:
            if df_col.lower() == 'fold_enrichment':
                df.rename(columns={df_col: 'fold_enrichment'}, inplace=True)
                has_fold_enrichment = True
                print("  Found 'fold_enrichment' column - will be saved to output")
                break
        
        if not has_fold_enrichment:
            print("  'fold_enrichment' column not found - will not be saved to output")
    
    # Standardize chr column
    df['chr'] = df['chr'].astype(str).str.replace('chr', '', regex=False)
    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)

    raw_len = len(df)
    # Filter out regions that are too close to the edges of the genome
    chrom_lengths = dict(zip(genome_open.references, genome_open.lengths))
    df = df[df['start'] >= 16384]
    df = df[df.apply(lambda row: row['end'] <= chrom_lengths[row['chr']] - 16384, axis=1)]

    print(f"Filtered out {raw_len - len(df)} regions that are too close to the edges of the genome")

    return df, has_fold_enrichment


def build_gene_index(transcriptome):
    """Build a spatial index for genes by chromosome and ATG position.
    
    Args:
        transcriptome: Transcriptome object with genes
    
    Returns:
        Dictionary mapping chromosome to sorted list of (atg_pos, gene_id, gene_obj)
    """
    print("Building gene spatial index...")
    gene_index = defaultdict(list)
    
    skipped_genes = 0
    for gene_id, gene in transcriptome.genes.items():
        # Get gene ATG position
        if gene.strand == "+":
            atg_pos = gene.cds_start()
        else:
            atg_pos = gene.cds_end()
        
        # Check if atg_pos is valid
        if atg_pos is None or atg_pos < 0:
            skipped_genes += 1
            continue
        
        gene_index[gene.chrom].append((atg_pos, gene_id, gene))
    
    # Sort by ATG position for binary search
    for chrom in gene_index:
        gene_index[chrom].sort(key=lambda x: x[0])
    
    total_genes = sum(len(genes) for genes in gene_index.values())
    print(f"  Indexed {total_genes} genes across {len(gene_index)} chromosomes")
    if skipped_genes > 0:
        print(f"  Skipped {skipped_genes} genes without valid CDS")
    
    return gene_index


def find_nearest_gene_fast(region_chr, region_start, region_end, gene_index, max_flanking=16384):
    """Find the nearest gene to a region using spatial index.
    
    The entire region (both start and end) must be within max_flanking distance from gene ATG.
    Uses binary search for O(log n) lookup per region instead of O(n).
    
    Args:
        region_chr: Chromosome of region
        region_start: Start position of region
        region_end: End position of region
        gene_index: Pre-built gene index from build_gene_index()
        max_flanking: Maximum distance from gene ATG to consider
    
    Returns:
        Tuple of (gene_id, gene_obj, distance) or (None, None, None) if no gene found
    """
    if region_chr not in gene_index:
        return None, None, None
    
    genes_on_chrom = gene_index[region_chr]
    
    # Binary search to find genes near the region
    # Search range: genes whose ATG could be within max_flanking of region start or end
    search_start = region_start - max_flanking
    search_end = region_end + max_flanking
    
    # Find insertion points using binary search
    import bisect
    left_idx = bisect.bisect_left(genes_on_chrom, (search_start,))
    right_idx = bisect.bisect_right(genes_on_chrom, (search_end,))
    
    # Check only genes in the candidate range
    nearest_gene_id = None
    nearest_gene = None
    min_distance = float('inf')
    
    for i in range(left_idx, right_idx):
        atg_pos, gene_id, gene = genes_on_chrom[i]
        
        # Check if ENTIRE region is within flanking distance from ATG
        distance_to_start = abs(region_start - atg_pos)
        distance_to_end = abs(region_end - atg_pos)
        
        # The entire region must be within max_flanking
        if distance_to_start <= max_flanking and distance_to_end <= max_flanking:
            distance = max(abs(distance_to_start), abs(distance_to_end))
            if distance < min_distance:
                min_distance = distance
                nearest_gene_id = gene_id
                nearest_gene = gene
    
    if nearest_gene_id is None:
        return None, None, None
    
    return nearest_gene_id, nearest_gene, min_distance


def make_seq_1hot(genome_open, chrm, start, end, seq_len):
    """Extract sequence and convert to one-hot encoding."""
    if start < 0:
        seq_dna = "N" * (-start) + genome_open.fetch(chrm, 0, end)
    else:
        seq_dna = genome_open.fetch(chrm, start, end)
    
    # Extend to full length if needed
    if len(seq_dna) < seq_len:
        seq_dna += "N" * (seq_len - len(seq_dna))
    
    seq_1hot = dna_io.dna_1hot(seq_dna)
    return seq_1hot

def make_seq_1hot_co_interval(genome_open, chrm, start, end, seq_len, strand, atg_not_found_count=0):
    """Extract sequence and convert to one-hot encoding. Assumes start and end are 1-based coordinate."""
    start = start - 1
    end = end - 1
    if start < 0:
        seq_dna = "N" * (-start) + genome_open.fetch(chrm, 0, end)
    else:
        seq_dna = genome_open.fetch(chrm, start, end)
    # Extend to full length if needed
    if len(seq_dna) < seq_len:
        print(f"Warning: Extending sequence to full length: {seq_len} - {len(seq_dna)} = {seq_len - len(seq_dna)} nucleotides")
        seq_dna += "N" * (seq_len - len(seq_dna))
    if strand == "-":
        seq_dna = dna_io.dna_rc(seq_dna)
    seq_1hot = dna_io.dna_1hot(seq_dna)

    if seq_dna[16384:16387].upper() != "ATG":
        print(f"Warning: ATG not found at position chr:start-end(1-based) (strand: {strand}): {chrm}:{start+1}-{end+1}, instead {seq_dna[16384:16387]}")
    atg_not_found_count += 1
        

    return seq_1hot, atg_not_found_count

def unaugment_grads(grads, fwdrc=True, shift=0):
    """Undo sequence augmentation."""
    # Handle different gradient array shapes
    if len(grads.shape) == 2:
        grads = grads[:, :, np.newaxis]
        squeeze_output = True
    elif len(grads.shape) == 3:
        squeeze_output = False
    else:
        raise ValueError(f"Unexpected gradient shape: {grads.shape}")
    
    # Reverse complement
    if not fwdrc:
        grads = grads[::-1, :, :]
        grads[:, [0, 3], :] = grads[:, [3, 0], :]
        grads[:, [1, 2], :] = grads[:, [2, 1], :]
    
    # Undo shift
    if shift < 0:
        grads[-shift:, :, :] = grads[:shift, :, :]
        grads[:-shift, :, :] = 0
    elif shift > 0:
        grads[:-shift, :, :] = grads[shift:, :, :]
        grads[-shift:, :, :] = 0
    
    if squeeze_output and len(grads.shape) == 3 and grads.shape[-1] == 1:
        grads = grads[:, :, 0]
    
    return grads


def main():
    options, args = parse_args()
    
    if len(args) != 3:
        print("ERROR: Must provide params_file, genome_fasta, genes_gff")
        sys.exit(1)
    
    params_file = args[0]
    genome_fasta = args[1]
    genes_gff = args[2]
    
    if options.model_path is None:
        print("ERROR: --model_path is required")
        sys.exit(1)
    
    if options.regions_csv is None:
        print("ERROR: --regions_csv is required")
        sys.exit(1)
    
    regions_csv = options.regions_csv
    
    if not os.path.isfile(options.model_path):
        print(f"ERROR: Model file not found: {options.model_path}")
        sys.exit(1)
    
    options.shifts = [int(shift) for shift in options.shifts.split(",")]
    
    # Parse tissue indices
    if options.tissue_indices is not None:
        tissue_indices = [int(idx) for idx in options.tissue_indices.split(",")]
    else:
        tissue_indices = None
    
    print("="*70)
    print("Integrated Gradients for Specific Regions")
    print("="*70)
    print(f"Regions CSV:    {regions_csv}")
    print(f"Genome FASTA:   {genome_fasta}")
    print(f"Genes GFF:      {genes_gff}")
    print(f"Model:          {options.model_path}")
    print(f"Output:         {options.out_file}")
    print(f"Max flanking:   {options.max_flanking} bp")
    print("="*70)
    
    # Read model parameters
    with open(params_file) as params_open:
        params = yaml.safe_load(params_open)
    params_model = params["model"]
    seq_len = params_model["seq_length"]
    model_type = params_model.get("model_type", "2d_to_1d")
    num_targets = params_model["num_targets"]
    
    if model_type != "2d_to_1d":
        raise ValueError(f"Model type {model_type} not supported")
    
    print(f"\nModel: seq_length={seq_len}, num_targets={num_targets}")
    
    # Load model
    print("\nLoading model...")
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(options.model_path, options.head_i)
    
    genome_open = pysam.Fastafile(genome_fasta)

    # Load regions while filtering out regions that are too close to the edges of the genome
    regions_df, has_fold_enrichment = load_regions_csv(regions_csv, genome_open)

    # Load genes
    print(f"\nLoading genes from {genes_gff}")
    transcriptome = pgene.Transcriptome(genes_gff)
    print(f"Loaded {len(transcriptome.genes)} genes")
    
    # Build gene spatial index for fast lookup
    gene_index = build_gene_index(transcriptome)
        
    # Find nearest gene for each region
    print(f"\nFinding nearest genes (within {options.max_flanking} bp)...")
    valid_regions = []
    skipped_regions = []
    
    for idx, row in regions_df.iterrows():
        if idx > 0 and idx % 5000 == 0:
            print(f"  Processed {idx}/{len(regions_df)} regions...")
        
        region_chr = str(row['chr'])
        region_start = int(row['start'])
        region_end = int(row['end'])
        
        gene_id, gene, distance = find_nearest_gene_fast(
            region_chr, region_start, region_end, 
            gene_index, options.max_flanking
        )
        
        if gene_id is None:
            skipped_regions.append({
                'chr': region_chr,
                'start': region_start,
                'end': region_end,
                'reason': 'No gene within flanking distance'
            })
            continue
        
        # Get ATG position
        if gene.strand == "+":
            # get the absolute position of the "A" of ATG start codon (0-indexed position)
            atg_pos = gene.cds_start()
        else:
            # get the absolute position of the "A" of ATG start codon (0-indexed position)
            atg_pos = gene.cds_end()
        
        # Calculate sequence coordinates(1-based) (center on ATG)
        if gene.strand == "+":
            seq_start = atg_pos + 1 - seq_len // 2
            seq_end = seq_start + seq_len 
            region_rel_start = region_start - 1 - atg_pos 
            region_rel_end = region_end - 1 - atg_pos 
        elif gene.strand == "-":
            seq_start = atg_pos + 1 - seq_len // 2 + 1 # First "+1" means convert to 1-based coordinate. Second "+1" makes sure its ATG is in center right when reverse complement.
            seq_end = seq_start + seq_len
            region_rel_end = atg_pos + 1 - region_start 
            region_rel_start = atg_pos + 1 - region_end 
        
        region_info = {
            'region_chr': region_chr,
            'region_start': region_start,
            'region_end': region_end,
            'region_rel_start': region_rel_start,
            'region_rel_end': region_rel_end,
            'gene_id': gene_id,
            'gene_chr': gene.chrom,
            'gene_strand': gene.strand,
            'atg_pos': atg_pos,
            'seq_start': seq_start,
            'seq_end': seq_end,
            'distance_to_atg': distance,
            'csv_index': idx
        }
        
        # Add fold_enrichment if available
        if has_fold_enrichment:
            region_info['fold_enrichment'] = row['fold_enrichment']
        
        valid_regions.append(region_info)
    
    print(f"Valid regions: {len(valid_regions)}")
    print(f"Skipped regions: {len(skipped_regions)}")
    
    if len(valid_regions) == 0:
        print("ERROR: No valid regions found")
        sys.exit(1)
    
    # Save skipped regions info
    if len(skipped_regions) > 0:
        skipped_df = pd.DataFrame(skipped_regions)
        skipped_path = options.out_file.replace('.h5', '_skipped.csv')
        skipped_df.to_csv(skipped_path, index=False)
        print(f"Saved skipped regions to: {skipped_path}")
    
    # Determine output shape
    if tissue_indices is not None:
        num_output_targets = len(tissue_indices)
        print(f"\nComputing gradients for {num_output_targets} selected tissues: {tissue_indices}")
    else:
        num_output_targets = num_targets
        tissue_indices = list(range(num_targets))
        print(f"\nComputing gradients for all {num_targets} tissues")
    
    # Initialize output HDF5
    print(f"\nInitializing output file: {options.out_file}")
    if os.path.isfile(options.out_file):
        os.remove(options.out_file)
    
    scores_h5 = h5py.File(options.out_file, "w")
    
    # Create datasets - store only region-specific gradients
    num_valid = len(valid_regions)
    max_region_len = max([r['region_rel_end'] - r['region_rel_start'] for r in valid_regions])
    
    # Store variable-length gradients for each region
    scores_h5.create_dataset("igs", dtype="float16", shape=(num_valid, max_region_len, 4, num_output_targets))
    scores_h5.create_dataset("region_chr", data=np.array([r['region_chr'] for r in valid_regions], dtype="S"))
    scores_h5.create_dataset("region_start", data=np.array([r['region_start'] for r in valid_regions], dtype="int32"))
    scores_h5.create_dataset("region_end", data=np.array([r['region_end'] for r in valid_regions], dtype="int32"))
    scores_h5.create_dataset("region_length", data=np.array([r['region_rel_end'] - r['region_rel_start'] for r in valid_regions], dtype="int32"))
    scores_h5.create_dataset("gene_id", data=np.array([r['gene_id'] for r in valid_regions], dtype="S"))
    
    # Add fold_enrichment only if it exists in the CSV
    if has_fold_enrichment:
        scores_h5.create_dataset("fold_enrichment", data=np.array([r['fold_enrichment'] for r in valid_regions], dtype="float32"))
        print("  Added 'fold_enrichment' dataset to output")
    
    scores_h5.create_dataset("distance_to_atg", data=np.array([r['distance_to_atg'] for r in valid_regions], dtype="int32"))
    scores_h5.create_dataset("tissue_indices", data=np.array(tissue_indices, dtype="int32"))
    
    # Create baseline
    if options.baseline_type == "zero":
        baseline = None
    elif options.baseline_type == "uniform":
        baseline = np.ones((1, seq_len, 4), dtype="float32") * 0.25
    elif options.baseline_type == "gc":
        gc_content = options.gc_content
        at_freq = (1.0 - gc_content) / 2.0
        gc_freq = gc_content / 2.0
        baseline = np.ones((1, seq_len, 4), dtype="float32")
        baseline[0, :, 0] = at_freq
        baseline[0, :, 1] = gc_freq
        baseline[0, :, 2] = gc_freq
        baseline[0, :, 3] = at_freq
    else:
        raise ValueError(f"Unknown baseline type: {options.baseline_type}")
    
    # Compute integrated gradients
    print(f"\nComputing Integrated Gradients...")
    print(f"Using {options.num_steps} interpolation steps")
    print(f"Baseline type: {options.baseline_type}")
    
    buffer_size = 100
    atg_not_found_count = 0
    for shift in options.shifts:
        print(f"\nProcessing shift {shift}")
        
        for rev_comp in [False, True] if options.rc else [False]:
            if options.rc:
                print(f"Fwd/rev = {'rev' if rev_comp else 'fwd'}")
            
            seq_1hots = []
            region_indices = []
            
            for ri, region_info in enumerate(valid_regions):
                if ri % 200 == 0 and ri > 0:
                    print(f"  Processing region {ri}/{len(valid_regions)}")
                
                seq_1hot, atg_not_found_count = make_seq_1hot_co_interval(
                    genome_open,
                    region_info['region_chr'],
                    region_info['seq_start'],
                    region_info['seq_end'],
                    seq_len,
                    region_info['gene_strand'],
                    atg_not_found_count
                )

                seq_1hot = dna_io.hot1_augment(seq_1hot, shift=shift)
                if rev_comp:
                    seq_1hot = dna_io.hot1_rc(seq_1hot)
                
                seq_1hots.append(seq_1hot[None, ...])
                region_indices.append(ri)
                
                if ri == len(valid_regions) - 1 or len(seq_1hots) >= buffer_size:
                    seq_1hots = np.concatenate(seq_1hots, axis=0)
                    
                    # Prepare baseline
                    if baseline is None:
                        batch_baseline = None
                    else:
                        batch_baseline = dna_io.hot1_augment(baseline[0], shift=shift)
                        if rev_comp:
                            batch_baseline = dna_io.hot1_rc(batch_baseline)
                        batch_baseline = np.tile(batch_baseline[None, ...], (len(seq_1hots), 1, 1))
                    
                    # Compute integrated gradients
                    igs = seqnn_model.integrated_gradients(
                        seq_1hots,
                        baseline=batch_baseline,
                        num_steps=options.num_steps,
                        head_i=options.head_i,
                        target_slice=np.array([[i] for i in tissue_indices]),
                        pos_slice=np.array([[0]]),
                        chunk_size=buffer_size,
                        batch_size=options.batch_size,
                        track_scale=options.track_scale,
                        no_untransform=options.no_untransform,
                        subtract_avg=True,
                        tissue_slab=options.tissue_slab,
                        dtype="float16",
                    )
                    
                    # Extract and save only region-specific gradients
                    for i, ri in enumerate(region_indices):
                        region_info = valid_regions[ri]
                        
                        # Unaugment full sequence gradients
                        ig_full = unaugment_grads(igs[i], fwdrc=(not rev_comp), shift=shift)
                        
                        # Extract region-specific gradients
                        rel_start = region_info['region_rel_start']
                        rel_end = region_info['region_rel_end']
                        ig_region = ig_full[rel_start:rel_end, :, :]
                        
                        # Store in HDF5 (pad if needed)
                        region_len = ig_region.shape[0]
                        scores_h5["igs"][ri, :region_len, :, :] += ig_region
                    
                    seq_1hots = []
                    region_indices = []
                    gc.collect()
    
    print(f"ATG not found count: {atg_not_found_count}")

    # Normalize gradients
    print("\nNormalizing gradients...")
    norm_factor = float(len(options.shifts) * (2 if options.rc else 1))
    if norm_factor > 0:
        for ri in range(len(valid_regions)):
            region_len = valid_regions[ri]['region_rel_end'] - valid_regions[ri]['region_rel_start']
            scores_h5["igs"][ri, :region_len, :, :] /= norm_factor
    
    scores_h5.close()
    genome_open.close()
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)
    print(f"Results saved to: {options.out_file}")
    print(f"Valid regions processed: {len(valid_regions)}")
    print(f"Skipped regions: {len(skipped_regions)}")
    print("="*70)


if __name__ == "__main__":
    main()

