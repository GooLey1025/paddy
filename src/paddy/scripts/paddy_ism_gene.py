#!/usr/bin/env python
"""
paddy_ism_gene.py

Compute traditional In Silico Mutagenesis (ISM) attribution scores for all genes in a GFF file.
For each gene, extracts 32kb sequence centered on ATG, computes ISM scores by mutating
each position to all 3 alternative nucleotides and measuring the prediction difference.

Attribution score: mut_pred - ref_pred (for each of 4 nucleotides × 23 tissues)
Reference nucleotide attribution is always 0.
Total attribution: mean(abs(attribution)) across all 4 nucleotides and 23 tissues

Note: All coordinates stored in the output HDF5 file are 1-based.
"""

from __future__ import print_function
from optparse import OptionParser
import gc
import yaml
import os
import sys
import h5py
import numpy as np
import pysam
import pandas as pd
import tensorflow as tf

from paddy import dna as dna_io
from paddy import gene as pgene
from paddy import seqnn


def parse_args():
    """Parse command line arguments."""
    usage = "usage: %prog [options] <params_file> <genome_fasta> <genes_gff>"
    parser = OptionParser(usage)
    
    parser.add_option(
        "-o",
        dest="out_file",
        default="ism_gene_out.h5",
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
        help="Batch size for predictions [Default: %default]",
    )
    parser.add_option(
        "--regions_tsv",
        dest="regions_tsv",
        default=None,
        type="str",
        help="Optional TSV file with regions to mutate (1-based: chr, start, end). If not provided, all positions will be processed [Default: %default]",
    )
    
    return parser.parse_args()


def make_seq_1hot_co_interval(genome_open, chrm, start, end, seq_len, strand):
    """Extract sequence and convert to one-hot encoding. Assumes start and end are 1-based coordinate."""
    start = start - 1
    end = end - 1
    if start < 0:
        seq_dna = "N" * (-start) + genome_open.fetch(chrm, 0, end)
    else:
        seq_dna = genome_open.fetch(chrm, start, end)
    # Extend to full length if needed
    if len(seq_dna) < seq_len:
        seq_dna += "N" * (seq_len - len(seq_dna))
    if strand == "-":
        seq_dna = dna_io.dna_rc(seq_dna)
    seq_1hot = dna_io.dna_1hot(seq_dna)
    return seq_1hot


def predict_ensemble(seqnn_model, seq_1hot, shifts, rc):
    """Make ensemble predictions with shifts and reverse complement.
    
    Args:
        seqnn_model: SeqNN model
        seq_1hot: 1-hot encoded sequence (seq_len, 4)
        shifts: List of shift values
        rc: Whether to include reverse complement
        
    Returns:
        Averaged predictions across ensemble (num_targets,)
    """
    preds_list = []
    
    for shift in shifts:
        # Apply shift
        seq_shifted = dna_io.hot1_augment(seq_1hot, shift=shift)
        
        # Forward strand
        pred_fwd = seqnn_model(seq_shifted[None, ...])[0]
        preds_list.append(pred_fwd)
        
        # Reverse complement
        if rc:
            seq_rc = dna_io.hot1_rc(seq_shifted)
            pred_rc = seqnn_model(seq_rc[None, ...])[0]
            preds_list.append(pred_rc)
    
    # Average predictions
    preds = np.mean(preds_list, axis=0)
    return preds


def predict_ensemble_batch(seqnn_model, seqs_1hot, shifts, rc):
    """Make ensemble predictions for a batch of sequences.
    
    Args:
        seqnn_model: SeqNN model
        seqs_1hot: Batch of 1-hot encoded sequences (batch_size, seq_len, 4)
        shifts: List of shift values
        rc: Whether to include reverse complement
        
    Returns:
        Averaged predictions across ensemble (batch_size, num_targets)
    """
    batch_size = seqs_1hot.shape[0]
    preds_list = []
    
    for shift in shifts:
        # Apply shift to all sequences
        seqs_shifted = np.array([dna_io.hot1_augment(seq, shift=shift) for seq in seqs_1hot])
        
        # Forward strand
        preds_fwd = seqnn_model(seqs_shifted)
        preds_list.append(preds_fwd)
        
        # Reverse complement
        if rc:
            seqs_rc = np.array([dna_io.hot1_rc(seq) for seq in seqs_shifted])
            preds_rc = seqnn_model(seqs_rc)
            preds_list.append(preds_rc)
    
    # Average predictions across ensemble
    preds_batch = np.mean(preds_list, axis=0)
    return preds_batch


def load_regions_filter(regions_tsv, gene_windows):
    """Load regions from TSV and create position masks for each gene.
    
    Args:
        regions_tsv: Path to TSV file with header (chr, start, end) in 1-based coordinates
        gene_windows: Dict mapping gene_id to gene metadata dict
        
    Returns:
        Dict mapping gene_id to boolean mask (seq_len,)
    """
    print(f"\nLoading region filters from {regions_tsv}")
    
    # Load regions from TSV (with header)
    regions_df = pd.read_csv(regions_tsv, sep='\t')
    print(f"Loaded {len(regions_df)} regions")
    
    # Group regions by chromosome for efficient lookup
    regions_by_chr = {}
    for _, row in regions_df.iterrows():
        chrm = str(row['chr'])
        start = int(row['start'])  # 1-based
        end = int(row['end'])  # 1-based inclusive
        if chrm not in regions_by_chr:
            regions_by_chr[chrm] = []
        regions_by_chr[chrm].append((start, end))
    
    # Create masks for each gene
    gene_masks = {}
    for gene_id, gene_info in gene_windows.items():
        seq_len = gene_info['seq_end'] - gene_info['seq_start']
        mask = np.zeros(seq_len, dtype=bool)
        
        gene_chr = gene_info['gene_chr']
        seq_start = gene_info['seq_start']  # 1-based
        gene_strand = gene_info['gene_strand']
        
        if gene_chr in regions_by_chr:
            for region_start, region_end in regions_by_chr[gene_chr]:
                # Find overlap between region and gene's sequence window
                # Convert to relative positions in the sequence
                rel_start = max(0, region_start - seq_start)
                rel_end = min(seq_len, region_end - seq_start + 1)
                
                if rel_start < rel_end:
                    # For minus strand genes, the sequence is reverse complemented
                    # So we need to flip the positions
                    if gene_strand == "-":
                        # Flip positions for RC sequences
                        rc_start = seq_len - rel_end
                        rc_end = seq_len - rel_start
                        mask[rc_start:rc_end] = True
                    else:
                        mask[rel_start:rel_end] = True
        
        gene_masks[gene_id] = mask
        if mask.sum() > 0:
            print(f"  Gene {gene_id}: {mask.sum()} positions in filter")
    
    total_filtered = sum(m.sum() for m in gene_masks.values())
    print(f"Total positions to process: {total_filtered}")
    
    return gene_masks


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
    
    if not os.path.isfile(options.model_path):
        print(f"ERROR: Model file not found: {options.model_path}")
        sys.exit(1)
    
    options.shifts = [int(shift) for shift in options.shifts.split(",")]
    
    print("="*70)
    print("Traditional ISM Attribution for All Genes")
    print("="*70)
    print(f"Genome FASTA:   {genome_fasta}")
    print(f"Genes GFF:      {genes_gff}")
    print(f"Model:          {options.model_path}")
    print(f"Output:         {options.out_file}")
    if options.regions_tsv:
        print(f"Regions filter: {options.regions_tsv}")
    print("="*70)
    
    # Read model parameters
    with open(params_file) as params_open:
        params = yaml.safe_load(params_open)
    params_model = params["model"]
    seq_len = params_model["seq_length"]
    num_targets = params_model["num_targets"]
    
    print(f"\nModel: seq_length={seq_len}, num_targets={num_targets}")
    
    # Load model
    print("\nLoading model...")
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(options.model_path, options.head_i)
    
    genome_open = pysam.Fastafile(genome_fasta)
    
    # Get chromosome lengths
    chrom_lengths = dict(zip(genome_open.references, genome_open.lengths))
    
    # Load genes
    print(f"\nLoading genes from {genes_gff}")
    transcriptome = pgene.Transcriptome(genes_gff)
    print(f"Loaded {len(transcriptome.genes)} genes")
    
    # Process all genes and validate 32kb window boundaries
    print(f"\nValidating genes (32kb window must fit within chromosome boundaries)...")
    valid_genes = []
    skipped_genes = []
    
    for gene_id, gene in transcriptome.genes.items():
        # Get gene span (0-based coordinates)
        gene_start_0based, gene_end_0based = gene.span()
        
        # Get gene ATG position (0-based)
        if gene.strand == "+":
            atg_pos_0based = gene.cds_start()
        else:
            atg_pos_0based = gene.cds_end()
        
        # Check if atg_pos is valid
        if atg_pos_0based is None or atg_pos_0based < 0:
            skipped_genes.append({
                'gene_id': gene_id,
                'chr': gene.chrom,
                'reason': 'No valid CDS/ATG position'
            })
            continue
        
        # Calculate sequence coordinates (1-based) (center on ATG)
        if gene.strand == "+":
            seq_start = atg_pos_0based + 1 - seq_len // 2
            seq_end = seq_start + seq_len
        elif gene.strand == "-":
            seq_start = atg_pos_0based + 1 - seq_len // 2 + 1
            seq_end = seq_start + seq_len
        else:
            skipped_genes.append({
                'gene_id': gene_id,
                'chr': gene.chrom,
                'reason': f'Invalid strand: {gene.strand}'
            })
            continue
        
        # Check if 32kb window fits within chromosome boundaries
        if gene.chrom not in chrom_lengths:
            skipped_genes.append({
                'gene_id': gene_id,
                'chr': gene.chrom,
                'reason': f'Chromosome not found in genome'
            })
            continue
        
        chrom_len = chrom_lengths[gene.chrom]
        
        if seq_start < 1:
            skipped_genes.append({
                'gene_id': gene_id,
                'chr': gene.chrom,
                'reason': f'Window extends before chromosome start (seq_start={seq_start})'
            })
            continue
        
        if seq_end > chrom_len:
            skipped_genes.append({
                'gene_id': gene_id,
                'chr': gene.chrom,
                'reason': f'Window extends beyond chromosome end (seq_end={seq_end}, chrom_len={chrom_len})'
            })
            continue
        
        # Get CDS boundaries (0-based)
        cds_start_0based = gene.cds_start() if gene.cds_start() is not None else -1
        cds_end_0based = gene.cds_end() if gene.cds_end() is not None else -1
        
        # Gene is valid - store metadata (convert all coordinates to 1-based for output)
        gene_info = {
            'gene_id': gene_id,
            'gene_chr': gene.chrom,
            'gene_strand': gene.strand,
            'gene_start': gene_start_0based + 1,
            'gene_end': gene_end_0based,
            'atg_pos': atg_pos_0based + 1,
            'seq_start': seq_start,
            'seq_end': seq_end,
            'cds_start': cds_start_0based + 1 if cds_start_0based >= 0 else -1,
            'cds_end': cds_end_0based + 1 if cds_end_0based >= 0 else -1,
        }
        
        valid_genes.append(gene_info)
    
    print(f"Valid genes: {len(valid_genes)}")
    print(f"Skipped genes: {len(skipped_genes)}")
    
    if len(valid_genes) == 0:
        print("ERROR: No valid genes found")
        sys.exit(1)
    
    # Save skipped genes info
    if len(skipped_genes) > 0:
        skipped_df = pd.DataFrame(skipped_genes)
        skipped_path = options.out_file.replace('.h5', '_skipped.csv')
        skipped_df.to_csv(skipped_path, index=False)
        print(f"Saved skipped genes to: {skipped_path}")
    
    # Load region filters if provided
    gene_masks = None
    if options.regions_tsv:
        gene_windows = {g['gene_id']: g for g in valid_genes}
        gene_masks = load_regions_filter(options.regions_tsv, gene_windows)
    
    # Initialize output HDF5
    print(f"\nInitializing output file: {options.out_file}")
    if os.path.isfile(options.out_file):
        os.remove(options.out_file)
    
    scores_h5 = h5py.File(options.out_file, "w")
    
    # Create datasets
    num_valid = len(valid_genes)
    
    # ISM attribution scores: (num_genes, seq_len, 4, num_targets)
    # 4 nucleotides: A=0, C=1, G=2, T=3
    scores_h5.create_dataset("ism_scores", dtype="float16", shape=(num_valid, seq_len, 4, num_targets))
    # Total attribution per position: mean(abs(scores)) across all nucleotides and tissues
    scores_h5.create_dataset("ism_total", dtype="float16", shape=(num_valid, seq_len))
    # Reference nucleotide at each position (0=A, 1=C, 2=G, 3=T, 4=N)
    scores_h5.create_dataset("ref_nucleotides", dtype="uint8", shape=(num_valid, seq_len))
    
    # Gene metadata (all coordinates are 1-based)
    scores_h5.create_dataset("gene_id", data=np.array([g['gene_id'] for g in valid_genes], dtype="S"))
    scores_h5.create_dataset("gene_chr", data=np.array([g['gene_chr'] for g in valid_genes], dtype="S"))
    scores_h5.create_dataset("gene_strand", data=np.array([g['gene_strand'] for g in valid_genes], dtype="S"))
    scores_h5.create_dataset("gene_start", data=np.array([g['gene_start'] for g in valid_genes], dtype="int32"))
    scores_h5.create_dataset("gene_end", data=np.array([g['gene_end'] for g in valid_genes], dtype="int32"))
    scores_h5.create_dataset("atg_pos", data=np.array([g['atg_pos'] for g in valid_genes], dtype="int32"))
    scores_h5.create_dataset("seq_start", data=np.array([g['seq_start'] for g in valid_genes], dtype="int32"))
    scores_h5.create_dataset("seq_end", data=np.array([g['seq_end'] for g in valid_genes], dtype="int32"))
    scores_h5.create_dataset("cds_start", data=np.array([g['cds_start'] for g in valid_genes], dtype="int32"))
    scores_h5.create_dataset("cds_end", data=np.array([g['cds_end'] for g in valid_genes], dtype="int32"))
    
    # Save position masks if region filtering is used
    if gene_masks is not None:
        mask_array = np.zeros((num_valid, seq_len), dtype=bool)
        for gi, gene_info in enumerate(valid_genes):
            mask_array[gi] = gene_masks[gene_info['gene_id']]
        scores_h5.create_dataset("position_mask", data=mask_array)
    
    # Compute ISM scores
    print(f"\nComputing ISM scores...")
    print(f"Batch size: {options.batch_size}")
    
    # Nucleotide mapping
    nt_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    idx_to_nt = ['A', 'C', 'G', 'T']
    
    for gi, gene_info in enumerate(valid_genes):
        if gi % 100 == 0 and gi > 0:
            print(f"  Processing gene {gi}/{len(valid_genes)}")
        
        # Extract reference sequence
        seq_1hot = make_seq_1hot_co_interval(
            genome_open,
            gene_info['gene_chr'],
            gene_info['seq_start'],
            gene_info['seq_end'],
            seq_len,
            gene_info['gene_strand']
        )
        
        # Get reference prediction (with ensemble)
        ref_pred = predict_ensemble(seqnn_model, seq_1hot, options.shifts, options.rc)
        
        # Store reference nucleotides
        for pos in range(seq_len):
            if seq_1hot[pos, 0] == 1:
                scores_h5["ref_nucleotides"][gi, pos] = 0  # A
            elif seq_1hot[pos, 1] == 1:
                scores_h5["ref_nucleotides"][gi, pos] = 1  # C
            elif seq_1hot[pos, 2] == 1:
                scores_h5["ref_nucleotides"][gi, pos] = 2  # G
            elif seq_1hot[pos, 3] == 1:
                scores_h5["ref_nucleotides"][gi, pos] = 3  # T
            else:
                scores_h5["ref_nucleotides"][gi, pos] = 4  # N
        
        # Determine which positions to process
        if gene_masks is not None:
            positions_to_process = np.where(gene_masks[gene_info['gene_id']])[0]
        else:
            positions_to_process = np.arange(seq_len)
        
        # Process each position
        for pos in positions_to_process:
            # Get reference nucleotide at this position
            ref_nt_idx = scores_h5["ref_nucleotides"][gi, pos]
            
            # Skip if reference is N
            if ref_nt_idx == 4:
                continue
            
            # Create mutated sequences for all 4 nucleotides
            mutated_seqs = []
            nt_indices = []
            
            for nt_idx in range(4):
                mut_seq = seq_1hot.copy()
                # Set position to the nucleotide
                mut_seq[pos, :] = 0
                mut_seq[pos, nt_idx] = 1
                mutated_seqs.append(mut_seq)
                nt_indices.append(nt_idx)
            
            mutated_seqs = np.array(mutated_seqs)
            
            # Get predictions for all mutations (with ensemble)
            # Process in batch for efficiency
            mut_preds = predict_ensemble_batch(seqnn_model, mutated_seqs, options.shifts, options.rc)
            
            # Compute attribution scores: mut_pred - ref_pred
            for i, nt_idx in enumerate(nt_indices):
                if nt_idx == ref_nt_idx:
                    # Reference nucleotide: attribution is 0
                    attribution = np.zeros(num_targets)
                else:
                    # Alternative nucleotide: mut_pred - ref_pred
                    attribution = mut_preds[i] - ref_pred
                
                scores_h5["ism_scores"][gi, pos, nt_idx, :] = attribution.astype('float16')
            
            # Compute total attribution: mean(abs()) across all 4 nucleotides and tissues
            total_attr = np.mean(np.abs(scores_h5["ism_scores"][gi, pos, :, :]))
            scores_h5["ism_total"][gi, pos] = total_attr
        
        # Clear memory periodically
        if gi % 50 == 0:
            gc.collect()
    
    scores_h5.close()
    genome_open.close()
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)
    print(f"Results saved to: {options.out_file}")
    print(f"Valid genes processed: {len(valid_genes)}")
    print(f"Skipped genes: {len(skipped_genes)}")
    print("="*70)


if __name__ == "__main__":
    main()

