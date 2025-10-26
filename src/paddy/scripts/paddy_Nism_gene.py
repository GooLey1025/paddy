#!/usr/bin/env python
"""
paddy_Nism_gene.py

Compute N-masked In Silico Mutagenesis (N-ISM) attribution scores for all genes in a GFF file.
For each gene, extracts 32kb sequence centered on ATG, computes N-ISM scores by replacing
each position with 'N' and measuring the prediction difference.

Attribution score: ref_pred - masked_pred (for each of 23 tissues)
Total attribution: mean(abs(attribution across tissues))

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
import time

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
        default="nism_gene_out.h5",
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
        default=32,
        type="int",
        help="Batch size for predictions [Default: %default]",
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
    print("N-ISM Attribution for All Genes")
    print("="*70)
    print(f"Genome FASTA:   {genome_fasta}")
    print(f"Genes GFF:      {genes_gff}")
    print(f"Model:          {options.model_path}")
    print(f"Output:         {options.out_file}")
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
    
    # Initialize output HDF5
    print(f"\nInitializing output file: {options.out_file}")
    if os.path.isfile(options.out_file):
        os.remove(options.out_file)
    
    scores_h5 = h5py.File(options.out_file, "w")
    
    # Create datasets
    num_valid = len(valid_genes)
    
    # N-ISM attribution scores: (num_genes, seq_len, num_targets)
    scores_h5.create_dataset("nism_scores", dtype="float16", shape=(num_valid, seq_len, num_targets))
    # Total attribution per position: mean(abs(scores)) across tissues
    scores_h5.create_dataset("nism_total", dtype="float16", shape=(num_valid, seq_len))
    
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
    
    # Compute N-ISM scores
    print(f"\nComputing N-ISM scores...")
    print(f"Batch size: {options.batch_size}")
    
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
        
        # Process all positions in batches
        batch_size = options.batch_size
        num_batches = (seq_len + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, seq_len)
            batch_positions = range(start_idx, end_idx)
            
            # Create batch of masked sequences
            masked_seqs = []
            for pos in batch_positions:
                masked_seq = seq_1hot.copy()
                masked_seq[pos, :] = 0  # Set to N (all zeros)
                masked_seqs.append(masked_seq)
            
            masked_seqs = np.array(masked_seqs)
            
            # Get predictions for masked sequences (with ensemble)
            masked_preds = predict_ensemble_batch(seqnn_model, masked_seqs, options.shifts, options.rc)
            
            # Compute attribution scores: ref_pred - masked_pred
            for i, pos in enumerate(batch_positions):
                attribution = ref_pred - masked_preds[i]
                scores_h5["nism_scores"][gi, pos, :] = attribution.astype('float16')
                # Total attribution: mean of absolute values across tissues
                scores_h5["nism_total"][gi, pos] = np.mean(np.abs(attribution)).astype('float16')

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

