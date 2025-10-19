#!/usr/bin/env python
"""
paddy_n_ism.py

Compute N-masked In Silico Mutagenesis (ISM) scores comparing reference sequences
with N-masked alternate sequences. Unlike borzoi_sed.py, this does NOT filter by
gene/exon overlap - all sequences are processed.

Supports multiple input formats:
1. FASTA file with ref and alt sequences (paired)
2. Reference genome FASTA + GTF + BED file specifying regions

For format 2, sequences are centered on ATG positions from GTF, and BED regions
must fall within seq_length/2 of the ATG site.
"""

from __future__ import print_function
from optparse import OptionParser
from collections import OrderedDict
import gc
import yaml
import os
import sys
from multiprocessing import Pool
from functools import partial

import h5py
import numpy as np
import pandas as pd
import pysam

from paddy import dna as dna_io
from paddy import gene as pgene
from paddy import seqnn


################################################################################
# main
################################################################################
def main():
    usage = """usage: %prog [options] <params_file> <input_file>
    
    Two input modes (auto-detected):
    1. FASTA mode: input_file is a FASTA with paired ref/alt sequences
    2. BED mode: input_file is a BED file, requires --ref (reference genome) and --gtf"""
    parser = OptionParser(usage)
    parser.add_option(
        "--ref",
        dest="genome_fasta",
        default=None,
        help="Reference genome FASTA (for BED mode) [Default: %default]",
    )
    parser.add_option(
        "--gtf",
        dest="genes_gtf",
        default=None,
        help="GTF file with gene annotations (for BED mode) [Default: %default]",
    )
    
    # Model options
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
    
    # Output options
    parser.add_option(
        "-o",
        dest="out_dir",
        default="n_ism_out",
        help="Output directory [Default: %default]",
    )
    parser.add_option(
        "--stats",
        dest="ism_stats",
        default="logSED,logD2,JS",
        help="Comma-separated list of stats to save [Default: %default]. Options: SED, logSED, D1, logD2, nD2, JS, REF, ALT",
    )
    
    # Ensemble options
    parser.add_option(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Average forward and reverse complement predictions [Default: %default]. Forward means plus strand of gene, reverse means minus strand.",
    )
    parser.add_option(
        "--shifts",
        dest="shifts",
        default="0",
        type="str",
        help="Ensemble prediction shifts [Default: %default]",
    )
    
    # Transformation options
    parser.add_option(
        "--no_untransform",
        dest="no_untransform",
        default=False,
        action="store_true",
        help="Skip untransformation of predictions [Default: %default]",
    )
    parser.add_option(
        "--pseudo",
        dest="pseudo",
        default=1e-6,
        type="float",
        help="Pseudocount for log transforms [Default: %default]. Note: JS metric will use minimum of 1e-6 to avoid Inf/NaN",
    )
    parser.add_option(
        "--batch_size",
        dest="batch_size",
        default=64,
        type="int",
        help="Batch size for predictions [Default: %default]",
    )
    parser.add_option(
        "-p",
        "--processes",
        dest="processes",
        default=12,
        type="int",
        help="Number of parallel processes for BED region loading (by chromosome) [Default: %default]",
    )
    
    (options, args) = parser.parse_args()
    
    if len(args) != 2:
        parser.error("Must provide parameters file and input file (FASTA or BED)")
    
    params_file = args[0]
    input_file = args[1]
    
    # Auto-detect input mode based on options
    # If both --ref and --gtf are provided, treat input_file as BED
    # Otherwise, treat input_file as FASTA with paired sequences
    if options.genome_fasta is not None and options.genes_gtf is not None:
        input_mode = "bed"
        bed_file = input_file
        print(f"BED mode detected:")
        print(f"  - BED file: {input_file}")
        print(f"  - Reference genome: {options.genome_fasta}")
        print(f"  - GTF file: {options.genes_gtf}")
    else:
        input_mode = "fasta"
        print(f"FASTA mode detected:")
        print(f"  - Input FASTA: {input_file}")
        if options.genome_fasta is not None or options.genes_gtf is not None:
            parser.error("For BED mode, both --ref and --gtf are required")
    
    if options.model_path is None:
        parser.error("--model_path is required")
    
    if not os.path.isfile(options.model_path):
        parser.error(f"Model file not found: {options.model_path}")
    
    if not os.path.isdir(options.out_dir):
        os.makedirs(options.out_dir)
    
    options.shifts = [int(shift) for shift in options.shifts.split(",")]
    options.ism_stats = options.ism_stats.split(",")
    
    #################################################################
    # read parameters
    
    with open(params_file) as params_open:
        params = yaml.safe_load(params_open)
    params_model = params["model"]
    seq_len = params_model["seq_length"]
    
    #################################################################
    # load model
    
    print("Loading model...")
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(options.model_path, options.head_i)
    num_targets = params_model["num_targets"]
    
    #################################################################
    # load sequences based on input mode
    
    if input_mode == "fasta":
        print("Loading sequences from FASTA file...")
        sequences = load_fasta_pairs(input_file, seq_len)
        
    elif input_mode == "bed":
        print("Loading sequences from genome + GTF + BED...")
        sequences = load_bed_sequences(
            options.genome_fasta,
            options.genes_gtf,
            bed_file,
            seq_len,
            options.processes
        )
    
    num_sequences = len(sequences)
    print(f"Loaded {num_sequences} sequence pairs")
    
    if num_sequences == 0:
        print("No valid sequences found. Exiting.")
        return
    
    #################################################################
    # setup output
    
    ism_out = initialize_output_h5(
        options.out_dir,
        options.ism_stats,
        sequences,
        num_targets
    )
    
    #################################################################
    # predict scores for each sequence pair (batch processing)
    
    print("Computing predictions in batches...")
    batch_size = options.batch_size
    num_batches = (num_sequences + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_sequences)
        batch_seqs = sequences[start_idx:end_idx]
        
        print(f"Processing batch {batch_idx + 1}/{num_batches} (sequences {start_idx}-{end_idx})")
        
        # Collect sequences for batch
        ref_seqs_batch = np.array([seq_info['ref_1hot'] for seq_info in batch_seqs])
        alt_seqs_batch = np.array([seq_info['alt_1hot'] for seq_info in batch_seqs])
        
        # Get batch predictions with ensembling
        ref_preds_batch = predict_ensemble_batch(
            seqnn_model, ref_seqs_batch, options.shifts, options.rc
        )
        alt_preds_batch = predict_ensemble_batch(
            seqnn_model, alt_seqs_batch, options.shifts, options.rc
        )
        
        # Write statistics for each sequence in batch
        for bi, (ref_preds, alt_preds) in enumerate(zip(ref_preds_batch, alt_preds_batch)):
            si = start_idx + bi
            write_scores(
                ref_preds,
                alt_preds,
                ism_out,
                si,
                options.ism_stats,
                options.pseudo
            )
    
    ism_out.close()
    print(f"Results written to {options.out_dir}/ism.h5")


def load_fasta_pairs(fasta_file, seq_len):
    """Load paired ref/alt sequences from FASTA file.
    
    Expects alternating ref and alt sequences, or sequences with _ref/_alt suffixes.
    
    Args:
        fasta_file: Path to FASTA file
        seq_len: Expected sequence length
        
    Returns:
        List of dicts with id, ref_1hot, alt_1hot
    """
    sequences = []
    fasta_open = pysam.Fastafile(fasta_file)
    headers = list(fasta_open.references)
    
    # Try to detect pairing pattern
    ref_headers = [h for h in headers if '_ref' in h.lower() or h.endswith('_R')]
    alt_headers = [h for h in headers if '_alt' in h.lower() or h.endswith('_A')]
    
    if len(ref_headers) > 0 and len(alt_headers) > 0:
        # Explicit ref/alt naming
        print(f"Detected {len(ref_headers)} ref and {len(alt_headers)} alt sequences")
        
        for ref_h in ref_headers:
            # Find matching alt
            base_id = ref_h.replace('_ref', '').replace('_R', '').replace('_REF', '')
            
            alt_h = None
            for ah in alt_headers:
                alt_base = ah.replace('_alt', '').replace('_A', '').replace('_ALT', '')
                if alt_base == base_id:
                    alt_h = ah
                    break
            
            if alt_h is None:
                print(f"Warning: No matching alt sequence for {ref_h}")
                continue
            
            ref_seq = fasta_open.fetch(ref_h)
            alt_seq = fasta_open.fetch(alt_h)
            
            if len(ref_seq) != len(alt_seq):
                print(f"Warning: Ref and alt sequences have different lengths for {base_id}")
                continue
            
            # Process sequences
            ref_1hot = make_seq_1hot(ref_seq, seq_len)
            alt_1hot = make_seq_1hot(alt_seq, seq_len)
            
            sequences.append({
                'id': base_id,
                'ref_1hot': ref_1hot,
                'alt_1hot': alt_1hot,
                'chr': None,
                'pos': None,
                'strand': None
            })
    
    else:
        # Assume alternating ref/alt pairs
        print(f"Detected {len(headers)} sequences, assuming alternating ref/alt pairs")
        
        for i in range(0, len(headers) - 1, 2):
            ref_h = headers[i]
            alt_h = headers[i + 1]
            
            ref_seq = fasta_open.fetch(ref_h)
            alt_seq = fasta_open.fetch(alt_h)
            
            if len(ref_seq) != len(alt_seq):
                print(f"Warning: Sequences {i} and {i+1} have different lengths")
                continue
            
            ref_1hot = make_seq_1hot(ref_seq, seq_len)
            alt_1hot = make_seq_1hot(alt_seq, seq_len)
            
            seq_id = f"{ref_h}_{alt_h}"
            
            sequences.append({
                'id': seq_id,
                'ref_1hot': ref_1hot,
                'alt_1hot': alt_1hot,
                'chr': None,
                'pos': None,
                'strand': None
            })
    
    fasta_open.close()
    return sequences


def load_bed_sequences(genome_fasta, genes_gtf, bed_file, seq_len, num_processes=1):
    """Load sequences from genome using BED file and GTF for ATG centering.
    
    Args:
        genome_fasta: Path to genome FASTA
        genes_gtf: Path to GTF file
        bed_file: Path to BED file with regions to mask
        seq_len: Sequence length
        num_processes: Number of parallel processes (by chromosome)
        
    Returns:
        List of dicts with id, ref_1hot, alt_1hot, and metadata
    """
    
    # Load genome
    genome_open = pysam.Fastafile(genome_fasta)
    
    # Load transcriptome
    print("Loading GTF...")
    transcriptome = pgene.Transcriptome(genes_gtf)
    
    # Pre-compute gene information for efficient lookup (like paddy_grad_gene.py)
    print("Pre-computing gene windows...")
    gene_windows = {}  # {chr: [(window_start, window_end, gene_id, atg_pos, strand)]}
    
    for gene_id, gene in transcriptome.genes.items():
        # Get ATG position (like paddy_grad_gene.py does)
        if gene.strand == "+":
            atg_pos = gene.cds_start()
        else:
            atg_pos = gene.cds_end()
        
        # Calculate ATG-centered window
        window_start = max(0, atg_pos - seq_len // 2)
        window_end = atg_pos + seq_len // 2
        
        # Store in chromosome-indexed dict for fast lookup
        if gene.chrom not in gene_windows:
            gene_windows[gene.chrom] = []
        gene_windows[gene.chrom].append((window_start, window_end, gene_id, atg_pos, gene.strand))
    
    # Sort windows by start position for each chromosome
    for chrom in gene_windows:
        gene_windows[chrom].sort(key=lambda x: x[0])
    
    # Load BED file (3-column format: chr, start, end)
    print("Loading BED file...")
    bed_regions = []
    with open(bed_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('\t')
            if len(parts) < 3:
                print(f"Warning: Skipping line {line_num} - expected at least 3 columns, got {len(parts)}")
                continue
            
            try:
                chrom = parts[0]
                start = int(parts[1])
                end = int(parts[2])
                bed_regions.append((chrom, start, end))
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping line {line_num} - invalid format: {e}")
                continue
    
    print(f"Processing {len(bed_regions)} BED regions...")
    
    # Group regions by chromosome for parallel processing
    regions_by_chrom = {}
    for chrom, mask_start, mask_end in bed_regions:
        if chrom not in regions_by_chrom:
            regions_by_chrom[chrom] = []
        regions_by_chrom[chrom].append((mask_start, mask_end))
    
    print(f"Found regions across {len(regions_by_chrom)} chromosomes")
    
    # Process chromosomes in parallel
    if num_processes > 1:
        print(f"Using {num_processes} parallel processes...")
        with Pool(processes=num_processes) as pool:
            process_func = partial(
                process_chromosome_regions,
                genome_fasta=genome_fasta,
                gene_windows=gene_windows,
                seq_len=seq_len
            )
            results = pool.map(process_func, regions_by_chrom.items())
        
        # Combine results from all chromosomes
        sequences = []
        skip_stats = {
            'no_gene_window': 0,
            'fetch_error': 0,
            'no_atg_found': 0,
            'total_processed': 0
        }
        for chrom_sequences, chrom_stats in results:
            sequences.extend(chrom_sequences)
            for key in skip_stats:
                skip_stats[key] += chrom_stats[key]
    else:
        # Serial processing
        skip_stats = {
            'no_gene_window': 0,
            'fetch_error': 0,
            'no_atg_found': 0,
            'total_processed': 0
        }
        sequences = []
        for idx, (chrom, regions) in enumerate(regions_by_chrom.items()):
            print(f"  Processing chromosome {chrom} ({idx+1}/{len(regions_by_chrom)})...")
            chrom_sequences, chrom_stats = process_chromosome_regions(
                (chrom, regions), genome_fasta, gene_windows, seq_len
            )
            sequences.extend(chrom_sequences)
            for key in skip_stats:
                skip_stats[key] += chrom_stats[key]
    
    genome_open.close()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("BED Region Processing Summary:")
    print("="*70)
    print(f"Total BED regions loaded:           {len(bed_regions)}")
    print(f"Successfully processed:             {skip_stats['total_processed']}")
    print(f"\nSkipped regions breakdown:")
    print(f"  - No gene window found:           {skip_stats['no_gene_window']}")
    print(f"  - Sequence fetch error:           {skip_stats['fetch_error']}")
    print(f"  - ATG not found at center:        {skip_stats['no_atg_found']}")
    total_skipped = skip_stats['no_gene_window'] + skip_stats['fetch_error'] + skip_stats['no_atg_found']
    print(f"  - Total skipped:                  {total_skipped}")
    print(f"\nSuccess rate: {100.0 * skip_stats['total_processed'] / len(bed_regions):.2f}%")
    print("="*70 + "\n")
    
    return sequences


def process_chromosome_regions(chrom_data, genome_fasta, gene_windows, seq_len):
    """Process all regions for a single chromosome.
    
    Args:
        chrom_data: Tuple of (chromosome_name, list_of_(start, end)_tuples)
        genome_fasta: Path to genome FASTA
        gene_windows: Pre-computed gene windows dict
        seq_len: Sequence length
        
    Returns:
        Tuple of (sequences_list, skip_stats_dict)
    """
    chrom, regions = chrom_data
    sequences = []
    skip_stats = {
        'no_gene_window': 0,
        'fetch_error': 0,
        'no_atg_found': 0,
        'total_processed': 0
    }
    
    # Open genome for this process
    genome_open = pysam.Fastafile(genome_fasta)
    
    for mask_start, mask_end in regions:
        # Generate a region ID
        region_id = f"{chrom}_{mask_start}_{mask_end}"
        
        # Find gene whose ATG-centered window contains this BED region
        gene_id = None
        atg_pos = None
        gene_strand = None
        
        # Fast lookup using pre-computed windows
        if chrom in gene_windows:
            for window_start, window_end, gid, g_atg, g_strand in gene_windows[chrom]:
                # Check if BED region falls completely within this window
                if window_start <= mask_start and mask_end <= window_end:
                    gene_id = gid
                    atg_pos = g_atg
                    gene_strand = g_strand
                    break
                # Early exit if we've passed possible matches
                if window_start > mask_end:
                    break
        
        # If no suitable gene found, skip this region
        if gene_id is None:
            skip_stats['no_gene_window'] += 1
            continue
        
        # Center sequence on ATG
        seq_start = max(0, atg_pos - seq_len // 2)
        seq_end = seq_start + seq_len
        
        # Extract reference sequence
        try:
            seq_dna = genome_open.fetch(chrom, seq_start, seq_end)
        except Exception as e:
            skip_stats['fetch_error'] += 1
            continue
        
        # Extend if needed
        if len(seq_dna) < seq_len:
            seq_dna += 'N' * (seq_len - len(seq_dna))
        
        # Convert to 1-hot
        ref_1hot = dna_io.dna_1hot(seq_dna)
        
        # Calculate positions to mask in the extracted sequence (before RC)
        mask_start_rel = max(0, mask_start - seq_start)
        mask_end_rel = min(seq_len, mask_end - seq_start)
        
        # Create N-masked alternative sequence (before RC)
        alt_1hot = ref_1hot.copy()
        
        # For minus strand genes, we need to handle masking carefully
        # The mask coordinates are in genomic coordinates, so we mask before RC
        if mask_start_rel < mask_end_rel:
            alt_1hot[mask_start_rel:mask_end_rel, :] = 0
        
        # NOW handle strand: forward sequence should be in plus strand orientation
        # If gene is on minus strand, reverse complement so ATG is centered correctly
        if gene_strand == "-":
            ref_1hot = dna_io.hot1_rc(ref_1hot)
            alt_1hot = dna_io.hot1_rc(alt_1hot)
        
        # Verify ATG is at center-right position (for both strands after processing)
        center_start = seq_len // 2
        center_end = center_start + 3
        center_seq_1hot = ref_1hot[center_start:center_end, :]
        center_seq = dna_io.hot1_dna(center_seq_1hot)
        
        if center_seq != "ATG":
            skip_stats['no_atg_found'] += 1
            continue
        
        # Successfully processed this region
        skip_stats['total_processed'] += 1
        
        # Create sequence ID
        seq_id = f"{gene_id}_{chrom}_{mask_start}_{mask_end}"
        
        sequences.append({
            'id': seq_id,
            'ref_1hot': ref_1hot,
            'alt_1hot': alt_1hot,
            'chr': chrom,
            'pos': atg_pos,
            'strand': gene_strand,
            'mask_start': mask_start,
            'mask_end': mask_end
        })
    
    genome_open.close()
    return sequences, skip_stats


def make_seq_1hot(seq_dna, seq_len):
    """Convert DNA sequence to 1-hot encoding.
    
    Args:
        seq_dna: DNA sequence string
        seq_len: Target sequence length
        
    Returns:
        1-hot encoded sequence
    """
    # Trim or extend sequence
    if len(seq_dna) > seq_len:
        center = len(seq_dna) // 2
        start = center - seq_len // 2
        seq_dna = seq_dna[start:start + seq_len]
    elif len(seq_dna) < seq_len:
        seq_dna += 'N' * (seq_len - len(seq_dna))
    
    return dna_io.dna_1hot(seq_dna)


def predict_ensemble(seqnn_model, seq_1hot, shifts, rc):
    """Make ensemble predictions with shifts and reverse complement.
    
    Args:
        seqnn_model: SeqNN model
        seq_1hot: 1-hot encoded sequence
        shifts: List of shift values
        rc: Whether to include reverse complement
        
    Returns:
        Averaged predictions across ensemble
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
        seqs_1hot: Batch of 1-hot encoded sequences (batch_size x seq_len x 4)
        shifts: List of shift values
        rc: Whether to include reverse complement
        
    Returns:
        Averaged predictions across ensemble for each sequence (batch_size x num_targets)
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


def initialize_output_h5(out_dir, ism_stats, sequences, num_targets):
    """Initialize output HDF5 file.
    
    Args:
        out_dir: Output directory
        ism_stats: List of statistics to compute
        sequences: List of sequence dicts
        num_targets: Number of prediction targets
        
    Returns:
        HDF5 file handle
    """
    ism_out = h5py.File(f"{out_dir}/ism.h5", "w")
    
    num_seqs = len(sequences)
    
    # Write sequence identifiers
    seq_ids = [s['id'] for s in sequences]
    ism_out.create_dataset("seq_id", data=np.array(seq_ids, "S"))
    
    # Write chromosome info if available
    if sequences[0]['chr'] is not None:
        seq_chrs = [s['chr'] if s['chr'] else "" for s in sequences]
        seq_pos = [s['pos'] if s['pos'] else 0 for s in sequences]
        seq_strands = [s['strand'] if s['strand'] else "" for s in sequences]
        
        ism_out.create_dataset("chr", data=np.array(seq_chrs, "S"))
        ism_out.create_dataset("pos", data=np.array(seq_pos, dtype="int32"))
        ism_out.create_dataset("strand", data=np.array(seq_strands, "S"))
        
        # Write mask positions if available
        if 'mask_start' in sequences[0]:
            mask_starts = [s.get('mask_start', 0) for s in sequences]
            mask_ends = [s.get('mask_end', 0) for s in sequences]
            ism_out.create_dataset("mask_start", data=np.array(mask_starts, dtype="int32"))
            ism_out.create_dataset("mask_end", data=np.array(mask_ends, dtype="int32"))
    
    # Initialize statistics datasets
    for stat in ism_stats:
        ism_out.create_dataset(
            stat, shape=(num_seqs, num_targets), dtype="float32"
        )
    
    return ism_out


def write_scores(ref_preds, alt_preds, ism_out, si, ism_stats, pseudo):
    """Compute and write ISM statistics.
    
    Args:
        ref_preds: Reference predictions
        alt_preds: Alternative predictions
        ism_out: Output HDF5 file
        si: Sequence index
        ism_stats: List of statistics to compute
        pseudo: Pseudocount for log transforms
    """
    from scipy.special import rel_entr
    
    # Clip negative predictions to zero (neural networks can produce small negative values)
    ref_preds = np.maximum(ref_preds, 0.0)
    alt_preds = np.maximum(alt_preds, 0.0)
    
    # Simple difference
    if 'SED' in ism_stats:
        sed = alt_preds - ref_preds
        ism_out['SED'][si] = sed.astype('float32')
    
    # Log difference
    if 'logSED' in ism_stats:
        log_sed = np.log2(alt_preds + pseudo + 1) - np.log2(ref_preds + pseudo + 1)
        ism_out['logSED'][si] = log_sed.astype('float32')
    
    # L1 norm
    if 'D1' in ism_stats:
        d1 = np.abs(alt_preds - ref_preds)
        ism_out['D1'][si] = d1.astype('float32')
    
    # Log L1 norm
    if 'logD1' in ism_stats:
        ref_preds_log = np.log2(ref_preds + pseudo + 1)
        alt_preds_log = np.log2(alt_preds + pseudo + 1)
        log_d1 = np.abs(ref_preds_log - alt_preds_log)
        ism_out['logD1'][si] = log_d1.astype('float32')
    
    # L2 norm (Euclidean distance)
    if 'D2' in ism_stats:
        d2 = np.sqrt(np.power(alt_preds - ref_preds, 2))
        ism_out['D2'][si] = d2.astype('float32')
    
    # Log L2 norm
    if 'logD2' in ism_stats:
        ref_preds_log = np.log2(ref_preds + pseudo + 1)
        alt_preds_log = np.log2(alt_preds + pseudo + 1)
        log_d2 = np.sqrt(np.power(ref_preds_log - alt_preds_log, 2))
        ism_out['logD2'][si] = log_d2.astype('float32')
    
    # Normalized difference
    if 'nD2' in ism_stats:
        ref_norm = ref_preds / (ref_preds.sum() + 1e-6)
        alt_norm = alt_preds / (alt_preds.sum() + 1e-6)
        nd2 = np.sqrt(np.power(ref_norm - alt_norm, 2))
        ism_out['nD2'][si] = nd2.astype('float32')
    
    # Jensen-Shannon divergence
    if 'JS' in ism_stats:
        # JS requires non-zero probabilities to avoid Inf/NaN
        # Use minimum pseudocount of 1e-6 if user didn't specify one
        js_pseudo = max(pseudo, 1e-6)
        ref_norm = ref_preds + js_pseudo
        ref_norm = ref_norm / ref_norm.sum()
        alt_norm = alt_preds + js_pseudo
        alt_norm = alt_norm / alt_norm.sum()
        
        ref_alt_entr = rel_entr(ref_norm, alt_norm)
        alt_ref_entr = rel_entr(alt_norm, ref_norm)
        js_dist = (ref_alt_entr + alt_ref_entr) / 2
        ism_out['JS'][si] = js_dist.astype('float32')
    
    # Store raw predictions
    if 'REF' in ism_stats:
        ism_out['REF'][si] = ref_preds.astype('float32')
    
    if 'ALT' in ism_stats:
        ism_out['ALT'][si] = alt_preds.astype('float32')


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()

