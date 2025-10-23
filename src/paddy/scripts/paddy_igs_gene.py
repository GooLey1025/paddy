#!/usr/bin/env python
"""
paddy_igs_gene.py

Compute Integrated Gradients for all genes in a GFF file.
For each gene, extracts 32kb sequence centered on ATG, computes integrated gradients,
and saves the full 32kb gradients along with comprehensive gene metadata.

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
        default="igs_gene_out.h5",
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
    
    return parser.parse_args()


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
    print("Integrated Gradients for All Genes")
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
        # seq_start is 1-based, so it should be >= 1
        # seq_end is 1-based exclusive, so it should be <= chrom_length
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
            'gene_start': gene_start_0based + 1,  # Convert to 1-based
            'gene_end': gene_end_0based,  # gene_end is already right-exclusive in 0-based, so it equals 1-based inclusive end
            'atg_pos': atg_pos_0based + 1,  # Convert to 1-based
            'seq_start': seq_start,  # Already 1-based
            'seq_end': seq_end,  # Already 1-based (exclusive)
            'cds_start': cds_start_0based + 1 if cds_start_0based >= 0 else -1,  # Convert to 1-based
            'cds_end': cds_end_0based + 1 if cds_end_0based >= 0 else -1,  # Convert to 1-based (cds_end() returns 0-based inclusive position)
        }
        
        valid_genes.append(gene_info)
    
    print(f"Valid genes: {len(valid_genes)}")
    print(f"Skipped genes: {len(skipped_genes)}")
    
    if len(valid_genes) == 0:
        print("ERROR: No valid genes found")
        sys.exit(1)
    
    # Save skipped genes info
    if len(skipped_genes) > 0:
        import pandas as pd
        skipped_df = pd.DataFrame(skipped_genes)
        skipped_path = options.out_file.replace('.h5', '_skipped.csv')
        skipped_df.to_csv(skipped_path, index=False)
        print(f"Saved skipped genes to: {skipped_path}")
    
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
    
    # Create datasets - store full 32kb gradients
    num_valid = len(valid_genes)
    
    # Store full sequence gradients for each gene
    scores_h5.create_dataset("igs", dtype="float16", shape=(num_valid, seq_len, 4, num_output_targets))
    
    # Gene metadata (all coordinates are 1-based)
    scores_h5.create_dataset("gene_id", data=np.array([g['gene_id'] for g in valid_genes], dtype="S"))
    scores_h5.create_dataset("gene_chr", data=np.array([g['gene_chr'] for g in valid_genes], dtype="S"))
    scores_h5.create_dataset("gene_strand", data=np.array([g['gene_strand'] for g in valid_genes], dtype="S"))
    scores_h5.create_dataset("gene_start", data=np.array([g['gene_start'] for g in valid_genes], dtype="int32"))  # 1-based inclusive
    scores_h5.create_dataset("gene_end", data=np.array([g['gene_end'] for g in valid_genes], dtype="int32"))  # 1-based inclusive
    scores_h5.create_dataset("atg_pos", data=np.array([g['atg_pos'] for g in valid_genes], dtype="int32"))  # 1-based
    scores_h5.create_dataset("seq_start", data=np.array([g['seq_start'] for g in valid_genes], dtype="int32"))  # 1-based inclusive
    scores_h5.create_dataset("seq_end", data=np.array([g['seq_end'] for g in valid_genes], dtype="int32"))  # 1-based exclusive
    scores_h5.create_dataset("cds_start", data=np.array([g['cds_start'] for g in valid_genes], dtype="int32"))  # 1-based inclusive
    scores_h5.create_dataset("cds_end", data=np.array([g['cds_end'] for g in valid_genes], dtype="int32"))  # 1-based inclusive
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
            gene_indices = []
            
            for gi, gene_info in enumerate(valid_genes):
                if gi % 200 == 0 and gi > 0:
                    print(f"  Processing gene {gi}/{len(valid_genes)}")
                
                seq_1hot, atg_not_found_count = make_seq_1hot_co_interval(
                    genome_open,
                    gene_info['gene_chr'],
                    gene_info['seq_start'],
                    gene_info['seq_end'],
                    seq_len,
                    gene_info['gene_strand'],
                    atg_not_found_count
                )

                seq_1hot = dna_io.hot1_augment(seq_1hot, shift=shift)
                if rev_comp:
                    seq_1hot = dna_io.hot1_rc(seq_1hot)
                
                seq_1hots.append(seq_1hot[None, ...])
                gene_indices.append(gi)
                
                if gi == len(valid_genes) - 1 or len(seq_1hots) >= buffer_size:
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
                    
                    # Store full sequence gradients
                    for i, gi in enumerate(gene_indices):
                        # Unaugment full sequence gradients
                        ig_full = unaugment_grads(igs[i], fwdrc=(not rev_comp), shift=shift)
                        
                        # Store complete gradients (no region extraction)
                        scores_h5["igs"][gi, :, :, :] += ig_full
                    
                    seq_1hots = []
                    gene_indices = []
                    gc.collect()
    
    print(f"ATG not found count: {atg_not_found_count}")

    # Normalize gradients
    print("\nNormalizing gradients...")
    norm_factor = float(len(options.shifts) * (2 if options.rc else 1))
    if norm_factor > 0:
        scores_h5["igs"][:] /= norm_factor
    
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

