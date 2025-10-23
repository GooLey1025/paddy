#!/usr/bin/env python
"""
make_borzoi_vcf.py

Convert STARRseq CSV to VCF format for Borzoi SED analysis.
Creates N-masked alternate alleles from BED regions.

Input CSV format: chr, start, end, ...
Output VCF format: CHROM, POS, ID, REF, ALT, ...

The ALT allele is the REF sequence with all bases replaced by 'N'.
"""

import argparse
import sys
import pysam


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert STARRseq CSV to VCF with N-masked alternates',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--csv',
        required=True,
        help='Input CSV file with chr, start, end columns'
    )
    
    parser.add_argument(
        '--ref',
        required=True,
        help='Reference genome FASTA file'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='output.vcf',
        help='Output VCF file'
    )
    
    parser.add_argument(
        '--chr-prefix',
        default='chr',
        help='Chromosome prefix in reference genome (e.g., "chr" or "")'
    )
    
    return parser.parse_args()


def read_csv_regions(csv_file):
    """Read BED regions from CSV file.
    
    Args:
        csv_file: Path to CSV file
        
    Returns:
        List of tuples (chr, start, end)
    """
    regions = []
    
    with open(csv_file, 'r', encoding='utf-8-sig') as f:  # utf-8-sig handles BOM
        header_line = f.readline().strip()
        print(f"CSV header: {header_line}")
        
        # Parse header to find column indices
        headers = header_line.split(',')
        
        # Clean headers - remove BOM and whitespace
        headers = [h.strip().lstrip('\ufeff') for h in headers]
        print(f"Cleaned headers: {headers}")
        
        # Find chr, start, end columns
        chr_idx = None
        start_idx = None
        end_idx = None
        
        for i, h in enumerate(headers):
            h_lower = h.lower().strip()
            if h_lower in ['chr', 'chrom', 'chromosome']:
                chr_idx = i
            elif h_lower in ['start', 'chromstart']:
                start_idx = i
            elif h_lower in ['end', 'chromend']:
                end_idx = i
        
        if chr_idx is None or start_idx is None or end_idx is None:
            raise ValueError(f"Could not find chr/start/end columns in CSV. Found: {headers}")
        
        print(f"Using columns: chr={chr_idx}, start={start_idx}, end={end_idx}")
        
        # Read data lines
        for line_num, line in enumerate(f, 2):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) < max(chr_idx, start_idx, end_idx) + 1:
                print(f"Warning: Line {line_num} has insufficient columns, skipping")
                continue
            
            try:
                chrom = parts[chr_idx]
                start = int(parts[start_idx])
                end = int(parts[end_idx])
                regions.append((chrom, start, end))
            except (ValueError, IndexError) as e:
                print(f"Warning: Line {line_num} has invalid format: {e}")
                continue
    
    print(f"Loaded {len(regions)} regions from CSV")
    return regions


def create_vcf(regions, ref_fasta, output_vcf, chr_prefix='chr'):
    """Create VCF file with N-masked alternate alleles.
    
    Args:
        regions: List of (chr, start, end) tuples
        ref_fasta: Path to reference genome FASTA
        output_vcf: Path to output VCF file
        chr_prefix: Chromosome prefix in reference genome
    """
    print(f"\nOpening reference genome: {ref_fasta}")
    genome = pysam.Fastafile(ref_fasta)
    
    # Get available chromosomes in reference
    available_chroms = set(genome.references)
    print(f"Available chromosomes in reference: {len(available_chroms)}")
    
    # Write VCF
    print(f"\nWriting VCF to: {output_vcf}")
    
    with open(output_vcf, 'w') as vcf_out:
        # Write VCF header
        vcf_out.write("##fileformat=VCFv4.2\n")
        vcf_out.write("##source=make_borzoi_vcf.py\n")
        vcf_out.write("##INFO=<ID=REGION,Number=1,Type=String,Description=\"Original genomic region\">\n")
        vcf_out.write("##INFO=<ID=LENGTH,Number=1,Type=Integer,Description=\"Length of masked region\">\n")
        vcf_out.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        
        skipped = 0
        written = 0
        
        for chrom, start, end in regions:
            # Normalize chromosome name
            chrom_clean = chrom.replace('chr', '')
            
            # Try to find matching chromosome in reference
            ref_chrom = None
            for test_chrom in [chrom, f'chr{chrom_clean}', chrom_clean, f'Chr{chrom_clean}']:
                if test_chrom in available_chroms:
                    ref_chrom = test_chrom
                    break
            
            if ref_chrom is None:
                if skipped < 10:  # Only print first 10 warnings
                    print(f"Warning: Chromosome '{chrom}' not found in reference, skipping")
                skipped += 1
                continue
            
            # Extract reference sequence for this region
            try:
                ref_seq = genome.fetch(ref_chrom, start, end).upper()
            except Exception as e:
                if skipped < 10:
                    print(f"Warning: Could not fetch {ref_chrom}:{start}-{end}: {e}")
                skipped += 1
                continue
            
            if not ref_seq or len(ref_seq) == 0:
                if skipped < 10:
                    print(f"Warning: Empty sequence for {ref_chrom}:{start}-{end}, skipping")
                skipped += 1
                continue
            
            # Create N-masked alternate allele
            alt_seq = 'N' * len(ref_seq)
            
            # VCF uses 1-based coordinates, BED uses 0-based
            # POS should be start + 1 for VCF
            vcf_pos = start + 1
            
            # Create variant ID
            var_id = f"{chrom}:{start}-{end}"
            
            # INFO field
            info = f"REGION={chrom}:{start}-{end};LENGTH={len(ref_seq)}"
            
            # Write VCF line
            vcf_line = f"{chrom}\t{vcf_pos}\t{var_id}\t{ref_seq}\t{alt_seq}\t.\t.\t{info}\n"
            vcf_out.write(vcf_line)
            written += 1
            
            if written % 1000 == 0:
                print(f"  Processed {written} variants...")
    
    genome.close()
    
    print(f"\n{'='*70}")
    print(f"VCF creation complete!")
    print(f"{'='*70}")
    print(f"Total variants written: {written}")
    print(f"Regions skipped:        {skipped}")
    print(f"Success rate:           {100.0 * written / (written + skipped):.2f}%")
    print(f"Output file:            {output_vcf}")
    print(f"{'='*70}")


def main():
    args = parse_args()
    
    # Validate inputs
    import os
    if not os.path.exists(args.csv):
        print(f"ERROR: CSV file not found: {args.csv}")
        sys.exit(1)
    
    if not os.path.exists(args.ref):
        print(f"ERROR: Reference FASTA not found: {args.ref}")
        sys.exit(1)
    
    print("="*70)
    print("Creating VCF from STARRseq CSV")
    print("="*70)
    print(f"Input CSV:       {args.csv}")
    print(f"Reference FASTA: {args.ref}")
    print(f"Output VCF:      {args.output}")
    print(f"Chr prefix:      '{args.chr_prefix}'")
    print("="*70)
    
    # Read regions from CSV
    regions = read_csv_regions(args.csv)
    
    if len(regions) == 0:
        print("\nERROR: No valid regions found in CSV file")
        sys.exit(1)
    
    # Create VCF
    create_vcf(regions, args.ref, args.output, args.chr_prefix)
    
    print(f"\n✅ Done! VCF file created: {args.output}")
    print(f"\nYou can now use this VCF with Borzoi SED:")
    print(f"  python borzoi_sed.py <params> <model> {args.output} -o <output_dir>")


if __name__ == "__main__":
    main()

