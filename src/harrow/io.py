# io.py
# Data I/O utilities for loading sequences
# Migrated from tangermeme to work with Paddy
# Author: Adapted for Paddy/Harrow

import numpy
import pandas
from .utils import one_hot_encode


def extract_loci(fasta_file, loci, window=None, alphabet=['A', 'C', 'G', 'T']):
    """Extract sequences from FASTA at specified loci.
    
    Parameters
    ----------
    fasta_file: str
        Path to FASTA file.
    
    loci: pandas.DataFrame
        BED-formatted dataframe with columns [chrom, start, end].
    
    window: int or None, optional
        If provided, extract a window of this size centered on each locus.
        Default is None (use start/end from loci).
    
    alphabet: list, optional
        The alphabet for one-hot encoding. Default is ['A', 'C', 'G', 'T'].
    
    Returns
    -------
    sequences: numpy.ndarray
        One-hot encoded sequences, shape (n_loci, length, alphabet_size).
    """
    
    try:
        from pyfaidx import Fasta
    except ImportError:
        raise ImportError("pyfaidx is required for FASTA loading. Install with: pip install pyfaidx")
    
    fasta = Fasta(fasta_file)
    sequences = []
    
    for idx, row in loci.iterrows():
        chrom = str(row.iloc[0])
        start = int(row.iloc[1])
        end = int(row.iloc[2])
        
        if window is not None:
            mid = (start + end) // 2
            start = mid - window // 2
            end = mid + window // 2 + window % 2
        
        # Extract sequence
        try:
            seq_str = str(fasta[chrom][start:end])
            seq_onehot = one_hot_encode(seq_str.upper(), alphabet=alphabet)
            sequences.append(seq_onehot)
        except Exception as e:
            print(f"Warning: Could not extract {chrom}:{start}-{end}: {e}")
            # Return N's
            seq_onehot = numpy.zeros((end - start, len(alphabet)))
            sequences.append(seq_onehot)
    
    return numpy.array(sequences)


def read_bed(bed_file, header=None):
    """Read BED format file.
    
    Parameters
    ----------
    bed_file: str
        Path to BED file.
    
    header: list or None, optional
        Column names. If None, uses standard BED names. Default is None.
    
    Returns
    -------
    bed_df: pandas.DataFrame
        BED-formatted dataframe.
    """
    
    if header is None:
        header = ['chrom', 'start', 'end']
    
    # Try to infer number of columns
    with open(bed_file, 'r') as f:
        first_line = f.readline().strip()
        n_cols = len(first_line.split('\t'))
    
    # Extend header if needed
    while len(header) < n_cols:
        header.append(f'col_{len(header)}')
    
    bed_df = pandas.read_csv(bed_file, sep='\t', header=None, names=header[:n_cols])
    
    return bed_df


def read_gff(gff_file):
    """Read GFF format file.
    
    Parameters
    ----------
    gff_file: str
        Path to GFF file.
    
    Returns
    -------
    gff_df: pandas.DataFrame
        GFF-formatted dataframe.
    """
    
    header = ['seqname', 'source', 'feature', 'start', 'end', 
              'score', 'strand', 'frame', 'attribute']
    
    gff_df = pandas.read_csv(gff_file, sep='\t', comment='#', header=None, names=header)
    
    return gff_df


def read_fasta(fasta_file, alphabet=['A', 'C', 'G', 'T']):
    """Read all sequences from a FASTA file.
    
    Parameters
    ----------
    fasta_file: str
        Path to FASTA file.
    
    alphabet: list, optional
        The alphabet for one-hot encoding. Default is ['A', 'C', 'G', 'T'].
    
    Returns
    -------
    sequences: numpy.ndarray
        One-hot encoded sequences.
    
    names: list
        Sequence names/IDs.
    """
    
    try:
        from pyfaidx import Fasta
    except ImportError:
        raise ImportError("pyfaidx is required for FASTA loading. Install with: pip install pyfaidx")
    
    fasta = Fasta(fasta_file)
    
    sequences = []
    names = []
    
    for record_name in fasta.keys():
        seq_str = str(fasta[record_name][:])
        seq_onehot = one_hot_encode(seq_str.upper(), alphabet=alphabet)
        sequences.append(seq_onehot)
        names.append(record_name)
    
    # Pad to same length if needed
    max_len = max(seq.shape[0] for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        if seq.shape[0] < max_len:
            padding = numpy.zeros((max_len - seq.shape[0], len(alphabet)))
            seq_padded = numpy.concatenate([seq, padding], axis=0)
        else:
            seq_padded = seq
        padded_sequences.append(seq_padded)
    
    return numpy.array(padded_sequences), names

