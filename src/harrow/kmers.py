# kmers.py
# K-mer analysis
# Simplified implementation for Paddy/Harrow
# Author: Adapted for Paddy/Harrow

import numpy
from collections import Counter
from .utils import characters


def kmer_frequencies(X, k=6):
    """Compute k-mer frequencies.
    
    Parameters
    ----------
    X: numpy.ndarray, shape=(-1, length, alphabet_size)
        One-hot encoded sequences.
    
    k: int, optional
        K-mer length. Default is 6.
    
    Returns
    -------
    frequencies: list of dict
        K-mer frequencies for each sequence.
    """
    
    frequencies = []
    
    for ex_idx in range(X.shape[0]):
        seq_str = characters(X[ex_idx])
        
        kmers = [seq_str[i:i+k] for i in range(len(seq_str) - k + 1)]
        kmer_counts = Counter(kmers)
        
        # Normalize to frequencies
        total = sum(kmer_counts.values())
        kmer_freqs = {kmer: count/total for kmer, count in kmer_counts.items()}
        
        frequencies.append(kmer_freqs)
    
    return frequencies

