# match.py
# Motif matching/scanning
# Simplified implementation for Paddy/Harrow
# Author: Adapted for Paddy/Harrow

import numpy
from scipy import signal


def match_pwm(X, pwm, threshold=0.8):
    """Scan sequences for PWM matches.
    
    Parameters
    ----------
    X: numpy.ndarray, shape=(-1, length, alphabet_size)
        One-hot encoded sequences.
    
    pwm: numpy.ndarray, shape=(motif_length, alphabet_size)
        Position weight matrix.
    
    threshold: float, optional
        Score threshold for matches (as fraction of max possible score). Default is 0.8.
    
    Returns
    -------
    matches: list of tuples
        List of (example_idx, position, score, strand) for each match.
    """
    
    matches = []
    max_score = pwm.max(axis=1).sum()
    
    for ex_idx in range(X.shape[0]):
        seq = X[ex_idx]
        
        # Forward strand
        for pos in range(seq.shape[0] - pwm.shape[0] + 1):
            window = seq[pos:pos+pwm.shape[0], :]
            score = (window * pwm).sum()
            if score >= threshold * max_score:
                matches.append((ex_idx, pos, score, '+'))
        
        # Reverse strand
        seq_rc = numpy.flip(seq, axis=0)[:, [3, 2, 1, 0]]
        for pos in range(seq_rc.shape[0] - pwm.shape[0] + 1):
            window = seq_rc[pos:pos+pwm.shape[0], :]
            score = (window * pwm).sum()
            if score >= threshold * max_score:
                matches.append((ex_idx, seq.shape[0] - pos - pwm.shape[0], score, '-'))
    
    return matches

