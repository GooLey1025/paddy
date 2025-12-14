# product.py
# Cartesian product analysis
# Simplified implementation for Paddy/Harrow
# Author: Adapted for Paddy/Harrow

import numpy
import itertools
from .ersatz import substitute
from .predict import predict


def cartesian_product(model, X, motifs, start=None, alphabet=['A', 'C', 'G', 'T'],
                     func=predict, additional_func_kwargs=None, **kwargs):
    """Test all pairwise motif combinations.
    
    This function tests all combinations of motifs from a provided list.
    
    Parameters
    ----------
    model: paddy.seqnn.SeqNN
        A Paddy SeqNN model.
    
    X: numpy.ndarray, shape=(-1, length, alphabet_size)
        Background sequences.
    
    motifs: list of str or numpy.ndarray
        List of motifs to test.
    
    start: int or None, optional
        Starting position. Default is None (center).
    
    alphabet: list, optional
        The alphabet. Default is ['A', 'C', 'G', 'T'].
    
    func: function, optional
        Function to apply. Default is `predict`.
    
    additional_func_kwargs: dict, optional
        Additional arguments for func. Default is None.
    
    kwargs: optional
        Additional arguments for func.
    
    Returns
    -------
    results: dict
        Dictionary mapping motif pairs to predictions.
    """
    
    additional_func_kwargs = additional_func_kwargs or {}
    
    results = {}
    for m1, m2 in itertools.product(motifs, repeat=2):
        key = (str(m1), str(m2))
        X_sub = substitute(X, m1, start=start, alphabet=alphabet)
        X_sub = substitute(X_sub, m2, start=start, alphabet=alphabet)
        y = func(model, X_sub, **kwargs, **additional_func_kwargs)
        results[key] = y
    
    return results

