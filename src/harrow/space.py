# space.py
# Motif spacing analysis
# Simplified implementation for Paddy/Harrow
# Author: Adapted for Paddy/Harrow

import numpy
from .ersatz import multisubstitute
from .predict import predict


def space(model, X, motif1, motif2, spacings, start=None, alphabet=['A', 'C', 'G', 'T'],
         func=predict, additional_func_kwargs=None, **kwargs):
    """Analyze the effect of motif spacing.
    
    This function tests how the spacing between two motifs affects model predictions.
    
    Parameters
    ----------
    model: paddy.seqnn.SeqNN
        A Paddy SeqNN model.
    
    X: numpy.ndarray, shape=(-1, length, alphabet_size)
        Background sequences.
    
    motif1: str or numpy.ndarray
        The first motif.
    
    motif2: str or numpy.ndarray
        The second motif.
    
    spacings: list of int
        List of spacings to test between the two motifs.
    
    start: int or None, optional
        Starting position for the first motif. Default is None (center).
    
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
    results: list of numpy.ndarray
        Predictions for each spacing.
    """
    
    additional_func_kwargs = additional_func_kwargs or {}
    
    results = []
    for spacing in spacings:
        X_spaced = multisubstitute(X, [motif1, motif2], [spacing],
                                   start=start, alphabet=alphabet)
        y = func(model, X_spaced, **kwargs, **additional_func_kwargs)
        results.append(y)
    
    return results

