# annotate.py
# Annotation-based analysis
# Simplified implementation for Paddy/Harrow
# Author: Adapted for Paddy/Harrow

import numpy
from .utils import _cast_as_array


def scan_sequences(X, annotations, func, **kwargs):
    """Apply function to annotated regions.
    
    Parameters
    ----------
    X: numpy.ndarray, shape=(-1, length, alphabet_size)
        One-hot encoded sequences.
    
    annotations: numpy.ndarray, shape=(n_annotations, 3)
        Annotations where columns are [example_idx, start, end].
    
    func: function
        Function to apply to each annotated region.
    
    kwargs: optional
        Additional arguments for func.
    
    Returns
    -------
    results: list
        Results from applying func to each annotation.
    """
    
    X = _cast_as_array(X)
    annotations = _cast_as_array(annotations)
    
    results = []
    for idx, start, end in annotations:
        idx, start, end = int(idx), int(start), int(end)
        region = X[idx, start:end, :]
        result = func(region, **kwargs)
        results.append(result)
    
    return results

