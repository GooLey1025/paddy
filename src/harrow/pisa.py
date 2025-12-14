# pisa.py
# Positional importance scoring analysis
# Simplified implementation for Paddy/Harrow
# Author: Adapted for Paddy/Harrow

import numpy
from .ersatz import randomize
from .predict import predict


def positional_importance(model, X, window_size=10, n_shuffles=20,
                         random_state=None, batch_size=32, verbose=False):
    """Compute position-specific importance scores.
    
    This function measures the importance of each window by randomizing it
    and measuring the change in predictions.
    
    Parameters
    ----------
    model: paddy.seqnn.SeqNN
        A Paddy SeqNN model.
    
    X: numpy.ndarray, shape=(-1, length, alphabet_size)
        One-hot encoded sequences.
    
    window_size: int, optional
        Size of windows to shuffle. Default is 10.
    
    n_shuffles: int, optional
        Number of shuffles per window. Default is 20.
    
    random_state: int or None, optional
        Random seed. Default is None.
    
    batch_size: int, optional
        Batch size for predictions. Default is 32.
    
    verbose: bool, optional
        Whether to show progress. Default is False.
    
    Returns
    -------
    importance: numpy.ndarray, shape=(-1, n_windows)
        Importance score for each window.
    """
    
    # Get original predictions
    y_orig = predict(model, X, batch_size=batch_size, verbose=verbose)
    
    n_windows = (X.shape[1] - window_size) // window_size + 1
    importance = numpy.zeros((X.shape[0], n_windows))
    
    for w_idx in range(n_windows):
        start = w_idx * window_size
        end = min(start + window_size, X.shape[1])
        
        # Randomize this window
        X_rand = randomize(X, start=start, end=end, n=n_shuffles, random_state=random_state)
        
        # Get predictions for each shuffle
        scores = []
        for s_idx in range(n_shuffles):
            y_rand = predict(model, X_rand[:, s_idx, :, :], batch_size=batch_size, verbose=False)
            # Compute difference
            diff = numpy.abs(y_rand - y_orig).mean(axis=-1)
            scores.append(diff)
        
        importance[:, w_idx] = numpy.mean(scores, axis=0)
    
    return importance

