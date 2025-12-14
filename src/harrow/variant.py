# variant.py
# Variant effect prediction
# Migrated from tangermeme to TensorFlow/NumPy
# Author: Adapted for Paddy/Harrow

import numpy
from .utils import _cast_as_array
from .predict import predict
from .ersatz import delete, insert


def substitution_effect(model, X, substitutions, func=predict,
                       additional_func_kwargs=None, **kwargs):
    """Apply a function before and after including one or more substitutions.
    
    This function calculates the effect that substitutions have on the
    output from a model. Any number of substitutions can be added to each
    sequence and the provided `func` is applied before and after these
    substitutions are included in the sequences.
    
    Parameters
    ----------
    model: paddy.seqnn.SeqNN
        A Paddy SeqNN model to use for making predictions.
    
    X: numpy.ndarray, shape=(-1, length, alphabet_size)
        A one-hot encoded set of sequences to have substitutions included in.
    
    substitutions: numpy.ndarray, shape=(-1, 3)
        A set of variants formatted like a COO-sparse matrix where each row
        is a single variant: [example_idx, position, nucleotide_idx].
    
    func: function, optional
        A function to apply before and after incorporating the substitution.
        Default is `predict`.
    
    additional_func_kwargs: dict, optional
        Additional named arguments to pass into the function. Default is None.
    
    kwargs: optional
        Additional named arguments for the function.
    
    Returns
    -------
    y_before: numpy.ndarray
        The output from `func` before variants are included.
    
    y_after: numpy.ndarray
        The output from `func` after the variants are included.
    """
    
    X = _cast_as_array(X)
    substitutions = _cast_as_array(substitutions).astype(int)
    additional_func_kwargs = additional_func_kwargs or {}
    
    X_var = X.copy()
    for idx, pos, nuc in substitutions:
        X_var[idx, pos, :] = 0
        X_var[idx, pos, nuc] = 1
    
    y_before = func(model, X, **kwargs, **additional_func_kwargs)
    y_after = func(model, X_var, **kwargs, **additional_func_kwargs)
    
    return y_before, y_after


def deletion_effect(model, X, start, end, trim_left=True, func=predict,
                   additional_func_kwargs=None, **kwargs):
    """Calculate the effect of deletions on model outputs.
    
    Parameters
    ----------
    model: paddy.seqnn.SeqNN
        A Paddy SeqNN model to use for making predictions.
    
    X: numpy.ndarray, shape=(-1, length, alphabet_size)
        A one-hot encoded set of sequences.
    
    start: int
        The starting position of the deletion, inclusive.
    
    end: int
        The ending position of the deletion, not inclusive.
    
    trim_left: bool, optional
        Whether to trim from the left (True) or right (False) to maintain
        sequence length. Default is True.
    
    func: function, optional
        A function to apply before and after the deletion. Default is `predict`.
    
    additional_func_kwargs: dict, optional
        Additional named arguments for the function. Default is None.
    
    kwargs: optional
        Additional named arguments for the function.
    
    Returns
    -------
    y_before: numpy.ndarray
        The output before the deletion.
    
    y_after: numpy.ndarray
        The output after the deletion.
    """
    
    X = _cast_as_array(X)
    additional_func_kwargs = additional_func_kwargs or {}
    
    # Delete the region
    X_del = delete(X, start, end)
    
    # Pad to maintain length
    pad_size = end - start
    if trim_left:
        # Pad on the left with N's (all zeros)
        padding = numpy.zeros((X.shape[0], pad_size, X.shape[2]), dtype=X.dtype)
        X_del = numpy.concatenate([padding, X_del], axis=1)
    else:
        # Pad on the right with N's
        padding = numpy.zeros((X.shape[0], pad_size, X.shape[2]), dtype=X.dtype)
        X_del = numpy.concatenate([X_del, padding], axis=1)
    
    y_before = func(model, X, **kwargs, **additional_func_kwargs)
    y_after = func(model, X_del, **kwargs, **additional_func_kwargs)
    
    return y_before, y_after


def insertion_effect(model, X, motif, start, trim_left=True, func=predict,
                    additional_func_kwargs=None, **kwargs):
    """Calculate the effect of insertions on model outputs.
    
    Parameters
    ----------
    model: paddy.seqnn.SeqNN
        A Paddy SeqNN model to use for making predictions.
    
    X: numpy.ndarray, shape=(-1, length, alphabet_size)
        A one-hot encoded set of sequences.
    
    motif: str or numpy.ndarray
        The motif to insert.
    
    start: int
        The starting position for insertion.
    
    trim_left: bool, optional
        Whether to trim from the left (True) or right (False) to maintain
        sequence length. Default is True.
    
    func: function, optional
        A function to apply before and after the insertion. Default is `predict`.
    
    additional_func_kwargs: dict, optional
        Additional named arguments for the function. Default is None.
    
    kwargs: optional
        Additional named arguments for the function.
    
    Returns
    -------
    y_before: numpy.ndarray
        The output before the insertion.
    
    y_after: numpy.ndarray
        The output after the insertion.
    """
    
    X = _cast_as_array(X)
    additional_func_kwargs = additional_func_kwargs or {}
    
    # Insert the motif
    X_ins = insert(X, motif, start=start)
    
    # Trim to maintain length
    motif_len = len(motif) if isinstance(motif, str) else motif.shape[1]
    if trim_left:
        # Trim from the left
        X_ins = X_ins[:, motif_len:, :]
    else:
        # Trim from the right
        X_ins = X_ins[:, :-motif_len, :]
    
    y_before = func(model, X, **kwargs, **additional_func_kwargs)
    y_after = func(model, X_ins, **kwargs, **additional_func_kwargs)
    
    return y_before, y_after

