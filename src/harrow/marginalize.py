# marginalize.py
# Motif marginalization experiments
# Migrated from tangermeme to TensorFlow/NumPy

import numpy
from .utils import _cast_as_array, one_hot_encode, validate_input
from .ersatz import substitute
from .predict import predict


def marginalize(model, X, motif, start=None, alphabet=['A', 'C', 'G', 'T'],
               func=predict, additional_func_kwargs=None, **kwargs):
    """Apply a function before and after substituting a motif into sequences.
    
    A marginalization experiment applies a function before and after substituting
    something into a set of sequences. The sequences are meant to be background
    sequences and the difference in output represents the "marginal" effect of
    adding that motif into the sequences.
    
    Parameters
    ----------
    model: paddy.seqnn.SeqNN
        A Paddy SeqNN model to use for making predictions.
    
    X: numpy.ndarray, shape=(-1, length, len(alphabet))
        A one-hot encoded set of sequences to have a motif inserted into.
    
    motif: str or numpy.ndarray
        A string or one-hot encoded motif to insert into sequences.
    
    start: int or None, optional
        The starting position of where to substitute the motif. If None,
        substitute the motif into the middle of the sequence. Default is None.
    
    alphabet: list, optional
        The alphabet for encoding. Default is ['A', 'C', 'G', 'T'].
    
    func: function, optional
        A function to apply before and after making the substitution.
        Default is `predict`.
    
    additional_func_kwargs: dict, optional
        Additional named arguments to pass into the function. Default is None.
    
    kwargs: optional
        Additional named arguments that will get passed into the function.
    
    Returns
    -------
    y_before: numpy.ndarray
        The output from the function before inserting the motif in.
    
    y_after: numpy.ndarray
        The output from the function after inserting the motif in.
    """
    
    X = _cast_as_array(X)
    validate_input(X, "X", shape=(-1, -1, len(alphabet)), ohe=True, ohe_dim=2, allow_N=True)
    additional_func_kwargs = additional_func_kwargs or {}
    
    X_perturb = substitute(X, motif, start=start, alphabet=alphabet)
    y_before = func(model, X, **kwargs, **additional_func_kwargs)
    y_after = func(model, X_perturb, **kwargs, **additional_func_kwargs)
    
    return y_before, y_after


def marginalize_annotations(model, X, X0, annotations, **kwargs):
    """Perform marginalizations on each annotation individually.
    
    This function takes in a model, a set of sequences, a set of background
    sequences, and a set of annotations, and returns the marginalization values
    for each annotation. For each annotation, the sequence in `X` is extracted
    and substituted into `X0`.
    
    Parameters
    ----------
    model: paddy.seqnn.SeqNN
        A Paddy SeqNN model to use for making predictions.
    
    X: numpy.ndarray, shape=(-1, length, alphabet_size)
        A one-hot encoded set of sequences corresponding to the annotations.
    
    X0: numpy.ndarray, shape=(-1, length, alphabet_size)
        A one-hot encoded set of sequences that motifs will be substituted into.
    
    annotations: numpy.ndarray, shape=(n_annotations, 3)
        A array of annotations where the first column is the example_idx, the
        second column is the start position (0-indexed) and the third column is
        the end position (0-indexed, not inclusive).
    
    kwargs: arguments
        Additional optional arguments to pass into the `marginalize` function.
    
    Returns
    -------
    y_befores: numpy.ndarray or list
        The application of the function BEFORE inserting the motif.
    
    y_afters: numpy.ndarray or list
        The application of the function AFTER inserting the motif.
    """
    
    X = _cast_as_array(X)
    X0 = _cast_as_array(X0)
    annotations = _cast_as_array(annotations)
    
    y_befores, y_afters = [], []
    
    for idx, start, end in annotations:
        idx, start, end = int(idx), int(start), int(end)
        seq = X[idx:idx+1, start:end, :]
        
        y_before, y_after = marginalize(model, X0, seq, **kwargs)
        y_befores.append(y_before)
        y_afters.append(y_after)
    
    y_befores = numpy.stack(y_befores, axis=0)
    y_afters = numpy.stack(y_afters, axis=0)
    
    return y_befores, y_afters

