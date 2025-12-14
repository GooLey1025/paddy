# ablate.py
# Ablation experiments (shuffling to remove signal)
# Migrated from tangermeme to TensorFlow/NumPy
# Author: Adapted for Paddy/Harrow

import numpy
import inspect
from .utils import _cast_as_array, validate_input
from .ersatz import shuffle
from .predict import predict


def ablate(model, X, start, end, n=20, shuffle_fn=shuffle, args=None,
          random_state=None, func=predict, additional_func_kwargs=None, **kwargs):
    """Make predictions before and after shuffling a region of sequences.
    
    An ablation experiment shuffles a motif (or region of interest) to remove
    any potential signal. Outputs are returned before and after the region is
    shuffled for a given number of shuffles.
    
    Parameters
    ----------
    model: paddy.seqnn.SeqNN
        A Paddy SeqNN model to use for making predictions.
    
    X: numpy.ndarray, shape=(-1, length, alphabet_size)
        A one-hot encoded set of sequences to have a region shuffled.
    
    start: int
        The starting position of where to shuffle the sequence, inclusive.
    
    end: int
        The ending position of where to shuffle the sequence, not inclusive.
    
    n: int, optional
        The number of times to shuffle that region. Default is 20.
    
    shuffle_fn: function, optional
        A function that will shuffle a portion of the sequence. Default is
        `shuffle` from ersatz.
    
    args: tuple or None, optional
        An optional set of additional arguments to pass into the model.
        Default is None.
    
    random_state: int or numpy.random.RandomState or None, optional
        The random seed to use to ensure determinism. Default is None.
    
    func: function, optional
        A function to apply before and after making the ablation. Default
        is `predict`.
    
    additional_func_kwargs: dict, optional
        Additional named arguments to pass into the function. Default is None.
    
    kwargs: optional
        Additional named arguments that will get passed into the function.
    
    Returns
    -------
    y_before: numpy.ndarray
        The output from the function before shuffling.
    
    y_after: numpy.ndarray or list
        The output from the function after shuffling, for each of n shuffles.
    """
    
    X = _cast_as_array(X)
    validate_input(X, "X", ohe=True, ohe_dim=2)
    additional_func_kwargs = additional_func_kwargs or {}
    
    # Add random_state to func kwargs if it accepts it
    if random_state is not None:
        func_params = inspect.signature(func).parameters
        if 'random_state' in func_params and 'random_state' not in additional_func_kwargs:
            additional_func_kwargs['random_state'] = random_state
    
    # Predictions before shuffling
    y_before = func(model, X, **kwargs, **additional_func_kwargs)
    
    # Shuffle the region n times
    X_shuffled = shuffle_fn(X, start=start, end=end, n=n, random_state=random_state)
    
    # Predictions after shuffling (for each shuffle)
    y_after = []
    for i in range(n):
        X_shuf_i = X_shuffled[:, i, :, :]
        y_shuf_i = func(model, X_shuf_i, **kwargs, **additional_func_kwargs)
        y_after.append(y_shuf_i)
    
    y_after = numpy.stack(y_after, axis=1)  # Shape: (n_examples, n_shuffles, ...)
    
    return y_before, y_after


def ablate_annotations(model, X, annotations, **kwargs):
    """Ablate each annotation individually.
    
    This function takes in a model, a set of sequences, and a set of annotations,
    and returns the ablation values for each annotation.
    
    Parameters
    ----------
    model: paddy.seqnn.SeqNN
        A Paddy SeqNN model to use for making predictions.
    
    X: numpy.ndarray, shape=(-1, length, alphabet_size)
        A one-hot encoded set of sequences with annotations.
    
    annotations: numpy.ndarray, shape=(n_annotations, 3)
        A array of annotations where the first column is the example_idx, the
        second column is the start position, and the third column is the end position.
    
    kwargs: arguments
        Additional optional arguments to pass into the `ablate` function.
    
    Returns
    -------
    y_befores: numpy.ndarray
        The outputs BEFORE ablation for each annotation.
    
    y_afters: numpy.ndarray
        The outputs AFTER ablation for each annotation (includes all shuffles).
    """
    
    X = _cast_as_array(X)
    annotations = _cast_as_array(annotations)
    
    y_befores, y_afters = [], []
    
    for idx, start, end in annotations:
        idx, start, end = int(idx), int(start), int(end)
        
        y_before, y_after = ablate(model, X[idx:idx+1], start=start, end=end, **kwargs)
        y_befores.append(y_before)
        y_afters.append(y_after)
    
    y_befores = numpy.concatenate(y_befores, axis=0)
    y_afters = numpy.concatenate(y_afters, axis=0)
    
    return y_befores, y_afters

