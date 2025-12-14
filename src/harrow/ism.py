# ism.py
# In-silico saturation mutagenesis
# Migrated from tangermeme to TensorFlow/NumPy

import numpy
import itertools
from .predict import predict
from .utils import _cast_as_array


def _edit_distance_one(X, start, end):
    """Generate all sequences of edit distance 1.

    Parameters
    ----------
    X: numpy.ndarray, shape=(length, alphabet_size)
        A single one-hot encoded sequence.

    start: int
        The first position to begin making edits on, inclusive.

    end: int
        The end of the span, not inclusive.

    Returns
    -------
    X_: numpy.ndarray, shape=(length*alphabet_size, length, alphabet_size)
        All one-hot encoded sequences with edit distance 1 from original.
    """

    end = end if end >= 0 else X.shape[0] + 1 + end
    X_ = numpy.repeat(X[numpy.newaxis, :, :], (end-start)*X.shape[1], axis=0)

    coords = itertools.product(range(X.shape[1]), range(start, end))
    for i, (j, k) in enumerate(coords):
        X_[i, k, :] = 0
        X_[i, k, j] = 1

    return X_


def saturation_mutagenesis(model, X, start=0, end=-1, batch_size=32,
                           target=None, hypothetical=False, raw_outputs=False,
                           dtype='float32', device='gpu', verbose=False):
    """Performs in-silico saturation mutagenesis on sequences.

    This function performs in-silico saturation mutagenesis on a set of
    sequences and returns attribution values or raw predictions.

    Parameters
    ----------
    model: paddy.seqnn.SeqNN
        The Paddy SeqNN model to use for predictions.

    X: numpy.ndarray, shape=(-1, length, alphabet_size)
        A set of one-hot encoded sequences to calculate attributions for.

    start: int, optional
        The start of where to begin making perturbations. Default is 0.

    end: int, optional
        The end of where to make perturbations. Default is -1 (entire sequence).

    batch_size: int, optional
        The number of examples to predict at once. Default is 32.

    target: int or slice or None, optional
        Which output/slice of outputs to focus on. If None, use all. Default is None.

    hypothetical: bool, optional
        Whether to return attributions for all possible characters at each
        position. Default is False.

    raw_outputs: bool, optional
        Whether to return raw predictions or processed attributions. Default is False.

    dtype: str, optional
        The dtype for returned values. Default is 'float32'.

    device: str, optional
        The device to run predictions on. Options are 'cpu' or 'gpu'. Default is 'gpu'.

    verbose: bool, optional
        Whether to display a progress bar. Default is False.

    Returns
    -------
    attr: numpy.ndarray
        Processed attribution values (if raw_outputs=False), shape:
        (n_examples, length, alphabet_size) or (n_examples, length, alphabet_size, n_targets)

    -- or, if raw_outputs=True --

    y0: numpy.ndarray
        Predictions for the reference sequences.

    y_hat: numpy.ndarray
        Predictions for each of the perturbed sequences.
    """

    X = _cast_as_array(X)
    y0 = predict(model, X, batch_size=batch_size,
                 dtype=dtype, device=device, verbose=verbose)

    if end < 0:
        end = X.shape[1] + 1 + end

    y_hat = []
    for i in range(X.shape[0]):
        X_ = _edit_distance_one(X[i], start, end)

        y_hat_ = predict(model, X_, batch_size=batch_size,
                         dtype=dtype, device=device, verbose=verbose)
        y_hat.append(y_hat_)

    y_hat = numpy.stack(y_hat)
    # Reshape: (n_examples, alphabet_size*(end-start), ...) -> (n_examples, alphabet_size, end-start, ...)
    y_hat = y_hat.reshape(X.shape[0], X.shape[2], end-start, *y_hat_.shape[1:])

    if raw_outputs:
        return y0, y_hat

    # Calculate attributions
    if target is not None:
        attr = y_hat[:, :, :, target] - \
            y0[:, numpy.newaxis, numpy.newaxis, target]
    else:
        attr = y_hat - y0[:, numpy.newaxis, numpy.newaxis, :]

    # Subtract per-position mean
    attr -= numpy.mean(attr, axis=1, keepdims=True)

    # Average across targets if multiple
    if len(attr.shape) > 3 and attr.shape[-1] > 1:
        attr = numpy.mean(attr, axis=tuple(range(3, len(attr.shape))))

    # If not hypothetical, only return attribution for actual nucleotide
    if not hypothetical:
        # Extract attributions only for the actual nucleotide at each position
        attr_actual = numpy.zeros((X.shape[0], X.shape[1]), dtype=dtype)
        for i in range(X.shape[0]):
            for pos in range(start, end):
                nuc_idx = X[i, pos, :].argmax()
                attr_actual[i, pos] = attr[i, nuc_idx, pos-start]
        return attr_actual

    # Expand back to full sequence length
    attr_full = numpy.zeros((X.shape[0], X.shape[1], X.shape[2]), dtype=dtype)
    attr_full[:, start:end, :] = numpy.transpose(attr, (0, 2, 1))

    return attr_full
