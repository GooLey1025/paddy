# predict.py
# Batched prediction wrapper for Paddy's SeqNN models
# Migrated from tangermeme (PyTorch) to TensorFlow

import numpy
import tensorflow as tf
from tqdm import tqdm

from .utils import _cast_as_array


def predict(model, X, args=None, batch_size=32, head_i=None, dtype='float32',
            device='gpu', verbose=False, **kwargs):
    """Make batched predictions in a memory-efficient manner.

    This function will take a Paddy SeqNN model or a standard TensorFlow/Keras model
    and make predictions from it. The function processes data in batches to handle
    large datasets that don't fit in memory.

    Parameters
    ----------
    model: paddy.seqnn.SeqNN or tf.keras.Model
        The model to use to make predictions. Can be either a Paddy SeqNN model
        or a standard TensorFlow/Keras model.

    X: numpy.ndarray or tf.Tensor, shape=(-1, seq_length, 4) or any model-compatible shape
        A one-hot encoded set of sequences (or any input compatible with the model)
        to make predictions for.

    args: tuple or list or None, optional
        An optional set of additional arguments to pass into the model. If provided,
        each element in the tuple or list is one input to the model and the element
        must be formatted to be the same batch size as `X`. Only applicable for
        standard Keras models. Default is None.

    batch_size: int, optional
        The number of examples to make predictions for at a time. Default is 32.

    head_i: int or None, optional
        The model head index to use for predictions. Only applicable for Paddy SeqNN
        models. If None, uses the default head. Default is None.

    dtype: str or numpy.dtype, optional
        The dtype to return predictions as. Default is 'float32'.

    device: str, optional
        The device to run predictions on. Options are 'cpu' or 'gpu'. For TensorFlow,
        this controls whether to use CPU or GPU. Default is 'gpu'.

    verbose: bool, optional
        Whether to display a progress bar during predictions. Default is False.

    **kwargs: optional
        Additional keyword arguments to pass to the model's call method.
        For example, alpha, beta for custom models, or training=False.

    Returns
    -------
    y: numpy.ndarray
        The output from the model for each input example. The precise format
        is determined by the model architecture.

    Examples
    --------
    >>> # Paddy SeqNN model
    >>> from paddy.seqnn import SeqNN
    >>> model = SeqNN(model_params)
    >>> y = predict(model, X, batch_size=32, head_i=0)

    >>> # Standard Keras model
    >>> model = tf.keras.Sequential([...])
    >>> y = predict(model, X, batch_size=32, device='gpu')

    >>> # Custom model with extra parameters
    >>> y = predict(model, X, batch_size=32, alpha=0.5, beta=2.0)

    >>> # Model with additional arguments
    >>> y = predict(model, X, args=[arg1, arg2], batch_size=32)
    """

    X = _cast_as_array(X)

    # Validate args if provided
    if args is not None:
        for arg in args:
            if arg.shape[0] != X.shape[0]:
                raise ValueError(
                    "Arguments must have the same first dimension as X")

    # Convert to float32 for model inference if needed
    if X.dtype != numpy.float32:
        X = X.astype(numpy.float32)

    # Set device context for TensorFlow
    if device.lower() in ['gpu', 'cuda']:
        device_context = '/GPU:0'
    else:
        device_context = '/CPU:0'

    y = []
    batch_size = min(batch_size, X.shape[0])

    # Detect if this is a Paddy SeqNN model (has __call__ with head_i parameter)
    is_seqnn = hasattr(model, '__class__') and model.__class__.__name__ in [
        'SeqNN', 'SlopeNN']

    with tf.device(device_context):
        for start in tqdm(range(0, X.shape[0], batch_size), disable=not verbose,
                          desc="Predicting"):
            end = min(start + batch_size, X.shape[0])
            X_batch = X[start:end]

            if X_batch.shape[0] == 0:
                continue

            # Prepare additional arguments if provided
            if args is not None:
                args_batch = [arg[start:end] for arg in args]

            # Call model based on type
            if is_seqnn:
                # Paddy SeqNN model: supports head_i and dtype
                y_batch = model(X_batch, head_i=head_i, dtype=dtype)
            else:
                # Standard TensorFlow/Keras model: use standard call
                if args is not None:
                    y_batch = model(X_batch, *args_batch, **kwargs)
                else:
                    y_batch = model(X_batch, **kwargs)

            # Convert to numpy if it's a tensor
            if isinstance(y_batch, tf.Tensor):
                y_batch = y_batch.numpy()

            y.append(y_batch)

    # Concatenate all predictions
    y = numpy.concatenate(y, axis=0).astype(dtype)

    return y
