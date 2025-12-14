# attribution.py
# Attribution methods wrapping Paddy's SeqNN TensorFlow implementations

import numpy
import tensorflow as tf
from .utils import _cast_as_array


def integrated_gradients(model, X, baseline=None, num_steps=50,
                         batch_size=32, head_i=None, target_index=None,
                         track_scale=1.0, pseudo_count=0.0, no_untransform=False,
                         subtract_avg=True, use_mean_target=False, tissue_slab=None,
                         dtype='float32', verbose=False):
    """Compute integrated gradients for sequences.

    This is a simplified wrapper around Paddy SeqNN's integrated_gradients method.
    Integrated Gradients (Sundararajan et al., 2017) computes gradients along the
    path from baseline to input and integrates them.

    Parameters
    ----------
    model: paddy.seqnn.SeqNN
        The Paddy SeqNN model.

    X: numpy.ndarray, shape=(-1, seq_length, 4)
        One-hot encoded sequences.

    baseline: numpy.ndarray or None, optional
        Baseline sequences. If None, uses zero baseline. Default is None.

    num_steps: int, optional
        Number of interpolation steps for Riemann approximation. Default is 50.

    batch_size: int, optional
        Batch size for processing. Default is 32.

    head_i: int or None, optional
        Model head index. Default is None.

    target_index: int or None, optional
        Specific target/tissue index to compute gradients for (0-based).
        If provided, only computes gradients for this single target.
        With optimized implementation, this provides genuine speedup by reducing
        the number of tissues in batch_jacobian computation.
        If None, computes for all targets. Default is None.
        Example: target_index=0 for first tissue only.
        Note: Ignored if use_mean_target=True.

    track_scale: float, optional
        Scaling factor for tracks. Default is 1.0.

    pseudo_count: float, optional
        Pseudo count for numerical stability. Default is 0.0.

    no_untransform: bool, optional
        Whether to skip untransformation. Default is False.

    subtract_avg: bool, optional
        Whether to subtract mean across nucleotides. Default is True.

    use_mean_target: bool, optional
        If True, compute gradient w.r.t. mean of all targets/tissues, giving global
        importance scores. If False (default), compute gradients for selected targets.
        For 2d_to_1d models with multiple targets, you can either:
        - Set use_mean_target=True to get shape (..., 1) - global importance
        - Keep use_mean_target=False to get selected targets (..., num_targets)
        - Use target_index to compute for specific tissue(s)
        Default is False.

    tissue_slab: int or None, optional
        Number of tissues to process at once for memory control.
        If None, processes all tissues together (fastest, more memory).
        If < num_tissues, chunks tissue computation (slower, less memory).
        Only relevant when use_mean_target=False.
        Example: tissue_slab=5 processes 5 tissues at a time.
        Default is None (no chunking).

    dtype: str, optional
        Data type for output. Default is 'float16'.

    verbose: bool, optional
        Whether to show progress. Default is False.

    Returns
    -------
    igs: numpy.ndarray
        Integrated gradients. 
        Shape: (batch, seq_length, 4, num_targets) if target_index=None and use_mean_target=False
               (batch, seq_length, 4, 1) if target_index is specified or use_mean_target=True
    """

    X = _cast_as_array(X).astype('float32')

    # Prepare baseline
    if baseline is None:
        baseline_batch = None
    else:
        baseline = _cast_as_array(baseline).astype('float32')
        if baseline.shape[0] == 1 and X.shape[0] > 1:
            baseline_batch = numpy.tile(baseline, (X.shape[0], 1, 1))
        else:
            baseline_batch = baseline

    # Handle target selection
    # target_slice should have shape (num_samples, num_targets_per_sample)
    # Each row contains the target indices to compute for that sample
    num_samples = X.shape[0]

    if target_index is not None and not use_mean_target:
        # Compute for single target only
        num_targets = 1
        # Replicate target_index for all samples: shape (num_samples, 1)
        target_slice = numpy.full(
            (num_samples, 1), target_index, dtype='int32')
    else:
        # Compute for all targets
        # Handle both method and property cases
        if callable(model.num_targets):
            num_targets = model.num_targets(head_i=head_i)
        else:
            num_targets = model.num_targets
        # Create target list and replicate for all samples: shape (num_samples, num_targets)
        targets_list = numpy.array(
            [i for i in range(num_targets)], dtype='int32')
        target_slice = numpy.tile(targets_list, (num_samples, 1))

    # Call model's integrated_gradients method
    igs = model.integrated_gradients(
        X,
        baseline=baseline_batch,
        num_steps=num_steps,
        head_i=head_i,
        target_slice=target_slice,
        # Single position slice (not used for 2d_to_1d)
        pos_slice=numpy.array([[0]]),
        chunk_size=X.shape[0],
        batch_size=batch_size,
        track_scale=track_scale,
        pseudo_count=pseudo_count,
        no_untransform=no_untransform,
        subtract_avg=subtract_avg,
        tissue_slab=tissue_slab,
        dtype=dtype,
        use_mean_target=use_mean_target
    )

    return igs


def gradients(model, X, batch_size=32, head_i=None,
              track_scale=1.0, pseudo_count=0.0, no_untransform=False,
              subtract_avg=True, input_gate=True, dtype='float16', verbose=False):
    """Compute vanilla gradients for sequences.

    This is a simplified wrapper around Paddy SeqNN's gradients method.

    Parameters
    ----------
    model: paddy.seqnn.SeqNN
        The Paddy SeqNN model.

    X: numpy.ndarray, shape=(-1, seq_length, 4)
        One-hot encoded sequences.

    batch_size: int, optional
        Batch size for processing. Default is 32.

    head_i: int or None, optional
        Model head index. Default is None.

    track_scale: float, optional
        Scaling factor for tracks. Default is 1.0.

    pseudo_count: float, optional
        Pseudo count for numerical stability. Default is 0.0.

    no_untransform: bool, optional
        Whether to skip untransformation. Default is False.

    subtract_avg: bool, optional
        Whether to subtract mean across nucleotides. Default is True.

    input_gate: bool, optional
        Whether to multiply by input (input * gradient). Default is True.

    dtype: str, optional
        Data type for output. Default is 'float16'.

    verbose: bool, optional
        Whether to show progress. Default is False.

    Returns
    -------
    grads: numpy.ndarray
        Gradients with respect to input. 
        Shape: (batch, seq_length, 4, num_targets)
        To get gradients for a specific target, manually slice: result[..., 0:1]
    """

    X = _cast_as_array(X).astype('float32')

    # Use all targets - user can slice the output manually if needed
    # Handle both method and property cases
    if callable(model.num_targets):
        num_targets = model.num_targets(head_i=head_i)
    else:
        num_targets = model.num_targets
    num_samples = X.shape[0]
    # target_slice should have shape (num_samples, num_targets_per_sample)
    # Create target list and replicate for all samples: shape (num_samples, num_targets)
    targets_list = numpy.array([i for i in range(num_targets)], dtype='int32')
    target_slice = numpy.tile(targets_list, (num_samples, 1))

    # Call model's gradients method
    grads = model.gradients(
        X,
        head_i=head_i,
        target_slice=target_slice,
        pos_slice=numpy.array([[0]]),  # Single position slice
        chunk_size=X.shape[0],
        batch_size=batch_size,
        track_scale=track_scale,
        pseudo_count=pseudo_count,
        no_untransform=no_untransform,
        subtract_avg=subtract_avg,
        input_gate=input_gate,
        dtype=dtype
    )

    return grads


def hypothetical_attributions(multipliers, X, references):
    """Aggregate contributions into hypothetical attributions.

    When handling categorical data like one-hot encodings, gradients may need
    to be modified because the choice of one character explicitly means the
    other characters are not there. This accounts for each character change
    being the addition of one character AND subtraction of another.

    Adapted from tangermeme's deep_lift_shap implementation for TensorFlow.

    Parameters
    ----------
    multipliers: numpy.ndarray, shape=(n_baselines, seq_length, alphabet_size)
        The multipliers/gradients calculated by a method like DeepLIFT/SHAP.

    X: numpy.ndarray, shape=(n_baselines, seq_length, alphabet_size)
        The one-hot encoded sequence being explained.

    references: numpy.ndarray, shape=(n_baselines, seq_length, alphabet_size)
        The one-hot encoded reference sequences.

    Returns
    -------
    projected_contribs: numpy.ndarray, shape=(seq_length, alphabet_size)
        The attribution values for each nucleotide in the input.
    """

    multipliers = _cast_as_array(multipliers)
    X = _cast_as_array(X)
    references = _cast_as_array(references)

    projected_contribs = numpy.zeros_like(references, dtype=X.dtype)

    for i in range(X.shape[2]):  # For each character in alphabet
        hypothetical_input = numpy.zeros_like(X, dtype=X.dtype)
        hypothetical_input[:, :, i] = 1.0
        hypothetical_diffs = hypothetical_input - references
        hypothetical_contribs = hypothetical_diffs * multipliers

        projected_contribs[:, :, i] = numpy.sum(hypothetical_contribs, axis=2)

    return projected_contribs
