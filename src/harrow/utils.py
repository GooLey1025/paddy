# utils.py
# Migrated from tangermeme (PyTorch) to NumPy/TensorFlow
# Author: Adapted for Paddy/Harrow

import numpy
import numba
import tensorflow as tf
from tqdm import tqdm


def validate_input(X, name, shape=None, dtype=None, min_value=None,
                   max_value=None, ohe=False, ohe_dim=1, allow_N=False):
    """Validate properties of the input array.

    This function will take in an object and verify characteristics of it, such
    as the type, the datatype of the elements, its shape, etc. If any of these
    characteristics are not met, an error will be raised.

    Parameters
    ----------
    X: numpy.ndarray or tf.Tensor
        The object to be verified.

    name: str
        The name to reference the array by if an error is raised.

    shape: tuple or None, optional
        The shape the array must have. If a -1 is provided at any axis, that
        position is ignored. If not provided, no check is performed. Default is
        None.

    dtype: numpy.dtype or None, optional
        The dtype the array must have. If not provided, no check is performed.
        Default is None.

    min_value: float or None, optional
        The minimum value that can be in the array, inclusive. If None, no
        check is performed. Default is None.

    max_value: float or None, optional
        The maximum value that can be in the array, inclusive. If None, no
        check is performed. Default is None.

    ohe: bool, optional
        Whether the input must be a one-hot encoding, i.e., only consist of
        zeroes and ones. Default is False.

    ohe_dim: int, optional
        The dimension that should sum to 1 for one-hot encoding. Default is 1.

    allow_N: bool, optional
        Whether to allow the return of the character 'N' in the sequence, i.e.
        if one-hot at a position is all 0's. Default is False.

    Returns
    -------
    X: numpy.ndarray or tf.Tensor
        The same object, unmodified, for convenience.
    """

    if not isinstance(X, (numpy.ndarray, tf.Tensor)):
        raise ValueError(
            "{} must be a numpy.ndarray or tf.Tensor object".format(name))

    if shape is not None:
        if len(shape) != len(X.shape):
            raise ValueError("{} has shape {} but must have shape {}".format(name,
                                                                             X.shape, shape))

        for i in range(len(shape)):
            if shape[i] != -1 and shape[i] != X.shape[i]:
                raise ValueError("{} has shape {} but must have shape {}".format(name,
                                                                                 X.shape, shape))

    if dtype is not None and X.dtype != dtype:
        raise ValueError("{} has dtype {} but must have dtype {}".format(name, X.dtype,
                                                                         dtype))

    if min_value is not None and X.min() < min_value:
        raise ValueError("{} cannot have a value below {}".format(name,
                                                                  min_value))

    if max_value is not None and X.max() > max_value:
        raise ValueError("{} cannot have a value above {}".format(name,
                                                                  max_value))

    if ohe:
        values = numpy.unique(X)
        if len(values) != 2:
            raise ValueError("{} must be one-hot encoded.".format(name))

        if not all(values == numpy.array([0, 1])):
            raise ValueError("{} must be one-hot encoded.".format(name))

        if allow_N:
            if not numpy.all(numpy.sum(X, axis=ohe_dim) <= 1):
                raise ValueError("{} must be one-hot encoded ".format(name) +
                                 "and contain unknown characters as all-zeroes.")
        else:
            if not numpy.all(X.sum(axis=ohe_dim) == 1):
                raise ValueError("{} must be one-hot encoded ".format(name) +
                                 "and cannot have unknown characters.")

    return X


def _cast_as_array(value, dtype=None):
    """Cast input as a numpy array.

    This function will take some array-like input and cast it as a numpy
    array with optionally a desired dtype.

    Parameters
    ----------
    value: array-like
        An array-like object to be cast into a numpy.ndarray

    dtype: numpy.dtype or None, optional
        A numpy dtype to cast the values to. If None, do not
        do casting. Default is None.

    Returns
    -------
    value: numpy.ndarray
        A numpy array that has been created from the array-like with the provided
        dtype.
    """

    if value is None:
        return None

    if isinstance(value, tf.Tensor):
        value = value.numpy()

    if isinstance(value, numpy.ndarray):
        if dtype is None:
            return value
        elif value.dtype == dtype:
            return value
        else:
            return value.astype(dtype)

    if isinstance(value, list):
        if all(isinstance(v, numpy.ndarray) for v in value):
            value = numpy.array(value)

    if isinstance(value, (float, int, list, tuple, numpy.ndarray)):
        if dtype is None:
            return numpy.array(value)
        else:
            return numpy.array(value, dtype=dtype)


def characters(pwm, alphabet=['A', 'C', 'G', 'T'], force=False, allow_N=False):
    """Converts a PWM/one-hot encoding to a string sequence.

    This function takes in a PWM or one-hot encoding and converts it to the
    most likely sequence. When the input is a one-hot encoding, this is the
    opposite of the `one_hot_encode` function.

    Parameters
    ----------
    pwm: numpy.ndarray or tf.Tensor, shape=(seq_len, alphabet_len)
        A numeric representation of the sequence. This can be one-hot encoded
        or contain numeric values. These numerics can be probabilities but can
        also be frequencies. Note: shape is (length, alphabet) in harrow.

    alphabet : set or tuple or list
        A pre-defined alphabet where the ordering of the symbols is the same
        as the index into the returned tensor. This is used to determine the
        letters in the returned sequence. Default is the DNA alphabet.

    force: bool, optional
        Whether to force a sequence to be produced even when there are ties.
        At each position that there is a tie, the character earlier in the
        sequence will be used. Default is False.

    allow_N: bool, optional
        Whether to allow the return of the character 'N' in the sequence, i.e.
        if pwm at a position is all 0's return N. Default is False.

    Returns
    -------
    seq: str
        A string where the length is the first dimension of PWM.
    """

    # Convert to numpy if needed
    if isinstance(pwm, tf.Tensor):
        pwm = pwm.numpy()

    # If (batch, seq_len, alphabet_size) and batch = 1, remove batch axis
    if len(pwm.shape) == 3 and pwm.shape[0] == 1:
        pwm = pwm[0]

    if len(pwm.shape) != 2:
        raise ValueError("PWM must have two dimensions where the " +
                         "first dimension is the length of the sequence and the second " +
                         "dimension is the length of the alphabet.")

    if pwm.shape[1] != len(alphabet):
        raise ValueError("PWM must have the same alphabet size as the " +
                         "provided alphabet.")

    pwm_max = pwm.max(axis=1, keepdims=True)
    pwm_ismax = pwm == pwm_max
    if pwm_ismax.sum(axis=1).max() > 1 and force == False and allow_N == False:
        raise ValueError("At least one position in the PWM has multiple " +
                         "letters with the same probability.")

    alphabet = numpy.array(alphabet)

    if allow_N:
        n_inds = numpy.where(pwm.sum(axis=1) == 0)[0]
        dna_chars = alphabet[pwm.argmax(axis=1)]
        dna_chars[n_inds] = 'N'
    else:
        dna_chars = alphabet[pwm.argmax(axis=1)]

    return ''.join(dna_chars)


@numba.njit("void(int8[:, :], int8[:], int8[:])", cache=True)
def _fast_one_hot_encode(X_ohe, seq, mapping):
    """An internal function for quickly converting bytes to one-hot indexes.

    This uses numba JIT compilation for fast encoding.
    """

    for i in range(len(seq)):
        idx = mapping[seq[i]]
        if idx == -1:
            continue

        if idx == -2:
            raise ValueError("Encountered character that is not in " +
                             "`alphabet` or in `ignore`.")

        X_ohe[i, idx] = 1


def one_hot_encode(sequence, alphabet=['A', 'C', 'G', 'T'], dtype=numpy.int8,
                   ignore=['N'], desc=None, verbose=False, **kwargs):
    """Converts a string or list of characters into a one-hot encoding.

    This function will take in either a string or a list and convert it into a
    one-hot encoding using fast numba-compiled code. If the input is a string, 
    each character is assumed to be a different symbol, e.g. 'ACGT' is assumed 
    to be a sequence of four characters.

    Although this function will be used here primarily to convert nucleotide
    sequences into one-hot encoding with an alphabet of size 4, in principle
    this function can be used for any types of sequences.

    Parameters
    ----------
    sequence : str or list
        The sequence to convert to a one-hot encoding.

    alphabet : set or tuple or list
        A pre-defined alphabet where the ordering of the symbols is the same
        as the index into the returned array, i.e., for the alphabet ['A', 'B']
        the returned array will have a 1 at index 0 if the character was 'A'.
        Characters outside the alphabet are ignored and none of the indexes are
        set to 1. Default is ['A', 'C', 'G', 'T'].

    dtype : numpy.dtype, optional
        The data type of the returned encoding. Default is int8.

    ignore: list, optional
        A list of characters to ignore in the sequence, meaning that no bits
        are set to 1 in the returned one-hot encoding. Put another way, the
        sum across characters is equal to 1 for all positions except those
        where the original sequence is in this list. Default is ['N'].

    Returns
    -------
    ohe : numpy.ndarray, shape=(sequence_length, alphabet_size)
        A binary matrix where the first dimension is sequence_length and
        the second dimension is alphabet_size. This matches the Paddy/TF
        format of (length, channels).
    """

    for char in ignore:
        if char in alphabet:
            raise ValueError("Character {} in the alphabet ".format(char) +
                             "and also in the list of ignored characters.")

    if isinstance(alphabet, list):
        alphabet = ''.join(alphabet)

    ignore = ''.join(ignore)

    e = "utf8"
    seq_idxs = numpy.frombuffer(bytearray(sequence, e), dtype=numpy.int8)
    alpha_idxs = numpy.frombuffer(bytearray(alphabet, e), dtype=numpy.int8)
    ignore_idxs = numpy.frombuffer(bytearray(ignore, e), dtype=numpy.int8)

    one_hot_mapping = numpy.zeros(256, dtype=numpy.int8) - 2
    for i, idx in enumerate(alpha_idxs):
        one_hot_mapping[idx] = i

    for i, idx in enumerate(ignore_idxs):
        one_hot_mapping[idx] = -1

    n, m = len(sequence), len(alphabet)

    one_hot_encoding = numpy.zeros((n, m), dtype=numpy.int8)
    _fast_one_hot_encode(one_hot_encoding, seq_idxs, one_hot_mapping)

    # Return as (length, alphabet) format for TensorFlow/Paddy compatibility
    return tf.convert_to_tensor(one_hot_encoding.astype(dtype))


def reverse_complement(seq, ohe=True, complement_map={"A": "T", "C": "G", "G": "C",
                                                      "T": "A"}, allow_N=True):
    """Return the reverse complement of a sequence.

    This function will take in a one-hot encoded sequence, a string, or a
    batch of sequences, and return the reverse complement. 

    Parameters
    ----------
    seq: str or numpy.ndarray
        The sequence(s) to be reverse complemented.
        - If str: DNA sequence string (e.g., "ATCG")
        - If ohe=True: numpy.ndarray w/ shape (length, alphabet_size) or 
          (batch, length, alphabet_size) for one-hot encoded sequences
        - If ohe=False: numpy.ndarray w/ shape (batch, length, alphabet_size)
          for non-one-hot encoded sequences (just reverses sequence axis)

    ohe: bool, optional
        Whether the input is one-hot encoded. If True, applies nucleotide
        complement transformation. If False, only reverses the sequence axis.
        Default is True.

    complement_map: dict, optional
        The ordering and complement of each nucleotide. When the input is a
        string, this is used to directly convert characters. When the input is
        a numpy array with ohe=True, the ordering of *keys* is assumed to be 
        the order of characters in the alphabet. Default is {"A": "T", "C": "G", 
        "G": "C", "T": "A"}.

    allow_N: bool, optional
        Whether to allow N characters when doing the reverse complement. Only
        matters when reverse complementing strings. Default is True.

    Returns
    -------
    rev_comp: str or numpy.ndarray
        The reverse complemented sequence(s) with the same shape as input.

    Examples
    --------
    >>> # String input
    >>> reverse_complement("ATCG")
    'CGAT'

    >>> # One-hot encoded single sequence
    >>> seq_ohe = numpy.array([[1,0,0,0], [0,0,0,1]])  # "AT"
    >>> reverse_complement(seq_ohe, ohe=True)
    # Returns one-hot for "AT" (complement of "TA")

    >>> # Batch of sequences, non-one-hot (just reverse)
    >>> batch = numpy.random.rand(10, 100, 4)
    >>> reverse_complement(batch, ohe=False)
    # Returns batch with reversed sequence axis
    """

    if isinstance(seq, str):
        # String input: apply complement map
        seq_rc = []
        for char in seq:
            if char in complement_map:
                seq_rc.append(complement_map[char])
            elif char == 'N' and allow_N:
                seq_rc.append('N')
            else:
                raise ValueError("'{}' not in complement map".format(char))
        seq_rc = ''.join(reversed(seq_rc))

    elif isinstance(seq, (numpy.ndarray, tf.Tensor)):
        if isinstance(seq, tf.Tensor):
            seq = seq.numpy()

        if ohe:
            # One-hot encoded: flip sequence axis and swap nucleotide positions
            chars = list(complement_map.keys())
            idxs = [chars.index(char) for char in complement_map.values()]

            # Determine sequence axis based on shape
            if seq.ndim == 2:
                # Single sequence: (length, alphabet_size)
                seq_rc = numpy.flip(seq, axis=0)[:, idxs]
            elif seq.ndim == 3:
                # Batch of sequences: (batch, length, alphabet_size)
                seq_rc = numpy.flip(seq, axis=1)[:, :, idxs]
            else:
                raise ValueError(
                    "Input array must be 2D or 3D, got shape {}".format(seq.shape))
        else:
            # Non-one-hot: just reverse the sequence axis
            if seq.ndim == 2:
                seq_rc = numpy.flip(seq, axis=0)
            elif seq.ndim == 3:
                seq_rc = numpy.flip(seq, axis=1)
            else:
                raise ValueError(
                    "Input array must be 2D or 3D, got shape {}".format(seq.shape))
    else:
        raise TypeError("Input must be str, numpy.ndarray, or tf.Tensor")

    return seq_rc


def random_one_hot(shape, probs=None, dtype='int8', random_state=None):
    """Generate random one-hot encodings. Useful for debugging.

    This function will generate random one-hot encodings where the last
    dimension has a single element being a one and every other element
    being a zero. Primarily used for debugging and generating backgrounds.

    Parameters
    ----------
    shape: tuple
        The shape of the 3D array to generate: (batch_size, seq_length, alphabet_size).

    probs: tuple, list, numpy.ndarray, or None, optional
        A 2D array of probabilities where the first dimension is the batch size
        equal to the batch size of shape and the second dimension is the alphabet
        size. The values should be the probability of that character occurring
        in that sequence. If a batch size of 1 is used when the batch size of
        shape is greater than 1, the same probabilities are used for each sequence.
        The sum of probabilities across the alphabet axis must be equal to 1.
        If None, use a uniform distribution. Default is None.

    dtype: str or numpy.dtype, optional
        The datatype to return the matrix as. Default is 'int8'.

    random_state: int or numpy.random.RandomState or None, optional
        The random state to use for generation. If None, do not use a
        deterministic seed. Default is None.

    Returns
    -------
    ohe: numpy.ndarray, shape=(batch_size, seq_length, alphabet_size)
        A numpy array with the specified shape that is one-hot encoded.
    """

    if not isinstance(shape, tuple) or len(shape) != 3:
        raise ValueError(
            "Shape must be a tuple with 3 dimensions: (batch, length, alphabet).")

    if not isinstance(random_state, numpy.random.RandomState):
        random_state = numpy.random.RandomState(random_state)

    if isinstance(probs, list):
        probs = numpy.array(probs)

    n = shape[2]  # alphabet size
    ohe = numpy.zeros(shape, dtype=dtype)

    for i in range(ohe.shape[0]):
        if probs is None:
            probs_ = None
        elif probs.ndim == 1:
            probs_ = probs
        elif probs.shape[0] == 1:
            probs_ = probs[0]
        else:
            probs_ = probs[i]

        choices = random_state.choice(n, size=shape[1], p=probs_)
        ohe[i, numpy.arange(shape[1]), choices] = 1

    return tf.convert_to_tensor(ohe)


def example_to_fasta_coords(example_df, loci_df, window=None, one_indexed=False):
    """Converts coordinates within a given example to those in a FASTA file.

    Many analyses involve extracting windows from the genome and processing them
    independently. When coordinates are identified on these windows, e.g.,
    seqlets or other spans, all that is recorded is the example index and the
    relative positions from the start of the extracted window. Frequently, though,
    one wants to know these coordinates on the original genome or other entity
    encoded in a FASTA file. This function cross-references the example indexes
    and BED-formatted locus file that the examples were extracted from to get
    the coordinates on the original sequence.

    Parameters
    ----------
    example_df: pandas.DataFrame
        A BED-formatted dataframe where the first three columns must be the example
        index, the start, and the end coordinates.

    loci_df: pandas.DataFrame
        A BED-formatted dataframe where the first three columns must be the example
        index, the start, and the end coordinates.

    window: int or None, optional
        If None, use the start and end coordinates in `loci_df` directly. If an
        integer, these start and end coordinates will be replaced with a window
        centered on the middle of these original coordinates that spans this
        window length. Default is None.

    one_indexed: bool, optional
        Whether `example_df` is one-indexed. Default is False.

    Returns
    -------
    coords_df: pandas.DataFrame
        A dataframe of corrected coordinates where the first three columns are
        chrom, start, and end.
    """

    loci_chroms = loci_df.iloc[:, 0].values
    loci_starts = loci_df.iloc[:, 1].values
    loci_ends = loci_df.iloc[:, 2].values

    if window is not None:
        mid = (loci_ends + loci_starts) // 2

        loci_starts = mid - window // 2
        loci_ends = mid + window // 2 + window % 2

    coords_ = example_df.iloc[:, :3].values
    idxs = coords_[:, 0]

    example_chroms = loci_chroms[idxs]
    example_starts = loci_starts[idxs] + coords_[:, 1] - one_indexed
    example_ends = loci_starts[idxs] + coords_[:, 2] - one_indexed

    names = loci_df.columns[:3]
    key_map = dict(zip(example_df.columns[:3], names))
    coords_df = example_df.copy()
    coords_df = coords_df.rename(key_map, axis='columns')
    coords_df[names[0]] = example_chroms
    coords_df[names[1]] = example_starts
    coords_df[names[2]] = example_ends
    return coords_df


def extract_signal(loci, X, verbose=False):
    """Extracts the signal at coordinates from an array of examples.

    This function takes in a dataframe with the first three columns being the
    example index, the start (inclusive) and the end (not inclusive) and
    returns the signal sum across those coordinates from the array. This can
    be used, for instance, to extract attributions or genomics signal from
    windows. This sum is done separately for each signal in the array.

    Parameters
    ----------
    loci: pandas.DataFrame
        A set of loci to extract signal from. Multiple loci can be present on
        the same example in X and the starts and ends can overlap.

    X: numpy.ndarray or tf.Tensor, shape=(-1, sequence_length, n_signals)
        A 3D array where the first dimension corresponds to the examples, the
        second dimension corresponds to sequence length, and the third
        dimension corresponds to the number of signals being measured.

    verbose: bool, optional
        Whether to print a progress bar tracking the extraction process. Default
        is False.

    Returns
    -------
    Y: numpy.ndarray, shape=(n_loci, n_signals)
        The sum of the signal across all positions for each of the loci.
    """

    validate_input(X, "X", shape=(-1, -1, -1))

    if isinstance(X, tf.Tensor):
        X = X.numpy()

    Y = numpy.zeros((loci.shape[0], X.shape[2]), dtype=X.dtype)

    loci_values = loci.values[:, :3].astype(int)
    for i, (idx, start, end) in enumerate(tqdm(loci_values, disable=not verbose)):
        Y[i] = X[idx, start:end, :].sum(axis=0)

    return Y
