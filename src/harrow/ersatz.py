# ersatz.py
# Sequence manipulation operations
# Migrated from tangermeme (PyTorch) to NumPy/TensorFlow

import numba
import numpy
import tensorflow as tf
from tqdm import tqdm

from .utils import validate_input, one_hot_encode, random_one_hot, _cast_as_array, reverse_complement


def insert(X, motif, start=None, alphabet=['A', 'C', 'G', 'T']):
    """Insert a motif into a set of sequences at a defined position.

    This function will take in an array of one-hot encoded sequences or a string
    that can be one-hot encoded and insert the motif into the defined
    position. It will then return a copy of the data with the insertion,
    leaving the original data unperturbed.

    Importantly, an *insertion* means that the entire original sequence is still
    present, albeit in two halves with the inserted motif in the middle.
    The returned sequence will be longer than the original sequence.

    Parameters
    ----------
    X: numpy.ndarray, shape=(-1, length, len(alphabet))
        A one-hot encoded set of sequences to have a motif inserted into.

    motif: numpy.ndarray or str, shape=(-1, motif_length, len(alphabet))
        A one-hot encoded version of a short motif or a string to insert.

    start: int or None, optional
        The starting position of where to insert the motif. If None,
        insert the motif into the middle of the sequence. Default is None.

    alphabet: set or tuple or list, optional
        The alphabet for one-hot encoding if motif is a string.
        Default is ['A', 'C', 'G', 'T'].

    Returns
    -------
    Y: numpy.ndarray, shape=(-1, length+motif_length, len(alphabet))
        A one-hot encoded set of sequences with the motif inserted.
    """

    if isinstance(motif, str):
        motif = one_hot_encode(motif, alphabet=alphabet)[numpy.newaxis, :, :]

    motif = _cast_as_array(motif)
    X = _cast_as_array(X)

    if motif.shape[0] == 1:
        motif = numpy.repeat(motif, X.shape[0], axis=0)

    validate_input(X, "X", ohe=True, ohe_dim=2)
    validate_input(motif, "motif", shape=(-1, -1,
                   X.shape[2]), ohe=True, ohe_dim=2)

    if start is not None:
        if start < 0 or start > X.shape[1]:
            raise ValueError(
                "Provided start falls off the end of the sequence")
    else:
        start = X.shape[1] // 2

    return numpy.concatenate([X[:, :start, :], motif, X[:, start:, :]], axis=1)


def substitute(X, motif, start=None, alphabet=['A', 'C', 'G', 'T'], ignore=['N']):
    """Substitute a motif into a set of sequences at a defined position.

    This function will take in an array of one-hot encoded sequences or a string
    that can be one-hot encoded and will substitute a motif at a defined
    position. A *substitution* means that part of the original sequence will
    be missing. The returned sequence will be the same length as the original.

    If all-zeros positions are present in a motif (e.g., from 'N' characters),
    the original sequence will be kept at those positions.

    Parameters
    ----------
    X: numpy.ndarray, shape=(-1, length, len(alphabet))
        A one-hot encoded set of sequences to have a motif substituted into.

    motif: numpy.ndarray or str, shape=(-1, motif_length, len(alphabet))
        A one-hot encoded version of a short motif or a string to substitute.

    start: int or None, optional
        The starting position of where to substitute the motif. If None,
        substitute the motif into the middle of the sequence. Default is None.

    alphabet: set or tuple or list, optional
        The alphabet for one-hot encoding if motif is a string.
        Default is ['A', 'C', 'G', 'T'].

    ignore: set or tuple or list, optional
        A set of characters indicating that the original value of the sequence
        should be maintained at this position. Default is ['N'].

    Returns
    -------
    Y: numpy.ndarray, shape=(-1, length, len(alphabet))
        A one-hot encoded set of sequences with the motif substituted.
    """

    if isinstance(motif, str):
        motif = one_hot_encode(motif, alphabet=alphabet, ignore=ignore)[
            numpy.newaxis, :, :]

    motif = _cast_as_array(motif)
    X = _cast_as_array(X).copy()  # Make a copy to avoid modifying original

    validate_input(X, "X", ohe=True, ohe_dim=2, allow_N=False)
    validate_input(motif, "motif", shape=(-1, -1,
                   X.shape[2]), ohe=True, ohe_dim=2, allow_N=True)

    if motif.shape[1] > X.shape[1]:
        raise ValueError("Motif cannot be longer than sequence.")

    if start is not None:
        if start < 0 or start > (X.shape[1] - motif.shape[1]):
            raise ValueError(
                "Provided start falls off the end of the sequence")
    else:
        start = X.shape[1] // 2 - motif.shape[1] // 2

    # Only substitute positions where motif is not all zeros (not 'N')
    for i in range(motif.shape[1]):
        idxs = motif[:, i, :].sum(axis=1) == 1

        if idxs.all():
            X[:, start+i, :] = motif[:, i, :]
        elif idxs.sum() > 0:
            X[idxs, start+i, :] = motif[idxs, i, :]

    return X


def multisubstitute(X, motifs, spacing, start=None, alphabet=['A', 'C', 'G', 'T'],
                    ignore=['N']):
    """Substitute a set of motifs into sequences with provided spacings.

    This function will take in a list of strings or one-hot encoded motifs
    and will substitute them into the sequences given the provided spacings.

    Parameters
    ----------
    X: numpy.ndarray, shape=(-1, length, len(alphabet))
        A one-hot encoded set of sequences to have motifs substituted into.

    motifs: list of numpy.ndarray or str
        A list of strings or one-hot encoded motifs to substitute.

    spacing: list or int
        An integer specifying a constant spacing between all motifs or a list
        of spacings of length equal to n-1 where n is the number of motifs.

    start: int or None, optional
        start defines the first position of the substitution (0-based).
        The motif begins at this index and extends forward for its full length.

    alphabet: set or tuple or list, optional
        The alphabet for one-hot encoding. Default is ['A', 'C', 'G', 'T'].

    ignore: set or tuple or list, optional
        Characters to ignore during substitution. Default is ['N'].

    Returns
    -------
    Y: numpy.ndarray, shape=(-1, length, len(alphabet))
        A one-hot encoded set of sequences with the motifs substituted.
    """

    motif_lengths = [len(motif) if isinstance(motif, str) else motif.shape[1]
                     for motif in motifs]

    if not isinstance(spacing, (int, list)):
        raise ValueError("Spacings must be an integer or a list of integers.")

    if isinstance(spacing, int):
        spacing = [spacing for _ in range(len(motifs) - 1)]

    if len(spacing) != (len(motifs) - 1):
        raise ValueError("Must provide n-1 spacings for n motifs.")

    for l in spacing:
        if l < 0 or l >= X.shape[1]:
            raise ValueError("Spacing cannot be smaller than zero or " +
                             "larger than the sequence being inserted into.")

    if start is None:
        n = sum(spacing) + sum(motif_lengths)
        start = X.shape[1] // 2 - n // 2

        if start < 0:
            raise ValueError("Sum of motif lengths and spacing cannot be " +
                             "larger than the sequence being inserted into.")

    for i in range(len(spacing)):
        X = substitute(X, motifs[i], start=start, alphabet=alphabet,
                       ignore=ignore)
        start += motif_lengths[i] + spacing[i]

    X = substitute(X, motifs[-1], start=start,
                   alphabet=alphabet, ignore=ignore)
    return X


def delete(X, start, end):
    """Delete a portion of a sequence.

    This function will take in an array of one-hot encoded sequences and a pair
    of numbers representing the start and the end of the portion to remove, and
    will return an array that is missing those positions.

    The sequence returned will be shorter than the original sequence.

    Parameters
    ----------
    X: numpy.ndarray, shape=(-1, length, len(alphabet))
        A one-hot encoded set of sequences to have a portion deleted.

    start: int
        The starting position to remove, inclusive.

    end: int
        The final position to remove, not inclusive.

    Returns
    -------
    Y: numpy.ndarray, shape=(-1, length-(end-start), len(alphabet))
        A one-hot encoded set of sequences with a portion deleted.
    """

    X = _cast_as_array(X)

    if start < 0 or start > X.shape[1]:
        raise ValueError("Start cannot be below zero or greater than " +
                         "the length of the sequence.")

    if end < 0 or end > X.shape[1] or end <= start:
        raise ValueError("End must come after start, must be greater " +
                         "than zero, and cannot be greater than the length of the sequence.")

    validate_input(X, "X", ohe=True, ohe_dim=2)
    return numpy.concatenate([X[:, :start, :], X[:, end:, :]], axis=1)


def shuffle(X, start=0, end=-1, n=1, random_state=None):
    """Replace a region of the provided sequences with a shuffled version.

    This function will take in a batch of sequences and shuffle the specified
    region between `start` and `end`. This means that the returned sequences
    will have the same number of each nucleotide in the specified region, but
    in different positions.

    Importantly, the i-th shuffle of each sequence uses the same shuffling.
    Put another way, every sequence is shuffled *the same way* each iteration.

    Parameters
    ----------
    X: numpy.ndarray, shape=(-1, length, len(alphabet))
        A one-hot encoded set of sequences where a portion will be shuffled.

    start: int, optional
        The starting position of where to shuffle the sequence, inclusive.
        Default is 0, shuffling the entire sequence.

    end: int, optional
        The ending position of where to shuffle the sequence. If end is
        negative then it is inclusive. Default is -1, shuffling the entire sequence.

    n: int, optional
        The number of times to shuffle that region. Default is 1.

    random_state: int, numpy.random.RandomState, or None, optional
        Whether to use a specific random seed when generating the shuffle to
        ensure reproducibility. If None, do not use a reproducible seed. Default
        is None.

    Returns
    -------
    Y: numpy.ndarray, shape=(-1, n, length, len(alphabet))
        A one-hot encoded set of sequences with a shuffled portion.
    """

    X = _cast_as_array(X)
    validate_input(X, "X", ohe=True, ohe_dim=2)

    if end < 0:
        end = X.shape[1] + 1 + end

    if end <= start:
        raise ValueError("End must come after start.")

    if end > X.shape[1] or start < 0:
        raise ValueError("Start or end are falling off the edge of X.")

    if not isinstance(random_state, numpy.random.RandomState):
        random_state = numpy.random.RandomState(random_state)

    X_shufs = []
    for i in range(n):
        idxs = numpy.arange(end-start)
        random_state.shuffle(idxs)

        X_ = X.copy()
        X_[:, start:end, :] = X[:, start:end, :][:, idxs, :]
        X_shufs.append(X_)

    # Stack: (batch, n_shuffles, length, alphabet)
    return numpy.stack(X_shufs, axis=1)


params = 'void(int64, int64, int32[:], int32[:, :], int32[:], '
params += 'int32[:, :], float32[:, :, :], int32)'


@numba.jit(params, nopython=False, cache=True)
def _fast_dinuc_shuffle(n_shuffles, n_chars, idxs, next_idxs, next_idxs_counts,
                        counters, shuffled_sequences, random_state):
    """An internal function for fast dinucleotide shuffling using numba."""

    numpy.random.seed(random_state)

    for i in range(n_shuffles):
        for char in range(n_chars):
            n = next_idxs_counts[char]

            next_idxs_ = numpy.arange(n)
            next_idxs_[:-1] = numpy.random.permutation(n-1)  # Keep last index
            next_idxs[char, :n] = next_idxs[char, :n][next_idxs_]

        idx = 0
        shuffled_sequences[i, 0, idxs[idx]] = 1
        for j in range(1, len(idxs)):
            char = idxs[idx]
            count = counters[i, char]
            idx = next_idxs[char, count]

            counters[i, char] += 1
            shuffled_sequences[i, j, idxs[idx]] = 1


def _dinucleotide_shuffle_single(X, n_shuffles=1, random_state=None, verbose=False):
    """An internal function for dinucleotide shuffling a single sequence.

    This function takes in a one-hot encoded sequence and returns a set of
    one-hot encoded sequences that are dinucleotide shuffled. The approach
    constructs a transition matrix between nucleotides, keeps the first and
    last nucleotide constant, and then randomly selects transitions until all
    nucleotides have been observed (Eulerian path).

    Adapted from:
    https://github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py

    Parameters
    ----------
    X: numpy.ndarray, shape=(length, alphabet_size)
        A single one-hot encoded sequence to be shuffled.

    n_shuffles: int, optional
        The number of shuffles to generate. Default is 1.

    random_state: int or None, optional
        Random seed for reproducibility. Default is None.

    verbose: bool, optional
        Whether to print warnings. Default is False.

    Returns
    -------
    shuffled_sequences: numpy.ndarray, shape=(n, length, alphabet_size)
        The shuffled sequences.
    """

    if random_state is None:
        random_state = numpy.random.randint(0, 9999999)

    seq_len, n_chars = X.shape
    idxs = X.argmax(axis=1).astype(numpy.int32)

    next_idxs = numpy.zeros((n_chars, seq_len), dtype=numpy.int32)
    next_idxs_counts = numpy.zeros(n_chars, dtype=numpy.int32)

    for char in range(n_chars):
        next_idxs_ = numpy.where(idxs[:-1] == char)[0]
        n = len(next_idxs_)

        next_idxs[char][:n] = next_idxs_ + 1
        next_idxs_counts[char] = n

    shuffled_sequences = numpy.zeros((n_shuffles, seq_len, n_chars),
                                     dtype=numpy.float32)
    counters = numpy.zeros((n_shuffles, n_chars), dtype=numpy.int32)

    _fast_dinuc_shuffle(n_shuffles, n_chars, idxs, next_idxs, next_idxs_counts,
                        counters, shuffled_sequences, random_state)
    
    # Only check conserved positions if sequence length > 2
    # (for length <= 2, conserved = shuffled_sequences[:, 1:-1, :] would be empty)
    if seq_len > 2:
        conserved = shuffled_sequences[:, 1:-1, :].sum(axis=0)
        if conserved.size > 0:  # Additional safety check
            if conserved.max() == n_shuffles and n_shuffles > 1:
                if verbose:
                    print("Warning: At least one position in dinucleotide shuffle " +
                          "is identical across all shuffles.")
            if conserved.max(axis=0).min() == n_shuffles and n_shuffles > 1:
                raise ValueError("All dinucleotide shuffles yield identical " +
                                 "sequences, potentially due to a lack of diversity in sequence.")

    return shuffled_sequences


def dinucleotide_shuffle(X, start=0, end=-1, n=20, random_state=None,
                         verbose=False):
    """Given a one-hot encoded sequence, dinucleotide shuffle it.

    This function takes in one-hot encoded sequences and returns a set of
    one-hot encoded sequences that are dinucleotide shuffled. The approach
    constructs a transition matrix between nucleotides, keeps the first and
    last nucleotide constant, and then randomly selects transitions until all
    nucleotides have been observed (Eulerian path).

    This preserves dinucleotide frequencies within the shuffled region.

    Adapted from:
    https://github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py

    Parameters
    ----------
    X: numpy.ndarray, shape=(-1, length, alphabet_size)
        A one-hot encoded set of sequences to be shuffled.

    start: int, optional
        The starting position of where to shuffle the sequence, inclusive.
        Default is 0, shuffling the entire sequence.

    end: int, optional
        The ending position of where to shuffle the sequence. If end is
        negative then it is inclusive. Default is -1, shuffling the entire sequence.

    n: int, optional
        The number of times to shuffle that region. Default is 20.

    random_state: int or None, optional
        Whether to use a specific random seed when generating the shuffle,
        to ensure reproducibility. If None, do not use a reproducible seed.
        Default is None.

    verbose: bool, optional
        Whether to print warnings about conserved positions. Default is False.

    Returns
    -------
    shuffled_sequences: numpy.ndarray, shape=(-1, n, length, alphabet_size)
        The shuffled sequences.
    """

    X = _cast_as_array(X)
    validate_input(X, "X", shape=(-1, -1, -1), ohe=True, ohe_dim=2)

    if end < 0:
        end = X.shape[1] + 1 + end

    if random_state is None:
        random_state = numpy.random.randint(0, 9999999)

    X_shufs = []
    for i in range(X.shape[0]):
        insert_ = _dinucleotide_shuffle_single(X[i, start:end, :], n_shuffles=n,
                                               random_state=random_state+i, verbose=verbose)

        X_shuf = numpy.repeat(X[i:i+1, :, :], n, axis=0)
        X_shuf[:, start:end, :] = insert_
        X_shufs.append(X_shuf)

    return numpy.stack(X_shufs, axis=0)


def randomize(X, start, end, probs=[[0.25, 0.25, 0.25, 0.25]], n=1,
              random_state=None):
    """Replace a region of the provided sequences with randomly drawn sequence.

    This function will take in a batch of sequences and replace the region
    specified by `start` and `end` with randomly generated sequences. It will
    do this `n` times for each sequence in X.

    This function does not shuffle the sequence in the specified region but
    replaces it with random substitutions.

    Parameters
    ----------
    X: numpy.ndarray, shape=(-1, length, alphabet_size)
        A one-hot encoded set of sequences where a portion should be randomized.

    start: int
        The starting position of where to randomize the sequence, inclusive.

    end: int
        The ending position of where to randomize the sequence, not inclusive.

    probs: 2D array, optional
        A 2D array of probabilities. The shape is either (1, alphabet_size) or
        (len(X), alphabet_size). Default is [[0.25, 0.25, 0.25, 0.25]].

    n: int, optional
        The number of times to randomize that region. Default is 1.

    random_state: int, numpy.random.RandomState, or None, optional
        Whether to use a specific random seed to ensure reproducibility.
        If None, do not use a reproducible seed. Default is None.

    Returns
    -------
    X_rands: numpy.ndarray, shape=(-1, n, length, alphabet_size)
        A one-hot encoded set of sequences with randomized regions.
    """

    X = _cast_as_array(X)
    probs = _cast_as_array(probs)

    if not isinstance(random_state, numpy.random.RandomState):
        random_state = numpy.random.RandomState(random_state)

    validate_input(X, "X", ohe=True, ohe_dim=2)
    validate_input(probs, "probs", shape=(-1, -1), min_value=0, max_value=1)

    if end <= start:
        raise ValueError("End must come after start.")

    if end > X.shape[1] or start < 0:
        raise ValueError("Start or end are falling off the edge of X.")

    X_rands = []
    for i in range(n):
        substitute_ohe = random_one_hot((X.shape[0], end-start, probs.shape[1]),
                                        probs=probs, random_state=random_state)

        X_rand = substitute(X, substitute_ohe, start=start)
        X_rands.append(X_rand)

    return numpy.stack(X_rands, axis=1)
