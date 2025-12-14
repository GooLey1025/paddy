# seqlet.py
# Seqlet extraction from attributions
# Migrated from tangermeme (PyTorch) to TensorFlow/NumPy
# Original authors: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code written by Avanti Shrikumar

import math
import numpy
import numba
import tensorflow as tf
import pandas

from .utils import validate_input
from .utils import characters

from sklearn.isotonic import IsotonicRegression


def _laplacian_null(X_sum, num_to_samp=10000, random_state=1234):
    """An internal function for calculating a null distribution.

    The TF-MoDISCo seqlet calling procedure works by constructing a null
    distribution of values using Laplacian distributions for the positive and
    negative values separately. This method constructs those Laplacian
    distributions and then samples a given number of values from them.


    Parameters
    ----------
    X_sum: numpy.ndarray, shape=(-1, length)
        A tensor of summed attribution values over the provided window size.

    num_to_samp: int
        The number of values to sample from the fit Laplacian distributions.

    random_state: int, optional
        The random state to use for the sampling. Default is 1234.


    Returns
    -------
    pos_values: numpy.ndarray, shape=(-1,)
        A vector of sampled values from the Laplacian distribution fit to the
        positive values.

    neg_values: numpy.ndarray, shape=(-1,)
        A vector of sampled values from the Laplacian distribution fit to the
        negative values.
    """

    values = X_sum.flatten()

    hist, bin_edges = numpy.histogram(values, bins=1000)
    peak = numpy.argmax(hist)
    l_edge, r_edge = bin_edges[peak:peak+2]
    top_values = values[(l_edge < values) & (values < r_edge)]

    hist, bin_edges = numpy.histogram(top_values, bins=1000)
    peak = numpy.argmax(hist)
    mu = (bin_edges[peak] + bin_edges[peak+1]) / 2

    pos_values = values[values >= mu]
    neg_values = values[values <= mu]

    # Take the most aggressive lambda over all percentiles
    quantiles = numpy.arange(19) * 5. / 100
    lq = -numpy.log(1 - quantiles)
    pos_q = numpy.quantile(a=pos_values, q=quantiles)
    neg_q = numpy.quantile(a=neg_values, q=1-quantiles)

    pos_lambda = numpy.max(lq / (pos_q - mu))
    neg_lambda = numpy.max(lq / (numpy.abs(neg_q - mu)))

    prob_pos = len(pos_values) / len(values)
    urand = numpy.random.RandomState(random_state).uniform(size=(num_to_samp, 2))

    icdf = numpy.log(1 - urand[:, 1])

    sampled_vals = numpy.where(urand[:, 0] < prob_pos, -icdf / pos_lambda + mu,
                                mu + icdf / neg_lambda)
    return sampled_vals[sampled_vals >= 0], sampled_vals[sampled_vals < 0]


def _iterative_extract_seqlets(X_sum, window_size, flank, suppress):
    """This internal function iteratively identifies seqlets.

    This function extracts seqlets from the signal by iteratively identifying
    the position across all examples with maximum windowed attribution sum and
    zeroing out the attributions around that position to ensure another seqlet
    cannot be identified from the same position.


    Parameters
    ----------
    X_sum: numpy.ndarray, shape=(-1, length)
        A tensor of summed attribution values over the provided window size.

    window_size: int
        The size of the window of attribution values to sum over when
        identifying seqlets. This is not the only component of seqlet size but
        is the most important.

    flank: int
        A number of characters on either end of the window to add to each
        seqlet. This is done primarily to remove the effect of surrounding
        positions and not have overlapping seqlets.

    suppress: int
        The number of positions to the left and right of the maximum attribution
        position to zero out the attributions of.


    Returns
    -------
    seqlets: list
        A list of tuples containing the example index, the start of the seqlet,
        the end of the seqlet, and the sum of attributions within the seqlet.
    """

    n, d = X_sum.shape
    seqlets = []
    for i in range(n):
        while True:
            argmax = numpy.argmax(X_sum[i], axis=0)
            max_val = X_sum[i, argmax]
            if max_val == -numpy.inf:
                break

            seqlet = i, argmax - flank, argmax + window_size + flank
            seqlets.append(seqlet)

            l_idx = int(max(numpy.floor(argmax + 0.5 - suppress), 0))
            r_idx = int(min(numpy.ceil(argmax + 0.5 + suppress), d))
            X_sum[i, l_idx:r_idx] = -numpy.inf

    return seqlets


def _isotonic_thresholds(values, null_values, increasing, target_fdr,
                         min_frac_neg=0.95):
    """This function uses an isotonic regression to find FDR thresholds.

    Given a set of attribution values summed over the provided window and a
    set of sampled negative values from the Laplacian distribution, find the
    threshold at which real attributions are separated from the null values at
    a given FDR threshold.

    As an implementation note, this method uses the scikit-learn
    IsotonicRegression function. This method works with numpy arrays.


    Parameters
    ----------
    values: numpy.ndarray, shape=(-1,)
        Sampled attribution window sums from the real data.

    null_values: numpy.ndarray, shape=(-1,)
        Sampled values from the Laplacian distribution that represents the null
        data.

    increasing: bool
        Whether the data should be modeled as increasing or decreasing in size.

    target_fdr: float
        The FDR threshold to use to separate positive and negative attribution
        values.

    min_frac_neg: float, optional
        The minimum number of values that need to be assigned to the null
        distribution. Default is 0.95.


    Returns
    -------
    threshold: float
        A single value representing the threshold to use on attribution window
        sums to separate seqlets from background given the FDR threshold.
    """

    n1, n2 = len(values), len(null_values)

    X = numpy.concatenate([values, null_values], axis=0)
    y = numpy.concatenate([numpy.ones(n1), numpy.zeros(n2)], axis=0)

    w = len(values) / len(null_values)
    sample_weight = numpy.concatenate([numpy.ones(n1), numpy.ones(n2)*w], axis=0)

    model = IsotonicRegression(out_of_bounds='clip', increasing=increasing)
    model.fit(X, y, sample_weight=sample_weight)
    y_hat = model.transform(values)

    min_prec_x = model.X_min_ if increasing else model.X_max_
    min_precision = max(model.transform([min_prec_x])[0], 1e-7)
    implied_frac_neg = -1 / (1 - (1 / min_precision))

    if (implied_frac_neg > 1.0 or implied_frac_neg < min_frac_neg):
        implied_frac_neg = max(min(1.0, implied_frac_neg), min_frac_neg)

    precisions = numpy.minimum(numpy.maximum(1 + implied_frac_neg*(
        1 - (1 / numpy.maximum(y_hat, 1e-7))), 0.0), 1.0)
    precisions[-1] = 1
    return values[precisions >= (1 - target_fdr)][0]


def tfmodisco_seqlets(X_attr, window_size=21, flank=10, target_fdr=0.2,
                      min_passing_frac=0.03, max_passing_frac=0.2,
                      weak_threshold_for_counting_sign=0.8):
    """Extract seqlets using the procedure from TF-MoDISco.

    Seqlets are contiguous spans of high attribution characters. This method
    for identifying them is the one that is implemented in the TF-MoDISco
    algorithm. Importantly, TF-MoDISco does several post-processing steps
    on these seqlets that are interleaved in the pattern identification
    procedure so the final set of seqlets actually used by patterns in
    TF-MoDISco will be smaller than the set that are returned here.

    The seqlets returned by this procedure have been optimized to be useful
    for motif discovery, and so are generally much longer and less sensitive
    than one might initially expect. The seqlets are longer because the local
    context that patterns occur in might be useful, and because uninformative
    characters on the flanks can easily be trimmed off. The seqlets are also
    less sensitive, in the sense that sometimes spans that one might call a
    seqlet by eye are missed, to prevent noise from contaminating the found
    patterns.


    Parameters
    ----------
    X_attr: numpy.ndarray or tf.Tensor, shape=(-1, length)
        A tensor of attribution values for each position in the sequence.
        The attributions here will be summed across the length of the alphabet
        so the values must be amenable to that. This means that, most likely,
        it should be attribution values multiplied by the one-hot encodings
        so only the present characters have attributions.

    window_size: int, optional
        The size of the window of attribution values to sum over when
        identifying seqlets. This is not the only component of seqlet size but
        is the most important. Default is 21.

    flank: int, optional
        A number of characters on either end of the window to add to each
        seqlet. This is done primarily to remove the effect of surrounding
        positions and not have overlapping seqlets. Default is 10.

    target_fdr: float, optional
        A FDR value to set on attribution score sums over windows when
        separating called seqlets from background. Default is 0.2.

    min_passing_frac: float, optional
        Require that at least this proportion of windows pass seqlet
        identification. Default is 0.03.

    max_passing_frac: float, optional
        Require that no more than this proportion of windows pass seqlet
        identification. Default is 0.2.

    weak_threshold_for_counting_sign: float, optional
        A minimal threshold to use when setting the final threshold value
        separating seqlets from non-seqlets.


    Returns
    -------
    seqlets: pandas.DataFrame, shape=(-1, 4)
        A tensor containing the example index, start position, end position,
        and attribution sum for each seqlet that passes the thresholds.
    """

    # Convert TensorFlow tensor to numpy if needed
    if isinstance(X_attr, tf.Tensor):
        X_attr = X_attr.numpy()

    validate_input(X_attr, "X_attr", shape=(-1, -1))
    suppress = int(0.5*window_size) + flank

    # Compute windowed sums using numpy
    # This is equivalent to PyTorch's unfold operation
    n, length = X_attr.shape
    X_sum = numpy.zeros((n, length - window_size + 1))
    for i in range(length - window_size + 1):
        X_sum[:, i] = X_attr[:, i:i+window_size].sum(axis=1)

    values = X_sum.flatten()
    if len(values) > 1000000:
        values = numpy.random.RandomState(1234).choice(
            a=values, size=1000000, replace=False)

    pos_values = values[values >= 0]
    neg_values = values[values < 0]
    pos_null_values, neg_null_values = _laplacian_null(X_sum)

    pos_threshold = _isotonic_thresholds(pos_values, pos_null_values,
                                         increasing=True, target_fdr=target_fdr)
    neg_threshold = _isotonic_thresholds(neg_values, neg_null_values,
                                         increasing=False, target_fdr=target_fdr)

    values = numpy.concatenate([pos_values, neg_values])
    frac_passing = (sum(values >= pos_threshold) +
                    sum(values <= neg_threshold)) / len(values)

    if frac_passing < min_passing_frac:
        pos_threshold = numpy.quantile(numpy.abs(values), q=1-min_passing_frac)
        neg_threshold = -pos_threshold

    if frac_passing > max_passing_frac:
        pos_threshold = numpy.quantile(numpy.abs(values), q=1-max_passing_frac)
        neg_threshold = -pos_threshold

    distribution = numpy.sort(numpy.abs(X_sum.flatten()))

    transformed_pos_threshold = numpy.sign(pos_threshold) * numpy.searchsorted(
        distribution, v=abs(pos_threshold)) / len(distribution)
    transformed_neg_threshold = numpy.sign(neg_threshold) * numpy.searchsorted(
        distribution, v=abs(neg_threshold)) / len(distribution)

    idxs = (X_sum >= pos_threshold) | (X_sum <= neg_threshold)

    X_sum[idxs] = numpy.abs(X_sum[idxs])
    X_sum[~idxs] = -numpy.inf

    if flank > 0:
        X_sum[:, :flank] = -numpy.inf
        X_sum[:, -flank:] = -numpy.inf

    seqlets = _iterative_extract_seqlets(X_sum=X_sum, window_size=window_size,
                                         flank=flank, suppress=suppress)

    # Find the weakest transformed threshold used across all tasks
    weak_thresh = min(min(transformed_pos_threshold,
                          abs(transformed_neg_threshold)) - 0.0001,
                      weak_threshold_for_counting_sign)

    threshold = distribution[int(weak_thresh * len(distribution))]

    seqlets_ = []
    for example_id, start, end in seqlets:
        attr_flank = int(0.5 * ((end-start) - window_size))
        attr_start, attr_end = start + attr_flank, end - attr_flank
        attr = X_attr[example_id, attr_start:attr_end].sum(axis=-1)

        seqlet = example_id, start, end, attr
        seqlets_.append(seqlet)

    names = 'example_idx', 'start', 'end', 'attribution'
    return pandas.DataFrame(seqlets_, columns=names)


###


@numba.njit
def _recursive_seqlets(X, threshold=0.01, min_seqlet_len=4, max_seqlet_len=25,
                       additional_flanks=0, n_bins=1000):
    """An internal function implementing the recursive seqlet algorithm.

    This algorithm has four steps.

    (1) Convert attribution scores into integer bins and calculate a histogram
    (2) Convert these histograms into null distributions across lengths
    (3) Use the null distributions to calculate p-values for each possible length
    (4) Decode this matrix of p-values to find the longest seqlets
    """

    n, l = X.shape
    m = n*l

    ###
    # Step 1: Calculate a histogram of binned scores
    ###

    xmax, xmin = X.max(), X.min()
    bin_width = (xmax - xmin) / (n_bins - 1)

    f = numpy.zeros(n_bins, dtype=numpy.float64)

    for i in range(n):
        for j in range(l):
            x_bin = math.floor((X[i, j] - xmin) / bin_width)
            f[x_bin] += 1

    f = f / m

    ###
    # Step 2: Calculate null distributions across lengths
    ###

    scores = numpy.zeros((max_seqlet_len+1, n_bins*max_seqlet_len),
                         dtype=numpy.float64)
    scores[1, :n_bins] = f

    rcdfs = numpy.zeros_like(scores)
    rcdfs[:, 0] = 1.0

    for seqlet_len in range(2, max_seqlet_len+1):
        for i in range(n_bins * (seqlet_len-1)):
            for j in range(n_bins):
                scores[seqlet_len, i+j] += scores[seqlet_len-1, i] * f[j]

        for i in range(1, n_bins * seqlet_len):
            rcdfs[seqlet_len, i] = max(rcdfs[seqlet_len, i-1] - scores[seqlet_len, i], 0)

    ###
    # Step 3: Calculate p-values given these 1-CDFs
    ###

    X_csum = numpy.zeros((n, l+1))
    for i in range(n):
        for j in range(l):
            X_csum[i, j+1] = X_csum[i, j] + X[i, j]

    ###
    # Step 4: Decode p-values into seqlets
    ###

    seqlets = []

    for i in range(n):
        p_value = numpy.ones((max_seqlet_len+1, l), dtype=numpy.float64)
        p_value[:min_seqlet_len] = 0
        p_value[:, -min_seqlet_len:] = 1

        for seqlet_len in range(min_seqlet_len, max_seqlet_len+1):
            for k in range(l-seqlet_len+1):
                x_ = X_csum[i, k+seqlet_len] - X_csum[i, k]
                x_ = math.floor((x_ - xmin*seqlet_len) / bin_width)

                p_value[seqlet_len, k] = max(rcdfs[seqlet_len, x_],
                                              p_value[seqlet_len-1, k])

        # Iteratively identify spans, from longest to shortest, that satisfy the
        # recursive p-value threshold.
        for j in range(max_seqlet_len - min_seqlet_len + 1):
            seqlet_len = max_seqlet_len - j

            while True:
                start = p_value[seqlet_len].argmin()
                p = p_value[seqlet_len, start]
                p_value[seqlet_len, start] = 1

                if p >= threshold:
                    break

                for k in range(1, seqlet_len):
                    if p_value[seqlet_len-k, start+k] >= threshold:
                        break

                else:
                    for end in range(start, min(start+seqlet_len, l-1)):
                        p_value[:, end] = 1

                    end = min(start + seqlet_len + additional_flanks, l-1)
                    start = max(start - additional_flanks, 0)
                    attr = X_csum[i, end] - X_csum[i, start]
                    seqlets.append((i, start, end, attr, p))

    return seqlets


def recursive_seqlets(X, threshold=0.01, min_seqlet_len=4, max_seqlet_len=25,
                      additional_flanks=0, n_bins=1000):
    """A seqlet caller implementing the recursive seqlet algorithm.

    NOTE: Currently only *positive* seqlets will be identified. The easiest way
    to get negative seqlets is to run this on the absolute value of the
    attribution values and then re-extract the attribution sums given the
    boundaries.

    This algorithm identifies spans of high attribution characters, called
    seqlets, using a simple approach derived from the Tomtom/FIMO algorithms.
    First, distributions of attribution sums are created for all potential
    seqlet lengths by discretizing the attribution sum into integers. Then,
    CDFs are calculated for each distribution (or, more specifically, 1-CDFs).
    Finally, p-values are calculated via lookup to these 1-CDFs for all
    potential CDFs, yielding a (n_positions, n_lengths) matrix of p-values.

    This algorithm then identifies seqlets by defining them to have a key
    property: all internal spans of a seqlet must also have been called a
    seqlet. This means that all spans from `min_seqlet_len` to `max_seqlet_len`,
    starting at any position in the seqlet, and fully contained by the borders,
    must have a p-value below the threshold. Functionally, this means finding
    entries where the upper left triangle rooted in it is comprised entirely of
    values below the threshold. Graphically, for a candidate seqlet starting at
    X and ending at Y to be called a seqlet, all the values within the bounds
    (in addition to X) must also have a p-value below the threshold.


                            min_seqlet_len
                             --------
    . . . . . . . | . . . . / . . . . . . . .
    . . . . . . . | . . . / . . . . . . . . .
    . . . . . . . | . . / . . . . . . . . . .
    . . . . . . . | . / . . . . . . . . . . .
    . . . . . . . | / . . . . . . . . . . . .
    . . . . . . . X . . . . . . . . Y . . . .
    . . . . . . . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . . . . . . . .


    The seqlets identified by this approach will usually be much smaller than
    those identified by the TF-MoDISco approach, including sometimes missing
    important characters on the flanks. You can set `additional_flanks` to
    a higher value if you want to include additional positions on either side.
    Importantly, the initial seqlet calls cannot overlap, but these additional
    characters are not considered when making that determination. This means
    that seqlets may appear to overlap when `additional_flanks` is set to a
    higher value.


    Parameters
    ----------
    X: tf.Tensor or numpy.ndarray, shape=(-1, length)
        Attributions for each position in each example. The identity of the
        characters is not relevant for seqlet calling, so this should be the
        "projected" attributions, i.e., the attribution of the observed
        characters.

    threshold: float, optional
        The p-value threshold for calling seqlets. All positions within the
        triangle (as detailed above) must be below this threshold. Default is
        0.01.

    min_seqlet_len: int, optional
        The minimum length that a seqlet must be, and the minimal length of
        span that must be identified as a seqlet in the recursive property.
        Default is 4.

    max_seqlet_len: int, optional
        The maximum length that a seqlet can be. Default is 25.

    additional_flanks: int, optional
        An additional value to subtract from the start, and to add to the end,
        of all called seqlets. Does not affect the called seqlets.

    n_bins: int, optional
        The number of bins to use when estimating the PDFs and CDFs. Default is
        1000.


    Returns
    -------
    seqlets: pandas.DataFrame, shape=(-1, 5)
        A BED-formatted dataframe containing the called seqlets, ranked from
        lowest p-value to higher p-value. The returned p-value is the p-value
        of the (location, length) span and is not influenced by the other
        values within the triangle.
    """

    if isinstance(X, tf.Tensor):
        X = X.numpy()
    elif not isinstance(X, numpy.ndarray):
        raise ValueError("`X` must be either a tf.Tensor or numpy.ndarray.")

    columns = ['example_idx', 'start', 'end', 'attribution', 'p-value']
    seqlets = _recursive_seqlets(X, threshold, min_seqlet_len, max_seqlet_len,
                                 additional_flanks, n_bins)
    seqlets = pandas.DataFrame(seqlets, columns=columns)
    return seqlets.sort_values("p-value").reset_index(drop=True)


# Convenience wrapper for backward compatibility
def extract_seqlets(X_attr, method='tfmodisco', **kwargs):
    """Extract seqlets using the specified method.

    Parameters
    ----------
    X_attr: numpy.ndarray or tf.Tensor
        Attribution values for sequences.

    method: str, optional
        Method to use for seqlet extraction. Either 'tfmodisco' or 'recursive'.
        Default is 'tfmodisco'.

    **kwargs
        Additional keyword arguments passed to the specific seqlet extraction
        function.

    Returns
    -------
    seqlets: pandas.DataFrame
        DataFrame containing seqlet information (example_idx, start, end, etc.)
    """
    if method == 'tfmodisco':
        return tfmodisco_seqlets(X_attr, **kwargs)
    elif method == 'recursive':
        return recursive_seqlets(X_attr, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'tfmodisco' or 'recursive'.")
