from typing import List, Sequence
import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import jax
from jax.scipy.special import erf


def split_into_batch(
    dataset: np.ndarray, batch_size: int, axis: int | None = None
) -> List[np.ndarray]:
    """Split an array into batches along a given axis.

    Parameters
    ----------
    dataset : np.ndarray
        Input array to split.
    batch_size : int
        Desired batch size along ``axis``.
    axis : int, optional
        Axis along which to split; defaults to 0.

    Returns
    -------
    list[np.ndarray]
        List of array chunks.
    """
    if axis is None:
        axis = 0
    if dataset.shape[axis] < batch_size:
        # if the dataset is smaller than the batch size, just use the whole dataset as one batch
        return [dataset]
    else:
        # split the dataset into batches
        return np.array_split(dataset, dataset.shape[axis] // batch_size, axis=axis)


@jax.jit
def compute_r_percentiles(
    key: jax.Array,
    mus: jnp.ndarray,
    vars: jnp.ndarray,
    nsamples: int = 1_000_000,
    percentiles: jnp.ndarray = jnp.array([50, 5.0, 95.0]),
) -> jnp.ndarray:
    """Sample from Gaussian field and return vector norm percentiles."""
    samples = jnp.linalg.norm(
        jax.random.normal(key, mus.shape + (nsamples,)) * jnp.sqrt(vars)[..., None]
        + mus[..., None],
        axis=0,
    )  # (  N, nsamples)
    percs = jnp.percentile(samples, percentiles, axis=-1)  # (   N)
    return percs


def compute_stationary_distribution(Q: np.ndarray) -> np.ndarray:
    """
    Compute the stationary distribution π for a column-normalized generator matrix Q.

    The stationary distribution satisfies Q @ π = 0 and sum(π) = 1.

    Parameters
    ----------
    Q : np.ndarray
        Generator matrix of size (n, n).

    Returns
    -------
    np.ndarray
        Stationary distribution π of size (n,).
    """
    # Ensure Q is a NumPy array
    Q = jnp.asarray(Q)

    # Compute eigenvalues and eigenvectors of Q
    evals, evecs = jnp.linalg.eig(Q)

    # Find the eigenvector corresponding to the eigenvalue closest to zero
    idx = jnp.argmin(jnp.abs(evals))
    pi = jnp.real(evecs[:, idx]).astype(jnp.float64)

    # Normalize to ensure sum(π) = 1
    return pi / jnp.sum(pi)


@jax.jit
def find_closest(t: float, array: jnp.ndarray, point_times: jnp.ndarray):
    """Return the array slice closest in time to ``t``.

    Parameters
    ----------
    t : float
        Target time.
    array : jnp.ndarray
        Array with leading time dimension matching ``point_times``.
    point_times : jnp.ndarray
        Time grid.

    Returns
    -------
    jnp.ndarray
        Values at the closest time index.
    """

    # Find the closest point time to t
    closest_index = jnp.argmin(jnp.abs(point_times - t))

    # Get the corresponding theta values for all rows
    out = array[:, closest_index]

    return out


@jax.jit
def find_previous(t, array, point_times):
    """Return array slice just before time ``t`` along leading axis."""
    # Compute a mask for valid indices
    mask = point_times <= t
    # Use jnp.argmax to find the last valid index (argmax returns the first True from the right)
    previous_index = jnp.argmax(mask[::-1])  # Reverse the mask
    previous_index = (
        mask.shape[0] - 1 - previous_index
    )  # Convert back to the original index

    # Get the corresponding values and time
    out = array[previous_index]

    out_time = point_times[previous_index]

    return out, out_time


@jax.jit
def find_next(t, array, point_times):
    """Return array slice just after time ``t`` along leading axis."""

    # Compute a mask for valid indices
    mask = jnp.concatenate([point_times[:-1] >= t, jnp.array([True])])
    # Use jnp.argmax to find the first valid index
    next_index = jnp.argmax(mask)
    # Get the corresponding values and time
    out = array[next_index]

    out_time = point_times[next_index]
    return out, out_time


def nanpad(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Right-pad 1D arrays with NaNs to a common length."""
    max_len = max(len(arr) for arr in arrays)
    padded_arrays = [
        np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan) for arr in arrays
    ]
    return np.array(padded_arrays)


def Proximal_MS2_kernel(t: float, t1: float, t2: float, Imax: float):
    """Compute piecewise-linear proximal MS2 kernel at time ``t``."""
    rise_term = jnp.heaviside(t, 0) * jnp.heaviside(t1 - t, 1) * t * Imax / t1
    plateau_term = jnp.heaviside(t - t1, 0) * jnp.heaviside(t1 + t2 - t, 1) * Imax
    return rise_term + plateau_term


def Fit_gaussian(
    data: np.ndarray, nbins: int = 200, exclude_below: int = 0, plot: bool = False
):
    """Fit a Gaussian to histogrammed data via unweighted chi-squared minimization."""
    Imin, Imax = np.nanpercentile(data, [0.1, 95])
    bins = np.linspace(Imin, Imax, nbins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    bin_counts = np.histogram(data, bins=bins)[0]

    def gaussian(x, mu, sigma):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - mu) / sigma) ** 2
        )

    def chi2(params):
        mu, sigma, N = params
        # Calculate the expected counts
        expected_counts = N * gaussian(bin_centers, mu, sigma) * bin_width
        # Calculate the chi-squared statistic
        exclude_mask = bin_counts < exclude_below
        chi2_stat = np.sum(((bin_counts - expected_counts) ** 2)[~exclude_mask])
        return chi2_stat

    out = minimize(
        chi2,
        x0=[np.nanmean(data), np.nanstd(data), np.sum(~np.isnan(data))],
        bounds=[(Imin, Imax), (0.01, None), (0.01, None)],
    )
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(bin_centers, bin_counts, label="Observed Counts")
        ax.plot(
            bin_centers,
            out.x[2] * gaussian(bin_centers, out.x[0], out.x[1]) * bin_width,
            label=f"Fitted Gaussian\nmu={out.x[0]:.2f}\nsigma={out.x[1]:.2f}\nN={out.x[2]:.2f}",
        )
        ax.legend()
        ax.set(xlabel="Signal Intensity", ylabel="Counts")
    return out


@jax.jit
def smooth_piecewise_constant_batch(centers, values, delta, tau, eval_point):
    """Smooth batched piecewise-constant signals with Gaussian-like kernels.

    Parameters
    ----------
    centers : jnp.ndarray
        Bin centers, shape ``(N,)``.
    values : jnp.ndarray
        Piecewise-constant values per batch, shape ``(B, N)``.
    delta : float
        Bin width.
    tau : float
        Smoothing scale.
    eval_point : jnp.ndarray
        Evaluation points, shape ``(M,)``.

    Returns
    -------
    jnp.ndarray
        Smoothed values, shape ``(B, M)``.
    """
    sqrt2tau = jnp.sqrt(2.0) * tau

    # centers: (N,)
    # values: (B, N)
    # eval_points: (M,)

    # Expand dimensions to broadcast:
    # eval_points -> (1, M, 1)
    # centers -> (1, 1, N)
    centers = centers[None, :]  # (B, 1, N)
    values = values[:, :]  # (B, 1, N)

    left = (eval_point - centers - delta / 2) / sqrt2tau  # (B, M, N)
    right = (eval_point - centers + delta / 2) / sqrt2tau  # (B, M, N)

    contributions = 0.5 * (erf(right) - erf(left)) * values  # (B, M, N)

    smoothed = jnp.sum(contributions, axis=-1)  # sum over N -> (B, M)

    return smoothed
