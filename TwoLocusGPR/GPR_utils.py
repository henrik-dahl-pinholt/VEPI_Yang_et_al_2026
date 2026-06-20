"""Utility functions for GP sampling, prediction, and MSD estimation (JAX)."""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from tqdm.auto import tqdm

# import stuff to do type hints


@partial(jax.jit, static_argnames=["num_samples"])
def sample_gauss(
    num_samples: int, mu: jnp.ndarray, covmat: jnp.ndarray, seed: int
) -> jnp.ndarray:
    """Batch sample num_samples from a gaussian with mean mu and covariance covmat.

    Parameters
    ----------
    num_samples : int
        Integer number of samples to draw per batch entry
    mu : jnp.ndarray, shape (ndat,ndim)
        Mean of the gaussian
    covmat : jnp.ndarray, shape (ndat,ndat)
        Covariance matrix of the gaussian
    seed : int
        Random seed for reproducibility

    Returns
    -------
    jnp.ndarray, shape (num_samples, ndat, ndim)
        Samples drawn from the specified Gaussian distribution
    """
    # Generate gaussian samples
    key = jax.random.PRNGKey(seed)
    gaussian_rdws = jax.random.normal(
        key, shape=(num_samples, *mu.shape)
    )  # (num_samples,ndat,ndim)

    # Perform cholesky decomposition of the covariance matrix (with jitter for numerical stability)
    cholesky = jnp.linalg.cholesky(
        covmat + jnp.eye(covmat.shape[-1])[None, :] * 1e-6, upper=True
    )  # (ndim,ndat,ndat)
    samples = jnp.einsum("ijk,kjl->ilk", gaussian_rdws, cholesky) + mu[None, :, :]
    return samples


@jax.jit
def sample_gauss_cmat(mus: jnp.ndarray, covmats: jnp.ndarray, seed: int) -> jnp.ndarray:
    """Batch sample a single sample across a set of gaussians defined by means and covariance matrices.

    Parameters
    ----------
    mus : jnp.ndarray, shape (batch, ndat)
        Means of the gaussians
    covmats : jnp.ndarray, shape (batch, ndat, ndat)
        Covariance matrices of the gaussians
    seed : int
        Random seed for reproducibility
    Returns
    -------
    jnp.ndarray
        Samples drawn from the specified Gaussian distributions
    """
    # Generate gaussian samples
    key = jax.random.PRNGKey(seed)
    gaussian_rdws = jax.random.normal(key, shape=mus.shape)

    # Perform cholesky decomposition of the covariance matrices and make sample
    cholesky = jnp.linalg.cholesky(
        covmats + jnp.eye(covmats.shape[-1])[None, :] * 1e-6
    )  # (batch,ndat,ndat)
    samples = mus + jnp.matvec(cholesky, gaussian_rdws)  # (batch,ndat)
    return samples


@jax.jit
def get_mat_for_cholesky(
    data: jnp.ndarray, covmat: jnp.ndarray, noise: jnp.ndarray
) -> tuple:
    """Add diagonal noise and filter out nans by setting their noise to a large value in the matrix for cholesky decomposition in GPR.

    Parameters
    ----------
    data : jnp.ndarray, shape (ndat, ndim)
        Input data with potential NaN entries
    covmat : jnp.ndarray, shape (ndim, ndat, ndat)
        Prior covariance matrix for the Gaussian Process
    noise : jnp.ndarray, shape (ndim,)
        Per-dimension observational noise std; broadcast across time.

    Returns
    -------
    nan_entries : jnp.ndarray, shape (ndat, ndim)
        Boolean array indicating positions of NaN entries in the data
    masked_data : jnp.ndarray, shape (ndat, ndim)
        Data with NaN entries replaced by zeros
    mats_for_cholesky : jnp.ndarray, shape (ndim, ndat, ndat)
        Covariance matrices adjusted for noise and NaN entries, ready for Cholesky
    """
    # handle nans by setting noise to inf
    nan_entries = jnp.isnan(data)  # (ndat,ndim)
    noise_vals = 2 * jnp.where(nan_entries, noise * 0 + 1e6, noise) ** 2  # (ndat,ndim)

    # add noise and remove nans
    diag_noise = jax.vmap(jnp.diag)((noise_vals).T)  # (ndim,ndat,ndat)
    masked_data = jnp.where(nan_entries, 0, data)  # (ndat,ndim)
    mats_for_cholesky = covmat + diag_noise  # (ndim,ndat,ndat)

    return nan_entries, masked_data, mats_for_cholesky


@jax.jit
def predict_mean_single(
    prediction_kernel: jnp.ndarray,
    data: jnp.ndarray,
    covmat: jnp.ndarray,
    noise: jnp.ndarray,
) -> jnp.ndarray:
    """Predict the mean of the posterior Gaussian Process at new points for a single track.

    Parameters
    ----------
    prediction_kernel : jnp.ndarray, shape (ndim,npred, ndat)
        Kernel matrix between prediction points and training data points
    data : jnp.ndarray, shape (ndat, ndim)
        Observed data points
    covmat : jnp.ndarray, shape (ndim, ndat, ndat)
        Covariance matrix of the Gaussian Process prior
    noise : jnp.ndarray, shape (ndim)
        Noise levels associated with the data

    Returns
    -------
    jnp.ndarray, shape (npred, ndim)
        Predicted mean of the posterior Gaussian Process at new points
    """
    # Prepare matrices for cholesky decomposition
    # Handle nans by setting their noise to a large value
    nan_entries, masked_data, mats_for_cholesky = get_mat_for_cholesky(
        data, covmat, noise
    )
    # Cholesky decomposition
    cholesky_pred = jnp.linalg.cholesky(mats_for_cholesky)  # (ndim,ndat,ndat)

    # Solve for alpha
    z = jax.scipy.linalg.solve_triangular(
        cholesky_pred, masked_data.T, lower=True
    )  # (ndim,ndat)
    x = jax.scipy.linalg.solve_triangular(
        cholesky_pred.swapaxes(-1, -2), z, lower=False
    )  # (ndim,ndat)

    # Predict mean
    mean_pred = jnp.matvec(prediction_kernel, x).T  # (npred,ndim)

    return mean_pred


@jax.jit
def predict_cov_single(
    prediction_kernel: jnp.ndarray,
    covmat: jnp.ndarray,
    data: jnp.ndarray,
    prediction_covmat: jnp.ndarray,
    noise: jnp.ndarray,
) -> jnp.ndarray:
    """Predict the covariance matrix for a dataset
    Parameters
    ----------
    prediction_kernel : jnp.ndarray, shape (ndim,npred, ndat)
        Kernel matrix between prediction points and training data points
    covmat : jnp.ndarray, shape (ndim, ndat, ndat)
        Covariance matrix of the Gaussian Process prior
    data : jnp.ndarray, shape (ndat, ndim)
        Observed data points
    prediction_covmat : jnp.ndarray, shape (ndim,npred,npred)
        Prior covariance matrix at the prediction points
    noise : jnp.ndarray, shape (ndim,)
        Per-dimension observational noise std

    Returns
    -------
    jnp.ndarray, shape (npred,ndim,ndim)
        Predicted covariance matrix of the posterior Gaussian Process at new points

    """
    # Prepare matrices for cholesky decomposition
    # Handle nans by setting their noise to a large value
    nan_entries, masked_data, mats_for_cholesky = get_mat_for_cholesky(
        data, covmat, noise
    )

    # Cholesky decomposition
    cholesky_pred = jnp.linalg.cholesky(mats_for_cholesky)  # (ndim,ndat,ndat)

    # Solve for V
    V = jax.scipy.linalg.solve_triangular(
        cholesky_pred, prediction_kernel.swapaxes(-1, -2), lower=True
    )

    # Compute predicted covariance
    cov_pred = prediction_covmat - V.swapaxes(-1, -2) @ V  # (ndim,npred,npred)
    return cov_pred.transpose(1, 2, 0)  # (npred,ndim,ndim)


@jax.jit
def predict_cov_single_diag(
    prediction_kernel: jnp.ndarray,
    covmat: jnp.ndarray,
    data: jnp.ndarray,
    prediction_covmat: jnp.ndarray,
    noise: jnp.ndarray,
) -> jnp.ndarray:
    """Predict only the diagonal of the covariance matrix for a dataset

    Parameters
    ----------
    prediction_kernel : jnp.ndarray, shape (ndim,npred, ndat)
        Kernel matrix between prediction points and training data points
    covmat : jnp.ndarray, shape (ndim, ndat, ndat)
        Covariance matrix of the Gaussian Process prior
    data : jnp.ndarray, shape (ndat, ndim)
        Observed data points
    prediction_covmat : jnp.ndarray, shape (ndim,npred,npred)
        Prior covariance matrix at the prediction points
    noise : jnp.ndarray, shape (ndim,)
        Per-dimension observational noise std
    Returns
    -------
    jnp.ndarray, shape (npred,ndim,ndim)
        Predicted diagonal of the covariance matrix of the posterior Gaussian Process at new points
    """
    # Prepare matrices for cholesky decomposition
    nan_entries, masked_data, mats_for_cholesky = get_mat_for_cholesky(
        data, covmat, noise
    )

    # Cholesky decomposition
    cholesky_pred = jnp.linalg.cholesky(mats_for_cholesky)  # (ndim,ndat,ndat)

    # Solve for V
    V = jax.scipy.linalg.solve_triangular(
        cholesky_pred, prediction_kernel.swapaxes(-1, -2), lower=True
    )  # (ndim,npred,npred)

    # Compute predicted covariance diagonal
    cov_pred = jax.vmap(jnp.diag)(prediction_covmat) - jnp.sum(
        V**2, axis=-2
    )  # (ndim,npred)
    return cov_pred.swapaxes(-1, -2)  # (npred,ndim,ndim)


# Vectorized versions of the prediction functions
vmap_pred_cov = jax.vmap(predict_cov_single_diag, in_axes=(None, None, 0, None, None))
vmap_pred_cov_full = jax.vmap(predict_cov_single, in_axes=(None, None, 0, None, None))
v_pred = jax.vmap(predict_mean_single, in_axes=(None, 0, None, None))


@jax.jit
def cholesky_inverse(L: jnp.ndarray) -> jnp.ndarray:
    """Compute the inverse of a matrix given its Cholesky decomposition L (lower triangular matrix such that A = L @ L.T)

    Parameters
    ----------
    L : jnp.ndarray, shape (n,n)
        Lower triangular matrix from Cholesky decomposition

    Returns
    -------
    jnp.ndarray
        Inverse of the original matrix A
    """

    # Solve L @ Y = I => Y = L^{-1}
    identity = jnp.eye(L.shape[0], dtype=L.dtype)
    Linv = jax.scipy.linalg.solve_triangular(L, identity, lower=True)

    # A^{-1} = L^{-T} @ L^{-1}
    Ainv = Linv.T @ Linv
    return Ainv


# Vectorized version of cholesky_inverse
vmap_cholesky_inverse = jax.vmap(cholesky_inverse)


def restore_np(
    arr_short: jnp.ndarray, removed_indices: jnp.ndarray, removed_values: jnp.ndarray
) -> jnp.ndarray:
    """Restore an array that had values removed at certain indices.

    Parameters
    ----------
    arr_short : jnp.ndarray, shape (n,)
        Array with values removed
    removed_indices : jnp.ndarray, shape (m,)
        Indices where values were removed
    removed_values : jnp.ndarray, shape (m,)
        Values that were removed

    Returns
    -------
    jnp.ndarray, shape (n + m,)
        Restored array with values reinserted at the specified indices
    """
    # Convert inputs to jax arrays
    idx = jnp.asarray(removed_indices)
    vals = jnp.asarray(removed_values)

    # ensure ascending order so ranks are 0,1,2,...
    order = jnp.argsort(idx)
    idx = idx[order]
    vals = vals[order]

    # each insertion position is original_index - number_of_prior_removals
    insert_pos = idx - jnp.arange(len(idx))

    return jnp.insert(arr_short, insert_pos, vals)


@jax.jit
def single_msds(tseries: jnp.ndarray, lags: jnp.ndarray) -> jnp.ndarray:
    """Compute mean squared displacements for a single time series over given lags.

    Parameters
    ----------
    tseries : jnp.ndarray, shape (n,)
        Time series data
    lags : jnp.ndarray, shape (m,)
        Lags at which to compute the mean squared displacement

    Returns
    -------
    jnp.ndarray, shape (m,)
        Mean squared displacements at the specified lags
    """
    # pad end to avoid wrap-around
    padded_tseries = jnp.concatenate([tseries, jnp.zeros_like(tseries) * jnp.nan])

    # compute msd for each lag
    inds = jnp.arange(len(tseries))
    shifted_arr = padded_tseries[inds[None, :] + lags[:, None]]
    msd = jnp.nanmean((shifted_arr - tseries[None, :]) ** 2, axis=1)
    return msd


# @jax.jit
# def msd_single_dim(X: jnp.ndarray, lags: jnp.ndarray) -> jnp.ndarray:
#     """ Compute mean squared displacements for a single dimension across multiple time series over given lags.

#     Parameters
#     ----------
#     X : jnp.ndarray, shape (n_series, n_timepoints)
#         Time series data for a single dimension across multiple series
#     lags : jnp.ndarray, shape (m,)
#         Lags at which to compute the mean squared displacement

#     Returns
#     -------
#     jnp.ndarray, shape (m,)
#         Mean squared displacements at the specified lags
#     """
#     msds = jax.vmap(single_msds, in_axes=(0, None))(X, lags)
#     nan_counts = jnp.sum(~jnp.isnan(X), axis=1)
#     N_ntotal = jnp.sum(nan_counts)
#     msd = jnp.nansum(msds * nan_counts[:, None], axis=0) / N_ntotal


#     return msd
@jax.jit
def msd_single_dim(X: jnp.ndarray, lags: jnp.ndarray) -> jnp.ndarray:
    n_series, T = X.shape

    padded = jnp.concatenate([X, jnp.full_like(X, jnp.nan)], axis=1)
    inds = jnp.arange(T)
    shifted = padded[:, inds[None, :] + lags[:, None]]  # shape: (n_series, n_lags, T)
    shifted = jnp.swapaxes(shifted, 1, 2)  # (n_series, T, n_lags)

    diffsq = (shifted - X[:, :, None]) ** 2
    return jnp.nanmean(diffsq, axis=(0, 1))


@jax.jit
def msd(X: jnp.ndarray, lags: jnp.ndarray) -> jnp.ndarray:
    """Compute mean squared displacements for multiple dimensions across multiple time series over given lags.

    Parameters
    ----------
    X : jnp.ndarray, shape (n_series, n_timepoints, n_dimensions)
        Time series data for multiple dimensions across multiple series

    lags : jnp.ndarray, shape (m,)
        Lags at which to compute the mean squared displacement

    Returns
    -------
    jnp.ndarray, shape (m, n_dimensions)
        Mean squared displacements at the specified lags for each dimension
    """
    return jax.vmap(msd_single_dim, in_axes=(2, None))(X, lags).T
