"""Posterior-derived geometry utilities and anomaly detection helpers."""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.scipy.special import logsumexp
from typing import Optional
from tqdm import tqdm
from . import GPR_utils
from .GPR import GPR


def gauss_legendre_nodes_weights(n: int):
    """Return Gauss-Legendre nodes and weights as JAX arrays."""
    x, w = np.polynomial.legendre.leggauss(n)
    return jnp.asarray(x), jnp.asarray(w)


def make_radial_quadrature(nr: int):
    """Gauss-Legendre quadrature on the unit-radius coordinate ``s in [0, 1]``."""
    x, w = gauss_legendre_nodes_weights(nr)
    s = 0.5 * (x + 1.0)
    ws = 0.5 * w
    return s, ws


def make_sphere_quadrature(nz: int, nphi: int):
    """Build a quadrature rule on the unit sphere ``S^2``."""
    z, wz = gauss_legendre_nodes_weights(nz)
    phi = jnp.linspace(0.0, 2.0 * jnp.pi, nphi, endpoint=False)
    wphi = (2.0 * jnp.pi) / nphi

    zz, pp = jnp.meshgrid(z, phi, indexing="ij")
    rho_xy = jnp.sqrt(jnp.clip(1.0 - zz**2, 0.0, 1.0))

    ux = rho_xy * jnp.cos(pp)
    uy = rho_xy * jnp.sin(pp)
    uz = zz

    u = jnp.stack([ux, uy, uz], axis=-1).reshape(-1, 3)
    w_ang = jnp.broadcast_to(wz[:, None] * wphi, zz.shape).reshape(-1)
    return u, w_ang


def make_ball_quadrature(nr: int = 32, nz: int = 24, nphi: int = 48):
    """Precompute quadrature points and log-weights on the unit ball."""
    s, ws = make_radial_quadrature(nr)
    u, w_ang = make_sphere_quadrature(nz, nphi)

    q = s[:, None, None] * u[None, :, :]
    q = q.reshape(-1, 3)

    tiny = jnp.asarray(1e-300, dtype=q.dtype)
    logw_base = (
        jnp.log(ws)[:, None]
        + jnp.log(w_ang)[None, :]
        + 2.0 * jnp.log(jnp.maximum(s, tiny))[:, None]
    ).reshape(-1)

    return q, logw_base


def _pad_to_multiple(
    x: jnp.ndarray, multiple: int, axis: int = 0, pad_value: float = 0.0
):
    """Pad an array so a given axis is divisible by ``multiple``."""
    n = x.shape[axis]
    n_pad = (-n) % multiple
    if n_pad == 0:
        return x, n, 0

    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, n_pad)
    x_pad = jnp.pad(x, pad_width, constant_values=pad_value)
    return x_pad, n, n_pad


@partial(jax.jit, static_argnames=("chunk_size",))
def log_ball_prob_bt_chunked(
    r: float,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    q: jnp.ndarray,
    logw_base: jnp.ndarray,
    chunk_size: int = 32,
) -> jnp.ndarray:
    """Approximate ``log P(||X|| <= r)`` for diagonal Gaussians via ball quadrature.

    Parameters
    ----------
    r : float
        Sphere radius.
    mu : jnp.ndarray
        Means with shape ``(B, T, 3)``.
    sigma : jnp.ndarray
        Standard deviations with shape ``(B, T, 3)``.
    q : jnp.ndarray
        Unit-ball quadrature points with shape ``(Q, 3)``.
    logw_base : jnp.ndarray
        Corresponding log quadrature weights excluding the ``r^3`` prefactor.
    chunk_size : int, optional
        Number of flattened ``(B, T)`` entries processed per scan step.

    Returns
    -------
    jnp.ndarray
        Log probabilities with shape ``(B, T)``.
    """
    dtype = mu.dtype
    B, T, D = mu.shape
    if D != 3:
        raise ValueError(f"Expected last dimension to be 3, got {D}.")

    r = jnp.asarray(r, dtype=dtype)
    tiny = jnp.asarray(1e-300, dtype=dtype)
    r_safe = jnp.maximum(r, tiny)

    mu_flat = mu.reshape(-1, 3)
    sigma_safe = jnp.maximum(sigma.reshape(-1, 3), tiny)
    n_flat = mu_flat.shape[0]

    mu_pad, n_orig, _ = _pad_to_multiple(mu_flat, chunk_size, axis=0, pad_value=0.0)
    sigma_pad, _, _ = _pad_to_multiple(sigma_safe, chunk_size, axis=0, pad_value=1.0)

    n_chunks = mu_pad.shape[0] // chunk_size
    mu_chunks = mu_pad.reshape(n_chunks, chunk_size, 3)
    sigma_chunks = sigma_pad.reshape(n_chunks, chunk_size, 3)

    xq = r * q
    logw = logw_base + 3.0 * jnp.log(r_safe)
    log2pi = jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=dtype))

    def one_chunk(_, xs):
        mu_c, sigma_c = xs
        z = (xq[None, :, :] - mu_c[:, None, :]) / sigma_c[:, None, :]
        quad = jnp.sum(z * z, axis=-1)
        logdet = 2.0 * jnp.sum(jnp.log(sigma_c), axis=-1)
        logpdf = -0.5 * (quad + 3.0 * log2pi + logdet[:, None])
        log_terms = logpdf + logw[None, :]
        logf_c = logsumexp(log_terms, axis=-1)
        return None, logf_c

    _, logf_chunks = jax.lax.scan(one_chunk, None, (mu_chunks, sigma_chunks))
    logf_flat = logf_chunks.reshape(-1)[:n_orig]
    logf_flat = jnp.where(r > 0, logf_flat, -jnp.inf)
    logf_flat = jnp.minimum(logf_flat, jnp.asarray(0.0, dtype=dtype))

    return logf_flat.reshape(B, T)


def log_prob_in_sphere_quadrature(
    mu: jnp.ndarray,
    var: jnp.ndarray,
    R: float,
    nr: int = 32,
    nz: int = 24,
    nphi: int = 48,
    chunk_size: int = 32,
    q: Optional[jnp.ndarray] = None,
    logw_base: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Deterministic log probability of a diagonal Gaussian lying in a sphere.

    Parameters
    ----------
    mu : jnp.ndarray
        Means, shape ``(N, n, 3)``.
    var : jnp.ndarray
        Diagonal variances, shape ``(N, n, 3)``.
    R : float
        Sphere radius.
    nr, nz, nphi : int, optional
        Quadrature orders for the radial and angular rules.
    chunk_size : int, optional
        Number of flattened ``(track, time)`` entries to process per scan step.
    q, logw_base : jnp.ndarray, optional
        Precomputed quadrature points and base log weights from
        :func:`make_ball_quadrature`.

    Returns
    -------
    jnp.ndarray
        Log probabilities with shape ``(n, N)``, matching :func:`prob_in_sphere`.
    """
    mu = jnp.asarray(mu)
    var = jnp.asarray(var)
    sigma = jnp.sqrt(jnp.maximum(var, 0.0))

    if q is None or logw_base is None:
        q, logw_base = make_ball_quadrature(nr=nr, nz=nz, nphi=nphi)

    logf = log_ball_prob_bt_chunked(
        R,
        mu,
        sigma,
        q,
        logw_base,
        chunk_size=chunk_size,
    )
    return logf.transpose((1, 0))


def prob_in_sphere_quadrature(
    mu: jnp.ndarray,
    var: jnp.ndarray,
    R: float,
    nr: int = 32,
    nz: int = 24,
    nphi: int = 48,
    chunk_size: int = 32,
    q: Optional[jnp.ndarray] = None,
    logw_base: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Deterministic quadrature approximation for ``P(||X|| <= R)``.

    This is an alternative to :func:`prob_in_sphere` that avoids the Monte Carlo
    zeros responsible for severe underflow when the probability is very small.
    """
    logp = log_prob_in_sphere_quadrature(
        mu,
        var,
        R,
        nr=nr,
        nz=nz,
        nphi=nphi,
        chunk_size=chunk_size,
        q=q,
        logw_base=logw_base,
    )
    logtiny = np.log(np.finfo(np.float64).tiny)
    return jnp.exp(jnp.maximum(logp, logtiny))


@partial(jax.jit, static_argnames=["num_samples"])
def prob_in_sphere_batch(
    mu: jnp.ndarray, sigma: jnp.ndarray, R: float, num_samples=10000
) -> jnp.ndarray:
    """Monte Carlo probability a 3D diagonal Gaussian lies inside a sphere.

    Parameters
    ----------
    mu : jnp.ndarray
        Means, shape ``(n, 3, N)``.
    sigma : jnp.ndarray
        Variances (diagonal covariances), shape ``(n, 3, N)``.
    R : float
        Sphere radius.
    num_samples : int, optional
        Number of Monte Carlo samples per Gaussian.

    Returns
    -------
    jnp.ndarray
        Probabilities, shape ``(n, N)``.
    """
    n, _, N = mu.shape
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, n * N)
    keys = keys.reshape((n, N, 2))

    # Reparameterized sampling of standard normal
    @jax.jit
    def sample_and_estimate(mu_ij, sigma_ij, key):
        eps = jax.random.normal(key, (num_samples, 3))
        samples = mu_ij + jnp.sqrt(sigma_ij) * eps
        within = jnp.linalg.norm(samples, axis=-1) <= R
        return jnp.mean(within)

    v_estimate = jax.vmap(
        jax.vmap(sample_and_estimate, in_axes=(0, 0, 0)),  # over N
        in_axes=(0, 0, 0),  # over n
    )
    return v_estimate(mu.transpose(0, 2, 1), sigma.transpose(0, 2, 1), keys)


def prob_in_sphere(
    mu: jnp.ndarray,
    var: jnp.ndarray,
    R: float,
    num_samples=10000,
    batch_size=10,
    verbose=True,
) -> jnp.ndarray:
    """Batch wrapper for probability mass of 3D diagonal Gaussians inside a sphere.

    Parameters
    ----------
    mu : jnp.ndarray
        Means, shape ``(N, n, 3)``.
    var : jnp.ndarray
        Variances (diagonal covariances), shape ``(N, n, 3)``.
    R : float
        Sphere radius.
    num_samples : int, optional
        Number of Monte Carlo samples per Gaussian.
    batch_size : int, optional
        Batch size for processing along the first axis.
    verbose : bool, optional
        Show progress bar when True.

    Returns
    -------
    jnp.ndarray
        Probabilities, shape ``(n, N)``.
    """
    means, vars = mu.transpose((1, 2, 0)), var.transpose((1, 2, 0))
    results = np.zeros((means.shape[0], means.shape[2]))

    # Create batches
    batches = np.array_split(np.arange(means.shape[0]), len(means) // batch_size)
    if verbose:
        batches = tqdm(batches, desc="Computing probabilities", total=len(batches))
    else:
        batches = iter(batches)

    # Process each batch
    for i in batches:
        inds = i
        p_means_batch = means[inds]
        p_vars_batch = vars[inds]

        probs = prob_in_sphere_batch(
            p_means_batch, p_vars_batch, R, num_samples=num_samples
        )
        results[inds] = np.array(probs)
    return results


@partial(jax.jit, static_argnames=("num_samples", "percentiles"))
def gaussian_radius_percentiles(
    mu: jnp.ndarray,
    sigma2: jnp.ndarray,
    num_samples: int = 10000,
    percentiles=(5, 50, 95),
    seed: int = 0,
) -> jnp.ndarray:
    """Monte Carlo percentiles of the radius for 3D diagonal Gaussians.

    Parameters
    ----------
    mu : jnp.ndarray
        Means, shape ``(N, T, d)``.
    sigma2 : jnp.ndarray
        Variances (diagonal), shape ``(N, T, d)``.
    num_samples : int, optional
        Number of Monte Carlo samples.
    percentiles : tuple, optional
        Percentiles to compute (0-100 range).
    seed : int, optional
        PRNG seed.

    Returns
    -------
    jnp.ndarray
        Radius percentiles, shape ``(N, T, len(percentiles))``.
    """
    key = jax.random.PRNGKey(seed)
    N, T, d = mu.shape

    # Sample standard normal: shape (num_samples, N, T, d)
    std_normal = jax.random.normal(key, shape=(int(num_samples), N, T, d))

    # Reparameterize: x = mu + sqrt(sigma2) * eps
    stddev = jnp.sqrt(sigma2)  # (N, T, d)
    samples = mu[None, ...] + stddev[None, ...] * std_normal  # (S, N, T, d)

    # Compute radius: sqrt(x^2 + y^2 + z^2)
    radii = jnp.linalg.norm(samples, axis=-1)  # (S, N, T)

    # Compute percentiles along sample axis
    radii_percentiles = jnp.nanpercentile(
        radii, q=jnp.array(percentiles), axis=0
    )  # (len(percentiles), N, T)

    # Reorder to (N, T, len(percentiles))
    return jnp.moveaxis(radii_percentiles, 0, -1)


def batched_radius_percentile_comp(
    mu: jnp.ndarray,
    vars: jnp.ndarray,
    batch_size: int = 10,
    verbose: bool = True,
    percentiles=(5, 50, 95),
    num_samples: int = 10000,
) -> jnp.ndarray:
    """Compute percentiles of radius vectors in batches.

    Parameters
    ----------
    mu : jnp.ndarray
        Means, shape ``(N, T, d)``.
    vars : jnp.ndarray
        Variances, shape ``(N, T, d)``.
    batch_size : int, optional
        Batch size for processing.
    verbose : bool, optional
        Show progress bar when True.
    percentiles : tuple, optional
        Percentiles to compute (0-100 range).

    Returns
    -------
    jnp.ndarray
        Percentiles (5th, 50th, 95th), shape ``(N, T, 3)``.
    """
    mu_batches = np.array_split(mu, len(mu) // batch_size + 1)
    var_batches = np.array_split(vars, len(vars) // batch_size + 1)
    if verbose:
        mu_batches = tqdm(
            mu_batches, desc="Computing percentiles", total=len(mu_batches)
        )
    else:
        mu_batches = iter(mu_batches)

    percentiles = np.zeros((len(mu), mu.shape[1], 3))
    count = 0
    for mu_batch, var_batch in zip(mu_batches, var_batches):
        percentiles[count : count + len(mu_batch)] = gaussian_radius_percentiles(
            mu_batch, var_batch
        )
        count += len(mu_batch)
    return percentiles


def detect_anomalies(
    x: jnp.ndarray,
    pred_regressor: GPR,
    track: jnp.ndarray,
    min_size: int = 1,
    max_size: int = 100,
    verbose: bool = True,
    threshold: float = 0.05,
):
    """Detect anomalies in a multivariate time series using a GP regressor.

    Parameters
    ----------
    x : jnp.ndarray
        Model parameters and noise, shape ``(n_params + n_dims,)``.
    pred_regressor : GPR
        Gaussian process regressor used for predictions.
    track : jnp.ndarray
        Time series data, shape ``(n_timepoints, n_dims)``.
    min_size : int, optional
        Minimum block size to test.
    max_size : int, optional
        Maximum block size to test.
    verbose : bool, optional
        Show progress bar when True.
    threshold : float, optional
        Significance threshold (Bonferroni corrected inside the routine).

    Returns
    -------
    significant : jnp.ndarray
        Binary flags per block size and time point, shape ``(n_block_sizes, n_timepoints)``.
    statistic : jnp.ndarray
        Test statistic per block size and time point, same shape as ``significant``.
    """

    ndims = track.shape[-1]
    block_sizes = jnp.unique(
        jnp.logspace(jnp.log10(min_size), jnp.log10(max_size), 10).astype(int)
    )

    times = pred_regressor.ts

    tracklength = len(track)
    all_inds = jnp.arange(tracklength)

    noise = jnp.abs(x[-ndims:])  # last dim is noise
    params = jnp.abs(x[:-ndims])  # all but last dim are parameters
    final_params = jnp.concatenate((jnp.tile(params, ndims), noise))
    theta = final_params[:-ndims]  # all but last dim are parameters

    # Function to run anomaly detection for a specific block size
    def run_block(block_size):
        start_inds = jnp.arange(0, tracklength - block_size + 1)
        in_block_inds = jnp.arange(block_size)[None, :] + start_inds[:, None]
        out_of_block_inds = jax.vmap(
            lambda a, b: jnp.setdiff1d(
                a, b, assume_unique=True, size=tracklength - block_size
            ),
            in_axes=(None, 0),
        )(all_inds, in_block_inds)

        arr_times = jnp.array(times)

        @jax.jit
        def compute_pval(ind):
            in_times = arr_times[in_block_inds[ind]]  # (nblocks,block_size)
            out_times = arr_times[out_of_block_inds[ind]]  # (nblocks,block_size)

            in_block_data = track[in_block_inds[ind]]  # (nblocks,block_size,ndims)
            out_block_data = track[
                out_of_block_inds[ind]
            ]  # (nblocks,tracklength-block_size,ndims)

            K_00 = pred_regressor.covbuilder(theta, in_times, in_times)
            K_01 = pred_regressor.covbuilder(theta, in_times, out_times)
            K_11 = pred_regressor.covbuilder(theta, out_times, out_times)

            prediction_covmat = utils.get_mat_for_cholesky(
                in_block_data, K_00, jnp.array(noise)
            )[2]

            mean_pred = utils.predict_mean_single(
                K_01, out_block_data, K_11, jnp.array(noise)
            )
            cov_pred = utils.predict_cov_single(
                K_01, K_11, out_block_data, prediction_covmat, jnp.array(noise)
            )

            residual = (in_block_data - mean_pred).T  # (ndims, block_size)

            Ls = jnp.linalg.cholesky(
                cov_pred.transpose(2, 0, 1)
            )  # (ndims, block_size, block_size)
            whitened_res = jax.scipy.linalg.solve_triangular(
                Ls, residual, lower=True
            ).T  # (block_size, ndims)
            statistic = jnp.nansum(whitened_res**2)
            ndeg = jnp.sum(~jnp.isnan(whitened_res))
            tail = jax.scipy.stats.chi2.cdf(statistic, ndeg)
            pval = jnp.min(jnp.array([tail, 1 - tail])) * 2  # two-sided
            return statistic, pval

        vmap_compute_pval = jax.vmap(compute_pval)
        batch_size = 200
        ind_batches = np.array_split(
            jnp.arange(len(start_inds)), len(start_inds) // batch_size + 1
        )
        pval_res = np.zeros(len(start_inds))
        statistic_res = np.zeros(len(start_inds))
        for batch_inds in ind_batches:
            statistic_res[batch_inds], pval_res[batch_inds] = vmap_compute_pval(
                batch_inds
            )
        return statistic_res, pval_res

    results = []
    if verbose:
        iterator = tqdm(block_sizes)
    else:
        iterator = block_sizes
    for block_size in iterator:
        statistic_res, pval_res = run_block(block_size)
        results.append((statistic_res, pval_res))

    n_tests = np.sum([len(res) for res in results])
    threshold = threshold / n_tests  # Bonferroni corrected threshold
    significant = jnp.zeros((len(results), len(times)))
    statistic = jnp.zeros((len(results), len(times)))
    for block_ind, block_size, (statistic_res, pval_res) in zip(
        jnp.arange(len(block_sizes)), block_sizes, results
    ):
        start_inds = jnp.arange(0, tracklength - block_size + 1)
        sig_inds = start_inds[pval_res < threshold]
        for ind in sig_inds:
            significant = significant.at[block_ind, ind : ind + block_size].set(1)
            statistic = statistic.at[block_ind, ind : ind + block_size].set(
                statistic_res[ind]
            )
    return significant, statistic, block_sizes
