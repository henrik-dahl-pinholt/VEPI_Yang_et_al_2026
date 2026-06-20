"""Gaussian process regression utilities built on JAX for MSD-based kernels."""

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm
from scipy.optimize import minimize, root_scalar

from . import GPR_utils


def _as_scope(scope):
    if scope is None:
        return ()
    if isinstance(scope, str):
        scope = (scope,)
    scope = tuple(scope)
    valid = {"dataset", "dim"}
    if any(entry not in valid for entry in scope):
        raise ValueError("Parameter layout scopes must use 'dataset' and/or 'dim'")
    if len(set(scope)) != len(scope):
        raise ValueError("Parameter layout scopes cannot repeat an axis")
    return scope


def _scope_width(scope, n_datasets, dim):
    width = 1
    if "dataset" in scope:
        width *= n_datasets
    if "dim" in scope:
        width *= dim
    return width


def _sanitize_label(label):
    return (
        label.replace("[", "_")
        .replace("]", "")
        .replace("=", "")
        .replace(", ", "_")
        .replace(",", "_")
    )


def _interp_threshold_crossing(x0, x1, y0, y1, threshold):
    """Linearly interpolate the x value where y crosses threshold."""
    if not np.all(np.isfinite([x0, x1, y0, y1, threshold])):
        return np.nan
    if x0 == x1:
        return np.nan
    if y0 == y1:
        if y0 == threshold:
            return min(x0, x1)
        return np.nan
    return x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)


def _find_profile_ci_side(prof_losses, grid, threshold, side="lower"):
    """Find one profile-likelihood CI endpoint from a sampled profile grid."""
    prof_losses = np.asarray(prof_losses, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if prof_losses.shape != grid.shape:
        raise ValueError("Profile losses and grid must have the same shape")
    if len(grid) < 2:
        return np.nan

    center = len(grid) // 2
    if side == "lower":
        intervals = ((ind - 1, ind) for ind in range(center, 0, -1))
    elif side == "upper":
        intervals = ((ind, ind + 1) for ind in range(center, len(grid) - 1))
    else:
        raise ValueError("side must be 'lower' or 'upper'")

    for left_ind, right_ind in intervals:
        x0, x1 = grid[left_ind], grid[right_ind]
        y0, y1 = prof_losses[left_ind], prof_losses[right_ind]
        if not np.all(np.isfinite([x0, x1, y0, y1])):
            continue
        if (y0 - threshold) * (y1 - threshold) <= 0:
            return _interp_threshold_crossing(x0, x1, y0, y1, threshold)
    return np.nan


def _make_restricted(loss, x_full0, fix_idx, fix_val_log, bounds=None, grad=None):
    """Pin one parameter and build reduced objective/grad/HVP.

    Parameters
    ----------
    loss : callable
        Objective in log space.
    x_full0 : np.ndarray
        Reference parameter vector for sizing.
    fix_idx : int
        Index to hold fixed.
    fix_val_log : float
        Fixed value (log scale) for ``fix_idx``.
    bounds : list of tuples or None, optional
        Bounds for the full optimization in natural scale, passed as a list of (lower, upper) pairs for each parameter. If None, no bounds are applied.

    Returns
    -------
    tuple
        ``(free_idx, assemble_full, f_red, g_red, bounds_red)`` where each
        entry operates in the reduced parameter space.
    """
    P = x_full0.size
    free_idx = np.array([i for i in range(P) if i != fix_idx], dtype=int)

    def assemble_full(x_free):
        x_full = np.array(x_full0, copy=True)
        x_full[free_idx] = x_free
        x_full[fix_idx] = fix_val_log
        return x_full

    # Reduced objective: f(x_free) = loss( scatter(x_free, fix=const) )
    def f_red(x_free):
        return float(loss(assemble_full(x_free)))

    def g_red(x_free):
        return np.asarray(grad(assemble_full(x_free)))[free_idx]

    if bounds is not None:
        bounds_red = [bounds[i] for i in range(P) if i != fix_idx]
    else:
        bounds_red = None

    return (
        free_idx,
        assemble_full,
        f_red,
        g_red if grad is not None else None,
        bounds_red,
    )


def _profile_one_param(
    loss,
    x_hat_full,
    fix_idx,
    grid_vals_natural,
    method="Nelder-Mead",
    bounds=None,
    grad=None,
    verbose=True,
):
    """Profile a single parameter over a grid while optimizing the rest.

    Parameters
    ----------
    loss : callable
        Objective in log space.
    x_hat_full : np.ndarray
        Best-fit parameters in log space.
    fix_idx : int
        Index of the parameter to fix during profiling.
    grid_vals_natural : array-like
        Grid of values (natural scale) to scan for the fixed parameter.
    method : str, optional
        SciPy optimizer for the reduced problem.
    bounds : list of tuples or None, optional
        Bounds for the full optimization in natural scale, passed as a list of (lower, upper) pairs for each parameter. If None, no bounds are applied.
    verbose : bool, optional
        Whether to display progress information.
    Returns
    -------
    dict
        Contains natural-scale grid, profiled loss values, and profiled
        parameter vectors in natural scale.
    """
    grid_log = np.log(np.asarray(grid_vals_natural))
    prof_losses = []
    prof_params_full_nat = []

    # warm start: begin from x_hat_full; then reuse previous solution
    x_free_current = np.delete(np.asarray(x_hat_full), fix_idx)

    if verbose:
        pbar = tqdm(total=len(grid_log))
        pbar.set_description(f"Profiling param {fix_idx}")

    def runfit(gv_log, x0):
        free_idx, assemble_full, f_red, g_red, sub_bounds = _make_restricted(
            loss, x_hat_full, fix_idx, gv_log, bounds=bounds, grad=grad
        )
        minimize_kwargs = {}
        if str(method).upper() == "L-BFGS-B" and g_red is not None:
            minimize_kwargs["jac"] = g_red

        res = minimize(
            f_red,
            x0,
            method=method,
            bounds=sub_bounds,
            **minimize_kwargs,
        )
        return res, free_idx

    # do right_side first
    right_val = []
    right_params_full_nat = []
    for gv_log in grid_log[len(grid_log) // 2 :]:
        res, free_idx = runfit(gv_log, x_free_current)
        x_free_current = res.x  # warm start next

        # collect profiled optimum in FULL parameter space (natural)
        x_full_prof_log = np.array(x_hat_full, copy=True)
        x_full_prof_log[free_idx] = res.x
        x_full_prof_log[fix_idx] = gv_log

        # prof_losses.append(float(res.fun))
        right_params_full_nat.append(np.exp(x_full_prof_log))
        right_val.append(float(res.fun))

        if verbose:
            pbar.update(1)
    # then left side, reset warm start
    x_free_current = np.delete(np.asarray(x_hat_full), fix_idx)
    left_val = []
    left_params_full_nat = []
    for gv_log in grid_log[: len(grid_log) // 2][::-1]:
        res, free_idx = runfit(gv_log, x_free_current)
        x_free_current = res.x  # warm start next

        # collect profiled optimum in FULL parameter space (natural)
        x_full_prof_log = np.array(x_hat_full, copy=True)
        x_full_prof_log[free_idx] = res.x
        x_full_prof_log[fix_idx] = gv_log

        # prof_losses.append(float(res.fun))
        left_params_full_nat.append(np.exp(x_full_prof_log))
        left_val.append(float(res.fun))

        if verbose:
            pbar.update(1)
    if verbose:
        pbar.close()
    prof_losses = left_val[::-1] + right_val
    prof_params_full_nat = left_params_full_nat[::-1] + right_params_full_nat

    return {
        "grid_natural": np.asarray(grid_vals_natural),
        "profile_loss": np.asarray(prof_losses),  # this is your profiled NLL
        "profile_params_full": np.asarray(prof_params_full_nat),  # exp(log x)
    }


@jax.jit
def LLH(data: jnp.ndarray, noise: jnp.ndarray, covmat: jnp.ndarray) -> jnp.ndarray:
    """Log-likelihood of a batch of single-track observations under a GP.

    Parameters
    ----------
    data : jnp.ndarray, shape (ndat, ndim)
        One track with optional NaNs for missing points.
    noise : jnp.ndarray, shape (ndim,)
        Per-dimension observational noise std.
    covmat : jnp.ndarray, shape (ndim, ndat, ndat)
        Prior covariance for each dimension at the observed times.

    Returns
    -------
    jnp.ndarray
        Scalar log-likelihood summed over dimensions and valid entries.
    """
    nan_entries, masked_data, mats_for_cholesky = GPR_utils.get_mat_for_cholesky(
        data, covmat, noise
    )
    cholesky_pred = jnp.linalg.cholesky(mats_for_cholesky)  # (ndim,ndat,ndat)

    z = jax.scipy.linalg.solve_triangular(
        cholesky_pred, masked_data.T, lower=True
    )  # (ndim,ndat)
    x = jax.scipy.linalg.solve_triangular(
        cholesky_pred.swapaxes(-1, -2), z, lower=False
    )  # (ndim,ndat)

    log_diag = jnp.linalg.diagonal(jnp.log(cholesky_pred))
    logdet = jnp.sum(log_diag, axis=-1)  # (ndim,)
    data_term = jnp.einsum("ji,ij->i", masked_data, x)  # (ndim,)
    llh = -0.5 * data_term - logdet  # (ndim,)

    return jnp.sum(llh) - jnp.sum(~nan_entries) * jnp.log(2 * jnp.pi) / 2


@jax.jit
def LLH_value_and_grad(data, noise, covmat, paramderivs):
    """Log-likelihood and gradient w.r.t. kernel parameters for one track.

    Parameters
    ----------
    data : jnp.ndarray, shape (ndat, ndim)
        One track with optional NaNs for missing points.
    noise : jnp.ndarray, shape (ndim,)
        Per-dimension observational noise std.
    covmat : jnp.ndarray, shape (ndim, ndat, ndat)
        Prior covariance for each dimension at the observed times.
    paramderivs : jnp.ndarray, shape (ndim, ndat, ndat, n_params)
        Covariance derivatives per parameter for each dimension.

    Returns
    -------
    llh : jnp.ndarray
        Scalar log-likelihood summed over dimensions and valid entries.
    grad : jnp.ndarray, shape (n_params,)
        Gradient of the log-likelihood with respect to parameters, with noise
        gradients stacked last.
    """
    n, d = data.shape

    nan_entries, y, mats_for_cholesky = GPR_utils.get_mat_for_cholesky(
        data, covmat, noise
    )

    # Cholesky factor for each dim
    L = jnp.linalg.cholesky(mats_for_cholesky)  # (d, n, n)

    z = jax.scipy.linalg.solve_triangular(L, y.T, lower=True)  # (d, n)
    alpha = jax.scipy.linalg.solve_triangular(
        jnp.swapaxes(L, -1, -2), z, lower=False
    )  # (d, n)

    # Log-determinant: log|K| = 2 * sum(log(diag(L)))
    logdet_per_dim = jnp.sum(
        jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=-1
    )  # (d,)
    # Data term: -1/2 y^T K^{-1} y = -1/2 sum_i alpha_i * y_i  (per dim)
    llh_data_terms = -0.5 * jnp.sum(alpha * y.T, axis=-1)  # (d,)

    # Combine per-dim terms; constant term only counts observed entries
    n_obs_total = jnp.sum(~nan_entries)  # scalar
    llh = jnp.sum(llh_data_terms - logdet_per_dim) - 0.5 * n_obs_total * jnp.log(
        2.0 * jnp.pi
    )

    # ===== Gradient =====
    # Rearrange paramderivs to (d, P, n, n) to align with dimensions
    dK = jnp.transpose(paramderivs, (0, 3, 1, 2))  # (d, P, n, n)

    # Term 1: 0.5 * alpha^T (dK/dtheta) alpha   -> shape (d, P)
    # Compute bilinear form without forming xx^T
    term1 = 0.5 * jnp.einsum("di,dpij,dj->dp", alpha, dK, alpha)

    # Term 2: 0.5 * tr(K^{-1} dK/dtheta)
    # Compute K^{-1} once per dim via two triangular solves with identity
    I = jnp.eye(n)
    tmp = jax.scipy.linalg.solve_triangular(
        L, jnp.broadcast_to(I, (d, n, n)), lower=True
    )  # (d, n, n)
    K_inv = jax.scipy.linalg.solve_triangular(
        jnp.swapaxes(L, -1, -2), tmp, lower=False
    )  # (d, n, n)

    # Frobenius inner product to get the trace for each (dim, param)
    # tr(K^{-1} dK) = sum_ij K_inv_{ij} * dK_{ji}
    term2 = 0.5 * jnp.einsum("dij,dpji->dp", K_inv, dK)

    # Sum over dims, result is (P,)
    grad = term1 - term2  # (d, P)
    noise_grad, paramgrad = grad[:, -1], grad[:, :-1]  # (d,1), (d, P-1)

    # flatten and concat to get final gradient
    grad = jnp.concatenate((paramgrad.flatten(), noise_grad))  # (P,)

    return llh, grad


vmap_LLH = jax.vmap(LLH, in_axes=(0, None, None))
vmap_value_and_grad = jax.vmap(LLH_value_and_grad, in_axes=(0, None, None, 0))


class GPR:
    """Gaussian Process regressor with MSD-parameterized covariance.

    Parameters
    ----------
    ts : array-like
        Observation times for training data.
    MSD_func : callable
        Function ``msd(t, params)`` returning the MSD at lag ``t``.
    dim : int
        Number of spatial dimensions in each track.
    per_dim_mask : sequence of bool, optional
        Layout of MSD parameters in compact fitting vectors. ``False`` means the
        parameter is shared across dimensions and appears once; ``True`` means it
        is fit separately per dimension and therefore appears ``dim`` times.
        Noise terms remain per-dimension.
    param_layout : dict or sequence, optional
        General compact layout for MSD parameters. Each scope is one of ``()``,
        ``("dim",)``, ``("dataset",)``, or ``("dataset", "dim")``. Dict keys are
        used as parameter names and insertion order is treated as MSD parameter
        order. When omitted, ``per_dim_mask`` or legacy all-shared MSD behavior is
        used.
    noise_layout : sequence or str, optional
        Scope for noise terms when ``param_layout`` is used. Defaults to
        per-dimension noise for single datasets.
    param_names : sequence of str, optional
        Names for MSD parameters when ``param_layout`` is a sequence.

    Notes
    -----
    ``sample_prior`` and ``LLH`` consume the fully expanded parameter vector
    ``[params_dim0, params_dim1, ..., noise_per_dim]``. ``fit_hyperparams``,
    ``Predict``, ``sample_posterior``, and ``MCMC`` consume a compact vector.
    Multi-dataset fitting is enabled by passing data as a list of arrays or as
    an array with leading dataset axis and by providing ``param_layout``.
    NaNs in data are handled by inflating diagonal noise and masking values.
    """

    def __init__(
        self,
        ts,
        MSD_func,
        dim,
        per_dim_mask=None,
        param_layout=None,
        noise_layout=("dim",),
        param_names=None,
    ):
        self.ts = ts
        self.MSD_func = MSD_func
        self.dim = dim
        self.noise_layout = _as_scope(noise_layout)
        self.param_names = None
        self.param_layout = None
        if per_dim_mask is None:
            self.per_dim_mask = None
        else:
            mask = np.asarray(per_dim_mask, dtype=bool)
            if mask.ndim != 1:
                raise ValueError("per_dim_mask must be a 1D sequence of booleans")
            self.per_dim_mask = tuple(bool(entry) for entry in mask.tolist())
        if param_layout is not None:
            if self.per_dim_mask is not None:
                raise ValueError("Use either per_dim_mask or param_layout, not both")
            if isinstance(param_layout, dict):
                self.param_names = tuple(param_layout.keys())
                self.param_layout = tuple(
                    _as_scope(scope) for scope in param_layout.values()
                )
            else:
                self.param_layout = tuple(_as_scope(scope) for scope in param_layout)
                if param_names is None:
                    self.param_names = tuple(
                        f"param_{idx}" for idx in range(len(self.param_layout))
                    )
                else:
                    if len(param_names) != len(self.param_layout):
                        raise ValueError("param_names must match param_layout length")
                    self.param_names = tuple(param_names)
        self._Construct_covbuilder()

    def _compact_msd_param_count(self):
        if self.param_layout is not None:
            raise ValueError(
                "_compact_msd_param_count requires n_datasets when param_layout is used"
            )
        if self.per_dim_mask is None:
            return None
        return sum(self.dim if is_per_dim else 1 for is_per_dim in self.per_dim_mask)

    def _param_layout_for_n(self, n_datasets):
        if self.param_layout is not None:
            return self.param_layout
        if self.per_dim_mask is not None:
            return tuple(
                ("dim",) if is_per_dim else () for is_per_dim in self.per_dim_mask
            )
        if n_datasets != 1:
            raise ValueError(
                "Multi-dataset compact parameters require param_layout or per_dim_mask"
            )
        return None

    def _compact_param_count_for_n(self, n_datasets):
        layout = self._param_layout_for_n(n_datasets)
        if layout is None:
            return None
        return sum(_scope_width(scope, n_datasets, self.dim) for scope in layout)

    def _compact_total_count_for_n(self, n_datasets):
        n_msd = self._compact_param_count_for_n(n_datasets)
        if n_msd is None:
            return None
        return n_msd + _scope_width(self.noise_layout, n_datasets, self.dim)

    def _expand_scoped_block(self, block, scope, dataset_index, n_datasets):
        if scope == ():
            return jnp.repeat(block[0], self.dim)
        if scope == ("dim",):
            return block
        if scope == ("dataset",):
            if dataset_index is None:
                raise ValueError(
                    "dataset_index is required for dataset-scoped parameters"
                )
            return jnp.repeat(block[dataset_index], self.dim)
        if scope == ("dataset", "dim"):
            if dataset_index is None:
                raise ValueError(
                    "dataset_index is required for dataset-scoped parameters"
                )
            return block.reshape(n_datasets, self.dim)[dataset_index]
        if scope == ("dim", "dataset"):
            if dataset_index is None:
                raise ValueError(
                    "dataset_index is required for dataset-scoped parameters"
                )
            return block.reshape(self.dim, n_datasets)[:, dataset_index]
        raise ValueError(f"Unsupported parameter scope {scope}")

    def _compact_param_labels(self, n_datasets):
        layout = self._param_layout_for_n(n_datasets)
        if layout is None:
            return None
        labels = []
        names = self.param_names
        if names is None:
            names = tuple(f"param_{idx}" for idx in range(len(layout)))
        for name, scope in zip(names, layout):
            if scope == ():
                labels.append(name)
            elif scope == ("dim",):
                labels.extend(f"{name}[dim={dim}]" for dim in range(self.dim))
            elif scope == ("dataset",):
                labels.extend(
                    f"{name}[dataset={dataset}]" for dataset in range(n_datasets)
                )
            elif scope == ("dataset", "dim"):
                labels.extend(
                    f"{name}[dataset={dataset}, dim={dim}]"
                    for dataset in range(n_datasets)
                    for dim in range(self.dim)
                )
            elif scope == ("dim", "dataset"):
                labels.extend(
                    f"{name}[dim={dim}, dataset={dataset}]"
                    for dim in range(self.dim)
                    for dataset in range(n_datasets)
                )
        noise_scope = self.noise_layout
        if noise_scope == ():
            labels.append("noise")
        elif noise_scope == ("dim",):
            labels.extend(f"noise[dim={dim}]" for dim in range(self.dim))
        elif noise_scope == ("dataset",):
            labels.extend(f"noise[dataset={dataset}]" for dataset in range(n_datasets))
        elif noise_scope == ("dataset", "dim"):
            labels.extend(
                f"noise[dataset={dataset}, dim={dim}]"
                for dataset in range(n_datasets)
                for dim in range(self.dim)
            )
        elif noise_scope == ("dim", "dataset"):
            labels.extend(
                f"noise[dim={dim}, dataset={dataset}]"
                for dim in range(self.dim)
                for dataset in range(n_datasets)
            )
        return labels

    def expand_compact_theta(self, theta, dataset_index=None, n_datasets=1):
        """Expand a compact fit vector into one dataset's full parameter vector."""
        theta = jnp.asarray(theta)
        if theta.ndim != 1:
            raise ValueError("theta must be a 1D parameter vector")
        if n_datasets < 1:
            raise ValueError("n_datasets must be positive")

        layout = self._param_layout_for_n(n_datasets)
        if layout is None:
            if theta.shape[0] < self.dim:
                raise ValueError("theta must include one noise parameter per dimension")
            noise = theta[-self.dim :]
            params = theta[: -self.dim]
            expanded_params = jnp.tile(params, self.dim)
            return jnp.concatenate((expanded_params, noise))

        expected = self._compact_total_count_for_n(n_datasets)
        if theta.shape[0] != expected:
            raise ValueError(
                f"Compact parameter vector has length {theta.shape[0]}, "
                f"expected {expected} from param_layout/noise_layout"
            )

        if dataset_index is None:
            if n_datasets == 1:
                dataset_index = 0
            else:
                raise ValueError("dataset_index is required when n_datasets > 1")

        per_param = []
        start = 0
        for scope in layout:
            width = _scope_width(scope, n_datasets, self.dim)
            param_block = theta[start : start + width]
            per_param.append(
                self._expand_scoped_block(param_block, scope, dataset_index, n_datasets)
            )
            start += width

        noise_width = _scope_width(self.noise_layout, n_datasets, self.dim)
        noise = self._expand_scoped_block(
            theta[start : start + noise_width],
            self.noise_layout,
            dataset_index,
            n_datasets,
        )

        if per_param:
            expanded_params = jnp.stack(per_param, axis=1).reshape(-1)
        else:
            expanded_params = jnp.zeros((0,), dtype=theta.dtype)

        return jnp.concatenate((expanded_params, noise))

    def _Construct_covbuilder(self):
        """Pre-build JIT-ed covariance and gradient builders for the MSD kernel."""
        msd_grad = jax.jacfwd(
            self.MSD_func, argnums=1
        )  # gradient of MSD with respect to parameters
        vmap_msd_grad = jax.vmap(
            msd_grad, in_axes=(None, 0)
        )  # vectorized gradient of MSD to run across dims
        vmap_msd = jax.vmap(
            self.MSD_func, in_axes=(None, 0)
        )  # vectorized MSD to run across dims

        def plateau_lag(theta):
            max_safe = jnp.sqrt(
                jnp.asarray(np.finfo(theta.dtype).max, dtype=theta.dtype)
            ) * jnp.asarray(0.1, dtype=theta.dtype)
            return jnp.minimum(jnp.asarray(1e23, dtype=theta.dtype), max_safe)

        @jax.jit
        def Build_covmat(theta, t1s, t2s):
            params_per_dim = theta.reshape(self.dim, -1)  # (ndim, nparams)
            lag_inf = plateau_lag(theta)

            covmat = 0.5 * (
                vmap_msd(lag_inf, params_per_dim)[:, None, None]
                - vmap_msd(t1s[:, None] - t2s[None, :], params_per_dim)
            )
            return covmat  # (ndim,ndat,ndat)

        @jax.jit
        def Build_covmat_grad(theta, data, t1s, t2s):
            ndim = data.shape[-1]
            # unpack parameters
            noise = theta[-ndim:]
            params = theta[:-ndim]
            params_per_dim = params.reshape(self.dim, -1)  # (ndim, nparams)
            lag_inf = plateau_lag(theta)
            param_grads = 0.5 * (
                vmap_msd_grad(lag_inf, params_per_dim)[:, None, None, :]
                - vmap_msd_grad(t1s[:, None] - t2s[None, :], params_per_dim)
            )  # (ndims,ndat,ndat,nparams)
            # covmat = 0.5 * (msd_grad(1e23,params_per_dim) - msd_grad(t1s[None,:, None,None] - t2s[None,None, :,None],params_per_dim[:,None,None,:])) # (ndims,ndat,ndat,nparams)

            # add noise
            nan_entries = jnp.isnan(data)  # (ndat,ndim)
            noise_vals = jnp.where(
                nan_entries, 0.0 * jnp.ones(len(noise)), 4 * noise
            )  # (ndat,ndim)

            diag_noise = jax.vmap(jnp.diag)(noise_vals.T)  # (ndim,ndat,ndat)

            # noise_grads = jnp.broadcast_to(diag_noise,(ndim,*covmat.shape) ) # (ndim,ndat,ndat,n_noise)
            # param_grads = jnp.broadcast_to(covmat,(ndim,*covmat.shape) )# (ndim,ndat,ndat,nparams)

            # merge the two gradients along the last axis
            mats_for_cholesky = jnp.concatenate(
                (param_grads, diag_noise[..., None]), axis=-1
            )  # (ndim,ndat,ndat,nparams+1)

            return mats_for_cholesky

        self.covbuilder = Build_covmat
        self.grad = jax.vmap(Build_covmat_grad, in_axes=(None, 0, None, None))

    def sample_prior(self, theta, num_samples, nanfrac=0.0, seed=42, batch_size=10):
        """Draw prior GP samples and optionally drop NaNs.

        Parameters
        ----------
        theta : array-like, shape (n_params * dim + dim,)
            Fully expanded parameters tiled per-dimension followed by
            per-dimension noise. Use ``expand_compact_theta`` to convert the
            output of fitting methods when ``per_dim_mask`` is set.
        num_samples : int
            Number of tracks to sample.
        nanfrac : float, optional
            Fraction of entries to mask as NaN in each track.
        seed : int, optional
            PRNG seed.
        batch_size : int, optional
            Batch size for sampling to limit memory.
        """
        noise = theta[-self.dim :]  # last dim is noise
        paramvec = theta[: -self.dim]  # all but last dim are parameters

        key = jax.random.PRNGKey(seed)

        mu = jnp.zeros((num_samples, len(self.ts), self.dim))
        nan_mask = (
            jax.random.uniform(key, shape=(num_samples, len(self.ts), self.dim))
            < nanfrac
        )
        nan_mask = jnp.where(nan_mask, jnp.nan, 1.0)

        mu_iterator = mu * nan_mask

        covmat = self.covbuilder(paramvec, self.ts, self.ts)  # (ndim,ndat,ndat)

        if num_samples <= batch_size:
            batch_inds = [np.arange(num_samples)]
        else:
            batch_inds = np.array_split(
                np.arange(num_samples),
                num_samples // batch_size + 1,
            )
        sample_list = []

        seeds = (
            seed
            + jnp.arange(len(batch_inds))[:, None]
            + jnp.arange(self.dim)[None, :] * len(batch_inds)
        )

        for seedval, inds in zip(seeds, batch_inds):
            mus = mu_iterator[inds]

            samples = jax.vmap(GPR_utils.sample_gauss_cmat, in_axes=(2, 0, 0))(
                mus,
                covmat,
                seedval,
            ).transpose(1, 2, 0)

            sample_list.append(samples)
        samples = np.concatenate(sample_list, axis=0)

        # output = [samples]

        key, subkey_noise, subkey_nan = jax.random.split(key, 3)

        noisy_samples = samples + np.sqrt(2) * noise[None, None, :] * jax.random.normal(
            subkey_noise, shape=samples.shape
        )

        return samples, noisy_samples

    def Predict(
        self,
        x,
        prediction_points,
        noisy_samples,
        batch_size=100,
        verbose=True,
    ):
        """Posterior mean/variance at prediction points for many tracks.

        Parameters
        ----------
        x : array-like
            Compact parameter vector in abs space, interpreted according to
            ``per_dim_mask`` before being expanded internally.
        prediction_points : array-like
            Times at which to predict.
        noisy_samples : array-like, shape (n_tracks, n_time, dim)
            Observed tracks (can include NaNs).
        batch_size : int, optional
            Batch size for prediction loops.
        verbose : bool, optional
            Show progress bar when True.
        """
        final_params = self.expand_compact_theta(jnp.abs(x))
        noise = final_params[-self.dim :]  # last dim is noise
        theta = final_params[: -self.dim]  # all but last dim are parameters

        prediction_covmat = self.covbuilder(theta, prediction_points, prediction_points)
        prediction_kernel = self.covbuilder(theta, prediction_points, self.ts)
        covmat = self.covbuilder(theta, self.ts, self.ts)

        mean_out = np.zeros((len(noisy_samples), len(prediction_points), self.dim))
        var_out = np.zeros((len(noisy_samples), len(prediction_points), self.dim))

        data_batches = np.array_split(
            noisy_samples, len(noisy_samples) // batch_size + 1
        )
        if verbose:
            data_batches = tqdm(
                data_batches, desc="Predicting", total=len(data_batches)
            )
        else:
            data_batches = iter(data_batches)
        count = 0
        for batch in data_batches:
            # run prediction at float 64
            mean_pred = GPR_utils.v_pred(prediction_kernel, batch, covmat, noise)

            cov_pred = GPR_utils.vmap_pred_cov(
                prediction_kernel, covmat, batch, prediction_covmat, noise
            )
            mean_out[count : count + len(batch)] = mean_pred
            var_out[count : count + len(batch)] = cov_pred
            count += len(batch)
        # mean_pred = v_pred(prediction_kernel, noisy_samples, covmat,noise)
        # cov_pred = vmap_pred_cov(prediction_kernel, covmat, noisy_samples, prediction_covmat, noise)

        # return mean_pred,cov_pred
        return mean_out, var_out

    def sample_posterior(
        self,
        x,
        prediction_points,
        noisy_samples,
        num_samples=100,
        seed=42,
        verbose=True,
        batch_size=100,
    ):
        """Draw posterior GP samples at prediction points given observed tracks.

        Parameters
        ----------
        x : array-like
            Compact parameter vector in abs space, interpreted according to
            ``per_dim_mask`` before being expanded internally.
        prediction_points : array-like
            Times at which to sample the posterior.
        noisy_samples : array-like, shape (n_tracks, n_time, dim)
            Observed tracks (can include NaNs).
        num_samples : int, optional
            Number of posterior samples per track.
        seed : int, optional
            PRNG seed for sampling.
        verbose : bool, optional
            Show progress bar when True.
        batch_size : int, optional
            Batch size for prediction loops.
        """
        final_params = self.expand_compact_theta(jnp.abs(x))
        noise = final_params[-self.dim :]  # last dim is noise
        theta = final_params[: -self.dim]  # all but last dim are parameters

        prediction_covmat = self.covbuilder(theta, prediction_points, prediction_points)
        prediction_kernel = self.covbuilder(theta, prediction_points, self.ts)
        covmat = self.covbuilder(theta, self.ts, self.ts)  # (ndim,ndat,ndat)

        out_samps = np.zeros(
            (len(noisy_samples), num_samples, len(prediction_points), self.dim)
        )
        data_batches = np.array_split(
            noisy_samples, len(noisy_samples) // batch_size + 1
        )
        if verbose:
            iterator = tqdm(
                data_batches, desc="Sampling posterior", total=len(data_batches)
            )
        else:
            iterator = data_batches
        count = 0
        # for i in iterator:
        for batch in iterator:
            # mean_pred = predict_mean_single(prediction_kernel, noisy_samples[i], covmat, noise)
            # cov_pred = predict_cov_single(prediction_kernel, covmat, noisy_samples[i], prediction_covmat, noise)
            # out_samps[i] = sample_gauss(num_samples, mean_pred, cov_pred, seed=seed+i)
            mean_pred = GPR_utils.v_pred(prediction_kernel, batch, covmat, noise)
            cov_pred = GPR_utils.vmap_pred_cov_full(
                prediction_kernel, covmat, batch, prediction_covmat, noise
            )

            out_samps[count : count + len(batch)] = jax.vmap(
                GPR_utils.sample_gauss, in_axes=(None, 0, 0, None)
            )(num_samples, mean_pred, cov_pred.transpose(0, 3, 1, 2), seed + count)
            count += len(batch)
        return out_samps

    def LLH(self, theta, data, batch_size=500, verbose=False):
        """Total log-likelihood across a batch of tracks."""
        return self._LLH_with_ts(
            theta, data, self.ts, batch_size=batch_size, verbose=verbose
        )

    def _LLH_with_ts(self, theta, data, ts, batch_size=500, verbose=False):
        """Total log-likelihood for one dataset at one time grid."""
        noise = theta[-self.dim :]  # last dim is noise
        paramvec = theta[: -self.dim]  # all but last dim are parameters

        covmat = self.covbuilder(paramvec, ts, ts)

        data_batches = np.array_split(data, len(data) // batch_size + 1)
        llhs = 0
        if verbose:
            data_batches = tqdm(
                data_batches, desc="Computing LLH", total=len(data_batches)
            )
        else:
            data_batches = iter(data_batches)
        for batch in data_batches:
            llhs += jnp.sum(vmap_LLH(batch, noise, covmat))

        return llhs

    def _normalize_data_list(self, data):
        if isinstance(data, (list, tuple)):
            datasets = [jnp.asarray(entry) for entry in data]
        else:
            arr = jnp.asarray(data)
            if arr.ndim == 3:
                datasets = [arr]
            elif arr.ndim == 4:
                datasets = [arr[i] for i in range(arr.shape[0])]
            else:
                raise ValueError(
                    "data must have shape (n_tracks, n_time, dim), "
                    "(n_datasets, n_tracks, n_time, dim), or be a list of datasets"
                )
        for dataset in datasets:
            if dataset.ndim != 3 or dataset.shape[-1] != self.dim:
                raise ValueError("Each dataset must have shape (n_tracks, n_time, dim)")
        return datasets

    def _normalize_ts_list(self, n_datasets):
        ts = self.ts
        if isinstance(ts, (list, tuple)):
            ts_list = [jnp.asarray(entry) for entry in ts]
        else:
            ts_arr = jnp.asarray(ts)
            if ts_arr.ndim == 1:
                ts_list = [ts_arr]
            elif ts_arr.ndim == 2:
                ts_list = [ts_arr[i] for i in range(ts_arr.shape[0])]
            else:
                raise ValueError(
                    "ts must be a 1D time grid, list of grids, or 2D array"
                )
        if len(ts_list) == 1 and n_datasets > 1:
            ts_list = ts_list * n_datasets
        if len(ts_list) != n_datasets:
            raise ValueError(f"Got {len(ts_list)} time grids for {n_datasets} datasets")
        return ts_list

    def get_objective(self, data):
        """Return a JIT-ed log-likelihood closure in log-parameter space."""
        datasets = self._normalize_data_list(data)
        ts_list = self._normalize_ts_list(len(datasets))
        n_datasets = len(datasets)

        @jax.jit
        def objective(x):
            params = jnp.exp(x)
            llh = 0.0
            for dataset_index, (dataset, ts) in enumerate(zip(datasets, ts_list)):
                final_params = self.expand_compact_theta(
                    params,
                    dataset_index=dataset_index,
                    n_datasets=n_datasets,
                )
                llh = llh + self._LLH_with_ts(final_params, dataset, ts)

            return -llh

        return objective

    def get_objective_grad(self, data, batch_size=8):
        """Return the analytic gradient of the negative log-likelihood."""
        datasets = self._normalize_data_list(data)
        ts_list = self._normalize_ts_list(len(datasets))
        n_datasets = len(datasets)

        def make_batch_grad(dataset_index, ts):
            @jax.jit
            def batch_grad(params, batch):
                final_params = self.expand_compact_theta(
                    params,
                    dataset_index=dataset_index,
                    n_datasets=n_datasets,
                )
                noise = final_params[-self.dim :]
                paramvec = final_params[: -self.dim]
                covmat = self.covbuilder(paramvec, ts, ts)
                paramderivs = self.grad(final_params, batch, ts, ts)
                _, full_grads = vmap_value_and_grad(batch, noise, covmat, paramderivs)
                full_grad = jnp.sum(full_grads, axis=0)
                _, pullback = jax.vjp(
                    lambda compact_params: self.expand_compact_theta(
                        compact_params,
                        dataset_index=dataset_index,
                        n_datasets=n_datasets,
                    ),
                    params,
                )
                return pullback(full_grad)[0]

            return batch_grad

        batch_grad_fns = [
            make_batch_grad(dataset_index, ts)
            for dataset_index, ts in enumerate(ts_list)
        ]
        data_batches = [
            np.array_split(dataset, len(dataset) // batch_size + 1)
            for dataset in datasets
        ]

        def objective_grad(x):
            params = jnp.exp(jnp.asarray(x))
            grad = jnp.zeros_like(params)
            for batch_grad, batches in zip(batch_grad_fns, data_batches):
                for batch in batches:
                    grad = grad + batch_grad(params, batch)
            return np.asarray(-grad * params, dtype=float)

        return objective_grad

    def fit_hyperparams(
        self,
        data,
        initial_guess,
        method="Nelder-Mead",
        verbose=True,
        profile=True,
        profile_method="Nelder-Mead",
        inds_to_profile=None,
        bounds=None,
        nfits_to_run=1,
        init_scale=1.0,
        grad_batch_size=8,
        **kwargs,
    ):
        datasets = self._normalize_data_list(data)
        ts_list = self._normalize_ts_list(len(datasets))
        n_datasets = len(datasets)
        self.expand_compact_theta(
            jnp.asarray(initial_guess), dataset_index=0, n_datasets=n_datasets
        )
        param_labels = self._compact_param_labels(n_datasets)
        objective = self.get_objective(data)
        minimize_kwargs = dict(kwargs)
        objective_grad = minimize_kwargs.get("jac")
        if str(method).upper() == "L-BFGS-B" and not callable(objective_grad):
            objective_grad = self.get_objective_grad(data, batch_size=grad_batch_size)
            minimize_kwargs["jac"] = objective_grad
        evidences = []
        res_objs = []
        nfailed = 0
        if bounds is not None:
            bounds_log = []
            for b in bounds:
                if b is None:
                    bounds_log.append((None, None))
                else:
                    if len(b) != 2:
                        raise ValueError(
                            "Bounds should be a list of (lower, upper) pairs or None"
                        )
                    bound_entry = []
                    for sub_b in b:
                        if sub_b is None:
                            bound_entry.append(None)
                        else:
                            if sub_b <= 0:
                                raise ValueError(
                                    "Bounds must be positive since parameters are in log space"
                                )
                            bound_entry.append(np.log(sub_b))
                    bounds_log.append(tuple(bound_entry))
        else:
            bounds_log = None
        if verbose:
            pbar = tqdm()
        for fit_num in range(nfits_to_run):

            if verbose:

                if len(evidences) == 0:
                    best_str = "N/A"
                else:
                    best_str = f"{max(evidences):.2f}"

                def callback(xk):
                    pbar.update()
                    pbar.set_description(
                        f"Fit {fit_num + 1}/{nfits_to_run} nfailed: {nfailed}: Evidence: {-objective(xk):.2f} | Current best: {best_str}"
                    )

            else:
                callback = None

            if fit_num == 0:
                init_guess = jnp.log(initial_guess)
            else:
                best_idx = np.argmax(evidences)
                best_params = res_objs[best_idx].x

                perturbation = np.random.randn(*best_params.shape) * init_scale
                init_guess = best_params + perturbation

                obj_val = objective(init_guess)
                while not np.isfinite(obj_val):
                    perturbation = np.random.randn(*best_params.shape) * init_scale
                    init_guess = best_params + perturbation
                    obj_val = objective(init_guess)

            res = minimize(
                objective,
                init_guess,
                method=method,
                callback=callback,
                bounds=bounds_log,
                **minimize_kwargs,
            )
            if not res.success:
                nfailed += 1

            res_objs.append(res)
            if np.isfinite(res.fun):
                evidences.append(-res.fun)
            else:
                evidences.append(-np.inf)

        best_arg = np.argmax(evidences)
        res = res_objs[best_arg]

        vmap_msd = jax.vmap(self.MSD_func, in_axes=(None, 0))

        def MSDfunc(params, dataset_index=0):
            final_params = self.expand_compact_theta(
                jnp.exp(params),
                dataset_index=dataset_index,
                n_datasets=n_datasets,
            )
            noises = final_params[-self.dim :]
            params_per_dim = final_params[: -self.dim].reshape(self.dim, -1)
            ts = ts_list[dataset_index]
            msd = vmap_msd(ts[1:], params_per_dim).T
            return msd + 4 * noises[None, :] ** 2  # (max_lag-1,ndims)

        if verbose:
            pbar.close()
        results = res

        results["all_results"] = res_objs
        results["all_evidences"] = evidences

        if n_datasets == 1:
            lags = jnp.arange(1, datasets[0].shape[1])
            results["lags"] = lags
            results["data msd"] = GPR_utils.msd(datasets[0], lags)
            results["predicted msd"] = MSDfunc(res.x, dataset_index=0)
        else:
            lags = [jnp.arange(1, dataset.shape[1]) for dataset in datasets]
            results["lags"] = lags
            results["data msd"] = [
                GPR_utils.msd(dataset, dataset_lags)
                for dataset, dataset_lags in zip(datasets, lags)
            ]
            results["predicted msd"] = [
                MSDfunc(res.x, dataset_index=dataset_index)
                for dataset_index in range(n_datasets)
            ]
        results["msd_func"] = MSDfunc
        results["final_params"] = jnp.exp(res.x)
        if n_datasets == 1:
            results["final_params_full"] = self.expand_compact_theta(jnp.exp(res.x))
        else:
            results["final_params_full"] = [
                self.expand_compact_theta(
                    jnp.exp(res.x),
                    dataset_index=dataset_index,
                    n_datasets=n_datasets,
                )
                for dataset_index in range(n_datasets)
            ]
        results["per_dim_mask"] = self.per_dim_mask
        results["param_layout"] = self.param_layout
        results["noise_layout"] = self.noise_layout
        results["param_labels"] = param_labels
        x_hat = res.x  # log-parameters at the best run

        if profile:
            if str(profile_method).upper() == "L-BFGS-B" and not callable(
                objective_grad
            ):
                objective_grad = self.get_objective_grad(
                    data, batch_size=grad_batch_size
                )
            # compute profile likelihoods of params
            CIs = []
            if inds_to_profile is None:
                inds_to_profile = np.arange(len(x_hat))
            # for i in range(len(x_hat)):
            for i in np.arange(len(res.x))[inds_to_profile]:
                if not jnp.isfinite(x_hat[i]):
                    raise ValueError("Non-finite parameter in MLE, cannot profile")
                # create grid around MLE
                # factors = np.geomspace(0.25, 4, 25)  # 25 points from 0.25x to 4x
                r = (4 / 0.25) ** (1 / (25 - 1))

                grid_factors = np.geomspace(0.25 / r**25, 4 * r**25, 75)
                grid = np.exp(x_hat[i]) * grid_factors
                prof = _profile_one_param(
                    objective,
                    x_hat,
                    i,
                    grid,
                    method=profile_method,
                    bounds=bounds_log,
                    grad=objective_grad,
                    verbose=verbose,
                )
                results[f"profile_param_{i}"] = prof
                if param_labels is not None:
                    results[f"profile_{_sanitize_label(param_labels[i])}"] = prof

                # compute confidence intervals from profiled likelihood
                prof_losses = prof["profile_loss"]
                profile_grid = prof["grid_natural"]
                LLH_best = res.fun
                threshold = LLH_best + 0.5 * 3.84  # 95% CI, chi2 with 1 dof

                CIs.append(
                    (
                        _find_profile_ci_side(
                            prof_losses, profile_grid, threshold, side="lower"
                        ),
                        _find_profile_ci_side(
                            prof_losses, profile_grid, threshold, side="upper"
                        ),
                    )
                )
            results["CIs"] = np.array(CIs)  # shape (P, 2) lower, upper
            if param_labels is not None:
                results["CI_labels"] = [param_labels[i] for i in inds_to_profile]

        return results

    def MCMC(
        self,
        data,
        initial_guess,
        step_sizes,
        verbose=True,
        seed=42,
        n_samples=1000,
        fixed_params=[],
    ):
        """Simple Metropolis-Hastings sampler over log-parameters.

        Parameters
        ----------
        data : jnp.ndarray, shape (n_tracks, n_time, dim)
            Observed tracks used to evaluate the likelihood.
        initial_guess : array-like
            Starting point in compact log-parameter space.
        step_sizes : array-like
            Proposal scales in log space.
        verbose : bool, optional
            Show sampling progress when True.
        seed : int, optional
            PRNG seed for reproducibility.
        n_samples : int, optional
            Number of MCMC samples to draw.
        fixed_params : list[int], optional
            Indices to freeze (step size set to zero).

        Returns
        -------
        results : jnp.ndarray, shape (n_samples, n_params)
            Sampled parameter states.
        llhs : list[float]
            Log-likelihood trace corresponding to samples.
        """

        objective = self.get_objective(data)
        freeze_mask = np.ones_like(initial_guess)
        for i in fixed_params:
            freeze_mask[i] = 0.0
        step_sizes = step_sizes * freeze_mask

        @jax.jit
        def do_step(rng_key, current_position, objective_current):
            proposal = (
                current_position
                + jax.random.normal(rng_key, shape=current_position.shape) * step_sizes
            )
            objective_proposal = objective(proposal)
            accept_prob = jnp.exp(objective_proposal - objective_current)
            rng_key, subkey = jax.random.split(rng_key)
            u = jax.random.uniform(subkey)
            accept = u < accept_prob
            new_position = jnp.where(accept, proposal, current_position)
            new_objective = jnp.where(accept, objective_proposal, objective_current)
            return new_position, new_objective

        rng_key = jax.random.PRNGKey(seed)
        results = []
        llhs = []
        state_position = initial_guess
        state_objective = objective(initial_guess)
        if verbose:
            iterator = tqdm(
                range(n_samples),
                desc="Sampling with MCMC",
                total=n_samples,
                leave=False,
            )
        else:
            iterator = range(n_samples)
        for i in iterator:
            rng_key, sample_key = jax.random.split(rng_key)
            state_position, state_objective = do_step(
                sample_key, state_position, state_objective
            )
            results.append(state_position)
            llhs.append(-state_objective)
        results = jnp.array(results)
        return results, llhs
