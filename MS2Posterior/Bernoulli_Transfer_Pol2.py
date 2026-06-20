"""Exact Bernoulli transfer-matrix solver for the MS2 Pol2 loading layer."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def _ms2_kernel_values(
    dt: float, t_rise: float, t_plateau: float, rna_intensity: np.ndarray | float
) -> np.ndarray:
    support = float(t_rise + t_plateau)
    n_window = int(support / dt) + 1
    dts = np.arange(n_window, dtype=float) * dt
    vals = (
        (dts >= 0.0)
        & (dts <= support)
    ).astype(float) * (
        (dts / t_rise) * (dts <= t_rise) + (dts > t_rise)
    )
    base = vals[::-1]
    rna = np.asarray(rna_intensity, dtype=float)
    if rna.ndim == 0:
        return float(rna) * base
    if rna.ndim == 1:
        return rna[:, None] * base[None, :]
    raise ValueError("rna_intensity must be scalar or 1D array of length ntraj")


def _state_bits(window_size: int) -> np.ndarray:
    states = np.arange(1 << window_size, dtype=np.uint32)
    shifts = np.arange(window_size - 1, -1, -1, dtype=np.uint32)
    return ((states[:, None] >> shifts[None, :]) & 1).astype(float)


def _prepare_inputs(
    data: np.ndarray,
    prior_rates: np.ndarray,
    noise: np.ndarray | float,
    data_times: np.ndarray,
    fine_grid: np.ndarray,
    t_rise: float,
    t_plateau: float,
    rna_intensity: np.ndarray | float,
    prior_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[None, :]
    prior_rates = np.asarray(prior_rates, dtype=float)
    if prior_rates.ndim == 1:
        prior_rates = np.broadcast_to(prior_rates[None, :], (data.shape[0], prior_rates.size))
    if prior_mask is None:
        prior_mask_arr = np.isfinite(prior_rates)
    else:
        prior_mask_arr = np.asarray(prior_mask, dtype=bool)
        if prior_mask_arr.ndim == 1:
            prior_mask_arr = np.broadcast_to(
                prior_mask_arr[None, :], (data.shape[0], prior_mask_arr.size)
            )
        prior_mask_arr = prior_mask_arr & np.isfinite(prior_rates)
    prior_rates = np.where(prior_mask_arr, prior_rates, 0.0)
    noise_arr = np.asarray(noise, dtype=float)
    if noise_arr.ndim == 0:
        noise_arr = np.full(data.shape[0], float(noise_arr))
    noise_arr = np.broadcast_to(noise_arr, (data.shape[0],))

    data_times = np.asarray(data_times, dtype=float)
    fine_grid = np.asarray(fine_grid, dtype=float)
    if fine_grid.ndim != 1 or fine_grid.size < 2:
        raise ValueError("fine_grid must be a one-dimensional grid with at least two entries")
    if data.shape[0] != prior_rates.shape[0]:
        raise ValueError("data and prior_rates must have the same number of tracks")
    if data.shape[1] != data_times.size:
        raise ValueError("data_times must match the number of data columns")
    if prior_rates.shape[1] != fine_grid.size:
        raise ValueError("fine_grid must match the number of prior-rate columns")
    if prior_mask_arr.shape != prior_rates.shape:
        raise ValueError("prior_mask must match prior_rates")

    dt = float(fine_grid[1] - fine_grid[0])
    weights = _ms2_kernel_values(dt, t_rise, t_plateau, rna_intensity)
    if weights.ndim == 1:
        weights = np.broadcast_to(weights[None, :], (data.shape[0], weights.size))
    elif weights.ndim == 2:
        if weights.shape[0] != data.shape[0]:
            raise ValueError(
                "rna_intensity must be scalar or length ntraj, got "
                f"{weights.shape[0]} for {data.shape[0]} trajectories"
            )
    else:
        raise ValueError("rna_intensity must be scalar or 1D array of length ntraj")
    window_size = int(weights.shape[1])
    raw_starts = np.rint(
        (data_times - float(t_rise + t_plateau) - fine_grid[0]) / dt
    ).astype(np.int32)
    if not np.allclose(
        fine_grid[0] + raw_starts * dt,
        data_times - float(t_rise + t_plateau),
        atol=max(1e-9, abs(dt) * 1e-6),
    ):
        raise ValueError("data_times must align with the fine grid spacing")
    pad_left = int(max(0, -np.min(raw_starts)))
    starts = raw_starts + pad_left
    shifts = np.diff(starts, prepend=starts[0]).astype(np.int32)
    if np.any(shifts < 0):
        raise ValueError("observation windows must be monotonic")

    prior_probs = np.clip(1.0 - np.exp(-prior_rates * dt), 1e-12, 1.0 - 1e-12)
    if pad_left:
        prior_probs = np.pad(
            prior_probs,
            ((0, 0), (pad_left, 0)),
            mode="constant",
            constant_values=1e-12,
        )
    needed = int(np.max(starts) + window_size)
    if needed > prior_probs.shape[1]:
        pad = needed - prior_probs.shape[1]
        prior_probs = np.pad(prior_probs, ((0, 0), (0, pad)), mode="edge")

    return {
        "data": data,
        "prior_probs": prior_probs,
        "original_grid_size": int(fine_grid.size),
        "pad_left": pad_left,
        "noise": noise_arr,
        "starts": starts,
        "shifts": shifts,
        "append_inds": (starts + window_size - 1).astype(np.int32),
        "weights": weights,
        "window_size": window_size,
    }


@partial(
    jax.jit,
    static_argnames=("starts0", "window_size", "max_shift", "n_states"),
)
def _transfer_posterior_value_and_grad(
    field,
    batch_data,
    batch_logits,
    batch_noise,
    bits,
    emission_means,
    shifts,
    append_inds,
    log2pi,
    *,
    starts0: int,
    window_size: int,
    max_shift: int,
    n_states: int,
):
    def loglik_fn(local_field):
        def initial_alpha():
            logits = jax.lax.dynamic_slice(
                batch_logits + local_field,
                (0, starts0),
                (batch_logits.shape[0], window_size),
            )
            logp1 = -jax.nn.softplus(-logits)
            logp0 = -jax.nn.softplus(logits)
            logp = bits[None, :, :] * logp1[:, None, :]
            logp = logp + (1.0 - bits[None, :, :]) * logp0[:, None, :]
            logp = jnp.sum(logp, axis=2)
            logp = logp - jnp.max(logp, axis=1, keepdims=True)
            alpha = jnp.exp(logp)
            return alpha / jnp.sum(alpha, axis=1, keepdims=True)

        def append_state(alpha, p_new):
            if window_size == 1:
                return jnp.stack([1.0 - p_new, p_new], axis=1)
            collapsed = alpha.reshape(
                (alpha.shape[0], 2, 1 << (window_size - 1))
            ).sum(axis=1)
            return jnp.stack(
                [collapsed * (1.0 - p_new)[:, None], collapsed * p_new[:, None]],
                axis=2,
            ).reshape((alpha.shape[0], n_states))

        alpha0 = initial_alpha()
        loglik0 = jnp.zeros((batch_data.shape[0],), dtype=batch_data.dtype)

        def step(carry, inputs):
            alpha, loglik = carry
            y_t, shift_t, append_ind_t = inputs

            def append_one(i, a):
                append_ind = append_ind_t - shift_t + 1 + i

                def do_append(aa):
                    p_new = jax.nn.sigmoid((batch_logits + local_field)[:, append_ind])
                    return append_state(aa, p_new)

                return jax.lax.cond(i < shift_t, do_append, lambda aa: aa, a)

            alpha = jax.lax.fori_loop(0, max_shift, append_one, alpha)
            finite = jnp.isfinite(y_t)
            residual = y_t[:, None] - emission_means
            obs_logp = -0.5 * (log2pi + 2.0 * jnp.log(batch_noise[:, None]))
            obs_logp = obs_logp - 0.5 * residual * residual / (
                batch_noise[:, None] ** 2
            )
            obs_logp = jnp.where(finite[:, None], obs_logp, 0.0)
            joint_logp = (
                jnp.log(jnp.maximum(alpha, jnp.finfo(batch_data.dtype).tiny))
                + obs_logp
            )
            max_logp = jnp.max(joint_logp, axis=1)
            weighted = jnp.exp(joint_logp - max_logp[:, None])
            norm = jnp.sum(weighted, axis=1)
            updated = weighted / norm[:, None]
            alpha = jnp.where(finite[:, None], updated, alpha)
            loglik = loglik + jnp.where(finite, max_logp + jnp.log(norm), 0.0)
            return (alpha, loglik), None

        (_, loglik), _ = jax.lax.scan(
            step,
            (alpha0, loglik0),
            (batch_data.T, shifts, append_inds),
        )
        return jnp.sum(loglik)

    return jax.value_and_grad(loglik_fn)(field)


def exact_bernoulli_transfer_loglikelihood(
    data: np.ndarray,
    prior_rates: np.ndarray,
    noise: np.ndarray | float,
    data_times: np.ndarray,
    fine_grid: np.ndarray,
    t_rise: float,
    t_plateau: float,
    rna_intensity: np.ndarray | float,
    *,
    batch_size: int = 16,
    use_x64: bool = False,
    prior_mask: np.ndarray | None = None,
    survival_rate_correction: float | np.ndarray = 0.0,
) -> dict[str, Any]:
    """Return the exact marginal log-likelihood under binary Pol2 loading."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", use_x64)
    dtype = jnp.float64 if use_x64 else jnp.float32

    prepared = _prepare_inputs(
        data,
        prior_rates,
        noise,
        data_times,
        fine_grid,
        t_rise,
        t_plateau,
        rna_intensity,
        prior_mask=prior_mask,
    )
    data = prepared["data"]
    prior_probs = prepared["prior_probs"]
    noise_arr = prepared["noise"]
    starts = prepared["starts"]
    shifts = prepared["shifts"]
    append_inds = prepared["append_inds"]
    weights = prepared["weights"]
    window_size = prepared["window_size"]
    n_states = int(1 << window_size)
    max_shift = int(np.max(shifts))

    bits = jnp.asarray(_state_bits(window_size), dtype=dtype)
    shifts_j = jnp.asarray(shifts, dtype=jnp.int32)
    append_inds_j = jnp.asarray(append_inds, dtype=jnp.int32)
    log2pi = jnp.asarray(np.log(2.0 * np.pi), dtype=dtype)

    def initial_alpha(batch_probs):
        window = jax.lax.dynamic_slice(
            batch_probs, (0, int(starts[0])), (batch_probs.shape[0], window_size)
        )
        window = jnp.clip(window, 1e-12, 1.0 - 1e-12)
        logp = bits[None, :, :] * jnp.log(window[:, None, :])
        logp = logp + (1.0 - bits[None, :, :]) * jnp.log1p(-window[:, None, :])
        logp = jnp.sum(logp, axis=2)
        logp = logp - jnp.max(logp, axis=1, keepdims=True)
        alpha = jnp.exp(logp)
        return alpha / jnp.sum(alpha, axis=1, keepdims=True)

    def append_state(alpha, p_new):
        if window_size == 1:
            return jnp.stack([1.0 - p_new, p_new], axis=1)
        collapsed = alpha.reshape((alpha.shape[0], 2, 1 << (window_size - 1))).sum(axis=1)
        return jnp.stack(
            [collapsed * (1.0 - p_new)[:, None], collapsed * p_new[:, None]],
            axis=2,
        ).reshape((alpha.shape[0], n_states))

    @jax.jit
    def run_batch(batch_data, batch_probs, batch_noise, batch_weights):
        emission_means = jnp.einsum("sk,bk->bs", bits, batch_weights)
        alpha0 = initial_alpha(batch_probs)
        loglik0 = jnp.zeros((batch_data.shape[0],), dtype=dtype)

        def step(carry, inputs):
            alpha, loglik = carry
            y_t, shift_t, append_ind_t = inputs

            def append_one(i, a):
                append_ind = append_ind_t - shift_t + 1 + i
                return jax.lax.cond(
                    i < shift_t,
                    lambda aa: append_state(aa, batch_probs[:, append_ind]),
                    lambda aa: aa,
                    a,
                )

            alpha = jax.lax.fori_loop(0, max_shift, append_one, alpha)
            finite = jnp.isfinite(y_t)
            residual = y_t[:, None] - emission_means
            obs_logp = -0.5 * (log2pi + 2.0 * jnp.log(batch_noise[:, None]))
            obs_logp = obs_logp - 0.5 * residual * residual / (batch_noise[:, None] ** 2)
            obs_logp = jnp.where(finite[:, None], obs_logp, 0.0)
            joint_logp = jnp.log(jnp.maximum(alpha, jnp.finfo(dtype).tiny)) + obs_logp
            max_logp = jnp.max(joint_logp, axis=1)
            weighted = jnp.exp(joint_logp - max_logp[:, None])
            norm = jnp.sum(weighted, axis=1)
            updated = weighted / norm[:, None]
            alpha = jnp.where(finite[:, None], updated, alpha)
            loglik = loglik + jnp.where(finite, max_logp + jnp.log(norm), 0.0)
            return (alpha, loglik), None

        (_, loglik), _ = jax.lax.scan(
            step,
            (alpha0, loglik0),
            (batch_data.T, shifts_j, append_inds_j),
        )
        return loglik

    loglik_parts = []
    for start in range(0, data.shape[0], batch_size):
        stop = min(start + batch_size, data.shape[0])
        batch_loglik = run_batch(
            jnp.asarray(data[start:stop], dtype=dtype),
            jnp.asarray(prior_probs[start:stop], dtype=dtype),
            jnp.asarray(noise_arr[start:stop], dtype=dtype),
            jnp.asarray(weights[start:stop], dtype=dtype),
        )
        loglik_parts.append(np.asarray(batch_loglik, dtype=float))

    loglik = np.concatenate(loglik_parts)
    correction = float(np.sum(np.asarray(survival_rate_correction, dtype=float)))
    raw_loglik_sum = float(np.sum(loglik))
    return {
        "loglik": loglik,
        "loglik_sum": raw_loglik_sum + correction,
        "loglik_sum_raw": raw_loglik_sum,
        "survival_rate_correction": correction,
        "window_size": window_size,
        "n_states": n_states,
        "weights": weights,
        "starts": starts,
    }


def exact_bernoulli_transfer_posterior(
    data: np.ndarray,
    prior_rates: np.ndarray,
    noise: np.ndarray | float,
    data_times: np.ndarray,
    fine_grid: np.ndarray,
    t_rise: float,
    t_plateau: float,
    rna_intensity: np.ndarray | float,
    *,
    batch_size: int = 4,
    use_x64: bool = False,
) -> dict[str, Any]:
    """Return exact binary load marginals and predicted MS2 by autodiff.

    The gradient is with respect to a normalized prior-logit field.  For a
    Bernoulli prior p_phi=logistic(logit(p)+phi), d log p(y)/d phi_t equals
    posterior_p(load_t=1) - p_t.
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", use_x64)
    dtype = jnp.float64 if use_x64 else jnp.float32

    prepared = _prepare_inputs(
        data,
        prior_rates,
        noise,
        data_times,
        fine_grid,
        t_rise,
        t_plateau,
        rna_intensity,
    )
    data = prepared["data"]
    prior_probs = prepared["prior_probs"]
    original_grid_size = prepared["original_grid_size"]
    pad_left = prepared["pad_left"]
    noise_arr = prepared["noise"]
    starts = prepared["starts"]
    shifts = prepared["shifts"]
    append_inds = prepared["append_inds"]
    weights = prepared["weights"]
    window_size = prepared["window_size"]
    n_states = int(1 << window_size)
    max_shift = int(np.max(shifts))

    prior_logits_np = np.log(prior_probs) - np.log1p(-prior_probs)
    bits = jnp.asarray(_state_bits(window_size), dtype=dtype)
    shifts_j = jnp.asarray(shifts, dtype=jnp.int32)
    append_inds_j = jnp.asarray(append_inds, dtype=jnp.int32)
    log2pi = jnp.asarray(np.log(2.0 * np.pi), dtype=dtype)

    def predict_from_load_probs(load_probs: np.ndarray) -> np.ndarray:
        out = np.empty((load_probs.shape[0], data.shape[1]), dtype=float)
        for obs_idx, start in enumerate(starts):
            out[:, obs_idx] = np.sum(
                load_probs[:, start : start + window_size] * weights, axis=1
            )
        out[~np.isfinite(data)] = np.nan
        return out

    loglik_parts = []
    posterior_parts = []
    for start in range(0, data.shape[0], batch_size):
        stop = min(start + batch_size, data.shape[0])
        batch_data = jnp.asarray(data[start:stop], dtype=dtype)
        batch_logits = jnp.asarray(prior_logits_np[start:stop], dtype=dtype)
        batch_noise = jnp.asarray(noise_arr[start:stop], dtype=dtype)
        batch_weights = jnp.asarray(weights[start:stop], dtype=dtype)
        emission_means = jnp.einsum("sk,bk->bs", bits, batch_weights)
        field0 = jnp.zeros_like(batch_logits)
        value, grad = _transfer_posterior_value_and_grad(
            field0,
            batch_data,
            batch_logits,
            batch_noise,
            bits,
            emission_means,
            shifts_j,
            append_inds_j,
            log2pi,
            starts0=int(starts[0]),
            window_size=int(window_size),
            max_shift=int(max_shift),
            n_states=int(n_states),
        )
        posterior_batch = np.clip(
            np.asarray(grad, dtype=float) + prior_probs[start:stop], 0.0, 1.0
        )
        loglik_parts.append(float(value))
        posterior_parts.append(posterior_batch)

    posterior_padded = np.concatenate(posterior_parts, axis=0)
    predicted_ms2 = predict_from_load_probs(posterior_padded)
    grid_slice = slice(pad_left, pad_left + original_grid_size)
    posterior = posterior_padded[:, grid_slice]
    return {
        "loglik_sum": float(np.sum(loglik_parts)),
        "posterior_load_prob": posterior,
        "predicted_ms2": predicted_ms2,
        "prior_load_prob": prior_probs[:, grid_slice],
        "window_size": window_size,
        "n_states": n_states,
        "weights": weights,
        "starts": starts,
    }
