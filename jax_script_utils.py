from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def acf_jax(dataset, lags):
    data = jnp.asarray(dataset)
    t_len = data.shape[1]

    def acf_at_lag(lag):
        lag_abs = jnp.abs(lag)
        shifted = jnp.roll(data, shift=-lag, axis=1)
        valid = jnp.arange(t_len) < (t_len - lag_abs)
        valid = valid[None, :]
        return (
            jnp.nanmean(jnp.where(valid, data * shifted, jnp.nan))
            - jnp.nanmean(data) ** 2
        )

    return jax.vmap(acf_at_lag)(lags)


@jax.jit
def cross_corr_jax(dataset1, dataset2, lags):
    data1 = jnp.asarray(dataset1)
    data1 = (data1 - jnp.nanmean(data1, axis=1, keepdims=True)) / jnp.nanstd(
        data1, axis=1, keepdims=True
    )
    data2 = jnp.asarray(dataset2)
    data2 = (data2 - jnp.nanmean(data2, axis=1, keepdims=True)) / jnp.nanstd(
        data2, axis=1, keepdims=True
    )
    t_len = data1.shape[1]

    def cross_corr_at_lag(lag):
        lag_abs = jnp.abs(lag)
        shifted = jnp.roll(data2, shift=-lag, axis=1)
        valid = jnp.where(
            lag >= 0,
            jnp.arange(t_len) < (t_len - lag_abs),
            jnp.arange(t_len) >= lag_abs,
        )
        valid = valid[None, :]
        return jnp.nanmean(jnp.where(valid, data1 * shifted, jnp.nan))

    return jax.vmap(cross_corr_at_lag)(lags)


@partial(jax.jit, static_argnames=("window_size",))
def pcontact_hmm_loglik_batch(
    observed_ms2: jnp.ndarray,
    p_contact_interval: jnp.ndarray,
    kon: float,
    koff: float,
    dt: float,
    load_probs: jnp.ndarray,
    emission_means: jnp.ndarray,
    noise_std: jnp.ndarray,
    window_size: int,
) -> jnp.ndarray:
    """Return track log-likelihoods with shared or per-track MS2 emission params."""
    batch_size = observed_ms2.shape[0]
    n_bit_states = 1 << int(window_size)
    half_states = 1 << (int(window_size) - 1)
    tiny = jnp.finfo(jnp.float32).tiny
    p_off = -jnp.expm1(-jnp.maximum(koff, 0.0) * dt)

    load_probs = jnp.asarray(load_probs, dtype=jnp.float32)
    if load_probs.ndim == 1:
        if load_probs.shape[0] == 2:
            load_probs = jnp.broadcast_to(load_probs[None, :], (batch_size, 2))
        else:
            load_probs = load_probs.reshape(batch_size, 2)

    emission_means = jnp.asarray(emission_means, dtype=jnp.float32)
    if emission_means.ndim == 1:
        if emission_means.shape[0] == n_bit_states:
            emission_means = jnp.broadcast_to(
                emission_means[None, :], (batch_size, n_bit_states)
            )
        else:
            emission_means = emission_means.reshape(batch_size, n_bit_states)

    noise_std = jnp.asarray(noise_std, dtype=jnp.float32)
    if noise_std.ndim == 0:
        noise_std = jnp.broadcast_to(noise_std, (batch_size,))
    log_norm = jnp.log(noise_std) + 0.5 * jnp.log(2.0 * jnp.pi)

    def emission_scaled(obs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        valid = jnp.isfinite(obs)
        obs_filled = jnp.where(valid, obs, 0.0)
        residual = (obs_filled[:, None] - emission_means) / noise_std[:, None]
        logp = -0.5 * residual * residual - log_norm[:, None]
        logp = jnp.where(valid[:, None], logp, 0.0)
        shift = jnp.max(logp, axis=1)
        return jnp.exp(logp - shift[:, None]), shift

    def apply_emission(alpha: jnp.ndarray, obs: jnp.ndarray):
        emission, shift = emission_scaled(obs)
        weighted = alpha * emission[:, None, :]
        norm = jnp.maximum(jnp.sum(weighted, axis=(1, 2)), tiny)
        return weighted / norm[:, None, None], jnp.log(norm) + shift

    def append_load(mixed: jnp.ndarray) -> jnp.ndarray:
        by_next_state = []
        for promoter_state in range(2):
            collapsed = (
                mixed[:, promoter_state, :]
                .reshape(batch_size, 2, half_states)
                .sum(axis=1)
            )
            p_load = load_probs[:, promoter_state]
            shifted = jnp.stack(
                [collapsed * (1.0 - p_load[:, None]), collapsed * p_load[:, None]],
                axis=2,
            ).reshape(batch_size, n_bit_states)
            by_next_state.append(shifted)
        return jnp.stack(by_next_state, axis=1)

    alpha = jnp.zeros((batch_size, 2, n_bit_states), dtype=jnp.float32)
    alpha = alpha.at[:, 0, 0].set(1.0)
    alpha, loglik = apply_emission(alpha, observed_ms2[:, 0])
    if p_contact_interval.shape[1] == observed_ms2.shape[1]:
        p_contact_interval = p_contact_interval[:, 1:]

    def step(carry, inputs):
        alpha, loglik = carry
        p_contact_t, obs = inputs
        p_on = -jnp.expm1(-jnp.maximum(kon, 0.0) * jnp.clip(p_contact_t, 0.0, 1.0) * dt)
        mixed_off = (1.0 - p_on)[:, None] * alpha[:, 0, :] + p_off * alpha[:, 1, :]
        mixed_on = p_on[:, None] * alpha[:, 0, :] + (1.0 - p_off) * alpha[:, 1, :]
        mixed = jnp.stack([mixed_off, mixed_on], axis=1)
        predicted = append_load(mixed)
        alpha, obs_loglik = apply_emission(predicted, obs)
        return (alpha, loglik + obs_loglik), None

    (_, loglik), _ = jax.lax.scan(
        step,
        (alpha, loglik),
        (p_contact_interval.T, observed_ms2[:, 1:].T),
    )
    return loglik


def state_bits(window_size: int) -> np.ndarray:
    states = np.arange(1 << int(window_size), dtype=np.uint32)
    shifts = np.arange(int(window_size) - 1, -1, -1, dtype=np.uint32)
    return ((states[:, None] >> shifts[None, :]) & 1).astype(np.float32)


def ms2_kernel_weights(
    dt: float, t_rise: float, t_plateau: float, alpha: float
) -> np.ndarray:
    support = float(t_rise + t_plateau)
    window_size = int(support / float(dt)) + 1
    ages = np.arange(window_size, dtype=np.float64) * float(dt)
    kernel = ((ages >= 0.0) & (ages <= support)).astype(np.float64) * (
        (ages / float(t_rise)) * (ages <= float(t_rise)) + (ages > float(t_rise))
    )
    return (float(alpha) * kernel[::-1]).astype(np.float32)
