from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from simulation_utils import _sample_loading_events, _sample_promoter_states




def _build_ms2_signal(
    observation_times: np.ndarray,
    loading_times: np.ndarray,
    t_increase: float,
    t_plateau: float,
    alpha: float,
    ms2_noise: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n_obs = observation_times.shape[0]
    signal = np.zeros(n_obs, dtype=np.float64)
    for event_time in loading_times:
        for i in range(n_obs):
            dt = observation_times[i] - event_time
            if dt < 0.0:
                continue
            if dt <= t_increase:
                signal[i] += alpha * dt / t_increase
            elif dt <= t_increase + t_plateau:
                signal[i] += alpha
    noisy = np.empty(n_obs, dtype=np.float64)
    for i in range(n_obs):
        noisy[i] = signal[i] + ms2_noise * np.random.normal()
    return signal, noisy

def _simulate_fromR(
    epfunc: callable,
    kon: float,
    koff: float,
    t_max: float,
    rc: float,
    pol2_loading_rate: np.ndarray,
    alpha: float,
    ms2_noise: float,
    t_increase: float,
    t_plateau: float,
    activation_delay: float,
    p_contact: float,
    measurement_interval: float,
):
    n_events = np.random.poisson(kon * t_max)
    on_times = np.empty(n_events, dtype=np.float64)
    for i in range(n_events):
        on_times[i] = t_max * np.random.random()
    if n_events > 1:
        on_times.sort()

    observation_times = np.arange(
        0.0, t_max + measurement_interval, measurement_interval
    )
    n_obs = observation_times.shape[0]

    merged_times = np.concatenate((observation_times, on_times))
    merged_is_obs = np.concatenate(
        (np.ones(n_obs, dtype=np.uint8), np.zeros(n_events, dtype=np.uint8))
    )
    if merged_times.shape[0] > 1:
        order = np.argsort(merged_times, kind="mergesort")
        merged_times = merged_times[order]
        merged_is_obs = merged_is_obs[order]
    event_r = epfunc(on_times)
    thinned_on_times = on_times[event_r < rc] + activation_delay
    thinned_on_times = thinned_on_times[thinned_on_times <= t_max]
    if thinned_on_times.shape[0] > 1:
        thinned_on_times.sort()

    state_sequence, state_times = _sample_promoter_states(
        p_contact=p_contact,
        kon=kon,
        koff=koff,
        t_max=t_max,
        thinned_on_times=thinned_on_times,
        nstates=2,
    )
    loading_times = _sample_loading_events(
        times=state_times,
        states=state_sequence,
        state_vals=pol2_loading_rate,
        t_max=t_max,
    )
    ms2_signal, ms2_noisy = _build_ms2_signal(
        observation_times=observation_times,
        loading_times=loading_times,
        t_increase=t_increase,
        t_plateau=t_plateau,
        alpha=alpha,
        ms2_noise=ms2_noise,
    )
    return observation_times, ms2_signal, ms2_noisy, loading_times
