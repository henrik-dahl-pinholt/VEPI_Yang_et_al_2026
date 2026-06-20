from __future__ import annotations

from typing import Tuple

import numpy as np


def _sample_loading_events(
    times: np.ndarray,
    states: np.ndarray,
    state_vals: np.ndarray,
    t_max: float,
) -> np.ndarray:
    n_segments = states.shape[0]
    event_counts = np.zeros(n_segments, dtype=np.int64)
    n_events = 0
    for i in range(n_segments):
        t0 = times[i]
        t1 = t_max if i == n_segments - 1 else times[i + 1]
        duration = t1 - t0
        lam = state_vals[states[i]] * duration
        event_counts[i] = np.random.poisson(lam)
        n_events += event_counts[i]
    out = np.empty(n_events, dtype=np.float64)
    cursor = 0
    for i in range(n_segments):
        t0 = times[i]
        t1 = t_max if i == n_segments - 1 else times[i + 1]
        duration = t1 - t0
        draw = event_counts[i]
        for _ in range(draw):
            out[cursor] = t0 + duration * np.random.random()
            cursor += 1
    if n_events > 1:
        out.sort()
    return out


def _sample_promoter_states(
    p_contact: float,
    kon: float,
    koff: float,
    t_max: float,
    thinned_on_times: np.ndarray,
    nstates: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    weights = np.empty(nstates, dtype=np.float64)
    norm = 0.0
    ratio = p_contact * kon / koff
    for i in range(nstates):
        weights[i] = ratio**i
        norm += weights[i]
    for i in range(nstates):
        weights[i] /= norm

    u0 = np.random.random()
    current_state = 0
    cdf = 0.0
    for i in range(nstates):
        cdf += weights[i]
        if u0 <= cdf:
            current_state = i
            break

    n_off_events = np.random.poisson(koff * t_max)
    off_times = np.empty(n_off_events, dtype=np.float64)
    for i in range(n_off_events):
        off_times[i] = t_max * np.random.random()
    if n_off_events > 1:
        off_times.sort()

    on_types = np.ones(thinned_on_times.shape[0], dtype=np.int64)
    off_types = -np.ones(n_off_events, dtype=np.int64)
    all_times = np.concatenate((thinned_on_times, off_times))
    all_types = np.concatenate((on_types, off_types))
    if all_times.shape[0] > 1:
        order = np.argsort(all_times)
        all_times = all_times[order]
        all_types = all_types[order]

    state_sequence = np.empty(all_times.shape[0] + 1, dtype=np.int64)
    state_times = np.empty(all_times.shape[0] + 1, dtype=np.float64)
    state_sequence[0] = current_state
    state_times[0] = 0.0
    n_states = 1
    current_time = 0.0

    while current_time < t_max and all_times.shape[0] > 0:
        if current_state == 0:
            relevant_times = thinned_on_times
            relevant_types = on_types
        elif current_state == nstates - 1:
            relevant_times = off_times
            relevant_types = off_types
        else:
            relevant_times = all_times
            relevant_types = all_types
        if relevant_times.shape[0] == 0 or current_time >= relevant_times[-1]:
            break
        idx = np.searchsorted(relevant_times, current_time, side="left")
        if idx >= relevant_times.shape[0]:
            break
        next_time = relevant_times[idx]
        current_state += relevant_types[idx]
        state_sequence[n_states] = current_state
        state_times[n_states] = next_time
        current_time = next_time
        n_states += 1

    return state_sequence[:n_states], state_times[:n_states]
