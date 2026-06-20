import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple
from tqdm.auto import tqdm


@jax.jit
def Proximal_MS2_kernel(t, t1, t2, Imax):
    """
    Compute the kernel function for the Proximal MS2 model.

    Parameters
    ----------
    t : float
        Time point at which to evaluate the kernel.
    t1 : float
        Time to go through array.
    t2 : float
        Time to go through rest of gene and release RNA.
    Imax : float
        Maximum intensity of the kernel.

    Returns
    -------
    float
        Value of the kernel function at time t.
    """
    rise_term = jnp.heaviside(t, 0) * jnp.heaviside(t1 - t, 1) * t * Imax / t1
    plateau_term = jnp.heaviside(t - t1, 0) * jnp.heaviside(t1 + t2 - t, 1) * Imax
    return rise_term + plateau_term


def get_ss(Q: np.ndarray, acts_on: str = "col") -> np.ndarray:
    """Compute steady-state distribution of a CTMC generator.

    Parameters
    ----------
    Q : np.ndarray
        Generator matrix of shape ``(n, n)``.
    acts_on : str, default="col"
        If "col", solve ``Q @ π = 0``; if "row", solve ``Q.T @ π = 0``.

    Returns
    -------
    np.ndarray
        Stationary distribution ``π`` of shape ``(n,)``.
    """
    n = Q.shape[0]
    A = Q if acts_on == "col" else Q.T  # A @ pi = 0
    # Constrained least squares: [A; 1^T] pi = [0; 1]
    M = np.vstack([A, np.ones((1, n))])
    # b = np.zeros((n + 1,)).at[-1].set(1.0)
    b = np.zeros((n + 1,))
    b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(M, b, rcond=None)

    # Numerical hygiene: nonnegative + renormalize (keeps JIT-friendly ops)
    pi = np.clip(pi, 0.0, np.inf)
    pi = pi / np.sum(pi)
    return pi


@jax.jit
def get_ss_jax(Q: jnp.ndarray, acts_on: str = "col") -> jnp.ndarray:
    """JAX version of :func:`get_ss` for computing CTMC steady state."""
    n = Q.shape[0]
    A = Q if acts_on == "col" else Q.T  # A @ pi = 0
    # Constrained least squares: [A; 1^T] pi = [0; 1]
    M = jnp.vstack([A, jnp.ones((1, n))])
    b = jnp.zeros((n + 1,)).at[-1].set(1.0)
    pi, *_ = jnp.linalg.lstsq(M, b, rcond=None)

    # Numerical hygiene: nonnegative + renormalize (keeps JIT-friendly ops)
    pi = jnp.clip(pi, 0.0, jnp.inf)
    pi = pi / jnp.sum(pi)
    return pi


def sample_promoter_states(Transition_matrix: np.ndarray, T: int, seed: int):
    """Sample promoter state sequence via Gillespie simulation.

    Parameters
    ----------
    Transition_matrix : np.ndarray
        CTMC generator with columns summing to zero (shape ``(n, n)``).
    T : int
        Maximum simulation time.
    seed : int
        RNG seed.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays of states and corresponding transition times.
    """
    np.random.seed(seed)
    ss_dist = np.array(get_ss(Transition_matrix))
    # renormalize
    ss_dist = ss_dist / np.sum(ss_dist)

    # initialize state sequence
    current_state = np.random.choice(np.arange(Transition_matrix.shape[0]), p=ss_dist)
    states = [current_state]
    times = [0]
    t = 0
    while t < T:
        exit_rate = -Transition_matrix[current_state, current_state]
        wait_time = np.random.exponential(1 / exit_rate)
        if t + wait_time > T:
            break
        t += wait_time

        tmat_row = Transition_matrix[:, current_state]
        out_rates = tmat_row[np.arange(Transition_matrix.shape[0]) != current_state]
        state_candidates = np.arange(Transition_matrix.shape[0])[
            np.arange(Transition_matrix.shape[0]) != current_state
        ]
        new_state = np.random.choice(state_candidates, p=out_rates / np.sum(out_rates))
        current_state = new_state
        states.append(current_state)
        times.append(t)
    return np.array(states), np.array(times)


def sample_loading_events(
    times: np.ndarray,
    states: np.ndarray,
    state_vals: np.ndarray,
    T_max: float,
    seed: int | None = None,
) -> np.ndarray:
    """Generate Poisson loading events given a CTMC state trajectory.

    Parameters
    ----------
    times : np.ndarray
        State transition times, shape ``(n_transitions,)``.
    states : np.ndarray
        State sequence, shape ``(n_transitions,)``.
    state_vals : np.ndarray
        Loading rate per state, shape ``(nstates,)``.
    T_max : float
        Maximum simulation time.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Sorted loading event times.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate the number of events in each segment
    state_durations = np.diff(np.append(times, T_max))
    state_integrated_rates = state_vals[states] * state_durations
    event_numbers = np.random.poisson(state_integrated_rates)

    # Generate a uniform number for each event (handling case if no events)
    try:
        uniforms = np.random.uniform(0, 1, (np.sum(event_numbers),))
    except Exception as e:
        print(
            np.sum(event_numbers),
            event_numbers,
            state_integrated_rates,
            state_durations,
        )

    # Generate the time of each event
    event_labels = np.repeat(np.arange(len(event_numbers)), event_numbers)
    event_times = times[event_labels] + state_durations[event_labels] * uniforms
    return np.sort(event_times)


def generate_MS2_signal(
    sampling_times: np.ndarray,
    loading_times: np.ndarray,
    kernel_function: callable,
    noise: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate MS2 fluorescence trace from loading events and kernel.

    Parameters
    ----------
    sampling_times : np.ndarray
        Observation times, shape ``(ntimes,)``.
    loading_times : np.ndarray
        Loading event times, shape ``(nevents,)``.
    kernel_function : callable
        Function mapping time offsets to intensity contributions.
    noise : float
        Gaussian noise standard deviation.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Noise-free and noisy MS2 signals, each shape ``(ntimes,)``.
    """
    noise_array = np.random.normal(0, noise, size=sampling_times.shape)
    signal = kernel_function(sampling_times[:, None] - loading_times[None, :]).sum(
        axis=-1
    )

    return signal, signal + noise_array


def generate_MS2_trajectory(
    tmat: np.ndarray,
    Tmax: float,
    sampling_times: np.ndarray,
    noise: float,
    kernel: callable,
    loading_rates: np.ndarray,
    kernel_T: float,
    seed: int,
):
    """Simulate a single MS2 trajectory from CTMC dynamics and loading events.

    Parameters
    ----------
    tmat : np.ndarray
        CTMC generator, shape ``(nstates, nstates)``.
    Tmax : float
        Maximum observation time.
    sampling_times : np.ndarray
        Observation times, shape ``(ntimes,)``.
    noise : float
        Gaussian noise standard deviation.
    kernel : callable
        Kernel function mapping time offsets to fluorescence.
    loading_rates : np.ndarray
        Loading rates per state, shape ``(nstates,)``.
    kernel_T : float
        Kernel support duration.
    seed : int
        RNG seed.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        State times, state sequence, loading events, clean MS2 trace, noisy trace.
    """

    Tmax_w_pad = Tmax + kernel_T
    states, times = sample_promoter_states(tmat, Tmax_w_pad, seed)

    events = sample_loading_events(times, states, loading_rates, Tmax_w_pad)

    ms2_signal, noisy_signal = generate_MS2_signal(
        sampling_times=sampling_times + kernel_T,
        loading_times=events,
        kernel_function=kernel,
        noise=noise,
    )

    return times, states, events, ms2_signal, noisy_signal


def sample_dataset(
    parameter_dict: dict, verbose: bool = True, seed: int | None = None
) -> Tuple[list[np.ndarray], list[tuple]]:
    """Generate a dataset of simulated MS2 trajectories.

    Parameters
    ----------
    parameter_dict : dict
        Dictionary containing CTMC, kernel, and simulation parameters.
    verbose : bool, default=True
        If True, show a progress bar.
    seed : int | None, optional
        Random seed for reproducible dataset generation.

    Returns
    -------
    tuple[list[np.ndarray], list[tuple]]
        Observed noisy trajectories and corresponding hidden simulation details.
    """
    tmat = parameter_dict["tmat"]
    Tmax = parameter_dict["Tmax"]
    sampling_times = parameter_dict["sampling_times"]
    noise = parameter_dict["noise"]
    kernel = parameter_dict["kernel"]
    loading_rates = parameter_dict["loading_rates"]
    T_rise = parameter_dict["T_rise"]
    T_plateau = parameter_dict["T_plateau"]
    ntraj = parameter_dict["ntraj"]
    nanfrac = parameter_dict.get("nanfrac", 0.0)
    variation_scale = parameter_dict.get(
        "loading_variation_scale", np.zeros_like(loading_rates) + 1e-9
    )
    rng = np.random.default_rng(seed)

    # if variation_scale > 0:
    shape = np.ones_like(loading_rates) / (variation_scale**2)
    scale = loading_rates * (variation_scale**2)

    loading_rates = rng.gamma(shape, scale, size=(ntraj, len(loading_rates)))
    # else:
    #     loading_rates = np.tile(loading_rates, (ntraj, 1))
    observed_dataset = []
    hidden_dataset = []
    if verbose:
        iterator = tqdm(list(range(ntraj)))
    else:
        iterator = list(range(ntraj))
    for i in iterator:
        traj_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        times, states, events, ms2_signal, noisy_signal = generate_MS2_trajectory(
            tmat,
            Tmax,
            sampling_times,
            noise,
            kernel,
            loading_rates[i],
            T_rise + T_plateau,
            traj_seed,
        )

        # introduce nans
        nan_indices = rng.choice(
            len(sampling_times), size=int(nanfrac * len(sampling_times)), replace=False
        )
        noisy_signal = np.array(noisy_signal)
        noisy_signal[nan_indices] = np.nan
        # introduce random lengths
        length = int(rng.integers(len(sampling_times) // 2, len(sampling_times)))
        noisy_signal[length:] = np.nan

        observed_dataset.append(noisy_signal)
        hidden_dataset.append((times, states, events, ms2_signal))
    return observed_dataset, hidden_dataset
