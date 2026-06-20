import numpy as np
from tqdm import tqdm
from typing import Tuple, List


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
    rise_term = np.heaviside(t, 0) * np.heaviside(t1 - t, 1) * t * Imax / t1
    plateau_term = np.heaviside(t - t1, 0) * np.heaviside(t1 + t2 - t, 1) * Imax
    return rise_term + plateau_term


def Get_eigensystem(N):
    """Generate the eigenvectors and eigenvalues of the spring matrix in the Fourier basis. These are used to convert between physical coordinates and the Rouse modes in which the dynamics are diagonal.

    Parameters
    ----------
    N : int
        The number of beads in the polymer

    Returns
    -------
    Qmat : (N,N-1) array
        The N-1 eigenvectors of the spring matrix in the Fourier basis.
    eigvals : (N-1) array
        The eigenvalues corresponding to the N-1 eigenvectors.
    """
    ii, jj = np.meshgrid(np.arange(1, N + 1), np.arange(2, N + 1))
    Qmat = np.sqrt(2 / N) * np.cos((ii - 1 / 2) * (jj - 1) * np.pi / N).T
    eigvals = 2 * (1 - np.cos((np.pi / N) * (np.arange(2, N + 1) - 1)))
    return Qmat, eigvals


def Get_propagators(N, D, k, dt):
    """Compute the propagator and variance of bead coordinates for a given time step.

    Parameters
    ----------
    N : int
        Number of beads in the polymer
    D : float
        Diffusion constant of the Rouse model
    k : float
        Spring constant of the Rouse model
    dt : float
        Time step for the propagation

    Returns
    -------
    mean_prop : (N,N) jax array
        Propagator for the mean of the Rouse modes
    var_prop : (N,N) jax array
        Propagator for the variance of the Rouse modes
    var_prop_inverse : (N,N) jax array
        Inverse of the propagator for the variance of the Rouse modes
    var_scaler : (N,N) jax array
        Scaling matrix to apply to univariate gaussian samples to sample propagated samples from a deterministic initial condition.
    """
    Qmat, eigvals = Get_eigensystem(N)
    eigexp = np.diag(np.exp(-k * eigvals * dt))
    var_out = np.diag((D / (k * eigvals)) * (1 - np.exp(-2 * k * eigvals * dt)))
    var_out_inv = np.diag(
        1 / ((D / (k * eigvals)) * (1 - np.exp(-2 * k * eigvals * dt)))
    )
    return (
        Qmat @ eigexp @ Qmat.T,
        Qmat @ var_out @ Qmat.T,
        Qmat @ (var_out_inv) @ Qmat.T,
        Qmat @ np.sqrt(var_out),
    )


def sample_loading_events(
    times: np.ndarray,
    states: np.ndarray,
    state_vals: np.ndarray,
    T_max: float,
) -> np.ndarray:
    """
    Generate loading events based on the CTMC simulation output.

    Parameters
    ----------
    times : np.ndarray
        Times of state transitions from the CTMC simulation.
    states : np.ndarray
        Sequence of states visited from the CTMC simulation.
    state_vals : np.ndarray
        Array of values corresponding to each state.
    T_max : float
        Maximum simulation time.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Event times.
    """

    # Generate the number of events in each segment
    state_durations = np.diff(np.append(times, T_max))
    state_integrated_rates = state_vals[states] * state_durations
    event_numbers = np.random.poisson(state_integrated_rates)

    # Generate a uniform number for each event
    uniforms = np.random.uniform(size=(np.sum(event_numbers),), low=0, high=1)

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
    """
    Generate a synthetic MS2 signal based on loading times and a kernel function.

    Parameters
    ----------
    sampling_times : np.ndarray
        Times at which to sample the signal.
    loading_times : np.ndarray
        Times of loading events.
    kernel_function : callable
        Function to compute the kernel for each loading event.
    noise : float
        Standard deviation of Gaussian noise to add to the signal.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Synthetic MS2 signal without noise and with noise, sampled at the specified times.
    """
    noise_array = np.random.normal(0, noise, size=sampling_times.shape)
    signal = kernel_function(sampling_times[:, None] - loading_times[None, :]).sum(
        axis=-1
    )

    return signal, signal + noise_array


def gen_steady_state_samples(n_trajectories, D, k, N, dim):
    gaussian_samples = np.random.normal(size=(n_trajectories, dim, N - 1))
    Qmat, eigvals = Get_eigensystem(N)
    mean_0, covar_0 = np.zeros((n_trajectories, dim, N - 1)), (
        D / k / eigvals
    ) * np.ones((n_trajectories, dim, N - 1))
    samples = (mean_0 + gaussian_samples * np.sqrt(covar_0)) @ Qmat.T
    return samples


def sample_Rouse_trajectories_fixed_interval(
    N, D, k, dim, delta_t, ndatapoints, n_trajectories
):
    mean_prop, cov_prop, _, var_scaler = Get_propagators(N, D, k, delta_t)

    outputs = np.zeros((n_trajectories, dim, ndatapoints, N))
    outputs[:, :, 0] = gen_steady_state_samples(n_trajectories, D, k, N, dim)
    for i in tqdm(range(1, ndatapoints)):
        # Propagate the mean and variance of the Rouse modes
        outputs[:, :, i] = (
            outputs[:, :, i - 1] @ mean_prop.T
            + np.random.normal(size=(n_trajectories, dim, N - 1)) @ var_scaler.T
        )
    return outputs


def Generate_data(key, outputs, w, data_noise, observe_every, delta_t):
    """Generate noisy data from the outputs of the Rouse model.

    Parameters
    ----------
    outputs : (n_trajectories, dim, ndatapoints, N) array
        The outputs of the Rouse model.
    w : (N,) array
        The weights for the end-to-end distance.
    data_noise : float
        The noise level to add to the data.
    observe_every : int
        The frequency at which to observe the data.

    Returns
    -------
    ep_dist : (n_trajectories, ndatapoints//observe_every, 1) array
        The end-to-end distances with noise.
    """
    ndatapoints = outputs.shape[2]
    ep_dist = outputs @ w
    # key, subkey = jax.random.split(key)
    ep_dist_w_noise = (
        ep_dist + np.random.normal(size=ep_dist.shape) * np.sqrt(2) * data_noise
    )
    ep_dist_downsampled = ep_dist[..., ::observe_every]
    ep_dist_w_noise_downsampled = ep_dist_w_noise[..., ::observe_every]

    times_downsampled = np.arange(0, ndatapoints, observe_every) * delta_t
    return (
        ep_dist,
        ep_dist_w_noise,
        ep_dist_downsampled,
        ep_dist_w_noise_downsampled,
        times_downsampled,
    )


def sample_polymer_and_on_events(
    kon, T, N, D, k, dim, w, localization_errors, measurement_interval, rc
):
    """Sample a polymer trajectory and on-events."""
    # sample candidate on-times
    N_events = np.random.poisson(kon * T)
    on_times = np.sort(np.random.uniform(0, T, N_events))

    # Merge with observation times
    observation_times = np.arange(0, T + measurement_interval, measurement_interval)
    merged_times = np.unique(np.concatenate((on_times, observation_times)))

    # sample the polymer trajectory
    outputs = np.zeros((dim, len(merged_times), N))
    outputs[:, 0] = gen_steady_state_samples(1, D, k, N, dim)[0]
    for i in range(len(merged_times) - 1):
        # Calculate the time step
        dt = merged_times[i + 1] - merged_times[i]

        # get the propagator
        mean_prop, cov_prop, _, var_scaler = Get_propagators(N, D, k, dt)

        # Get the next state using the get_next function
        outputs[:, i + 1] = (
            outputs[:, i] @ mean_prop.T
            + np.random.normal(size=(1, dim, N - 1)) @ var_scaler.T
        )

    # convert to ep_separation vector
    ep_sep = outputs @ w

    # separate into observation and on-times
    data_mask = np.isin(merged_times, observation_times)
    event_mask = np.isin(merged_times, on_times)
    ep_sep_obs, ep_sep_at_event = (ep_sep[:, data_mask], ep_sep[:, event_mask])

    # add noise to the observations
    ep_sep_obs_noisy = ep_sep_obs + np.random.normal(
        scale=np.sqrt(2) * localization_errors[:, None], size=ep_sep_obs.shape
    )

    # thin the on-events based on contact
    contact = np.linalg.norm(ep_sep_at_event, axis=0) < rc
    thinned_on_times = on_times[contact]

    return observation_times, ep_sep_obs, ep_sep_obs_noisy, thinned_on_times, on_times


def sample_promoter_states(p_contact, kon, koff, T, thinned_on_times, nstates=2):
    thinned_on_times = np.sort(thinned_on_times)
    if nstates < 2:
        raise ValueError("Number of states must be at least 2.")

    # sample initial condition
    weights = (p_contact * kon / koff) ** np.arange(nstates)
    weights = weights / np.sum(weights)
    # p_on = p_contact * kon / (p_contact * kon + koff)
    # current_state = int(np.random.random() < p_on)  # 0 for off, 1 for on
    current_state = np.random.choice(np.arange(nstates), p=weights)

    # generate the event times and dictionary to hold them
    N_off_events = np.random.poisson(koff * T)
    off_times = np.sort(np.random.uniform(0, T, N_off_events))
    # times = {0: thinned_on_times, 1: off_times}
    concat_times = np.concatenate((thinned_on_times, off_times))

    on_types, off_types = np.ones_like(thinned_on_times).astype(int), -np.ones_like(
        off_times
    ).astype(int)
    concat_type = np.concatenate((on_types, off_types))
    sort_inds = np.argsort(concat_times)

    times = concat_times[sort_inds]
    types = concat_type[sort_inds]

    state_sequence, state_times = [current_state], [0.0]
    current_time = 0.0

    while current_time < T:
        # if len(relevant_times) == 0:
        # relevant_times = times[current_state][times[current_state] >= current_time]
        if current_state == 0:
            relevant_times = thinned_on_times
            relevant_types = on_types
        elif current_state == nstates - 1:
            relevant_times = off_times
            relevant_types = off_types
        else:
            relevant_times = times
            relevant_types = types
        if len(relevant_times) == 0:
            break
        if current_time >= relevant_times[-1]:
            break

        next_ind = np.searchsorted(relevant_times, current_time, side="left")
        next_time = relevant_times[next_ind]
        next_type = relevant_types[next_ind]

        current_state += next_type  # toggle state

        state_sequence.append(current_state)
        state_times.append(next_time)
        current_time = next_time
    state_sequence = np.array(state_sequence)
    state_times = np.array(state_times)

    return state_sequence, state_times


def sample_ep_experiment(
    kon,
    koff,
    T,
    N,
    k,
    D,
    dim,
    rc,
    localization_errors,
    measurement_interval,
    pol2_loading_rate,
    alpha,
    MS2_noise,
    T_increase,
    T_plateau,
    a,
    b,
    seed=1,
):
    np.random.seed(seed)
    w = np.zeros(N)
    w[a] = 1
    w[b] = -1
    observation_times, ep_sep_obs, ep_sep_obs_noisy, thinned_on_times, on_times = (
        sample_polymer_and_on_events(
            kon=kon,
            T=T,
            N=N,
            D=D,
            k=k,
            dim=dim,
            w=w,
            localization_errors=localization_errors,
            measurement_interval=measurement_interval,
            rc=rc,
        )
    )
    # compute contact probability
    p_contact = np.mean(
        np.linalg.norm(gen_steady_state_samples(10_000, D, k, N, dim) @ w, axis=-1) < rc
    )
    state_sequence, state_times = sample_promoter_states(
        p_contact=p_contact,
        kon=kon,
        koff=koff,
        T=T,
        thinned_on_times=thinned_on_times,
    )

    # generate the pol2 loading events
    pol2_loading_events = sample_loading_events(
        times=state_times, states=state_sequence, state_vals=pol2_loading_rate, T_max=T
    )

    # generate the MS2 signal
    def kernel(t):
        return Proximal_MS2_kernel(t, T_increase, T_plateau, alpha)

    MS2_signal, MS2_signal_noisy = generate_MS2_signal(
        sampling_times=observation_times,
        loading_times=pol2_loading_events,
        kernel_function=kernel,
        noise=MS2_noise,
    )
    return (
        observation_times,
        ep_sep_obs,
        ep_sep_obs_noisy,
        thinned_on_times,
        on_times,
        state_sequence,
        state_times,
        MS2_signal_noisy,
        pol2_loading_events,
        MS2_signal,
    )
