import jax
from jax import numpy as jnp
from utils import find_previous, find_next

vmap_expm_two_ax = jax.vmap(jax.vmap(jax.scipy.linalg.expm, in_axes=0), in_axes=0)
vmap_expm_one_ax = jax.vmap(jax.scipy.linalg.expm, in_axes=0)


def _safe_log_matrix(matrix):
    return jnp.log(jnp.clip(matrix, jnp.finfo(matrix.dtype).tiny, None))


@jax.jit
def Do_forward_pass(
    Q_tilde: jnp.ndarray, f_mats: jnp.ndarray, dt: float, init_cond: jnp.ndarray
) -> jnp.ndarray:
    """Forward pass for tilted CTMC in log space.

    Parameters
    ----------
    Q_tilde : jnp.ndarray
        Generator matrices, shape ``(batch, ntimes, nstates, nstates)`` or broadcastable.
    f_mats : jnp.ndarray
        Tilt matrices matching ``Q_tilde``.
    dt : float
        Time step.
    init_cond : jnp.ndarray
        Initial state distribution, shape ``(batch, nstates)``.

    Returns
    -------
    jnp.ndarray
        Forward log-messages over time, shape ``(ntimes+1, batch, nstates)``.
    """
    matrices = Q_tilde - f_mats

    out = vmap_expm_two_ax(matrices * dt)  # (batch_size, n_times, n_states, n_states)

    @jax.jit
    def scan_fn(carry, matrix):
        after_carry = jax.scipy.special.logsumexp(
            _safe_log_matrix(matrix) + carry[:, None, :], axis=-1
        )  # Shape (batch_size, nstates)
        return after_carry, carry

    final_output, out_traj = jax.lax.scan(
        scan_fn,
        jnp.log(init_cond),
        out.transpose(1, 0, 2, 3),
    )
    return jnp.concatenate([out_traj, final_output[None, :]])


@jax.jit
def Do_backward_pass(
    Q_tilde: jnp.ndarray, f_mats: jnp.ndarray, dt: float
) -> jnp.ndarray:
    """Backward pass for tilted CTMC in log space.

    Parameters
    ----------
    Q_tilde : jnp.ndarray
        Generator matrices, shape ``(batch, ntimes, nstates, nstates)`` or broadcastable.
    f_mats : jnp.ndarray
        Tilt matrices matching ``Q_tilde``.
    dt : float
        Time step.

    Returns
    -------
    jnp.ndarray
        Backward log-messages over time (reversed), shape ``(ntimes+1, batch, nstates)``.
    """
    matrices = Q_tilde.swapaxes(-1, -2) - f_mats

    out = vmap_expm_two_ax(matrices * dt)

    init_cond = jnp.ones((len(matrices), Q_tilde.shape[-1]))

    @jax.jit
    def scan_fn(carry, matrix):
        after_carry = jax.scipy.special.logsumexp(
            _safe_log_matrix(matrix) + carry[:, None, :], axis=-1
        )
        return after_carry, carry

    final_output, out_traj = jax.lax.scan(
        scan_fn,
        jnp.log(init_cond),
        out.transpose(1, 0, 2, 3)[::-1],
    )
    return jnp.concatenate([out_traj, final_output[None, :]])


def Compute_partition_function(
    Q_tilde: jnp.ndarray,
    f_mats: jnp.ndarray,
    dt: float,
    init_cond: jnp.ndarray,
    nanmask=None,
) -> jnp.ndarray:
    """Compute log normalizing constant (partition function) for tilted CTMC."""
    out_traj = Do_forward_pass(Q_tilde, f_mats, dt, init_cond)
    if nanmask is not None:
        terminal_rows = jnp.sum(jnp.isfinite(nanmask), axis=0) - 1
        batch = jnp.arange(out_traj.shape[1])
        terminal_out = out_traj[terminal_rows, batch, :]
        return jnp.sum(jax.scipy.special.logsumexp(terminal_out, axis=-1))
    else:
        return jnp.sum(jax.scipy.special.logsumexp(out_traj[-1], -1), axis=0)


@jax.jit
def Interpolate_posterior(t, f_mats, Q_tilde, point_times, fpass, bpass):
    """
    Interpolate the posterior distribution at a time point.

    Parameters
    ----------
    t : float
        Time point at which to interpolate the posterior.
    f_mats : jnp.ndarray
        Array of shape (batch_size, n_times) containing the f matrices.
    Q_tilde : jnp.ndarray
        Generator matrix for the tilted CTMC.
    f_maker : function
        Function to create the f matrix.
    dt : float
        Time step size.
    point_times : jnp.ndarray
        Array of time points where the forward and backward passes were computed.
    fpass : jnp.ndarray
        Forward pass results.
    bpass : jnp.ndarray
        Backward pass results.

    Returns
    -------
    jnp.ndarray
        Interpolated posterior distribution at the given time points.
    """

    fpass_interpolated = Interpolate_forward(t, f_mats, Q_tilde, point_times, fpass)
    bpass_interpolated = Interpolate_backward(t, f_mats, Q_tilde, point_times, bpass)

    interpolated_posterior = fpass_interpolated + bpass_interpolated
    interpolated_posterior = interpolated_posterior - jax.scipy.special.logsumexp(
        interpolated_posterior, axis=1, keepdims=True
    )  # Normalize the posterior distribution

    return jnp.exp(interpolated_posterior)


@jax.jit
def Interpolate_forward(t, f_mats, Q_tilde, point_times, fpass):
    """
    Interpolate the forward distribution at a time point.

    Parameters
    ----------
    t : float
        Time point at which to interpolate the forward.
    f_mats : jnp.ndarray
        Array of shape (batch_size, n_times) containing the f matrices.
    Q_tilde : jnp.ndarray
        Generator matrix for the tilted CTMC.
    f_maker : function
        Function to create the f matrix.
    dt : float
        Time step size.
    point_times : jnp.ndarray
        Array of time points where the forward and backward passes were computed.
    fpass : jnp.ndarray
        Forward pass results.

    Returns
    -------
    jnp.ndarray
        Interpolated forward distribution at the given time points.
    """

    # Get the closest column in fpass and bpass to each time point in times
    fpass_closest, fpass_time = find_previous(
        t, fpass, point_times
    )  # (batch_size, n_states)

    fmat, _ = find_previous(
        t, f_mats.swapaxes(0, 1), point_times
    )  # (batch_size, n_states)
    Qmat, _ = find_previous(
        t, Q_tilde.swapaxes(0, 1), point_times
    )  # (batch_size, n_states)

    dt_fpass = t - fpass_time

    # Generate the matrix exponential for the forward and backward passes
    fpass_mat = Qmat - fmat

    fpass_mat = vmap_expm_one_ax(
        fpass_mat * dt_fpass
    )  # (batch_size, n_states, n_states)

    # Compute the posterior distribution
    fpass_interpolated = jax.scipy.special.logsumexp(
        _safe_log_matrix(fpass_mat) + fpass_closest[:, None, :], axis=-1
    )  # Shape (batch_size, n_states)

    return fpass_interpolated


@jax.jit
def Interpolate_backward(t, f_mats, Q_tilde, point_times, bpass):
    """
    Interpolate the backward distribution at a time point.

    Parameters
    ----------
    t : float
        Time point at which to interpolate the backward.
    f_mats : jnp.ndarray
        Array of shape (batch_size, n_times) containing the f matrices.
    Q_tilde : jnp.ndarray
        Generator matrix for the tilted CTMC.
    f_maker : function
        Function to create the f matrix.
    dt : float
        Time step size.
    point_times : jnp.ndarray
        Array of time points where the forward and backward passes were computed.
    bpass : jnp.ndarray
        Backward pass results.

    Returns
    -------
    jnp.ndarray
        Interpolated backward distribution at the given time points.
    """

    # Get the closest column in fpass and bpass to each time point in times
    bpass_closest, bpass_time = find_next(t, bpass, point_times)
    # (batch_size, n_states)
    fmat, _ = find_next(t, f_mats.swapaxes(0, 1), point_times)  # (batch_size, n_states)
    Qmat = find_next(t, Q_tilde.swapaxes(0, 1), point_times)  # (batch_size, n_states)
    dt_bpass = bpass_time - t

    # Generate the matrix exponential for the forward and backward passes
    bpass_mat = Qmat.swapaxes(-1, -2) - fmat
    bpass_mat = vmap_expm_one_ax(
        bpass_mat * dt_bpass
    )  # (batch_size, n_states, n_states)

    # Compute the posterior distribution
    bpass_interpolated = jax.scipy.special.logsumexp(
        _safe_log_matrix(bpass_mat) + bpass_closest[:, None, :], axis=-1
    )  # Shape (batch_size, n_states)
    return bpass_interpolated


@jax.jit
def Interpolate_joint(t, f_mats, Q_tilde, point_times, fpass, bpass):
    """
    Interpolate the joint posterior distribution at a time point.

    Parameters
    ----------
    t : float
        Time point at which to interpolate the posterior.
    f_mats : jnp.ndarray
        Array of shape (batch_size, n_times) containing the f matrices.
    Q_tilde : jnp.ndarray
        Generator matrix for the tilted CTMC.
    f_maker : function
        Function to create the f matrix.
    dt : float
        Time step size.
    point_times : jnp.ndarray
        Array of time points where the forward and backward passes were computed.
    fpass : jnp.ndarray
        Forward pass results.
    bpass : jnp.ndarray
        Backward pass results.

    Returns
    -------
    jnp.ndarray
        Interpolated posterior distribution at the given time points.
    """

    fpass_interpolated = Interpolate_forward(t, f_mats, Q_tilde, point_times, fpass)
    bpass_interpolated = Interpolate_backward(t, f_mats, Q_tilde, point_times, bpass)
    Q_offdiag = Q_tilde * (1 - jnp.eye(Q_tilde.shape[-1]))[None, :]

    interpolated_posterior = (
        jnp.log(Q_offdiag)
        + fpass_interpolated[:, None, :]
        + bpass_interpolated[:, :, None]
    )
    normalizing_constant = jax.scipy.special.logsumexp(
        fpass[-1], axis=-1, keepdims=True
    )
    normalized_posterior = interpolated_posterior - normalizing_constant[:, :, None]
    return jnp.exp(normalized_posterior)


@jax.jit
def gen_posterior_tmats(Q_tilde, f_mats, bpass, dt):
    """Construct transition matrices for posterior sampling from forward/backward passes."""
    backward_factor_mats = (
        bpass[1:, :, :, None] - bpass[1:, :, None, :]
    )  # (ntimes,batch_size,nstates,nstates)
    backward_factor_mats = backward_factor_mats.swapaxes(
        0, 1
    )  # (batch_size,ntimes,nstates,nstates)

    matrices = (Q_tilde - f_mats) * jnp.exp(backward_factor_mats)
    row_sum = jnp.sum(matrices, axis=2)
    diag_rowsum = jnp.eye(matrices.shape[-1])[None, None, :, :] * row_sum[:, :, :, None]
    matrices = matrices - diag_rowsum  # (batch_size, n_times, n_states, n_states)

    out = vmap_expm_two_ax(matrices * dt)  # (batch_size, n_times, n_states, n_states)
    transposedout = out.transpose(1, 0, 2, 3)  # Transpose to scan over the ntimes axis
    return transposedout


def sample_posterior(Q_tilde, f_mats, bpass, fpass, key, nsamples, dt):
    """Sample posterior state paths using categorical transitions."""

    transposedout = gen_posterior_tmats(Q_tilde, f_mats, bpass, dt)

    vmap_categorical = jax.vmap(jax.random.categorical, in_axes=(0, 0))

    inds = jnp.arange(bpass.shape[1])

    @jax.jit
    def sample_single(key):

        init_prob = (
            bpass[0]
            + fpass[0]
            - jax.scipy.special.logsumexp(bpass[0] + fpass[0], axis=-1, keepdims=True)
        )  # (batch_size, n_states)
        init_state = jax.random.categorical(key, init_prob, axis=-1)  # (batch_size,)

        @jax.jit
        def scan(carry, x):
            prev_state, key = carry
            mat = x
            keys = jax.random.split(key, len(mat))

            next_state = vmap_categorical(keys, jnp.log(mat[inds, :, prev_state[inds]]))
            newkey = jax.random.split(key)[0]
            return (next_state, newkey), next_state

        _, samples = jax.lax.scan(scan, (init_state, key), transposedout)
        return samples

    samples = jax.vmap(sample_single)(jax.random.split(key, nsamples)).transpose(
        2, 0, 1
    )  # (batch_size, nsamples, ntimes)
    return samples


@jax.jit
def most_likely_path(Q_tilde, f_mats, bpass, fpass, dt):
    """
    Compute the most likely (MAP) posterior path for each batch.

    Returns
    -------
    jnp.ndarray
        Integer array of shape ``(batch_size, ntimes+1)`` with MAP state sequences.
    """
    transposedout = gen_posterior_tmats(Q_tilde, f_mats, bpass, dt)  # (T, B, S, S)
    eps = 1e-32
    log_trans = jnp.log(transposedout + eps)  # (T, B, S, S)

    # Initial log-probs (same as in sampler)
    init_log = bpass[0] + fpass[0]
    init_log = init_log - jax.scipy.special.logsumexp(
        init_log, axis=-1, keepdims=True
    )  # (B, S)

    def step(carry, log_trans_t):
        """
        Viterbi step:
        carry: (B, S_prev) – best log-prob ending in each state at current time.
        log_trans_t: (B, S_next, S_prev) – transition probs for this time step.
        """
        prev_log = carry  # (B, S_prev)
        scores = log_trans_t + prev_log[:, None, :]  # (B, S_next, S_prev)
        new_log = jnp.max(scores, axis=-1)  # (B, S_next)
        backptr = jnp.argmax(scores, axis=-1).astype(jnp.int32)  # (B, S_next)
        return new_log, backptr

    # Scan over time to compute best log-probs and backpointers
    final_log, backptrs = jax.lax.scan(step, init_log, log_trans)
    # backptrs: (T, B, S_next)

    T = backptrs.shape[0]  # number of transitions = ntimes
    B = backptrs.shape[1]

    # Choose best final state for each batch
    last_state = jnp.argmax(final_log, axis=-1).astype(jnp.int32)  # (B,)

    # Backtrack to recover full path (length T+1)
    path = jnp.zeros((T + 1, B), dtype=jnp.int32)
    path = path.at[T].set(last_state)

    def backtrack(k, path):
        t = T - 1 - k
        idx = jnp.arange(B)
        prev = backptrs[t, idx, path[t + 1]]  # (B,)
        return path.at[t].set(prev)

    path = jax.lax.fori_loop(0, T, backtrack, path)  # (T+1, B)

    # Return as (batch_size, T+1)
    return path.transpose(1, 0)


class TiltedCTMC:
    """Convenience wrapper for tilted CTMC inference and sampling."""

    def __init__(self, f_mats, transition_matrix, initial_state, time_grid):
        if len(transition_matrix) == 2:
            self.transition_matrix = transition_matrix[None, None, :] - f_mats * 0
        else:
            self.transition_matrix = transition_matrix
        self.initial_state = initial_state
        self.f_mats = f_mats

        try:
            dt = time_grid[1] - time_grid[0]
            self.point_times = time_grid
            self.dt = dt
        except IndexError:
            self.dt = time_grid
            time_grid = jnp.arange(0, f_mats.shape[0] * self.dt, self.dt)

        self.Do_pass()

    def Log_Z(self, nanmask=None):
        """Return log partition function of the tilted CTMC."""
        return Compute_partition_function(
            self.transition_matrix,
            self.f_mats,
            self.dt,
            self.initial_state,
            nanmask=nanmask,
        )

    def Do_pass(self):
        """Recompute forward and backward messages."""

        self.fpass = Do_forward_pass(
            self.transition_matrix, self.f_mats, self.dt, self.initial_state
        )

        self.bpass = Do_backward_pass(self.transition_matrix, self.f_mats, self.dt)[
            ::-1
        ]

    def Posterior(self, t):
        """Interpolate marginal posterior at time ``t``."""
        return Interpolate_posterior(
            t,
            self.f_mats,
            self.transition_matrix,
            self.point_times,
            self.fpass,
            self.bpass,
        )

    def Joint(self, t):
        """Interpolate joint posterior for adjacent times around ``t``."""
        return Interpolate_joint(
            t,
            self.f_mats,
            self.transition_matrix,
            self.point_times,
            self.fpass,
            self.bpass,
        )

    def Update(self, f_mats, transition_matrix, initial_state):
        """Update parameters and recompute messages."""
        self.transition_matrix = transition_matrix
        self.initial_state = initial_state
        self.f_mats = f_mats
        self.Do_pass()

    def get_posterior(self):
        """Return normalized marginal posterior over time grid."""
        pos = self.fpass + self.bpass
        # avg_pos = 0.5*(pos[1:]+pos[:-1])
        avg_pos_norm = pos - jax.scipy.special.logsumexp(
            pos, axis=-1, keepdims=True
        )  # Normalize the posterior distribution

        return jnp.exp(avg_pos_norm)

    def get_joint(self):
        """Return joint posterior over consecutive time steps on the grid."""

        Q_tilde = self.transition_matrix

        log_Q_offdiag = (jnp.log(Q_tilde)).swapaxes(1, 0)  # (n_states,n_states)

        joint = (
            log_Q_offdiag + self.fpass[1:, :, None, :] + self.bpass[1:, :, :, None]
        )  # (ntimes,batch_size,n_states,n_states)
        normalizing_constant = jax.scipy.special.logsumexp(
            self.fpass[-1], axis=-1, keepdims=True
        )  # (batch_size, n_states)

        return jnp.exp(
            joint - normalizing_constant[None, :, :, None]
        )  # (ntimes,batch_size,n_states,n_states)

    def sample_posterior(self, nsamples, seed=0):
        """Sample posterior state paths with specified number of draws."""
        key = jax.random.PRNGKey(seed)
        return sample_posterior(
            self.transition_matrix,
            self.f_mats,
            self.bpass,
            self.fpass,
            key,
            nsamples,
            self.dt,
        )

    def most_likely_path(self):
        """Compute MAP trajectory across the entire grid."""
        return most_likely_path(
            self.transition_matrix, self.f_mats, self.bpass, self.fpass, self.dt
        )
