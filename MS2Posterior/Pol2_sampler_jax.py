from typing import Tuple
import jax
import jax.numpy as jnp


@jax.jit
def setup_sampler(
    time_grid: jnp.ndarray, rate_vals: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Precompute CDFs and integrated rates on a temporal grid.

    Parameters
    ----------
    time_grid : jnp.ndarray
        Monotonic grid of candidate event times, shape ``(ngrid,)``.
    rate_vals : jnp.ndarray
        Loading rates evaluated on ``time_grid``, shape ``(ntraj, ngrid)``.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        Rate CDFs per trajectory (``(ntraj, ngrid)``) and integrated rates ``∫λ dt`` (``(ntraj,)``).
    """

    rate_sum = rate_vals.sum(-1)
    # Guard against degenerate inputs
    rate_pdf = rate_vals / rate_sum[:, None]  # (ntraj, ngrid)
    rate_cdf = jnp.cumsum(rate_pdf, axis=-1)

    # set last entry to 1 to avoid numerical issues
    rate_cdf = rate_cdf.at[:, -1].set(1.0)

    # This is now the exact integral on the grid (used below in acceptance ratios)
    dt = time_grid[1] - time_grid[0]
    int_loading_rates = rate_sum * dt
    return rate_cdf, int_loading_rates


vmap_search = jax.vmap(
    lambda cdf, u: jax.numpy.searchsorted(cdf, u, side="left"), in_axes=(0, 0)
)
vmap_search_constant_ref = jax.vmap(
    lambda cdf, u: jax.numpy.searchsorted(cdf, u, side="left"), in_axes=(None, 0)
)


@jax.jit
def proposal_gen(
    key: jax.Array, time_grid: jnp.ndarray, cdfs: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample candidate event times via inverse-CDF on the grid.

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    time_grid : jnp.ndarray
        Grid of times, shape ``(ngrid,)``.
    cdfs : jnp.ndarray
        Precomputed CDFs per trajectory, shape ``(ntraj, ngrid)``.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        Sampled times and their grid indices, each shape ``(ntraj,)``.
    """
    # inverse-CDF sampling on the precomputed grid (no while-loop)
    u = jax.random.uniform(key, shape=(cdfs.shape[0],))
    i = vmap_search(cdfs, u)

    return time_grid[i], i


@jax.jit
def support_inds(
    x: jnp.ndarray, sampling_times: jnp.ndarray, arange: jnp.ndarray
) -> jnp.ndarray:
    """Indices where the kernel centered at ``x`` overlaps the sampling grid.

    Assumes a causal kernel with support ``[0, S_support]``.

    Parameters
    ----------
    x : jnp.ndarray
        Event times, shape ``(ntraj,)``.
    sampling_times : jnp.ndarray
        Observation times, shape ``(ntimes,)``.
    arange : jnp.ndarray
        Integer offsets covering kernel support, shape ``(nsupport,)``.

    Returns
    -------
    jnp.ndarray
        Indices into ``sampling_times`` of overlapping points, shape ``(ntraj, nsupport)``.
    """
    i0 = vmap_search_constant_ref(sampling_times, x)  # (ntraj)
    return i0[:, None] + arange[None, :]


@jax.jit
def MS2_kernel(x: jnp.ndarray, T_rise: float, S_support: float) -> jnp.ndarray:
    """Piecewise-linear MS2 kernel (rise then plateau)."""
    return (x >= 0) * (x <= S_support) * ((x / T_rise) * (x <= T_rise) + (x > T_rise))


@jax.jit
def add_point_update(
    x: jnp.ndarray,
    ind: jnp.ndarray,
    S: jnp.ndarray,
    arange: jnp.ndarray,
    count_grid: jnp.ndarray,
    log_prior_sum: jnp.ndarray,
    T_rise: float,
    S_support: float,
    RNA_int: jnp.ndarray | float,
    sampling_times: jnp.ndarray,
    signs: jnp.ndarray,
    rate_at_x: jnp.ndarray,
):
    """Incrementally update counts, overlap field, and prior term for a birth/death move.

    Parameters
    ----------
    x : jnp.ndarray
        Event times, shape ``(ntraj,)``.
    ind : jnp.ndarray
        Grid indices corresponding to ``x``.
    S : jnp.ndarray
        Current overlap field, shape ``(ntraj, ntimes)``.
    arange : jnp.ndarray
        Integer offsets covering kernel support.
    count_grid : jnp.ndarray
        Event counts on fine grid, shape ``(ntraj, ngrid)``.
    log_prior_sum : jnp.ndarray
        Accumulated log prior contributions per trajectory, shape ``(ntraj,)``.
    T_rise, S_support, RNA_int, sampling_times
        Kernel and sampling parameters.
    signs : jnp.ndarray
        +1 for birth, -1 for death per trajectory, shape ``(ntraj,)``.
    rate_at_x : jnp.ndarray
        Loading rate at sampled locations, shape ``(ntraj,)``.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        Updated count grid, overlap field, and log prior sum.
    """
    indexer = jnp.arange(S.shape[0])[:, None]

    # update count grid
    count_grid = count_grid.at[jnp.arange(count_grid.shape[0]), ind].add(signs)
    # update overlap field
    inds = support_inds(x, sampling_times, arange)
    t_slice = sampling_times[inds]  # (ntraj, nsupport)
    rna_scale = jnp.broadcast_to(RNA_int, (S.shape[0],))
    Kx = MS2_kernel(t_slice - x[:, None], T_rise, S_support) * rna_scale[:, None]

    S = S.at[indexer, inds].add(signs[:, None] * Kx)
    log_prior_sum += jnp.log(rate_at_x + 1e-9) * signs
    return count_grid, S, log_prior_sum


@jax.jit
def Papangelou(
    x: jnp.ndarray,
    loading_rates: jnp.ndarray,
    sampling_times: jnp.ndarray,
    A: jnp.ndarray,
    M: jnp.ndarray,
    S: jnp.ndarray,
    T_rise: float,
    S_support: float,
    RNA_int: jnp.ndarray | float,
    beta: float,
    arange: jnp.ndarray,
):
    """Compute Papangelou conditional intensity for each trajectory."""
    # compute kernel support
    inds = support_inds(x, sampling_times, arange)
    t_slice = sampling_times[inds]  # (ntraj, nsupport)
    rna_scale = jnp.broadcast_to(RNA_int, (S.shape[0],))
    Kx = MS2_kernel(t_slice - x[:, None], T_rise, S_support) * rna_scale[:, None]
    indexer = jnp.arange(A.shape[0])[:, None]
    M_indexed = M[indexer, inds]
    S_indexed = S[indexer, inds]
    AK = jnp.vecdot(A[indexer, inds], Kx)
    MK2 = jnp.vecdot(M_indexed, Kx * Kx)

    # compute h and J fields
    h_val = AK - MK2 * 0.5
    J_val = jnp.vecdot(M_indexed * S_indexed, Kx)

    return loading_rates * jnp.exp(beta * (h_val - J_val))


@jax.jit
def do_birth_move(
    key: jax.Array,
    Count_grid: jnp.ndarray,
    log_prior_sum: jnp.ndarray,
    S: jnp.ndarray,
    rates_on_grid: jnp.ndarray,
    rate_cdfs: jnp.ndarray,
    sampling_times: jnp.ndarray,
    T_rise: float,
    S_support: float,
    RNA_int: jnp.ndarray | float,
    int_loading_rates: jnp.ndarray,
    arange: jnp.ndarray,
    A: jnp.ndarray,
    M: jnp.ndarray,
    fine_grid: jnp.ndarray,
    beta: float,
):
    """Perform a birth move in the reversible-jump sampler."""

    samples, inds = proposal_gen(key, fine_grid, rate_cdfs)  # (ntraj,)
    rates_at_samps = rates_on_grid[jnp.arange(len(samples)), inds]  # (ntraj,)
    papangelou_vals = Papangelou(
        samples,
        rates_at_samps,
        sampling_times,
        A,
        M,
        S,
        T_rise,
        S_support,
        RNA_int,
        beta=beta,
        arange=arange,
    )  # (ntraj,)
    n = jnp.sum(Count_grid, axis=1)

    birth_density = rates_at_samps / int_loading_rates
    alphas = jnp.minimum(1.0, papangelou_vals / (n + 1.0) / birth_density)
    key, _ = jax.random.split(key)
    accepted = jax.random.uniform(key, shape=alphas.shape) < alphas

    Count_grid, S, log_prior_sum = add_point_update(
        samples,
        inds,
        S,
        arange,
        Count_grid,
        log_prior_sum,
        T_rise,
        S_support,
        RNA_int,
        sampling_times,
        signs=accepted,
        rate_at_x=rates_at_samps,
    )
    return Count_grid, S, log_prior_sum


vmap_ind_choice = jax.vmap(
    lambda key, n, p: jax.random.choice(key, n, p=p),
    in_axes=(0, None, 0),
)


@jax.jit
def sample_death_inds(key, Count_grid):
    """Sample grid indices for death moves proportional to current counts."""
    count_sums = jnp.sum(Count_grid, axis=1)
    ps = Count_grid / count_sums[:, None]
    keys = jax.random.split(key, ps.shape[0])
    index_choices = vmap_ind_choice(keys, ps.shape[1], ps)
    return index_choices  # (ntraj,)


@jax.jit
def do_death_move(
    key: jax.Array,
    Count_grid: jnp.ndarray,
    log_prior_sum: jnp.ndarray,
    S: jnp.ndarray,
    rates_on_grid: jnp.ndarray,
    rate_cdfs: jnp.ndarray,
    sampling_times: jnp.ndarray,
    T_rise: float,
    S_support: float,
    RNA_int: jnp.ndarray | float,
    int_loading_rates: jnp.ndarray,
    arange: jnp.ndarray,
    A: jnp.ndarray,
    M: jnp.ndarray,
    fine_grid: jnp.ndarray,
    beta: float,
):
    """Perform a death move in the reversible-jump sampler."""

    n = jnp.sum(Count_grid, axis=1)
    death_inds = sample_death_inds(key, Count_grid)
    remove_coord = fine_grid[death_inds]
    # remove point to be able to eval Papangelou with particle removed
    signs = -jnp.ones_like(death_inds) * (n > 0)  # only remove if n>0
    rate_at_x = rates_on_grid[jnp.arange(len(death_inds)), death_inds]
    Count_grid, S, log_prior_sum = add_point_update(
        remove_coord,
        death_inds,
        S,
        arange,
        Count_grid,
        log_prior_sum,
        T_rise,
        S_support,
        RNA_int,
        sampling_times,
        signs=signs,
        rate_at_x=rate_at_x,
    )
    Papangelou_vals = Papangelou(
        remove_coord,
        rate_at_x,
        sampling_times,
        A,
        M,
        S,
        T_rise,
        S_support,
        RNA_int,
        beta=beta,
        arange=arange,
    )  # (ntraj,)
    birth_density = (
        rates_on_grid[jnp.arange(len(death_inds)), death_inds] / int_loading_rates
    )

    alpha = jnp.minimum(1.0, birth_density * n / Papangelou_vals)
    key, _ = jax.random.split(key)
    u = jax.random.uniform(key, shape=alpha.shape)
    revert = (u > alpha) * (n > 0)  # only revert if n>0

    Count_grid, S, log_prior_sum = add_point_update(
        remove_coord,
        death_inds,
        S,
        arange,
        Count_grid,
        log_prior_sum,
        T_rise,
        S_support,
        RNA_int,
        sampling_times,
        signs=revert,
        rate_at_x=rate_at_x,
    )
    return Count_grid, S, log_prior_sum


@jax.jit
def do_move(
    key: jax.Array,
    Count_grid: jnp.ndarray,
    log_prior_sum: jnp.ndarray,
    S: jnp.ndarray,
    rates_on_grid: jnp.ndarray,
    rate_cdfs: jnp.ndarray,
    sampling_times: jnp.ndarray,
    T_rise: float,
    S_support: float,
    RNA_int: jnp.ndarray | float,
    int_loading_rates: jnp.ndarray,
    arange: jnp.ndarray,
    A: jnp.ndarray,
    M: jnp.ndarray,
    fine_grid: jnp.ndarray,
    beta: float,
):
    """Randomly choose a birth or death move and execute it."""

    do_birth = jax.random.bernoulli(key, p=0.5)
    newkey, _ = jax.random.split(key)
    return jax.lax.cond(
        do_birth,
        lambda args: do_birth_move(*args),
        lambda args: do_death_move(*args),
        (
            newkey,
            Count_grid,
            log_prior_sum,
            S,
            rates_on_grid,
            rate_cdfs,
            sampling_times,
            T_rise,
            S_support,
            RNA_int,
            int_loading_rates,
            arange,
            A,
            M,
            fine_grid,
            beta,
        ),
    )


@jax.jit
def Update_mean(
    mean: jnp.ndarray, new_sample: jnp.ndarray, count: int, start_iter: int
) -> jnp.ndarray:
    """Online mean update starting after a burn-in iteration."""
    update = count >= start_iter
    new_mean = mean + (new_sample - mean) / (count - start_iter + 1)
    return jax.lax.cond(update, lambda _: new_mean, lambda _: mean, operand=None)


@jax.jit
def Energy(
    AA: jnp.ndarray, S: jnp.ndarray, M: jnp.ndarray, noise: jnp.ndarray
) -> jnp.ndarray:
    """Compute energy (negative log posterior up to constants) for current overlap field."""

    Norm = jnp.sum(
        jnp.log(2 * jnp.pi * noise[:, None] ** 2) * noise[:, None] ** 2 * M / 2, axis=1
    )

    return (
        -jnp.vecdot(AA, AA * noise[:, None] ** 2) / 2
        + jnp.vecdot(AA, S)
        - 0.5 * jnp.vecdot(M, S * S)
        - Norm
    )


@jax.jit
def scan(carry, key):

    params, count, mean_grid, sig_predgrid, Count_grid, S, log_prior_sum, mean_E = carry
    (
        rates_on_grid,
        rate_cdfs,
        sampling_times,
        T_rise,
        S_support,
        RNA_int,
        int_loading_rates,
        arange,
        A,
        M,
        fine_grid,
        beta,
        n_iter,
        noise,
    ) = params
    Count_grid, S, log_prior_sum = do_move(
        key,
        Count_grid,
        log_prior_sum,
        S,
        rates_on_grid,
        rate_cdfs,
        sampling_times,
        T_rise,
        S_support,
        RNA_int,
        int_loading_rates,
        arange,
        A,
        M,
        fine_grid,
        beta,
    )
    mean_grid = Update_mean(mean_grid, Count_grid, count, n_iter // 2)
    sig_predgrid = Update_mean(sig_predgrid, S, count, n_iter // 2)
    E_current = Energy(A, S, M, noise) + log_prior_sum

    mean_E = Update_mean(mean_E, E_current, count, n_iter // 2)
    count += 1
    return (
        params,
        count,
        mean_grid,
        sig_predgrid,
        Count_grid,
        S,
        log_prior_sum,
        mean_E,
    ), (
        jnp.sum(Count_grid, axis=1),
        E_current,  # ,Count_grid,S
    )


def sample_loadings(
    in_arr_dat: jnp.ndarray,
    noise: jnp.ndarray | float,
    T_rise: float,
    T_plateau: float,
    sampling_times: jnp.ndarray,
    in_rates_on_grid: jnp.ndarray,
    fine_grid: jnp.ndarray,
    seed: int,
    RNA_int: jnp.ndarray | float,
    n_iter: int = 15000,
    beta: float = 1.0,
    nrepeat: int = 100,
):
    """Sample polymerase loading trajectories using reversible-jump MCMC."""
    iterable = False
    try:
        a = len(noise)
        iterable = True
    except:
        pass

    if not iterable:
        noise = jnp.ones((in_arr_dat.shape[0] * nrepeat,)) * noise
    else:
        noise = jnp.ones((in_arr_dat.shape[0] * nrepeat,)) * noise.repeat(
            nrepeat
        )  # (ntraj*nrepeat,)
    base_ntraj = in_arr_dat.shape[0]
    rna_base = jnp.asarray(RNA_int)
    if rna_base.ndim == 0:
        rna_base = jnp.full((base_ntraj,), rna_base)
    elif rna_base.ndim == 1:
        if rna_base.shape[0] != base_ntraj:
            raise ValueError(
                "RNA_int must be scalar or shape (ntraj,), got"
                f" {rna_base.shape} for {base_ntraj} trajectories"
            )
    else:
        raise ValueError("RNA_int must be scalar or 1D array of length ntraj")
    RNA_int = jnp.tile(rna_base, nrepeat)
    # repeat the data nrepeat to get many samples per trajectory
    arr_dat = jnp.tile(in_arr_dat, (nrepeat, 1))
    rates_on_grid = jnp.tile(in_rates_on_grid, (nrepeat, 1))
    ntraj = arr_dat.shape[0]
    dt = sampling_times[1] - sampling_times[0]

    mask = ~jnp.isnan(arr_dat)
    A = jnp.where(mask, arr_dat / (noise[:, None] ** 2), 0.0)
    M = jnp.where(mask, 1.0 / (noise[:, None] ** 2), 0.0)
    S_support = T_plateau + T_rise
    S = jnp.zeros_like(arr_dat)
    Count_grid = jnp.zeros((ntraj, fine_grid.shape[0]))
    sig_grid = jnp.zeros((ntraj, arr_dat.shape[1]))

    sampling_times = jnp.array(sampling_times)

    rate_cdfs, int_loading_rates = setup_sampler(fine_grid, rates_on_grid)
    arange = jnp.arange(0, jnp.ceil((T_rise + T_plateau) / dt) + 1).astype(int)

    init_params = (
        rates_on_grid,
        rate_cdfs,
        sampling_times,
        T_rise,
        S_support,
        RNA_int,
        int_loading_rates,
        arange,
        A,
        M,
        fine_grid,
        beta,
        n_iter,
        noise,
    )

    carry_init = (
        init_params,
        0,
        jnp.zeros_like(Count_grid),
        sig_grid,
        Count_grid,
        S,
        jnp.zeros((ntraj,)),
        jnp.zeros((ntraj,)),
    )
    keys = jax.random.split(jax.random.PRNGKey(seed), n_iter)

    (_, count, mean, sig_predgrid, Count_grid, S, log_prior_sum, mean_E), (
        n_particles_trace,
        E_trace,
    ) = jax.lax.scan(scan, carry_init, keys)

    posterior_rate = mean / (fine_grid[1] - fine_grid[0])

    # average posterior rate and mean_E over repeats
    posterior_rate = posterior_rate.reshape(nrepeat, ntraj // nrepeat, -1).mean(axis=0)
    sig_pred = sig_predgrid.reshape(nrepeat, ntraj // nrepeat, -1).mean(axis=0)
    mean_E = mean_E.reshape(nrepeat, ntraj // nrepeat).mean(axis=0)

    return (
        posterior_rate,
        sig_pred,
        n_particles_trace,
        E_trace,
        Count_grid,
        S,
        mean_E,
    )


def compute_log_Z(
    in_arr_dat: jnp.ndarray,
    noise: jnp.ndarray | float,
    T_rise: float,
    T_plateau: float,
    sampling_times: jnp.ndarray,
    in_rates_on_grid: jnp.ndarray,
    fine_grid: jnp.ndarray,
    seed: int,
    RNA_int: jnp.ndarray | float,
    n_iter: int = 10_000,
    nsteps: int = 10,
    nrepeat: int = 20,
):
    """Thermodynamic integration estimate of log normalizing constant."""
    iterable = False
    try:
        a = len(noise)
        iterable = True
    except:
        pass
    if not iterable:
        noise = jnp.ones((in_arr_dat.shape[0] * nrepeat,)) * noise
    else:
        noise = jnp.ones((in_arr_dat.shape[0] * nrepeat,)) * noise.repeat(
            nrepeat
        )  # (ntraj*nrepeat,)
    base_ntraj = in_arr_dat.shape[0]
    rna_base = jnp.asarray(RNA_int)
    if rna_base.ndim == 0:
        rna_base = jnp.full((base_ntraj,), rna_base)
    elif rna_base.ndim == 1:
        if rna_base.shape[0] != base_ntraj:
            raise ValueError(
                "RNA_int must be scalar or shape (ntraj,), got"
                f" {rna_base.shape} for {base_ntraj} trajectories"
            )
    else:
        raise ValueError("RNA_int must be scalar or 1D array of length ntraj")
    RNA_int = jnp.tile(rna_base, nrepeat)
    # repeat the data nrepeat to get many samples per trajectory
    arr_dat = jnp.tile(in_arr_dat, (nrepeat, 1))
    rates_on_grid = jnp.tile(in_rates_on_grid, (nrepeat, 1))

    ntraj = arr_dat.shape[0]
    dt = sampling_times[1] - sampling_times[0]
    mask = ~jnp.isnan(arr_dat)
    A = jnp.where(mask, arr_dat / (noise[:, None] ** 2), 0.0)
    M = jnp.where(mask, 1.0 / (noise[:, None] ** 2), 0.0)
    S_support = T_plateau + T_rise

    sampling_times = jnp.array(sampling_times)
    rates_on_grid = jnp.array(rates_on_grid)

    rate_cdfs, int_loading_rates = setup_sampler(fine_grid, rates_on_grid)

    arange = jnp.arange(0, jnp.ceil((T_rise + T_plateau) / dt) + 1).astype(int)

    betas = jnp.concatenate(
        [
            jnp.zeros(1),
            jnp.logspace(jnp.log10(1 / nsteps / 10), 0, nsteps),
        ]
    )

    @jax.jit
    def run_single_beta(beta):
        S = jnp.zeros_like(arr_dat)
        Count_grid = jnp.zeros((ntraj, fine_grid.shape[0]))
        carry_init = (
            0,
            jnp.zeros_like(Count_grid),
            Count_grid,
            S,
            jnp.zeros((ntraj,)),
            jnp.zeros((ntraj,)),
        )
        keys = jax.random.split(jax.random.PRNGKey(seed), n_iter)

        @jax.jit
        def scan(carry, key):
            count, mean_grid, Count_grid, S, log_prior_sum, mean_E = carry
            Count_grid, S, log_prior_sum = do_move(
                key,
                Count_grid,
                log_prior_sum,
                S,
                rates_on_grid,
                rate_cdfs,
                sampling_times,
                T_rise,
                S_support,
                RNA_int,
                int_loading_rates,
                arange,
                A,
                M,
                fine_grid,
                beta,
            )
            mean_grid = Update_mean(mean_grid, Count_grid, count, n_iter // 2)
            E_current = Energy(A, S, M, noise)  # + log_prior_sum
            mean_E = Update_mean(mean_E, E_current, count, n_iter // 2)
            count += 1

            return (count, mean_grid, Count_grid, S, log_prior_sum, mean_E), (
                jnp.sum(Count_grid, axis=1),
                E_current,
            )

        (count, mean, Count_grid, S, log_prior_sum, mean_E), (
            n_particles_trace,
            E_trace,
        ) = jax.lax.scan(scan, carry_init, keys)
        # average mean_E and log_prior_sum over repeats
        mean_E = mean_E.reshape(nrepeat, ntraj // nrepeat).mean(axis=0)
        log_prior_sum = log_prior_sum.reshape(nrepeat, ntraj // nrepeat).mean(axis=0)

        return jnp.sum(mean_E), jnp.sum(log_prior_sum)

    energies, log_prior_sums = jax.vmap(run_single_beta)(betas)

    log_Z = jnp.trapezoid(energies, betas)
    return log_Z, energies[-1] + log_prior_sums[-1]
