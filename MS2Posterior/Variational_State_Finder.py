from typing import Dict, Any, Iterable, Tuple, Set, Optional
import jax.numpy as jnp
import jax
import numpy as np
from scipy.special import digamma
from Tilted_CTMC import TiltedCTMC
from Pol2_sampler_jax import sample_loadings
from tqdm.auto import tqdm
import time

Array = jnp.ndarray
Msg = Dict[str, Any]  # messages are dicts of named expected stats


# ---------- Node interface (works for latent or observed) ----------
class Node:
    def __init__(self, name: str, parents: Tuple[str, ...], children: Tuple[str, ...]):
        self.name = name
        self.parents = parents
        self.children = children

    # What stats this node can publish to others (cached into MessageStore)
    def moments(self) -> Msg:
        """Return expectations others might consume (e.g., ``E[x]``, ``E[log x]``).

        Returns
        -------
        Dict[str, Any]
            Mapping of statistic names to arrays.
        """
        raise NotImplementedError

    # What this node needs from its Markov blanket to do a closed-form update
    def needs_for_update(self) -> Dict[str, Iterable[str]]:
        """Describe required messages for a coordinate update.

        Returns
        -------
        dict
                Keys ``parents``, ``children``, ``co_parents`` with iterable of message names.
        """
        raise NotImplementedError

    # Do the closed-form update given messages from parents/children/coparents
    def update(self, msgs: Dict[str, Msg], rho: float = 1.0) -> "Node":
        """Return an updated node given Markov blanket messages.

        Parameters
        ----------
        msgs : dict
            ``{"parents": {p: Msg}, "children": {c: Msg}, "co_parents": {k: Msg}}``.
        rho : float, default=1.0
            Damping factor applied internally by the node.

        Returns
        -------
        Node
            Updated node (usually ``self`` mutated).
        """
        raise NotImplementedError


# ---------- Graph structure and message store ----------


class Graph:
    def __init__(self, nodes: Dict[str, Node]):
        self.nodes = nodes

    def parents(self, i: str) -> Set[str]:
        return set(self.nodes[i].parents)

    def children(self, i: str) -> Set[str]:
        return set(self.nodes[i].children)

    def co_parents(self, i: str) -> Set[str]:
        cps = set()
        for c in self.children(i):
            cps |= set(self.nodes[c].parents)
        cps.discard(i)
        return cps

    def markov_blanket(self, i: str) -> Tuple[Set[str], Set[str], Set[str]]:
        return self.parents(i), self.children(i), self.co_parents(i)

    def __getitem__(self, key: str) -> Node:
        try:
            return self.nodes[key]
        except KeyError:
            raise KeyError(
                f"Node '{key}' not found in graph. Valid nodes are: {list(self.nodes.keys())}"
            )


# simple cache: last-published moments per node
class MessageStore:
    def __init__(
        self,
        cache: Dict[str, Msg] = None,
        fix_messages: Dict[str, Msg] = None,
    ):
        self.cache = cache if cache is not None else {}
        self.fix_messages = fix_messages if fix_messages is not None else {}

    def get(self, node_name: str) -> Msg:
        """Return last-published message for a node."""
        return self.cache[node_name]

    def publish(self, node_name: str, msg: Msg):
        """Store new messages"""
        self.cache[node_name] = msg


def cavi_sweep(
    G: Graph,
    M: MessageStore,
    schedule: Iterable[str],
    rho: float = 0.75,
    verbose: bool = False,
    message_manipulator=None,
):
    """Perform one coordinate-ascent variational inference sweep over the graph."""
    if message_manipulator is not None:
        M = message_manipulator(M)
    t0 = time.time()
    for i in schedule:
        t2 = time.time()
        if verbose:
            print(f"Updating node: {i}")
        node = G.nodes[i]
        needs = node.needs_for_update()

        # Collect messages from Markov blanket
        parents, children, coparents = G.markov_blanket(i)
        msgs = {"parents": {}, "children": {}, "co_parents": {}}

        for p in parents:
            pub = M.get(p)
            msgs["parents"].update({k: pub[k] for k in needs["parents"] if k in pub})

        for c in children:
            pub = M.get(c)
            msgs["children"].update({k: pub[k] for k in needs["children"] if k in pub})

        for k in coparents:
            pub = M.get(k)
            msgs["co_parents"].update(
                {k2: pub[k2] for k2 in needs["co_parents"] if k2 in pub}
            )
        # Closed-form update (node applies its own masking + damping)
        new_node = node.update(msgs, rho=rho)
        G.nodes[i] = new_node

        # Immediately publish its new moments
        M.publish(i, new_node.moments())
        # Apply any message manipulation (e.g., fixing certain messages)
        if message_manipulator is not None:
            M = message_manipulator(M)

        if verbose:
            print(f" Updated node: {i} in {time.time()-t2:.2f} seconds.")

    if verbose:
        print(f"CAVI sweep completed in {time.time()-t0:.2f} seconds.")


def _mix(old, new, rho, mask, groups=None):
    """Damped parameter update with optional mask and grouping.
    When groups are specified, all entries in a group are tied together and updated to the same value (still with damping) which is the sum over the new values in that group.

    Parameters
    ----------
    old : array
        Old parameter values.
    new : array
        New parameter values.
    rho : float
        Damping factor. New values are weighted by rho, old values by (1-rho).
    mask : array
        Boolean array of same shape as old/new; True entries are frozen to old values.
    groups : array, optional
        Array of same shape as old/new; entries with the same group index are tied together.
    """
    if groups is None:
        groups = np.arange(len(old.flatten())).reshape(old.shape)
    summed_new = np.zeros_like(old)
    for group in np.unique(groups):
        # construct an aggregated new summed over the group entries
        group_mask = groups == group
        summed_entry = np.sum(new[group_mask])
        # copy entry to all entries in the group
        summed_new[group_mask] = summed_entry

    return np.where(mask, old, (1.0 - rho) * old + rho * summed_new)


class GammaPrior(Node):
    """Variational node for Gamma parameters"""

    def __init__(
        self,
        name: str,
        parents: Tuple[str, ...],
        children: Tuple[str, ...],
        n: int,
        prior_shape: Optional[Array] = None,
        prior_rate: Optional[Array] = None,
        _mask: Optional[Array] = None,
        _fixed_groups: Optional[Array] = None,
    ):
        self._fixed_groups = _fixed_groups  # groups of indices to be tied together
        self.name = name
        self.parents = parents
        self.children = children
        self.n = n
        self.shape = None
        self.rate = None
        self.prior_shape = prior_shape
        self.prior_rate = prior_rate
        self._mask = _mask

        # check that prior shapes are correct
        for array, name in zip(
            [self.prior_shape, self.prior_rate], ["prior_shape", "prior_rate"]
        ):
            if array is not None:
                if array.shape != (self.n,):
                    raise ValueError(
                        f"'{name}' has shape {array.shape}, expected ({self.n},)"
                    )
        # Initialize fixed groups
        if self._fixed_groups is None:
            self._fixed_groups = np.arange(self.n)
        else:
            # check that _fixed_groups shape is correct
            if self._fixed_groups.shape != (self.n,):
                raise ValueError(
                    f"'_fixed_groups' has shape {self._fixed_groups.shape}, expected ({self.n},)"
                )

        # Initialize mask
        if self._mask is None:
            self._mask = np.zeros(self.n).astype(bool)
        else:
            # check that _mask shape is correct
            if self._mask.shape != (self.n,):
                raise ValueError(
                    f"'_mask' has shape {self._mask.shape}, expected ({self.n},)"
                )
            # ensure that all entries in a group is masked if any entry is masked
            for group in np.unique(self._fixed_groups):
                group_mask = self._mask[self._fixed_groups == group]
                if np.any(group_mask):
                    self._mask[self._fixed_groups == group] = True

        # initialize shape
        if self.prior_shape is None:
            self.prior_shape = 1 * np.ones(self.n)  # uninformative prior for stability

        # initialize rate
        if self.prior_rate is None:
            self.prior_rate = 1e-3 * np.ones(
                self.n
            )  # uninformative prior for stability
        self.shape = self.prior_shape
        self.rate = self.prior_rate

    def moments(self) -> Msg:
        mean = self.shape / self.rate
        log_avg = digamma(self.shape) - np.log(self.rate)
        std_dev = np.sqrt(self.shape) / self.rate
        return {
            f"{self.name} <k>": mean,
            f"{self.name} <log k>": log_avg,
            f"{self.name} <std k>": std_dev,
        }


"""Utility functions """


def _unwrap_matrix(mat: Array) -> Array:
    """Convert a square matrix into a vector by taking only the off-diagonal elements row-wise."""
    n = mat.shape[0]
    inds = np.array([(i, j) for i in range(n) for j in range(n) if i != j])
    return mat[inds[:, 0], inds[:, 1]]


def _wrap_matrix(vec: Array) -> Array:
    """Convert a vector of off-diagonal elements into a square matrix by filling in zeros on the diagonal and then column normalizing to zero."""
    len_vec = vec.shape[0]
    nstates = int(0.5 * np.sqrt(4 * len_vec + 1) + 0.5)
    mat = np.zeros((nstates, nstates))
    inds = np.array([(i, j) for i in range(nstates) for j in range(nstates) if i != j])
    mat[inds[:, 0], inds[:, 1]] = vec
    return mat - np.diag(np.sum(mat, axis=0))


def _as_rate_series(rate_series, reference=None):
    rate_series = jnp.asarray(rate_series)
    if rate_series.ndim == 2:
        rate_series = rate_series[None, :, :]
    if reference is not None:
        reference = jnp.asarray(reference)
        if reference.ndim == 2:
            reference = reference[None, :, :]
        if rate_series.shape[0] == 1 and reference.shape[0] != 1:
            rate_series = jnp.broadcast_to(rate_series, reference.shape)
    return rate_series


def _effective_contact_rate_swaps(rate_series, rate_inds, Q_avg, dt, contact_prob):
    """Convert interval any-contact probabilities into rate-series multipliers.

    For an interval contact probability ``p_any`` and base contact rate ``k``,
    use the interval rate
    ``lambda_eff = -log(1 - p_any * (1 - exp(-k * dt))) / dt``.  This reduces
    to ``k * p_any`` for small ``k * dt`` and saturates at
    ``-log(1 - p_any) / dt`` for large ``k * dt``.  When multiple transitions
    leave the same source state, distribute the effective exit rate in
    proportion to the original proposed driven rates.
    """
    if contact_prob is None:
        return None
    if dt is None:
        raise ValueError("dt must be provided for effective contact rate swaps.")

    rate_inds = np.asarray(rate_inds)
    rate_series = _as_rate_series(rate_series, None)
    if rate_series.shape[0] == 1 and len(rate_inds) != 1:
        rate_series = jnp.broadcast_to(
            rate_series, (len(rate_inds),) + rate_series.shape[1:]
        )
    contact_prob = _as_rate_series(contact_prob, rate_series)
    effective_swaps = jnp.zeros_like(rate_series)

    for from_ind in np.unique(rate_inds[:, 1]):
        swap_rows = np.where(rate_inds[:, 1] == from_ind)[0]
        p_contact = jnp.clip(contact_prob[swap_rows[0]], 0.0, 1.0)
        plug_in_exit = jnp.zeros_like(p_contact)
        plug_in_rates = []
        for row in swap_rows:
            to_ind = rate_inds[row, 0]
            plug_in_rate = Q_avg[to_ind, from_ind] * rate_series[row]
            plug_in_rates.append(plug_in_rate)
            plug_in_exit = plug_in_exit + plug_in_rate

        contact_rate = jnp.where(
            p_contact > 1e-12, plug_in_exit / (p_contact + 1e-300), 0.0
        )
        contact_rate_dt = contact_rate * dt
        p_log = jnp.minimum(p_contact, 1.0 - 1e-7)
        log_survival = jnp.where(
            p_contact >= 1.0 - 1e-7,
            -contact_rate_dt,
            jnp.log1p(-p_log * (-jnp.expm1(-contact_rate_dt))),
        )
        effective_exit = -log_survival / dt

        for row, plug_in_rate in zip(swap_rows, plug_in_rates):
            to_ind = rate_inds[row, 0]
            row_effective_rate = jnp.where(
                plug_in_exit > 1e-12,
                effective_exit * plug_in_rate / (plug_in_exit + 1e-300),
                0.0,
            )
            row_swap = jnp.where(
                Q_avg[to_ind, from_ind] > 1e-12,
                row_effective_rate / Q_avg[to_ind, from_ind],
                0.0,
            )
            effective_swaps = effective_swaps.at[row].set(row_swap)

    return effective_swaps


def _Insert_rate_series(
    Q_tilde,
    f_mats,
    rate_series,
    rate_inds,
    Q_avg,
    dt=None,
    contact_prob=None,
):
    """
    Insert the rate series into the Q_tilde matrix and f_mats.

    Parameters
    ----------
    Q_tilde : array (n_states, n_states)
        The normalized transition rate matrix.
    f_mats : array (batch_size, ntimes, n_states, n_states)
        The f_mats to be modified.
    rate_series : array (nswaps,batch_size, ntimes)
        The rate series to be inserted.
    rate_inds : array (nswaps,2)
        The indices of reaction in which to insert the rate series.
    Q_avg : array (n_states, n_states)
        The average transition rate matrix.

    Returns
    -------
    tuple
        Updated Q_tilde and f_mats.
    """

    Q_tilde_arr = np.array(Q_tilde)
    rate_inds = np.asarray(rate_inds)
    rate_series = _as_rate_series(rate_series, None)
    if rate_series.shape[0] == 1 and len(rate_inds) != 1:
        rate_series = jnp.broadcast_to(
            rate_series, (len(rate_inds),) + rate_series.shape[1:]
        )
    if contact_prob is not None:
        rate_series = _effective_contact_rate_swaps(
            rate_series, rate_inds, Q_avg, dt, contact_prob
        )
        contact_prob = None
    Q_tilde_new = np.array(Q_tilde)[None, None, :] + np.zeros_like(f_mats)
    for rate, rate_ind in zip(rate_series, rate_inds):
        # reweight the off-diagonal entry
        Q_tilde_new[:, :, rate_ind[0], rate_ind[1]] *= rate

        # adjust the diagonal entry to keep the row sum zero
        Q_tilde_new[:, :, rate_ind[1], rate_ind[1]] += (
            Q_tilde_arr[None, None, rate_ind[0], rate_ind[1]]
            - Q_tilde_new[:, :, rate_ind[0], rate_ind[1]]
        )
        # swap into f_mats
        f_mats = f_mats.at[:, :, rate_ind[1], rate_ind[1]].add(
            (
                Q_avg[None, None, rate_ind[0], rate_ind[1]]
                - Q_tilde_arr[None, None, rate_ind[0], rate_ind[1]]
            )
            * (rate - 1.0)
        )

    return jnp.array(Q_tilde_new), f_mats


@jax.jit
def int_jax(y, dt):
    return jnp.nansum(y, axis=0) * dt


_vmap_int_jax = jax.vmap(int_jax, in_axes=(0, None))
"""---------- Specific node implementations for Variational Inference of MS2 data----------"""


class LoadingRates(GammaPrior):
    """Variational node for polymerase loading rates"""

    def __init__(
        self,
        dt,
        n,
        **kwargs,
    ):
        super().__init__(
            name="Loading Rates",
            parents=(),
            children=("Polymerase Loadings",),
            n=n,
            **kwargs,
        )
        self.dt = dt

    def needs_for_update(self) -> Dict[str, Iterable[str]]:
        return {
            "parents": [],
            "children": [
                "Pol2_posterior",
            ],
            "co_parents": ["masked_posterior"],  # ["State_posterior"],
        }

    def update(self, msgs: Dict[str, Msg], rho: float = 1.0) -> "GammaPrior":
        # Gather expected sufficient statistics from children
        p2_loadings = msgs["children"].get("Pol2_posterior", None)  # (ntraj,ntimesteps)
        promoter_state = (
            msgs["co_parents"].get("masked_posterior", None)[1:].swapaxes(0, 1)
        )  # (ntraj,ntimesteps,nstates)
        if p2_loadings is not None and promoter_state is not None:
            # compute sufficient statistics from co-parents
            integrand = (
                p2_loadings[:, :, None] * promoter_state
            )  # (ntraj,ntimesteps,nstates)

            # identify if the rates are per track or shared across tracks by comparing the shape of the rates to the number of states
            if self.n == promoter_state.shape[-1]:
                # rates are shared across tracks, so we sum over all trajectories to get the expected transitions and times for each state

                expected_transitions = _vmap_int_jax(
                    jnp.where(jnp.isnan(integrand), 0.0, integrand), self.dt
                ).sum(
                    axis=0
                )  # (nstates,)

                integrand = promoter_state  # (ntraj,ntimesteps,nstates)
                expected_times = _vmap_int_jax(
                    jnp.where(jnp.isnan(integrand), 0.0, integrand), self.dt
                ).sum(
                    axis=0
                )  # (nstates,)
            else:
                # rates are per track, so we need to keep track of the expected transitions and times for each state and each track
                expected_transitions = _vmap_int_jax(
                    jnp.where(jnp.isnan(integrand), 0.0, integrand), self.dt
                ).reshape(
                    -1
                )  # (ntraj*nstates,)

                integrand = promoter_state  # (ntraj,ntimesteps,nstates)
                expected_times = _vmap_int_jax(
                    jnp.where(jnp.isnan(integrand), 0.0, integrand), self.dt
                ).reshape(
                    -1
                )  # (ntraj*nstates)

            # Update shape and rate parameters using the gathered statistics
            new_shape = expected_transitions + self.prior_shape
            new_rate = expected_times + self.prior_rate
            # Apply masking, damping, and grouping
            self.shape = _mix(
                self.shape,
                new_shape,
                rho,
                self._mask,
                groups=self._fixed_groups,
            )
            self.rate = _mix(
                self.rate, new_rate, rho, self._mask, groups=self._fixed_groups
            )
        else:
            print(
                "Warning: LoadingRates update called without required messages from children."
            )
        return self


class TransitionRates(GammaPrior):
    """Variational node for promoter state transition rates"""

    def __init__(
        self,
        dt,
        n,
        name="Transition Rates",
        parents=(),
        children=("Promoter State",),
        driven_rate_bounds=(1e-8, 1000.0),
        driven_rate_map_xatol=1e-4,
        **kwargs,
    ):
        super().__init__(name=name, parents=parents, children=children, n=n, **kwargs)
        self.dt = dt
        self.rate_swaps = None
        self.rate_inds = None
        self.contact_survival_probs = None
        self.driven_rate_bounds = driven_rate_bounds
        self.driven_rate_map_xatol = driven_rate_map_xatol
        self._map_rate_mask = np.zeros(self.n, dtype=bool)
        self._map_rate_values = np.full(self.n, np.nan)

    def needs_for_update(self) -> Dict[str, Iterable[str]]:
        return {
            "parents": [],
            "children": [
                "masked_posterior",
                "masked_joint",
            ],  # ["State_posterior", "Jump_posterior"],
            "co_parents": [""],
        }

    def update(self, msgs: Dict[str, Msg], rho: float = 1.0) -> "GammaPrior":
        # Gather expected sufficient statistics from children
        promoter_state = msgs["children"].get("masked_posterior", None)
        Jumps = msgs["children"].get(
            "masked_joint", None
        )  # (ntraj,ntimesteps,nstates,nstates)
        if promoter_state is not None and Jumps is not None:

            # compute sufficient statistics from co-parents
            integrand = Jumps  # (ntraj,ntimesteps,nstates,nstates)
            expected_transitions = _vmap_int_jax(integrand, self.dt).sum(
                axis=0
            )  # (nstates,nstates)
            integrand = promoter_state  # (ntraj,ntimesteps,nstates)
            expected_times = jnp.nansum(
                _vmap_int_jax(integrand, self.dt), axis=0
            )  # (nstates,)

            # the transitions are stored as a linear array so unwrap the matrix
            unwrapped_transitions = _unwrap_matrix(
                expected_transitions
            )  # (nstates*(nstates-1),)

            # the expected times are the amount of time spent in the state before the transition
            col_arr = (
                np.arange(promoter_state.shape[-1])[None, :]
                * np.ones(promoter_state.shape[-1], dtype=int)[:, None]
            )
            unwrapped_col_arr = _unwrap_matrix(col_arr)

            expected_times = expected_times[unwrapped_col_arr]  # (nstates*(nstates-1),)
            if self.rate_swaps is not None and self.rate_inds is not None:
                for swap, inds in zip(self.rate_swaps, self.rate_inds):
                    to_ind, from_ind = inds
                    vec_ind = self._transition_index(
                        to_ind, from_ind, promoter_state.shape[-1]
                    )
                    swapped_time = jnp.nansum(
                        _vmap_int_jax(
                            integrand[1:, :, from_ind] * swap.T,
                            self.dt,
                        )
                    )
                    expected_times = expected_times.at[vec_ind].set(swapped_time)

            # Update shape and rate parameters using the gathered statistics
            new_shape = unwrapped_transitions + self.prior_shape
            new_rate = expected_times + self.prior_rate

            # Apply masking and damping
            self.shape = _mix(
                self.shape,
                new_shape,
                rho,
                self._mask,
                groups=self._fixed_groups,
            )
            self.rate = _mix(
                self.rate, new_rate, rho, self._mask, groups=self._fixed_groups
            )
            if (
                self.contact_survival_probs is not None
                and self.rate_swaps is not None
                and self.rate_inds is not None
            ):
                self._update_contact_survival_map_rates(promoter_state, Jumps, rho)
            else:
                self._map_rate_mask[:] = False

        else:
            print(
                "Warning: TransitionRates update called without required messages from children."
            )
        return self

    def _transition_index(self, to_ind: int, from_ind: int, nstates: int) -> int:
        return to_ind * (nstates - 1) + from_ind - int(from_ind > to_ind)

    def _time_batch_view(
        self, values: Array, target_shape: Tuple[int, int]
    ) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.shape == target_shape:
            return values
        if values.ndim == 2 and values.T.shape == target_shape:
            return values.T
        if values.ndim == 1:
            if values.shape[0] == target_shape[0]:
                return np.broadcast_to(values[:, None], target_shape)
            if values.shape[0] == target_shape[1]:
                return np.broadcast_to(values[None, :], target_shape)
        return np.broadcast_to(values, target_shape)

    def moments(self) -> Msg:
        out = super().moments()
        if np.any(self._map_rate_mask):
            rate = np.asarray(out[f"{self.name} <k>"]).copy()
            log_rate = np.asarray(out[f"{self.name} <log k>"]).copy()
            std_rate = np.asarray(out[f"{self.name} <std k>"]).copy()
            map_values = self._map_rate_values[self._map_rate_mask]
            rate[self._map_rate_mask] = map_values
            log_rate[self._map_rate_mask] = np.log(map_values)
            std_rate[self._map_rate_mask] = 0.0
            out[f"{self.name} <k>"] = rate
            out[f"{self.name} <log k>"] = log_rate
            out[f"{self.name} <std k>"] = std_rate
        return out


class InitialStateProbabilities(Node):
    """Variational node for initial state probabilities"""

    def __init__(
        self,
        Nsets,
        dim,
        name="Initial State Probabilities",
        parents=(),
        children=("Promoter State",),
        prior_concentration: Optional[Array] = None,
        _mask: Optional[Array] = None,
        **kwargs,
    ):
        self.Nsets = Nsets
        self.dim = dim
        self.prior_concentration = prior_concentration
        super().__init__(name=name, parents=parents, children=children, **kwargs)
        self._mask = _mask

        # check that prior concentration is correct
        if self.prior_concentration is not None:
            if self.prior_concentration.shape != (
                self.Nsets,
                self.dim,
            ):
                raise ValueError(
                    f"'prior_concentration' has shape {self.prior_concentration.shape}, expected ({self.Nsets}, {self.dim})"
                )

        # Initialize mask
        if self._mask is None:
            self._mask = np.zeros((self.Nsets, self.dim)).astype(bool)

        # initialize concentration
        if self.prior_concentration is None:
            self.prior_concentration = np.zeros(
                (self.Nsets, self.dim)
            )  # uninformative prior
        self.concentration = self.prior_concentration

    def moments(self) -> Msg:
        # Keep the initial-state message numerically stable with the current
        # zero-pseudocount prior; exact-zero occupancies are valid here.
        # mean = self.concentration / np.sum(self.concentration, axis=-1, keepdims=True)
        # log_avg = digamma(self.concentration) - digamma(
        #     np.sum(self.concentration, axis=-1, keepdims=True)
        # )
        log_avg = np.log(self.concentration + 1e-9)

        # check that there are no nans in log_avg
        if jnp.any(~jnp.isfinite(log_avg)):
            raise ValueError(
                f"NaN values found in log average of InitialStateProbabilities.\nNonfinite in concentration: {jnp.any(~jnp.isfinite(self.concentration))}.\nNonfinite in log_avg: {jnp.any(~jnp.isfinite(log_avg))}."
            )
        return {
            f"{self.name} <log pi>": log_avg,
        }

    def needs_for_update(self) -> Dict[str, Iterable[str]]:
        return {
            "parents": [],
            "children": [
                "State_posterior",
            ],
            "co_parents": [],
        }

    def update(
        self, msgs: Dict[str, Msg], rho: float = 1.0
    ) -> "InitialStateProbabilities":
        # Gather expected sufficient statistics from children
        promoter_state = (
            msgs["children"].get("State_posterior", None).swapaxes(0, 1)
        )  # (ntraj,ntimesteps,nstates)

        if promoter_state is not None:
            # compute sufficient statistics from co-parents
            expected_occupancy = promoter_state[:, 0, :]  # (ntraj,nstates)

            # Update concentration parameters using the gathered statistics
            new_concentration = expected_occupancy + self.prior_concentration
            self.concentration = _mix(
                self.concentration, new_concentration, rho, self._mask
            )

        else:
            print(
                "Warning: InitialStateProbabilities update called without required messages from children."
            )
        return self


@jax.jit
def TCMC_log_avg(posterior, joint, f_mats, Q_tilde, init_prob, dt):
    avg_init_prob = (jnp.log(init_prob + 1e-9) * posterior[:, 0, :]).sum()
    if len(Q_tilde.shape) == 2:
        Q_tilde = Q_tilde[None, None, :]

    exit_rates = -jnp.diagonal(
        Q_tilde, axis1=-2, axis2=-1
    )  # (1,1,nstates) or (ntraj,ntimesteps,nstates)

    f_mat_diag = jnp.diagonal(f_mats, axis1=-2, axis2=-1)  # (ntraj,ntimesteps,nstates)

    # do the integral over local-time terms

    local_integrand = jnp.sum(
        -(exit_rates + f_mat_diag) * posterior[:, 1:], axis=-1
    )  # (ntraj,ntimesteps)

    local_integral = jnp.sum(_vmap_int_jax(local_integrand, dt))  # (ntraj,)

    log_Q = jnp.log(Q_tilde)
    log_Q = jnp.where(jnp.isfinite(log_Q), log_Q, 0.0)

    # do the integral over jump terms
    jump_integrand = jnp.sum(
        joint * log_Q,
        axis=(-2, -1),
    )  # (ntraj,ntimesteps)
    jump_integral = jnp.sum(_vmap_int_jax(jump_integrand, dt))  # (ntraj,)

    return jnp.sum(avg_init_prob + local_integral + jump_integral)


class Promoter_State(Node):
    """Variational node for the promoter state"""

    def __init__(
        self,
        time_grid,
        nstates,
        max_times,
        rate_swaps=None,
        rate_inds=None,
        contact_survival_probs=None,
        name="Promoter State",
        parents=(
            "Transition Rates",
            "Initial State Probabilities",
        ),
        children=("Polymerase Loadings",),
    ):
        self.dt = time_grid[1] - time_grid[0]
        self.max_times = max_times
        self.time_grid = time_grid
        self.time_masks = jnp.array(
            [time_grid <= tmax for tmax in max_times]
        ).T  # (ntimesteps,Nsets)
        self.time_masks = jnp.where(self.time_masks, 1.0, jnp.nan)
        # add an extra one for the initial time
        self.time_masks = jnp.concatenate(
            [jnp.ones((1, self.time_masks.shape[1])), self.time_masks], axis=0
        )  # (ntimesteps+1,Nsets)
        self.nstates = nstates
        self.rate_swaps = rate_swaps
        self.rate_inds = rate_inds
        self.contact_survival_probs = contact_survival_probs
        super().__init__(name=name, parents=parents, children=children)

    def needs_for_update(self) -> Dict[str, Iterable[str]]:

        return {
            "parents": [
                "Transition Rates <k>",
                "Transition Rates <log k>",
                "Initial State Probabilities <log pi>",
            ],
            "children": ["Pol2_posterior"],
            "co_parents": ["Loading Rates <k>", "Loading Rates <log k>"],
        }

    # Do the closed-form update given messages from parents/children/coparents
    def update(self, msgs: Dict[str, Msg], rho: float = 1.0) -> "Node":
        """
        msgs: {"parents": {p: Msg}, "children": {c: Msg}, "co_parents": {k: Msg}}
        Must apply damping/masking internally.
        """
        # compute the prior
        log_prior = msgs["parents"]["Initial State Probabilities <log pi>"]
        self.init_prob = jnp.exp(
            log_prior - jax.scipy.special.logsumexp(log_prior, axis=-1, keepdims=True)
        )

        # compute the generator matrices
        log_rates = msgs["parents"]["Transition Rates <log k>"]
        rates = msgs["parents"]["Transition Rates <k>"]
        self.Q_tilde = _wrap_matrix(np.exp(log_rates))
        self.Q = _wrap_matrix(rates)

        # compute the tilt factor
        loading_rate = msgs["co_parents"]["Loading Rates <k>"]
        log_loading_rate = msgs["co_parents"]["Loading Rates <log k>"]
        pol2_rate = msgs["children"]["Pol2_posterior"]  # (ntraj,ntimesteps)

        Q_exit = -jnp.diag(self.Q)
        Q_tilde_exit = -jnp.diag(self.Q_tilde)

        # check if the polymerase loading rates are per track or per state and reshape accordingly
        if len(loading_rate) == self.nstates:
            tilt = (
                Q_exit[None, None, :]
                - Q_tilde_exit[None, None, :]
                + loading_rate[None, None, :]
                - log_loading_rate * pol2_rate[:, :, None]
            )  # (ntraj,ntimesteps,nstates)
        else:
            reshaped = loading_rate.reshape(
                len(self.init_prob), self.nstates
            )  # (ntracks,nstates)
            reshaped_logged = log_loading_rate.reshape(
                len(self.init_prob), self.nstates
            )  # (ntracks,nstates)
            tilt = (
                Q_exit[None, None, :]
                - Q_tilde_exit[None, None, :]
                + reshaped[:, None, :]
                - reshaped_logged[:, None, :] * pol2_rate[:, :, None]
            )  # (ntraj,ntimesteps,nstates)

        norm_tilt = tilt - jnp.min(tilt, axis=-1, keepdims=True)
        self.fmat = (
            jnp.eye(self.nstates)[None, None, :, :] * norm_tilt[..., None]
        )  # (ntraj,ntimesteps,nstates,nstates)

        # apply rate series if provided
        if self.rate_swaps is not None and self.rate_inds is not None:
            q_bef, f_bef = self.Q_tilde, self.fmat
            self.Q_tilde, self.fmat = _Insert_rate_series(
                self.Q_tilde,
                self.fmat,
                self.rate_swaps,
                self.rate_inds,
                self.Q,
                dt=self.dt,
                contact_prob=self.contact_survival_probs,
            )
        return self

    def moments(self) -> Msg:
        TCTMC, fmat, Q = self._runTCMC()
        del TCTMC, fmat, Q
        # check that there are no nans
        if jnp.any(jnp.isnan(self.posterior)):
            raise ValueError("NaNs detected in promoter state posterior.")
        if jnp.any(jnp.isnan(self.joint)):
            raise ValueError("NaNs detected in promoter state joint.")
        return {
            "State_posterior": self.posterior,
            "Jump_posterior": self.joint,
            "masked_posterior": self.masked_posterior,
            "masked_joint": self.masked_joint,
            "MAP": self.MAP,
        }

    def _runTCMC(self):

        if (
            self.rate_swaps is not None
            and self.rate_inds is not None
            and len(self.Q_tilde.shape) == 2
        ):
            Q_tilde, fmat = _Insert_rate_series(
                self.Q_tilde,
                self.fmat,
                self.rate_swaps,
                self.rate_inds,
                self.Q,
                dt=self.dt,
                contact_prob=self.contact_survival_probs,
            )
            # Q = _Insert_rate_series(Q, self.fmat, self.rate_swaps, self.rate_inds)[0]
            # log_avg_Q = jnp.log(jnp.abs(q_t) + 1e-100)
        else:
            fmat = self.fmat
            Q_tilde = self.Q_tilde
        TCTMC = TiltedCTMC(fmat, Q_tilde, self.init_prob, self.time_grid)
        self.posterior = TCTMC.get_posterior()
        self.MAP = TCTMC.most_likely_path()

        joint_vals = TCTMC.get_joint()
        self.joint = jnp.where(jnp.isnan(joint_vals), 0, joint_vals)

        self.masked_posterior = self.posterior * self.time_masks[:, :, None]
        self.masked_joint = self.joint * self.time_masks[1:, :, None, None]

        return TCTMC, fmat, Q_tilde


@jax.jit
def kernel_jax(x, T_plateau, T_increase):
    return (
        jnp.heaviside(x, 1)
        * jnp.heaviside(T_plateau + T_increase - x, 1)
        * (
            (x / T_increase) * jnp.heaviside(T_increase - x, 1)
            + jnp.heaviside(x - T_increase, 1)
        )
    )


def coarse_grain_rate(rate_fine, block):
    if (rate_fine.shape[0] - 1) % block != 0:
        raise ValueError(
            f"Fine rate length {rate_fine.shape[0]} is not compatible with block {block}"
        )
    coarse = rate_fine[:-1].reshape(-1, block).mean(axis=1)
    return jnp.concatenate([coarse, rate_fine[-1:]])


def upsample_rate(rate_coarse, block):
    return jnp.concatenate([jnp.repeat(rate_coarse[:-1], block), rate_coarse[-1:]])


def coarse_grain_mask(mask_fine, block):
    if (mask_fine.shape[0] - 1) % block != 0:
        raise ValueError(
            f"Fine mask length {mask_fine.shape[0]} is not compatible with block {block}"
        )
    coarse = mask_fine[:-1].reshape(-1, block).any(axis=1)
    return jnp.concatenate([coarse, mask_fine[-1:]])


vmap_upsample_rate = jax.vmap(upsample_rate, in_axes=(0, None))
vmap_coarse_grain_rate = jax.vmap(coarse_grain_rate, in_axes=(0, None))
vmap_coarse_grain_mask = jax.vmap(coarse_grain_mask, in_axes=(0, None))


class Polymerase_Loadings(Node):
    """Variational node for polymerase loading events"""

    def __init__(
        self,
        time_grid,
        noise_std,
        T_rise,
        T_plateau,
        sampling_times,
        RNA_int,
        nrepeat=10,
        niter=10_000,
        **kwargs,
    ):

        self.nrepeat = nrepeat
        self.niter = niter
        self.time_grid = time_grid
        self.dt = time_grid[1] - time_grid[0]
        self.noise_std = noise_std  # (batch_size,)
        self.T_rise = T_rise
        self.T_plateau = T_plateau
        self.sampling_times = sampling_times
        self.RNA_int = RNA_int

        self.time_grid = time_grid

        super().__init__(
            name="Polymerase Loadings",
            parents=("Loading Rates", "Promoter State"),
            children=("MS2 Data",),
            **kwargs,
        )

    def needs_for_update(self) -> Dict[str, Iterable[str]]:
        return {
            "parents": [
                "Loading Rates <log k>",
                "Loading Rates <k>",
                "State_posterior",
                "masked_posterior",
            ],
            "children": ["data"],
            "co_parents": [],
        }

    def comp_logavg(self, promoter_state, log_loading_rate, loading_rate):
        nstates = promoter_state.shape[-1]
        # check if the loading rates are per track or shared across tracks
        if len(log_loading_rate) == nstates:
            per_track = False
        else:
            per_track = True

        if not per_track:
            self.log_avg = jnp.sum(
                promoter_state * log_loading_rate[None, None, :], axis=-1
            )  # (ntraj,ntimesteps)
            self.avg_rate = jnp.sum(
                promoter_state * loading_rate[None, None, :], axis=-1
            )
        else:
            ntraj = promoter_state.shape[0]
            log_loading_rate_reshaped = log_loading_rate.reshape(
                ntraj, nstates
            )  # (ntraj,nstates)
            loading_rate_reshaped = loading_rate.reshape(
                ntraj, nstates
            )  # (ntraj,nstates)
            self.log_avg = jnp.sum(
                promoter_state * log_loading_rate_reshaped[:, None, :], axis=-1
            )  # (ntraj,ntimesteps)
            self.avg_rate = jnp.sum(
                promoter_state * loading_rate_reshaped[:, None, :], axis=-1
            )

        self.rate = jnp.exp(self.log_avg)  # (ntraj,ntimesteps)

    def update(self, msgs: Dict[str, Msg], rho: float = 1.0) -> "Node":
        # Gather expected sufficient statistics from parents
        log_loading_rate = msgs["parents"]["Loading Rates <log k>"]
        loading_rate = msgs["parents"]["Loading Rates <k>"]
        promoter_state = msgs["parents"]["State_posterior"].swapaxes(0, 1)[
            :, 1:
        ]  # (ntraj,ntimesteps,nstates)
        masked_posterior = msgs["parents"].get(
            "masked_posterior", msgs["parents"]["State_posterior"]
        )
        promoter_state_masked = masked_posterior.swapaxes(0, 1)[
            :, 1:
        ]  # (ntraj,ntimesteps,nstates)

        data = msgs["children"]["data"]  # (ntraj,ntimesteps)

        # compute the prior rate function
        if log_loading_rate is not None and promoter_state is not None:
            self.comp_logavg(promoter_state, log_loading_rate, loading_rate)
            nanmask = jnp.isnan(
                promoter_state_masked.sum(axis=-1)
            )  # (ntraj,ntimesteps)
            self.nan_mask = nanmask

        else:
            print(
                "Warning: Polymerase_Loadings update called without required messages from parents."
            )

        if data is not None:
            self.data = data
        else:
            print(
                "Warning: Polymerase_Loadings update called without required messages from parents/children."
            )
        return self

    def moments(self) -> Msg:

        (
            self.posterior_rate,
            self.pred_sig,
            n_particles_trace,
            E_trace,
            Count_grid,
            S,
            self.mean_E,
        ) = sample_loadings(
            self.data,
            self.noise_std,
            self.T_rise,
            self.T_plateau,
            self.sampling_times,
            self.rate,
            self.time_grid,
            seed=np.random.randint(1e6),
            RNA_int=self.RNA_int,
            nrepeat=self.nrepeat,
            n_iter=self.niter,
        )

        return {"Pol2_posterior": self.posterior_rate, "Predicted MS2": self.pred_sig}


class MS2_data(Node):
    def __init__(
        self,
        data: Array,
    ):
        self.data = data
        self.name = "MS2 Data"
        self.parents = ("Polymerase Loadings",)
        self.children = ()
        super().__init__(self.name, self.parents, self.children)

    def moments(self) -> Msg:
        return {"data": self.data}  # trivial

    def needs_for_update(self) -> Dict[str, Iterable[str]]:
        return {"parents": [], "children": [], "co_parents": []}

    def update(self, msgs, rho=1.0) -> "Node":
        return self  # no-op


class MS2Posterior:
    """Coordinate-ascent variational inference driver for MS2 datasets."""

    def __init__(
        self,
        fine_grid,
        nstates,
        sampling_times,
        arr_dat,
        T_rise,
        T_plateau,
        MS2_intensity,
        noise,
        per_track_rates=False,
        argdict={},
    ):
        """Initialize factor graph nodes and message store.

        Parameters
        ----------
        fine_grid : jnp.ndarray
            Temporal grid for latent loadings.
        nstates : int
            Number of promoter states.
        sampling_times : jnp.ndarray
            Observation times for MS2 traces.
        arr_dat : np.ndarray
            Observed dataset, shape ``(ntraj, ntimes)``.
        T_rise, T_plateau : float
            MS2 kernel parameters.
        MS2_intensity : float
            Fluorescence per polymerase.
        noise : float or np.ndarray
            Observation noise (scalar or per-trajectory).
        per_track_rates : bool
            Whether to use per-track loading rates (True) or shared loading rates (False), default False.
        argdict : dict, optional
            Optional overrides for node initialization. The keys should be the
            names of the nodes to be modified, and the values should be dictionaries
            of parameters to be passed to the node constructors. Allowed keys are:
            "Loading Rates", "Transition Rates", "Initial State Probabilities",
            "Promoter State", and "Polymerase Loadings".
        """
        self.per_track_rates = per_track_rates
        self.sampling_times = sampling_times
        self.nstates = nstates
        self.fine_grid = fine_grid
        self.tmaxes = np.array(
            [
                self.sampling_times[np.argmax(self.sampling_times * (~np.isnan(d)))]
                for d in arr_dat
            ]
        )
        self.Tmax = np.max(self.tmaxes)
        self.ntransitions = nstates * (nstates - 1)
        self.dt_fine = self.fine_grid[1] - self.fine_grid[0]
        self.dt = self.sampling_times[1] - self.sampling_times[0]
        nodes_objects = {MS2_data(arr_dat)}
        self.promoter_grid = self.fine_grid
        self.dt_promoter = self.promoter_grid[1] - self.promoter_grid[0]

        # add LoadingRates node
        if not self.per_track_rates:
            LoadingRates_params = {"n": nstates}
        else:
            LoadingRates_params = {"n": nstates * len(arr_dat)}
        if "Loading Rates" in argdict:
            LoadingRates_params.update(argdict["Loading Rates"])
        nodes_objects.add(LoadingRates(self.dt_promoter, **LoadingRates_params))

        # add TransitionRates node
        TransitionRates_params = {"n": self.ntransitions}
        if "Transition Rates" in argdict:
            TransitionRates_params.update(argdict["Transition Rates"])
        nodes_objects.add(TransitionRates(self.dt_promoter, **TransitionRates_params))

        # add InitialStateProbabilities node
        InitialStateProbabilities_params = {}
        if "Initial State Probabilities" in argdict:
            InitialStateProbabilities_params.update(
                argdict["Initial State Probabilities"]
            )
        nodes_objects.add(
            InitialStateProbabilities(
                len(arr_dat), nstates, **InitialStateProbabilities_params
            )
        )

        # add PromoterState node
        Promoter_State_params = {"nstates": nstates, "max_times": self.tmaxes}
        if "Promoter State" in argdict:
            Promoter_State_params.update(argdict["Promoter State"])
        nodes_objects.add(Promoter_State(self.promoter_grid, **Promoter_State_params))

        # add PolymeraseLoadings node
        Polymerase_Loadings_params = {}
        if "Polymerase Loadings" in argdict:
            Polymerase_Loadings_params.update(argdict["Polymerase Loadings"])
        nodes_objects.add(
            Polymerase_Loadings(
                fine_grid,
                noise,
                T_rise,
                T_plateau,
                sampling_times,
                MS2_intensity,
                **Polymerase_Loadings_params,
            )
        )

        # Create graph
        self.nodes = {node.name: node for node in nodes_objects}
        self.G = Graph(self.nodes)

        max_time = self.Tmax

        # lrate_vals = 1 / np.linspace(
        #     self.dt / 10, self.dt * 5, self.G["Loading Rates"].n, endpoint=True
        # )
        lrate_vals = np.logspace(
            np.log10(self.dt / 100), np.log10(self.dt * 5), self.G["Loading Rates"].n
        )[::-1]

        trate_vals = (
            1
            / np.logspace(
                np.log10(self.dt * 10),
                np.log10(max_time / 10),
                self.G["Transition Rates"].n,
            )
        )[::-1]

        self.time_masks = jnp.array(
            [self.promoter_grid < tmax for tmax in self.tmaxes]
        ).T  # (ntimesteps,Nsets)
        # add an extra one for the initial time
        self.time_masks = jnp.concatenate(
            [jnp.ones((1, self.time_masks.shape[1])), self.time_masks], axis=0
        )  # (ntimesteps+1,Nsets)
        self.time_masks = jnp.where(self.time_masks, 1.0, jnp.nan)

        # pstate_pos = self.time_masks[:,:,None]*jnp.ones((len(fine_grid) + 1, len(arr_dat), nstates)) / nstates

        pstate_pos = np.random.uniform(
            0, 1, (len(self.promoter_grid) + 1, len(arr_dat), nstates)
        )
        pstate_pos = self.time_masks[:, :, None] * jnp.array(
            pstate_pos / np.sum(pstate_pos, axis=-1, keepdims=True)
        )

        init_prob = jnp.ones((len(arr_dat), nstates)) / nstates

        # Store in dictionary
        self.init_dict = {
            "Loading Rates": {
                "Loading Rates <k>": lrate_vals,
                "Loading Rates <log k>": np.log(lrate_vals),
            },
            "Transition Rates": {
                "Transition Rates <k>": trate_vals,
                "Transition Rates <log k>": np.log(trate_vals),
            },
            "Promoter State": {
                "masked_posterior": pstate_pos,
                "State_posterior": pstate_pos,
            },
            "Initial State Probabilities": {
                "Initial State Probabilities <log pi>": jnp.log(init_prob)
            },
            "MS2 Data": {"data": arr_dat},
            "Polymerase Loadings": {},
            "Background": {},
        }

        # modify with any provided initializations
        if "init_vals" in argdict:
            for key in argdict["init_vals"]:
                self.init_dict[key].update(argdict["init_vals"][key])

        # Create message store which holds current moments
        self.M = MessageStore(cache=self.init_dict)

        self.schedule = [
            "Polymerase Loadings",
            "Loading Rates",
            "Promoter State",
            "Transition Rates",
            "Initial State Probabilities",
            "Promoter State",
            "Transition Rates",
            "Initial State Probabilities",
            "Promoter State",
            "Transition Rates",
            "Initial State Probabilities",
        ]

    def Swap_Rate_Series(
        self,
        rate_swaps: Array,
        rate_inds: Array,
        contact_survival_probs: Optional[Array] = None,
    ):
        """Insert rate series into promoter state node and transition rate node.

        Parameters
        ----------
        rate_swaps : Array
            Array of shape (n_swaps,) containing multiplicative factors to apply to the specified rates
        rate_inds : Array            Array of shape (n_swaps, 2) containing the indices of the rates to swap, where each row is (to_ind, from_ind) indicating that the rate at from_ind should be multiplied by the corresponding factor in rate_swaps and inserted at to_ind.
        contact_survival_probs : Array, optional
            Actual contact probabilities p(t) for the swapped rates. When
            provided, the promoter posterior and ELBO use the local Bernoulli
            contact-averaged survival term instead of exp(-k p(t) dt).
        """
        self.G["Promoter State"].rate_swaps = rate_swaps
        self.G["Promoter State"].rate_inds = rate_inds
        self.G["Promoter State"].contact_survival_probs = contact_survival_probs
        self.G["Transition Rates"].rate_swaps = rate_swaps
        self.G["Transition Rates"].rate_inds = rate_inds
        self.G["Transition Rates"].contact_survival_probs = contact_survival_probs

    def Run(
        self,
        verbose=False,
        rtol=5e-3,
        atol=1e-10,
        miniter=0,
        maxiter=np.inf,
        rho=0.75,
        message_manipulator=None,
        eval_conds=["loading", "transition"],
    ):
        """Iteratively run CAVI until convergence or minimum iterations met."""

        self.params = {
            "Loading Rates": [],
            "Transition Rates": [],
            "Loading Rates <std k>": [],
            "Transition Rates <std k>": [],
        }
        # check eval_conds
        for cond in eval_conds:
            if cond not in ("loading", "transition", "init_prob"):
                raise ValueError(
                    f"Invalid eval_cond: {cond}. Must be one of 'loading', 'transition', or 'init_prob'."
                )

        previous_init_prob, current_init_prob = None, None
        pbar = tqdm(leave=False, desc="CAVI Sweeps")
        while True:
            cavi_sweep(
                self.G,
                self.M,
                self.schedule,
                verbose=verbose,
                message_manipulator=message_manipulator,
                rho=rho,
            )

            self.params["Loading Rates"].append(
                self.M.get("Loading Rates")["Loading Rates <k>"]
            )
            lrats = self.M.get("Loading Rates")["Loading Rates <k>"]
            if len(lrats) > self.nstates:
                print(
                    self.M.get("Loading Rates")["Loading Rates <k>"][
                        : self.nstates * 2
                    ],
                    self.M.get("Transition Rates")["Transition Rates <k>"],
                )
            else:
                print(
                    self.M.get("Loading Rates")["Loading Rates <k>"][: self.nstates],
                    self.M.get("Transition Rates")["Transition Rates <k>"],
                )
            self.params["Loading Rates <std k>"].append(
                self.M.get("Loading Rates")["Loading Rates <std k>"]
            )
            self.params["Transition Rates"].append(
                self.M.get("Transition Rates")["Transition Rates <k>"]
            )
            self.params["Transition Rates <std k>"].append(
                self.M.get("Transition Rates")["Transition Rates <std k>"]
            )

            previous_init_prob = current_init_prob
            current_init_prob = self.M.get("Initial State Probabilities")[
                "Initial State Probabilities <log pi>"
            ]
            # check for nans in params and initial prob
            if (
                np.any(~np.isfinite(self.params["Loading Rates"][-1]))
                or np.any(~np.isfinite(self.params["Transition Rates"][-1]))
                or np.any(~np.isfinite(current_init_prob))
            ):
                raise ValueError(
                    "nonfinite values encountered in parameters during CAVI. Try different initialization or check data for issues.\nLoading Rates: {}\nTransition Rates: {}\nInitial Probabilities: {}".format(
                        self.params["Loading Rates"][-1],
                        self.params["Transition Rates"][-1],
                        current_init_prob,
                    )
                )
            if len(self.params["Loading Rates"]) > 1:
                tol_val_init_prob = np.max(
                    np.abs(current_init_prob - previous_init_prob)
                    - atol
                    - rtol * np.abs(previous_init_prob)
                )
                tol_val_loading = np.max(
                    np.abs(
                        self.params["Loading Rates"][-1]
                        - self.params["Loading Rates"][-2]
                    )
                    - atol
                    - rtol * np.abs(self.params["Loading Rates"][-2])
                )

                tol_val_transitio0n = np.max(
                    np.abs(
                        self.params["Transition Rates"][-1]
                        - self.params["Transition Rates"][-2]
                    )
                    - atol
                    - rtol * np.abs(self.params["Transition Rates"][-2])
                )
                pbar.set_description(
                    f"CAVI Sweeps | ΔLoading Rate: {tol_val_loading:.2e} | ΔTransition Rate: {tol_val_transitio0n:.2e}"  # | ΔInit Prob: {tol_val_init_prob:.2e} "
                )
                condition = True

                for cond in eval_conds:
                    if cond == "loading":
                        condition = condition and np.allclose(
                            self.params["Loading Rates"][-1],
                            self.params["Loading Rates"][-2],
                            atol=atol,
                            rtol=rtol,
                        )
                    if cond == "transition":
                        condition = condition and np.allclose(
                            self.params["Transition Rates"][-1],
                            self.params["Transition Rates"][-2],
                            atol=atol,
                            rtol=rtol,
                        )
                    if cond == "init_prob":
                        condition = condition and np.allclose(
                            current_init_prob, previous_init_prob, atol=atol, rtol=rtol
                        )
                if condition and pbar.n >= miniter:
                    break

            pbar.update(1)
            if pbar.n >= maxiter:
                print(f"Reached maximum iterations ({maxiter}) without convergence.")
                break

        self.params["Transition Rates"] = np.array(self.params["Transition Rates"])
        self.params["Loading Rates"] = np.array(self.params["Loading Rates"])
