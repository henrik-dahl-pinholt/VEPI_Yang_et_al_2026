from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import multiprocessing as mp
import os
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from Simulation_thinning import (
    Get_eigensystem,
    gen_steady_state_samples,
    sample_loading_events,
    sample_promoter_states,
)


SCRIPT_ROOT = Path(__file__).resolve().parent

koff = 1.0 / 60
loading_rates = np.array([0.5, 0.000001])
T_rise, T_plateau = 2.0, 4.0
seed = 42
grid_factor = 10

noise = 1.0
track_dur = 5 * 60.0
ndata_trajs = 1000
seconds_per_minute = 60.0
spring_rate = (1 / 0.177) * seconds_per_minute
diffusion = 2810 * seconds_per_minute
polymer_length = 334
overwrite = False
alpha = 0.5

rcs_to_run = [30]
dts_to_run = [0.5]
loc_errs_to_run = [40]
seps_to_run = [90]

@dataclass
class RouseProjection:
    eigvals: np.ndarray
    q_w: np.ndarray
    mode_std: np.ndarray


def make_rouse_projection(N: int, D: float, k: float, a: int, b: int):
    qmat, eigvals = Get_eigensystem(N)
    w = np.zeros(N)
    w[a] = 1.0
    w[b] = -1.0
    return RouseProjection(
        eigvals=eigvals,
        q_w=qmat.T @ w,
        mode_std=np.sqrt(D / (k * eigvals)),
    )


def analytic_contact_probability(projection: RouseProjection, rc: float) -> float:
    sigma2 = float(np.sum((projection.q_w**2) * projection.mode_std**2))
    if sigma2 <= 0.0:
        return float(rc > 0.0)

    x = rc / math.sqrt(sigma2)
    return float(
        math.erf(x / math.sqrt(2.0))
        - math.sqrt(2.0 / math.pi) * x * math.exp(-0.5 * x * x)
    )


def sample_polymer_and_on_events_fast(
    kon: float,
    T: float,
    N: int,
    D: float,
    k: float,
    dim: int,
    localization_errors: np.ndarray,
    measurement_interval: float,
    rc: float,
    projection: RouseProjection,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_events = np.random.poisson(kon * T)
    on_times = np.sort(np.random.uniform(0.0, T, n_events))

    observation_times = np.arange(0.0, T + measurement_interval, measurement_interval)
    merged_times = np.unique(np.concatenate((on_times, observation_times)))

    modes = np.random.normal(size=(dim, N - 1)) * projection.mode_std
    ep_sep = np.empty((dim, merged_times.shape[0]), dtype=np.float64)
    ep_sep[:, 0] = modes @ projection.q_w

    for i in range(merged_times.shape[0] - 1):
        dt = merged_times[i + 1] - merged_times[i]
        decay = np.exp(-k * projection.eigvals * dt)
        innovation_std = projection.mode_std * np.sqrt(
            np.maximum(0.0, 1.0 - decay * decay)
        )
        modes = modes * decay + np.random.normal(size=(dim, N - 1)) * innovation_std
        ep_sep[:, i + 1] = modes @ projection.q_w

    data_mask = np.isin(merged_times, observation_times)
    event_mask = np.isin(merged_times, on_times)
    ep_sep_obs = ep_sep[:, data_mask]
    ep_sep_at_event = ep_sep[:, event_mask]

    ep_sep_obs_noisy = ep_sep_obs + np.random.normal(
        scale=np.sqrt(2.0) * localization_errors[:, None],
        size=ep_sep_obs.shape,
    )

    contact = np.linalg.norm(ep_sep_at_event, axis=0) < rc
    return observation_times, ep_sep_obs, ep_sep_obs_noisy, on_times[contact], on_times


def legacy_mc_contact_probability(
    N: int,
    D: float,
    k: float,
    dim: int,
    rc: float,
    a: int,
    b: int,
) -> float:
    w = np.zeros(N)
    w[a] = 1.0
    w[b] = -1.0
    return float(
        np.mean(
            np.linalg.norm(gen_steady_state_samples(10_000, D, k, N, dim) @ w, axis=-1)
            < rc
        )
    )


def generate_ms2_signal_compact(
    sampling_times: np.ndarray,
    loading_times: np.ndarray,
    T_increase: float,
    T_plateau: float,
    alpha: float,
    noise: float,
) -> Tuple[np.ndarray, np.ndarray]:
    noise_array = np.random.normal(0.0, noise, size=sampling_times.shape)
    signal = np.zeros(sampling_times.shape, dtype=np.float64)
    support_end = T_increase + T_plateau

    for loading_time in loading_times:
        start = np.searchsorted(sampling_times, loading_time, side="right")
        plateau_start = np.searchsorted(
            sampling_times, loading_time + T_increase, side="right"
        )
        end = np.searchsorted(sampling_times, loading_time + support_end, side="right")

        if plateau_start > start:
            dt = sampling_times[start:plateau_start] - loading_time
            signal[start:plateau_start] += alpha * dt / T_increase
        if end > plateau_start:
            signal[plateau_start:end] += alpha

    return signal, signal + noise_array


def sample_ep_experiment_fast(
    *,
    kon: float,
    koff: float,
    T: float,
    N: int,
    k: float,
    D: float,
    dim: int,
    rc: float,
    localization_errors: np.ndarray,
    measurement_interval: float,
    pol2_loading_rate: np.ndarray,
    alpha: float,
    MS2_noise: float,
    T_increase: float,
    T_plateau: float,
    a: int,
    b: int,
    seed: int,
    projection: RouseProjection,
    p_contact_method: str,
    analytic_p_contact: Optional[float] = None,
):
    np.random.seed(seed)
    (
        observation_times,
        ep_sep_obs,
        ep_sep_obs_noisy,
        thinned_on_times,
        on_times,
    ) = sample_polymer_and_on_events_fast(
        kon=kon,
        T=T,
        N=N,
        D=D,
        k=k,
        dim=dim,
        localization_errors=localization_errors,
        measurement_interval=measurement_interval,
        rc=rc,
        projection=projection,
    )

    if p_contact_method == "legacy_mc":
        p_contact = legacy_mc_contact_probability(
            N=N,
            D=D,
            k=k,
            dim=dim,
            rc=rc,
            a=a,
            b=b,
        )
    elif p_contact_method == "analytic":
        p_contact = analytic_p_contact
        if p_contact is None:
            p_contact = analytic_contact_probability(projection, rc)
    else:
        raise ValueError(f"Unknown p_contact_method: {p_contact_method}")

    state_sequence, state_times = sample_promoter_states(
        p_contact=p_contact,
        kon=kon,
        koff=koff,
        T=T,
        thinned_on_times=thinned_on_times,
    )

    pol2_loading_events = sample_loading_events(
        times=state_times,
        states=state_sequence,
        state_vals=pol2_loading_rate,
        T_max=T,
    )

    MS2_signal, MS2_signal_noisy = generate_ms2_signal_compact(
        sampling_times=observation_times,
        loading_times=pol2_loading_events,
        T_increase=T_increase,
        T_plateau=T_plateau,
        alpha=alpha,
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


_WORKER_CONTEXT = None


def _init_worker(context):
    global _WORKER_CONTEXT
    params = context["params"]
    projection = make_rouse_projection(
        N=params["N"],
        D=params["D"],
        k=params["k"],
        a=params["a"],
        b=params["b"],
    )
    analytic_p_contact = None
    if context["p_contact_method"] == "analytic":
        analytic_p_contact = analytic_contact_probability(projection, params["rc"])
    _WORKER_CONTEXT = {
        "params": params,
        "projection": projection,
        "p_contact_method": context["p_contact_method"],
        "analytic_p_contact": analytic_p_contact,
    }


def _run_single(seed_value):
    context = _WORKER_CONTEXT
    return sample_ep_experiment_fast(
        seed=seed_value,
        projection=context["projection"],
        p_contact_method=context["p_contact_method"],
        analytic_p_contact=context["analytic_p_contact"],
        **context["params"],
    )


def parse_float_list(value: str):
    return [float(item) for item in value.split(",") if item]


def parse_int_list(value: str):
    return [int(item) for item in value.split(",") if item]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate exact-law test data using fast Rouse-mode propagation."
    )
    parser.add_argument("--n-trajs", type=int, default=ndata_trajs)
    parser.add_argument("--processes", type=int, default=mp.cpu_count())
    parser.add_argument("--chunksize", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", default=overwrite)
    parser.add_argument(
        "--p-contact-method",
        choices=("legacy_mc", "analytic"),
        default=os.environ.get("P_CONTACT_METHOD", "legacy_mc"),
        help=(
            "legacy_mc reproduces the old per-trajectory 10,000-sample estimate; "
            "analytic uses the exact steady-state Maxwell probability and is faster."
        ),
    )
    parser.add_argument("--dts", type=parse_float_list, default=dts_to_run)
    parser.add_argument("--loc-errs", type=parse_int_list, default=loc_errs_to_run)
    parser.add_argument("--seps", type=parse_int_list, default=seps_to_run)
    parser.add_argument("--rcs", type=parse_int_list, default=rcs_to_run)
    parser.add_argument("--cache-subdir", default="4_simdata")
    return parser


def run_dataset(
    *,
    params,
    seeds,
    processes,
    chunksize,
    p_contact_method,
):
    context = {
        "params": params,
        "p_contact_method": p_contact_method,
    }
    worker_count = max(1, min(processes, len(seeds)))
    if chunksize is None:
        chunksize = max(1, len(seeds) // (worker_count * 8))

    if worker_count == 1:
        _init_worker(context)
        return [
            _run_single(seed_value)
            for seed_value in tqdm(seeds, total=len(seeds), smoothing=0)
        ]

    with mp.Pool(
        processes=worker_count,
        initializer=_init_worker,
        initargs=(context,),
    ) as pool:
        return list(
            tqdm(
                pool.imap(_run_single, seeds, chunksize=chunksize),
                total=len(seeds),
                smoothing=0,
            )
        )


def main():
    args = build_parser().parse_args()
    cache_dir = SCRIPT_ROOT / "cache" / args.cache_subdir
    cache_dir.mkdir(parents=True, exist_ok=True)

    seeds = [seed + i for i in range(args.n_trajs)]

    for dt in args.dts:
        for loc_err in args.loc_errs:
            print(f"\trunning loc err {loc_err}")
            localization_errors = np.array([loc_err, loc_err, loc_err])
            for sep in args.seps:
                left_locus = polymer_length // 2 - sep // 2
                right_locus = polymer_length // 2 + sep // 2
                Gamma = 2 * diffusion / np.sqrt(spring_rate * np.pi)
                J = abs(left_locus - right_locus) * diffusion / spring_rate
                for rc_true in args.rcs:
                    print(f"\t\trunning {rc_true}")
                    store_path = (
                        cache_dir / f"test_data_{dt:.2f}_{loc_err}_{sep}_{rc_true}.pkl"
                    )
                    if store_path.exists() and not args.overwrite:
                        continue

                    kon = 20.0 * (sep / 90) ** (3 / 2) * (30.0 / rc_true) ** 3
                    params = {
                        "kon": kon,
                        "koff": koff,
                        "T": track_dur,
                        "N": polymer_length,
                        "k": spring_rate,
                        "D": diffusion,
                        "dim": 3,
                        "rc": rc_true,
                        "localization_errors": localization_errors,
                        "measurement_interval": dt,
                        "pol2_loading_rate": loading_rates[::-1],
                        "alpha": alpha,
                        "MS2_noise": noise,
                        "T_increase": T_rise,
                        "T_plateau": T_plateau,
                        "a": left_locus,
                        "b": right_locus,
                    }
                    results = run_dataset(
                        params=params,
                        seeds=seeds,
                        processes=args.processes,
                        chunksize=args.chunksize,
                        p_contact_method=args.p_contact_method,
                    )

                    with open(store_path, "wb") as f:
                        resdict = {
                            "results": results,
                            "parameters": {
                                "kon": kon,
                                "koff": koff,
                                "T_rise": T_rise,
                                "T_plateau": T_plateau,
                                "alpha": alpha,
                                "loading_rates": loading_rates,
                                "localization_errors": localization_errors,
                                "noise": noise,
                                "dt": dt,
                                "track_dur": track_dur,
                                "polymer_length": polymer_length,
                                "spring_rate": spring_rate,
                                "diffusion": diffusion,
                                "rc_true": rc_true,
                                "Gamma": Gamma,
                                "J": J,
                                "loc_err": loc_err,
                                "sep": sep,
                                "simulator": "rouse_mode_fast",
                                "p_contact_method": args.p_contact_method,
                            },
                        }
                        pickle.dump(resdict, f)


if __name__ == "__main__":
    main()
