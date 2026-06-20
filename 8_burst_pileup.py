from pathlib import Path
import pickle
import jax
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
import os,sys

SCRIPT_ROOT = Path(__file__).resolve().parent
from jax import numpy as jnp
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


TWOLOCUSGPR_ROOT = SCRIPT_ROOT / "TwoLocusGPR"
sys.path.insert(0, str(TWOLOCUSGPR_ROOT))

MSD_PATH = SCRIPT_ROOT / "cache" / "MSD_fits" / "fit_results_individual_v28_StatisticalFilter_Alpha0p37_final"

from TwoLocusGPR.GPR import GPR
from TwoLocusGPR.MSD_functions import FD_MSD
from TwoLocusGPR.Posterior_analysis import make_ball_quadrature
from script_utils import (
    _lookup_param,
    fit_amplitude_only,
    predict_w0_pon_crosscorr,
    reassign_tracks_by_duration,
    smooth_lag_curve,
    time_interval_label,
    time_interval_seconds,
)

RESULT_PATH = SCRIPT_ROOT / "result" / "2_run_inference"
OUTPUT_PATH = RESULT_PATH.parent / "8_burst_pileup"

if not OUTPUT_PATH.exists():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

DATAPATH = SCRIPT_ROOT / "Data" / "filtered_data_v6"
NPY_CONDITION_MAPPING = {
    "85kb_None": "15C-A2_GSK",
    "340kb_IAA": "G7B8G2_GSK_100uM_IAAadded2hrBefore",
    "340kb_dTAG": "G7B8G2_GSK_500nMdTAG13added2hrBefore",
    "340kb_None": "S+V-A6B8_GSK",
    "255kb_None": "15A-A6G9_GSK",
    "170kb_None": "15B-18G9_GSK",
    "340kb_Cp_None": "SCre-G5-H7_GSK",
    "340kb_Ce_None": "VCre-C2b_GSK",
    "340kb_Ce_Cp_None": "G7B8G2_GSK",
    "340kb_Ce_Cp_noE_noP_None": "E11G8_GSK",
    "340kb_Ce_Cp_noE_None": "Vika-H11_GSK",
    "2.362kb_noE_noP_None": "14A-A11_GSK",
    "1.5kb_None": "14B5-F10",
    "0.395kb_noE_noP_None": "14A-A11E6_GSK",
    "30s_85kb_IAA": "15C-A2_GSK_IAA(100uM)added2hrBefore",
    "30s_170kb_IAA": "15B-18G9_GSK_IAA(100uM)added2hrBefore",
    "30s_255kb_IAA": "15A-A6G9_GSK_IAA(100uM)added2hrBefore",
    "30s_340kb_IAA": "S+V-A6B8_GSK_IAA(100uM)added2hrBefore",
    "30s_340kb_Ce_Cp_IAAdTAG": "G7B8G2_GSK_dTAG13(500uM)andIAA(100uM)added2hrBefore",
}

def load_model_res(model_result_path):
    with np.load(model_result_path) as model_res:
        required = {
            "original_ep_data",
            "original_ms2_data",
            "turnon",
        }
        missing = required.difference(model_res.files)
        if missing:
            if verbose:
                print(
                    f"Missing keys {sorted(missing)} in {model_result_path}. Skipping."
                )
            return None

        ep_data = np.asarray(model_res["original_ep_data"], dtype=float)
        if ep_data.ndim != 3:
            raise ValueError(
                f"Expected original_ep_data to have shape tracks x time x xyz, got {ep_data.shape}"
            )
        distance = np.linalg.norm(ep_data, axis=-1)
        ms2 = np.asarray(model_res["original_ms2_data"], dtype=float)
        turn_on = model_res["turnon"].T

        observation_times = (
            np.asarray(model_res["observation_times"], dtype=float)
            if "observation_times" in model_res.files
            else None
        )
    return {
        "ep_data": ep_data,
        "distance": distance,
        "ms2": ms2,
        "turn_on": turn_on,
        "observation_times": observation_times,
    }









def resolve_msd_fit(condition, treatment, time_interval):
    """Find the cached MSD fit pickle and condition key for this dataset."""
    frame_label = time_interval_label(time_interval)
    mapping_keys = [
        f"{frame_label}_{condition}_{treatment}",
        f"{condition}_{treatment}",
    ]
    for mapping_key in mapping_keys:
        fit_condition = NPY_CONDITION_MAPPING.get(mapping_key)
        if fit_condition is None:
            continue
        fit_path = MSD_PATH / f"fit_params_{fit_condition}.pkl"
        if fit_path.exists():
            return fit_path, fit_condition

    tried = [
        str(MSD_PATH / f"fit_params_{NPY_CONDITION_MAPPING[key]}.pkl")
        for key in mapping_keys
        if key in NPY_CONDITION_MAPPING
    ]
    raise FileNotFoundError(
        f"No cached MSD fit found for {condition=}, {treatment=}, "
        f"{time_interval=}. Tried: {tried}"
    )


def load_fd_msd_fit_params(
    condition,
    treatment,
    time_interval,
    *,
    position_scale=1000.0,
    time_units="minutes",
):
    """Load FD-MSD fit parameters in the units used by this inference script.

    The MSD fits were performed in frame lags and microns. The inference data
    loaded here are in nm, and observation_times are in minutes.
    """
    frame_label = time_interval_label(time_interval)
    frame_seconds = time_interval_seconds(time_interval)
    try:
        fit_path, fit_condition = resolve_msd_fit(condition, treatment, time_interval)
    except FileNotFoundError as e:
        print(str(e))
        return None

    with open(fit_path, "rb") as f:
        fit_result = pickle.load(f)
    params = (
        fit_result["params"]
        if isinstance(fit_result, dict) and "params" in fit_result
        else fit_result
    )

    prefix_candidates = [f"{frame_label}_{fit_condition}"]
    if frame_label == "0.5s":
        prefix_candidates.append(f"0p5s_{fit_condition}")

    alpha = _lookup_param(
        params,
        [f"{prefix} a (dim 0)" for prefix in prefix_candidates]
        + [f"{fit_condition}_a", "global_a"],
        default=0.37,
    )
    log_tau_frames = _lookup_param(
        params,
        [f"{prefix} log(t) (dim 0)" for prefix in prefix_candidates]
        + [f"{fit_condition}_log(t)"],
    )
    log_J_um2 = _lookup_param(
        params,
        [f"{prefix} log(J) (dim 0)" for prefix in prefix_candidates]
        + [f"{fit_condition}_log(J)"],
    )
    log_noise_um2 = np.array(
        [
            _lookup_param(
                params,
                [f"{prefix} log(s²) (dim {dim})" for prefix in prefix_candidates]
                + [f"{frame_label} log(s²) (dim {dim})", f"log(s²) (dim {dim})"],
            )
            for dim in range(3)
        ]
    )

    if time_units == "frames":
        tau_scale = 1.0
    elif time_units == "seconds":
        tau_scale = frame_seconds
    elif time_units == "minutes":
        tau_scale = frame_seconds / 60.0
    else:
        raise ValueError("time_units must be 'frames', 'seconds', or 'minutes'")

    tau = np.exp(log_tau_frames) * tau_scale
    J = np.exp(log_J_um2) * position_scale**2
    noise_std = np.sqrt(np.exp(log_noise_um2)) * position_scale
    # GPR internally uses sqrt(2) * noise as the observed-coordinate noise std.
    gpr_noise_std = noise_std / np.sqrt(2.0)
    fd_params = np.array([tau, J, alpha], dtype=float)
    compact_gpr_params = np.concatenate([fd_params, gpr_noise_std])

    return {
        "fit_path": fit_path,
        "fit_condition": fit_condition,
        "frame_label": frame_label,
        "fd_params": fd_params,
        "noise_std": noise_std,
        "gpr_noise_std": gpr_noise_std,
        "compact_gpr_params": compact_gpr_params,
        "raw_params": params,
    }


def mean_radius_diag_gaussian(means, vars, quadrature_order=15):
    means = np.asarray(means, dtype=float)
    vars = np.maximum(np.asarray(vars, dtype=float), 0.0)
    nodes_1d, weights_1d = np.polynomial.hermite.hermgauss(quadrature_order)
    nodes_1d = np.sqrt(2.0) * nodes_1d
    weights_1d = weights_1d / np.sqrt(np.pi)
    grid = np.stack(
        np.meshgrid(nodes_1d, nodes_1d, nodes_1d, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    grid_weights = np.prod(
        np.stack(
            np.meshgrid(weights_1d, weights_1d, weights_1d, indexing="ij"), axis=-1
        ),
        axis=-1,
    ).reshape(-1)
    samples = means[..., None, :] + np.sqrt(vars)[..., None, :] * grid[None, None, :, :]
    return np.sum(np.linalg.norm(samples, axis=-1) * grid_weights[None, None, :], axis=-1)


def weighted_mean_radius_diag_gaussian(
    means,
    vars,
    weights,
    quadrature_order=7,
    point_chunk_size=128,
):
    means = np.asarray(means, dtype=float)
    vars = np.maximum(np.asarray(vars, dtype=float), 0.0)
    weights = np.asarray(weights, dtype=float)

    nodes_1d, weights_1d = np.polynomial.hermite.hermgauss(quadrature_order)
    nodes_1d = np.sqrt(2.0) * nodes_1d
    weights_1d = weights_1d / np.sqrt(np.pi)
    grid = np.stack(
        np.meshgrid(nodes_1d, nodes_1d, nodes_1d, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    grid_weights = np.prod(
        np.stack(
            np.meshgrid(weights_1d, weights_1d, weights_1d, indexing="ij"), axis=-1
        ),
        axis=-1,
    ).reshape(-1)

    n_points, n_times, _ = means.shape
    sqrt_vars = np.sqrt(vars)
    out = np.zeros(n_times, dtype=float)
    for start in range(0, n_points, point_chunk_size):
        stop = min(start + point_chunk_size, n_points)
        weight_chunk = weights[start:stop]
        mean_chunk = means[start:stop]
        for time_idx in range(n_times):
            samples = (
                mean_chunk[:, time_idx, None, :]
                + sqrt_vars[time_idx][None, None, :] * grid[None, :, :]
            )
            expected = np.sum(
                np.linalg.norm(samples, axis=-1) * grid_weights[None, :],
                axis=1,
            )
            out[time_idx] += np.sum(weight_chunk * expected)
    return out


def compute_instantaneous_contact_prediction(
    msd_fit,
    window_time,
    rc=40,
    ball_quadrature=(8, 8, 16),
    radius_quadrature_order=7,
):
    base_paramvec = np.array(msd_fit["compact_gpr_params"], dtype=float, copy=True)
    observed_noise_var = (np.sqrt(2.0) * np.asarray(msd_fit["gpr_noise_std"])) ** 2

    predictor = GPR(
        jnp.asarray([0.0]),
        FD_MSD,
        3,
        param_layout=((), (), ()),
        noise_layout=("dim",),
        param_names=("tau", "J", "alpha"),
    )

    rc = float(rc)
    if rc <= 0:
        raise ValueError("rc must be positive")

    nr, nz, nphi = ball_quadrature
    unit_ball_points, logw_base = make_ball_quadrature(nr=nr, nz=nz, nphi=nphi)
    points = np.asarray(rc * unit_ball_points, dtype=float)
    log_volume_weights = np.asarray(logw_base, dtype=float) + 3.0 * np.log(rc)

    final_params = predictor.expand_compact_theta(jnp.abs(jnp.asarray(base_paramvec)))
    theta = final_params[:-3]
    stationary_var = np.asarray(
        predictor.covbuilder(theta, jnp.asarray([0.0]), jnp.asarray([0.0]))[:, 0, 0],
        dtype=float,
    )
    stationary_var = np.maximum(stationary_var, np.finfo(float).tiny)

    # Conditioning on ||X_0|| <= rc uses the stationary GP prior truncated to
    # the ball, not a uniform distribution over the ball.
    log_prior_weights = -0.5 * np.sum(points**2 / stationary_var[None, :], axis=1)
    log_weights = log_volume_weights + log_prior_weights
    finite_weights = np.isfinite(log_weights)
    if not np.any(finite_weights):
        raise ValueError("No finite quadrature weights for contact prediction")
    log_weights = log_weights[finite_weights]
    points = points[finite_weights]
    weights = np.exp(log_weights - np.max(log_weights))
    weights = weights / np.sum(weights)

    cross_cov = np.asarray(
        predictor.covbuilder(theta, jnp.asarray(window_time), jnp.asarray([0.0]))[
            :, :, 0
        ].T,
        dtype=float,
    )
    response = cross_cov / stationary_var[None, :]
    conditional_var = stationary_var[None, :] - cross_cov**2 / stationary_var[None, :]
    conditional_var = np.maximum(conditional_var, 0.0) + observed_noise_var[None, :]
    conditional_means = points[:, None, :] * response[None, :, :]
    return weighted_mean_radius_diag_gaussian(
        conditional_means,
        conditional_var,
        weights,
        quadrature_order=radius_quadrature_order,
    )


def safe_divide(num, den):
    return np.divide(
        num,
        den,
        out=np.full_like(num, np.nan, dtype=float),
        where=den > 0,
    )


def random_center_window_stats(
    windowed_values, center_weights, center_idx, rng, n_repeats
):
    values = np.asarray(windowed_values, dtype=float)
    weights = np.asarray(center_weights, dtype=float)
    n_tracks, _, this_window_size = values.shape
    num = np.zeros((n_repeats, this_window_size), dtype=float)
    den = np.zeros((n_repeats, this_window_size), dtype=float)
    source_counts = np.zeros(n_tracks, dtype=int)
    candidate_counts = np.zeros(n_tracks, dtype=int)

    for track_idx in range(n_tracks):
        center_valid = np.isfinite(values[track_idx, :, center_idx]) & np.isfinite(
            weights[track_idx]
        )
        source_idx = np.flatnonzero(center_valid)
        candidate_idx = np.flatnonzero(center_valid)
        source_counts[track_idx] = len(source_idx)
        candidate_counts[track_idx] = len(candidate_idx)
        if len(source_idx) == 0 or len(candidate_idx) == 0:
            continue

        source_weights = weights[track_idx, source_idx]
        for repeat_idx in range(n_repeats):
            random_idx = rng.choice(candidate_idx, size=len(source_idx), replace=True)
            sampled_windows = values[track_idx, random_idx]
            valid = np.isfinite(sampled_windows)
            num[repeat_idx] += np.sum(
                np.where(valid, sampled_windows * source_weights[:, None], 0.0),
                axis=0,
            )
            den[repeat_idx] += np.sum(
                np.where(valid, source_weights[:, None], 0.0),
                axis=0,
            )

    return {
        "num": num,
        "den": den,
        "mean": safe_divide(num, den),
        "source_counts_by_track": source_counts,
        "candidate_counts_by_track": candidate_counts,
    }


def random_center_distance_samples(
    windowed_distances, center_weights, center_idx, rng, n_repeats
):
    """Sample center distances from random windows for the distance histogram."""
    distances = np.asarray(windowed_distances, dtype=float)
    weights = np.asarray(center_weights, dtype=float)
    n_tracks = distances.shape[0]
    sampled_distances = []
    sampled_weights = []

    for track_idx in range(n_tracks):
        center_valid = np.isfinite(distances[track_idx, :, center_idx]) & np.isfinite(
            weights[track_idx]
        )
        source_idx = np.flatnonzero(center_valid)
        candidate_idx = np.flatnonzero(center_valid)
        if len(source_idx) == 0 or len(candidate_idx) == 0:
            continue

        source_weights = weights[track_idx, source_idx]
        for _ in range(n_repeats):
            random_idx = rng.choice(candidate_idx, size=len(source_idx), replace=True)
            sampled_distances.append(distances[track_idx, random_idx, center_idx])
            sampled_weights.append(source_weights)

    if not sampled_distances:
        return {
            "distances": np.array([], dtype=np.float32),
            "weights": np.array([], dtype=np.float32),
        }

    return {
        "distances": np.concatenate(sampled_distances).astype(np.float32),
        "weights": np.concatenate(sampled_weights).astype(np.float32),
    }


@jax.jit
def get_num_den(values, weights, center_idx):
    center_valid = jnp.isfinite(values[..., center_idx]) & jnp.isfinite(weights)
    valid = jnp.isfinite(values) & center_valid[..., None]
    weighted_values = jnp.where(valid, values * weights[..., None], 0.0)
    weighted_counts = jnp.where(valid, weights[..., None], 0.0)
    num_by_track = np.sum(weighted_values, axis=1)
    den_by_track = np.sum(weighted_counts, axis=1)
    num = np.sum(num_by_track, axis=0)
    den = np.sum(den_by_track, axis=0)
    return num, den


@jax.jit
def comp_pileup(values, weights, center_idx):
    num, den = get_num_den(values, weights, center_idx)
    return num / den


@jax.jit
def get_weights_and_vals_forcenter_hist(values, weights, center_idx):
    center_valid = jnp.isfinite(values[..., center_idx]) & jnp.isfinite(weights)
    center_value = jnp.where(center_valid, values[..., center_idx], 0.0)
    weighted_counts = jnp.where(center_valid, weights, 0.0)
    return center_value, weighted_counts


def agg_stats(
    conditions,
    treatments_to_fit,
    time_interval,
    window_size,
    kon,
    rc,
    fit_type,
):
    center_idx = window_size // 2
    dt = float(time_interval.strip("s")) / 60
    window_time = (np.arange(window_size) - center_idx) * dt
    num_ms2 = np.zeros(window_size)
    num_dist = np.zeros(window_size)
    den_ms2 = np.zeros(window_size)
    den_dist = np.zeros(window_size)
    num_ms2_random, den_ms2_random = np.zeros(window_size), np.zeros(window_size)
    num_dist_random, den_dist_random = np.zeros(window_size), np.zeros(window_size)
    agg_dists, agg_turnons, agg_ms2 = [], [], []
    agg_random_dist, agg_random_dist_weights = [], []
    predictions = []
    for condition in conditions:
        for treatment in treatments_to_fit:
            if (fit_type, condition, treatment, time_interval, kon, rc) in pileups:
                vals = pileups[(fit_type, condition, treatment, time_interval, kon, rc)]
                ms2_n, ms2_d = get_num_den(vals[1], vals[2], center_idx)
                dist_n, dist_d = get_num_den(vals[0], vals[2], center_idx)
                ms2_mean = safe_divide(np.asarray(ms2_n), np.asarray(ms2_d))
                dist_mean = safe_divide(np.asarray(dist_n), np.asarray(dist_d))
                num_ms2 += np.nan_to_num(ms2_mean, nan=0.0)
                den_ms2 += np.isfinite(ms2_mean)
                num_dist += np.nan_to_num(dist_mean, nan=0.0)
                den_dist += np.isfinite(dist_mean)
                nwindows = vals[0].shape[0] * vals[0].shape[1]
                agg_dists.append(vals[0].reshape(nwindows, window_size))

                hist_weights = vals[2].flatten()
                hist_weight_sum = np.nansum(hist_weights)
                if hist_weight_sum > 0:
                    hist_weights = hist_weights / hist_weight_sum
                agg_turnons.append(hist_weights)
                agg_ms2.append(vals[1].reshape(nwindows, window_size))
                ms2_random_mean = safe_divide(
                    np.sum(vals[4]["num"], axis=0),
                    np.sum(vals[4]["den"], axis=0),
                )
                dist_random_mean = safe_divide(
                    np.sum(vals[3]["num"], axis=0),
                    np.sum(vals[3]["den"], axis=0),
                )
                random_dist = random_center_distance_samples(
                    windowed_distances=vals[0],
                    center_weights=vals[2],
                    center_idx=center_idx,
                    rng=np.random.default_rng(random_seed),
                    n_repeats=random_control_repeats,
                )
                random_dist_weights = random_dist["weights"]
                random_dist_weight_sum = np.nansum(random_dist_weights)
                if random_dist_weight_sum > 0:
                    random_dist_weights = random_dist_weights / random_dist_weight_sum
                agg_random_dist.append(random_dist["distances"])
                agg_random_dist_weights.append(random_dist_weights)
                num_ms2_random += np.nan_to_num(ms2_random_mean, nan=0.0)
                den_ms2_random += np.isfinite(ms2_random_mean)
                num_dist_random += np.nan_to_num(dist_random_mean, nan=0.0)
                den_dist_random += np.isfinite(dist_random_mean)
                predictions.append(vals[5])

    arr_agg_dists = jnp.concatenate(agg_dists)
    arr_agg_turnons = jnp.concatenate(agg_turnons)
    arr_agg_ms2 = jnp.concatenate(agg_ms2)
    arr_agg_random_dist = np.concatenate(agg_random_dist)
    arr_agg_random_dist_weights = np.concatenate(agg_random_dist_weights)
    agg_stats = {
        "prediction": np.mean(predictions, axis=0),
        "window_time": window_time,
        "num_dist": num_dist,
        "den_dist": den_dist,
        "num_ms2": num_ms2,
        "den_ms2": den_ms2,
        "num_ms2_random": num_ms2_random,
        "den_ms2_random": den_ms2_random,
        "num_dist_random": num_dist_random,
        "den_dist_random": den_dist_random,
        "arr_agg_dists": arr_agg_dists,
        "arr_agg_turnons": arr_agg_turnons,
        "arr_agg_ms2": arr_agg_ms2,
        "agg_random_dist": arr_agg_random_dist,
        "agg_random_dist_weights": arr_agg_random_dist_weights,
        "center_idx": center_idx,
        "window_size": window_size,
    }
    return agg_stats


conditions = (
    "85kb",
    # "340kb_Ce_Cp",
    "170kb",
    "340kb_Ce",
    "255kb",
    "340kb_Cp",
    "340kb",
)
treatments_to_fit = ["None"]
possible_time_intervals = ["30s", "5s"]
types_to_test = ["Regular", "Shifted", "Shuffled"]
minlength = 50
nsteps_to_run = 20
rcs_to_test = [265.6, 40][::-1]
window_size = 51
window_size5s = 101
kons_to_test = [2.31e-2, 1.15][::-1]
random_control_repeats = 100
random_seed = 12345
verbose = True
overwrite = False
center_idx = window_size // 2 
pileups = {}
prediction_cache = {}

for fit_type in types_to_test:
    for condition in conditions:
        for treatment in treatments_to_fit:
            condition_treatment_path = RESULT_PATH / condition / treatment
            for time_interval in possible_time_intervals:
                window_size_used = window_size5s if time_interval == "5s" else window_size
                for kon, rc in zip(kons_to_test, rcs_to_test):
                    savename = f"pileups_{window_size}_{fit_type}_{condition}_{treatment}_{time_interval}_rc{rc}.pkl"
                    if not (OUTPUT_PATH / savename).exists() or overwrite:
                        rundir = condition_treatment_path / f"rc={rc}"
                        fit_name = f"{time_interval}_{condition}_{treatment}"
                        
                        
                        
                        if fit_type == "Regular":
                            rc_fit_name = f"{fit_name}_rc={rc}_kon={kon:g}"
                        else:
                            rc_fit_name = f"{fit_name}_{fit_type}_rc={rc}_kon={kon:g}"
                        
                        model_result_path = rundir / f"{rc_fit_name}_model_results.npz"
                        if not model_result_path.exists():
                            if verbose:
                                print(f"Missing {model_result_path}. Skipping.")
                            continue
                        model_res = load_model_res(model_result_path)
                        if model_res is None:
                            if verbose:
                                print(
                                    f"Failed to load model results for {model_result_path}. Skipping."
                                )
                            continue

                        distance = model_res["distance"]
                        ms2 = model_res["ms2"]
                        turn_on = model_res["turn_on"]
                        observation_times = model_res["observation_times"]
                        dt = time_interval_seconds(time_interval) / 60.0
                        window_time = (np.arange(window_size) - center_idx) * dt
                        msd_fit = load_fd_msd_fit_params(
                            condition,
                            treatment,
                            time_interval,
                            position_scale=1000.0,
                            time_units="minutes",
                        )
                        if msd_fit is None:
                            continue
                        prediction_key = (
                            condition,
                            treatment,
                            time_interval,
                            float(rc),
                            len(window_time),
                        )
                        if prediction_key not in prediction_cache:
                            prediction_cache[prediction_key] = (
                                compute_instantaneous_contact_prediction(
                                    msd_fit,
                                    window_time,
                                    rc=rc,
                                )
                            )
                        prediction = prediction_cache[prediction_key]

                        distance_windowed = np.lib.stride_tricks.sliding_window_view(
                            distance, window_shape=window_size, axis=1
                        )
                        ms2_windowed = np.lib.stride_tricks.sliding_window_view(
                            ms2, window_shape=window_size, axis=1
                        )
                        turn_on_windowed = np.lib.stride_tricks.sliding_window_view(
                            turn_on, window_shape=window_size, axis=1
                        )

                        random_stats_dist = random_center_window_stats(
                            windowed_values=distance_windowed,
                            center_weights=turn_on_windowed[..., center_idx],
                            center_idx=center_idx,
                            rng=np.random.default_rng(random_seed),
                            n_repeats=random_control_repeats,
                        )
                        random_stats_ms2 = random_center_window_stats(
                            windowed_values=ms2_windowed,
                            center_weights=turn_on_windowed[..., center_idx],
                            center_idx=center_idx,
                            rng=np.random.default_rng(random_seed),
                            n_repeats=random_control_repeats,
                        )

                        resdict = {
                            "distance": distance_windowed,
                            "ms2": ms2_windowed,
                            "turn_on": turn_on_windowed,
                            "random_stats_dist": random_stats_dist,
                            "random_stats_ms2": random_stats_ms2,
                            "prediction": prediction,
                        }

                        with open(OUTPUT_PATH / savename, "wb") as f:
                            pickle.dump(resdict, f)
                    else:
                        with open(OUTPUT_PATH / savename, "rb") as f:
                            resdict = pickle.load(f)
                            distance_windowed = resdict["distance"]
                            ms2_windowed = resdict["ms2"]
                            turn_on_windowed = resdict["turn_on"]
                            random_stats_dist = resdict["random_stats_dist"]
                            random_stats_ms2 = resdict["random_stats_ms2"]
                            prediction = resdict["prediction"]
                    pileups[
                        (fit_type, condition, treatment, time_interval, kon, rc)
                    ] = (
                        distance_windowed,
                        ms2_windowed,
                        turn_on_windowed[..., center_idx],
                        random_stats_dist,
                        random_stats_ms2,
                        prediction,
                    )
 

treatment = "None"
time_interval = "30s"
kon, rc = kons_to_test[0], rcs_to_test[0]
fit_type = "Regular"
conds_minus_Ce_Cp = [cond for cond in conditions if cond not in ["340kb_Ce_Cp"]]


def plot_burst_pileup(stats, out_dir=None, savename=None):

    colors_pt = {
        "blue": "#0077BB",
        "magenta": "#EE3377",
        "teal": "#009988",
        "orange": "#EE7733",
        "cyan": "#33BBEE",
        "red": "#CC3311",
        "grey": "#BBBBBB",
    }

    pt_heatmap_cmap = LinearSegmentedColormap.from_list(
        "PT_Inferno",
        [
            (0.00, "#000000"),
            (0.35, colors_pt["blue"]),
            (0.65, colors_pt["magenta"]),
            (0.85, colors_pt["orange"]),
            (1.00, "#FFDDAA"),
        ],
    )
    arr_agg_dists = stats["arr_agg_dists"]
    arr_agg_turnons = stats["arr_agg_turnons"]
    arr_agg_ms2 = stats["arr_agg_ms2"]
    agg_random_dist = stats["agg_random_dist"]
    agg_random_dist_weights = stats["agg_random_dist_weights"]
    window_size = stats["window_size"]
    center_idx = stats["center_idx"]
    window_time = stats["window_time"]
    num_dist, den_dist = stats["num_dist"], stats["den_dist"]
    num_ms2, den_ms2 = stats["num_ms2"], stats["den_ms2"]
    num_dist_random, den_dist_random = (
        stats["num_dist_random"],
        stats["den_dist_random"],
    )
    num_ms2_random, den_ms2_random = stats["num_ms2_random"], stats["den_ms2_random"]
    # make histograms for each window index
    dmin, dmax = 0, 300
    ms2_min, ms2_max = -5, 15
    nbins = 100
    hist_vals = []
    hist_ms2_vals = []
    for i in range(window_size):
        valid = (~jnp.isnan(arr_agg_dists[:, i])) * (~jnp.isnan(arr_agg_turnons))
        vals, edges = np.histogram(
            arr_agg_dists[:, i][valid],
            weights=arr_agg_turnons[valid],
            bins=nbins,
            range=(dmin, dmax),
        )
        hist_vals.append(vals)
        valid_ms2 = (~jnp.isnan(arr_agg_ms2[:, i])) * (~jnp.isnan(arr_agg_turnons))
        vals, edges = np.histogram(
            arr_agg_ms2[:, i][valid_ms2],
            weights=arr_agg_turnons[valid_ms2],
            bins=nbins,
            range=(ms2_min, ms2_max),
        )
        hist_ms2_vals.append(vals)

    fig, ax = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
    im1 = ax[0].imshow(
        np.array(hist_vals).T,
        aspect="auto",
        extent=[
            -window_time[-1],
            window_time[-1],
            dmin,
            dmax,
        ],
        origin="lower",
        interpolation="none",
        cmap=pt_heatmap_cmap,
    )
    ax[0].plot(window_time, num_dist / den_dist, color="white", label="Average")
    ax[0].plot(
        window_time,
        num_dist_random / den_dist_random,
        color="white",
        linestyle="dashed",
        label="Random",
    )
    ax[0].plot(
        window_time,
        stats["prediction"],
        color="white",
        linestyle="dotted",
        label="Prediction",
    )
    ax[1].plot(window_time, num_ms2 / den_ms2, color="white", label="Average")
    ax[1].plot(
        window_time,
        num_ms2_random / den_ms2_random,
        color="white",
        linestyle="dashed",
        label="Random",
    )
    im2 = ax[1].imshow(
        np.array(hist_ms2_vals).T,
        aspect="auto",
        extent=[
            -window_time[-1],
            window_time[-1],
            ms2_min,
            ms2_max,
        ],
        origin="lower",
        interpolation="none",
        cmap=pt_heatmap_cmap,
    )
    ax[1].legend(loc="lower right")
    ax[0].legend(loc="lower right")
    # add colorbars
    cbar0 = plt.colorbar(im1, ax=ax[0])
    cbar1 = plt.colorbar(im2, ax=ax[1])
    cbar0.set_label("Density [1/nm]", rotation=270, labelpad=20)
    cbar1.set_label("Density [1/a.u.]", rotation=270, labelpad=20)

    proximal_control = np.load(DATAPATH / "30s_0.395kb_noE_noP_None.npz")
    dist_control = np.linalg.norm(
        proximal_control["enhancer_coordinate"]
        - proximal_control["promoter_coordinate"],
        axis=-1,
    )
    nan_mask = np.isnan(dist_control)
    ax[0].set(ylabel="Enhancer-Promoter Distance [nm]")
    ax[1].set(xlabel="Time around window center (Minutes)", ylabel="MS2 Signal [a.u.]")
    fig.tight_layout()
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{savename[0]}.pdf")

    rmin = 0
    rmax = 700
    nbins = 50
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    vals, weight = arr_agg_dists[:, center_idx], arr_agg_turnons
    valid = (~np.isnan(vals)) * (~np.isnan(arr_agg_turnons))
    vals, weights = vals[valid], weight[valid]
    _ = ax.hist(
        dist_control[~nan_mask],
        bins=nbins,
        range=(rmin, rmax),
        color="#EE7733",
        density=True,
        label="Proximal control",
    )
    _ = ax.hist(
        vals.flatten(),
        weights=weights.flatten(),
        color="#0077BB",
        bins=nbins,
        density=True,
        alpha=0.5,
        range=(rmin, rmax),
        label="Turn-on-weighted\ndistance",
    )
    random_valid = np.isfinite(agg_random_dist) & np.isfinite(agg_random_dist_weights)
    _ = ax.hist(
        agg_random_dist[random_valid],
        weights=agg_random_dist_weights[random_valid],
        color="#009988",
        bins=nbins,
        density=True,
        alpha=0.5,
        range=(rmin, rmax),
        label="Random-window\ndistance",
    )
    ax.set(xlabel="Enhancer-Promoter Distance [nm]", ylabel="Density [1/nm]")
    ax.legend()
    fig.tight_layout()
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{savename[1]}.pdf")


# plot the main fig pileup
stats = agg_stats(conditions, ["None"], time_interval, window_size, kon, rc, fit_type)
main_fig_dir = SCRIPT_ROOT / "figures" / "main_figure"
plot_burst_pileup(
    stats, main_fig_dir, ("pileup", "dip_hist")
)  # ,",save=True")

# plot the 5s pileup
stats = agg_stats(conditions,["None"],"5s",window_size,kon,rc,fit_type)
main_fig_dir = SCRIPT_ROOT / "figures"/"main_figure"
plot_burst_pileup(stats,main_fig_dir, ("pileup_5s", "dip_hist_5s")
)



# stats = agg_stats(conditions,["None"],"5s",window_size,kon,rc,"Shifted")
# curve_shifted = stats["num_dist"]/stats["den_dist"]
# stats = agg_stats(conditions,["None"],"5s",window_size,kon,rc,"Regular")
# curve_regular = stats["num_dist"]/stats["den_dist"]
# stats = agg_stats(conditions,["None"],"5s",window_size,kon,rc,"Shuffled")
# curve_shuffle = stats["num_dist"]/stats["den_dist"]
# pred = stats["prediction"]
# plt.plot(stats["window_time"],curve_regular,label="Regular")
# plt.plot(stats["window_time"],curve_shifted,label="Shifted")
# plt.plot(stats["window_time"],curve_shuffle,label="Shuffled")
# plt.plot(stats["window_time"],pred,label="Prediction")
# plt.legend()

# plot_burst_pileup(stats)
