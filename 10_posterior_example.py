import sys
import os
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
MAIN_FIG_DIR = SCRIPT_ROOT / "figures" / "main_figure"
MAIN_FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = SCRIPT_ROOT / "cache" / "10_posterior_example"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MSD_PATH = SCRIPT_ROOT / "cache" / "MSD_fits" / "fit_results_individual_v28_StatisticalFilter_Alpha0p37_final"
MS2POSTERIOR_ROOT = SCRIPT_ROOT / "MS2Posterior"
TWOLOCUSGPR_ROOT = SCRIPT_ROOT / "TwoLocusGPR"
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(MS2POSTERIOR_ROOT))
sys.path.insert(0, str(TWOLOCUSGPR_ROOT))
from TwoLocusGPR.GPR import GPR
from TwoLocusGPR.MSD_functions import FD_MSD
from TwoLocusGPR.Posterior_analysis import (
    batched_radius_percentile_comp,
)
from MS2Posterior.Pol2_sampler_jax import sample_loadings
from script_utils import (
    _lookup_param,
    reassign_tracks_by_duration,
    time_interval_label,
    time_interval_seconds,
)
import numpy as np
import matplotlib.pyplot as plt
import pickle

SAMPLE_CACHE_PATH = CACHE_DIR / "sample_loadings_track8_170kb_None_30s_rc40_kon1p15_v1.npz"
SAMPLE_CACHE_KEYS = (
    "posterior_rate",
    "pred_sig",
    "n_particles_trace",
    "E_trace",
    "Count_grid",
    "S",
    "mean_E",
)
SAMPLE_NREPEAT = int(os.environ.get("POSTERIOR_EXAMPLE_NREPEAT", "512"))
SAMPLE_NITER = int(os.environ.get("POSTERIOR_EXAMPLE_NITER", "1024"))
SAMPLE_SEED = int(os.environ.get("POSTERIOR_EXAMPLE_SEED", "0"))

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

RESULT_PATH = SCRIPT_ROOT / "result" / "2_run_inference"
TWOLOCUSGPR_ROOT = SCRIPT_ROOT / "TwoLocusGPR"






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


def load_model_res(model_result_path):
    with np.load(model_result_path) as model_res:
        required = {
            "original_ep_data",
            "original_ms2_data",
            "turnon",
        }
        p2_load = np.asarray(model_res["pol2_loading_posterior"], dtype=float)
        pcont = np.asarray(model_res["pcont"], dtype=float)
        missing = required.difference(model_res.files)
        if missing:
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
        pon = np.asarray(model_res["pon"], dtype=float)
        predicted_ms2 = np.asarray(model_res["predicted_ms2"], dtype=float)
        MAP = np.asarray(model_res["MAP"], dtype=float)
    return {
        "ep_data": ep_data,
        "distance": distance,
        "ms2": ms2,
        "turnon": turn_on,
        "observation_times": observation_times,
        "pol2_loading_posterior": p2_load,
        "pcont": pcont,
        "pon": pon,
        "predicted_ms2": predicted_ms2,
        "MAP": MAP,
    }


condition = "170kb"
time_interval = "30s"
treatment = "None"
rc = 40
kon = 1.15
fit_type = "Regular"

condition_treatment_path = RESULT_PATH / condition / treatment

rundir = condition_treatment_path / f"rc={rc}"
fit_name = f"{time_interval}_{condition}_{treatment}"
if fit_type == "Regular":
    rc_fit_name = f"{fit_name}_rc={rc}_kon={kon:g}"
else:
    rc_fit_name = f"{fit_name}_{fit_type}_rc={rc}_kon={kon:g}"
model_result_path = rundir / f"{rc_fit_name}_model_results.npz"

model_results = load_model_res(model_result_path)

pon = model_results["pon"]
predicted_ms2 = model_results["predicted_ms2"]
MAP = model_results["MAP"]
observation_times = model_results["observation_times"]
turnon = model_results["turnon"]
ep_dat, ms2_data = model_results["ep_data"], model_results["ms2"]
pcont = model_results["pcont"]
p2_loadings = model_results["pol2_loading_posterior"]
msd_fit = load_fd_msd_fit_params(
    condition,
    treatment,
    time_interval,
    position_scale=1000.0,
    time_units="minutes",
)


paramvec = msd_fit["compact_gpr_params"]
predictor = GPR(
    observation_times,
    FD_MSD,
    3,
    param_layout=((), (), ()),
    noise_layout=("dim",),
    param_names=("tau", "J", "alpha"),
)
means, vars = predictor.Predict(paramvec, observation_times, ep_dat)
# 8 170

ind = 8
percentiles = batched_radius_percentile_comp(
    means[ind][None, :], vars[ind][None, :], verbose=False
)

model_pkl_path = rundir / f"{rc_fit_name}_model_object.pkl"
with open(model_pkl_path, "rb") as f:
    model = pickle.load(f)

# fine_grid = np.linspace(observation_times[0], observation_times[-1], 10000)

self = model.G.nodes["Polymerase Loadings"]
# upsampled_rate = np.interp(fine_grid, observation_times, self.rate[ind])


def load_or_sample_polymerase_trace():
    if SAMPLE_CACHE_PATH.exists():
        with np.load(SAMPLE_CACHE_PATH) as cached:
            return tuple(cached[key] for key in SAMPLE_CACHE_KEYS)

    sampled = sample_loadings(
        self.data[ind][None, :],
        self.noise_std[ind],
        self.T_rise,
        self.T_plateau,
        self.sampling_times,
        self.rate[ind][None, :],
        self.time_grid,
        seed=SAMPLE_SEED,
        RNA_int=self.RNA_int,
        nrepeat=SAMPLE_NREPEAT,
        n_iter=SAMPLE_NITER,
    )
    np.savez_compressed(
        SAMPLE_CACHE_PATH,
        **{key: np.asarray(value) for key, value in zip(SAMPLE_CACHE_KEYS, sampled)},
        cache_version=np.asarray(1),
        seed=np.asarray(SAMPLE_SEED),
        nrepeat=np.asarray(SAMPLE_NREPEAT),
        n_iter=np.asarray(SAMPLE_NITER),
    )
    return sampled


(
    self.posterior_rate,
    self.pred_sig,
    n_particles_trace,
    E_trace,
    Count_grid,
    S,
    self.mean_E,
) = load_or_sample_polymerase_trace()

std = np.std(S, axis=0)

start_time = 160
duration = 74
fig = plt.figure(figsize=(5.0, 10))
# make ax0 and ax3 lie on a marger of the first two and last tworows in a 6x1 grid and ax1,ax2 take the rows 3 and 4. use gridspec
gs = fig.add_gridspec(6, 1, hspace=0.5)
ax0 = fig.add_subplot(gs[0:2, 0])
ax1 = fig.add_subplot(gs[2:3, 0])
ax2 = fig.add_subplot(gs[3:4, 0])
ax3 = fig.add_subplot(gs[4:6, 0])
ax = [ax0, ax1, ax2, ax3]

# fig, ax = plt.subplots(4, 1, figsize=(5.0, 7.5), sharex=True)
ax[0].plot(
    observation_times - start_time,
    np.linalg.norm(ep_dat[ind], axis=-1),
    ".",
    label="Data",
    alpha=0.5,
)
ax[0].plot(
    observation_times - start_time,
    percentiles[0, :, 1],
    label="Posterior median",
    color="C0",
)
ax[0].fill_between(
    observation_times - start_time,
    percentiles[0, :, 0],
    percentiles[0, :, -1],
    color="C0",
    alpha=0.2,
    label=f"95% CI",
)

ax[0].axhline(rc, linestyle="--", color="red", label="R_EP")
dt = observation_times[1] - observation_times[0]
ax[0].set(
    ylim=(0, 500),
    ylabel="E-P distance (nm)",
    xlim=(0, np.max(observation_times[~np.isnan(ms2_data[ind])]) + dt),
)
ax[0].legend()

ax[1].plot(observation_times - start_time, pcont[ind])
ax[1].set(ylabel=f"P(contact)")


ax[2].fill_between(
    observation_times - start_time,
    0,
    pon[ind],
    label="Promoter state",
    color="C1",
    alpha=0.2,
    zorder=-1,
)

ax[2].fill_between(
    observation_times - start_time,
    0,
    turnon[ind] / np.nanmax(turnon[ind]),
    label="Posterior on-rate",
    color="dimgrey",
    alpha=0.8,
    zorder=1,
)
ax[2].set(ylabel="Promoter state", ylim=(0, 1.1))
ax[3].plot(
    observation_times - start_time,
    ms2_data[ind],
    ".",
    label="MS2 data",
    alpha=0.5,
    zorder=4,
)
ax[3].fill_between(
    observation_times - start_time,
    0,
    p2_loadings[ind] * 5,
    label="Pol2 loadings",
    color="dimgrey",
    alpha=0.2,
    zorder=-1,
)
ax[3].plot(
    observation_times - start_time,
    predicted_ms2[ind],
    label="Predicted MS2",
    color="C0",
    zorder=3,
)
ax[3].fill_between(
    observation_times - start_time,
    predicted_ms2[ind] - 1.96 * std,
    predicted_ms2[ind] + 1.96 * std,
    color="darkgrey",
    alpha=0.5,
    label="90% CI",
)

ax[3].set(xlabel="Time (min)", ylabel="MS2 intensity", ylim=(-1, 20))
ax[3].legend()
for a in ax:
    a.set(xlim=(0, duration))
    # remove upper and right spines
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
# fig.tight_layout()

fig.savefig(MAIN_FIG_DIR / "example.pdf")
