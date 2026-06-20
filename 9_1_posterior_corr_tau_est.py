from pathlib import Path
import pickle
import sys
import jax
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
import os
from jax import numpy as jnp
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
trapz = getattr(np, "trapezoid", np.trapz)

SCRIPT_ROOT = Path(__file__).resolve().parent
MAIN_FIG_DIR = SCRIPT_ROOT / "figures" / "main_figure"
MAIN_FIG_DIR.mkdir(parents=True, exist_ok=True)
MS2POSTERIOR_ROOT = SCRIPT_ROOT / "MS2Posterior"
TWOLOCUSGPR_ROOT = SCRIPT_ROOT / "TwoLocusGPR"
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(MS2POSTERIOR_ROOT))
sys.path.insert(0, str(TWOLOCUSGPR_ROOT))

MSD_PATH = SCRIPT_ROOT / "cache" / "MSD_fits" / "fit_results_individual_v28_StatisticalFilter_Alpha0p37_final"


from MS2Posterior.Pol2_sampler_jax import sample_loadings
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

from TwoLocusGPR.GPR import GPR
from TwoLocusGPR.MSD_functions import FD_MSD
from script_utils import (
    _lookup_param,
    fit_amplitude_only,
    predict_w0_pon_crosscorr,
    reassign_tracks_by_duration,
    smooth_lag_curve,
    time_interval_label,
    time_interval_seconds,
)
from jax_script_utils import acf_jax, cross_corr_jax

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
RESULT_PATH = SCRIPT_ROOT / "result" / "2_run_inference"

DATAPATH = SCRIPT_ROOT / "Data" / "filtered_data_v6"











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
        lrates = model_res["loading_rates"]
        
    return {
        "ep_data": ep_data,
        "distance": distance,
        "ms2": ms2,
        "turn_on": turn_on,
        "observation_times": observation_times,
        "pol2_loading_posterior": p2_load,
        "pcont": pcont,
        "pon": pon,
        "loading_rates": lrates,
    }


conditions = (
    "85kb",
    # "340kb_Ce_Cp",
    "170kb",
    "340kb_Ce",
    "255kb",
    "340kb_Cp",
    "340kb",
)
treatment = "None"
time_interval = "30s"


nsteps_to_run = 20
rcs_to_test = [40]

kon = 1.15
rc = 40


frame_dt_min = time_interval_seconds(time_interval) / 60.0
lags_30,lags_5 = np.arange(0, 700, 1), np.arange(0, 700, 1)

agg_turnon_auto = []

for time_interval in ["30s", "5s"]:
    interval_agg = []
    lags = lags_30 if time_interval == "30s" else lags_5
    for fit_type in ["Regular", "Shifted", "Shuffled"]:
        turnon_auto = []
        for condition in conditions:
            
            condition_treatment_path = RESULT_PATH / condition / treatment

            rundir = condition_treatment_path / f"rc={rc}"
            fit_name = f"{time_interval}_{condition}_{treatment}"
            if fit_type == "Regular":
                rc_fit_name = f"{fit_name}_rc={rc}_kon={kon:g}"
            else:
                rc_fit_name = f"{fit_name}_{fit_type}_rc={rc}_kon={kon:g}"
            model_result_path = rundir / f"{rc_fit_name}_model_results.npz"
            if not model_result_path.exists():
                print(f"Model result not found: {model_result_path}")
                continue
            model_res = load_model_res(model_result_path)
            
            turnon_auto.append(acf_jax(model_res["turn_on"], lags))
        interval_agg.append(turnon_auto)

    agg_turnon_auto.append(interval_agg)




fig,ax = plt.subplots(2, 1, figsize=(5, 5))
xlims = [(0, 400), (0, 100)]
for i, (time_interval, interval_agg) in enumerate(zip(["30s", "5s"], agg_turnon_auto)):
    dt = float(time_interval.split("s")[0])
    arr_auto =  np.array(interval_agg[0])
    arr_auto = arr_auto/arr_auto[:,0][:,None]
    for k in range(arr_auto.shape[0]):
        if k == 0:
            ax[i].plot([lags_30,lags_5][i]*dt,arr_auto[k],alpha=0.5,color="dimgrey",label="Individual conditions")
        else:
            ax[i].plot([lags_30,lags_5][i]*dt,arr_auto[k],alpha=0.5,color="dimgrey")
    
    avg_auto = np.mean(arr_auto, axis=0)
    ax[i].plot([lags_30,lags_5][i]*dt,avg_auto,"o-",alpha=1.0,color="C0",label="Average")
    trapz_int = trapz(avg_auto, [lags_30,lags_5][i]*dt)
    ax[i].axvline(trapz_int, color="red", linestyle="--",label=r"$\tau_\mathrm{E-P}$"+f": {trapz_int:.2f}")
    ax[i].legend()
    ax[i].set(xlim=xlims[i],xlabel="Time [s]",ylabel="turnon-turnon\nautocorrelation")
fig.tight_layout()
fig.savefig(MAIN_FIG_DIR / "tau_est.pdf")
