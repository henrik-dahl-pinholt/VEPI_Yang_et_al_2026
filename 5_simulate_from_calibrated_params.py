import os
import pickle
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import sys
SCRIPT_ROOT = Path(__file__).resolve().parent
TWOLOCUSGPR_ROOT = SCRIPT_ROOT / "TwoLocusGPR"
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(TWOLOCUSGPR_ROOT))
import matplotlib.pyplot as plt
import numpy as np

import jax
from jax import numpy as jnp

from TwoLocusGPR.GPR import GPR
from TwoLocusGPR.MSD_functions import FD_MSD
from TwoLocusGPR.GPR_utils import msd

from typing import Dict, Tuple
import numpy as np
import json
from script_utils import _lookup_param, time_interval_label, time_interval_seconds
from simulation_utils import _sample_loading_events, _sample_promoter_states
from jax_script_utils import cross_corr_jax
CORR_FIT_PATH = SCRIPT_ROOT / "result" / "1_MS2_corrfit" / "fit_raw_intensity_0341338401d33fe53cf19f91401d8dac_summary.json"

RESULT_PATH = SCRIPT_ROOT / "result" / "5_simulate_from_calibrated_params"
INFERENCE_RESULT_PATH = SCRIPT_ROOT / "result" / "2_run_inference"
MSD_PATH = SCRIPT_ROOT / "cache" / "MSD_fits" / "fit_results_individual_v28_StatisticalFilter_Alpha0p37_final"
DATA_ROOT = SCRIPT_ROOT / "Data" / "filtered_data_v6"

NPY_CONDITION_MAPPING = {
    "85kb_None": "15C-A2_GSK",
    '340kb_IAA': 'G7B8G2_GSK_100uM_IAAadded2hrBefore',
    '340kb_dTAG': 'G7B8G2_GSK_500nMdTAG13added2hrBefore',
    '340kb_None': 'S+V-A6B8_GSK',
    '255kb_None': '15A-A6G9_GSK',
    '170kb_None': '15B-18G9_GSK',
    '340kb_Cp_None': 'SCre-G5-H7_GSK',
    '340kb_Ce_None': 'VCre-C2b_GSK',
    '340kb_Ce_Cp_None': 'G7B8G2_GSK',
    '340kb_Ce_Cp_noE_noP_None': 'E11G8_GSK',
    '340kb_Ce_Cp_noE_None': 'Vika-H11_GSK',
    '2.362kb_noE_noP_None': '14A-A11_GSK',
    '1.5kb_None':'14B5-F10',
    '0.395kb_noE_noP_None':'14A-A11E6_GSK',
    '30s_85kb_IAA':'15C-A2_GSK_IAA(100uM)added2hrBefore',
    '30s_170kb_IAA':'15B-18G9_GSK_IAA(100uM)added2hrBefore',
    '30s_255kb_IAA':'15A-A6G9_GSK_IAA(100uM)added2hrBefore',
    '30s_340kb_IAA':'S+V-A6B8_GSK_IAA(100uM)added2hrBefore',
    '30s_340kb_Ce_Cp_IAAdTAG':'G7B8G2_GSK_dTAG13(500uM)andIAA(100uM)added2hrBefore'
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


def read_data(condition, intensity_key, time_interval, treatment):

    time_interval_seconds = float(str(time_interval).removesuffix("s"))
    path = DATA_ROOT / f"{time_interval_seconds:g}s_{condition}_{treatment}.npz"
    if not path.exists():
        return None, None
    loaded_data = np.load(path, allow_pickle=True)
    ep_dat = loaded_data["enhancer_coordinate"]-loaded_data["promoter_coordinate"]
    ms2_data = loaded_data[intensity_key]
    
    return ep_dat, ms2_data



def _build_ms2_signal(
    observation_times: np.ndarray,
    loading_times: np.ndarray,
    t_increase: float,
    t_plateau: float,
    RNA_intensity: float,
    ms2_noise: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n_obs = observation_times.shape[0]
    signal = np.zeros(n_obs, dtype=np.float64)
    for event_time in loading_times:
        for i in range(n_obs):
            dt = observation_times[i] - event_time
            if dt < 0.0:
                continue
            if dt <= t_increase:
                signal[i] += RNA_intensity * dt / t_increase
            elif dt <= t_increase + t_plateau:
                signal[i] += RNA_intensity
    noisy = np.empty(n_obs, dtype=np.float64)
    for i in range(n_obs):
        noisy[i] = signal[i] + ms2_noise * np.random.normal()
    return signal, noisy

def _simulate_fromR_with_events(
    epfunc,
    kon,
    koff,
    t_max,
    rc,
    pol2_loading_rate,
    RNA_intensity,
    ms2_noise,
    t_increase,
    t_plateau,
    activation_delay,
    p_contact,
    measurement_interval,
):
    n_events = np.random.poisson(kon * t_max)
    on_times = t_max * np.random.random(size=n_events)
    if n_events > 1:
        on_times.sort()

    observation_times = np.arange(
        0.0, t_max + measurement_interval, measurement_interval
    )
    event_r = epfunc(on_times,observation_times)
    contact_on_times = on_times[event_r < rc]
    thinned_on_times = contact_on_times + activation_delay
    thinned_on_times = thinned_on_times[thinned_on_times <= t_max]
    if thinned_on_times.shape[0] > 1:
        thinned_on_times.sort()

    state_sequence, state_times = _sample_promoter_states(
        p_contact=p_contact,
        kon=kon,
        koff=koff,
        t_max=t_max,
        thinned_on_times=thinned_on_times,
        nstates=2,
    )
    loading_times = _sample_loading_events(
        times=state_times,
        states=state_sequence,
        state_vals=pol2_loading_rate,
        t_max=t_max,
    )
    ms2_signal, ms2_noisy = _build_ms2_signal(
        observation_times=observation_times,
        loading_times=loading_times,
        t_increase=t_increase,
        t_plateau=t_plateau,
        RNA_intensity=RNA_intensity,
        ms2_noise=ms2_noise,
    )

    transitions = np.diff(state_sequence)
    event_payload = {
        "candidate_on_times": on_times,
        "contact_on_times": contact_on_times,
        "true_on_times": state_times[1:][transitions == 1],
        "true_off_times": state_times[1:][transitions == -1],
        "state_times": state_times,
        "state_sequence": state_sequence,
        "loading_times": loading_times,
    }
    return observation_times, ms2_signal, ms2_noisy, event_payload


def _exact_event_metadata_from_lists(event_payloads):
    return {
        "candidate_on_times_by_trace": [
            np.asarray(payload["candidate_on_times"], dtype=np.float64)
            for payload in event_payloads
        ],
        "contact_on_times_by_trace": [
            np.asarray(payload["contact_on_times"], dtype=np.float64)
            for payload in event_payloads
        ],
        "true_on_times_by_trace": [
            np.asarray(payload["true_on_times"], dtype=np.float64)
            for payload in event_payloads
        ],
        "true_off_times_by_trace": [
            np.asarray(payload["true_off_times"], dtype=np.float64)
            for payload in event_payloads
        ],
        "state_times_by_trace": [
            np.asarray(payload["state_times"], dtype=np.float64)
            for payload in event_payloads
        ],
        "state_sequence_by_trace": [
            np.asarray(payload["state_sequence"], dtype=np.int64)
            for payload in event_payloads
        ],
        "loading_times_by_trace": [
            np.asarray(payload["loading_times"], dtype=np.float64)
            for payload in event_payloads
        ],
    }



def simulate_dataset(
    rc,
    ntrajectories,
    T,
    pol2_loading_rates,
    RNA_intensity,
    MS2_noises,
    T_increase,
    T_plateau,
    measurement_interval,
    tau,
    J,
    alpha,
    localization_errors,
    kon_true,
    koff,
    return_metadata=True,
):
    # fine_times = np.arange(
    #     0, T + measurement_interval / grid_factor, measurement_interval / grid_factor
    # )
    simpath = f"cache/5_simulate_from_calibrated_params{rc}.pkl"

    ms2_dat = []
    ms2_noisy_dat = []
    ep_dat_list = []
    ep_dat_noisy_list = []
    ms2_dat_fine = []
    ms2_noisy_dat_fine = []
    ep_dat_list_fine = []
    ep_dat_noisy_list_fine = []
    event_payloads = []
    for i in tqdm(list(range(ntrajectories))):
        class R_func:
            def __call__(self, x,observation_times):
                x_here = np.concatenate([x, observation_times])
                sort_idx = np.argsort(x_here)
                x_sorted = x_here[sort_idx]

                GPR_class = GPR(x_sorted, FD_MSD, 3)
                data, data_noisy = GPR_class.sample_prior(
                    np.array([tau, J, alpha] * 3 + list(localization_errors)),
                    1,
                    seed=i,
                )

                inverse_sort = np.empty_like(sort_idx)
                inverse_sort[sort_idx] = np.arange(sort_idx.size)

                event_pos = inverse_sort[: len(x)]
                obs_pos = inverse_sort[len(x) :]

                sample_at_x = data[0, event_pos]
                observation_data = data[0, obs_pos]
                noisy_observation_data = data_noisy[0, obs_pos]

                self.ep_data = np.asarray(observation_data, dtype=np.float32)
                self.ep_data_noisy = np.asarray(noisy_observation_data, dtype=np.float32)

                return np.linalg.norm(sample_at_x, axis=-1)

        r_func = R_func()
        np.random.seed(i)
        fine_observation_times, ms2_signal, ms2_noisy, event_payload = (
            _simulate_fromR_with_events(
                epfunc=r_func,
                kon=kon_true,
                koff=koff,
                t_max=T,
                rc=rc,
                pol2_loading_rate=pol2_loading_rates[i%len(pol2_loading_rates)],
                RNA_intensity=RNA_intensity,
                ms2_noise=MS2_noises[i%len(pol2_loading_rates)],
                t_increase=T_increase,
                t_plateau=T_plateau,
                activation_delay=0,
                p_contact=0.0,
                measurement_interval=measurement_interval ,
            )
        )
        event_payloads.append(event_payload)

        ms2_dat_fine.append(ms2_signal)
        ms2_noisy_dat_fine.append(ms2_noisy)
        ep_dat_list_fine.append(r_func.ep_data)
        ep_dat_noisy_list_fine.append(r_func.ep_data_noisy)
        ms2_dat.append(ms2_signal)
        ms2_noisy_dat.append(ms2_noisy)
        ep_dat_list.append(r_func.ep_data)
        ep_dat_noisy_list.append(r_func.ep_data_noisy)
        observation_times = fine_observation_times
    sim_data = {
        "ms2_dat": ms2_dat,
        "ms2_noisy_dat": ms2_noisy_dat,
        "ep_dat_list": ep_dat_list,
        "ep_dat_noisy_list": ep_dat_noisy_list,
        "ms2_dat_fine": ms2_dat_fine,
        "ms2_noisy_dat_fine": ms2_noisy_dat_fine,
        "ep_dat_list_fine": ep_dat_list_fine,
        "ep_dat_noisy_list_fine": ep_dat_noisy_list_fine,
        "fine_observation_times": fine_observation_times,
        "observation_times": observation_times,
    }
    sim_data.update(_exact_event_metadata_from_lists(event_payloads))
    with open(simpath, "wb") as handle:
        pickle.dump(sim_data, handle)
    result = (
        ep_dat_list,
        ep_dat_noisy_list,
        ms2_dat,
        ms2_noisy_dat,
    )
    if return_metadata:
        return result + (sim_data,)
    return result


# read the correlation fit results
with open(CORR_FIT_PATH, "r") as f:
    corr_fit_results = json.load(f)
keys = ["koff", "T_rise", "T_plateau", "RNA_intensity", "offset", "loading_rate"]
koff, T_rise, T_plateau, RNA_intensity, offset, loading_rate = [
    corr_fit_results["parameters"][key] for key in keys
]

# read the correlation fit results
with open(CORR_FIT_PATH, "r") as f:
    corr_fit_results = json.load(f)
keys = ["koff", "T_rise", "T_plateau", "RNA_intensity", "offset", "loading_rate"]
koff, T_rise, T_plateau, RNA_intensity, offset, loading_rate = [
    corr_fit_results["parameters"][key] for key in keys
]

treatment = "None"
time_interval = "30s"
conditions = (
    "170kb",
    "85kb",
    "340kb_Ce_Cp",
    "340kb_Ce",
    "255kb",
    "340kb_Cp",
    "340kb",
)
for condition in conditions:
    print(f"Simulating for {condition} {treatment} {time_interval}")

    fit_name = f"{time_interval}_{condition}_{treatment}"

    savedir  = INFERENCE_RESULT_PATH / condition / treatment
    savename = f"{fit_name}"

    res = np.load(savedir / f"{savename}_model_results.npz")
    data_length = res["pon"].shape[0]
    measurement_interval = float(time_interval.rstrip("s"))/60
    pol2_loading_rates = res["loading_rates"].reshape(data_length, -1)[:,::-1]
    T = res["observation_times"][-1]
    MS2_noise = res["noise"]

    # load the msd fit parameters and data for this condition, treatment, and time interval
    msd_fit = load_fd_msd_fit_params(
        condition,
        treatment,
        time_interval,
        position_scale=1000.0,
        time_units="minutes",
    )
    paramvec = msd_fit["compact_gpr_params"]  
    tau, J, alpha = paramvec[:3]
    localization_errors = paramvec[3:]
    ntrajectories = 1000
    lags = np.arange(-200,201)
    simulation_parameter_sets = [
        ("MLE", 265.6, 2.31e-2),
        ("RCest", 40, 1.15),
    ]
    sim_cross_corrs = []
    for label, rc, kon in simulation_parameter_sets:
        print(f"\tSimulating {label} for rc={rc}, kon={kon}")
        savepath = RESULT_PATH/ f"simulated_data_{condition}_{treatment}_{time_interval}_rc{rc}_kon{kon}.pkl"
        if not savepath.parent.exists():
            savepath.parent.mkdir(parents=True)
        if not savepath.exists():
            simres = simulate_dataset(
                rc,
                ntrajectories,
                T,
                pol2_loading_rates,
                RNA_intensity,
                MS2_noise,
                T_rise,
                T_plateau,
                measurement_interval,
                tau,
                J,
                alpha,
                localization_errors,
                kon,
                koff,
            )    
            # unpack the results and save
            (
                sim_ep_dat_list,
                sim_ep_dat_noisy_list,
                sim_ms2_dat,
                sim_ms2_noisy_dat,
            ) = [np.array(x) for x in simres[:4]]
            sim_data = simres[4]
            with open(savepath, "wb") as handle:
                resdict = {
                    "ep_dat_list": sim_ep_dat_list,
                    "ep_dat_noisy_list": sim_ep_dat_noisy_list,
                    "ms2_dat": sim_ms2_dat,
                    "ms2_noisy_dat": sim_ms2_noisy_dat,
                    "sim_data": sim_data,
                }
                pickle.dump(resdict, handle)
        else:
            with open(savepath, "rb") as handle:
                sim_data = pickle.load(handle)
                sim_ep_dat_list = sim_data["ep_dat_list"]
                sim_ep_dat_noisy_list = sim_data["ep_dat_noisy_list"]
                sim_ms2_dat = sim_data["ms2_dat"]
                sim_ms2_noisy_dat = sim_data["ms2_noisy_dat"]
        sim_dist = jnp.linalg.norm(sim_ep_dat_noisy_list,axis=-1)
        sim_cross_corrs.append(
            (label, cross_corr_jax(sim_dist, sim_ms2_noisy_dat, lags))
        )
        
    # load raw data
    ep_dat, ms2_data = read_data(
                    condition, "corrected_intensity", time_interval, treatment
                )

    cross_corr_data = cross_corr_jax(jnp.linalg.norm(ep_dat,axis=-1), ms2_data, lags)

    fig,ax = plt.subplots()
    for label, cross_corr in sim_cross_corrs:
        ax.plot(lags*measurement_interval, cross_corr, label=f"Simulated ({label})")
    ax.plot(lags*measurement_interval, cross_corr_data, label="Data")
    ax.set(xlabel="Lag (minutes)",ylabel="Cross-correlation [Intensity*nm]",title=f"{condition} {treatment} {time_interval}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT_PATH / f"cross_corr_{condition}_{treatment}_{time_interval}.png")
