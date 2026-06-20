import os
from pathlib import Path
import pickle
import sys
import numpy as np
from tqdm.auto import tqdm

SCRIPT_ROOT = Path(__file__).resolve().parent
TWOLOCUSGPR_ROOT = SCRIPT_ROOT / "TwoLocusGPR"
MS2POSTERIOR_ROOT = SCRIPT_ROOT / "MS2Posterior"
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(TWOLOCUSGPR_ROOT))
sys.path.insert(0, str(MS2POSTERIOR_ROOT))
from TwoLocusGPR.GPR import GPR
from TwoLocusGPR.MSD_functions import FD_MSD
from script_utils import (
    _lookup_param,
    file_ok,
    filter_short_tracks,
    reassign_tracks_by_duration,
    time_interval_label,
    time_interval_seconds,
)
from TwoLocusGPR.Posterior_analysis import (
    prob_in_sphere_quadrature,
    batched_radius_percentile_comp,
)
import matplotlib.pyplot as plt
from MS2Posterior.Variational_State_Finder import cavi_sweep

MSD_PATH = SCRIPT_ROOT / "cache" / "MSD_fits" / "fit_results_individual_v28_StatisticalFilter_Alpha0p37_final"

RESULT_PATH = SCRIPT_ROOT / "result" / "2_run_inference"

DATA_ROOT = SCRIPT_ROOT / "Data" / "filtered_data_v6"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


def Store_Results(model, ms2_data, ep_dat, pcont, save_structure=None):
    pon = model.M.get("Promoter State")["masked_posterior"][1:, :, 0].T
    predicted_ms2 = model.M.get("Polymerase Loadings")["Predicted MS2"]
    turnon = model.M.get("Promoter State")["masked_joint"][:, :, 0, 1]
    MAP = model.M.get("Promoter State")["MAP"][:, 1:]
    MAP = np.where(np.isnan(ms2_data), np.nan, MAP)
    pol2_loading_posterior = model.M.get("Polymerase Loadings")["Pol2_posterior"]

    loading_rates = model.M.get("Loading Rates")["Loading Rates <k>"].reshape(
        len(ms2_data), 2
    )
    transition_rates = model.M.get("Transition Rates")["Transition Rates <k>"]
    fit_data = model.M.get("MS2 Data")["data"]

    T_rise = model.G.nodes["Polymerase Loadings"].T_rise
    T_plateau = model.G.nodes["Polymerase Loadings"].T_plateau
    RNA_intensity = model.G.nodes["Polymerase Loadings"].RNA_int
    noise = model.G.nodes["Polymerase Loadings"].noise_std
    prior_shape = model.G.nodes["Loading Rates"].prior_shape.reshape(len(ms2_data), 2)
    prior_rate = model.G.nodes["Loading Rates"].prior_rate.reshape(len(ms2_data), 2)

    model_results = {
        "pon": pon,
        "predicted_ms2": predicted_ms2,
        "turnon": turnon,
        "MAP": MAP,
        "pcont": pcont,
        "pol2_loading_posterior": pol2_loading_posterior,
        "loading_rates": loading_rates,
        "transition_rates": transition_rates,
        "observation_times": model.sampling_times,
        "fine_grid": model.fine_grid,
        "original_ms2_data": ms2_data,
        "original_ep_data": ep_dat,
        "fit_data": fit_data,
        "T_rise": T_rise,
        "T_plateau": T_plateau,
        "RNA_intensity": RNA_intensity,
        "noise": noise,
        "loading_rate_prior_shape": prior_shape,
        "loading_rate_prior_rate": prior_rate,
        "Initial_Probabilities": np.exp(
            model.M.get("Initial State Probabilities")[
                "Initial State Probabilities <log pi>"
            ]
        )
        / np.sum(
            np.exp(
                model.M.get("Initial State Probabilities")[
                    "Initial State Probabilities <log pi>"
                ]
            ),
            axis=-1,
            keepdims=True,
        ),
    }

    if save_structure is not None:
        savedir, savename = save_structure
        np.savez(savedir / f"{savename}_model_results.npz", **model_results)
        with open(savedir / f"{savename}_model_object.pkl", "wb") as f:
            pickle.dump(model, f)
    return model_results




def read_data(condition, intensity_key, time_interval, treatment):

    time_interval_seconds = float(str(time_interval).removesuffix("s"))
    path = DATA_ROOT / f"{time_interval_seconds:g}s_{condition}_{treatment}.npz"
    if not path.exists():
        return None, None
    loaded_data = np.load(path, allow_pickle=True)
    ep_dat = loaded_data["enhancer_coordinate"] - loaded_data["promoter_coordinate"]
    ms2_data = loaded_data[intensity_key]

    return ep_dat, ms2_data


def plot_rc_predictions(
    model_results,
    ms2_data,
    pcont,
    ep_dat,
    means,
    vars,
    show_plot=False,
    save_structure=None,
):
    pon = model_results["pon"]
    predicted_ms2 = model_results["predicted_ms2"]
    MAP = model_results["MAP"]
    observation_times = model_results["observation_times"]
    turnon = model_results["turnon"]

    if save_structure is not None:
        savedir, savename = save_structure
        # make folder for example plots if it doesn't exist
        example_plot_folder = savedir / "example_plots"
        if not example_plot_folder.exists():
            example_plot_folder.mkdir()

    n_tracks_to_plot = min(
        25,
        len(ms2_data),
        len(ep_dat),
        means.shape[0],
        vars.shape[0],
        pcont.shape[0],
        predicted_ms2.shape[0],
        pon.shape[0],
        MAP.shape[0],
        turnon.T.shape[0],
    )
    indstoplot = np.arange(n_tracks_to_plot)
    for ind in indstoplot:
        percentiles = batched_radius_percentile_comp(
            means[ind][None, :], vars[ind][None, :], verbose=False
        )

        fig, ax = plt.subplots(3, 1, figsize=(16, 7), sharex=True)
        ax[0].plot(
            observation_times,
            np.linalg.norm(ep_dat[ind], axis=-1),
            ".",
            label="E-P distance",
            alpha=0.5,
        )
        ax[0].plot(
            observation_times,
            percentiles[0, :, 1],
            label="Posterior median E-P distance",
            color="C0",
        )
        ax[0].fill_between(
            observation_times,
            percentiles[0, :, 0],
            percentiles[0, :, -1],
            color="C0",
            alpha=0.2,
            label=f"95% credible interval for E-P distance",
        )
        dt = observation_times[1] - observation_times[0]
        ax[0].set(
            ylim=(0, np.nanpercentile(np.linalg.norm(ep_dat, axis=-1), 99)),
            ylabel="E-P distance (nm)",
            xlim=(0, np.max(observation_times[~np.isnan(ms2_data[ind])]) + dt),
        )
        ax[0].legend()

        ax[1].plot(observation_times, pcont[ind])
        ax[1].set(ylabel=f"Probability of E-P contact\n(r<{rc}nm)")

        ax[2].plot(
            observation_times, ms2_data[ind], ".", label="MS2 data", alpha=0.5, zorder=4
        )
        ax[2].plot(
            observation_times,
            predicted_ms2[ind],
            label="Predicted MS2",
            color="C0",
            zorder=3,
        )
        ax[2].fill_between(
            observation_times,
            0,
            pon[ind] * np.nanmax(ms2_data[ind]),
            label="Promoter state",
            color="C1",
            alpha=0.2,
            zorder=-1,
        )
        ax[2].plot(
            observation_times,
            np.abs(1 - MAP[ind]) * np.nanmax(ms2_data[ind]),
            label="MAP path",
            color="C1",
        )
        ax[2].fill_between(
            observation_times,
            0,
            turnon.T[ind] * np.nanmax(ms2_data[ind]),
            label="Posterior on-rate",
            color="dimgrey",
            alpha=0.8,
            zorder=1,
        )
        ax[2].set(xlabel="Time (min)", ylabel="MS2 intensity")
        ax[2].legend()
        fig.tight_layout()
        if save_structure is not None:
            savedir, savename = save_structure
            fig.savefig(example_plot_folder / f"{savename}_track{ind}.png")
        if not show_plot:
            plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    tmaxes = np.array(
        [
            np.max(observation_times[~np.isnan(ms2_data[ind])])
            for ind in range(len(ms2_data))
        ]
    )
    lensort = np.argsort(tmaxes)[::-1]
    ax[0].imshow(
        pon[lensort],
        aspect="auto",
        origin="lower",
        extent=[observation_times[0], observation_times[-1], 0, len(ms2_data)],
        interpolation="none",
    )
    cbar = fig.colorbar(ax[0].images[0], ax=ax[0])
    cbar.set_label(
        "Posterior probability of promoter being ON", rotation=270, labelpad=15
    )
    ax[0].set(
        title="Posterior probability of promoter being ON",
        xlabel="Time (min)",
        ylabel="Track index",
    )
    nanmask = np.isnan(ms2_data)
    masked_MAP = np.where(nanmask, np.nan, MAP)
    ax[1].imshow(
        masked_MAP[lensort],
        aspect="auto",
        origin="lower",
        extent=[observation_times[0], observation_times[-1], 0, len(ms2_data)],
        interpolation="none",
    )
    cbar = fig.colorbar(ax[1].images[0], ax=ax[1])
    cbar.set_label("MAP promoter state (0=OFF, 1=ON)", rotation=270, labelpad=15)
    ax[1].set(
        title="MAP promoter state",
        xlabel="Time (min)",
        ylabel="Track index",
    )
    fig.tight_layout()
    if save_structure is not None:
        savedir, savename = save_structure
        fig.savefig(savedir / f"{savename}_burst_calls_overview.png")
    if not show_plot:
        plt.close(fig)






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


rcs_to_test = [265.6, 40]
kons_to_test = [2.31e-2, 1.15]
conditions = (
    "85kb",
    "340kb_Ce_Cp",
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


for condition in conditions:
    for treatment in treatments_to_fit:
        for time_interval in possible_time_intervals:
            print(
                f"Running inference for condition {condition}, treatment {treatment}, time interval {time_interval}"
            )
            # load the observation time and T_rise ect.
            condition_treatment_path = RESULT_PATH / condition / treatment
            fit_name = f"{time_interval}_{condition}_{treatment}"
            fit_results_path = (
                condition_treatment_path / f"{fit_name}_model_results.npz"
            )
            if not file_ok(fit_results_path):
                print(
                    f"Missing model results for condition {condition}, treatment {treatment}, time interval {time_interval}. Skipping."
                )
                continue
            fit_res = np.load(fit_results_path)
            observation_times = fit_res["observation_times"]
            T_rise, T_plateau = fit_res["T_rise"], fit_res["T_plateau"]
            dt = observation_times[1] - observation_times[0]

            # read the data
            try:
                ep_dat, ms2_data = read_data(
                    condition, "corrected_intensity", time_interval, treatment
                )
                if ep_dat is None or ms2_data is None:
                    print(
                        f"Failed to read data for condition {condition}, treatment {treatment}, time interval {time_interval}. Skipping."
                    )
                    continue
            except FileNotFoundError as e:
                print(
                    f"File not found for condition {condition}, treatment {treatment}, time interval {time_interval}. Skipping."
                )
                continue

            # filter out short tracks
            ms2_data, ep_dat = filter_short_tracks(ms2_data, ep_dat, minlength)
            if len(ms2_data) == 0:
                print(
                    f"\tAfter filtering out tracks shorter than {minlength} time points, no tracks remain for condition {condition}, treatment {treatment}, time interval {time_interval}. Skipping inference."
                )
                continue

            # pad the data to allow for pol2 loadings before the measurement begins
            n_pad = len(observation_times) - ms2_data.shape[1]
            if n_pad < 0:
                print(
                    f"Data are longer than model observation times for condition {condition}, treatment {treatment}, time interval {time_interval}. Skipping."
                )
                continue
            ms2_data = np.pad(ms2_data, ((0, 0), (n_pad, 0)), constant_values=np.nan)
            ep_dat = np.pad(
                ep_dat,
                ((0, 0), (n_pad, 0), (0, 0)),
                constant_values=np.nan,
            )

            # load the msd fit parameters and data for this condition, treatment, and time interval
            try:
                msd_fit = load_fd_msd_fit_params(
                    condition,
                    treatment,
                    time_interval,
                    position_scale=1000.0,
                    time_units="minutes",
                )
            except (FileNotFoundError, KeyError):
                print(
                    f"Failed to load MSD fit parameters for condition {condition}, treatment {treatment}, time interval {time_interval}. Skipping."
                )
                continue
            if msd_fit is None:
                print(
                    f"Failed to load MSD fit parameters for condition {condition}, treatment {treatment}, time interval {time_interval}. Skipping."
                )
                continue

            paramvec = msd_fit["compact_gpr_params"]
            predictor = GPR(
                observation_times,
                FD_MSD,
                3,
                param_layout=((), (), ()),
                noise_layout=("dim",),
                param_names=("tau", "J", "alpha"),
            )

            # randomize the tracks for Shuffled condition
            for fit_type in types_to_test:
                if fit_type == "Shuffled":
                    fit_ms2_data, fit_ep_dat = reassign_tracks_by_duration(
                        ms2_data, ep_dat
                    )
                elif fit_type == "Regular":
                    fit_ms2_data, fit_ep_dat = ms2_data, ep_dat
                elif fit_type == "Shifted":
                    # shift the tracks backwards by 4 time points without wraparound
                    fit_ep_dat = np.full_like(ep_dat, np.nan)
                    fit_ep_dat[:, :-4] = ep_dat[:, 4:]
                    fit_ms2_data = ms2_data
                else:
                    raise ValueError(f"Unknown track fit_type: {fit_type}")

                # predict on the randomized data
                means, vars = predictor.Predict(paramvec, observation_times, fit_ep_dat)
                for rc, kon in zip(rcs_to_test, kons_to_test):
                    print(
                        f"\tRunning {fit_type} inference for condition {condition}, treatment {treatment}, rc={rc}nm, kon={kon}/min..."
                    )
                    # make rundir for this condition and rc if it doesn't exist
                    rundir = condition_treatment_path / f"rc={rc}"
                    if not rundir.exists():
                        rundir.mkdir()

                    if fit_type == "Regular":
                        rc_fit_name = f"{fit_name}_rc={rc}_kon={kon:g}"
                    else:
                        rc_fit_name = f"{fit_name}_{fit_type}_rc={rc}_kon={kon:g}"
                    model_path = rundir / (rc_fit_name + "_model_object.pkl")
                    rc_example_plot_folder = rundir / "example_plots"
                    rc_output_paths = [
                        rundir / f"{rc_fit_name}_model_results.npz",
                        model_path,
                        rundir / f"{rc_fit_name}_burst_calls_overview.png",
                    ] + [
                        rc_example_plot_folder / f"{rc_fit_name}_track{ind}.png"
                        for ind in range(25)
                    ]

                    pcont = prob_in_sphere_quadrature(means, vars, rc).T

                    if file_ok(model_path):
                        with open(model_path, "rb") as f:
                            model = pickle.load(f)
                    else:
                        # load base model
                        base_model_path = condition_treatment_path / (
                            fit_name + "_model_object.pkl"
                        )
                        if not file_ok(base_model_path):
                            print(
                                f"Missing base model for condition {condition}, treatment {treatment}, time interval {time_interval}. Skipping."
                            )
                            continue
                        with open(base_model_path, "rb") as f:
                            model = pickle.load(f)

                        # estimate promoter states and pol2 loadings with this rc and kon
                        kon_fitted = model.M.get("Transition Rates")[
                            "Transition Rates <k>"
                        ][0]
                        kon_inds = np.array([0, 1])[None, :]
                        k_t = (
                            kon * pcont[None, :] / kon_fitted
                        )  # swap out the previously fitted transition rates for the new ones
                        model.Swap_Rate_Series(k_t, kon_inds)
                        model.schedule = [
                            "Polymerase Loadings",
                            "Promoter State",
                            "Initial State Probabilities",
                        ]
                        pbar = tqdm(
                            leave=False, desc="CAVI Sweeps", total=nsteps_to_run
                        )
                        for _ in range(nsteps_to_run):
                            cavi_sweep(
                                model.G,
                                model.M,
                                model.schedule,
                                rho=1.0,
                            )
                            pon = np.nanmean(
                                model.M.get("Promoter State")["masked_posterior"]
                            )
                            avg_pol2 = np.nanmean(
                                model.M.get("Polymerase Loadings")["Pol2_posterior"]
                            )
                            pbar.set_postfix(
                                {"Promoter State": pon, "Polymerase Loadings": avg_pol2}
                            )
                            pbar.update(1)

                    model_results = Store_Results(
                        model,
                        fit_ms2_data,
                        fit_ep_dat,
                        pcont,
                        save_structure=(rundir, rc_fit_name),
                    )
                    plot_rc_predictions(
                        model_results,
                        fit_ms2_data,
                        pcont,
                        fit_ep_dat,
                        means,
                        vars,
                        show_plot=False,
                        save_structure=(rundir, rc_fit_name),
                    )
                    del model
