import os
import argparse


def parse_run_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu-index",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        help="GPU index to expose through CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards to split condition/treatment/time jobs across.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index to run.",
    )
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    return args


# %load_ext autoreload
# %autoreload 2
RUN_ARGS = parse_run_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(RUN_ARGS.gpu_index)
from pathlib import Path
from pyexpat import model
import numpy as np
import json
import sys
import matplotlib.pyplot as plt
import pickle
from script_utils import file_ok, filter_short_tracks, time_interval_seconds

SCRIPT_ROOT = Path(__file__).resolve().parent
MS2POSTERIOR_ROOT = SCRIPT_ROOT / "MS2Posterior"
TWOLOCUSGPR_ROOT = SCRIPT_ROOT / "TwoLocusGPR"
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(MS2POSTERIOR_ROOT))
sys.path.insert(0, str(TWOLOCUSGPR_ROOT))
from MS2Posterior.Variational_State_Finder import MS2Posterior
from TwoLocusGPR.Posterior_analysis import (
    prob_in_sphere_quadrature,
    batched_radius_percentile_comp,
)

DATA_ROOT = SCRIPT_ROOT / "Data" / "filtered_data_v6"
CORR_FIT_PATH = SCRIPT_ROOT / "result" / "1_MS2_corrfit" / "fit_raw_intensity_0341338401d33fe53cf19f91401d8dac_summary.json"

MSD_PATH = SCRIPT_ROOT / "cache" / "MSD_fits" / "fit_results_individual_v28_StatisticalFilter_Alpha0p37_final"
RESULT_PATH = SCRIPT_ROOT / "result" / "2_run_inference"


def read_data(condition, intensity_key, time_interval, treatment):

    time_interval_seconds = float(str(time_interval).removesuffix("s"))
    path = DATA_ROOT / f"{time_interval_seconds:g}s_{condition}_{treatment}.npz"
    if not path.exists():
        return None, None
    loaded_data = np.load(path, allow_pickle=True)
    ep_dat = loaded_data["enhancer_coordinate"]-loaded_data["promoter_coordinate"]
    ms2_data = loaded_data[intensity_key]
    
    return ep_dat, ms2_data


def single_dataset_compact_params_from_full(full_params):
    """Convert expanded [Gamma,J,alpha] per-dim params to compact single-dataset order."""
    full_params = np.asarray(full_params, dtype=float)
    msd_params = full_params[:-3].reshape(3, 3)
    return np.concatenate(
        [
            msd_params[:, 0],
            msd_params[:, 1],
            msd_params[:, 2],
            full_params[-3:],
        ]
    )




def Store_Results(model, ms2_data, save_structure=None):
    pon = model.M.get("Promoter State")["masked_posterior"][1:, :, 0].T
    predicted_ms2 = model.M.get("Polymerase Loadings")["Predicted MS2"]
    turnon = model.M.get("Promoter State")["masked_joint"][:, :, 0, 1]
    MAP = model.M.get("Promoter State")["MAP"][:, 1:]
    MAP = np.where(np.isnan(ms2_data), np.nan, MAP)

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
        "loading_rates": loading_rates,
        "transition_rates": transition_rates,
        "observation_times": model.sampling_times,
        "fine_grid": model.fine_grid,
        "original_ms2_data": ms2_data,
        "fit_data": fit_data,
        "T_rise": T_rise,
        "T_plateau": T_plateau,
        "RNA_intensity": RNA_intensity,
        "noise": noise,
        "loading_rate_prior_shape": prior_shape,
        "loading_rate_prior_rate": prior_rate,
        "Initial_Probabilities": np.exp(model.M.get("Initial State Probabilities")["Initial State Probabilities <log pi>"]) / np.sum(np.exp(model.M.get("Initial State Probabilities")["Initial State Probabilities <log pi>"]), axis=-1, keepdims=True),
    }

    if save_structure is not None:
        savedir, savename = save_structure
        np.savez(savedir / f"{savename}_model_results.npz", **model_results)
        with open(savedir / f"{savename}_model_object.pkl", "wb") as f:
            pickle.dump(model, f)
    return model_results


def plot_results(model_results, ms2_data, show_plot=True, save_structure=None):
    pon = model_results["pon"]
    predicted_ms2 = model_results["predicted_ms2"]
    MAP = model_results["MAP"]
    observation_times = model_results["observation_times"]

    nrows, ncols = 3, 5
    ntoplot = nrows * ncols
    inds = np.random.choice(len(ms2_data), np.min([ntoplot,len(ms2_data)]), replace=False)
    fig, ax = plt.subplots(nrows, ncols, figsize=(16, 9))
    for i, ind in enumerate(inds):
        ax.flatten()[i].plot(
            observation_times, ms2_data[ind], ".", label="MS2 data", alpha=0.2
        )
        ax.flatten()[i].plot(
            observation_times, predicted_ms2[ind], label="Predicted MS2", color="C0"
        )
        ax.flatten()[i].fill_between(
            observation_times,
            0,
            pon[ind] * np.nanmax(ms2_data[ind]),
            label="Promoter state",
            alpha=0.2,
            color="C1",
            zorder=-1,
        )
        ax.flatten()[i].plot(
            observation_times,
            np.abs(1 - MAP[ind]) * np.nanmax(ms2_data[ind]),
            label="MAP path",
            color="C1",
        )
        tmax = np.max(observation_times[~np.isnan(ms2_data[ind])])
        ax.flatten()[i].set(
            title=f"Track {ind}",
            xlabel="Time (min)",
            ylabel="MS2 Intensity",
            xlim=(observation_times[0], tmax + dt),
        )
        if i == 0:
            ax.flatten()[i].legend()
    fig.tight_layout()
    if save_structure is not None:
        savedir, savename = save_structure
        fig.savefig(savedir / f"{savename}_example_traces.png")
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

conditions = (
    "85kb",
    "1.5kb",
    "340kb_Ce_Cp",
    "170kb",
    "340kb_Ce",
    "255kb",
    "340kb_Cp",
    "340kb",
)
treatments_to_fit = ["None", "IAA", "dTAG", "IAAdTAG"]
possible_time_intervals = ["30s", "5s"]

time_interval = 30
minlength = 50
dt = time_interval / 60
# rcmin, rc_step, nvals = 10, 15, 14
# rcs_to_test = np.arange(rcmin, rcmin + rc_step * nvals, rc_step)

# read the correlation fit results
with open(CORR_FIT_PATH, "r") as f:
    corr_fit_results = json.load(f)
keys = ["koff", "T_rise", "T_plateau", "RNA_intensity", "offset", "loading_rate"]
koff, T_rise, T_plateau, RNA_intensity, offset, loading_rate = [
    corr_fit_results["parameters"][key] for key in keys
]

condition = "340kb"
treatment = "None"
time_interval = "30s"

work_item_index = 0
for condition in conditions:
    for treatment in treatments_to_fit:
        for fr_ind, time_interval in enumerate(possible_time_intervals):
            current_work_item = work_item_index
            work_item_index += 1
            if current_work_item % RUN_ARGS.num_shards != RUN_ARGS.shard_index:
                continue
            
            dt = time_interval_seconds(time_interval) / 60.0
            

            # read the data
            try:
                ep_dat, ms2_data = read_data(
                    condition, "corrected_intensity", time_interval, treatment
                )
                if ep_dat is None or ms2_data is None:
           
                    continue
            except FileNotFoundError as e:
                continue
            
            # filter out short tracks
            ms2_data, ep_dat = filter_short_tracks(ms2_data, ep_dat, minlength)
            if len(ms2_data) == 0:
                print(
                    f"\tAfter filtering out tracks shorter than {minlength} time points, no tracks remain for condition {condition}, treatment {treatment}, time interval {time_interval}. Skipping inference."
                )
                continue

            # pad the data to allow for pol2 loadings before the measurement begins
            n_pad = int((T_rise + T_plateau) / dt) + 1
            ms2_data = np.pad(ms2_data, ((0, 0), (n_pad, 0)), constant_values=np.nan)
            ep_dat = np.pad(
                ep_dat,
                ((0, 0), (n_pad, 0), (0, 0)),
                constant_values=np.nan,
            )
            observation_times = np.arange(ms2_data.shape[1]) * dt - n_pad * dt

            # make folder for this condition and treatment if it doesn't exist
            condition_treatment_path = RESULT_PATH / condition / treatment
            if not condition_treatment_path.exists():
                condition_treatment_path.mkdir(parents=True)
            fit_name = f"{time_interval}_{condition}_{treatment}"

            base_model_path = condition_treatment_path / (
                fit_name + "_model_object.pkl"
            )
            base_output_paths = [
                condition_treatment_path / f"{fit_name}_model_results.npz",
                base_model_path,
                condition_treatment_path / f"{fit_name}_example_traces.png",
                condition_treatment_path / f"{fit_name}_burst_calls_overview.png",
            ]
            if not all(file_ok(path) for path in base_output_paths):
                if file_ok(base_model_path):
                    with open(base_model_path, "rb") as f:
                        model = pickle.load(f)
                else:
                    # set up the model for the non-ep-driven fit
                    data_offset = ms2_data - offset
                    noise = np.nanmean(np.abs(np.diff(ms2_data, axis=1)), axis=1)
                    r_on = loading_rate
                    r_off = 0.0001
                    prior_shape = np.array([10.0, 100.0] * len(ms2_data))
                    prior_rate = np.array(
                        [
                            prior_shape[0] / r_on,
                            prior_shape[1] / r_off,
                        ]
                        * len(ms2_data)
                    )
                    model = MS2Posterior(
                        observation_times,
                        2,
                        observation_times,
                        data_offset,
                        T_rise,
                        T_plateau,
                        RNA_intensity,
                        noise,
                        per_track_rates=True,
                        argdict={
                            "Loading Rates": {
                                "prior_shape": prior_shape,
                                "prior_rate": prior_rate,
                            },
                        },
                    )

                    # run the inference
                    model.Run(eval_conds=["transition"])
                
                model_results = Store_Results(
                    model, ms2_data, save_structure=(condition_treatment_path, fit_name)
                )
                plot_results(
                    model_results,
                    ms2_data,
                    show_plot=False,
                    save_structure=(condition_treatment_path, fit_name),
                )
                del model


            
