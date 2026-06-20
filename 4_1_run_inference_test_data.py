import os
import argparse
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
SIMULATION_DATA_ROOT = Path(
    os.environ.get(
        "EP_INFERENCE_SIMDATA_ROOT",
        str(SCRIPT_ROOT),
    )
)

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = (
        Path(os.environ.get("TMPDIR", "/tmp")) / f"matplotlib-{os.getuid()}"
    )
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)


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
    parser.add_argument(
        "--cache-dir",
        default=str(SIMULATION_DATA_ROOT / "cache" / "4_simdata"),
        help="Directory containing simulated test_data_*.pkl files.",
    )
    parser.add_argument(
        "--pattern",
        default="test_data_*.pkl",
        help="Glob pattern for simulated data files inside --cache-dir.",
    )
    parser.add_argument(
        "--outdir",
        default=str(SIMULATION_DATA_ROOT / "result" / "4_1_run_inference_test_data"),
        help="Directory for inference outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run inference even when all expected outputs already exist.",
    )
    parser.add_argument(
        "--max-timepoints",
        type=int,
        default=6000,
        help="Use at most this many timepoints from each simulated trace.",
    )
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    return args


RUN_ARGS = parse_run_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(RUN_ARGS.gpu_index)
import numpy as np
import json
import sys
import matplotlib.pyplot as plt
import pickle

MS2POSTERIOR_ROOT = SCRIPT_ROOT / "MS2Posterior"
TWOLOCUSGPR_ROOT = SCRIPT_ROOT / "TwoLocusGPR"
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(MS2POSTERIOR_ROOT))
sys.path.insert(0, str(TWOLOCUSGPR_ROOT))
from MS2Posterior.Variational_State_Finder import MS2Posterior
from script_utils import file_ok


def resolve_script_path(path_arg):
    path = Path(path_arg)
    if path.is_absolute():
        return path
    return SCRIPT_ROOT / path


CACHE_DIR = resolve_script_path(RUN_ARGS.cache_dir)
RESULT_PATH = resolve_script_path(RUN_ARGS.outdir)
MAX_TIMEPOINTS = RUN_ARGS.max_timepoints

# Edit these lists to add more simulated conditions later.
DTS_TO_RUN = [0.5]
LOC_ERRS_TO_RUN = [40]
SEPS_TO_RUN = [90]
RCS_TRUE_TO_RUN = [30]



def broadcast_state_values(raw_values, n_tracks):
    raw_values = np.asarray(raw_values)
    if raw_values.size == 2:
        return np.broadcast_to(raw_values.reshape(1, 2), (n_tracks, 2))
    return raw_values.reshape(n_tracks, 2)


def Store_Results(model, ms2_data, save_structure=None):
    promoter_state = model.M.get("Promoter State")
    polymerase_loadings = model.M.get("Polymerase Loadings")
    loading_rates_node = model.M.get("Loading Rates")
    loading_rates_raw = np.asarray(loading_rates_node["Loading Rates <k>"])
    loading_rate_log_raw = np.asarray(loading_rates_node["Loading Rates <log k>"])
    loading_rate_std_raw = np.asarray(loading_rates_node["Loading Rates <std k>"])

    pon = promoter_state["masked_posterior"][1:, :, 0].T
    poff = 1 - pon
    predicted_ms2 = model.M.get("Polymerase Loadings")["Predicted MS2"]
    state0_to_state1 = promoter_state["masked_joint"][:, :, 0, 1]
    state1_to_state0 = promoter_state["masked_joint"][:, :, 1, 0]
    turnon = state1_to_state0
    turnoff = state0_to_state1
    MAP = promoter_state["MAP"][:, 1:]
    MAP = np.where(np.isnan(ms2_data), np.nan, MAP)

    loading_rates = broadcast_state_values(loading_rates_raw, len(ms2_data))
    loading_rate_log = broadcast_state_values(loading_rate_log_raw, len(ms2_data))
    loading_rate_std = broadcast_state_values(loading_rate_std_raw, len(ms2_data))
    transition_rates = model.M.get("Transition Rates")["Transition Rates <k>"]
    fit_data = model.M.get("MS2 Data")["data"]

    T_rise = model.G.nodes["Polymerase Loadings"].T_rise
    T_plateau = model.G.nodes["Polymerase Loadings"].T_plateau
    RNA_intensity = model.G.nodes["Polymerase Loadings"].RNA_int
    noise = model.G.nodes["Polymerase Loadings"].noise_std
    prior_shape_raw = np.asarray(model.G.nodes["Loading Rates"].prior_shape)
    prior_rate_raw = np.asarray(model.G.nodes["Loading Rates"].prior_rate)

    model_results = {
        "pon": pon,
        "poff": poff,
        "predicted_ms2": predicted_ms2,
        "turnon": turnon,
        "turnoff": turnoff,
        "state0_to_state1": state0_to_state1,
        "state1_to_state0": state1_to_state0,
        "MAP": MAP,
        "loading_rates": loading_rates,
        "loading_rates_shared": loading_rates_raw,
        "loading_rate_log": loading_rate_log,
        "loading_rate_log_shared": loading_rate_log_raw,
        "loading_rate_std": loading_rate_std,
        "loading_rate_std_shared": loading_rate_std_raw,
        "loading_rates_are_per_track": np.array(model.per_track_rates),
        "transition_rates": transition_rates,
        "pol2_posterior": polymerase_loadings.get(
            "Pol2_posterior", np.full_like(ms2_data, np.nan)
        ),
        "observation_times": model.sampling_times,
        "fine_grid": model.fine_grid,
        "original_ms2_data": ms2_data,
        "fit_data": fit_data,
        "T_rise": T_rise,
        "T_plateau": T_plateau,
        "RNA_intensity": RNA_intensity,
        "noise": noise,
        "loading_rate_prior_shape": prior_shape_raw,
        "loading_rate_prior_rate": prior_rate_raw,
        "loading_rate_prior_shape_per_track": broadcast_state_values(
            prior_shape_raw, len(ms2_data)
        ),
        "loading_rate_prior_rate_per_track": broadcast_state_values(
            prior_rate_raw, len(ms2_data)
        ),
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


def plot_results(model_results, ms2_data, show_plot=True, save_structure=None):
    pon = model_results["pon"]
    predicted_ms2 = model_results["predicted_ms2"]
    MAP = model_results["MAP"]
    observation_times = model_results["observation_times"]
    dt_plot = (
        float(np.nanmedian(np.diff(observation_times)))
        if len(observation_times) > 1
        else 0.0
    )

    nrows, ncols = 3, 5
    ntoplot = nrows * ncols
    inds = np.random.choice(
        len(ms2_data), np.min([ntoplot, len(ms2_data)]), replace=False
    )
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
            xlim=(observation_times[0], tmax + dt_plot),
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


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(val) for val in value]
    return value


def parse_cache_params(cache_path):
    parts = cache_path.stem.split("_")
    if len(parts) != 6 or parts[:2] != ["test", "data"]:
        raise ValueError(f"Unexpected simulated cache filename: {cache_path.name}")
    return {
        "dt": float(parts[2]),
        "loc_err": int(parts[3]),
        "sep": int(parts[4]),
        "rc_true": int(parts[5]),
    }


def should_run_cache(cache_path):
    params = parse_cache_params(cache_path)
    return (
        any(np.isclose(params["dt"], dt) for dt in DTS_TO_RUN)
        and params["loc_err"] in LOC_ERRS_TO_RUN
        and params["sep"] in SEPS_TO_RUN
        and params["rc_true"] in RCS_TRUE_TO_RUN
    )


def load_simulated_ms2(cache_path, max_timepoints=None):
    with open(cache_path, "rb") as f:
        sim_data = pickle.load(f)

    results = sim_data["results"]
    parameters = sim_data["parameters"]
    observation_times = np.asarray(results[0][0], dtype=float)
    ms2_data = np.asarray([np.asarray(track[7], dtype=float) for track in results])
    original_timepoints = len(observation_times)
    if max_timepoints is not None and max_timepoints > 0:
        n_timepoints = min(int(max_timepoints), original_timepoints)
        observation_times = observation_times[:n_timepoints]
        ms2_data = ms2_data[:, :n_timepoints]
    return observation_times, ms2_data, parameters, original_timepoints


def loading_rate_prior(ms2_data, parameters):
    loading_rates = np.asarray(parameters["loading_rates"], dtype=float)
    positive_rates = loading_rates[loading_rates > 0]
    if len(positive_rates) == 0:
        raise ValueError("Simulated data has no positive loading_rates entry.")

    r_on = float(np.max(positive_rates))
    r_off = float(np.min(positive_rates))
    prior_shape = np.array([10.0, 100.0])
    prior_rate = np.array(
        [
            prior_shape[0] / r_on,
            prior_shape[1] / r_off,
        ]
    )
    return prior_shape, prior_rate


def expected_output_paths(savedir, fit_name):
    return [
        savedir / f"{fit_name}_model_results.npz",
        savedir / f"{fit_name}_model_object.pkl",
        savedir / f"{fit_name}_example_traces.png",
        savedir / f"{fit_name}_burst_calls_overview.png",
        savedir / f"{fit_name}_metadata.json",
    ]


def outputs_complete_for_current_settings(savedir, fit_name):
    output_paths = expected_output_paths(savedir, fit_name)
    if not all(file_ok(path) for path in output_paths):
        return False
    try:
        with open(savedir / f"{fit_name}_metadata.json", "r") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return metadata.get("max_timepoints") == MAX_TIMEPOINTS


def main():

    RESULT_PATH.mkdir(parents=True, exist_ok=True)
    cache_paths = sorted(CACHE_DIR.glob(RUN_ARGS.pattern))
    if not cache_paths:
        raise FileNotFoundError(
            f"No simulated data files matching {RUN_ARGS.pattern!r} in {CACHE_DIR}"
        )
    cache_paths = [path for path in cache_paths if should_run_cache(path)]
    if not cache_paths:
        raise FileNotFoundError(
            "No simulated data files matched the requested parameter lists: "
            f"{DTS_TO_RUN=}, {LOC_ERRS_TO_RUN=}, {SEPS_TO_RUN=}, {RCS_TRUE_TO_RUN=}"
        )
    print(f"Selected {len(cache_paths)} simulated dataset(s) for inference.")

    for work_item_index, cache_path in enumerate(cache_paths):
        if work_item_index % RUN_ARGS.num_shards != RUN_ARGS.shard_index:
            continue

        fit_name = cache_path.stem
        savedir = RESULT_PATH / fit_name
        savedir.mkdir(parents=True, exist_ok=True)
        model_path = savedir / f"{fit_name}_model_object.pkl"

        if not RUN_ARGS.overwrite and outputs_complete_for_current_settings(
            savedir, fit_name
        ):
            print(f"Skipping complete fit: {fit_name}")
            continue

        print(f"Running MS2-only inference for {fit_name}")
        observation_times, ms2_data, parameters, original_timepoints = (
            load_simulated_ms2(cache_path, max_timepoints=MAX_TIMEPOINTS)
        )

        model = None
        if file_ok(model_path) and not RUN_ARGS.overwrite:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            if len(model.sampling_times) != ms2_data.shape[1]:
                print(
                    f"Existing model for {fit_name} has "
                    f"{len(model.sampling_times)} timepoints; refitting with "
                    f"{ms2_data.shape[1]} timepoints."
                )
                model = None

        if model is None:
            T_rise = float(parameters["T_rise"])
            T_plateau = float(parameters["T_plateau"])
            RNA_intensity = float(parameters["alpha"])
            noise = np.full(len(ms2_data), float(parameters["noise"]))
            prior_shape, prior_rate = loading_rate_prior(ms2_data, parameters)

            model = MS2Posterior(
                observation_times,
                2,
                observation_times,
                ms2_data,
                T_rise,
                T_plateau,
                RNA_intensity,
                noise,
                per_track_rates=False,
                argdict={
                    "Loading Rates": {
                        "prior_shape": prior_shape,
                        "prior_rate": prior_rate,
                    },
                },
            )
            model.Run()

        model_results = Store_Results(
            model, ms2_data, save_structure=(savedir, fit_name)
        )
        plot_results(
            model_results,
            ms2_data,
            show_plot=False,
            save_structure=(savedir, fit_name),
        )
        metadata = {
            "source_cache": str(cache_path),
            "fit_name": fit_name,
            "sim_parameters": json_ready(parameters),
            "ms2_source_tuple_index": 7,
            "ms2_source": "MS2_signal_noisy",
            "original_timepoints": int(original_timepoints),
            "used_timepoints": int(ms2_data.shape[1]),
            "max_timepoints": MAX_TIMEPOINTS,
        }
        with open(savedir / f"{fit_name}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        del model


if __name__ == "__main__":
    main()
