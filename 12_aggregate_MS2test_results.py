from pathlib import Path
import numpy as np
import os
import argparse

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import pickle
from scipy.ndimage import label
from tqdm.auto import tqdm
import multiprocessing as mp
import jax
import pandas as pd
import h5py

# set jax to use CPU only
jax.config.update("jax_platform_name", "cpu")

SCRIPT_ROOT = Path(__file__).resolve().parent
TRUE_DATA_DIR = SCRIPT_ROOT / "Data" / "MS2_test_data"
RESULTS_DIR = TRUE_DATA_DIR / "inference_results"
DATA_DIR = SCRIPT_ROOT / "Data"
SUMMARY_CSV = DATA_DIR / "aggregated_MS2_test_results.csv"
SCATTER_H5 = DATA_DIR / "aggregated_MS2_test_scatter_values.h5"
SCATTER_INDEX_CSV = DATA_DIR / "aggregated_MS2_test_scatter_index.csv"

T_RISE = 3
T_PLATEAU = 5
TMAX = 500

def _load_true_files_by_ind():
    if not TRUE_DATA_DIR.exists():
        return {}
    return {
        f.split("ind=")[1].split("_")[0]: TRUE_DATA_DIR / f
        for f in os.listdir(TRUE_DATA_DIR)
        if f.endswith(".pkl") and "ind=" in f
    }


TRUE_FILES_BY_IND = _load_true_files_by_ind()


def _metadata_from_filename(filename):
    return {
        "ind": int(filename.split("ind=")[1].split("_")[0]),
        "kon": float(filename.split("_kon=")[1].split("_")[0]),
        "koff": float(filename.split("_koff=")[1].split("_")[0]),
        "loading_rate": float(filename.split("_lrate=")[1].split("_")[0]),
        "snr": float(filename.split("_snr=")[1].split("_")[0]),
        "dt": float(filename.split("_dt=")[1].split("_")[0]),
        "ntraj": int(filename.split("_ntraj=")[1].split(".pkl")[0]),
    }


def _true_file_for_result(filename):
    ind = filename.split("ind=")[1].split("_")[0]
    if ind not in TRUE_FILES_BY_IND:
        raise FileNotFoundError(f"Expected one true-data file for ind={ind}")
    return TRUE_FILES_BY_IND[ind]


def _bin_hidden_promoter_and_events(true_data, arr_dat, dt):
    sampling_times = np.linspace(0, TMAX, int(TMAX / dt) + 1)
    ngridpoints = arr_dat.shape[1]
    fine_grid = np.linspace(0, TMAX, ngridpoints)
    bw = fine_grid[1] - fine_grid[0]
    bin_edges = np.concatenate(([fine_grid[0] - bw / 2], fine_grid + bw / 2))
    bin_starts = bin_edges[:-1]

    binned_chopped_states = []
    binned_chopped_states_w_loadings = []
    binned_chopped_events = []
    for t_ind, hidden in enumerate(true_data["hidden_dataset"]):
        times, states, events, _ = hidden
        times = times - T_RISE - T_PLATEAU
        events = events - T_RISE - T_PLATEAU

        if len(times) > 1:
            state_inds = np.searchsorted(times[1:], bin_starts, side="right")
            state_inds = np.clip(state_inds, 0, len(states) - 1)
            abinned_states = np.abs(states[state_inds] - 1).astype(float)
        else:
            current_state = np.abs(states[0] - 1)
            abinned_states = np.full(len(bin_edges) - 1, current_state, dtype=float)

        abinned_events = np.histogram(events, bins=bin_edges)[0].astype(float)

        valid_obs = ~np.isnan(arr_dat[t_ind, :])
        if np.any(valid_obs):
            max_time = sampling_times[np.max(np.where(valid_obs))]
        else:
            max_time = -np.inf
        valid_time = fine_grid <= max_time

        abinned_states[~valid_time] = np.nan
        abinned_events[~valid_time] = np.nan

        labs, nfeatures = label(abinned_states == 1)
        states_w_loadings = np.zeros_like(abinned_states)
        states_w_loadings[~valid_time] = np.nan
        for feature_index in range(1, nfeatures + 1):
            inds = labs == feature_index
            states_w_loadings[inds] = float(np.any(abinned_events[inds] > 0))

        binned_chopped_states.append(abinned_states)
        binned_chopped_events.append(abinned_events)
        binned_chopped_states_w_loadings.append(states_w_loadings)

    return (
        np.asarray(binned_chopped_states),
        np.asarray(binned_chopped_states_w_loadings),
        np.asarray(binned_chopped_events),
        fine_grid,
    )


def run_aggregation(filename, include_scatter_values=False):
    metadata = _metadata_from_filename(filename)
    kon = metadata["kon"]
    koff = metadata["koff"]
    loading_rate = metadata["loading_rate"]
    snr = metadata["snr"]
    dt = metadata["dt"]
    ntraj = metadata["ntraj"]

    with open(RESULTS_DIR / filename, "rb") as f:
        data = pickle.load(f)
    with open(_true_file_for_result(filename), "rb") as f:
        true_data = pickle.load(f)

    arr_dat = np.array(true_data["observed_dataset"])
    (
        binned_chopped_states,
        binned_chopped_states_w_loadings,
        binned_chopped_events,
        fine_grid,
    ) = _bin_hidden_promoter_and_events(true_data, arr_dat, dt)

    predicted_state = data["M"].get("Promoter State")["masked_posterior"].swapaxes(0, 1)
    predicted_events = data["M"].get("Polymerase Loadings")["Pol2_posterior"]

    Pred_loading_rates = data["params"]["Loading Rates"][-1]
    pred_on_state = np.argmax(Pred_loading_rates)

    # analyze deviation in "regression error"
    state_deviation_mean = np.nanmean(
        np.abs(binned_chopped_states_w_loadings - predicted_state[:, 1:, pred_on_state])
    )
    state_deviation_std = np.nanstd(
        np.abs(binned_chopped_states_w_loadings - predicted_state[:, 1:, pred_on_state])
    )

    p_on, p_off = [], []
    for index in range(predicted_state.shape[0]):
        true_states_on = binned_chopped_states_w_loadings[index]
        true_states_off = binned_chopped_states[index]
        # max_ind = np.argmax(fine_grid[fine_grid<=np.max(sampling_times[~np.isnan(arr_dat[index,:])])])+1

        pred_stateprob = predicted_state[index, 1:, pred_on_state]
        p_on.append(pred_stateprob[true_states_on > 0])
        p_off.append(pred_stateprob[true_states_off == 0])

    # Analyze state prediction
    pons_flat, poff_flat = np.concatenate(p_on), np.concatenate(p_off)
    pons_flat, poff_flat = (
        pons_flat[~np.isnan(pons_flat)],
        poff_flat[~np.isnan(poff_flat)],
    )
    finite_probs = np.concatenate([pons_flat, poff_flat])
    positive_probs = finite_probs[finite_probs > 0]
    min_val = np.min(positive_probs) if len(positive_probs) else 1e-12
    max_val = np.max([np.max(pons_flat), np.max(poff_flat)])
    edges = np.logspace(np.log10(min_val), np.log10(max_val), 100)
    centers = np.sqrt(edges[:-1] * edges[1:])
    pon_hist, _ = np.histogram(pons_flat, edges, range=(0, 1), density=True)
    poff_hist, _ = np.histogram(poff_flat, edges, range=(0, 1), density=True)
    TPR, FPR = [], []
    for threshold in edges[1:]:
        TP = np.nansum(pons_flat[~np.isnan(pons_flat)] >= threshold)
        FN = np.nansum(pons_flat[~np.isnan(pons_flat)] < threshold)
        FP = np.nansum(poff_flat >= threshold)
        TN = np.nansum(poff_flat < threshold)
        TPR.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
        FPR.append(FP / (FP + TN) if (FP + TN) > 0 else 0)
    TPR, FPR = np.array(TPR), np.array(FPR)
    best_ind = np.argmax(TPR - FPR)
    best_TPR, best_FPR = TPR[best_ind], FPR[best_ind]
    sorted_FPR = np.argsort(FPR)
    AUC = np.trapz(TPR[sorted_FPR], FPR[sorted_FPR])

    # analyze event prediction
    deviations = np.cumsum(binned_chopped_events, axis=-1) - np.cumsum(
        (predicted_events) * (fine_grid[1] - fine_grid[0]), axis=-1
    )
    mean, std = np.nanmean(deviations), np.nanstd(deviations)
    avg_event_number = np.nanmean(np.nansum(binned_chopped_events, axis=-1))

    # rates
    off_state = np.abs(pred_on_state - 1)
    ind_order = np.array([pred_on_state, off_state])
    lrate_mu, lrate_std = (
        data["params"]["Loading Rates"][-1][ind_order],
        data["params"]["Loading Rates <std k>"][-1][ind_order],
    )
    trate_mu, trate_std = (
        data["params"]["Transition Rates"][-1][ind_order],
        data["params"]["Transition Rates <std k>"][-1][ind_order],
    )

    results = {
        "AUC": AUC,
        "Best_TPR": best_TPR,
        "Best_FPR": best_FPR,
        "mean_event_deviation": mean,
        "std_event_deviation": std,
        "avg_event_number": avg_event_number,
        "pred_loading_rate_on": lrate_mu[0],
        "pred_loading_rate_off": lrate_mu[1],
        "pred_loading_rate_on_std": lrate_std[0],
        "pred_loading_rate_off_std": lrate_std[1],
        "pred_kon": trate_mu[0],
        "pred_koff": trate_mu[1],
        "pred_kon_std": trate_std[0],
        "pred_koff_std": trate_std[1],
        "true_kon": kon,
        "true_koff": koff,
        "loading_rate": loading_rate,
        "snr": snr,
        "dt": dt,
        "ntraj": ntraj,
        "state_deviation_mean": state_deviation_mean,
        "state_deviation_std": state_deviation_std,
    }

    if not include_scatter_values:
        return results

    pred_promoter_state = np.asarray(predicted_state[:, 1:, pred_on_state])
    true_promoter_state = np.asarray(binned_chopped_states)
    true_loading_state = np.asarray(binned_chopped_states_w_loadings)
    n_state_time = min(
        true_promoter_state.shape[1],
        true_loading_state.shape[1],
        pred_promoter_state.shape[1],
    )
    true_promoter_state = true_promoter_state[:, :n_state_time]
    true_loading_state = true_loading_state[:, :n_state_time]
    pred_promoter_state = pred_promoter_state[:, :n_state_time]
    observed_state_mask = np.asarray(arr_dat[:, :n_state_time])
    state_mask = (
        np.isfinite(true_promoter_state)
        & np.isfinite(true_loading_state)
        & np.isfinite(pred_promoter_state)
        & np.isfinite(observed_state_mask)
    )

    true_ms2 = np.asarray([hidden[-1] for hidden in true_data["hidden_dataset"]])
    pred_ms2 = np.asarray(data["M"].get("Polymerase Loadings")["Predicted MS2"])
    n_ms2_time = min(true_ms2.shape[1], pred_ms2.shape[1], arr_dat.shape[1])
    true_ms2 = true_ms2[:, :n_ms2_time]
    pred_ms2 = pred_ms2[:, :n_ms2_time]
    observed_mask = np.asarray(arr_dat[:, :n_ms2_time])
    ms2_mask = (
        np.isfinite(observed_mask) & np.isfinite(true_ms2) & np.isfinite(pred_ms2)
    )

    scatter_values = {
        "true_promoter_state": true_promoter_state[state_mask].astype(np.uint8),
        "true_loading_state": true_loading_state[state_mask].astype(np.uint8),
        "pred_promoter_state": pred_promoter_state[state_mask].astype(np.float32),
        "true_ms2": true_ms2[ms2_mask].astype(np.float32),
        "pred_ms2": pred_ms2[ms2_mask].astype(np.float32),
    }
    return results, scatter_values


def build_scatter_values(filename):
    metadata = _metadata_from_filename(filename)
    with open(RESULTS_DIR / filename, "rb") as f:
        data = pickle.load(f)
    with open(_true_file_for_result(filename), "rb") as f:
        true_data = pickle.load(f)

    arr_dat = np.asarray(true_data["observed_dataset"])
    binned_chopped_states, binned_chopped_states_w_loadings, _, _ = (
        _bin_hidden_promoter_and_events(true_data, arr_dat, metadata["dt"])
    )

    predicted_state = data["M"].get("Promoter State")["masked_posterior"].swapaxes(0, 1)
    pred_on_state = np.argmax(data["params"]["Loading Rates"][-1])

    pred_promoter_state = np.asarray(predicted_state[:, 1:, pred_on_state])
    n_state_time = min(
        binned_chopped_states.shape[1],
        binned_chopped_states_w_loadings.shape[1],
        pred_promoter_state.shape[1],
        arr_dat.shape[1],
    )
    true_promoter_state = binned_chopped_states[:, :n_state_time]
    true_loading_state = binned_chopped_states_w_loadings[:, :n_state_time]
    pred_promoter_state = pred_promoter_state[:, :n_state_time]
    observed_state_mask = arr_dat[:, :n_state_time]
    state_mask = (
        np.isfinite(true_promoter_state)
        & np.isfinite(true_loading_state)
        & np.isfinite(pred_promoter_state)
        & np.isfinite(observed_state_mask)
    )

    true_ms2 = np.asarray([hidden[-1] for hidden in true_data["hidden_dataset"]])
    pred_ms2 = np.asarray(data["M"].get("Polymerase Loadings")["Predicted MS2"])
    n_ms2_time = min(true_ms2.shape[1], pred_ms2.shape[1], arr_dat.shape[1])
    true_ms2 = true_ms2[:, :n_ms2_time]
    pred_ms2 = pred_ms2[:, :n_ms2_time]
    observed_ms2_mask = arr_dat[:, :n_ms2_time]
    ms2_mask = (
        np.isfinite(observed_ms2_mask) & np.isfinite(true_ms2) & np.isfinite(pred_ms2)
    )

    return {
        "true_promoter_state": true_promoter_state[state_mask].astype(np.uint8),
        "true_loading_state": true_loading_state[state_mask].astype(np.uint8),
        "pred_promoter_state": pred_promoter_state[state_mask].astype(np.float32),
        "true_ms2": true_ms2[ms2_mask].astype(np.float32),
        "pred_ms2": pred_ms2[ms2_mask].astype(np.float32),
    }


def _create_scatter_datasets(handle):
    for name in ("true_promoter_state", "true_loading_state"):
        handle.create_dataset(
            name,
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint8,
            chunks=(1_000_000,),
            compression="lzf",
            shuffle=True,
        )
    for name in ("pred_promoter_state", "true_ms2", "pred_ms2"):
        handle.create_dataset(
            name,
            shape=(0,),
            maxshape=(None,),
            dtype=np.float32,
            chunks=(1_000_000,),
            compression="lzf",
            shuffle=True,
        )


def _append_dataset(dataset, values):
    start = dataset.shape[0]
    end = start + len(values)
    dataset.resize((end,))
    dataset[start:end] = values
    return start, len(values)


def write_scatter_values(result_files):
    records = []
    with h5py.File(SCATTER_H5, "w") as handle:
        handle.attrs["description"] = (
            "Frame-level true/predicted promoter-state and MS2 pairs aggregated "
            "from 2025_12_22_test_MS2Posterior inference results."
        )
        handle.attrs["promoter_state_definition"] = (
            "true_promoter_state is the hidden promoter state binned onto "
            "observation frames after subtracting T_rise + T_plateau; "
            "pred_promoter_state is the inferred posterior probability for the "
            "state with the larger fitted loading rate."
        )
        handle.attrs["ms2_definition"] = (
            "true_ms2 is the noiseless hidden MS2 signal and pred_ms2 is the "
            "Predicted MS2 output; both are restricted to observed frames."
        )
        _create_scatter_datasets(handle)

        for filename in tqdm(result_files, desc="Writing scatter values"):
            scatter_values = build_scatter_values(filename)
            metadata = _metadata_from_filename(filename)
            promoter_start, promoter_count = _append_dataset(
                handle["true_promoter_state"], scatter_values["true_promoter_state"]
            )
            _append_dataset(
                handle["true_loading_state"], scatter_values["true_loading_state"]
            )
            _append_dataset(
                handle["pred_promoter_state"], scatter_values["pred_promoter_state"]
            )
            ms2_start, ms2_count = _append_dataset(
                handle["true_ms2"], scatter_values["true_ms2"]
            )
            _append_dataset(handle["pred_ms2"], scatter_values["pred_ms2"])

            records.append(
                {
                    "filename": filename,
                    "promoter_start": promoter_start,
                    "promoter_count": promoter_count,
                    "ms2_start": ms2_start,
                    "ms2_count": ms2_count,
                    "ind": metadata["ind"],
                    "true_kon": metadata["kon"],
                    "true_koff": metadata["koff"],
                    "loading_rate": metadata["loading_rate"],
                    "snr": metadata["snr"],
                    "dt": metadata["dt"],
                    "ntraj": metadata["ntraj"],
                }
            )

    pd.DataFrame(records).to_csv(SCATTER_INDEX_CSV, index=False)


def aggregates_complete():
    return all(
        path.exists() and path.stat().st_size > 0
        for path in (SUMMARY_CSV, SCATTER_H5, SCATTER_INDEX_CSV)
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate aggregate outputs from Data/MS2_test_data.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    DATA_DIR.mkdir(exist_ok=True)
    if aggregates_complete() and not args.overwrite:
        print("Using cached MS2 validation aggregate outputs in Data/.")
        return
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(
            "Missing raw MS2 validation inference files. Expected them under "
            f"{RESULTS_DIR}. The published aggregate outputs are already in Data/; "
            "rerun with --overwrite only after adding the raw validation files."
        )
    result_files = sorted(f for f in os.listdir(RESULTS_DIR) if f.endswith(".pkl"))
    if not result_files:
        raise FileNotFoundError(
            "No raw MS2 validation inference_result_*.pkl files found under "
            f"{RESULTS_DIR}. The published aggregate outputs are already in Data/; "
            "rerun with --overwrite only after adding the raw validation files."
        )

    res = []
    n_workers = min(mp.cpu_count(), 16)
    with mp.Pool(processes=n_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(run_aggregation, result_files),
            total=len(result_files),
            desc="Aggregating run metrics",
        ):
            res.append(result)

    df = pd.DataFrame(res)
    df.to_csv(SUMMARY_CSV, index=False)
    write_scatter_values(result_files)


if __name__ == "__main__":
    main()
