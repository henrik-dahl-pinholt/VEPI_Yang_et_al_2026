from __future__ import annotations

from pathlib import Path
import hashlib
import os
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
import pickle
import json
from script_utils import (
    _array_digest,
    _json_dumps,
    _safe_name,
    _save_npz_atomic,
)
from jax_script_utils import acf_jax, cross_corr_jax
SCRIPT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_ROOT / "Data" / "filtered_data_v6"
CACHE_DIR = SCRIPT_ROOT / "cache" / "6_cross_correlation"
main_fig_dir = SCRIPT_ROOT / "figures" / "main_figure"
SIMDATA_RESPATH = SCRIPT_ROOT / "result" / "5_simulate_from_calibrated_params"
CORR_FIT_PATH = SCRIPT_ROOT / "result/1_MS2_corrfit/fit_raw_intensity_0341338401d33fe53cf19f91401d8dac_summary.json"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
main_fig_dir.mkdir(parents=True, exist_ok=True)
CROSS_CORR_BOOTSTRAP_NSAMPLES = int(os.environ.get("CROSS_CORR_BOOTSTRAP_NSAMPLES", "10000"))
CROSS_CORR_BOOTSTRAP_CONFIDENCE = float(os.environ.get("CROSS_CORR_BOOTSTRAP_CONFIDENCE", "95"))
CROSS_CORR_BOOTSTRAP_SEED = int(os.environ.get("CROSS_CORR_BOOTSTRAP_SEED", "0"))
CROSS_CORR_BOOTSTRAP_CACHE_VERSION = 3
with open(CORR_FIT_PATH, "r") as f:
    corr_fit_results = json.load(f)
offset = corr_fit_results["parameters"]["offset"]





def _cache_digest(metadata: dict, arrays) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(_json_dumps(metadata).encode())
    for arr in arrays:
        digest.update(_array_digest(arr).encode())
    return digest.hexdigest()



def read_data(condition, intensity_key, time_interval, treatment):

    time_interval_seconds = float(str(time_interval).removesuffix("s"))
    path = DATA_ROOT / f"{time_interval_seconds:g}s_{condition}_{treatment}.npz"
    if not path.exists():
        return None, None
    loaded_data = np.load(path, allow_pickle=True)
    ep_dat = loaded_data["enhancer_coordinate"]-loaded_data["promoter_coordinate"]
    ms2_data = loaded_data[intensity_key]
    
    return ep_dat, ms2_data

@jax.jit
def cross_corr_track_sums_counts(dataset1, dataset2, lags):
    data1 = jnp.asarray(dataset1)
    data1 = (data1 - jnp.nanmean(data1, axis=1, keepdims=True)) / jnp.nanstd(data1, axis=1, keepdims=True)
    data2 = jnp.asarray(dataset2)
    data2 = (data2 - jnp.nanmean(data2, axis=1, keepdims=True)) / jnp.nanstd(data2, axis=1, keepdims=True)
    t_len = data1.shape[1]

    def sums_counts_at_lag(lag):
        lag_abs = jnp.abs(lag)
        shifted = jnp.roll(data2, shift=-lag, axis=1)
        valid_time = jnp.where(
            lag >= 0,
            jnp.arange(t_len) < (t_len - lag_abs),
            jnp.arange(t_len) >= lag_abs,
        )
        products = data1 * shifted
        valid = valid_time[None, :] & jnp.isfinite(products)
        sums = jnp.sum(jnp.where(valid, products, 0.0), axis=1)
        counts = jnp.sum(valid, axis=1)
        return sums, counts

    sums, counts = jax.vmap(sums_counts_at_lag)(lags)
    return jnp.swapaxes(sums, 0, 1), jnp.swapaxes(counts, 0, 1)


def bootstrap_cross_corr_samples(dataset1, dataset2, lags, nsamples, seed):
    """Bootstrap cross-correlations by resampling paired trajectories."""
    if nsamples < 2:
        raise ValueError("At least two bootstrap samples are required.")
    track_sums, track_counts = cross_corr_track_sums_counts(dataset1, dataset2, lags)
    track_counts = track_counts.astype(track_sums.dtype)
    ntraj = track_sums.shape[0]
    keys = jax.random.split(jax.random.PRNGKey(seed), nsamples)

    @jax.jit
    def sample_all(sample_keys):
        def scan(_, key):
            traj_inds = jax.random.randint(key, shape=(ntraj,), minval=0, maxval=ntraj)
            total_sums = jnp.sum(track_sums[traj_inds], axis=0)
            total_counts = jnp.sum(track_counts[traj_inds], axis=0)
            corr = jnp.where(total_counts > 0, total_sums / total_counts, jnp.nan)
            return None, corr

        _, samples = jax.lax.scan(scan, None, sample_keys)
        return samples

    return sample_all(keys)


def merge_condition_bootstrap_samples(condition_samples):
    """Merge condition-level bootstrap correlation curves into one distribution."""
    condition_samples = np.asarray(condition_samples)
    return condition_samples.reshape(-1, condition_samples.shape[-1])


def _average_cross_corr_bootstrap_cache_path(
    label,
    conditions,
    ep_dist_datasets,
    ms2_datasets,
    lags,
    nsamples,
    confidence,
    seed,
) -> Path:
    metadata = {
        "cache_type": "merged_condition_cross_correlation_bootstrap",
        "version": CROSS_CORR_BOOTSTRAP_CACHE_VERSION,
        "label": label,
        "conditions": list(conditions),
        "nsamples": int(nsamples),
        "confidence": float(confidence),
        "seed": int(seed),
    }
    arrays = list(ep_dist_datasets) + list(ms2_datasets) + [lags]
    key = _cache_digest(metadata, arrays)
    return CACHE_DIR / f"{_safe_name(label)}_{key}.npz"


def load_or_compute_average_cross_corr_bootstrap(
    label,
    conditions,
    ep_dist_datasets,
    ms2_datasets,
    lags,
    nsamples=CROSS_CORR_BOOTSTRAP_NSAMPLES,
    confidence=CROSS_CORR_BOOTSTRAP_CONFIDENCE,
    seed=CROSS_CORR_BOOTSTRAP_SEED,
):
    cache_path = _average_cross_corr_bootstrap_cache_path(
        label,
        conditions,
        ep_dist_datasets,
        ms2_datasets,
        lags,
        nsamples,
        confidence,
        seed,
    )
    if cache_path.exists():
        print(f"Loaded cached cross-correlation bootstrap: {cache_path}", flush=True)
        with np.load(cache_path, allow_pickle=False) as cached:
            return cached["samples"], cached["lower"], cached["upper"]

    print(f"Computing merged cross-correlation bootstrap: {label} ({nsamples} samples per condition)", flush=True)
    condition_samples = []
    for index, (condition, ep_dist, ms2_data) in enumerate(
        zip(conditions, ep_dist_datasets, ms2_datasets)
    ):
        print(f"  {condition}", flush=True)
        samples = np.asarray(
            bootstrap_cross_corr_samples(ep_dist, ms2_data, lags, nsamples, seed + index)
        )
        condition_samples.append(samples)

    condition_samples = np.stack(condition_samples)
    samples = merge_condition_bootstrap_samples(condition_samples)
    alpha = (100.0 - confidence) / 2.0
    lower, upper = np.nanpercentile(samples, [alpha, 100.0 - alpha], axis=0)
    metadata = {
        "cache_type": "merged_condition_cross_correlation_bootstrap",
        "version": CROSS_CORR_BOOTSTRAP_CACHE_VERSION,
        "label": label,
        "conditions": list(conditions),
        "nsamples": int(nsamples),
        "confidence": float(confidence),
        "seed": int(seed),
        "lags": np.asarray(lags).tolist(),
        "resampling": "tracks_within_condition_then_merge_condition_bootstrap_curves",
    }
    _save_npz_atomic(
        cache_path,
        samples=samples,
        lower=lower,
        upper=upper,
        metadata=np.array(_json_dumps(metadata)),
    )
    print(f"Saved cross-correlation bootstrap cache: {cache_path}", flush=True)
    return samples, lower, upper


def randomize_tracks(dataset1, dataset2):
    """Pair up tracks from dataset1 and dataset2 by choosing the track with the closest duration that isn't itself or already taken"""
    durations = np.sum(~np.isnan(dataset1), axis=1)
    inds = np.arange(dataset1.shape[0])
    taken_inds = set()
    permuted_inds = []
    for i in range(dataset1.shape[0]):
        dur = durations[i]
        possible_inds = inds[~np.isin(inds, list(taken_inds)) & (inds != i)]
        if len(possible_inds) == 0:
            print("Warning: ran out of possible tracks to pair, taking a random one from the whole set that isn't itself")
            # if we run out of possible inds, just take a random one from the whole set that isn't itself
            possible_inds = inds[inds != i]
        closest_ind = possible_inds[np.argmin(np.abs(durations[possible_inds] - dur))]
        permuted_inds.append(closest_ind)
        taken_inds.add(closest_ind)
    permuted_dataset2 = dataset2[permuted_inds]
    return dataset1, permuted_dataset2
conditions = (
    "340kb_Ce_Cp",
    "85kb",
    "170kb",
    "340kb_Ce",
    "255kb",
    "340kb",
    "340kb_Cp",
)

      
simulation_parameter_sets = [
        ("MLE", 265.6, 2.31e-2),
        ("RCest", 40, 1.15),
    ]
possible_time_intervals = ["30s", "5s"]
lags_30s = jnp.arange(-200, 201)
lags_5s = jnp.arange(-400, 401)
corrs_30s = []
random_corrs_30s = []
random_corrs_5s = []
corrs_5s = []
means_30s = []
sim_means_30s = []
sim_std_30s = []
conditions_30s = []
ep_dists_30s = []
ms2_data_30s = []
corrs_sim_30s_MLE, corrs_sim_30s_RCest = [], []
for condition in conditions:
    for time_interval in possible_time_intervals:
        treatment = "None"
        ep_dat, ms2_data = read_data(
            condition, "corrected_intensity", time_interval, treatment
        )
        if ep_dat is None:
            continue
        if time_interval == "30s":
            means_30s.append(jnp.nanmean(ms2_data))
        ep_dist = jnp.linalg.norm(ep_dat, axis=-1)
        random_ep_dist, random_ms2_data = randomize_tracks(ep_dist, ms2_data)
        if time_interval == "30s":
            conditions_30s.append(condition)
            ep_dists_30s.append(ep_dist)
            ms2_data_30s.append(ms2_data)
            corrs_30s.append(cross_corr_jax(ep_dist, ms2_data, lags_30s))
            random_corrs_30s.append(cross_corr_jax(random_ep_dist, random_ms2_data, lags_30s))
        else:
            corrs_5s.append(cross_corr_jax(ep_dist, ms2_data, lags_5s))
            random_corrs_5s.append(cross_corr_jax(random_ep_dist, random_ms2_data, lags_5s))
            
        # compute cross-corr predictions from the simulated data
        if time_interval == "30s":
            for label, rc, kon in simulation_parameter_sets:
                savepath = SIMDATA_RESPATH / f"simulated_data_{condition}_{treatment}_{time_interval}_rc{rc}_kon{kon}.pkl"  
                with open(savepath, "rb") as handle:
                    sim_data = pickle.load(handle)
                    sim_ep_dat_list = sim_data["ep_dat_list"]
                    sim_ep_dat_noisy_list = sim_data["ep_dat_noisy_list"]
                    sim_ms2_dat = sim_data["ms2_dat"]
                    sim_ms2_noisy_dat = sim_data["ms2_noisy_dat"]
                    
                if label == "MLE":
                    corrs_sim_30s_MLE.append(cross_corr_jax(jnp.linalg.norm(sim_ep_dat_noisy_list,axis=-1), sim_ms2_noisy_dat, lags_30s))
                elif label == "RCest":
                    corrs_sim_30s_RCest.append(cross_corr_jax(jnp.linalg.norm(sim_ep_dat_noisy_list,axis=-1), sim_ms2_noisy_dat, lags_30s))
                    if time_interval == "30s":
                        sim_means_30s.append(jnp.nanmean(sim_ms2_noisy_dat + offset))
                        sim_std_30s.append(
                            jnp.nanstd(sim_ms2_noisy_dat)
                            / jnp.sqrt(jnp.sum(~jnp.isnan(sim_ms2_noisy_dat)))
                        )

avg_corr_30s = jnp.nanmean(jnp.stack(corrs_30s), axis=0)
std_corr_30s = jnp.nanstd(jnp.stack(corrs_30s), axis=0)
avg_corr_5s = jnp.nanmean(jnp.stack(corrs_5s), axis=0)
std_corr_5s = jnp.nanstd(jnp.stack(corrs_5s), axis=0)
avg_random_corr_30s = jnp.nanmean(jnp.stack(random_corrs_30s), axis=0)
std_random_corr_30s = jnp.nanstd(jnp.stack(random_corrs_30s), axis=0)
avg_random_corr_5s = jnp.nanmean(jnp.stack(random_corrs_5s), axis=0)
std_random_corr_5s = jnp.nanstd(jnp.stack(random_corrs_5s), axis=0)
avg_corr_sim_30s_MLE = jnp.nanmean(jnp.stack(corrs_sim_30s_MLE), axis=0)
std_corr_sim_30s_MLE = jnp.nanstd(jnp.stack(corrs_sim_30s_MLE), axis=0)
avg_corr_sim_30s_RCest = jnp.nanmean(jnp.stack(corrs_sim_30s_RCest), axis=0)
std_corr_sim_30s_RCest = jnp.nanstd(jnp.stack(corrs_sim_30s_RCest), axis=0)
avg_corr_30s_bootstrap_samples, _, _ = (
    load_or_compute_average_cross_corr_bootstrap(
        "corrected_intensity_30s_data_average",
        conditions_30s,
        ep_dists_30s,
        ms2_data_30s,
        lags_30s,
    )
)

block_size = 15
def block_average(data, block_size):
    data = jnp.asarray(data)
    n_blocks = data.shape[-1] // block_size
    data = data[..., :n_blocks*block_size]
    return jnp.nanmean(
        data.reshape(data.shape[:-1] + (n_blocks, block_size)),
        axis=-1,
    )
avg_corr_30s_block = block_average(avg_corr_30s, block_size)
std_corr_30s_block = block_average(std_corr_30s, block_size)
avg_corr_5s_block = block_average(avg_corr_5s, block_size)
std_corr_5s_block = block_average(std_corr_5s, block_size)
avg_random_corr_30s_block = block_average(avg_random_corr_30s, block_size)
std_random_corr_30s_block = block_average(std_random_corr_30s, block_size)
avg_random_corr_5s_block = block_average(avg_random_corr_5s, block_size)
std_random_corr_5s_block = block_average(std_random_corr_5s, block_size)
avg_corr_sim_30s_MLE_block = block_average(avg_corr_sim_30s_MLE, block_size)
std_corr_sim_30s_MLE_block = block_average(std_corr_sim_30s_MLE, block_size)
avg_corr_sim_30s_RCest_block = block_average(avg_corr_sim_30s_RCest, block_size)
std_corr_sim_30s_RCest_block = block_average(std_corr_sim_30s_RCest, block_size)
avg_corr_30s_bootstrap_samples_block = block_average(avg_corr_30s_bootstrap_samples, block_size)
std_corr_30s_bootstrap_block = np.nanstd(
    np.asarray(avg_corr_30s_bootstrap_samples_block),
    axis=0,
)
lags_30s_block = block_average(lags_30s, block_size)
lags_5s_block = block_average(lags_5s, block_size)
            
condition_map = {
    "340kb_Ce_Cp":"339CECP",
    "85kb":"87noC",
    "170kb":"170noC",
    "340kb_Ce":"339CE",
    "255kb":"255noC",
    "340kb":"339noC",
    "340kb_Cp":"339CP",
}      

fig,ax = plt.subplots(1,1,figsize=(5,5))            
ax.bar([condition_map[c] for c in conditions],means_30s,color="#0077BB",label="Data")
ax.errorbar(
    [condition_map[c] for c in conditions],
    sim_means_30s,
    yerr=sim_std_30s,
    fmt="o",
    color="#EE7733",
    markersize=7,
    capsize=3,
    label="Prediction",
)
ax.legend()
ax.set(xlabel="Condition", ylabel="Mean MS2 Intensity [a.u]")
#rotate x-axis labels
plt.xticks(rotation=-45)
fig.savefig(main_fig_dir / "mean_predictions.pdf")
                                          
fig,ax = plt.subplots(1,1, figsize=(5,4))
for i,corr in enumerate(corrs_30s):
    if i!=0:
        ax.plot(lags_30s*0.5, corr, alpha=0.2,color="dimgray")
    else:
        ax.plot(lags_30s*0.5, corr, alpha=0.2,color="dimgray", label="Individual\ncell-lines")
    
# ax.plot(lags_30s*0.5, avg_corr_30s, color="black", label="Average")
# ax.plot(lags_30s*0.5, avg_random_corr_30s, color="red", label="Randomized",zorder=-1)
# ax.fill_between(lags_30s*0.5, avg_random_corr_30s-std_random_corr_30s, avg_random_corr_30s+std_random_corr_30s, color="red", alpha=0.2, label="Randomized\nstd dev.",zorder=-1)

ax.fill_between(
    lags_30s_block*0.5,
    avg_corr_30s_block - std_corr_30s_bootstrap_block,
    avg_corr_30s_block + std_corr_30s_bootstrap_block,
    color="#0077BB",
    alpha=0.18,
    linewidth=0,
    label="Data bootstrap\nstd dev.",
)
ax.plot(lags_30s_block*0.5, avg_corr_30s_block, "-o",color="#0077BB", label="Data average")
ax.plot(lags_30s_block*0.5, avg_random_corr_30s_block,color="#CC3111",zorder=-1,alpha=0.5)
ax.fill_between(lags_30s_block*0.5, avg_random_corr_30s_block-std_random_corr_30s_block, avg_random_corr_30s_block+std_random_corr_30s_block, color="#CC3111", alpha=0.1, zorder=-1,label="Data randomized")
# ax.plot(lags_30s_block*0.5, avg_corr_sim_30s_MLE_block, color="blue", label="Prediction\nr_c=266nm")
ax.plot(lags_30s_block*0.5, avg_corr_sim_30s_RCest_block, color="#EE7733", label="Prediction\nr_c= 40nm")
ax.legend()
ax.set(xlabel="Lag (min)", ylabel="Cross-correlation")
fig.tight_layout()
fig.savefig(main_fig_dir / "cross_correlation_prediction.pdf")
