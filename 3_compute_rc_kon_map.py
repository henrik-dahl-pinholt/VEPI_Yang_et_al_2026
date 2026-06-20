from pathlib import Path
import pickle
import sys
import os
from matplotlib.colors import LinearSegmentedColormap
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter1d

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

SCRIPT_ROOT = Path(__file__).resolve().parent
TWOLOCUSGPR_ROOT = SCRIPT_ROOT / "TwoLocusGPR"
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(TWOLOCUSGPR_ROOT))


from TwoLocusGPR.Posterior_analysis import prob_in_sphere_quadrature


import jax
import json
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from TwoLocusGPR.GPR import GPR
from TwoLocusGPR.MSD_functions import FD_MSD
from script_utils import (
    _lookup_param,
    filter_short_tracks,
    time_interval_label,
    time_interval_seconds,
)
from jax_script_utils import (
    ms2_kernel_weights,
    pcontact_hmm_loglik_batch,
    state_bits,
)

DATA_ROOT = SCRIPT_ROOT / "Data" / "filtered_data_v6"
CORR_FIT_PATH = SCRIPT_ROOT / "result" / "1_MS2_corrfit" / "fit_raw_intensity_0341338401d33fe53cf19f91401d8dac_summary.json"

MSD_PATH = SCRIPT_ROOT / "cache" / "MSD_fits" / "fit_results_individual_v28_StatisticalFilter_Alpha0p37_final"
RESULT_PATH = SCRIPT_ROOT / "result" / "2_run_inference"

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
import pickle

path = MSD_PATH / "fit_params_G7B8G2_GSK_100uM_IAAadded2hrBefore.pkl"
with open(path, "rb") as f:
    fit_result = pickle.load(f)
fit_result





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
    ep_dat = loaded_data["enhancer_coordinate"] - loaded_data["promoter_coordinate"]
    ms2_data = loaded_data[intensity_key]

    return ep_dat, ms2_data


outdir = SCRIPT_ROOT / "result" / "3_compute_rc_kon_map"
outdir.mkdir(parents=True, exist_ok=True)

main_fig_dir = SCRIPT_ROOT / "figures" / "main_figure"
main_fig_dir.mkdir(parents=True, exist_ok=True)

konmin, kon_max = 1e-6, 1e6
kons_to_test = np.logspace(np.log10(konmin), np.log10(kon_max), 100)
rcs_to_test = np.logspace(0, 4, 100)
minlength = 50

# read the correlation fit results
with open(CORR_FIT_PATH, "r") as f:
    corr_fit_results = json.load(f)
keys = ["koff", "T_rise", "T_plateau", "RNA_intensity", "offset", "loading_rate"]
koff, T_rise, T_plateau, RNA_intensity, offset, loading_rate = [
    corr_fit_results["parameters"][key] for key in keys
]

time_interval = "30s"

conditions = (
    "85kb",
    "340kb_Ce_Cp",
    "170kb",
    "340kb_Ce",
    "255kb",
    "340kb_Cp",
    "340kb",
)
treatment = "None"

marginals_llh = []
marginals_burst = []
maps_llh = []
maps_burst = []

for condition in conditions:
    print(f"Processing condition {condition}...")
    fit_name = f"{time_interval}_{condition}_{treatment}"
    map_path = outdir / f"{fit_name}_kon_rc_map.npz"

    if not map_path.exists():

        base_res = np.load(
            RESULT_PATH / condition / treatment / f"{fit_name}_model_results.npz"
        )
        T_rise = float(base_res["T_rise"])
        T_plateau = float(base_res["T_plateau"])
        RNA_intensity = float(base_res["RNA_intensity"])

        dt = float(str(time_interval).removesuffix("s")) / 60
        # read the data
        ep_dat, ms2_data = read_data(
            condition, "corrected_intensity", time_interval, treatment
        )
        if ep_dat is None or ms2_data is None:
            print(
                f"\tNo data found for condition {condition}, treatment {treatment}, time interval {time_interval}. Skipping inference."
            )

        ms2_data, ep_dat = filter_short_tracks(ms2_data, ep_dat, minlength)

        # pad the data to allow for pol2 loadings before the measurement begins
        n_pad = int((T_rise + T_plateau) / dt) + 1
        ms2_data = np.pad(ms2_data, ((0, 0), (n_pad, 0)), constant_values=np.nan)
        ep_dat = np.pad(
            ep_dat,
            ((0, 0), (n_pad, 0), (0, 0)),
            constant_values=np.nan,
        )
        observation_times = np.arange(ms2_data.shape[1]) * dt - n_pad * dt

        # load the msd fit parameters and data for this condition, treatment, and time interval

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

        savedir = RESULT_PATH / condition / treatment
        savename = f"{fit_name}"
        res = np.load(savedir / f"{savename}_model_results.npz")
        turnon = res["turnon"].T
        poff = 1 - res["pon"]

        llh_list = []
        llh_list_burst = []
        for rc in rcs_to_test:
            rundir = RESULT_PATH / condition / treatment
            savedir = rundir / f"rc={rc}"
            savename = f"{fit_name}_rc={rc}"

            _, koff = res["transition_rates"]
            noises = res["noise"]
            loading_rates = res["loading_rates"][:, [1, 0]]
            pcont = prob_in_sphere_quadrature(means, vars, rc).T
            sub_llh = []
            sub_llh_burst = []
            for kon in kons_to_test:
                weights = ms2_kernel_weights(
                    dt=dt,
                    t_rise=float(res["T_rise"]),
                    t_plateau=float(res["T_plateau"]),
                    alpha=float(res["RNA_intensity"]),
                )
                bits = state_bits(len(weights))
                emission_means = bits @ weights
                load_probs = 1.0 - np.exp(-np.maximum(loading_rates, 0.0) * dt)

                llh = pcontact_hmm_loglik_batch(
                    # observed_ms2=np.roll(res["fit_data"], roll, axis=1),
                    observed_ms2=res["fit_data"],
                    p_contact_interval=pcont,
                    kon=kon,
                    koff=koff,
                    dt=dt,
                    load_probs=load_probs,
                    emission_means=emission_means,
                    noise_std=noises,
                    window_size=len(weights),
                )
                non_jump = jnp.nansum(-(poff * kon * pcont)) * dt
                jump = jnp.nansum(turnon * jnp.log(pcont * kon + 1e-10)) * dt

                sub_llh.append(jnp.sum(llh))
                sub_llh_burst.append(jump + non_jump)
            llh_list.append(sub_llh)
            llh_list_burst.append(sub_llh_burst)

        # store the map
        np.savez(
            map_path,
            rcs=rcs_to_test,
            kons=kons_to_test,
            llh=np.array(llh_list),
            llh_burst=np.array(llh_list_burst),
        )
    else:
        print(
            f"\tResults already exist for condition {condition}, treatment {treatment}, time interval {time_interval}. Skipping computation."
        )
        loaded = np.load(map_path)
        rcs_to_test = loaded["rcs"]
        kons_to_test = loaded["kons"]
        llh_list = loaded["llh"]
        llh_list_burst = loaded["llh_burst"]
    for llh_map, label in zip([llh_list, llh_list_burst], ["llh", "llh_burst"]):
        dkon = kons_to_test[1] - kons_to_test[0]
        argmax = np.unravel_index(np.argmax(llh_map), np.array(llh_map).shape)
        best_kon = kons_to_test[argmax[1]]
        best_rc = rcs_to_test[argmax[0]]

        marginalized = jax.scipy.special.logsumexp(jnp.array(llh_map), b=dkon, axis=1)

        if label == "llh":
            maps_llh.append(np.array(llh_map))
            marginals_llh.append(marginalized)
        else:
            maps_burst.append(np.array(llh_list_burst))
            marginals_burst.append(marginalized)

        fig, ax = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

        def log_edges(vals):
            vals = np.asarray(vals)
            mids = np.sqrt(vals[:-1] * vals[1:])
            return np.r_[vals[0] ** 2 / mids[0], mids, vals[-1] ** 2 / mids[-1]]

        rc_edges = log_edges(rcs_to_test)
        kon_edges = log_edges(kons_to_test)

        ax[0].pcolormesh(
            rc_edges,
            kon_edges,
            np.array(llh_map).T,
            shading="auto",
            cmap="viridis",
            vmin=np.nanmax(np.array(llh_map)) - 2000,
            vmax=np.nanmax(np.array(llh_map)),
        )

        ax[0].set(yscale="log", xscale="log", ylabel="kon (1/min)")
        ax[0].scatter(
            best_rc,
            best_kon,
            color="red",
            label=f"Best fit: kon={best_kon:.2e}, rc={best_rc:.1f}",
        )

        ax[0].legend()

        ax[1].plot(rcs_to_test, marginalized, marker="o")
        ax[1].axvline(
            rcs_to_test[np.argmax(marginalized)],
            color="red",
            linestyle="--",
            label=f"Best fit: rc={rcs_to_test[np.argmax(marginalized)]:.1f} nm",
        )
        ax[1].set(xlabel="Contact radius (nm)", ylabel="log sum_kon exp(llh(kon,rc))")
        fig.tight_layout()
        fig.suptitle(f"Kon-RC Map ({condition}, {label})")
        if label == "llh":
            fig.savefig(outdir / f"{fit_name}_kon_rc_map.png")
        else:
            fig.savefig(outdir / f"{fit_name}_kon_rc_map_{label}.png")

        plt.close(fig)
# for maps,marginals,label in zip([maps_llh,maps_burst],[marginals_llh,marginals_burst],["llh","llh_burst"]):


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
maps, marginals, label = maps_llh, marginals_llh, "llh"
rc_min_range, rc_max_range = 32, 52
rc_min_idx = np.argmin(np.abs(rcs_to_test - rc_min_range))
rc_max_idx = np.argmin(np.abs(rcs_to_test - rc_max_range))

ridge_kon = kons_to_test[np.argmax(maps[0], axis=1)]
smoothed_ridge_kon = gaussian_filter1d(ridge_kon, sigma=1)
kon_min_val = kons_to_test[np.argmax(maps[0].T[:, rc_max_idx])]
kon_max_val = kons_to_test[np.argmax(maps[0].T[:, rc_min_idx])]

shaded_rc_upper = smoothed_ridge_kon[rc_min_idx : rc_max_idx + 1]
shaed_kon_upper = np.concatenate(([kon_max_val] * rc_min_idx, shaded_rc_upper))


argmax = np.unravel_index(np.argmax(maps[0]), np.array(maps[0]).shape)
best_kon = kons_to_test[argmax[1]]
best_rc = rcs_to_test[argmax[0]]
fig = plt.figure(figsize=(5, 5))
ax1 = fig.add_subplot(4, 2, (1, 6))
ax2 = fig.add_subplot(4, 2, (7, 8))
ax = [ax1, ax2]
im = ax[0].pcolormesh(
    rc_edges,
    kon_edges,
    np.sum(maps, axis=0).T - np.nanmax(np.sum(maps, axis=0)),
    shading="auto",
    cmap=pt_heatmap_cmap,
    vmin=-5_000,
    vmax=0,
)
ax[0].contour(
    rc_edges[:-1],
    kon_edges[:-1],
    np.sum(maps, axis=0).T - np.nanmax(np.sum(maps, axis=0)),
    levels=-np.linspace(10, 10_000, 5)[::-1],
    colors="dimgrey",
    linestyles="solid",
    linewidths=1,
    alpha=0.3,
)
# ax[0].scatter(best_rc,best_kon, color="red", label=f"Best fit\nkon={best_kon:.2e}min^-1\nrc={best_rc:.1f}nm", zorder=5)


ax[0].plot(
    rcs_to_test[:50],
    1e7 * rcs_to_test[:50] ** (-3),
    color="white",
    linestyle="--",
    label="kon ~ rc^-3",
)

ax[0].fill_between(
    rcs_to_test[rc_min_idx : rc_max_idx + 1],
    kons_to_test[0],
    shaded_rc_upper,
    alpha=0.3,
    color="white",
    label=f"rc\in[32,52]\nkon\in[{kon_min_val:.1f}, {kon_max_val:.1f}]",
)
ax[0].fill_between(
    rcs_to_test[: rc_max_idx + 1],
    kon_min_val,
    shaed_kon_upper,
    alpha=0.3,
    color="white",
)
ax[0].legend()

ax[0].set(
    yscale="log",
    xscale="log",
    ylabel="kon (1/min)",
    xlim=(rcs_to_test[0], 1e3),
    ylim=(1e-4, kons_to_test[-1]),
)
marginalized = jax.scipy.special.logsumexp(np.sum(maps, axis=0), b=dkon, axis=1)

ax[1].plot(
    rcs_to_test,
    marginalized - np.nanmax(marginalized),
    color="k",
    label="Marginalized\nevidence",
)

for i, curve in enumerate(marginals):
    if i != 0:
        ax[1].plot(rcs_to_test, (curve - np.nanmax(curve)), color="gray", alpha=0.3)
    else:
        ax[1].plot(
            rcs_to_test,
            (curve - np.nanmax(curve)),
            color="gray",
            alpha=0.3,
            label="Individual\ncell-lines",
        )
ax[1].axvspan(
    rcs_to_test[rc_min_idx + 1], rcs_to_test[rc_max_idx], alpha=0.3, color="lightgray"
)
ax[1].set(
    xlabel="Contact radius (nm)",
    ylabel="Log p(r_c)",
    xlim=(rcs_to_test[0], 1e3),
    xscale="log",
)
ax[1].legend()
# add colorbar in a manner that doesn't mess with the axes being shared
cbar = plt.colorbar(ax[0].collections[0], ax=ax[0])
cbar.set_label("log p(kon,rc)", rotation=270, labelpad=20)

fig.suptitle(f"Combined Kon-RC Map ({label})")
fig.tight_layout()
fig.savefig(main_fig_dir / f"rc_kon_data.pdf", bbox_inches="tight")


fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(rcs_to_test, smoothed_ridge_kon)

ax.fill_between(
    rcs_to_test[rc_min_idx : rc_max_idx + 1],
    kons_to_test[0],
    shaded_rc_upper,
    alpha=0.3,
    color="dimgrey",
    label=f"rc\in[32,52]\nkon\in[{kon_min_val:.1f}, {kon_max_val:.1f}]",
)
ax.fill_between(
    rcs_to_test[: rc_max_idx + 1],
    kon_min_val,
    shaed_kon_upper,
    alpha=0.3,
    color="dimgrey",
)
kon = 1.15
rc = 40

ax.plot(
    rcs_to_test[:50],
    1e6 * rcs_to_test[:50] ** (-3),
    color="k",
    linestyle="--",
    label="kon ~ rc^-3",
)

ax.scatter(40, 1.15, label=f"Estimated kon={kon:.2f}min^-1\nrc={rc}nm")
ax.set(xscale="log", yscale="log", ylim=(1e-2, 1e3), xlim=(5e0, 1e3))
ax.legend()
fig.savefig(main_fig_dir / f"rc_kon_data_ridge.pdf", bbox_inches="tight")


# if label=="llh":
#     fig.savefig(main_fig_dir / f"combined_kon_rc_map.pdf", bbox_inches="tight")
# else:
#     fig.savefig(main_fig_dir / f"combined_kon_rc_map_{label}.pdf", bbox_inches="tight")


# maps,label = maps_llh,"llh"
# rc_min_range,rc_max_range = 32, 52
# rc_min_idx = np.argmin(np.abs(rcs_to_test-rc_min_range))
# rc_max_idx = np.argmin(np.abs(rcs_to_test-rc_max_range))

# ridge_kon = kons_to_test[np.argmax(maps[0],axis=1)]
# smoothed_ridge_kon = ridge_kon
# kon_min_val = kons_to_test[np.argmax(maps[0].T[:,rc_max_idx])]
# kon_max_val = kons_to_test[np.argmax(maps[0].T[:,rc_min_idx])]

# shaded_rc_upper = smoothed_ridge_kon[rc_min_idx:rc_max_idx+1]
# shaed_kon_upper = np.concatenate(([kon_max_val]*rc_min_idx, shaded_rc_upper))


# argmax = np.unravel_index(np.argmax(maps[0]), np.array(maps[0]).shape)
# best_kon = kons_to_test[argmax[1]]
# best_rc = rcs_to_test[argmax[0]]
# fig = plt.figure(figsize=(6,6))
# ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=2)
# ax2 = plt.subplot2grid((3, 4), (2, 0), colspan=3, rowspan=1)
# ax = [ax1, ax2]
# im = ax[0].pcolormesh(
#     rc_edges,
#     kon_edges,
#     np.sum(maps,axis=0).T-np.nanmax(np.sum(maps,axis=0)),
#     shading="auto",
#     cmap="viridis",vmin=-100,vmax=0)
# ax[0].contour(
#     rc_edges[:-1],
#     kon_edges[:-1],
#     np.sum(maps,axis=0).T-np.nanmax(np.sum(maps,axis=0)),
#     levels=-np.linspace(0,1000, 5)[::-1],colors="dimgrey", linestyles="solid", linewidths=1,alpha=0.3
# )
# ax[0].scatter(best_rc,best_kon, color="red", label=f"Best fit\nkon={best_kon:.2e}min^-1\nrc={best_rc:.1f}nm", zorder=5)

# ax[0].plot(rcs_to_test,smoothed_ridge_kon, color="red", label="Best kon at each rc")
# ax[0].plot(rcs_to_test[:30],1e4*rcs_to_test[:30]**(-3), color="k", linestyle="--", label="kon ~ rc^-3")

# ax[0].fill_between(rcs_to_test[rc_min_idx:rc_max_idx+1], kons_to_test[0], shaded_rc_upper, alpha=0.3,color="darkred", label="Range suggested\nby other figures")
# ax[0].fill_between(rcs_to_test[:rc_max_idx+1], kon_min_val,shaed_kon_upper, alpha=0.3,color="darkred")
# ax[0].legend()

# ax[0].set(yscale="log",xscale="log", ylabel="kon (1/min)",xlim=(1e2,1e3),ylim=(5e-3,3e-1))
# marginalized = jax.scipy.special.logsumexp(np.sum(maps,axis=0),b=dkon,axis=1)

# ax[1].plot(rcs_to_test,marginalized-np.nanmax(marginalized),color="k",label="Marginalized\nevidence")
# for i,curve in enumerate(marginals):
#     if i !=0:
#         ax[1].plot(rcs_to_test,(curve-np.nanmax(curve)),color="gray", alpha=0.3)
#     else:
#         ax[1].plot(rcs_to_test,(curve-np.nanmax(curve)),color="gray", alpha=0.3,label="Individual\ncell-lines")
# ax[1].set(xlabel="Contact radius (nm)", ylabel="Log p(r_c | data)",xscale="log",xlim=(1e2,1e3))
# ax[1].legend()
# # add colorbar in a manner that doesn't mess with the axes being shared
# cbar_ax = fig.add_axes([0.8, 0.45, 0.02, 0.5])
# cbar = fig.colorbar(im, cax=cbar_ax)
# cbar.set_label("Log p(r_c,k_on|data) (max-subtracted)", rotation=270, labelpad=15)
# fig.suptitle(f"Combined Kon-RC Map ({label})")
# fig.tight_layout()
