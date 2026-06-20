from pathlib import Path
import pickle
import sys
import os
from matplotlib.colors import LinearSegmentedColormap
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter1d
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

SCRIPT_ROOT = Path(__file__).resolve().parent
MAIN_FIG_DIR = SCRIPT_ROOT / "figures" / "main_figure"
MAIN_FIG_DIR.mkdir(parents=True, exist_ok=True)
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
from TwoLocusGPR.MSD_functions import Rouse_MSD
from TwoLocusGPR.GPR_utils import msd
from jax_script_utils import (
    ms2_kernel_weights,
    pcontact_hmm_loglik_batch,
    state_bits,
)




cache_dir = SCRIPT_ROOT / "cache" / "4_simdata"
dt = 0.5
loc_err = 40
sep = 90
rc_true = 30
store_path = (
                        cache_dir / f"test_data_{dt:.2f}_{loc_err}_{sep}_{rc_true}.pkl"
                    )
# load with pickle 
with open(store_path, "rb") as f:
    test_data = pickle.load(f)

alpha,Gamma,J = test_data["parameters"]["alpha"],test_data["parameters"]["Gamma"],test_data["parameters"]["J"]
simdat = test_data["results"]
observation_times = simdat[0][0]
ep_dat = np.array([s[2] for s in simdat]).swapaxes(1,2)
ms2_data = np.array([s[-3] for s in simdat])
T_rise,T_plateau,RNA_intensity = test_data["parameters"]["T_rise"],test_data["parameters"]["T_plateau"],test_data["parameters"]["alpha"]
noise = test_data["parameters"]["noise"]
loading_rates = test_data["parameters"]["loading_rates"]
koff = test_data["parameters"]["koff"]
localization_errors = test_data["parameters"]["localization_errors"]

konmin,kon_max = 1e-3,1e6
kons_to_test = np.logspace(np.log10(konmin), np.log10(kon_max), 100)
rcs_to_test = np.logspace(np.log10(5e0),3,100)
minlength = 50


marginals_llh = []
marginals_burst = []
maps_llh = []
maps_burst = []

outdir = SCRIPT_ROOT / "result"/"3_3_compute_rc_kon_map_simulations"

fit_name = f"{dt}_{loc_err}_{sep}_{rc_true}"
map_path = outdir / f"{fit_name}_kon_rc_map.npz"

if not map_path.exists():

    # pad the data to allow for pol2 loadings before the measurement begins
    n_pad = int((T_rise + T_plateau) / dt) + 1
    ms2_data = np.pad(ms2_data, ((0, 0), (n_pad, 0)), constant_values=np.nan)
    ep_dat = np.pad(
        ep_dat,
        ((0, 0), (n_pad, 0), (0, 0)),
        constant_values=np.nan,
    )
    observation_times = np.arange(ms2_data.shape[1]) * dt - n_pad * dt

    #load the msd fit parameters and data for this condition, treatment, and time interval
    paramvec = np.array([Gamma,J]+list(localization_errors))
    predictor = GPR(
        observation_times,
        Rouse_MSD,
        3,
        param_layout=((), ()),
        noise_layout=("dim",),
        param_names=("Gamma", "J"),
    )
    
    means, vars = predictor.Predict(paramvec, observation_times, ep_dat[:100])


    weights = ms2_kernel_weights(
                dt=dt,
                t_rise=T_rise,
                t_plateau=T_plateau,
                alpha=RNA_intensity,
            )
    bits = state_bits(len(weights))
    emission_means = bits @ weights
    llh_list = []
    # llh_list_burst = []
    for rc in tqdm(rcs_to_test):
        
        pcont = prob_in_sphere_quadrature(means, vars, rc).T
        sub_llh = []
        sub_llh_burst = []
        for kon in kons_to_test:
            
            load_probs = 1.0 - np.exp(-np.maximum(loading_rates[::-1], 0.0) * dt)

            llh = pcontact_hmm_loglik_batch(
                # observed_ms2=np.roll(res["fit_data"], roll, axis=1),
                observed_ms2=ms2_data[:100],
                p_contact_interval=pcont,
                kon=kon,
                koff=koff,
                dt=dt,
                load_probs=load_probs,
                emission_means=emission_means,
                noise_std=noise,
                window_size=len(weights),
            )
            # non_jump = jnp.nansum(-(poff*kon*pcont))*dt
            # jump = jnp.nansum(turnon*jnp.log(pcont*kon+1e-10))*dt
            
            sub_llh.append(jnp.sum(llh))
            # sub_llh_burst.append(jump+non_jump)
        llh_list.append(sub_llh)
        # llh_list_burst.append(sub_llh_burst)

    # store the map 
    np.savez(
        map_path,
        rcs=rcs_to_test,
        kons=kons_to_test,
        llh=np.array(llh_list),
        # llh_burst=np.array(llh_list_burst),
    )
else:
    loaded = np.load(map_path)
    rcs_to_test = loaded["rcs"]
    kons_to_test = loaded["kons"]
    llh_list = loaded["llh"]
    # llh_list_burst = loaded["llh_burst"]
# for llh_map,label in zip([llh_list,llh_list_burst],["llh","llh_burst"]):
llh_map,label = llh_list,"llh"
dkon = kons_to_test[1] - kons_to_test[0]
argmax = np.unravel_index(np.argmax(llh_map), np.array(llh_map).shape)
best_kon = kons_to_test[argmax[1]]
best_rc = rcs_to_test[argmax[0]]

marginalized = jax.scipy.special.logsumexp(jnp.array(llh_map),b=dkon,axis=1)

# if label=="llh":
maps_llh.append(np.array(llh_map))
marginals_llh.append(marginalized)
# else:
#     maps_burst.append(np.array(llh_list_burst))
#     marginals_burst.append(marginalized)


colors_pt = {
    "blue": "#0077BB",
    "magenta": "#EE3377",
    "teal": "#009988",
    "orange": "#EE7733",
    "cyan": "#33BBEE",
    "red": "#CC3311",
    "grey": "#BBBBBB"
}

pt_heatmap_cmap = LinearSegmentedColormap.from_list(
    'PT_Inferno',
    [
        (0.00, '#000000'),
        (0.35, colors_pt['blue']),
        (0.65, colors_pt['magenta']),
        (0.85, colors_pt['orange']),
        (1.00, '#FFDDAA')
    ]
)

fig = plt.figure(figsize=(5,5))
# ax1 should be a merger of the first 3x2 subplots in a 4x2 grid
ax1 = fig.add_subplot(4, 2, (1, 6))
ax2 = fig.add_subplot(4, 2, (7, 8))
ax = [ax1, ax2]

# fig,ax = plt.subplots(2,1,figsize=(5,5),sharex=True)

def log_edges(vals):
    vals = np.asarray(vals)
    mids = np.sqrt(vals[:-1] * vals[1:])
    return np.r_[vals[0] ** 2 / mids[0], mids, vals[-1] ** 2 / mids[-1]]

rc_edges = log_edges(rcs_to_test)
kon_edges = log_edges(kons_to_test)
ridge_kon = kons_to_test[np.argmax(llh_map,axis=1)]
smoothed_ridge_kon = gaussian_filter1d(ridge_kon, sigma=1)

ax[0].pcolormesh(
    rc_edges,
    kon_edges,
    np.array(llh_map).T,
    shading="auto",
    cmap=pt_heatmap_cmap,vmin=np.nanmax(np.array(llh_map))-200,vmax=np.nanmax(np.array(llh_map))
)

ax[0].set(yscale="log",xscale="log", ylabel="kon (1/min)",xlim=(rcs_to_test[0],rcs_to_test[-1]))
ax[0].scatter(best_rc,best_kon, color="k", label=f"Best fit: kon={best_kon:.1f}min^-1\nrc={round(best_rc)} nm")
ax[0].scatter(rc_true,test_data["parameters"]["kon"], color="#CC3311", label=f"True: kon={int(test_data['parameters']['kon'])}min^-1\nrc={rc_true:.1f} nm")
ax[0].legend()
# add colorbar
cbar = plt.colorbar(ax[0].collections[0], ax=ax[0])
cbar.set_label("log p(kon,rc)",rotation=270,labelpad=20)

ax[1].plot(rcs_to_test,marginalized,color="#0077BB")
ax[1].axvline(rcs_to_test[np.argmax(marginalized)], color="k", linestyle="--", label=f"Best fit: rc={rcs_to_test[np.argmax(marginalized)]:.1f} nm")
ax[1].axvline(rc_true, color="#CC3311", linestyle="--", label=f"True: rc={rc_true:.1f} nm")
ax[1].set(xlabel="Contact radius (nm)", ylabel="log p(r_c)",xscale="log",xlim=(rcs_to_test[0],rcs_to_test[-1]),ylim=(np.nanmin(marginalized)+80,np.nanmax(marginalized)+10))
fig.suptitle("Simulations")
fig.tight_layout()
fig.savefig(MAIN_FIG_DIR / "rc_kon_simdat.pdf")


fig,ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(rcs_to_test,smoothed_ridge_kon)
ax.scatter(rc_true,test_data["parameters"]["kon"], color="#CC3311", label=f"True: kon={int(test_data['parameters']['kon'])}min^-1\nrc={rc_true:.1f} nm")
ax.set(xscale="log",yscale="log",ylim=(1e-2,1e3),xlim=(5e0,1e3))
ax.legend()
fig.savefig(MAIN_FIG_DIR / "rc_kon_simdat_ridge.pdf")
