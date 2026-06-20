import sys
from jax import numpy as jnp
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
MS2POSTERIOR_ROOT = SCRIPT_ROOT / "MS2Posterior"
sys.path.insert(0, str(MS2POSTERIOR_ROOT))
import numpy as np
from simulation import sample_dataset


SNRS = np.linspace(0.5, 2, 5, endpoint=True)
kons = 1 / np.linspace(15, 225, 5, endpoint=True)
koffs = 1 / np.linspace(1, 15, 5, endpoint=True)

dts = [0.1, 0.5]
Ntraces = [50, 200]
# grid search parameters
from itertools import product

param_grid = list(product(SNRS, kons, koffs, dts, Ntraces))

for sim_index in range(len(param_grid)):
    print(f"Starting simulation {sim_index+1} out of {len(param_grid)}")
    snr, kon, koff, dt, ntraj = param_grid[sim_index]
    lrate = 1.0
    noise = 1 / snr
    loading_rates = np.array([lrate, 0.0])
    tmat = np.array([[-koff, kon], [koff, -kon]])

    Tmax = 500  # min
    sampling_times = np.linspace(0, Tmax, int(Tmax / dt) + 1)
    nanfrac = 0.1
    T_rise = 3
    T_plateau = 5

    def kernel_np(x):
        return (
            np.heaviside(x, 1)
            * np.heaviside(T_plateau + T_rise - x, 1)
            * ((x / T_rise) * np.heaviside(T_rise - x, 1) + np.heaviside(x - T_rise, 1))
        )

    parameter_dict = dict(
        tmat=tmat,
        Tmax=Tmax,
        sampling_times=sampling_times,
        noise=noise,
        kernel=kernel_np,
        loading_rates=loading_rates,
        T_rise=T_rise,
        T_plateau=T_plateau,
        ntraj=ntraj,
        nanfrac=nanfrac,
    )
    observed_dataset, hidden_dataset = sample_dataset(parameter_dict, verbose=False)
    arr_dat = jnp.array(observed_dataset)

    save_path = SCRIPT_ROOT / "Data" / "MS2_test_data"
    save_path.mkdir(parents=True, exist_ok=True)
    # store the data
    import pickle

    with open(
        save_path
        / f"test_data_ind={sim_index}_snr={snr:.2f}_lrate={lrate:.2f}_kon={kon:.3f}_koff={koff:.3f}_dt={dt:.3f}_ntraj={ntraj}.pkl",
        "wb",
    ) as f:
        pickle.dump(
            {
                "observed_dataset": observed_dataset,
                "hidden_dataset": hidden_dataset,
            },
            f,
        )
