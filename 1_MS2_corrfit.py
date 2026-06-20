from pathlib import Path
import hashlib
import json
import pickle
import time
import jax
from typing import Callable, Iterable
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm.auto import tqdm
from script_utils import (
    _array_digest,
    _json_dumps,
    _safe_name,
    _save_npz_atomic,
)
from jax_script_utils import acf_jax

SCRIPT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_ROOT / "Data" / "filtered_data_v6"
CACHE_DIR = SCRIPT_ROOT / "cache"/ "1_MS2_corrfit"
RESULT_DIR = SCRIPT_ROOT / "result" / "1_MS2_corrfit"
CACHE_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

BOOTSTRAP_NSAMPLES = 10_000
VARIANCE_CACHE_VERSION = 1
FIT_CACHE_VERSION = 1





def _cache_digest(metadata: dict, arrays: Iterable[np.ndarray | jnp.ndarray]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(_json_dumps(metadata).encode())
    for arr in arrays:
        digest.update(_array_digest(arr).encode())
    return digest.hexdigest()



def _save_pickle_atomic(path: Path, value):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(value, handle)
    tmp_path.replace(path)


def _save_json_atomic(path: Path, value: dict):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def _jsonable(value):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    return arr.tolist()


def _variance_cache_path(
    dataset_label: str,
    condition: str,
    dataset: jnp.ndarray,
    lags: jnp.ndarray,
    nsamples: int,
) -> Path:
    metadata = {
        "cache_type": "bootstrap_acf_variance",
        "version": VARIANCE_CACHE_VERSION,
        "dataset_label": dataset_label,
        "condition": condition,
        "nsamples": nsamples,
    }
    key = _cache_digest(metadata, arrays=(dataset, lags))
    # create cache dir if not there
    if not (CACHE_DIR / "Bootstrapped_variances").exists():
        (CACHE_DIR / "Bootstrapped_variances").mkdir(parents=True, exist_ok=True)
    return (
        CACHE_DIR
        / "Bootstrapped_variances"
        / f"variance_{_safe_name(dataset_label)}_{_safe_name(condition)}_{key}.npz"
    )


def load_or_compute_bootstrap_variance(
    dataset_label: str,
    condition: str,
    dataset: jnp.ndarray,
    lags: jnp.ndarray,
    nsamples: int = BOOTSTRAP_NSAMPLES,
) -> jnp.ndarray:
    cache_path = _variance_cache_path(dataset_label, condition, dataset, lags, nsamples)
    if cache_path.exists():
        print(f"Loaded cached variance: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as cached:
            return jnp.array(cached["variance"])

    print(f"Computing bootstrap variance: {dataset_label} / {condition}")
    covariance_matrix = np.asarray(bootstrap_acf(dataset, lags, nsamples=nsamples))
    variance = np.diag(covariance_matrix)
    metadata = {
        "cache_type": "bootstrap_acf_variance",
        "version": VARIANCE_CACHE_VERSION,
        "dataset_label": dataset_label,
        "condition": condition,
        "nsamples": nsamples,
        "dataset_shape": list(np.asarray(dataset).shape),
        "lags": np.asarray(lags).tolist(),
    }
    _save_npz_atomic(
        cache_path,
        covariance=covariance_matrix,
        variance=variance,
        metadata=np.array(_json_dumps(metadata)),
    )
    print(f"Saved variance cache: {cache_path}")
    return jnp.array(variance)


def _fit_result_path(dataset_label: str, metadata: dict, arrays: Iterable) -> Path:
    key = _cache_digest(metadata, arrays=arrays)
    return RESULT_DIR / f"fit_{_safe_name(dataset_label)}_{key}.pkl"


def _new_fit_state(metadata: dict, target_nfits: int) -> dict:
    return {
        "metadata": metadata,
        "target_nfits": target_nfits,
        "losses": [],
        "xs": [],
        "predictions": [],
        "success": [],
        "status": [],
        "messages": [],
        "nit": [],
        "nfev": [],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "updated_at": None,
    }


def _load_fit_state(path: Path, metadata: dict, target_nfits: int) -> dict:
    if not path.exists():
        return _new_fit_state(metadata, target_nfits)
    with path.open("rb") as handle:
        state = pickle.load(handle)
    return state


def run_fit_with_cache(
    dataset_label: str,
    metadata: dict,
    arrays_for_key: Iterable,
    nfits: int,
    init_guess: jnp.ndarray,
    init_scale: float,
    bounds: np.ndarray,
    loss_fn: Callable,
    loss_grad_fn: Callable,
    predict_fn: Callable,
) -> tuple[dict, Path, int]:
    result_path = _fit_result_path(dataset_label, metadata, arrays_for_key)
    state = _load_fit_state(result_path, metadata, nfits)
    completed = len(state["losses"])

    if completed >= nfits:
        print(f"Loaded cached fit: {result_path} ({completed}/{nfits} starts)")
    else:
        if completed:
            print(f"Resuming cached fit: {result_path} ({completed}/{nfits} starts)")
        else:
            print(f"Starting fit cache: {result_path}")
        pbar = tqdm(total=nfits, initial=completed)
        for i in range(completed, nfits):
            if i == 0:
                init_here = np.asarray(jnp.log(init_guess), dtype=float)
            else:
                best_fit = int(np.argmin(state["losses"]))
                best_params = np.asarray(state["xs"][best_fit], dtype=float)
                init_here = best_params + np.random.normal(
                    scale=init_scale, size=np.asarray(init_guess).shape
                )

            res = minimize(
                loss_fn, init_here, method="L-BFGS-B", jac=loss_grad_fn, bounds=bounds
            )
            pred = np.asarray(predict_fn(res.x))
            loss_value = float(np.asarray(loss_fn(res.x)))

            state["losses"].append(loss_value)
            state["xs"].append(np.asarray(res.x))
            state["predictions"].append(pred)
            state["success"].append(bool(res.success))
            state["status"].append(int(res.status))
            state["messages"].append(str(res.message))
            state["nit"].append(int(getattr(res, "nit", -1)))
            state["nfev"].append(int(getattr(res, "nfev", -1)))
            state["target_nfits"] = nfits
            state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            _save_pickle_atomic(result_path, state)

            best_fit = int(np.argmin(state["losses"]))
            pbar.set_description(f"Best fit so far: {state['losses'][best_fit]:.2f}")
            pbar.update()
        pbar.close()

    best_fit = int(np.argmin(state["losses"]))
    return state, result_path, best_fit


def write_fit_outputs(
    dataset_label: str,
    fit_state: dict,
    fit_state_path: Path,
    best_fit: int,
    best_x: np.ndarray,
    pred: jnp.ndarray,
    acf_after_filtering: jnp.ndarray,
    covars: jnp.ndarray,
    conditions: tuple[str, ...],
    parameters: dict,
) -> tuple[Path, Path]:
    base = fit_state_path.with_suffix("")
    arrays_path = base.with_name(base.name + "_best.npz")
    summary_path = base.with_name(base.name + "_summary.json")

    _save_npz_atomic(
        arrays_path,
        best_x=np.asarray(best_x),
        best_prediction=np.asarray(pred),
        acf_after_filtering=np.asarray(acf_after_filtering),
        covariance_diagonal=np.asarray(covars),
        losses=np.asarray(fit_state["losses"]),
        best_fit=np.array(best_fit),
        conditions=np.asarray(conditions),
    )
    summary = {
        "dataset_label": dataset_label,
        "fit_state": str(fit_state_path),
        "best_arrays": str(arrays_path),
        "best_fit": int(best_fit),
        "best_loss": float(fit_state["losses"][best_fit]),
        "n_completed": len(fit_state["losses"]),
        "target_nfits": int(fit_state["target_nfits"]),
        "parameters": parameters,
    }
    _save_json_atomic(summary_path, summary)
    return arrays_path, summary_path


@jax.jit
def get_ss_jax(Q, acts_on: str = "col"):
    """
    Steady state for a CTMC rate matrix Q.

    acts_on = "col": evolution is p' = Q p  (rows sum to 0)  -> solve Q @ pi = 0
    acts_on = "row": evolution is p'^T = p^T Q (cols sum to 0) -> solve Q.T @ pi = 0
    """
    n = Q.shape[0]
    A = Q if acts_on == "col" else Q.T  # A @ pi = 0
    # Constrained least squares: [A; 1^T] pi = [0; 1]
    M = jnp.vstack([A, jnp.ones((1, n))])
    b = jnp.zeros((n + 1,)).at[-1].set(1.0)
    pi, *_ = jnp.linalg.lstsq(M, b, rcond=None)

    # Numerical hygiene: nonnegative + renormalize (keeps JIT-friendly ops)
    pi = jnp.clip(pi, 0.0, jnp.inf)
    pi = pi / jnp.sum(pi)
    return pi


@jax.jit
def Lambda(t: float, ss: jnp.ndarray, l: jnp.ndarray, tmat: jnp.ndarray) -> jnp.ndarray:
    """Compute Lambda(t) = sum_{i,k} l_i [e^{Qt}]_{ik} pi_k l_k.

    Parameters
    ----------
    t : float
        Lag time.
    ss : jnp.ndarray
        Stationary distribution ``π`` of shape ``(nstates,)``.
    l : jnp.ndarray
        State-dependent correlation matrix of rates/labels, shape ``(nstates,nstates)``.
    tmat : jnp.ndarray
        CTMC generator matrix ``Q`` with shape ``(nstates, nstates)``.

    Returns
    -------
    jnp.ndarray
        Scalar kernel expectation at lag ``t``.
    """
    abs_t = jnp.abs(t)

    def pos(tt):
        P = jax.scipy.linalg.expm(tmat * tt)  # (n,n)
        return P * ss[None, :]

    def neg(tt):
        Pt = jax.scipy.linalg.expm(tmat.T * tt)  # (n,n)
        return Pt * ss[:, None]

    prop = jax.lax.cond(t < 0, neg, pos, abs_t)  # (n,n)
    return jnp.sum(l * prop)


dbl_vmap_lambda = jax.vmap(
    jax.vmap(Lambda, (0, None, None, None)), (0, None, None, None)
)
vmap_lambda = jax.vmap(Lambda, (0, None, None, None))


@jax.jit
def Kernel_autocorrelation(t: jnp.ndarray, a: float, b: float) -> jnp.ndarray:
    """Autocorrelation of a ramp-then-plateau MS2 kernel.

    Parameters
    ----------
    t : jnp.ndarray
        Lag times.
    a : float
        Rise time ``T_rise``.
    b : float
        Plateau duration ``T_plateau``.

    Returns
    -------
    jnp.ndarray
        Kernel autocorrelation at ``t``.
    """
    # R(t) = ∫ f(u) f(u+t) du, even, support |t| ≤ a+b
    K = lambda z: jnp.clip(z, 0.0, a)

    # rect ⋆ rect
    mm = jnp.maximum(0.0, b - jnp.abs(t))

    # ramp ⋆ rect (one direction)  **NOTE the 1/(2a)**
    def rm(s):
        return (1.0 / (2.0 * a)) * (K(a + b - s) ** 2 - K(a - s) ** 2)

    # include both directions to make it even
    rm_sym = rm(t) + rm(-t)

    # ramp ⋆ ramp
    U = K(a - t)  # = min(max(a - t, 0), a)
    L = K(-t)  # = min(max(-t, 0), a)
    rr = (1.0 / a**2) * ((U**3 - L**3) / 3.0 + 0.5 * t * (U**2 - L**2))

    return mm + rm_sym + rr


@jax.jit
def sumI1(
    t: jnp.ndarray,
    ts: jnp.ndarray,
    ss: jnp.ndarray,
    arr_lrates: jnp.ndarray,
    dt: float,
    tmat: jnp.ndarray,
    T_rise: float,
    T_plateau: float,
) -> jnp.ndarray:
    """Numerically integrate the cross term in the ACF expression.

    Parameters
    ----------
    t : jnp.ndarray
        Lag times.
    ts : jnp.ndarray
        Integration grid over kernel support.
    ss : jnp.ndarray
        Stationary distribution, shape ``(nstates,)``.
    arr_lrates : jnp.ndarray
        Correlation matrix of the state-dependent loading rates, shape ``(nstates,nstates,)``.
    dt : float
        Integration step for ``ts``.
    tmat : jnp.ndarray
        Generator matrix, shape ``(nstates, nstates)``.
    T_rise : float
        Kernel rise time.
    T_plateau : float
        Kernel plateau duration.

    Returns
    -------
    jnp.ndarray
        Integrated cross term evaluated at lag ``t``.
    """
    integrand = Kernel_autocorrelation(ts, T_rise, T_plateau) * vmap_lambda(
        t + ts, ss, arr_lrates, tmat
    )
    return jnp.sum(integrand * dt)


vmap_I1 = jax.vmap(sumI1, (0, None, None, None, None, None, None, None))


@jax.jit
def kernel_integral(a: float, b: float) -> float:
    """Integral of ramp-plateau kernel over time.

    Parameters
    ----------
    a : float
        Rise duration.
    b : float
        Plateau duration.

    Returns
    -------
    float
        Kernel area.
    """
    return a / 2 + b


@jax.jit
def corrfunc(
    lag_times: jnp.ndarray,
    tmat: jnp.ndarray,
    loading_rates: jnp.ndarray,
    T_plateau: float,
    T_rise: float,
    noise: float,
    RNA_int: float,
    loading_variation_scale: jnp.ndarray | None = None,
    npoints: int = 100,
) -> jnp.ndarray:
    """Closed-form kernel for the autocovariance of MS2 traces under CTMC loading.

    Parameters
    ----------
    lag_times : jnp.ndarray
        Lag times at which to evaluate the autocovariance, shape ``(nlags,)``.
    tmat : jnp.ndarray
        CTMC generator matrix, shape ``(nstates, nstates)``.
    loading_rates : jnp.ndarray
        Loading rates for each state, shape ``(nstates,)``.
    T_plateau : float
        Plateau duration of the MS2 kernel.
    T_rise : float
        Rise time of the MS2 kernel.
    noise : float
        Observation noise standard deviation.
    RNA_int : float
        Intensity per polymerase.
    loading_variation_scale : jnp.ndarray | None, optional
        If not None, relative standard deviation of variation in loading rates across trajectories.
        If None, no variation is applied and the same loading rates are used for all trajectories.
    npoints : int, default=1000
        Number of quadrature points for the kernel integral.
    Returns
    -------
    jnp.ndarray
        Concatenated mean term and autocovariance values, shape ``(nlags+1,)``.
    """
    if loading_variation_scale is None:
        loading_corr = jnp.outer(loading_rates, loading_rates)
    else:
        var = (loading_variation_scale * loading_rates) ** 2
        loading_corr = jnp.outer(loading_rates, loading_rates) + jnp.diag(var)
    ss = get_ss_jax(tmat)
    I2 = jnp.sum(ss * loading_rates) * Kernel_autocorrelation(
        lag_times, T_rise, T_plateau
    )
    tmin, tmax = -(T_plateau + T_rise), T_plateau + T_rise
    ts = jnp.linspace(tmin, tmax, npoints)
    dt = ts[1] - ts[0]
    I1 = vmap_I1(lag_times, ts, ss, loading_corr, dt, tmat, T_rise, T_plateau)

    res = I2 + I1
    noise_term = noise**2 * (lag_times == 0)
    mean_term = (
        RNA_int**2
        * jnp.sum(ss * loading_rates) ** 2
        * kernel_integral(T_rise, T_plateau) ** 2
    )
    mean = RNA_int * kernel_integral(T_rise, T_plateau) * jnp.sum(ss * loading_rates)
    return jnp.concatenate(
        [jnp.array([mean]), RNA_int**2 * (res) + noise_term - mean_term]
    )



@jax.jit
def acf_w_mean(dataset, lags):
    acf = acf_jax(dataset, lags)
    mean = jnp.nanmean(dataset)
    return jnp.concatenate([jnp.array([mean]), acf])


def read_data(condition, datakey, time_interval):
    path = DATA_ROOT / f"{time_interval}s_{condition}_None.npz"
    return jnp.array(np.load(path, allow_pickle=True)[datakey].astype(float))


from sklearn.covariance import OAS


def bootstrap_acf(
    arr_dat: jnp.ndarray,
    lags: Iterable[int],
    nsamples: int = 10_000,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Block-bootstrap the autocovariance function and its covariance.

    Parameters
    ----------
    arr_dat : jnp.ndarray
        Input trajectories with possible NaNs, shape ``(ntraj, ntimes)``.
    lags : Iterable[int]
        Lag times (in samples) to fit.
    nsamples : int, default=10_000
        Number of bootstrap replicates.


    acf_centering : {"global", "time"}, default="global"
        Centering convention for the ACF statistic. ``"global"`` subtracts the
        scalar sample mean squared. ``"time"`` subtracts each bootstrap sample's
        ensemble mean at each time point before computing the ACF.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        Covariance matrix of ACF estimates (shape ``(block_size+1, block_size+1)``)
        and the bootstrap mean vector (length ``block_size+1`` including the mean term).
    """
    ntraj = arr_dat.shape[0]

    @jax.jit
    def sample_experiment(key):
        traj_inds = jax.random.randint(key, shape=(ntraj,), minval=0, maxval=ntraj)
        return arr_dat[traj_inds]

    key = jax.random.PRNGKey(np.random.randint(0, 1e6))

    @jax.jit
    def scan(carry, key):
        n, mean, cov = carry

        bootstrap_sample = sample_experiment(key)

        # compute acf
        result = acf_w_mean(bootstrap_sample, lags)

        # update state
        n += 1
        newmean = mean + (result - mean) / n
        newcov = cov + (result - mean)[:, None] * (result - newmean)[None, :]

        return (n, newmean, newcov), result

    init_carry = (
        0,
        jnp.zeros((len(lags) + 1,)),
        jnp.zeros((len(lags) + 1, len(lags) + 1)),
    )
    for i in tqdm(list(range(nsamples))):
        key, subkey = jax.random.split(key)
        init_carry, result = scan(init_carry, subkey)
    n, mean, cov = init_carry
    covariance_matrix = cov / (n - 1)

    return covariance_matrix


@jax.jit
def gen_tmat(kon, koff):
    # tmat = jnp.array([[-koff, kon, 0], [0, -kon, kon], [koff, 0, -kon]])
    tmat = jnp.array([[-koff, kon], [koff, -kon]])
    return tmat



vmap_gen_tmat = jax.vmap(gen_tmat, (0, None))
vmap_corrfunc = jax.vmap(corrfunc, (0, 0, None, None, None, 0, None, None))

conditions = (
    "1.5kb",
    "340kb_Ce_Cp",
    "85kb",
    "170kb",
    "340kb_Ce",
    "255kb",
    "340kb_Cp",
    "340kb",
)


max_lag = 200
dt30 = 0.5  # minutes
nfits = 50
init_scale = 2.0
lags = jnp.arange(0, max_lag + 1)
lags_expanded = jnp.array([lags * dt30] * len(conditions))
dataset_label = "raw_intensity"

datasets_after_filtering = [read_data(condition, "intensity", 30) for condition in conditions]
acf_after_filtering = jnp.array(
    [acf_w_mean(dataset, lags) for dataset in datasets_after_filtering]
)

covars = jnp.array(
    [
        load_or_compute_bootstrap_variance(
            dataset_label,
            condition,
            dataset,
            lags,
            nsamples=BOOTSTRAP_NSAMPLES,
        )
        for condition, dataset in zip(conditions, datasets_after_filtering)
    ]
)

acf_after_filtering = jnp.array(acf_after_filtering)

@jax.jit
def predict(x):
    kons = jnp.exp(x)[: len(conditions)]

    noises = jnp.exp(x)[
        len(conditions) : len(datasets_after_filtering) + len(conditions)
    ]

    cv, koff, loading_rate, T_max, T_frac = jnp.exp(x)[
        len(datasets_after_filtering)
        + len(conditions) : len(datasets_after_filtering)
        + len(conditions)
        + 5
    ]
    T_rise = T_max * T_frac
    T_plateau = T_max * (1 - T_frac)
    RNA_intensity = jnp.exp(x)[-2]
    offset = jnp.exp(x)[-1]
    tmats = vmap_gen_tmat(kons, koff)
    pred = vmap_corrfunc(
        lags_expanded,
        tmats,
        jnp.array([loading_rate, 0.0]),
        T_plateau,
        T_rise,
        noises,
        RNA_intensity,
        cv,
    )
    # add offset to mean term
    pred = pred.at[:, 0].add(offset)
    return pred

vmap_solve = jax.vmap(jnp.linalg.solve, (0, 0))

@jax.jit
def loss(x):

    diff = predict(x) - acf_after_filtering  # shape (ndatasets, nlags)

    return jnp.sum(diff**2 / covars)  # shape ()

loss_grad = jax.grad(loss)

kon_bounds = [(1 / 10000, 10.0)] * len(conditions)
noise_bounds = [(1e-3, 1e3)] * len(datasets_after_filtering)
cv_bounds = [(1e-3, 10.0)]
koff_bounds = [(1 / 10000, 10.0)]
loading_rate_bounds = [(1e-3, 1e3)]
T_max_bounds = [(1.0, 10.0)]
T_frac_bounds = [(0.01, 0.99)]
RNA_intensity_bounds = [(1e-1, 1e2)]
offset_bounds = [(1e-3, 1e3)]
bounds = np.log(
    kon_bounds
    + noise_bounds
    + cv_bounds
    + koff_bounds
    + loading_rate_bounds
    + T_max_bounds
    + T_frac_bounds
    + RNA_intensity_bounds
    + offset_bounds
)

kon_guesses = jnp.array([1 / 60] * len(conditions))
loading_rate_guess = 1.0
koff_guess = 1 / 60
T_max_guess = 6.0
T_frac_guess = 0.5
RNA_intensity_guess = 2.0
offset_init = 1.0
cv = 0.5
noise_guess = jnp.array([1.0] * len(datasets_after_filtering))
init_guess = jnp.concatenate(
    [
        kon_guesses,
        noise_guess,
        jnp.array(
            [cv, koff_guess, loading_rate_guess, T_max_guess, T_frac_guess]
            + [RNA_intensity_guess, offset_init]
        ),
    ]
)

fit_metadata = {
    "cache_type": "ms2_corrfit",
    "version": FIT_CACHE_VERSION,
    "dataset_label": dataset_label,
    "conditions": list(conditions),
    "max_lag": int(max_lag),
    "dt30": float(dt30),
    "init_scale": float(init_scale),
    "bounds": np.asarray(bounds).tolist(),
}
fit_state, fit_state_path, best_fit = run_fit_with_cache(
    dataset_label=dataset_label,
    metadata=fit_metadata,
    arrays_for_key=(acf_after_filtering, covars, lags_expanded, init_guess),
    nfits=nfits,
    init_guess=init_guess,
    init_scale=init_scale,
    bounds=bounds,
    loss_fn=loss,
    loss_grad_fn=loss_grad,
    predict_fn=predict,
)
best_x = np.asarray(fit_state["xs"][best_fit])
pred = jnp.array(fit_state["predictions"][best_fit])

kons_fit = jnp.exp(best_x)[: len(conditions)]
noises = jnp.exp(best_x)[
    len(conditions) : len(conditions) + len(datasets_after_filtering)
]

cv, koff_fit, loading_rate_fit, T_max_fit, T_frac_fit = jnp.exp(best_x)[
    len(conditions)
    + len(datasets_after_filtering) : len(conditions)
    + len(datasets_after_filtering)
    + 5
]
T_rise_fit = T_max_fit * T_frac_fit
T_plateau_fit = T_max_fit * (1 - T_frac_fit)
RNA_intensity_fit = jnp.exp(best_x)[-2]
offset_fit = jnp.exp(best_x)[-1]
print(f"Best fit parameters:")
print(f"  kons: {kons_fit}")
print(f"  noises: {noises}")
print(f"  cv: {cv}")
print(f"  koff: {koff_fit}")
print(f"  loading_rate: {loading_rate_fit}")
print(f"  T_max: {T_max_fit}")
print(f"  T_frac: {T_frac_fit}")
print(f"  T_plateau: {T_plateau_fit}")
print(f"  T_rise: {T_rise_fit}")
print(f"  RNA_intensity: {RNA_intensity_fit}")
print(f"  offset: {offset_fit}")

fit_parameters = {
    "kons": _jsonable(kons_fit),
    "noises": _jsonable(noises),
    "cv": _jsonable(cv),
    "koff": _jsonable(koff_fit),
    "loading_rate": _jsonable(loading_rate_fit),
    "T_max": _jsonable(T_max_fit),
    "T_frac": _jsonable(T_frac_fit),
    "T_plateau": _jsonable(T_plateau_fit),
    "T_rise": _jsonable(T_rise_fit),
    "RNA_intensity": _jsonable(RNA_intensity_fit),
    "offset": _jsonable(offset_fit),
}
arrays_path, summary_path = write_fit_outputs(
    dataset_label=dataset_label,
    fit_state=fit_state,
    fit_state_path=fit_state_path,
    best_fit=best_fit,
    best_x=best_x,
    pred=pred,
    acf_after_filtering=acf_after_filtering,
    covars=covars,
    conditions=conditions,
    parameters=fit_parameters,
)
print(f"Saved best fit arrays: {arrays_path}")
print(f"Saved best fit summary: {summary_path}")

fig, ax = plt.subplots(2, 4, figsize=(12, 8))
for index, condition in enumerate(conditions):
    axis = ax.flatten()[index]

    axis.plot(lags_expanded[index], acf_after_filtering[index][1:], label=f"Data")
    axis.plot(lags_expanded[index], pred[index][1:], label=f"Fit", linestyle="--")
    axis.fill_between(
        lags_expanded[index],
        acf_after_filtering[index][1:] - jnp.sqrt(covars[index])[1:],
        acf_after_filtering[index][1:] + jnp.sqrt(covars[index])[1:],
        color="gray",
        alpha=0.3,
        label="Bootstrap 1σ",
    )
    axis.set(title=condition, xlabel="Lag [Minutes]", ylabel="ACF", yscale="log")
    axis.legend()
fig.suptitle(f"{dataset_label} data and fit")
fig.tight_layout()
acf_plot_path = RESULT_DIR / f"{dataset_label}_ACF_fits.png"
fig.savefig(acf_plot_path)
plt.close(fig)

means = [a[0] for a in acf_after_filtering]
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
means = [a[0] for a in acf_after_filtering]
pred_means = [a[0] for a in pred]
ax.plot(conditions, means, label="Data", marker="o")
ax.plot(
    conditions,
    pred_means,
    label="Fit",
    marker="s",
)

ax.set(ylabel="Mean intensity [AU]", xlabel="Condition")
ax.legend()
fig.tight_layout()
mean_plot_path = RESULT_DIR / f"{dataset_label}_mean_fit.png"
fig.savefig(mean_plot_path)
plt.close(fig)
print(f"Saved plots: {acf_plot_path}, {mean_plot_path}")
