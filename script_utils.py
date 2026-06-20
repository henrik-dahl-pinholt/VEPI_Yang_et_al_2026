from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def _json_dumps(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _array_digest(arr) -> str:
    arr_np = np.ascontiguousarray(np.asarray(arr))
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(arr_np.shape).encode())
    digest.update(arr_np.dtype.str.encode())
    digest.update(arr_np.view(np.uint8))
    return digest.hexdigest()


def _save_npz_atomic(path: Path, **arrays):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp_path.replace(path)


def file_ok(path):
    return path.exists() and path.stat().st_size > 0


def time_interval_seconds(time_interval):
    return float(str(time_interval).removesuffix("s"))


def time_interval_label(time_interval):
    return f"{time_interval_seconds(time_interval):g}s"


def filter_short_tracks(ms2_data, ep_dat, minlength):
    """Filter out tracks that are shorter than `minlength` time points."""
    track_lengths = np.sum(~np.isnan(ms2_data), axis=1)
    keep_mask = track_lengths >= minlength
    return ms2_data[keep_mask], ep_dat[keep_mask]


def reassign_tracks_by_duration(dataset1, dataset2):
    """Pair up tracks from dataset1 and dataset2 by choosing the track with the closest duration that isn't itself or already taken"""
    durations = np.sum(~np.isnan(dataset1), axis=1)
    inds = np.arange(dataset1.shape[0])
    taken_inds = set()
    permuted_inds = []
    for i in range(dataset1.shape[0]):
        dur = durations[i]
        possible_inds = inds[~np.isin(inds, list(taken_inds)) & (inds != i)]
        if len(possible_inds) == 0:
            print(
                "Warning: ran out of possible tracks to pair, taking a random one from the whole set that isn't itself"
            )
            # if we run out of possible inds, just take a random one from the whole set that isn't itself
            possible_inds = inds[inds != i]
        closest_ind = possible_inds[np.argmin(np.abs(durations[possible_inds] - dur))]
        permuted_inds.append(closest_ind)
        taken_inds.add(closest_ind)
    permuted_dataset2 = dataset2[permuted_inds]
    return dataset1, permuted_dataset2


def _lookup_param(params, candidate_keys, default=None):
    for key in candidate_keys:
        if key in params:
            value = params[key]
            if np.isfinite(value):
                return float(value)
    if default is not None:
        return default
    raise KeyError(f"None of these parameter keys were found: {candidate_keys}")


def smooth_lag_curve(curve, sigma_frames=2.0):
    if sigma_frames <= 0:
        return np.asarray(curve)
    radius = int(np.ceil(4 * sigma_frames))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma_frames) ** 2)
    kernel = kernel / kernel.sum()
    curve = np.asarray(curve, dtype=float)
    valid = np.isfinite(curve)
    numerator = np.convolve(np.where(valid, curve, 0.0), kernel, mode="same")
    denominator = np.convolve(valid.astype(float), kernel, mode="same")
    return numerator / np.maximum(denominator, 1e-12)


def predict_w0_pon_crosscorr(
    lags_min,
    contact_autocorr,
    pon_autocorr,
    dt_min,
    max_delay_min=60.0,
    response_lag_min=0.0,
):
    """Predict contact-Pol2 shape for W=0 contact and a pon-autocorr response."""
    delays = np.arange(0.0, max_delay_min + 0.5 * dt_min, dt_min)
    kernel = np.interp(delays, lags_min, pon_autocorr, left=np.nan, right=np.nan)
    kernel = np.clip(np.nan_to_num(kernel, nan=0.0), 0.0, np.inf)
    if not np.any(kernel > 0):
        kernel = np.zeros_like(delays)
        kernel[0] = 1.0
    kernel = kernel / np.sum(kernel)

    pred = np.zeros_like(lags_min, dtype=float)
    weight = np.zeros_like(lags_min, dtype=float)
    for delay, delay_weight in zip(delays, kernel):
        shifted = np.interp(
            lags_min - (delay + response_lag_min),
            lags_min,
            contact_autocorr,
            left=np.nan,
            right=np.nan,
        )
        valid = np.isfinite(shifted)
        pred[valid] += delay_weight * shifted[valid]
        weight[valid] += delay_weight
    return np.divide(pred, weight, out=np.full_like(pred, np.nan), where=weight > 0)


def fit_amplitude_only(observed, predicted, fit_mask):
    valid = fit_mask & np.isfinite(observed) & np.isfinite(predicted)
    if not np.any(valid):
        return np.nan
    denom = np.sum(predicted[valid] ** 2)
    if denom <= 0:
        return np.nan
    return float(np.sum(observed[valid] * predicted[valid]) / denom)
