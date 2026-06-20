from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT_DIR = SCRIPT_ROOT / "result" / "1_MS2_corrfit"
DEFAULT_OUTPUT_DIR = SCRIPT_ROOT / "figures" / "Supplement_figures" / "MS2_corrfit"
DEFAULT_DT_MINUTES = 0.5

DATA_COLOR = "#0077BB"
FIT_COLOR = "#EE7733"
BAND_COLOR = "0.55"

DISPLAY_LABELS = {
    "1.5kb": "1.5 kb",
    "85kb": "85 kb",
    "170kb": "170 kb",
    "255kb": "255 kb",
    "340kb": "340 kb",
    "340kb_Ce": "340 kb Ce",
    "340kb_Cp": "340 kb Cp",
    "340kb_Ce_Cp": "340 kb Ce+Cp",
}


@dataclass(frozen=True)
class FitBundle:
    dataset_label: str
    conditions: np.ndarray
    prediction: np.ndarray
    acf: np.ndarray
    variance: np.ndarray
    summary_path: Path
    arrays_path: Path
    summary: dict


def find_latest_summary(result_dir: Path) -> Path:
    summaries = sorted(result_dir.glob("fit_*_summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No fit summary JSON files found in {result_dir}")
    return max(summaries, key=lambda path: path.stat().st_mtime)


def resolve_arrays_path(summary_path: Path, summary: dict) -> Path:
    candidates = []
    if "best_arrays" in summary:
        candidates.append(Path(summary["best_arrays"]))
    if summary_path.name.endswith("_summary.json"):
        candidates.append(
            summary_path.with_name(summary_path.name.replace("_summary.json", "_best.npz"))
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"No best-fit array bundle found. Tried:\n{tried}")


def load_fit_bundle(summary_path: Path | None = None, result_dir: Path = DEFAULT_RESULT_DIR) -> FitBundle:
    summary_path = find_latest_summary(result_dir) if summary_path is None else summary_path
    with summary_path.open() as handle:
        summary = json.load(handle)

    arrays_path = resolve_arrays_path(summary_path, summary)
    with np.load(arrays_path, allow_pickle=False) as arrays:
        conditions = arrays["conditions"].astype(str)
        prediction = np.asarray(arrays["best_prediction"], dtype=float)
        acf = np.asarray(arrays["acf_after_filtering"], dtype=float)
        variance = np.asarray(arrays["covariance_diagonal"], dtype=float)

    expected_shape = prediction.shape
    if acf.shape != expected_shape or variance.shape != expected_shape:
        raise ValueError(
            "Fit arrays have inconsistent shapes: "
            f"prediction={prediction.shape}, acf={acf.shape}, variance={variance.shape}"
        )
    if len(conditions) != prediction.shape[0]:
        raise ValueError(
            f"Found {len(conditions)} conditions but {prediction.shape[0]} prediction rows"
        )

    return FitBundle(
        dataset_label=str(summary.get("dataset_label", "ms2")),
        conditions=conditions,
        prediction=prediction,
        acf=acf,
        variance=variance,
        summary_path=summary_path,
        arrays_path=arrays_path,
        summary=summary,
    )


def condition_labels(conditions: np.ndarray) -> list[str]:
    return [DISPLAY_LABELS.get(str(condition), str(condition)) for condition in conditions]


def positive_floor(*arrays: np.ndarray) -> float:
    positives = []
    for arr in arrays:
        finite_positive = np.asarray(arr)[np.isfinite(arr) & (np.asarray(arr) > 0)]
        if finite_positive.size:
            positives.append(np.nanmin(finite_positive))
    if not positives:
        return 1e-6
    return max(float(np.nanmin(positives)) * 0.25, 1e-9)


def plot_acf_fits(bundle: FitBundle, *, dt_minutes: float = DEFAULT_DT_MINUTES, log_y: bool = True):
    lag_minutes = np.arange(bundle.acf.shape[1] - 1) * dt_minutes
    labels = condition_labels(bundle.conditions)
    stderr = np.sqrt(np.clip(bundle.variance[:, 1:], 0.0, np.inf))

    fig, axes = plt.subplots(2, 4, figsize=(11.2, 5.8), sharex=True)
    axes_flat = axes.ravel()

    for index, (axis, label) in enumerate(zip(axes_flat, labels)):
        data = bundle.acf[index, 1:]
        fit = bundle.prediction[index, 1:]
        err = stderr[index]
        lower = data - err
        upper = data + err

        if log_y:
            floor = positive_floor(data, fit, upper)
            lower = np.maximum(lower, floor)
            upper = np.maximum(upper, floor)
            axis.set_yscale("log")
        else:
            floor = None

        axis.fill_between(
            lag_minutes,
            lower,
            upper,
            color=BAND_COLOR,
            alpha=0.22,
            linewidth=0,
            label="Bootstrap 1-sigma",
        )
        axis.plot(lag_minutes, data, color=DATA_COLOR, linewidth=1.8, label="Data")
        axis.plot(
            lag_minutes,
            fit,
            color=FIT_COLOR,
            linewidth=1.8,
            linestyle="--",
            label="Fit",
        )
        axis.set_title(label, fontsize=10)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=8)
        if floor is not None:
            axis.set_ylim(bottom=floor)
        if index == 0:
            axis.legend(frameon=False, fontsize=8, loc="upper right")

    for axis in axes_flat[len(labels) :]:
        axis.axis("off")

    fig.supxlabel("Lag (min)", fontsize=11)
    fig.supylabel("MS2 autocovariance [a.u.^2]", fontsize=11)
    yscale_label = "log scale" if log_y else "linear scale"
    fig.suptitle(f"{bundle.dataset_label} MS2 autocovariance fit ({yscale_label})", fontsize=12)
    fig.tight_layout()
    return fig


def plot_means(bundle: FitBundle):
    labels = condition_labels(bundle.conditions)
    x = np.arange(len(labels))
    data_means = bundle.acf[:, 0]
    fit_means = bundle.prediction[:, 0]
    data_stderr = np.sqrt(np.clip(bundle.variance[:, 0], 0.0, np.inf))

    fig, axis = plt.subplots(1, 1, figsize=(6.3, 3.5))
    axis.errorbar(
        x,
        data_means,
        yerr=data_stderr,
        fmt="o",
        color=DATA_COLOR,
        ecolor=DATA_COLOR,
        elinewidth=1.2,
        capsize=3,
        markersize=5,
        label="Data",
    )
    axis.plot(
        x,
        fit_means,
        marker="s",
        linestyle="--",
        color=FIT_COLOR,
        linewidth=1.6,
        markersize=5,
        label="Fit",
    )
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=-35, ha="left")
    axis.set_ylabel("Mean MS2 intensity [a.u.]")
    axis.set_xlabel("Condition")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    fig.tight_layout()
    return fig


def save_figures(
    bundle: FitBundle,
    output_dir: Path,
    *,
    dt_minutes: float = DEFAULT_DT_MINUTES,
    log_y: bool = True,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = bundle.dataset_label

    acf_fig = plot_acf_fits(bundle, dt_minutes=dt_minutes, log_y=log_y)
    mean_fig = plot_means(bundle)

    acf_path = output_dir / f"{prefix}_acf_fits.pdf"
    mean_path = output_dir / f"{prefix}_mean_fit.pdf"
    multipage_path = output_dir / f"{prefix}_corrfit_plots.pdf"

    acf_fig.savefig(acf_path)
    mean_fig.savefig(mean_path)
    with PdfPages(multipage_path) as pdf:
        pdf.savefig(acf_fig)
        pdf.savefig(mean_fig)

    plt.close(acf_fig)
    plt.close(mean_fig)
    return [acf_path, mean_path, multipage_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot fitted MS2 autocovariance functions and means as PDF files."
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Directory containing fit_*_summary.json and fit_*_best.npz.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Specific fit summary JSON to plot. Defaults to the newest summary in --result-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PDF figures are written.",
    )
    parser.add_argument(
        "--dt-minutes",
        type=float,
        default=DEFAULT_DT_MINUTES,
        help="Lag spacing in minutes. The 30 s MS2 fit uses 0.5.",
    )
    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="Plot autocovariance panels on a linear y-axis instead of the default log axis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_fit_bundle(args.summary, args.result_dir)
    paths = save_figures(
        bundle,
        args.output_dir,
        dt_minutes=args.dt_minutes,
        log_y=not args.linear_y,
    )
    print(f"Loaded summary: {bundle.summary_path}")
    print(f"Loaded arrays: {bundle.arrays_path}")
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
