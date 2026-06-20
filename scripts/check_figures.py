from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECK_ROOT = ROOT / ".repro_check"
DEFAULT_MIN_BYTES = 1024
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass


@dataclass(frozen=True)
class Step:
    name: str
    command: tuple[str, ...]
    outputs: tuple[str, ...]
    is_notebook: bool = False


def python_script(script_name: str) -> tuple[str, ...]:
    return (sys.executable, script_name)


def notebook_command(notebook_name: str) -> tuple[str, ...]:
    output_script = CHECK_ROOT / "executed_notebooks" / f"{Path(notebook_name).stem}.py"
    return (
        sys.executable,
        "scripts/run_notebook_as_script.py",
        notebook_name,
        "--output-script",
        str(output_script),
    )


def build_manifest() -> list[Step]:
    return [
        Step(
            "aggregate-ms2-validation-cache",
            python_script("12_aggregate_MS2test_results.py"),
            (),
        ),
        Step(
            "ms2-correlation-fit-supplement",
            python_script("1_1_plot_MS2_corrfit.py"),
            (
                "figures/Supplement_figures/MS2_corrfit/raw_intensity_acf_fits.pdf",
                "figures/Supplement_figures/MS2_corrfit/raw_intensity_mean_fit.pdf",
                "figures/Supplement_figures/MS2_corrfit/raw_intensity_corrfit_plots.pdf",
            ),
        ),
        Step(
            "posterior-example",
            python_script("10_posterior_example.py"),
            ("figures/main_figure/example.pdf",),
        ),
        Step(
            "simulation-rc-kon",
            python_script("4_2_compute_rc_kon_map_simulation.py"),
            (
                "figures/main_figure/rc_kon_simdat.pdf",
                "figures/main_figure/rc_kon_simdat_ridge.pdf",
            ),
        ),
        Step(
            "data-rc-kon",
            python_script("3_compute_rc_kon_map.py"),
            (
                "figures/main_figure/rc_kon_data.pdf",
                "figures/main_figure/rc_kon_data_ridge.pdf",
            ),
        ),
        Step(
            "cross-correlation",
            python_script("6_cross_correlation.py"),
            (
                "figures/main_figure/mean_predictions.pdf",
                "figures/main_figure/cross_correlation_prediction.pdf",
            ),
        ),
        Step(
            "burst-pileup",
            python_script("8_burst_pileup.py"),
            (
                "figures/main_figure/pileup.pdf",
                "figures/main_figure/dip_hist.pdf",
                "figures/main_figure/pileup_5s.pdf",
                "figures/main_figure/dip_hist_5s.pdf",
            ),
        ),
        Step(
            "posterior-correlation",
            python_script("9_posterior_corr.py"),
            ("figures/main_figure/contact_pol2_cross_corr.pdf",),
        ),
        Step(
            "tau-estimate",
            python_script("9_1_posterior_corr_tau_est.py"),
            ("figures/main_figure/tau_est.pdf",),
        ),
        Step(
            "ms2-validation-supplement",
            notebook_command("13_analyze_aggregate_MS2test.ipynb"),
            (
                "figures/Supplement_figures/MS2_validation/Rate_inference.pdf",
                "figures/Supplement_figures/MS2_validation/Promoter_state_reconstruction.pdf",
                "figures/Supplement_figures/MS2_validation/MS2_signal_prediction.pdf",
                "figures/Supplement_figures/MS2_validation/Prediction_example.pdf",
                "figures/Supplement_figures/MS2_validation/Example_trajectories.pdf",
            ),
            is_notebook=True,
        ),
        Step(
            "inference-no-drive-supplement",
            notebook_command("14_plot_inference_results_noRCDrive.ipynb"),
            (
                "figures/Supplement_figures/Inference_no_drive/burst_off_on.pdf",
                "figures/Supplement_figures/Inference_no_drive/Observed_vs_Predicted_MS2.pdf",
                "figures/Supplement_figures/Inference_no_drive/Rates.pdf",
                "figures/Supplement_figures/Inference_no_drive/pon_maps.pdf",
                "figures/Supplement_figures/Inference_no_drive/fraction_on_vs_observed_MS2.pdf",
            ),
            is_notebook=True,
        ),
        Step(
            "inference-rc-supplement",
            notebook_command("15_plot_inference_results_RC.ipynb"),
            ("figures/Supplement_figures/Inference_RC/ms2_histograms.pdf",),
            is_notebook=True,
        ),
    ]


def selected_steps(args: argparse.Namespace) -> list[Step]:
    steps = build_manifest()
    if args.skip_notebooks:
        steps = [step for step in steps if not step.is_notebook]
    if args.only:
        requested = set(args.only)
        steps = [step for step in steps if step.name in requested]
        missing = requested - {step.name for step in steps}
        if missing:
            raise SystemExit(f"Unknown step name(s): {', '.join(sorted(missing))}")
    return steps


def expected_outputs(steps: list[Step]) -> list[Path]:
    seen: set[Path] = set()
    outputs: list[Path] = []
    for step in steps:
        for rel in step.outputs:
            path = ROOT / rel
            if path not in seen:
                outputs.append(path)
                seen.add(path)
    return outputs


def assert_manifest_covers_existing_figures(outputs: list[Path], allow_extra: bool) -> None:
    if allow_extra:
        return
    figure_root = ROOT / "figures"
    if not figure_root.exists():
        return
    expected = {path.relative_to(ROOT) for path in outputs}
    existing = {path.relative_to(ROOT) for path in figure_root.rglob("*.pdf")}
    extra = sorted(existing - expected)
    if extra:
        formatted = "\n".join(f"  {path}" for path in extra)
        raise SystemExit(
            "The manifest does not claim every existing figure PDF:\n"
            f"{formatted}\n"
            "Add those PDFs to the manifest or rerun with --allow-extra-figures."
        )


def backup_outputs(outputs: list[Path], backup_root: Path) -> list[tuple[Path, Path]]:
    moved: list[tuple[Path, Path]] = []
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            continue
        rel = path.relative_to(ROOT)
        backup_path = backup_root / rel
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        path.replace(backup_path)
        moved.append((backup_path, path))
    return moved


def restore_backups(
    moved: list[tuple[Path, Path]],
    displaced_root: Path,
) -> None:
    for backup_path, original_path in moved:
        if not backup_path.exists():
            continue
        if original_path.exists():
            displaced_path = displaced_root / original_path.relative_to(ROOT)
            displaced_path.parent.mkdir(parents=True, exist_ok=True)
            original_path.replace(displaced_path)
        original_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.replace(original_path)


def validate_outputs(outputs: list[Path], min_bytes: int) -> list[str]:
    problems: list[str] = []
    for path in outputs:
        rel = path.relative_to(ROOT)
        if not path.exists():
            problems.append(f"{rel}: missing")
            continue
        size = path.stat().st_size
        if size < min_bytes:
            problems.append(f"{rel}: too small ({size} bytes)")
            continue
        with path.open("rb") as handle:
            header = handle.read(5)
        if header != b"%PDF-":
            problems.append(f"{rel}: does not start with a PDF header")
    return problems


def run_step(step: Step, env: dict[str, str], log_dir: Path) -> None:
    log_path = log_dir / f"{step.name}.log"
    print(f"\n==> {step.name}")
    print("    " + " ".join(step.command))
    completed = subprocess.run(
        step.command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(completed.stdout)
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-80:])
        raise RuntimeError(
            f"{step.name} failed with exit code {completed.returncode}. "
            f"Log: {log_path}\n{tail}"
        )
    print(f"    ok, log: {log_path.relative_to(ROOT)}")


def print_manifest(steps: list[Step]) -> None:
    for step in steps:
        print(step.name)
        print("  command:", " ".join(step.command))
        for output in step.outputs:
            print("  output:", output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the paper figure PDFs from local cached data and verify "
            "that each expected output was recreated."
        )
    )
    parser.add_argument("--list", action="store_true", help="Print the manifest and exit.")
    parser.add_argument(
        "--only",
        action="append",
        help="Run only the named manifest step. Can be repeated.",
    )
    parser.add_argument(
        "--skip-notebooks",
        action="store_true",
        help="Skip notebook-owned supplement figures.",
    )
    parser.add_argument(
        "--no-isolate",
        action="store_true",
        help="Do not move existing target PDFs aside before running.",
    )
    parser.add_argument(
        "--restore-originals",
        action="store_true",
        help="After a successful isolated run, restore the original backed-up PDFs.",
    )
    parser.add_argument(
        "--allow-extra-figures",
        action="store_true",
        help="Do not fail if figures/ contains PDFs not listed in the manifest.",
    )
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=DEFAULT_MIN_BYTES,
        help=f"Minimum acceptable PDF size in bytes. Default: {DEFAULT_MIN_BYTES}.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    steps = selected_steps(args)
    outputs = expected_outputs(steps)

    if args.list:
        print_manifest(steps)
        return 0

    assert_manifest_covers_existing_figures(outputs, args.allow_extra_figures or bool(args.only))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = CHECK_ROOT / f"run_{timestamp}"
    log_dir = run_root / "logs"
    backup_root = run_root / "figure_backup"
    displaced_root = run_root / "displaced_failed_outputs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (CHECK_ROOT / "executed_notebooks").mkdir(parents=True, exist_ok=True)

    moved: list[tuple[Path, Path]] = []
    if not args.no_isolate:
        moved = backup_outputs(outputs, backup_root)
        print(f"Moved {len(moved)} existing target PDF(s) to {backup_root.relative_to(ROOT)}.")

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
    env.setdefault("PYTHONUNBUFFERED", "1")

    try:
        for step in steps:
            run_step(step, env, log_dir)

        problems = validate_outputs(outputs, args.min_bytes)
        if problems:
            raise RuntimeError("Output validation failed:\n" + "\n".join(problems))

        if args.restore_originals and moved:
            restore_backups(moved, run_root / "displaced_success_outputs")
            print("Restored original PDFs after successful validation.")

    except BaseException:
        if moved and not args.no_isolate:
            restore_backups(moved, displaced_root)
            print(f"Restored original PDFs. Partial outputs moved to {displaced_root.relative_to(ROOT)}.")
        raise

    print(f"\nValidated {len(outputs)} PDF output(s).")
    print(f"Logs: {log_dir.relative_to(ROOT)}")
    if moved and not args.restore_originals:
        print(f"Original PDF backup: {backup_root.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
