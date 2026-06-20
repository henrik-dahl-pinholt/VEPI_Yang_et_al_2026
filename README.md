# Variational Enhancer-Promoter Inference (VEPI) reproducibility repo
This is a repository for reproducing the inference and analysis relating to Variational Enhancer-Promoter Inference (VEPI) in the manuscript "Live-cell imaging of enhancer-promoter dynamics
reveals transient contact-driven gene activation".

The repository uses cached computations for intermediate results so a user will not have to re-run all heavy computations from scratch. The large cache, data, and result files are distributed as GitHub release assets rather than normal Git files. Rerunning these computations is possible, and most scripts carry an overwrite option to allow for re-computation if needed. The scripts are put more or less in cronological order for how the analysis/inference was run and validated.

## Reproducing figures from a fresh clone

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate vepi-yang-2026
python --version
```

This repository was prepared and tested with Python 3.12.2. The environment file uses a minimal set of packages needed by the scripts and notebooks that regenerate the figures. It installs CPU JAX by default, which is sufficient for reproducing figures from the cached release assets. GPU JAX was used for the inference and can be installed separately if you plan to rerun heavy inference rather than only regenerate plots.

After cloning the repository, you still need the large-size cached computations if you do not intend to resimulate/infer everything from scratch.
These have been stored as release assets, allowing you to download and regenerate all figure plots using:

```bash
python scripts/download_repro_assets.py
python scripts/check_figures.py
```

The downloader reads `repro_assets_manifest.json`, downloads the release archives, verifies their SHA256 hashes, and extracts them into the local `Data`, `cache`, and `result` directories. `scripts/check_figures.py` then regenerates the figure PDFs from those local files and verifies that the expected outputs exist. Note that this will extract ~15Gb onto your disk, so make sure you are ready for that. 

The release tag used by the manifest is `repro-assets-v1`. The generated archives are staged in `release_assets/repro-assets-v1`, which is intentionally ignored by Git.

## Folder structure
The folders in the repository contain the following:
- cache, Data, result
    - Stores intermediate or semi-final computations to reduce user runtime when reproducing plots/analysis. These folders are populated by `scripts/download_repro_assets.py`.
- figures
    - Stores the generated figures.
- scripts
    - Stores small functions and utilities for the analysis/simulation
- MS2Posterior
    - VEPI module for doing Variational Inference on MS2 data.
- TwoLocusGPR
    - VEPI module for doing Gaussian Process Regression on two-locus tracking data.

## Note on plots
You may find that plots do not look as formatted in the paper. This is because the pdfs were formatted in Affinity Designer before being included in the figure. 
We stress that the information in the output figures here is equal to that in the paper. 

## Correlation function fitting
### Fig. S21
- 1_MS2_corrfit.py
    - Performs the MS2 correlation fitting
- 1_1_plot_MS2_corrfit.py
    - Defines the plotting functions for the MS2 correlation fitting
- 1_1_plot_MS2_corrfit.ipynb
    - Jupyter notebook for running and showing the MS2 correlation fitting results

## Base inference without E-P drive 
### 2_run_inference.py
- Runs the VEPI inference algorithm without E-P drive to provide per-track parameters and results for burst-call analysis

## Inference with E-P drive and validation
### 4_generate_test_data.py
- Generate simulation data of a Rouse model driving a two-state promoter to study identifiability of $k_\mathrm{on}$ and $R_\mathrm{E-P}$.
### 4_1_run_inference_test_data.py
- Runs the VEPI E-P driven inference algorithm on the generated test data.

### Fig. S20, Fig. 4J, and Fig. 4K
- 4_2_compute_rc_kon_map_simulation.py
    - Code for generating plot for S20A and Fig. 4J
- 3_compute_rc_kon_map.py
    - Code for generating plot for S20B and Fig. 4K

## Predictions from calibrated model
### 5_simulate_from_calibrated_params.py
- Runs E-P distance and MS2 end-to-end simulations based on the calibrated model obtained from the correlation function fits, MSD parameters (stored in the repo), and inference results.

### E-P driven inference on experimental data
### 7_run_ep_driven_inf.py
- Runs the VEPI E-P driven inference algorithm on the experimental data.

### Fig. 4
- 6_cross_correlation.py
    - Code for Fig. 4L
- 8_burst_pileup.py
    - Code for Fig. 4N, Fig. 4O, and Fig. 4P
- 9_posterior_corr.py
    - Code for Fig. 4M
- 9_1_posterior_corr_tau_est.py
    - Code for Fig. 4Q
- 10_posterior_example.py
    - Code for Fig. 4F-4I

## MS2 inference validation
### 11_generate_test_data.py
- Generate test data for validating the MS2 inference algorithm. (no E-P distance, just MS2 simulations)
### 12_aggregate_MS2_test_results.py
- Compute summary statistics for the MS2 inference validation results.

### Fig. S18 and S19
- 13_analyze_aggregate_MS2test.ipynb
    - Generates all plots in Fig. S18 and S19

### Fig. S22 and S23
- 14_plot_inference_results_noRCDrive.ipynb
    - Generates all plots in Fig. S22 and S23

### Fig. S24
- 15_plot_inference_results_RC.ipynb
    - Generates all plots in Fig. S24

Any other scripts are just utility scripts for various tasks.
