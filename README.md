# Time-aware entity-centric risk modeling for fraud detection

This repository contains the source code and configuration required to generate the tables and
figures listed in the script-to-result map below:

**Time-aware entity-centric risk modeling for fraud detection under temporal drift and cold-start conditions**

The repository covers the preprocessing, causal entity feature construction,
model training, fixed-FPR evaluation, ablation analysis, sensitivity analysis, runtime analysis,
bootstrap confidence intervals, figure generation, and the supplementary PaySim portability check
reported in Tables 5-13 and Figures 1-16.

## Repository availability

The public repository is available at:

```text
https://github.com/yqfan2026/fraud-causal-risk-framework
```

If a Zenodo archive is created for the released version, its DOI will be listed here and cited
together with the repository in the data and code availability statement.

## Repository structure

```text
fraud-causal-risk-framework/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── configs/
│   ├── default_ieee_cis.yaml
│   └── random_seed.txt
├── results/
│   └── .gitkeep
├── scripts/
│   ├── 01_preprocess.py
│   ├── 02_build_causal_features.py
│   ├── 03_train_lgbm.py
│   ├── 04_make_entity_groups.py
│   ├── 05_make_ablations.py
│   ├── run_temporal_new.py
│   ├── compute_bootstrap_new.py
│   ├── make_baseline_dir.py
│   ├── make_figures_new.py
│   ├── paysim_experiment.py
│   └── probe_features.py
```

## Data

The main experiments use the IEEE-CIS Fraud Detection dataset. A supplementary portability check
uses the PaySim dataset under a weak feature setting. Raw datasets are not included in this
repository; the expected directory layout is given below, and the processed feature matrices are
available from the DOI listed in the Data Availability statement.

The raw IEEE-CIS Fraud Detection dataset is publicly available from Kaggle:

https://www.kaggle.com/c/ieee-fraud-detection

After downloading the dataset, place the files under `data/ieee/`:

```text
data/ieee/train_transaction.csv
data/ieee/test_transaction.csv
data/ieee/train_identity.csv
data/ieee/test_identity.csv
```

The raw data files are not included in this repository because they are distributed by a
third-party repository and may be large. Users should download the raw data from the original
sources and comply with the corresponding terms of use.

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

Python 3.10 or newer is recommended. The main experiments used random seed `42`.

## Script-to-result map

| **Script** | **Generated outputs** |
| :--- | :--- |
| `scripts/01_preprocess.py` | Table 2 (dataset statistics); writes the processed train/test matrices consumed by every later script |
| `scripts/02_build_causal_features.py` | Table 3 (entity definitions); adds the causal entity-risk features used in Tables 5-11 and Figures 1-16 |
| `scripts/04_make_entity_groups.py` | Entity groups that define the cold-start protocol behind Tables 5-10 |
| `scripts/03_train_lgbm.py` | Table 5 (main results), Table 8 (fixed-FPR budgets), Table 9 (precision/recall/F1), Table 10 (cost), Table 11 (hyperparameter sweep, one run per setting) |
| `scripts/run_temporal_new.py` | Chronological temporal holdout rows of Table 5; Figures 11-14 |
| `scripts/compute_bootstrap_new.py` | Table 6 (bootstrap confidence intervals); Figure 2 |
| `scripts/05_make_ablations.py` | Table 7 (ablation study); Figure 3 |
| `scripts/make_baseline_dir.py` | Baseline run directory required by Table 5 and Figures 11-14 |
| `scripts/make_figures_new.py` | Figures 1-16, read from the outputs of the runs above |
| `scripts/paysim_experiment.py` | Table 13; Figure 10; PaySim row of Figure 12 |
| `scripts/probe_features.py` | Diagnostic probe of the PaySim feature set; produces no table or figure |

## Execution commands

All commands are run from the repository root and use relative paths only. Steps 9 and 10 read a
fixed layout, so the output directories below should be used as written.

### 1. Preprocess the raw IEEE-CIS data

```bash
python scripts/01_preprocess.py --data_dir ./data/ieee --out_dir ./outputs/ieee_proc
```

This writes `train_features.pkl`, `test_features.pkl`, `y.npy`, `train_txid.npy`,
`test_txid.npy`, and `meta.json` into `outputs/ieee_proc/`.

### 2. Build causal entity features

```bash
python scripts/02_build_causal_features.py \
  --proc_dir ./outputs/ieee_proc \
  --out_dir ./processed_features_full_h7 \
  --half_life_days 7 \
  --smooth_strength 10
```

### 3. Entity groups for the group cold-start protocol

```bash
python scripts/04_make_entity_groups.py --proc_dir ./processed_features_full_h7
```

This creates `processed_features_full_h7/entity.npy`.

### 4. Weak-feature baseline matrix

```bash
python scripts/make_baseline_dir.py \
  --src ./processed_features_full_h7 \
  --dst ./outputs/ieee_proc_baseline
```

### 5. Train and evaluate the full causal framework

```bash
python scripts/03_train_lgbm.py \
  --proc_dir ./processed_features_full_h7 \
  --out_dir ./runs_new_entities/full \
  --cv_mode group \
  --n_splits 5 \
  --seed 42 \
  --calibration platt
```

### 6. Train and evaluate the baseline

```bash
python scripts/03_train_lgbm.py \
  --proc_dir ./outputs/ieee_proc_baseline \
  --out_dir ./runs_new_entities/baseline \
  --cv_mode group \
  --n_splits 5 \
  --seed 42 \
  --calibration platt
```

### 7. Chronological temporal holdout

```bash
python scripts/run_temporal_new.py \
  --proc_dir ./processed_features_full_h7 \
  --out_dir ./runs_new_entities/temporal_full \
  --train_frac 0.8 --seed 42

python scripts/run_temporal_new.py \
  --proc_dir ./outputs/ieee_proc_baseline \
  --out_dir ./runs_new_entities/temporal_baseline \
  --train_frac 0.8 --seed 42
```

### 8. Ablation variants

```bash
python scripts/05_make_ablations.py \
  --src ./processed_features_full_h7 \
  --out_root ./outputs/ablations
```

Then train each variant into the directory names used by the figure script:

```bash
python scripts/03_train_lgbm.py --proc_dir ./outputs/ablations/ieee_ab_causal_er_rate \
  --out_dir ./runs_new_entities/ab_er_rate --cv_mode group --n_splits 5 --seed 42 --calibration platt

python scripts/03_train_lgbm.py --proc_dir ./outputs/ablations/ieee_ab_causal_er_time \
  --out_dir ./runs_new_entities/ab_er_time --cv_mode group --n_splits 5 --seed 42 --calibration platt

python scripts/03_train_lgbm.py --proc_dir ./outputs/ablations/ieee_ab_causal_full_nohier \
  --out_dir ./runs_new_entities/ab_full_nohier --cv_mode group --n_splits 5 --seed 42 --calibration platt
```

### 9. Bootstrap confidence intervals

```bash
FCRF_ROOT=. python scripts/compute_bootstrap_new.py
```

On Windows PowerShell:

```powershell
$env:FCRF_ROOT = "."; python scripts/compute_bootstrap_new.py
```

This reads `processed_features_full_h7/y.npy` and
`runs_new_entities/{baseline,full}/oof_cal.npy`, and writes
`outputs/figures/bootstrap_ci_new.json`.

### 10. Figure generation

```bash
export FCRF_RESULTS=./results
python scripts/make_figures_new.py --root . --figs 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
```

On Windows PowerShell:

```powershell
$env:FCRF_RESULTS = "./results"; python scripts/make_figures_new.py --root . --figs 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
```

`FCRF_RESULTS` points to the directory that holds `metrics_paysim.json`,
`runtime_table_for_paper.csv`, and `hyperparameter_sensitivity.csv`; it defaults to `./results`
and only needs to be set when those files live elsewhere. Figures 8-12 read these files directly,
so re-running the pipeline and updating them is sufficient to redraw the figures.

Note: The `results/` directory will be automatically created if it does not exist. Please ensure
the training scripts have been run first to generate the required JSON/CSV files.

Figures are written as PNG, PDF, and TIFF at 300 dpi into `outputs/figures/`. The script reads
`runs_new_entities/` and, for Figure 2, `outputs/figures/bootstrap_ci_new.json` from step 9.

### 11. PaySim supplementary experiment

```bash
python scripts/paysim_experiment.py \
  --csv ./data/paysim/PS_20174392719_1491204439457_log.csv \
  --out_dir ./outputs/paysim_experiment
```

### Hyperparameter sweep (Table 11)

Repeat steps 2 and 5 for each setting, using a distinct pair of directory names:

| Half-life / smoothing | `--out_dir` for step 2 | `--out_dir` for step 5 |
| :--- | :--- | :--- |
| 14 / 10 | `./processed_features_h14_s10` | `./runs_new_entities/hp_h14_s10` |
| 30 / 10 | `./processed_features_h30_s10` | `./runs_new_entities/hp_h30_s10` |
| 14 / 30 | `./processed_features_h14_s30` | `./runs_new_entities/hp_h14_s30` |
| 14 / 50 | `./processed_features_h14_s50` | `./runs_new_entities/hp_h14_s50` |

The 7 / 10 row is the main run from steps 2 and 5.

### Layout expected by steps 9 and 10

```text
./processed_features_full_h7/y.npy
./runs_new_entities/baseline/oof_cal.npy
./runs_new_entities/full/oof_cal.npy
./runs_new_entities/temporal_full/metrics_temporal.json
./runs_new_entities/temporal_baseline/metrics_temporal.json
./runs_new_entities/ab_er_rate/metrics_cv.json
./runs_new_entities/ab_er_time/metrics_cv.json
./runs_new_entities/ab_full_nohier/metrics_cv.json
./outputs/figures/bootstrap_ci_new.json
```

## Result tables

Running the scripts writes all result tables and metric files into `results/`. The directory is
shipped empty apart from its `.gitkeep` file, so that every reported number is produced by the
reader's own run rather than copied from the authors' machine.

The generated files follow the naming used for Tables 5-13: the main result table
(`main_results_for_paper_unified.csv`), the ablation table, fixed-FPR classification metrics,
FPR sweeps, cost-sensitive tables, hyperparameter sensitivity, and runtime analysis, together
with the underlying cross-validation metric files (`metrics_cv_*.json`,
`metrics_temporal_*.json`, `bootstrap_ci.json`) and the PaySim experiment summary.

## Notes on generated outputs

Large generated intermediate files, processed feature matrices, prediction arrays, and raw data
files are intentionally excluded from Git tracking; `data/` and `outputs/` are created locally on
first run and are ignored by Git.

## Note on recent updates

To ensure compatibility with newer software environments, the preprocessing scripts handle pandas
3.x string dtypes explicitly. This release also includes the scripts for the PaySim
supplementary experiment. The core algorithmic logic and interfaces of the main framework remain
unchanged.

PaySim scripts are provided in the `scripts/` folder. Core IEEE-CIS algorithms remain unchanged.

## License

This code is released under the MIT License.

