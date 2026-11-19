# Covariance Pooling on Tabular NetFlow (ICCIT 2026)

Code, split construction, configurations and per-run results for the paper

> N. Sakib, M. S. H. Sarker, S. Hassan, M. O. F. Bhuiyan, S. D. Joy and S. Paul, "Covariance Pooling Beats
> Mean but Not Max Pooling on Tabular NetFlow: A Mechanistic and Cross-Dataset Study," in *Proc. 29th
> International Conference on Computer and Information Technology (ICCIT)*, Cox's Bazar, Bangladesh, 2026.

The paper trains matched 1D convolutional backbones that differ only in their pooling head
(global average pooling GAP, global max pooling GMP, squeeze-and-excitation SE, and iSQRT-COV global
covariance pooling GCP) on four NetFlow-v2 datasets, and compares them in-distribution, under
leave-one-dataset-out shift, on two alternative backbones (a dilated residual TCN and an MLP-Mixer), and on synthetic data that isolates
covariance from mean signal.

Every number in the paper comes from a JSON file in `results/`. The tables and figures can be rebuilt
from those files on a laptop without a GPU or the datasets.

## Contents

```
src/                         library: data loading and splits, models, GCP layer, training, metrics, synthetic generator
scripts/                     experiment drivers, summaries, and the table and figure builders
reviewer_runs_2026-11-18/    alternative-backbone runner (TCN, MLP) added in the revision
results/                     per-seed result JSONs behind every table and figure (about 3 MB)
requirements.txt
```

## Setup

Python 3.11 or 3.12.

```bash
pip install -r requirements.txt
```

## Rebuild the tables and figures (no GPU, no data)

Run from the repository root. Outputs go to `outputs/`.

| Paper item | Command |
|---|---|
| Table I (in-distribution macro-F1, Holm column) | `python scripts/summarize_table2.py` (prints the table body) |
| Tables II and III | `python scripts/make_iccit_tables.py` → `outputs/tables/` |
| Fig. 1 (pipeline) | `python scripts/make_iccit_fig_pipeline.py` → `outputs/figures/` |
| Fig. 2 (channel-width sweep) | `python scripts/make_iccit_fig_width.py` → `outputs/figures/` |

The two tables built this way are byte-identical to the ones in the paper.

## Data

The four datasets are the NetFlow-v2 versions of UNSW-NB15, CSE-CIC-IDS2018, ToN-IoT and BoT-IoT
(Sarhan et al.). They are distributed by the University of Queensland as one combined file,
NF-UQ-NIDS-v2, at <https://staff.itee.uq.edu.au/marius/NIDS_datasets/>. The data is not included here;
please download it from that page under its own terms.

```bash
# 1. convert the downloaded CSV to parquet (streams, so it fits in memory)
python scripts/csv_to_parquet.py NF-UQ-NIDS-v2.csv data/raw/NF-UQ-NIDS-v2.parquet

# 2. build the four train/validation/test splits used in the paper
python scripts/build_splits.py
```

`build_splits.py` uses the paper's settings and prints the row counts next to the original ones, so
a rebuilt split can be checked before any training. The settings are: seed 42; a 70/15/15 split
stratified on the native class; validation and test rows whose feature hash occurs in training
dropped; constant columns and z-scoring fit on training only; and the three larger corpora capped at
400,000 rows per class (NF-UNSW-NB15-v2 uncapped). The parquet path can also be set with the
`NF_UQ_PARQUET` environment variable. Every result JSON from the TCN runs also records checksums of
the feature arrays it was trained on (`split_checksums`, the first 16 hex digits of a SHA-256), so a
rebuilt split can be compared to the original.

## Configuration

These are the defaults in the drivers and were used for every reported run unless a row below says
otherwise.

| Setting | Value |
|---|---|
| Seeds | 42, 123, 456, 789, 1337 (n = 5 per cell) |
| Backbone | three Conv1d layers, channels 16-32-32, kernel 3, no padding (L' = F − 6) |
| Heads | GAP, GMP, SE (reduction 4), GCP (iSQRT-COV, 3 Newton–Schulz iterations, 1e-5 ridge); GAP-wide and GMP-wide project to GCP's 528-dim output |
| Training | Adam, learning rate 1e-3, batch 512, 20 epochs, cross-entropy, binary attack/benign labels |
| Model selection | checkpoint with the best validation macro-F1; the test split is read once |
| TCN | three residual blocks of two kernel-3 convolutions, widths 16-32-32, dilations 1, 2, 4, same padding (L' = F) |
| MLP | MLP-Mixer without convolution: a 32-channel embedding per feature, two blocks mixing across features and across channels (L' = F) |
| Statistics | paired Student-t 95% intervals over seeds, Cohen's d_z; a difference is resolved when \|d_z\| > 1.242 (t₀.₉₇₅,₄/√5) |

## Re-running the experiments (GPU)

Each driver is resumable: a cell whose result JSON exists is skipped. Run `python <script> --help`
for its options.

| Paper item | Results folder | Produced by |
|---|---|---|
| Table I, GAP/SE/GCP on NF-UNSW-NB15-v2 | `results/repro/` | `scripts/run_repro_t6b.py` |
| Table I, GAP/SE/GCP on CSE-CIC-IDS2018 and ToN-IoT | `results/gateC/` | `scripts/run_gateC_headline.py` |
| Table I, all heads on BoT-IoT | `results/botredo/` | `scripts/run_botiot_same_driver.py` |
| Table I, GMP; GMP-wide control (all four corpora) | `results/maxpool/` | `scripts/run_maxpool_control.py` (`--heads gmp_wide` for the width-matched control) |
| Table I, gradient-boosted tree | `results/tree_baseline/` | `scripts/run_tree_baseline.py` |
| Table II, TCN rows | `results/reviewer_tcn/` | `reviewer_runs_2026-11-18/run_alt_arch.py --backbone tcn` (`run_all_alt_arch.sh` loops heads and seeds) |
| Table II, MLP rows | `results/reviewer_mlp/` | `reviewer_runs_2026-11-18/run_alt_arch.py --backbone mlp` |
| Table III top, leave-one-dataset-out | `results/gap1/`, `results/gap1_gmp_legacy/`, `results/gap1_repro_check/` | `scripts/run_gap1.py` (GMP cells with `--legacy-protocol`) |
| Table III bottom, synthetic sweeps | `results/gateB_synthetic_n40k/`, `results/gateB_synthetic_far/` | `scripts/run_gateB_synthetic.py` |
| 40-epoch budget check | `results/sens40/` | `scripts/run_maxpool_control.py --epochs 40 --results-dir results/sens40` |
| GAP-wide width control | `results/gapwide/` | `scripts/run_gapwide_control.py` |
| Fig. 2, channel-width sweep | `results/tier1/` | `scripts/run_tier1_cwidth.py` |
| Per-sample effective rank | `results/task1/` | `scripts/run_task1_widths.py`, `scripts/analyze_task1_effrank.py` |
| Dataset effective rank | `results/gateC/gateC_summary.json` | `scripts/run_gateC_headline.py` |
| Coupling-index negative | `results/efc/` | `scripts/run_efc_coupling.py`, `scripts/analyze_efc_*.py` |
| Mechanism probe | `results/gap3/`, `results/gap3_ton/` | `scripts/run_gap3.py`, `scripts/run_m4_ton_probe.py` |
| CPU inference cost | `results/latency/` | `scripts/run_latency.py` |
| Feature-permutation check | `results/feature_perm/` | `scripts/run_feature_permutation.py` |
| Extra Trees reference | `results/extratrees/` | `scripts/run_extratrees_baseline.py` |
| Dataset row and class counts | `results/data_inventory.json` | `src/data.py` |

Intervals in the summary files can be recomputed with `scripts/recompute_t_intervals.py`.

Two notes on provenance. The dataset effective-rank correlation stored in
`results/gateC/gateC_summary.json` (r = −0.236) uses an older BoT-IoT run; the paper's r = −0.15 uses
the BoT-IoT GCP−GAP difference of `results/botredo/` (0.0052), the same run as Table I. The GPU runs
were made on several machines (local GPUs and rented RTX 3090s), so a rerun reproduces the numbers to
within GPU nondeterminism rather than bit for bit; `results/gap1_repro_check/` shows one original
cell reproduced to every reported digit.

## Citation

See `CITATION.cff`.

## License

Code: MIT (see `LICENSE`). The datasets keep their own licences.
