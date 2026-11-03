# Analysis code

Python code of the analyses reported in the paper. It reads the archive in `../data/` and writes all outputs to
`../results/` (not part of the repository).

## Steps

| Script | Paper | What it does | Output |
|---|---|---|---|
| `step1_chronological_analysis.py` | Section 2.4 | Tunes the 21 candidate models on each training block (Optuna, grouped 5-fold CV), freezes the selected model before the incoming season, fits the slope–bias and bias-only corrections on the two transfer levels and evaluates the remaining test levels | `results/chronological/` |
| `step2_paper_tables_figures.py` | Sections 2.4 and 3 | Metrics with bootstrap confidence intervals, paired comparisons, transfer-level sensitivity and the tables and figures, all from the saved predictions of step 1 (no retraining) | `results/chronological/paper/` |
| `step3_digital_shadow_evaluation.py` | Section 2.3.2 | Residual baseline r = ρg − ρd, fault-free S4 monitoring with Shewhart, EWMA and CUSUM charts, synthetic offsets in mass, attenuation and phase, detection within five records, effect of the offsets on predicted MC | `results/digital_shadow/` |
| `prepare_archive.py` | Section 2.5 | Provenance of `data/lentil_microwave_archive.csv`: merges the raw archive files and reconstructs the level, filling and packing-state columns | `data/lentil_microwave_archive.csv` |

Run the steps in order from this folder:

```bash
python step1_chronological_analysis.py
python step2_paper_tables_figures.py
python step3_digital_shadow_evaluation.py
```

Step 1 takes several hours; it can be interrupted and restarted with the same settings, and finished combinations and
Optuna trials are read from disk. Steps 2 and 3 take minutes. Step 3 uses the frozen models (checkpoints) of step 1.
`prepare_archive.py` needs the raw archive files, which are not part of the repository; the merged CSV is.

## Modules

| Module | Content |
|---|---|
| `common.py` | Paths, data loading, timing, device selection and crash-safe file writing |
| `candidate_models.py` | Definitions and Optuna search spaces of the 15 core models, TabPFN versions, preflight check |
| `chronological.py` | Records and input sets, six further models (PLS, PCR, MLP, GPR, AdaBoost-SVR, 1-D CNN), tuning, freezing, transfer selection, corrections, metrics and paired bootstrap comparisons |
| `paper_outputs.py` | Statistics, tables (Excel, LaTeX, CSV) and figures of the paper |
| `archive.py` | Records, input blocks and transfer levels for the digital shadow evaluation |
| `dielectric.py` | Free-space relations for ε′ and ε″ (Section 2.2) and the synthetic offsets (Section 2.3.2) |
| `frozen_models.py` | Loading of the frozen models and the slope–bias calibration |
| `spc.py` | Shewhart, EWMA and CUSUM charts on the residual score z (Section 2.3.1) |
| `shadow_evaluation.py` | Digital shadow evaluation (Section 2.3.2) |

## Environment

The analyses were run on Windows 11 with Python 3.11.16 and an NVIDIA GPU (PyTorch 2.11.0, CUDA 12.8); the package
versions are listed in `requirements.txt`. Install PyTorch for your system first (see the comments in
`requirements.txt`), then:

```bash
pip install -r requirements.txt
```

**TabPFN.** TabPFN v2 is openly licensed. TabPFN v2.5 and v3 are gated: they need a Prior Labs API key
(https://ux.priorlabs.ai) and the acceptance of their licence. Set the key as the environment variable `TABPFN_TOKEN`
before starting Python, or put it in a file `local_keys.py` in this folder (`TABPFN_API_KEY = "..."`; optionally
`HF_TOKEN = "..."`). `local_keys.py` is listed in `.gitignore` and must never be committed. Models that cannot be
loaded are reported by the preflight check and skipped; to reproduce the paper, all 21 models must be available.

## Reproducibility

- Fixed random seeds: Optuna TPE sampler 42, model seeds 42, bootstrap resamples 0, digital shadow sequences 0.
- The pre-specified selection rule (lowest CV-RMSE) and the primary configuration (`paper_outputs.PRIMARY`) do not
  depend on the test results.
- TabPFN results can differ slightly between GPU and CPU computation.
