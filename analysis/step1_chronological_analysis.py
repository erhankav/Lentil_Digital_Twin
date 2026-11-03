# -*- coding: utf-8 -*-
"""
Step 1: chronological (forward-season) analysis, Section 2.4 of the paper.

Tunes the 21 candidate models on each training block, freezes the selected model before the incoming season is used,
fits the slope-bias and bias-only corrections on the transfer levels and evaluates the test levels.
Output: results/chronological/ (tables/, checkpoints/, tuning/, FINAL_REPORT.txt).

The settings below are those of the paper. The run takes several hours; it can be interrupted and restarted with the
same settings: finished combinations and Optuna trials are read from disk.

    python step1_chronological_analysis.py
"""
import chronological as kr
from common import RESULTS_ROOT

OUT_ROOT = RESULTS_ROOT / "chronological"
N_TRIALS = 30                                 # Optuna trials per model and combination
TRIALS_BY_MODEL = {"EBM": 10, "GPR": 20}      # fewer trials for the slow models (CNN1D has a fixed architecture)
CV_FOLDS = 5                                  # folds of the grouped CV used for tuning
N_BOOT = 2000                                 # bootstrap resamples over the test MC levels
RESUME = True                                 # continue an interrupted run


if __name__ == "__main__":
    ok_models = kr.preflight_check()          # 21 models; all must be available to reproduce the paper
    missing = [m for m in kr.ALL_MODEL_ORDER if m not in ok_models]
    if missing:
        print(f"\nWARNING: models not available in this environment: {missing}. The run continues without them, "
              "so model selection can differ from the paper.\n")
    kr.run_chronological(out_root=OUT_ROOT, n_trials=N_TRIALS, trials_by_model=TRIALS_BY_MODEL,
                         models=ok_models, resume=RESUME, n_boot=N_BOOT, cv_folds=CV_FOLDS)
