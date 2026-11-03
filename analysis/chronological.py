# -*- coding: utf-8 -*-
"""
CHRONOLOGICAL (FORWARD-SEASON) ANALYSIS - Section 2.4 of the paper                [version 2.0]

Transitions: S1->S2, S1-S2->S3, S1-S3->S4   (S1..S4 = 2014..2017)
Targets    : moisture content (MC, primary) and bulk density (BD, secondary)
Inputs     : all 41 frequencies, wide format (one record = one row)
               spectra   : dA, dphi                                   (82)
               microwave : dA, dphi, e1, e2, X = dA/dphi              (205)
               full (MC) : microwave + (e1-1)/rho_g + e2/rho_g        (287)   [MC only]
             The BD models do NOT use the mass / rho_g input (spectra and microwave).
Models     : (1) CORE: 15 models with the definitions and Optuna search spaces of candidate_models.py.
                 EBM is used without pairwise interactions (EBM_INTERACTIONS = 0).
             (2) EXTRA, common for spectral / wide inputs:
                 PLS, PCR, MLP (ANN), GPR (Gaussian process), AdaBoost_SVR, CNN1D (1-D convolutional network)
             Cultivar-specific (Ceren, Ciftci) and pooled (no cultivar indicator) models.
Tuning     : Optuna (TPE, seed 42); criterion = k-fold GroupKFold CV-RMSE (group = cultivar + season + MC level;
             k = CV_FOLDS = 5). Pre-specified selection rule: lowest CV-RMSE. X is standardised with training statistics;
             the target is not standardised (CNN and MLP scale the target internally).
Freezing   : the selected configuration is refitted on the whole training block and frozen before the incoming season
             is used.
Transfer   : in the incoming season, for each cultivar, the 2 levels closest to the 25th and 75th percentiles of the level
             MCs; the correction (b0 + b1*yhat, or b0 only) is fitted on these records, the REMAINING levels form the test set.
Comparison : paired difference on the same test records, 2000 bootstrap resamples over the test MC levels.

Version 2.0 additions (they do NOT change the design of the main analysis; supplementary / sensitivity analyses only):
  * the incoming-season predictions of ALL models are saved for every combination (tables/all_model_predictions.csv):
    reliability of the model selection, ensembles and the one-standard-error rule can be computed afterwards
  * extended metrics: MAE, MedAE, MaxErr, MAPE, RPD, RPIQ, slope, intercept
  * configuration signature: if the model list or the number of trials changes, old checkpoints are invalidated

Target values are NEVER modified. The level mean is used only to SORT the levels for the transfer selection.
This module does not contain the consistency monitoring of the digital twin (Section 2.3.2; see shadow_evaluation.py);
the out-of-fold residuals of the selected BD model on the S1-S3 training block are saved for it.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import TransformedTargetRegressor  # noqa: E402
from sklearn.cross_decomposition import PLSRegression  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import AdaBoostRegressor  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

# Core model definitions, search spaces and TabPFN versions come from candidate_models.py.
from candidate_models import (HYPERPARAMS, MODEL_ORDER, UNAVAILABLE, _suggest,  # noqa: E402
                                   model_factories)
from candidate_models import preflight_check as _candidates_preflight  # noqa: E402
from common import (DATA_ROOT, RESULTS_ROOT, FREQS, get_device, header, hms, load_dataset, now_str,  # noqa: E402
                            pickle_dump_atomic, pickle_load_safe, save_table, text_dump_atomic)

try:
    import torch
    import torch.nn as nn
except Exception as _exc:  # noqa: BLE001
    torch = None
    nn = None
    UNAVAILABLE["CNN1D"] = f"torch not available: {_exc}"

optuna.logging.set_verbosity(optuna.logging.WARNING)

CODE_VERSION = "2.0"
N_FREQ = len(FREQS)
SEASONS = {2014: "S1", 2015: "S2", 2016: "S3", 2017: "S4"}
TRANSITIONS = [((2014,), 2015), ((2014, 2015), 2016), ((2014, 2015, 2016), 2017)]
CULTIVARS = {"Ceren": "CEREN", "Ciftci": "CIFTCI"}          # model_type -> Type value
MODEL_TYPES = ["Ceren", "Ciftci", "Pooled"]

INPUT_SETS = {
    "spectra": ["dA", "dphi"],
    "microwave": ["dA", "dphi", "e1", "e2", "X"],
    "full": ["dA", "dphi", "e1", "e2", "X", "e1n", "e2n"],
}
TARGET_INPUTS = {"moisture": ["spectra", "microwave", "full"], "bulk_density": ["spectra", "microwave"]}
TARGET_COL = {"moisture": "MC", "bulk_density": "BD"}

# Number of Optuna trials (per model and combination). The inputs have 205-287 dimensions and every trial is
# a k-fold CV, so trials are expensive. A separate number can be given for very slow models.
DEFAULT_TRIALS = 30
DEFAULT_TRIALS_BY_MODEL = {"EBM": 10, "CNN1D": 0, "GPR": 20}

# Number of folds (k) of the grouped cross-validation used for tuning (Section 2.4: k = 5). With k = 5 every
# fold trains on 80 % of the training data; the folds are small (S1 Ceren has 12 levels -> 2-3 levels per fold).
#
CV_FOLDS = 5

# Pairwise interaction terms of EBM. With 205-287 inputs the interaction search is very slow (one fit took about
# 200 s at 285 x 205 and ran out of memory at 570 x 287) and is not feasible over 45 combinations x Optuna x k-fold
# CV. interactions = 0 (purely additive EBM) reduces one fit to 3-8 s.
# None uses the EBM default (not recommended: very slow, may run out of memory).
EBM_INTERACTIONS = 0


def trn_label(train_years, test_year):
    return "+".join(SEASONS[y] for y in train_years) + "to" + SEASONS[test_year]


# ----------------------------------------------------------------------------------------------
# Data: record (row) table and wide input blocks
# ----------------------------------------------------------------------------------------------
def build_records(df):
    df = df.sort_values(["Type", "Year", "Sample_ID", "Freq"]).reset_index(drop=True)
    if len(df) % N_FREQ:
        raise ValueError("the number of rows is not a multiple of 41")
    n = len(df) // N_FREQ

    def arr(col):
        return df[col].to_numpy(float).reshape(n, N_FREQ)

    if not (arr("Freq") == FREQS).all():
        raise ValueError("the frequency order is not the expected 2000..4000")
    for tcol in ("Moisture_Content", "Bulk_Density"):
        if np.ptp(arr(tcol), axis=1).max() > 0:           # the std may not be exactly 0 in floating point
            raise ValueError(f"{tcol} is not constant within a record")
    first = df.iloc[::N_FREQ].reset_index(drop=True)
    rec = first[["Type", "Year", "Level_Key", "Level_ID", "Filling_ID", "Packing_State", "Sample_ID"]].copy()
    rec["MC"] = first["Moisture_Content"].to_numpy(float)
    rec["BD"] = first["Bulk_Density"].to_numpy(float)
    rec["rid"] = np.arange(n)
    rho = rec["BD"].to_numpy()[:, None]
    dA, dphi, e1, e2 = arr("Attenuation"), arr("Phase_Shift"), arr("E1"), arr("E2")
    blocks = {"dA": dA, "dphi": dphi, "e1": e1, "e2": e2, "X": dA / dphi,
              "e1n": (e1 - 1.0) / rho, "e2n": e2 / rho}
    return rec, blocks


def feature_matrix(blocks, input_set):
    return np.hstack([blocks[b] for b in INPUT_SETS[input_set]])


def select_transfer_levels(rec):
    """For each (incoming season, cultivar): the two levels closest to the 25th and 75th percentiles (ties: lower MC).
    The level mean is only a sort key; target values are not modified."""
    out = {}
    for (year, typ), g in rec.groupby(["Year", "Type"]):
        lv = g.groupby("Level_Key")["MC"].mean().sort_values(kind="stable")
        vals, keys = lv.to_numpy(), lv.index.to_list()
        chosen = []
        for q in (25, 75):
            target = np.percentile(vals, q)             # linear interpolation
            dist = np.abs(vals - target)
            for j in np.argsort(dist, kind="stable"):    # stable sort: on ties the lower MC comes first
                if keys[j] not in chosen:
                    chosen.append(keys[j])
                    break
        out[(year, typ)] = chosen
    return out


# ----------------------------------------------------------------------------------------------
# EXTRA models: families commonly used for spectral / wide inputs
# ----------------------------------------------------------------------------------------------
EXTRA_MODEL_ORDER = ["PLS", "PCR", "MLP", "GPR", "AdaBoost_SVR", "CNN1D"]
ALL_MODEL_ORDER = list(MODEL_ORDER) + EXTRA_MODEL_ORDER

EXTRA_HYPERPARAMS = {
    "PLS": {"n_components": {"type": "int", "range": (1, 30)}},
    "PCR": {"n_components": {"type": "int", "range": (1, 30)}},
    "MLP": {"hidden": {"type": "categorical", "choices": ["32", "64", "64-32", "128-64"]},
            "alpha": {"type": "float", "range": (1e-5, 1e-1), "log": True},
            "learning_rate_init": {"type": "float", "range": (1e-4, 1e-2), "log": True}},
    "GPR": {"length_scale": {"type": "float", "range": (1.0, 300.0), "log": True},
            "noise": {"type": "float", "range": (1e-4, 1.0), "log": True}},
    "AdaBoost_SVR": {"C": {"type": "float", "range": (1e-1, 1e3), "log": True},
                     "epsilon": {"type": "float", "range": (1e-3, 1.0), "log": True},
                     "gamma": {"type": "float", "range": (1e-4, 1e1), "log": True},
                     "n_estimators": {"type": "int", "range": (10, 50)},
                     "learning_rate": {"type": "float", "range": (0.1, 1.0), "log": True}},
    "CNN1D": {},                                    # fixed architecture, no tuning (the CV score is still computed)
}
ALL_HYPERPARAMS = {**HYPERPARAMS, **EXTRA_HYPERPARAMS}

if torch is not None:
    class _CNNNet(nn.Module):
        def __init__(self, n_channels):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(n_channels, 16, 5, padding=2), nn.ReLU(),
                nn.Conv1d(16, 16, 5, padding=2), nn.ReLU(),
                nn.AdaptiveAvgPool1d(8), nn.Flatten())
            self.head = nn.Sequential(nn.Linear(16 * 8, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            return self.head(self.conv(x)).squeeze(-1)


class CNN1DRegressor:
    """Each input block is one channel; convolution over the 41 ordered frequencies. 2 convolutions (16 filters, width 5),
    adaptive average pooling to 8 positions, dense layer with 32 units, linear output.
    250 epochs, Adam, lr 3e-3, weight decay 1e-4, mini-batch 64. The target is standardised internally."""

    def __init__(self, n_channels=5, seed=42, epochs=250, lr=3e-3, weight_decay=1e-4, batch=64):
        self.n_channels, self.seed, self.epochs = n_channels, seed, epochs
        self.lr, self.weight_decay, self.batch = lr, weight_decay, batch
        self.net = None

    def _x(self, X):
        return torch.tensor(np.asarray(X, dtype=np.float32).reshape(len(X), self.n_channels, -1))

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.my, self.sy = float(y.mean()), float(y.std()) or 1.0
        torch.manual_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        net = _CNNNet(self.n_channels)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        Xt, yt = self._x(X), torch.tensor(((y - self.my) / self.sy).astype(np.float32))
        n = len(Xt)
        net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                opt.zero_grad()
                loss = torch.mean((net(Xt[idx]) - yt[idx]) ** 2)
                loss.backward()
                opt.step()
        net.eval()
        self.net = net
        return self

    def predict(self, X):
        with torch.no_grad():
            return self.net(self._x(X)).numpy() * self.sy + self.my


def extra_factories():
    """Factories of the extra models. Models that cannot be used (e.g. no torch) are left out."""
    f = {
        "PLS": lambda **kw: PLSRegression(n_components=kw.get("n_components", 5), scale=False),
        "PCR": lambda **kw: make_pipeline(PCA(n_components=kw.get("n_components", 5)), LinearRegression()),
        "MLP": lambda **kw: TransformedTargetRegressor(
            regressor=MLPRegressor(hidden_layer_sizes=tuple(int(x) for x in str(kw.get("hidden", "64")).split("-")),
                                   alpha=kw.get("alpha", 1e-3), learning_rate_init=kw.get("learning_rate_init", 1e-3),
                                   early_stopping=True, n_iter_no_change=20, max_iter=2000, random_state=42),
            transformer=StandardScaler()),
        "GPR": lambda **kw: GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-3, 1e3)) * RBF(kw.get("length_scale", 10.0), (1e-2, 1e4))
            + WhiteKernel(kw.get("noise", 0.1), (1e-6, 1e1)),
            normalize_y=True, n_restarts_optimizer=0, random_state=42),
        "AdaBoost_SVR": lambda **kw: AdaBoostRegressor(
            estimator=SVR(C=kw.get("C", 10.0), epsilon=kw.get("epsilon", 0.1), gamma=kw.get("gamma", 1e-3)),
            n_estimators=kw.get("n_estimators", 20), learning_rate=kw.get("learning_rate", 1.0), random_state=42),
    }
    if torch is not None:
        f["CNN1D"] = lambda n_channels=5, **kw: CNN1DRegressor(n_channels=n_channels, seed=42)
    return f


# ----------------------------------------------------------------------------------------------
# Model: X is standardised with training statistics, the target is not
# ----------------------------------------------------------------------------------------------
class FrozenModel:
    """Fitted model + training scaler. (The factory lambda is not stored, so the object can be pickled.)"""

    def __init__(self, name, params, scaler, model):
        self.name, self.params, self.sx, self.model = name, params, scaler, model

    def predict(self, X):
        return np.ravel(self.model.predict(self.sx.transform(X))).astype(float)


def fit_model(name, params, factory, X, y, n_channels=None):
    sx = StandardScaler().fit(X)
    kw = dict(params)
    if name == "CNN1D":
        kw["n_channels"] = n_channels or 5
    model = factory(**kw)
    model.fit(sx.transform(X), np.asarray(y, dtype=float))
    return FrozenModel(name, params, sx, model)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def tune_model(name, factory, X, y, groups, n_trials, base_path, resume, n_folds=None, n_channels=None):
    """Searches with Optuna (search spaces of candidate_models.py); criterion = k-fold grouped CV-RMSE (k = CV_FOLDS).
    The result (best parameters, CV-RMSE, out-of-fold predictions) is written to disk; the Optuna trials are kept in
    sqlite, so an interrupted search resumes from the last trial. If k changes, old results and trials are not used."""
    base_path = Path(base_path)
    n_folds = n_folds or CV_FOLDS
    ckpt = base_path.with_suffix(".pkl")
    spec = ALL_HYPERPARAMS.get(name)
    want_trials = n_trials if spec else 0
    if resume:
        done = pickle_load_safe(ckpt)
        if (done is not None and done.get("n_trials") == want_trials and done.get("version") == CODE_VERSION
                and done.get("cv_folds") == n_folds):
            return done
    base_path.parent.mkdir(parents=True, exist_ok=True)
    folds = list(GroupKFold(n_splits=min(n_folds, len(np.unique(groups)))).split(X, y, groups))

    def oof_for(params):
        oof = np.zeros(len(y))
        for tr, va in folds:
            oof[va] = fit_model(name, params, factory, X[tr], y[tr], n_channels).predict(X[va])
        return oof

    best_params = {}
    if spec:
        db = Path(f"{base_path}__k{n_folds}.db").resolve().as_posix()      # separate study if k differs
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42),
                                    study_name=f"{base_path.name}__k{n_folds}", storage=f"sqlite:///{db}",
                                    load_if_exists=True)
        n_done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)

        def objective(trial):
            try:
                return rmse(y, oof_for(_suggest(trial, spec)))
            except Exception:  # noqa: BLE001
                return 1e9

        if n_done < want_trials:
            study.optimize(objective, n_trials=want_trials - n_done, show_progress_bar=False)
        if study.best_value < 1e9:
            best_params = study.best_params
    oof = oof_for(best_params)                                  # TabPFN / LinearRegression / CNN1D: no tuning, CV only
    res = {"model": name, "params": best_params, "cv_rmse": rmse(y, oof), "oof": oof, "n_trials": want_trials,
           "version": CODE_VERSION, "cv_folds": n_folds}
    pickle_dump_atomic(res, ckpt)
    return res


def resolve_models(models=None):
    """Available models: core + extra, in the order of ALL_MODEL_ORDER."""
    factories = dict(model_factories(get_device()))
    factories.update(extra_factories())
    if "EBM" in factories and EBM_INTERACTIONS is not None:
        _ebm = factories["EBM"]
        factories["EBM"] = lambda _b=_ebm, **kw: _b(interactions=EBM_INTERACTIONS, **kw)
    wanted = list(models or ALL_MODEL_ORDER)
    return [m for m in ALL_MODEL_ORDER if m in wanted and m in factories], factories


def preflight_check(include_wide=True, ask_key=True):
    """Tries the core and the extra models at the wide input size of this analysis (205 features)."""
    _, factories = resolve_models()                      # the actual definitions of this analysis, including EBM_INTERACTIONS
    return _candidates_preflight(n_samples=120, n_features=205 if include_wide else None, ask_key=ask_key,
                         factories=factories)


# ----------------------------------------------------------------------------------------------
# Correction and metrics
# ----------------------------------------------------------------------------------------------
def correction_columns(g):
    """Fits the corrections on the transfer records of one (combination, cultivar) group.
    g: columns y_true, y_pred, is_transfer. Returns: p_none, p_slope_bias, p_bias_only, b0, b1."""
    tr = g[g["is_transfer"]]
    yp = g["y_pred"].to_numpy(float)
    if len(tr) >= 2 and np.std(tr["y_pred"]) > 0:
        b1, b0 = np.polyfit(tr["y_pred"].to_numpy(float), tr["y_true"].to_numpy(float), 1)
    else:
        b1, b0 = 1.0, 0.0
    shift = float(np.mean(tr["y_true"] - tr["y_pred"])) if len(tr) else 0.0
    return {"p_none": yp, "p_slope_bias": b0 + b1 * yp, "p_bias_only": yp + shift, "b0": b0, "b1": b1}


def apply_corrections(preds):
    """The three correction variants of the main predictions (none / slope_bias / bias_only)."""
    out = []
    for _, g in preds.groupby(["target", "input_set", "transition", "model_type", "Type"], sort=False):
        g = g.copy()
        for k, v in correction_columns(g).items():
            g[k] = v
        out.append(g)
    return pd.concat(out, ignore_index=True)


def metric_row(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    e = p - y
    bias = float(e.mean())
    sst = float(np.sum((y - y.mean()) ** 2))
    r = float(np.sqrt(np.mean(e ** 2)))
    q75, q25 = np.percentile(y, [75, 25])
    try:
        slope, intercept = np.polyfit(y, p, 1)               # prediction = intercept + slope * reference
    except Exception:  # noqa: BLE001
        slope, intercept = np.nan, np.nan
    return {"RMSEP": r, "MAE": float(np.mean(np.abs(e))), "MedAE": float(np.median(np.abs(e))),
            "MaxErr": float(np.max(np.abs(e))), "MAPE_pct": float(np.mean(np.abs(e) / np.abs(y)) * 100),
            "Bias": bias,
            "SEP": float(np.sqrt(np.sum((e - bias) ** 2) / (len(e) - 1))) if len(e) > 1 else np.nan,
            "R2": float(1 - np.sum(e ** 2) / sst) if sst > 0 else np.nan,
            "RPD": float(np.std(y, ddof=1) / r) if (r > 0 and len(y) > 1) else np.nan,
            "RPIQ": float((q75 - q25) / r) if r > 0 else np.nan,
            "Slope": float(slope), "Intercept": float(intercept)}


CORRECTIONS = {"none": "p_none", "slope_bias": "p_slope_bias", "bias_only": "p_bias_only"}


def test_metrics_table(corrected):
    test = corrected[~corrected["is_transfer"]]
    rows = []
    for keys, g in test.groupby(["target", "input_set", "transition", "model_type", "Type"], sort=False):
        for cname, col in CORRECTIONS.items():
            r = dict(zip(["target", "input_set", "transition", "model_type", "cultivar"], keys))
            r.update(correction=cname, N=len(g), N_levels=g["Level_Key"].nunique(),
                     **metric_row(g["y_true"].to_numpy(), g[col].to_numpy()))
            rows.append(r)
    return pd.DataFrame(rows)


def all_models_test_table(allpred):
    """Incoming-season performance of every (combination, cultivar, model) (NOT used for selection; reporting only).
    For every model the three corrections are computed from its own transfer records."""
    rows = []
    keys = ["target", "input_set", "transition", "model_type", "Type", "model"]
    for k, g in allpred.groupby(keys, sort=False):
        cc = correction_columns(g)
        test = ~g["is_transfer"].to_numpy()
        y = g["y_true"].to_numpy(float)[test]
        for cname, col in CORRECTIONS.items():
            m = metric_row(y, cc[col][test])
            rows.append({**dict(zip(keys, k)), "correction": cname, "N": int(test.sum()),
                         "RMSEP": m["RMSEP"], "MAE": m["MAE"], "Bias": m["Bias"], "R2": m["R2"]})
    return pd.DataFrame(rows).rename(columns={"Type": "cultivar"})


# ----------------------------------------------------------------------------------------------
# Combination (target x input set x transition x model type)
# ----------------------------------------------------------------------------------------------
def _signature(models, n_trials, trials_by_model, cv_folds):
    return {"version": CODE_VERSION, "models": list(models), "cv_folds": cv_folds,
            "trials": {m: trials_by_model.get(m, n_trials) for m in models}, "ebm_interactions": EBM_INTERACTIONS}


def run_combo(out_dir, rec, blocks, transfer, factories, target, input_set, train_years, test_year, model_type,
              models, n_trials, trials_by_model, resume, save_all_model_predictions=True, cv_folds=None):
    cv_folds = cv_folds or CV_FOLDS
    combo_id = f"{target}__{input_set}__{trn_label(train_years, test_year)}__{model_type}"
    ckpt = Path(out_dir) / "checkpoints" / f"{combo_id}.pkl"
    sig = _signature(models, n_trials, trials_by_model, cv_folds)
    if resume:
        done = pickle_load_safe(ckpt)
        if done is not None:
            if done.get("signature") == sig:
                return done, True
            print(f"      (configuration changed -> {combo_id} is recomputed)")
    tcol = TARGET_COL[target]
    X_all = feature_matrix(blocks, input_set)
    n_channels = len(INPUT_SETS[input_set])
    cultivars = list(CULTIVARS.values()) if model_type == "Pooled" else [CULTIVARS[model_type]]
    tr_mask = rec["Year"].isin(train_years).to_numpy() & rec["Type"].isin(cultivars).to_numpy()
    te_mask = (rec["Year"] == test_year).to_numpy() & rec["Type"].isin(cultivars).to_numpy()
    Xtr, ytr, groups = X_all[tr_mask], rec.loc[tr_mask, tcol].to_numpy(float), rec.loc[tr_mask, "Level_Key"].to_numpy()

    rows, tuned = [], {}
    for name in models:
        t0 = time.time()
        try:
            res = tune_model(name, factories[name], Xtr, ytr, groups, trials_by_model.get(name, n_trials),
                             Path(out_dir) / "tuning" / f"{combo_id}__{name}", resume, n_folds=cv_folds,
                             n_channels=n_channels)
        except Exception as exc:  # noqa: BLE001
            print(f"      ✗ {name}: {str(exc)[:90]}")
            text_dump_atomic(f"{exc}\n\n{traceback.format_exc()}", Path(out_dir) / "tuning" / f"FAILED_{combo_id}__{name}.txt")
            continue
        tuned[name] = res
        rows.append({"model": name, "cv_rmse": res["cv_rmse"], "params": str(res["params"]),
                     "n_trials": res["n_trials"], "sec": time.time() - t0})
    if not rows:
        raise RuntimeError("no model could be fitted")
    model_table = pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)
    sel = model_table.iloc[0]["model"]          # MAIN SELECTION RULE (pre-specified): lowest CV-RMSE

    def incoming_frame(fz, name):
        frames = []
        for typ in cultivars:
            m = te_mask & (rec["Type"] == typ).to_numpy()
            sub = rec[m].copy()
            sub["y_true"] = sub[tcol].to_numpy(float)
            sub["y_pred"] = fz.predict(X_all[m])
            sub["is_transfer"] = sub["Level_Key"].isin(transfer[(test_year, typ)])
            frames.append(sub[["rid", "Type", "Year", "Level_Key", "y_true", "y_pred", "is_transfer"]])
        out = pd.concat(frames, ignore_index=True)
        out.insert(0, "model", name)
        out.insert(0, "model_type", model_type)
        out.insert(0, "transition", trn_label(train_years, test_year))
        out.insert(0, "input_set", input_set)
        out.insert(0, "target", target)
        return out

    # The selected configuration is refitted on the whole training block and frozen
    frozen = fit_model(sel, tuned[sel]["params"], factories[sel], Xtr, ytr, n_channels)
    sel_frame = incoming_frame(frozen, sel)
    preds = sel_frame.drop(columns=["model"])

    all_preds = None
    if save_all_model_predictions:
        # NOT used for selection; reliability of the selection, ensembles and the one-standard-error rule are computed later
        frames = [sel_frame]
        for name in tuned:
            if name == sel:
                continue
            try:
                fz = fit_model(name, tuned[name]["params"], factories[name], Xtr, ytr, n_channels)
                frames.append(incoming_frame(fz, name))
            except Exception as exc:  # noqa: BLE001
                print(f"      x {name} (incoming-season prediction): {str(exc)[:80]}")
        all_preds = pd.concat(frames, ignore_index=True)

    result = {"combo_id": combo_id, "target": target, "input_set": input_set,
              "transition": trn_label(train_years, test_year), "model_type": model_type, "selected": sel,
              "selected_params": tuned[sel]["params"], "selected_cv_rmse": float(tuned[sel]["cv_rmse"]),
              "n_train": int(tr_mask.sum()), "n_features": int(Xtr.shape[1]), "model_table": model_table,
              "predictions": preds, "all_model_predictions": all_preds,
              "oof_selected": pd.DataFrame({"rid": rec.loc[tr_mask, "rid"].to_numpy(), "y_true": ytr,
                                            "oof_pred": tuned[sel]["oof"]}),
              "frozen": frozen, "signature": sig}
    pickle_dump_atomic(result, ckpt)
    return result, False


def paired_bootstrap(levels, err_a, err_b, n_boot=2000, seed=0):
    """Difference = RMSEP(B) - RMSEP(A); the test MC levels are resampled."""
    d = pd.DataFrame({"l": levels, "a": np.asarray(err_a) ** 2, "b": np.asarray(err_b) ** 2})
    g = d.groupby("l").agg(sa=("a", "sum"), sb=("b", "sum"), n=("a", "size"))
    sa, sb, n = g["sa"].to_numpy(), g["sb"].to_numpy(), g["n"].to_numpy()
    point = float(np.sqrt(sb.sum() / n.sum()) - np.sqrt(sa.sum() / n.sum()))
    idx = np.random.default_rng(seed).integers(0, len(g), size=(n_boot, len(g)))
    diffs = np.sqrt(sb[idx].sum(1) / n[idx].sum(1)) - np.sqrt(sa[idx].sum(1) / n[idx].sum(1))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, float(lo), float(hi), len(g)


def comparison_table(corrected, n_boot=2000):
    test = corrected[~corrected["is_transfer"]].copy()
    rows = []

    def add(kind, ctx, a, b, ccol_a, ccol_b, label_a, label_b):
        m = a.merge(b, on="rid", suffixes=("_a", "_b"))
        if len(m) < 2:
            return
        ea, eb = m[ccol_a + "_a"] - m["y_true_a"], m[ccol_b + "_b"] - m["y_true_b"]
        pt, lo, hi, nl = paired_bootstrap(m["Level_Key_a"].to_numpy(), ea.to_numpy(), eb.to_numpy(), n_boot)
        rows.append({"comparison": kind, **ctx, "A": label_a, "B": label_b, "diff_RMSEP_B_minus_A": pt,
                     "ci95_low": lo, "ci95_high": hi, "N": len(m), "N_levels": nl})

    base = ["target", "input_set", "transition", "model_type", "Type"]
    # (a) slope-bias correction: corrected - uncorrected
    for keys, g in test.groupby(base, sort=False):
        ctx = dict(zip(base[:-1], keys[:-1]), cultivar=keys[-1])
        add("slope_bias_vs_none", ctx, g, g, "p_none", "p_slope_bias", "none", "slope_bias")
        add("bias_only_vs_none", ctx, g, g, "p_none", "p_bias_only", "none", "bias_only")
    # (b) cultivar-specific vs pooled
    for (tg, ins, tr, typ), g in test.groupby(["target", "input_set", "transition", "Type"], sort=False):
        spec_name = [k for k, v in CULTIVARS.items() if v == typ][0]
        a = g[g["model_type"] == spec_name]
        b = g[g["model_type"] == "Pooled"]
        for cname, col in CORRECTIONS.items():
            if cname == "bias_only":
                continue
            add("pooled_vs_specific", {"target": tg, "input_set": ins, "transition": tr, "model_type": "-",
                                       "cultivar": typ, "correction": cname}, a, b, col, col, spec_name, "Pooled")
    # (c) input sets
    pairs = [("spectra", "microwave"), ("microwave", "full")]
    for (tg, tr, mt, typ), g in test.groupby(["target", "transition", "model_type", "Type"], sort=False):
        for sa_, sb_ in pairs:
            a, b = g[g["input_set"] == sa_], g[g["input_set"] == sb_]
            if a.empty or b.empty:
                continue
            for cname, col in CORRECTIONS.items():
                if cname == "bias_only":
                    continue
                add("inputset", {"target": tg, "input_set": f"{sa_}->{sb_}", "transition": tr, "model_type": mt,
                                 "cultivar": typ, "correction": cname}, a, b, col, col, sa_, sb_)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Quick summary plots (the figures of the paper are made by paper_outputs.py)
# ----------------------------------------------------------------------------------------------
def make_summary_plots(metrics, out_dir, dpi=200):
    out_dir = Path(out_dir)
    for target in metrics["target"].unique():
        m = metrics[(metrics["target"] == target) & (metrics["correction"].isin(["none", "slope_bias"]))]
        iset = "full" if target == "moisture" else "microwave"
        d = m[m["input_set"] == iset].copy()
        d["spec"] = np.where(d["model_type"] == "Pooled", "Pooled", "Specific")
        d["variant"] = d["spec"] + " / " + d["correction"]
        d = d[(d["spec"] == "Pooled") | (d["model_type"].map(CULTIVARS) == d["cultivar"])]
        piv = d.pivot_table(index=["cultivar", "transition"], columns="variant", values="RMSEP")
        if not piv.empty:
            ax = piv.plot(kind="bar", figsize=(10, 5), width=0.8)
            ax.set_ylabel("RMSEP")
            ax.set_title(f"{target}: {iset} input set — specific vs pooled, unchanged vs slope-bias")
            ax.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / f"summary_{target}_specific_vs_pooled.png", dpi=dpi)
            plt.close("all")


# ----------------------------------------------------------------------------------------------
# Main workflow
# ----------------------------------------------------------------------------------------------
def run_chronological(out_root=RESULTS_ROOT / "chronological", targets=("moisture", "bulk_density"),
                      n_trials=DEFAULT_TRIALS, trials_by_model=None, models=None, resume=True, n_boot=2000,
                      save_all_model_predictions=True, data_root=DATA_ROOT, transitions=None, model_types=None,
                      cv_folds=None):
    """models: names of the models to use (e.g. the output of preflight_check); None -> 15 core + 6 extra models.
    cv_folds: number of folds of the grouped CV used for tuning (None -> CV_FOLDS = 5)."""
    cv_folds = cv_folds or CV_FOLDS
    out_dir = Path(out_root)
    for sub in ("checkpoints", "tuning", "tables"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    trials_by_model = {**DEFAULT_TRIALS_BY_MODEL, **(trials_by_model or {})}
    use_models, factories = resolve_models(models)
    skipped = [m for m in (models or ALL_MODEL_ORDER) if m not in use_models]
    header(f"CHRONOLOGICAL ANALYSIS v{CODE_VERSION} | output: {out_dir} | Optuna trials: {n_trials} "
           f"(per model: {trials_by_model}) | CV folds k={cv_folds}")
    print("  Core models :", [m for m in use_models if m in MODEL_ORDER])
    print("  Extra models:", [m for m in use_models if m in EXTRA_MODEL_ORDER])
    if skipped:
        print(f"  ! Models not available, skipped: {skipped}  ({ {k: UNAVAILABLE.get(k, '?')[:40] for k in skipped} })")
    rec, blocks = build_records(load_dataset("Pooled", data_root))
    transfer = select_transfer_levels(rec)
    tinfo = pd.DataFrame([{"Year": y, "Type": t, "transfer_levels": ";".join(v),
                           "n_transfer_records": int(rec["Level_Key"].isin(v).sum()),
                           "n_incoming_records": int(((rec["Year"] == y) & (rec["Type"] == t)).sum())}
                          for (y, t), v in sorted(transfer.items()) if y in (2015, 2016, 2017)])
    tinfo["n_test_records"] = tinfo["n_incoming_records"] - tinfo["n_transfer_records"]
    save_table(tinfo, out_dir / "tables" / "transfer_subsets")
    print("  Transfer / test records (paper: Ceren S2 10/84, Ciftci S2 10/75, S3 30/195, S4 30/75):")
    print(tinfo[["Year", "Type", "n_transfer_records", "n_test_records"]].to_string(index=False))

    combos = []
    for target in targets:
        for iset in TARGET_INPUTS[target]:
            for train_years, test_year in (transitions or TRANSITIONS):
                for mt in (model_types or MODEL_TYPES):
                    combos.append((target, iset, train_years, test_year, mt))
    print(f"  Combinations in total: {len(combos)} (27 MC + 18 BD = 45)")

    results, t_all, t_new, n_new = [], time.time(), 0.0, 0
    for i, (target, iset, ty, te, mt) in enumerate(combos, 1):
        t0 = time.time()
        try:
            res, from_ckpt = run_combo(out_dir, rec, blocks, transfer, factories, target, iset, ty, te, mt,
                                       use_models, n_trials, trials_by_model, resume, save_all_model_predictions,
                                       cv_folds)
            results.append(res)
            if not from_ckpt:
                t_new += time.time() - t0
                n_new += 1
            eta = (t_new / n_new) * (len(combos) - i) if n_new else 0
            tag = "✓ checkpoint" if from_ckpt else hms(time.time() - t0)
            print(f"  [{i:2d}/{len(combos)}] {res['combo_id']:<48} → {res['selected']:<12} cv_rmse={res['selected_cv_rmse']:.4f}  "
                  f"({tag}; remaining ~ {hms(eta)})")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i:2d}/{len(combos)}] {target}/{iset}/{trn_label(ty, te)}/{mt} ERROR: {str(exc)[:100]}")
            text_dump_atomic(f"{exc}\n\n{traceback.format_exc()}",
                             out_dir / "checkpoints" / f"FAILED_{target}__{iset}__{trn_label(ty, te)}__{mt}.txt")

    header("Results")
    if not results:
        print("  No results.")
        return None
    frozen_tbl = pd.DataFrame([{k: r[k] for k in ("combo_id", "target", "input_set", "transition", "model_type",
                                                   "selected", "selected_params", "selected_cv_rmse", "n_train", "n_features")}
                               for r in results])
    save_table(frozen_tbl, out_dir / "tables" / "frozen_models")
    save_table(pd.concat([r["model_table"].assign(combo_id=r["combo_id"]) for r in results], ignore_index=True),
               out_dir / "tables" / "model_cv_scores")
    preds = pd.concat([r["predictions"] for r in results], ignore_index=True)
    corrected = apply_corrections(preds)
    corrected.to_csv(out_dir / "tables" / "predictions_all.csv", index=False, encoding="utf-8-sig")
    metrics = test_metrics_table(corrected)
    save_table(metrics, out_dir / "tables" / "test_metrics")
    comps = comparison_table(corrected, n_boot)
    save_table(comps, out_dir / "tables" / "paired_comparisons")
    allm = [r["all_model_predictions"] for r in results if r.get("all_model_predictions") is not None]
    if allm:
        allpred = pd.concat(allm, ignore_index=True)
        allpred.to_csv(out_dir / "tables" / "all_model_predictions.csv", index=False, encoding="utf-8-sig")
        save_table(all_models_test_table(allpred), out_dir / "tables" / "all_models_test_scores")
    oof = [r["oof_selected"].assign(combo_id=r["combo_id"]) for r in results
           if r["target"] == "bulk_density" and r["transition"] == "S1+S2+S3toS4"]
    if oof:
        pd.concat(oof, ignore_index=True).to_csv(out_dir / "tables" / "bd_oof_residual_baseline_S1-S3.csv",
                                                 index=False, encoding="utf-8-sig")
    make_summary_plots(metrics, out_dir / "tables")

    show = metrics[(metrics["correction"].isin(["none", "slope_bias"])) & (metrics["input_set"] == "microwave")]
    piv = show.pivot_table(index=["target", "cultivar", "model_type", "transition"], columns="correction", values="RMSEP")
    print("  RMSEP (microwave input set):")
    print(piv.round(4).to_string())
    text_dump_atomic(
        f"CHRONOLOGICAL ANALYSIS SUMMARY (version {CODE_VERSION})\nDate: {now_str()}\nCombinations: {len(results)}/{len(combos)}\n"
        f"Models: {use_models}\nOptuna trials: {n_trials} (per model: {trials_by_model})\nCV folds (k): {cv_folds}\n"
        f"Bootstrap: {n_boot}\n"
        f"Skipped models: {skipped}\n\nTransfer subsets:\n{tinfo.to_string(index=False)}\n\n"
        f"Frozen models:\n{frozen_tbl.to_string(index=False)}\n", out_dir / "FINAL_REPORT.txt")
    print(f"\n  Done -> {out_dir}")
    print("  Tables and figures of the paper: python step2_paper_tables_figures.py")
    return {"frozen": frozen_tbl, "metrics": metrics, "comparisons": comps}
