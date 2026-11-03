# -*- coding: utf-8 -*-
"""
Candidate regression models of the chronological analysis.

Model definitions, Optuna search spaces and TabPFN versions of the 15 core models. The core model set and its search
spaces follow the earlier wheat study with the same instrument (Saçılık et al., 2026); chronological.py adds six further
models (PLS, PCR, MLP, GPR, AdaBoost-SVR, 1D-CNN). Target values are never modified.

preflight_check() fits every model once on small synthetic data and returns the models that run in this environment.
"""


from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

from common import FEATURES, disable_interactive_auth, get_device, hf_login_if_available  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
disable_interactive_auth()      # without a key, TabPFN must fail at once instead of waiting for interactive input

UNAVAILABLE = {}   # model name -> reason it cannot be used

try:
    import xgboost as xgb
except Exception as exc:  # noqa: BLE001
    xgb = None
    UNAVAILABLE["XGBoost"] = f"xgboost not available: {exc}"
try:
    from lightgbm import LGBMRegressor
except Exception as exc:  # noqa: BLE001
    LGBMRegressor = None
    UNAVAILABLE["LightGBM"] = f"lightgbm not available: {exc}"
try:
    from catboost import CatBoostRegressor
except Exception as exc:  # noqa: BLE001
    CatBoostRegressor = None
    UNAVAILABLE["CatBoost"] = f"catboost not available: {exc}"
try:
    from interpret.glassbox import ExplainableBoostingRegressor
except Exception as exc:  # noqa: BLE001
    ExplainableBoostingRegressor = None
    UNAVAILABLE["EBM"] = f"interpret not available: {exc}"
try:
    from tabpfn import TabPFNRegressor
except Exception as exc:  # noqa: BLE001
    TabPFNRegressor = None
    UNAVAILABLE["TabPFN_v2"] = f"tabpfn not available: {exc}"


# ----------------------------------------------------------------------------------------------
# Models and hyperparameter search spaces
# ----------------------------------------------------------------------------------------------
HYPERPARAMS = {
    "Ridge": {"alpha": {"type": "float", "range": (1e-4, 1e3), "log": True}},
    "Lasso": {"alpha": {"type": "float", "range": (1e-4, 1e3), "log": True}},
    "ElasticNet": {"alpha": {"type": "float", "range": (1e-4, 1e3), "log": True},
                   "l1_ratio": {"type": "float", "range": (0.0, 1.0)}},
    "KNN": {"n_neighbors": {"type": "categorical", "choices": list(range(3, 51, 2))},
            "weights": {"type": "categorical", "choices": ["uniform", "distance"]},
            "p": {"type": "categorical", "choices": [1, 2]}},
    "RandomForest": {"n_estimators": {"type": "int", "range": (100, 500)},
                     "max_depth": {"type": "int", "range": (5, 30)},
                     "min_samples_split": {"type": "int", "range": (2, 20)},
                     "min_samples_leaf": {"type": "int", "range": (1, 10)}},
    "ExtraTrees": {"n_estimators": {"type": "int", "range": (100, 500)},
                   "max_depth": {"type": "int", "range": (5, 30)},
                   "min_samples_split": {"type": "int", "range": (2, 20)},
                   "min_samples_leaf": {"type": "int", "range": (1, 10)}},
    "SVR_RBF": {"C": {"type": "float", "range": (1e-1, 1e3), "log": True},
                "epsilon": {"type": "float", "range": (1e-3, 1.0), "log": True},
                "gamma": {"type": "float", "range": (1e-4, 1e1), "log": True}},
    "XGBoost": {"n_estimators": {"type": "int", "range": (100, 500)},
                "max_depth": {"type": "int", "range": (3, 10)},
                "learning_rate": {"type": "float", "range": (1e-3, 0.3), "log": True},
                "subsample": {"type": "float", "range": (0.5, 1.0)},
                "colsample_bytree": {"type": "float", "range": (0.5, 1.0)}},
    "LightGBM": {"n_estimators": {"type": "int", "range": (100, 500)},
                 "max_depth": {"type": "int", "range": (3, 10)},
                 "learning_rate": {"type": "float", "range": (1e-3, 0.3), "log": True},
                 "num_leaves": {"type": "int", "range": (20, 100)},
                 "subsample": {"type": "float", "range": (0.5, 1.0)}},
    "CatBoost": {"iterations": {"type": "int", "range": (100, 500)},
                 "depth": {"type": "int", "range": (3, 10)},
                 "learning_rate": {"type": "float", "range": (1e-3, 0.3), "log": True}},
    "EBM": {"max_bins": {"type": "int", "range": (32, 256)},
            "max_leaves": {"type": "int", "range": (3, 10)},
            "learning_rate": {"type": "float", "range": (1e-3, 0.1), "log": True}},
}

MODEL_ORDER = ["LinearRegression", "Ridge", "Lasso", "ElasticNet", "KNN", "RandomForest", "ExtraTrees",
               "SVR_RBF", "XGBoost", "LightGBM", "CatBoost", "EBM", "TabPFN_v2", "TabPFN_v2_5", "TabPFN_v3"]

# TabPFN versions: name -> (tabpfn ModelVersion name, gated Hugging Face repository | None)
# v2 is openly licensed (no key needed). v2.5 and v3 are GATED: a Prior Labs key and acceptance of their licence are required.
TABPFN_VARIANTS = {
    "TabPFN_v2": ("V2", None),
    "TabPFN_v2_5": ("V2_5", "tabpfn_2_5"),
    "TabPFN_v3": ("V3", "tabpfn_3"),
}


def model_factories(device):
    f = {
        "LinearRegression": lambda **kw: LinearRegression(**kw),
        "Ridge": lambda **kw: Ridge(random_state=42, **kw),
        "Lasso": lambda **kw: Lasso(random_state=42, **kw),
        "ElasticNet": lambda **kw: ElasticNet(random_state=42, **kw),
        "KNN": lambda **kw: KNeighborsRegressor(**kw),
        "RandomForest": lambda **kw: RandomForestRegressor(random_state=42, n_jobs=-2, **kw),
        "ExtraTrees": lambda **kw: ExtraTreesRegressor(random_state=42, n_jobs=-2, **kw),
        "SVR_RBF": lambda **kw: SVR(**kw),
    }
    if xgb is not None:
        f["XGBoost"] = lambda **kw: xgb.XGBRegressor(random_state=42, n_jobs=-2, verbosity=0, **kw)
    if LGBMRegressor is not None:
        f["LightGBM"] = lambda **kw: LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1, **kw)
    if CatBoostRegressor is not None:
        f["CatBoost"] = lambda **kw: CatBoostRegressor(random_state=42, verbose=0, allow_writing_files=False, **kw)
    if ExplainableBoostingRegressor is not None:
        f["EBM"] = lambda **kw: ExplainableBoostingRegressor(random_state=42, **kw)
    if TabPFNRegressor is not None:
        try:
            from tabpfn.constants import ModelVersion
            known = {v.name for v in ModelVersion}
        except ImportError:                    # older tabpfn: default model only (v2)
            known = {"V2"}
        for name, (vname, _repo) in TABPFN_VARIANTS.items():
            if vname in known:
                f[name] = (lambda _v=vname, **kw: _make_tabpfn(device, _v, **kw))
            else:
                UNAVAILABLE[name] = f"ModelVersion.{vname} is not available in the installed tabpfn (pip install -U tabpfn)"
    return {k: f[k] for k in MODEL_ORDER if k in f}


def _make_tabpfn(device, version_name, **kw):
    """Selects the version EXPLICITLY (with 'auto', a newer tabpfn may pick a gated version)."""
    # ignore_pretraining_limits: avoids rejecting training sets above the pretraining limits on the CPU
    kw.setdefault("ignore_pretraining_limits", True)
    try:
        from tabpfn.constants import ModelVersion
    except ImportError:                       # older tabpfn: the default is already v2
        return TabPFNRegressor(device=device, **kw)
    if not hasattr(TabPFNRegressor, "create_default_for_version"):
        return TabPFNRegressor(device=device, **kw)
    return TabPFNRegressor.create_default_for_version(getattr(ModelVersion, version_name), device=device, **kw)


# ----------------------------------------------------------------------------------------------
# Preflight check with the real model classes
# ----------------------------------------------------------------------------------------------
def _try_model(factory, X, y):
    """Returns (success, note); the note is the actual error message."""
    t0 = time.time()
    try:
        model = factory()
        model.fit(X, y)
        assert np.isfinite(np.asarray(model.predict(X), dtype=float)).all()
        return True, f"{time.time() - t0:.1f}s"
    except Exception as exc:  # noqa: BLE001
        msg = " ".join(str(exc).split())              # collapse line breaks
        return False, f"[{type(exc).__name__}] {msg[:300]}"


def _show_key_form(ok, gated, device, X, y):
    """Shows a hidden password field in the notebook cell output (ipywidgets). When the key is entered, the gated TabPFN
    versions are tried again at once and the working ones are APPENDED to `ok` (the same list object, so the caller's
    list is updated). Returns (pw, btn, out)."""
    try:
        import ipywidgets as w
        from IPython.display import display
    except Exception:  # noqa: BLE001
        print("\n  (The key field in the notebook needs ipywidgets: run 'pip install ipywidgets' and restart the kernel.\n"
              "   Alternatively, set the TABPFN_TOKEN environment variable before starting Python; see README.md.)")
        return None
    pw = w.Password(placeholder="tabpfn_sk_...", description="API Key:", layout=w.Layout(width="560px"))
    persist = w.Checkbox(value=True, indent=False, layout=w.Layout(width="auto"),
                         description="Also save in my user profile (~/.cache/tabpfn/auth_token; kept after a kernel restart)")
    btn = w.Button(description="Save key and retry TabPFN v2.5 / v3", button_style="primary",
                   layout=w.Layout(width="auto"))
    out = w.Output()

    def on_click(_):
        with out:
            out.clear_output()
            token = pw.value.strip().strip("\"' \r\n\t")
            pw.value = ""                                  # clear the field at once
            if not token:
                print("The key is empty.")
                return
            os.environ["TABPFN_TOKEN"] = token
            if persist.value:
                p = Path.home() / ".cache" / "tabpfn" / "auth_token"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(token)
                print(f"Key saved: {p}")
            print("Trying (the model weights are downloaded on the first run; this can take a few minutes)...")
            factories = model_factories(device)
            for name in gated:
                success, note = _try_model(factories[name], X, y)
                print(f"  {name:<14}{'OK' if success else 'ERROR'}  {note}")
                if success and name not in ok:
                    ok.append(name)
            ok.sort(key=MODEL_ORDER.index)
            print("Available models:", ok)
            if any(n not in ok for n in gated):
                print("If a version still fails, check that its licence is accepted at https://ux.priorlabs.ai and that TABPFN_TOKEN is set.")

    btn.on_click(on_click)
    display(w.VBox([w.HTML("<b>Prior Labs API key for TabPFN v2.5 / v3</b> (ux.priorlabs.ai account page; not a Hugging Face token)"),
                    pw, persist, btn, out]))
    return pw, btn, out


def preflight_check(n_samples=300, device=None, verbose=True, ask_key=True, n_features=None, factories=None):
    """Fits and predicts every model once on small synthetic data and returns the list of models that work.
    Run it once BEFORE a long run to avoid surprises.
    ask_key=True: if the gated TabPFN versions (v2.5, v3) fail, a key field is shown in the notebook cell output.
    n_features: None -> 5 features; a number such as 205 -> wide input (chronological analysis)."""
    device = device or get_device()
    disable_interactive_auth()
    hf_login_if_available()
    rng = np.random.default_rng(0)
    cols = FEATURES if n_features is None else [f"x{i}" for i in range(n_features)]
    X = pd.DataFrame(rng.normal(size=(n_samples, len(cols))), columns=cols)
    y = 3 * X.iloc[:, 0] - 2 * X.iloc[:, 1] + rng.normal(scale=0.1, size=n_samples)
    rows, ok = [], []
    locked = []                                       # TabPFN versions that failed
    print(f"Device: {device} | checking the Python packages...")
    for name, factory in (factories or model_factories(device)).items():
        success, note = _try_model(factory, X, y)
        rows.append((name, "OK" if success else "ERROR", note))
        if success:
            ok.append(name)
        elif name in TABPFN_VARIANTS:
            locked.append(name)
    for name, why in UNAVAILABLE.items():
        rows.append((name, "MISSING", why[:90]))
    try:
        import openpyxl  # noqa: F401
        rows.append(("openpyxl (Excel)", "OK", ""))
    except Exception:  # noqa: BLE001
        rows.append(("openpyxl (Excel)", "MISSING", "pip install openpyxl; without it only CSV files are written"))
    if verbose:
        print(f"\n{'Model':<20}{'Status':<8}Note")
        print("-" * 70)
        for name, status, note in rows:
            print(f"{name:<20}{status:<8}{note}")
        print(f"\nAvailable models: {len(ok)}/{len(factories) if factories else len(MODEL_ORDER)}")
        if locked:
            print(f"\nTabPFN versions that failed: {locked}  (the actual error messages are listed above)")
            if "TabPFN_v2" in locked:
                print("  TabPFN_v2 is openly licensed and needs no key. If it fails, make sure the installed tabpfn version matches\n"
                      "  requirements.txt and that the Python session was restarted after installing it.")
            gated = [n for n in locked if TABPFN_VARIANTS[n][1]]
            if gated:
                print("  TabPFN v2.5 and v3 are GATED. To use them:\n"
                      "   1. Get an API key from the account page at https://ux.priorlabs.ai and set it as the environment variable\n"
                      "      TABPFN_TOKEN before starting Python\n"
                      "      (a Prior Labs key, not a Hugging Face token).")
                for n in gated:
                    repo = TABPFN_VARIANTS[n][1]
                    print(f"   2. Accept the {n} licence: https://ux.priorlabs.ai/accept-license?hf_repo_id={repo}")
                print("   3. If the Hugging Face repository is gated as well (the error says 'gated'): accept its terms at\n"
                      "      https://huggingface.co/Prior-Labs/<repository> and log in with 'hf auth login' or HF_TOKEN.\n"
                      "   If the licences are accepted and the error remains, check that the key is valid and not expired.\n"
                      "      (see https://docs.priorlabs.ai for the TabPFN licence and key settings)")
            print("  (Failed models are skipped; the others run normally. After fixing a model, run the preflight check and the\n"
                  "   analysis again: finished models are skipped and only the added ones are trained.)")
    gated_failed = [n for n in locked if TABPFN_VARIANTS[n][1]]
    if ask_key and gated_failed:
        try:
            from IPython import get_ipython
            in_notebook = get_ipython() is not None
        except Exception:  # noqa: BLE001
            in_notebook = False
        if in_notebook:
            _show_key_form(ok, gated_failed, device, X, y)
    return ok


# ----------------------------------------------------------------------------------------------
# Optuna parameter suggestion
# ----------------------------------------------------------------------------------------------
def _suggest(trial, spec):
    out = {}
    for name, info in spec.items():
        if info["type"] == "int":
            out[name] = trial.suggest_int(name, *info["range"])
        elif info["type"] == "float":
            out[name] = trial.suggest_float(name, *info["range"], log=info.get("log", False))
        else:
            out[name] = trial.suggest_categorical(name, info["choices"])
    return out
