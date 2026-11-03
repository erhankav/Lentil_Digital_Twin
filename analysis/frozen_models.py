# -*- coding: utf-8 -*-
"""
Frozen models and calibration versions.

No model is trained here: the configurations frozen in the chronological analysis are read from its checkpoints.
Calibration maintenance is the slope-bias correction fitted on the transfer records of the incoming season
(y_c = b0 + b1 * y); the fitting rule is the same as in chronological.apply_corrections.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from archive import CHECKPOINT_DIR
from chronological import feature_matrix  # noqa: E402
from common import apply_local_keys, pickle_load_safe  # noqa: E402

MC_INPUT_SET = "full"            # MC predictor of the twin: 287 inputs (including the rho_g-normalised terms)
BD_INPUT_SET = "microwave"       # dielectric BD predictor rho_d: 205 inputs, NO mass / rho_g


@dataclass(frozen=True)
class ModelKey:
    target: str          # "moisture" | "bulk_density"
    input_set: str       # "spectra" | "microwave" | "full"
    transition: str      # e.g. "S1+S2+S3toS4"
    model_type: str      # "Ceren" | "Ciftci" | "Pooled"

    @property
    def combo_id(self):
        return f"{self.target}__{self.input_set}__{self.transition}__{self.model_type}"


@lru_cache(maxsize=None)
def _load_checkpoint(combo_id):
    try:
        apply_local_keys()               # for TabPFN v2.5 / v3 checkpoints (silently skipped if there is no key file)
    except Exception:  # noqa: BLE001
        pass
    path = CHECKPOINT_DIR / f"{combo_id}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {combo_id}. Run step1_chronological_analysis.py first.")
    res = pickle_load_safe(path)
    if res is None:                      # the file exists but cannot be opened: show the actual reason (e.g. a model saved on CUDA)
        import pickle
        try:
            with open(path, "rb") as fh:
                res = pickle.load(fh)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"checkpoint could not be opened: {combo_id} ({type(e).__name__}: {str(e)[:200]})") from e
    move_tabpfn_to_device(res)
    return res


def tabpfn_estimators(obj, seen=None, depth=0):
    """TabPFN estimators in the object tree (tabpfn classes with an executor_ attribute)."""
    seen = set() if seen is None else seen
    if id(obj) in seen or depth > 8:
        return
    seen.add(id(obj))
    if type(obj).__module__.startswith("tabpfn") and hasattr(obj, "executor_"):
        yield obj
        return
    if isinstance(obj, dict):
        kids = obj.values()
    elif isinstance(obj, (list, tuple)):
        kids = obj
    else:
        kids = getattr(obj, "__dict__", {}).values()
    for v in kids:
        yield from tabpfn_estimators(v, seen, depth + 1)


def move_tabpfn_to_device(obj):
    """Moves the TabPFN estimators to the device of this computer: the GPU (CUDA) if available, otherwise the CPU.
    Estimators that are already on the right device are left untouched."""
    ests = list(tabpfn_estimators(obj))
    if not ests:
        return
    import torch
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for est in ests:
        cur = getattr(est, "devices_", None) or ()
        if cur and all(d.type == dev.type for d in cur):
            continue
        est.device, est.devices_ = dev.type, (dev,)
        ex = getattr(est, "executor_", None)
        if ex is not None and hasattr(ex, "to"):
            ex.to([dev], getattr(ex, "force_inference_dtype", None), getattr(ex, "dtype_byte_size", 4))


class FrozenPredictor:
    """Frozen S1-...-S(k) model. Its version ID is stored with the results."""

    def __init__(self, key: ModelKey):
        self.key = key
        res = _load_checkpoint(key.combo_id)
        self.frozen, self.selected = res["frozen"], res["selected"]
        self.cv_rmse = float(res["selected_cv_rmse"])
        self.oof = res["oof_selected"]                 # DataFrame: rid, y_true, oof_pred (training block)

    @property
    def version(self):
        return f"{self.key.combo_id}::{self.selected}"

    def predict(self, blocks):
        return self.frozen.predict(feature_matrix(blocks, self.key.input_set))


def fit_slope_bias(y_pred, y_true):
    """Same rule as chronological.apply_corrections: OLS, y_true ~ b0 + b1 * y_pred."""
    y_pred, y_true = np.asarray(y_pred, float), np.asarray(y_true, float)
    if len(y_pred) >= 2 and np.std(y_pred) > 0:
        b1, b0 = np.polyfit(y_pred, y_true, 1)
        return float(b0), float(b1)
    return 0.0, 1.0


@dataclass(frozen=True)
class Calibration:
    b0: float
    b1: float
    version: str
    fitted_on: tuple = ()

    def apply(self, y):
        return self.b0 + self.b1 * np.asarray(y, float)


IDENTITY = Calibration(0.0, 1.0, "none")
