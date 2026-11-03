# -*- coding: utf-8 -*-
"""
Common helpers of the analysis: constants, data loading, timing, device selection and crash-safe file writing.
Target variables (Moisture_Content, Bulk_Density) are never modified.
"""
from __future__ import annotations

import math
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]   # repository root (this file is in analysis/)
DATA_ROOT = REPO_ROOT / "data"                    # merged archive: data/lentil_microwave_archive.csv
RESULTS_ROOT = REPO_ROOT / "results"              # outputs of the analysis steps
ARCHIVE_FILE = "lentil_microwave_archive.csv"
DATASETS = ("Pooled",)                            # one file with both cultivars
FEATURES = ["Phase_Shift", "Attenuation", "E1", "E2", "tanLoss"]   # measured + derived features
FREQS = np.arange(2000, 4001, 50)

REQUIRED_COLS = ["Type", "Year", "Level_Key", "Level_ID", "Filling_ID", "Packing_State", "Sample_ID",
                 "Freq", "Moisture_Content", "Bulk_Density", "Phase_Shift", "Attenuation", "E1", "E2", "tanLoss"]
# column names of the published CSV -> column names used in the code (in the same order)
ARCHIVE_COLUMNS = {"Cultivar": "Type", "Year": "Year", "Level_Key": "Level_Key", "Level_ID": "Level_ID",
                   "Filling_ID": "Filling_ID", "Packing_State": "Packing_State", "Record_No": "Sample_ID",
                   "Frequency_MHz": "Freq", "MC_wb_percent": "Moisture_Content", "Bulk_Density_g_cm3": "Bulk_Density",
                   "Phase_Shift_deg": "Phase_Shift", "Attenuation_dB": "Attenuation", "Eps_Real": "E1",
                   "Eps_Imag": "E2", "Loss_Tangent": "tanLoss"}


# ----------------------------------------------------------------------------------------------
# General helpers
# ----------------------------------------------------------------------------------------------
def apply_local_keys():
    """Copies the keys of the optional local file local_keys.py (never committed) to environment variables; values in the
    file take precedence. Does nothing if the file does not exist or is empty. Never prints a key."""
    try:
        import local_keys as keys
    except Exception:  # noqa: BLE001
        return False
    applied = False
    tab = str(getattr(keys, "TABPFN_API_KEY", "") or "").strip().strip("\"' ")
    if tab:
        os.environ["TABPFN_TOKEN"] = tab
        applied = True
    hf = str(getattr(keys, "HF_TOKEN", "") or "").strip().strip("\"' ")
    if hf:
        os.environ["HF_TOKEN"] = hf
        applied = True
    return applied


def disable_interactive_auth():
    """Without a valid key, TabPFN starts a browser / terminal login and WAITS for input, which blocks a notebook cell
    forever. This makes it fail at once so that the model is skipped. Also applies the optional local key file (local_keys.py)."""
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    apply_local_keys()


def hf_login_if_available():
    """Hugging Face login for the TabPFN weights. The token is never written in code; it is read from HF_TOKEN."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        return False
    try:
        from huggingface_hub import login
        login(token=token)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"Hugging Face login failed: {exc}")
        return False


def hms(seconds):
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return "N/A"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m:02d}m {s:02d}s"
    return f"{s:02d}s"


def header(txt):
    print("\n" + "=" * 100)
    print(f"  {txt}")
    print("=" * 100)


def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def save_table(df, path_without_ext):
    """Always writes CSV, and Excel in addition if openpyxl is available."""
    path_without_ext = str(path_without_ext)
    df.to_csv(path_without_ext + ".csv", index=False, encoding="utf-8-sig")
    try:
        df.to_excel(path_without_ext + ".xlsx", index=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  (Excel file not written, CSV only: {exc})")


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def load_dataset(name, root=DATA_ROOT):
    """Reads the merged archive data/lentil_microwave_archive.csv with the column names used in the code."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset: {name!r}; options: {DATASETS}")
    path = Path(root) / ARCHIVE_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path, float_precision="round_trip").rename(columns=ARCHIVE_COLUMNS)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    return df.reset_index(drop=True)


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------------------------------------------------------------------------
# Crash-safe file writing: write to a temporary file first, then move it into place.
# If the power fails during writing, the old file stays intact; a half-written file never appears.
# ----------------------------------------------------------------------------------------------
def _atomic_write(path, data: bytes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def pickle_dump_atomic(obj, path):
    import pickle
    _atomic_write(path, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def pickle_load_safe(path):
    """Returns None if the file does not exist or is damaged (it is then recomputed)."""
    import pickle
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception:  # noqa: BLE001
        return None


def json_dump_atomic(obj, path):
    import json
    _atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=1, default=_json_default).encode("utf-8"))


def text_dump_atomic(text, path):
    _atomic_write(path, text.encode("utf-8"))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


warnings.filterwarnings("ignore")
