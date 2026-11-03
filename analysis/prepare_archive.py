# -*- coding: utf-8 -*-
"""
Step 0 (provenance): builds data/lentil_microwave_archive.csv from the raw archive files of the measurement machine.

The raw files (one Excel file per cultivar and season, CEREN_2014.xlsx ... CIFTCI_2017.xlsx) are not part of the
repository; the merged CSV produced by this script is. The script documents how the CSV was made and can rebuild it
from the raw files:

    python prepare_archive.py <folder with the raw .xlsx files>

Raw files: 41 rows per record (2000-4000 MHz in 50 MHz steps) and eight columns, matched by the start of their
headers (MC is labelled "Nem", Turkish for moisture, in the raw files). Target values are copied unchanged.
The archive stores no level, filling or packing-state indices; they are reconstructed from the record order
(Section 2.5 of the paper):
  * a new filling starts where the bulk density drops relative to the preceding record;
  * the packing state is the position of the record within its filling (1-5);
  * fillings are assigned to MC levels in order: one filling per level in 2014-2015, three in 2016-2017.

The merged file used in the analysis was stored as an Excel workbook before it was exported to the published CSV.
Rebuilding from the raw files therefore gives identical values in every column except the loss tangent, which can
differ in the last binary digit (at most 6e-17). The loss tangent is not an input of the models.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import ARCHIVE_COLUMNS, ARCHIVE_FILE, DATA_ROOT

FREQS = np.arange(2000, 4001, 50)    # 41 frequencies, 50 MHz steps
N_FREQ = len(FREQS)
CULTIVARS = ("CEREN", "CIFTCI")      # order of the merged file: Ceren (green), then Ciftci (red)

# Fillings per MC level: one filling in 2014-2015, three fillings in 2016-2017 (five packing states each).
FILLINGS_PER_LEVEL = {2014: 1, 2015: 1, 2016: 3, 2017: 3}

# Columns are matched by the start of their header (robust to spaces and units)
PREFIX_MAP = [("freq", "Freq"), ("nem", "Moisture_Content"), ("bulk", "Bulk_Density"),
              ("phase", "Phase_Shift"), ("atten", "Attenuation"),
              ("e1", "E1"), ("e2", "E2"), ("tanloss", "tanLoss")]
RAW_COLS = ["Freq", "Moisture_Content", "Bulk_Density", "Phase_Shift",
            "Attenuation", "E1", "E2", "tanLoss"]
STRUCT_COLS = ["Type", "Year", "Level_Key", "Level_ID", "Filling_ID", "Packing_State", "Sample_ID"]


def standardize_columns(df):
    """Matches the columns by name; stops if anything unexpected is found."""
    mapping = {}
    for col in df.columns:
        key = str(col).strip().lower()
        hits = [new for prefix, new in PREFIX_MAP if key.startswith(prefix)]
        if len(hits) != 1:
            raise ValueError(f"column not recognised: {col!r}")
        mapping[col] = hits[0]
    df = df.rename(columns=mapping)
    if sorted(df.columns) != sorted(RAW_COLS):
        raise ValueError(f"expected columns not found: {list(df.columns)}")
    for c in RAW_COLS:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"{c} is not numeric")
    return df[RAW_COLS]


def add_structure(df, year):
    """Each block of 41 rows is one packing state (2000..4000 MHz).
    Filling boundary: the bulk density drops relative to the preceding record. Target columns are not modified."""
    n = len(df)
    if n % N_FREQ:
        raise ValueError(f"{n} rows, not a multiple of {N_FREQ}")
    if not (df["Freq"].to_numpy() == np.tile(FREQS, n // N_FREQ)).all():
        raise ValueError("the frequency order is not 2000..4000 (blocks of 41)")
    dens = df["Bulk_Density"].to_numpy()[::N_FREQ]
    filling = np.cumsum(np.r_[True, dens[1:] < dens[:-1]])            # 1, 2, 3, ...
    state = pd.Series(filling).groupby(filling).cumcount().to_numpy() + 1
    level = (filling - 1) // FILLINGS_PER_LEVEL[year] + 1
    sample = np.arange(1, len(dens) + 1)
    out = df.copy()
    for name, arr in [("Sample_ID", sample), ("Filling_ID", filling),
                      ("Level_ID", level), ("Packing_State", state)]:
        out[name] = np.repeat(arr, N_FREQ)
    return out


def read_one(path):
    type_name, year = path.stem.split("_")      # CEREN_2014.xlsx -> CEREN, 2014
    year = int(year)
    df = standardize_columns(pd.read_excel(path).dropna(how="all"))
    df = add_structure(df.reset_index(drop=True), year)
    df.insert(0, "Year", year)
    df.insert(0, "Type", type_name)
    df["Level_Key"] = f"{type_name}_{year}_L" + df["Level_ID"].astype(str).str.zfill(2)
    return df


def main(raw_dir, out_file=DATA_ROOT / ARCHIVE_FILE):
    raw_dir = Path(raw_dir)
    parts = []
    for cultivar in CULTIVARS:
        files = sorted(raw_dir.rglob(f"{cultivar}_20??.xlsx"))
        if not files:
            raise FileNotFoundError(f"no {cultivar}_YYYY.xlsx files in {raw_dir}")
        for p in files:
            part = read_one(p)
            print(f"{p.name:18s} rows={len(part):6d}  records={part['Sample_ID'].nunique():4d}  "
                  f"levels={part['Level_ID'].nunique():3d}  fillings={part['Filling_ID'].nunique():3d}")
            parts.append(part)
    df = pd.concat(parts, ignore_index=True)[STRUCT_COLS + RAW_COLS]
    df = df.rename(columns={v: k for k, v in ARCHIVE_COLUMNS.items()})
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False, encoding="utf-8", lineterminator="\n")
    print(f"written: {out_file}  ({len(df):,} rows, {len(df) // N_FREQ} records)")
    return df


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python prepare_archive.py <folder with the raw .xlsx files>")
    main(sys.argv[1])
