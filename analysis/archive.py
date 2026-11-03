# -*- coding: utf-8 -*-
"""
Archive records for the digital shadow evaluation. The record table, the input blocks and the transfer levels are
produced with the functions of the chronological analysis (chronological.build_records / select_transfer_levels), so
the inputs replayed here are identical to those the frozen models saw in training.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from chronological import build_records, select_transfer_levels  # noqa: E402
from common import DATA_ROOT, RESULTS_ROOT, load_dataset  # noqa: E402

# Frozen models and predictions of the chronological analysis (step 1).
ANALYSIS_VERSION = "v2"                               # version of the chronological analysis (CODE_VERSION 2.0)
CHRONO_DIR = RESULTS_ROOT / "chronological"
CHECKPOINT_DIR = CHRONO_DIR / "checkpoints"
PREDICTIONS_CSV = CHRONO_DIR / "tables" / "predictions_all.csv"

INCOMING_YEARS = (2015, 2016, 2017)
CULTIVAR_MODEL = {"CEREN": "Ceren", "CIFTCI": "Ciftci"}      # Type -> cultivar-specific model_type


@lru_cache(maxsize=1)
def load_archive():
    """(rec, blocks, transfer). Cached: the returned objects must NOT be modified (take a copy).

    Extra columns of rec: is_transfer (meaningful for incoming seasons only), level_mc (mean MC of the level; sort key).
    """
    rec, blocks = build_records(load_dataset("Pooled", DATA_ROOT))
    transfer = select_transfer_levels(rec)
    rec = rec.copy()
    rec["is_transfer"] = [bool(y in INCOMING_YEARS and lk in transfer[(y, t)])
                          for y, t, lk in zip(rec["Year"], rec["Type"], rec["Level_Key"])]
    rec["level_mc"] = rec.groupby("Level_Key")["MC"].transform("mean")
    for b in blocks.values():
        b.setflags(write=False)
    return rec, blocks, transfer


def subset_blocks(blocks, rows):
    """Writable copies of the blocks for the selected rows."""
    rows = np.asarray(rows)
    return {k: np.array(v[rows], dtype=float) for k, v in blocks.items()}


def measurement_order(rec):
    """Measurement order of the archive: season -> cultivar -> MC level -> filling -> packing state.
    The archive has no time stamps; the order of the cultivars within a day is unknown, so they are taken as consecutive blocks."""
    return rec.sort_values(["Year", "Type", "Level_ID", "Filling_ID", "Packing_State"], kind="stable")


def mc_bd_order(rec):
    """Ordering of Section 2.3.2: ascending by MC level (level mean), then by rho_g."""
    return rec.sort_values(["level_mc", "Level_Key", "BD"], kind="stable")
