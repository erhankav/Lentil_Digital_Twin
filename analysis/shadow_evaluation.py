# -*- coding: utf-8 -*-
"""
DIGITAL SHADOW EVALUATION (offline, on the archive) - Section 2.3.2 of the paper

1. Baseline     : mu0, sigma0 = out-of-fold residuals of the selected S1-S3 BD model (microwave, 205 inputs, no mass),
                  r = rho_g - rho_d, for each cultivar. The dependence of r on MC and rho_g is examined BEFORE the limits.
2. Fault-free   : chart settings fixed from S1-S3; the S4 test records (transfer levels excluded) are monitored as one
                  sequence ordered by MC level and rho_g -> alarm rate of every chart and of the combined rule.
3. Perturbation : offsets added to copies of the S4 test records (dA 0.25/0.5/1 dB; dphi 2/5/10 degrees; rho_g
                  0.005/0.010/0.020 g cm-3). eps', eps'', X and the rho_g-normalised terms are recomputed; the slope-bias
                  corrected S1-S3 models are applied. 3000 sequences per cultivar and condition: 20 fault-free + 10
                  perturbed records (drawn with replacement). Alarm rate in the fault-free part; for the remaining
                  sequences, detection within the first 5 records and the delay to the first alarm.
4. MC effect    : mean change in predicted MC caused by each offset.

Primary analysis: cultivar-specific models, slope-bias correction, S1-S3 OOF baseline, MC-rho_g ordering.
Sensitivity     : pooled models; uncorrected rho_d; re-baselining on the S4 transfer records; measurement order.
The offsets are test conditions, NOT detection limits. Perturbed records are synthetic; no command is issued or executed.

Run: python step3_digital_shadow_evaluation.py   (needs the checkpoints of step 1)
Output: results/digital_shadow/ with tables/, figures/, SUMMARY_digital_shadow.txt and manifest.json
"""
from __future__ import annotations

import hashlib
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from dielectric import OFFSET_GRID, OFFSET_UNITS, offset_label, perturb  # noqa: E402
from frozen_models import (BD_INPUT_SET, IDENTITY, MC_INPUT_SET, Calibration, FrozenPredictor,  # noqa: E402
                           ModelKey, fit_slope_bias)
from spc import CHARTS, SpcSettings, run_sequences  # noqa: E402
from archive import (CULTIVAR_MODEL, CHRONO_DIR, ANALYSIS_VERSION, PREDICTIONS_CSV, load_archive,  # noqa: E402
                     mc_bd_order, measurement_order, subset_blocks)
from common import (ARCHIVE_FILE, DATA_ROOT, RESULTS_ROOT, json_dump_atomic, now_str, save_table,  # noqa: E402
                    text_dump_atomic)

TRANSITION = "S1+S2+S3toS4"
TEST_YEAR = 2017
N_SEQ, N_CLEAN, N_PERT, DETECT_WITHIN = 3000, 20, 10, 5
SEED = 0
SETTINGS = SpcSettings()
CULTIVARS = ("CEREN", "CIFTCI")
CULTIVAR_LABEL = {"CEREN": "Ceren (green)", "CIFTCI": "Çiftçi (red)"}
SCOPES = ("specific", "pooled")
PRIMARY = {"scope": "specific", "correction": "slope_bias", "baseline": "S1-S3_OOF", "ordering": "MC_then_rho_g"}
CONDITIONS = [("clean", 0.0)] + [(k, v) for k, vals in OFFSET_GRID.items() for v in vals]
ILLUSTRATION = ("rho_g", 0.010, 51)      # illustrative figure: +0.010 g cm-3 from the 51st record on (synthetic)

# Colours: first four categorical colours of the reference palette (fixed order; checked in light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CULT_COLOR = {"CEREN": SERIES[0], "CIFTCI": SERIES[1]}
INK, INK2, GRID, ALARM = "#0b0b0b", "#52514e", "#e4e3df", "#0b0b0b"
CHART_LABEL = {"shewhart": "Shewhart", "ewma": "EWMA", "cusum": "CUSUM", "combined": "Combined rule"}
CHART_MARKER = {"shewhart": "o", "ewma": "s", "cusum": "^", "combined": "D"}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------------------------------
def residual_dependence(r, mc, rho, sigma0):
    X = np.column_stack([np.ones_like(mc), mc, rho])
    coef, *_ = np.linalg.lstsq(X, r, rcond=None)
    fit = X @ coef
    r2 = 1.0 - np.sum((r - fit) ** 2) / np.sum((r - r.mean()) ** 2)
    s_mc, s_rho = spearmanr(mc, r), spearmanr(rho, r)
    return {"b_MC": coef[1], "b_rho_g": coef[2], "R2_MC_rho_g": r2,
            "dz_per_1pct_MC": coef[1] / sigma0, "dz_per_0.01_rho_g": coef[2] * 0.01 / sigma0,
            "spearman_MC": s_mc.statistic, "spearman_MC_p": s_mc.pvalue,
            "spearman_rho_g": s_rho.statistic, "spearman_rho_g_p": s_rho.pvalue}


def ordered_stats(z):
    out = run_sequences(z[None, :], SETTINGS)
    rows = [{"chart": "warning_2sigma", "n_records": len(z), "n_alarm": int(out["warn"].sum()),
             "share_alarm": float(out["warn"].mean()),
             "first_alarm_record": int(np.argmax(out["warn"][0])) + 1 if out["warn"].any() else np.nan}]
    for ch in CHARTS:
        a = out[ch][0]
        rows.append({"chart": ch, "n_records": len(a), "n_alarm": int(a.sum()), "share_alarm": float(a.mean()),
                     "first_alarm_record": int(np.argmax(a)) + 1 if a.any() else np.nan})
    return rows, out


def sequence_stats(Zc, Zp):
    out = run_sequences(np.hstack([Zc, Zp]), SETTINGS)
    rows = []
    for ch in CHARTS:
        a = out[ch]
        fa = a[:, :Zc.shape[1]].any(axis=1)
        post = a[:, Zc.shape[1]:]
        det = post.any(axis=1)
        delay = np.where(det, post.argmax(axis=1) + 1, np.nan)
        ok = ~fa
        n_ok = int(ok.sum())
        dl = delay[ok & det]
        rows.append({"chart": ch, "n_seq": len(a), "p_false_alarm_fault_free_part": float(fa.mean()),
                     "n_evaluated": n_ok,
                     "p_detect_within_5": float(np.sum(ok & (delay <= DETECT_WITHIN)) / n_ok) if n_ok else np.nan,
                     "p_detect_within_10": float(np.sum(ok & det) / n_ok) if n_ok else np.nan,
                     "median_delay": float(np.median(dl)) if len(dl) else np.nan,
                     "mean_delay": float(np.mean(dl)) if len(dl) else np.nan})
    return rows


# ----------------------------------------------------------------------------------------------
# All computations for one (scope, cultivar)
# ----------------------------------------------------------------------------------------------
def evaluate_cultivar(rec, blocks, scope, typ, idx, preds_ref):
    mt = CULTIVAR_MODEL[typ] if scope == "specific" else "Pooled"
    bd = FrozenPredictor(ModelKey("bulk_density", BD_INPUT_SET, TRANSITION, mt))
    mc = FrozenPredictor(ModelKey("moisture", MC_INPUT_SET, TRANSITION, mt))

    # 1. Baseline: S1-S3 OOF residuals (records of this cultivar)
    oof = bd.oof.merge(rec[["rid", "Type", "MC", "BD", "Level_Key", "Year"]], on="rid")
    oof = oof[oof["Type"] == typ]
    if not np.allclose(oof["y_true"], oof["BD"]):
        raise RuntimeError("OOF y_true does not match rho_g")
    r_oof = (oof["y_true"] - oof["oof_pred"]).to_numpy()
    mu0, sd0 = float(r_oof.mean()), float(r_oof.std(ddof=1))
    base_row = {"scope": scope, "cultivar": typ, "bd_model": bd.version, "mc_model": mc.version,
                "n_records": len(r_oof), "n_levels": oof["Level_Key"].nunique(), "mu0": mu0, "sigma0": sd0,
                "bd_cv_rmse": bd.cv_rmse}
    dep_row = {"scope": scope, "cultivar": typ, "n_records": len(r_oof),
               **residual_dependence(r_oof, oof["MC"].to_numpy(float), oof["BD"].to_numpy(float), sd0)}
    oof_points = oof.assign(r=r_oof, scope=scope)[["scope", "Type", "Year", "Level_Key", "MC", "BD", "r"]]

    # 2. S4 records: transfer (calibration) and test (evaluation)
    s4 = rec[(rec["Year"] == TEST_YEAR) & (rec["Type"] == typ)]
    tr = s4[s4["is_transfer"]]
    te = mc_bd_order(s4[~s4["is_transfer"]])
    b_tr = subset_blocks(blocks, tr["rid"].to_numpy())
    cal = {}
    for name, model in (("bd", bd), ("mc", mc)):
        y_true = tr["BD"] if name == "bd" else tr["MC"]
        b0, b1 = fit_slope_bias(model.predict(b_tr), y_true)
        cal[name] = Calibration(b0, b1, f"slope_bias::{model.key.combo_id}::{typ}::S4_transfer",
                                tuple(sorted(tr["Level_Key"].unique())))
    rhod_tr_raw = bd.predict(b_tr)

    # Check: identical to the corrected predictions of the chronological analysis (replay fidelity)
    ref = preds_ref[(preds_ref.target == "bulk_density") & (preds_ref.input_set == BD_INPUT_SET)
                    & (preds_ref.transition == TRANSITION) & (preds_ref.model_type == mt) & (preds_ref.Type == typ)]
    ref = ref.set_index("rid").loc[te["rid"]]

    b_te = subset_blocks(blocks, te["rid"].to_numpy())
    rho_te = te["BD"].to_numpy(float)
    cond_pred = {}
    for kind, val in CONDITIONS:
        nb, nrho = perturb(b_te, rho_te, "dA" if kind == "clean" else kind, val)   # clean: zero offset = identity
        cond_pred[offset_label(kind, val)] = {"kind": kind, "value": val, "rho_g": nrho,
                                              "rhod_raw": bd.predict(nb), "mc_raw": mc.predict(nb)}
    clean = cond_pred["clean"]
    fid = np.abs(cal["bd"].apply(clean["rhod_raw"]) - ref["p_slope_bias"].to_numpy()).max()
    if fid > 1e-6:
        raise RuntimeError(f"replay fidelity broken ({scope}/{typ}): {fid:.2e}")

    corrections = {"slope_bias": cal, "none": {"bd": IDENTITY, "mc": IDENTITY}}
    r_tr_sb = tr["BD"].to_numpy(float) - cal["bd"].apply(rhod_tr_raw)
    baselines_by_corr = {"slope_bias": {"S1-S3_OOF": (mu0, sd0),
                                        "S4_transfer": (float(r_tr_sb.mean()), float(r_tr_sb.std(ddof=1)))},
                         "none": {"S1-S3_OOF": (mu0, sd0)}}
    base_row.update({"mu0_S4_transfer": baselines_by_corr["slope_bias"]["S4_transfer"][0],
                     "sigma0_S4_transfer": baselines_by_corr["slope_bias"]["S4_transfer"][1],
                     "n_transfer_records": len(tr), "n_test_records": len(te),
                     "cal_bd_b0": cal["bd"].b0, "cal_bd_b1": cal["bd"].b1,
                     "cal_mc_b0": cal["mc"].b0, "cal_mc_b1": cal["mc"].b1})

    order_meas = te.index.get_indexer(measurement_order(te).index)     # te (ordered by MC and rho_g) -> measurement order

    ordered_rows, seq_rows, effect_rows, traces = [], [], [], []
    for cname, cals in corrections.items():
        for bname, (mu, sd) in baselines_by_corr[cname].items():
            z = {lab: (cp["rho_g"] - cals["bd"].apply(cp["rhod_raw"]) - mu) / sd for lab, cp in cond_pred.items()}
            ctx = {"scope": scope, "cultivar": typ, "correction": cname, "baseline": bname}
            for oname, perm in (("MC_then_rho_g", np.arange(len(te))), ("measurement_order", order_meas)):
                rows, out = ordered_stats(z["clean"][perm])
                ordered_rows += [{**ctx, "ordering": oname, **r} for r in rows]
                if (scope, cname, bname, oname) == (PRIMARY["scope"], PRIMARY["correction"], PRIMARY["baseline"],
                                                    PRIMARY["ordering"]):
                    st = out["stats"]
                    traces.append(pd.DataFrame({
                        "cultivar": typ, "position": np.arange(1, len(te) + 1), "rid": te["rid"].to_numpy(),
                        "Level_Key": te["Level_Key"].to_numpy(), "MC_ref": te["MC"].to_numpy(),
                        "rho_g": rho_te, "rho_d": cals["bd"].apply(clean["rhod_raw"]),
                        "MC_pred": cals["mc"].apply(clean["mc_raw"]), "z": st["z"][0], "ewma": st["ewma"][0],
                        "ewma_limit": st["ewma_limit"][0], "cusum_pos": st["cpos"][0], "cusum_neg": st["cneg"][0],
                        "warn": out["warn"][0], **{f"alarm_{ch}": out[ch][0] for ch in CHARTS}}))
            for lab, cp in cond_pred.items():
                Zc, Zp = z["clean"][idx[:, :N_CLEAN]], z[lab][idx[:, N_CLEAN:]]
                seq_rows += [{**ctx, "condition": lab, "kind": cp["kind"], "value": cp["value"],
                              "unit": OFFSET_UNITS.get(cp["kind"], ""), **r} for r in sequence_stats(Zc, Zp)]
            if (cname, bname) == ("slope_bias", "S1-S3_OOF"):
                for lab, cp in cond_pred.items():
                    if lab == "clean":
                        continue
                    d_rhod = cals["bd"].apply(cp["rhod_raw"]) - cals["bd"].apply(clean["rhod_raw"])
                    d_r = (cp["rho_g"] - rho_te) - d_rhod
                    d_mc = cals["mc"].apply(cp["mc_raw"]) - cals["mc"].apply(clean["mc_raw"])
                    effect_rows.append({"scope": scope, "cultivar": typ, "condition": lab, "kind": cp["kind"],
                                        "value": cp["value"], "unit": OFFSET_UNITS[cp["kind"]],
                                        "mean_d_rho_d": d_rhod.mean(), "mean_d_r": d_r.mean(),
                                        "mean_d_z": d_r.mean() / sd, "sd_d_z": d_r.std(ddof=1) / sd,
                                        "mean_d_MC": d_mc.mean(), "sd_d_MC": d_mc.std(ddof=1),
                                        "min_d_MC": d_mc.min(), "max_d_MC": d_mc.max(),
                                        "mean_d_MC_uncorrected": (cp["mc_raw"] - clean["mc_raw"]).mean()})
    illus = None
    if scope == PRIMARY["scope"]:
        kind, val, onset = ILLUSTRATION
        zc = (clean["rho_g"] - cal["bd"].apply(clean["rhod_raw"]) - mu0) / sd0
        cp = cond_pred[offset_label(kind, val)]
        zp = (cp["rho_g"] - cal["bd"].apply(cp["rhod_raw"]) - mu0) / sd0
        zi = np.where(np.arange(len(te)) + 1 >= onset, zp, zc)
        illus = (typ, zi, onset, run_sequences(zi[None, :], SETTINGS))
    return {"baseline": base_row, "dependence": dep_row, "oof_points": oof_points, "ordered": ordered_rows,
            "sequences": seq_rows, "effects": effect_rows, "traces": traces, "illustration": illus,
            "level_breaks": np.flatnonzero(te["Level_Key"].to_numpy()[1:] != te["Level_Key"].to_numpy()[:-1]) + 1.5}


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
def _style():
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 300,
                         "figure.dpi": 110, "font.family": "DejaVu Sans"})


def _legend_if_any(ax):
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper left", fontsize=7.5)


def _save(fig, path):
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_dependence(oofp, base, out):
    seasons = {2014: ("S1", SERIES[0]), 2015: ("S2", SERIES[1]), 2016: ("S3", SERIES[2])}
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharey="row")
    for i, typ in enumerate(CULTIVARS):
        d = oofp[(oofp.scope == "specific") & (oofp.Type == typ)]
        sd0 = base[(base.scope == "specific") & (base.cultivar == typ)]["sigma0"].iloc[0]
        for j, (col, xl) in enumerate((("MC", "Reference MC (% w.b.)"), ("BD", r"$\rho_g$ (g cm$^{-3}$)"))):
            ax = axes[i, j]
            for yr, (lab, c) in seasons.items():
                s = d[d.Year == yr]
                ax.scatter(s[col], s["r"], s=10, color=c, alpha=0.75, linewidths=0, label=lab)
            ax.axhline(0, color=INK2, lw=0.8)
            for k in (-3, 3):
                ax.axhline(k * sd0, color=INK2, lw=0.8, ls="--")
            ax.set_xlabel(xl)
            if j == 0:
                ax.set_ylabel(f"{CULTIVAR_LABEL[typ]}\n" + r"$r=\rho_g-\rho_d$ (g cm$^{-3}$)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Out-of-fold residuals of the S1–S3 dielectric BD models (dashed: ±3σ₀)", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.legend(h, lab, title="Training season", ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               fontsize=7.5, title_fontsize=7.5)
    _save(fig, out / "fig_residual_dependence")


def _chart_panels(axes, z, ewma, ewma_lim, cpos, cneg, alarms, color, breaks, onset=None):
    x = np.arange(1, len(z) + 1)
    ax = axes[0]
    ax.plot(x, z, color=color, lw=1.2)
    for k, ls in ((2, ":"), (3, "--")):
        ax.axhline(k, color=INK2, lw=0.8, ls=ls)
        ax.axhline(-k, color=INK2, lw=0.8, ls=ls)
    a = alarms["shewhart"]
    ax.scatter(x[a], z[a], s=22, marker="x", color=ALARM, zorder=3, label="alarm")
    ax.set_ylabel("z")
    ax = axes[1]
    ax.plot(x, ewma, color=color, lw=1.2)
    ax.plot(x, ewma_lim, color=INK2, lw=0.8, ls="--")
    ax.plot(x, -ewma_lim, color=INK2, lw=0.8, ls="--")
    a = alarms["ewma"]
    ax.scatter(x[a], ewma[a], s=22, marker="x", color=ALARM, zorder=3)
    ax.set_ylabel("EWMA")
    ax = axes[2]
    ax.plot(x, cpos, color=color, lw=1.2, label="C⁺")
    ax.plot(x, cneg, color=color, lw=1.2, ls=(0, (3, 2)), label="C⁻")
    ax.axhline(SETTINGS.h, color=INK2, lw=0.8, ls="--")
    a = alarms["cusum"]
    ax.scatter(x[a], np.maximum(cpos, cneg)[a], s=22, marker="x", color=ALARM, zorder=3)
    ax.set_ylabel("CUSUM")
    ax.set_xlabel("S4 test record (ordered by MC level, then ρg)")
    for ax in axes:
        for b in breaks:
            ax.axvline(b, color=GRID, lw=0.8, zorder=0)
        if onset is not None:
            ax.axvline(onset - 0.5, color=INK, lw=1.0)


def fig_faultfree(traces, breaks, out):
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.0), sharex="col")
    for j, typ in enumerate(CULTIVARS):
        t = traces[traces.cultivar == typ]
        _chart_panels(axes[:, j], t.z.to_numpy(), t.ewma.to_numpy(), t.ewma_limit.to_numpy(),
                      t.cusum_pos.to_numpy(), t.cusum_neg.to_numpy(),
                      {ch: t[f"alarm_{ch}"].to_numpy(bool) for ch in ("shewhart", "ewma", "cusum")},
                      CULT_COLOR[typ], breaks[typ])
        axes[0, j].set_title(CULTIVAR_LABEL[typ])
    _legend_if_any(axes[2, 0])
    _legend_if_any(axes[0, 0])
    fig.suptitle("Fault-free S4 replay with frozen S1–S3 chart settings (slope–bias-corrected ρd; "
                 "vertical lines: MC levels)", fontsize=9, color=INK)
    fig.tight_layout()
    _save(fig, out / "fig_s4_faultfree_charts")


def fig_illustration(illus, breaks, out):
    kind, val, onset = ILLUSTRATION
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.0), sharex="col")
    for j, (typ, zi, ons, res) in enumerate(illus):
        st = res["stats"]
        _chart_panels(axes[:, j], st["z"][0], st["ewma"][0], st["ewma_limit"][0], st["cpos"][0], st["cneg"][0],
                      {ch: res[ch][0] for ch in ("shewhart", "ewma", "cusum")}, CULT_COLOR[typ], breaks[typ], ons)
        axes[0, j].set_title(CULTIVAR_LABEL[typ])
        lo, hi = axes[0, j].get_ylim()
        axes[0, j].text(ons - 1.5, lo + 0.04 * (hi - lo), f"synthetic ρg +{val:g} g cm⁻³ from here →",
                        ha="right", va="bottom", fontsize=7.5, color=INK)
    _legend_if_any(axes[2, 0])
    _legend_if_any(axes[0, 0])
    fig.suptitle(f"Illustrative replay: synthetic mass offset from record {onset} onward (not a recorded fault)",
                 fontsize=9, color=INK)
    fig.tight_layout()
    _save(fig, out / "fig_illustrative_injection")


def fig_detection(seq, out):
    allc = seq[(seq.scope == PRIMARY["scope"]) & (seq.correction == PRIMARY["correction"])
               & (seq.baseline == PRIMARY["baseline"])]
    d = allc[allc.condition != "clean"]
    ctrl = allc[(allc.condition == "clean") & (allc.chart == "combined")].set_index("cultivar")["p_detect_within_5"]
    kinds = [("dA", "Attenuation offset (dB)"), ("dphi", "Phase offset (°)"), ("rho_g", "ρg offset (g cm⁻³)")]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.6), sharey=True)
    for i, typ in enumerate(CULTIVARS):
        for j, (kind, xl) in enumerate(kinds):
            ax = axes[i, j]
            for c_i, ch in enumerate(CHARTS):
                s = d[(d.cultivar == typ) & (d.kind == kind) & (d.chart == ch)].sort_values("value")
                ax.plot(s["value"], s["p_detect_within_5"], color=SERIES[c_i], lw=1.2,
                        marker=CHART_MARKER[ch], ms=4.5, label=CHART_LABEL[ch])
            ax.axhline(ctrl[typ], color=INK2, lw=0.9, ls="--", label="No offset (combined)")
            ax.set_ylim(-0.03, 1.03)
            ax.set_xticks(OFFSET_GRID[kind])
            if i == 1:
                ax.set_xlabel(xl)
            if j == 0:
                ax.set_ylabel(f"{CULTIVAR_LABEL[typ]}\nP(detected ≤ 5 records)")
    axes[0, 0].legend(loc="upper left", fontsize=7, ncol=1)
    fig.suptitle("Detection of synthetic offsets after a 20-record fault-free run (3000 sequences)",
                 fontsize=9, color=INK)
    fig.tight_layout()
    _save(fig, out / "fig_detection_within5")


def fig_mc_effect(eff, out):
    d = eff[eff.scope == PRIMARY["scope"]]
    labels = [offset_label(k, v) for k, vals in OFFSET_GRID.items() for v in vals]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for i, typ in enumerate(CULTIVARS):
        s = d[d.cultivar == typ].set_index("condition").loc[labels]
        ax.bar(x + (i - 0.5) * w, s["mean_d_MC"], width=w - 0.04, color=CULT_COLOR[typ],
               yerr=s["sd_d_MC"], error_kw={"lw": 0.8, "ecolor": INK2, "capsize": 2}, label=CULTIVAR_LABEL[typ])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, [lab.replace("rho_g", "ρg").replace("dphi", "Δφ").replace("dA", "ΔA") for lab in labels],
                  rotation=30, ha="right")
    ax.set_ylabel("Mean change in predicted MC\n(% w.b., ± SD)")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_title("Effect of each synthetic offset on the slope–bias-corrected S1–S3 MC prediction (S4 test records)")
    fig.tight_layout()
    _save(fig, out / "fig_mc_change_per_offset")


# ----------------------------------------------------------------------------------------------
# Summary text
# ----------------------------------------------------------------------------------------------
def summary_text(base, dep, ordered, seq, eff, elapsed):
    p = PRIMARY
    L = [f"DIGITAL SHADOW EVALUATION (Section 2.3.2) - {now_str()} - models: {CHRONO_DIR.name}",
         f"Primary: {p}. Sequences: {N_SEQ} x ({N_CLEAN} fault-free + {N_PERT} perturbed), seed={SEED}. "
         f"Chart settings: {SETTINGS.as_dict()}", f"Duration: {elapsed:.0f} s", "",
         "1) BASELINE (S1-S3 OOF, r = rho_g - rho_d, g cm-3)"]
    L.append(base[["scope", "cultivar", "bd_model", "n_records", "n_levels", "mu0", "sigma0",
                   "mu0_S4_transfer", "sigma0_S4_transfer"]].to_string(index=False))
    L += ["", "2) DEPENDENCE of r on MC and rho_g (OOF; records are not independent, descriptive)",
          dep[["scope", "cultivar", "b_MC", "b_rho_g", "R2_MC_rho_g", "dz_per_1pct_MC", "dz_per_0.01_rho_g",
               "spearman_MC", "spearman_rho_g"]].to_string(index=False)]
    o = ordered[(ordered.ordering == p["ordering"])]
    L += ["", "3) FAULT-FREE S4 TEST SEQUENCE (ordered) - share of records with an alarm / first alarm",
          o.pivot_table(index=["scope", "correction", "baseline", "cultivar"], columns="chart",
                        values="share_alarm").round(3).to_string()]
    s = seq[(seq.scope == p["scope"]) & (seq.correction == p["correction"]) & (seq.baseline == p["baseline"])]
    L += ["", "4) PRIMARY - alarm rate in the 20-record fault-free part (clean condition)",
          s[s.condition == "clean"].pivot_table(index="cultivar", columns="chart",
                                                values="p_false_alarm_fault_free_part").round(3).to_string(),
          "", "5) PRIMARY - detection within the first 5 records (sequences without an alarm in the fault-free part)",
          s.pivot_table(index=["kind", "value"], columns=["cultivar", "chart"],
                        values="p_detect_within_5").round(3).to_string(),
          "", "6) PRIMARY - median delay (records)",
          s[s.condition != "clean"].pivot_table(index=["kind", "value"], columns=["cultivar", "chart"],
                                                values="median_delay").to_string(),
          "", "7) EFFECT OF THE OFFSETS (cultivar-specific, slope-bias corrected)",
          eff[eff.scope == "specific"][["cultivar", "condition", "mean_d_rho_d", "mean_d_r", "mean_d_z",
                                         "mean_d_MC", "sd_d_MC"]].round(4).to_string(index=False),
          "", "NOTE: The offsets are test conditions, not detection limits. Perturbed records are synthetic; the archive contains no commands.",
          "Sequences are drawn with replacement and carry no serial dependence between records; the ordered sequence (3) complements this."]
    return "\n".join(L)


# ----------------------------------------------------------------------------------------------
def main(out_root=None):
    t0 = time.time()
    out = (RESULTS_ROOT / "digital_shadow") if out_root is None else Path(out_root)
    tdir, fdir = out / "tables", out / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    rec, blocks, _ = load_archive()
    preds_ref = pd.read_csv(PREDICTIONS_CSV)

    res = []
    for ci, typ in enumerate(CULTIVARS):
        n_te = int(((rec.Year == TEST_YEAR) & (rec.Type == typ) & ~rec.is_transfer).sum())
        idx = np.random.default_rng([SEED, ci]).integers(0, n_te, size=(N_SEQ, N_CLEAN + N_PERT))
        for scope in SCOPES:
            print(f"  {scope:<9} {typ:<7} ...", flush=True)
            res.append(evaluate_cultivar(rec, blocks, scope, typ, idx, preds_ref))

    base = pd.DataFrame([r["baseline"] for r in res])
    dep = pd.DataFrame([r["dependence"] for r in res])
    ordered = pd.DataFrame([row for r in res for row in r["ordered"]])
    seq = pd.DataFrame([row for r in res for row in r["sequences"]])
    eff = pd.DataFrame([row for r in res for row in r["effects"]])
    traces = pd.concat([t for r in res for t in r["traces"]], ignore_index=True)
    oofp = pd.concat([r["oof_points"] for r in res], ignore_index=True)
    for name, df in (("baseline_mu0_sigma0", base), ("residual_dependence", dep),
                     ("faultfree_ordered_S4", ordered), ("sequence_detection", seq), ("offset_effects", eff),
                     ("trace_S4_primary", traces)):
        save_table(df, tdir / name)
    oofp.to_csv(tdir / "oof_residuals_S1-S3.csv", index=False, encoding="utf-8-sig")

    _style()
    breaks = {r["baseline"]["cultivar"]: r["level_breaks"] for r in res if r["baseline"]["scope"] == "specific"}
    fig_dependence(oofp, base, fdir)
    fig_faultfree(traces, breaks, fdir)
    fig_illustration([r["illustration"] for r in res if r["illustration"] is not None], breaks, fdir)
    fig_detection(seq, fdir)
    fig_mc_effect(eff, fdir)

    elapsed = time.time() - t0
    text = summary_text(base, dep, ordered, seq, eff, elapsed)
    text_dump_atomic(text, out / "SUMMARY_digital_shadow.txt")
    import sklearn
    import torch
    manifest = {"date": now_str(), "analysis_version": ANALYSIS_VERSION, "chrono_dir": CHRONO_DIR.name,
                "python": sys.version, "platform": platform.platform(),
                "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                "torch": torch.__version__, "seed": SEED, "n_seq": N_SEQ, "n_clean": N_CLEAN, "n_pert": N_PERT,
                "detect_within": DETECT_WITHIN, "spc_settings": SETTINGS.as_dict(), "primary": PRIMARY,
                "offset_grid": OFFSET_GRID, "illustration": ILLUSTRATION,
                "models": sorted(set(base["bd_model"]) | set(base["mc_model"])),
                "inputs_sha256": {ARCHIVE_FILE: _sha256(DATA_ROOT / ARCHIVE_FILE),
                                  "predictions_all.csv": _sha256(PREDICTIONS_CSV)},
                "elapsed_sec": elapsed}
    json_dump_atomic(manifest, out / "manifest.json")
    print(text)
    return {"baseline": base, "dependence": dep, "ordered": ordered, "sequences": seq, "effects": eff}


if __name__ == "__main__":
    main()
