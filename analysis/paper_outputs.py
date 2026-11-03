# -*- coding: utf-8 -*-
"""
PAPER OUTPUTS - publication tables and figures from the outputs of the chronological analysis.

Input : <out_root>/tables/*.csv (outputs of chronological.py) + data/lentil_microwave_archive.csv
Output: <out_root>/paper/figures/*.pdf|png   (vector PDF + 600 dpi PNG)
        <out_root>/paper/tables/Paper_Tables.xlsx  (formatted, with a README sheet)  +  *.tex  +  *.csv
        <out_root>/paper/Figure_Captions_Draft.md, README_paper.txt

Principles
  * NO retraining; everything is computed from the saved predictions. The design of the main analysis is unchanged.
  * Primary (pre-specified) configuration: MC = cultivar-specific model, `full` input, slope-bias correction;
    BD = cultivar-specific model, `microwave` input, slope-bias correction. Everything else is a supplementary /
    sensitivity analysis and is labelled as such. The primary configuration is NOT chosen from the test results.
  * Confidence intervals are bootstrapped over the test MC LEVELS (the independent unit is the level, not the record).
"""
from __future__ import annotations

import itertools
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
import matplotlib.text as mtext  # noqa: E402

from chronological import (CULTIVARS, SEASONS, TRANSITIONS, build_records, trn_label)  # noqa: E402
from common import DATA_ROOT, FREQS, RESULTS_ROOT, load_dataset, pickle_load_safe  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# ----------------------------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------------------------
PRIMARY = {"moisture": {"input_set": "full", "correction": "slope_bias"},
           "bulk_density": {"input_set": "microwave", "correction": "slope_bias"}}
SPEC = {"CEREN": "Ceren", "CIFTCI": "Ciftci"}                    # Type -> cultivar-specific model_type
TRANS_ORDER = ["S1toS2", "S1+S2toS3", "S1+S2+S3toS4"]
TLAB = {"S1toS2": "S1→S2", "S1+S2toS3": "S1–S2→S3", "S1+S2+S3toS4": "S1–S3→S4"}
TLAB_LONG = {"S1toS2": "S1→S2\n(test 2015)", "S1+S2toS3": "S1–S2→S3\n(test 2016)", "S1+S2+S3toS4": "S1–S3→S4\n(test 2017)"}
CLAB = {"CEREN": "Ceren (green lentil)", "CIFTCI": "Çiftçi (red lentil)"}
CSHORT = {"CEREN": "Ceren", "CIFTCI": "Çiftçi"}
CORR_LAB = {"none": "No correction", "bias_only": "Bias correction", "slope_bias": "Slope–bias correction"}
INPUT_LAB = {"spectra": "Spectra", "microwave": "Microwave", "full": "Full"}
TARGET_LAB = {"moisture": "Moisture content (% w.b.)", "bulk_density": "Bulk density (g cm$^{-3}$)"}
TARGET_SHORT = {"moisture": "MC", "bulk_density": "BD"}
UNIT = {"moisture": "% w.b.", "bulk_density": "g cm$^{-3}$"}
DEC = {"moisture": 3, "bulk_density": 4}

# Okabe-Ito palette (colour-blind safe)
C_CULT = {"CEREN": "#009E73", "CIFTCI": "#D55E00"}
C_CORR = {"none": "#7A7A7A", "bias_only": "#0072B2", "slope_bias": "#D55E00"}
C_INPUT = {"spectra": "#56B4E9", "microwave": "#E69F00", "full": "#CC79A7"}
C_TRAIN, C_TRANSFER, C_TEST = "#0072B2", "#E69F00", "#009E73"


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "legend.frameon": False, "axes.linewidth": 0.8, "xtick.major.width": 0.8,
        "ytick.major.width": 0.8, "xtick.major.size": 3, "ytick.major.size": 3, "lines.linewidth": 1.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "axes.grid": False, "mathtext.default": "regular"})


def _panel_labels(axes, x=-0.14, y=1.04):
    for i, ax in enumerate(np.ravel(axes)):
        ax.text(x, y, f"({chr(97 + i)})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def _points_display(ax):
    """Display coordinates of all data points (scatter and line) inside the axes."""
    pts = []
    for coll in ax.collections:
        try:
            off = np.asarray(coll.get_offsets())
            if off.ndim == 2 and len(off):
                pts.append(coll.get_offset_transform().transform(off))
        except Exception:  # noqa: BLE001
            pass
    for ln in ax.lines:
        xy = ln.get_xydata()
        if len(xy):
            pts.append(ln.get_transform().transform(xy))
    return np.vstack(pts) if pts else np.empty((0, 2))


def corner_text(ax, s, fontsize=6.3):
    """Places the statistics box in the corner that covers the FEWEST data points. Call after the layout (tight_layout) is fixed."""
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pts = _points_display(ax)
    best = None
    for (x, y, ha, va) in ((0.04, 0.96, "left", "top"), (0.96, 0.96, "right", "top"),
                           (0.04, 0.04, "left", "bottom"), (0.96, 0.04, "right", "bottom")):
        t = ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize)
        bb = t.get_window_extent(rend).expanded(1.18, 1.25)
        n = int(((pts[:, 0] >= bb.x0) & (pts[:, 0] <= bb.x1) & (pts[:, 1] >= bb.y0) & (pts[:, 1] <= bb.y1)).sum()) if len(pts) else 0
        t.remove()
        if best is None or n < best[0]:
            best = (n, x, y, ha, va)
    n, x, y, ha, va = best
    ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize, zorder=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CCCCCC", lw=0.5))
    return n


def check_overlaps(fig, min_area=6.0):
    """Readability check: do texts overlap? Does a box or legend cover data points?"""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    items = []
    hidden = set()          # axis labels outside the view limits (not drawn)
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            eps = (hi - lo) * 1e-6
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                loc = tk.get_loc()
                if not (lo - eps <= loc <= hi + eps):
                    hidden.update((id(tk.label1), id(tk.label2)))
    for t in fig.findobj(mtext.Text):
        if not t.get_visible() or not t.get_text().strip() or id(t) in hidden:
            continue
        try:
            bb = t.get_window_extent(rend)
        except Exception:  # noqa: BLE001
            continue
        if bb.width >= 1 and bb.height >= 1:
            items.append((t.get_text()[:30].replace("\n", " "), bb))
    issues = []
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            (la, ba), (lb, bb_) = items[a], items[b]
            w = min(ba.x1, bb_.x1) - max(ba.x0, bb_.x0)
            h = min(ba.y1, bb_.y1) - max(ba.y0, bb_.y0)
            if w > 1.5 and h > 1.5 and w * h > min_area:
                issues.append(f"text-text: '{la}' <-> '{lb}'")
    for ax in fig.axes:
        pts = _points_display(ax)
        if not len(pts):
            continue
        ext = [(t.get_text()[:30].replace("\n", " "), t.get_window_extent(rend).expanded(1.12, 1.18))
               for t in ax.texts if t.get_visible() and t.get_text().strip() and t.get_bbox_patch() is not None]
        if ax.get_legend() is not None:
            ext.append(("legend", ax.get_legend().get_window_extent(rend)))
        for lab, bb in ext:
            n = int(((pts[:, 0] >= bb.x0) & (pts[:, 0] <= bb.x1) & (pts[:, 1] >= bb.y0) & (pts[:, 1] <= bb.y1)).sum())
            if n > 0:
                issues.append(f"text-data: '{lab}' covers {n} data points")
    return issues


class _Saver:
    def __init__(self, figdir, formats, dpi):
        self.figdir, self.formats, self.dpi, self.saved, self.issues = Path(figdir), formats, dpi, [], {}
        self.figdir.mkdir(parents=True, exist_ok=True)

    def __call__(self, fig, name):
        issues = check_overlaps(fig)
        for fmt in self.formats:
            fig.savefig(self.figdir / f"{name}.{fmt}", format=fmt, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        self.saved.append(name)
        if issues:
            self.issues[name] = issues
        print(f"    ✓ {name}" + (f"   ! {len(issues)} readability warning(s)" if issues else ""))
        for m in issues[:6]:
            print(f"        - {m}")


# ----------------------------------------------------------------------------------------------
# Statistics helpers (bootstrap over the test LEVELS)
# ----------------------------------------------------------------------------------------------
def _level_sums(y, p, levels):
    e = np.asarray(p, float) - np.asarray(y, float)
    y = np.asarray(y, float)
    df = pd.DataFrame({"l": levels, "n": 1.0, "e": e, "e2": e ** 2, "ae": np.abs(e), "y": y, "y2": y ** 2})
    return df.groupby("l")[["n", "e", "e2", "ae", "y", "y2"]].sum().to_numpy()


def _metrics_from_sums(S):
    n, se, se2, sae, sy, sy2 = (S[..., i] for i in range(6))
    rmse = np.sqrt(se2 / n)
    sst = sy2 - sy ** 2 / n
    return {"RMSEP": rmse, "MAE": sae / n, "Bias": se / n,
            "SEP": np.sqrt(np.maximum(se2 - se ** 2 / n, 0) / np.maximum(n - 1, 1)),
            "R2": 1 - se2 / sst, "RPD": np.sqrt(sst / np.maximum(n - 1, 1)) / rmse}


def metrics_with_ci(y, p, levels, n_boot=2000, seed=0):
    S = _level_sums(y, p, levels)
    pt = _metrics_from_sums(S.sum(0))
    idx = np.random.default_rng(seed).integers(0, len(S), size=(n_boot, len(S)))
    bt = _metrics_from_sums(S[idx].sum(1))
    out = {}
    for k in pt:
        lo, hi = np.nanpercentile(bt[k], [2.5, 97.5])
        out[k], out[k + "_lo"], out[k + "_hi"] = float(pt[k]), float(lo), float(hi)
    y, p = np.asarray(y, float), np.asarray(p, float)
    q75, q25 = np.percentile(y, [75, 25])
    out["RPIQ"] = float((q75 - q25) / out["RMSEP"]) if out["RMSEP"] > 0 else np.nan
    try:
        out["Slope"], out["Intercept"] = (float(v) for v in np.polyfit(y, p, 1))
    except Exception:  # noqa: BLE001
        out["Slope"] = out["Intercept"] = np.nan
    out["MAPE_pct"] = float(np.mean(np.abs(p - y) / np.abs(y)) * 100)
    out["MaxErr"] = float(np.max(np.abs(p - y)))
    return out


def train_range(rec, transition, model_type, target):
    train_years = [y for y, s in SEASONS.items() if s in transition.replace("to", "+").split("+")[:-1]]
    types = list(CULTIVARS.values()) if model_type == "Pooled" else [CULTIVARS[model_type]]
    col = "MC" if target == "moisture" else "BD"
    d = rec[rec["Year"].isin(train_years) & rec["Type"].isin(types)][col]
    return float(d.min()), float(d.max())


def build_metric_table(P, rec, n_boot):
    """Metrics and 95 % CI for every configuration (target x input x transition x model type x cultivar x correction)."""
    T = P[~P["is_transfer"]]
    rows = []
    cols = {"none": "p_none", "bias_only": "p_bias_only", "slope_bias": "p_slope_bias"}
    for keys, g in T.groupby(["target", "input_set", "transition", "model_type", "Type"], sort=False):
        tg, ins, tr, mt, typ = keys
        lo, hi = train_range(rec, tr, mt, tg)
        outside = float(((g["y_true"] < lo) | (g["y_true"] > hi)).mean() * 100)
        for cname, col in cols.items():
            m = metrics_with_ci(g["y_true"], g[col], g["Level_Key"], n_boot)
            rows.append({"target": tg, "input_set": ins, "transition": tr, "model_type": mt, "cultivar": typ,
                         "correction": cname, "N": len(g), "N_levels": g["Level_Key"].nunique(),
                         "train_min": lo, "train_max": hi, "pct_test_outside_train_range": outside, **m})
    return pd.DataFrame(rows)


def _paired_full(levels, ea, eb, n_boot=2000, seed=0):
    d = pd.DataFrame({"l": levels, "a": np.asarray(ea) ** 2, "b": np.asarray(eb) ** 2})
    g = d.groupby("l").agg(sa=("a", "sum"), sb=("b", "sum"), n=("a", "size"))
    sa, sb, n = g["sa"].to_numpy(), g["sb"].to_numpy(), g["n"].to_numpy()
    point = float(np.sqrt(sb.sum() / n.sum()) - np.sqrt(sa.sum() / n.sum()))
    idx = np.random.default_rng(seed).integers(0, len(g), size=(n_boot, len(g)))
    diffs = np.sqrt(sb[idx].sum(1) / n[idx].sum(1)) - np.sqrt(sa[idx].sum(1) / n[idx].sum(1))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = min(1.0, 2 * min((np.sum(diffs <= 0) + 1) / (n_boot + 1), (np.sum(diffs >= 0) + 1) / (n_boot + 1)))
    return point, float(lo), float(hi), float(p), len(g)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def build_comparisons(P, n_boot):
    """The comparisons of chronological.comparison_table + bootstrap p and Holm adjustment (SUPPLEMENTARY)."""
    T = P[~P["is_transfer"]].copy()
    rows = []
    cmap = {"none": "p_none", "bias_only": "p_bias_only", "slope_bias": "p_slope_bias"}

    def add(kind, ctx, a, b, ca, cb, la, lb):
        m = a.merge(b, on="rid", suffixes=("_a", "_b"))
        if len(m) < 2:
            return
        ea, eb = m[ca + "_a"] - m["y_true_a"], m[cb + "_b"] - m["y_true_b"]
        pt, lo, hi, p, nl = _paired_full(m["Level_Key_a"].to_numpy(), ea.to_numpy(), eb.to_numpy(), n_boot)
        rows.append({"comparison": kind, **ctx, "A": la, "B": lb, "diff_RMSEP_B_minus_A": pt, "ci95_low": lo,
                     "ci95_high": hi, "p_boot": p, "N": len(m), "N_levels": nl})

    base = ["target", "input_set", "transition", "model_type", "Type"]
    for keys, g in T.groupby(base, sort=False):
        ctx = dict(zip(base[:-1], keys[:-1]), cultivar=keys[-1], correction="")
        add("slope_bias_vs_none", ctx, g, g, "p_none", "p_slope_bias", "none", "slope_bias")
        add("bias_only_vs_none", ctx, g, g, "p_none", "p_bias_only", "none", "bias_only")
    for (tg, ins, tr, typ), g in T.groupby(["target", "input_set", "transition", "Type"], sort=False):
        a, b = g[g["model_type"] == SPEC[typ]], g[g["model_type"] == "Pooled"]
        for cname in ("none", "slope_bias"):
            add("pooled_vs_specific", {"target": tg, "input_set": ins, "transition": tr, "model_type": "-",
                                       "cultivar": typ, "correction": cname}, a, b, cmap[cname], cmap[cname],
                SPEC[typ], "Pooled")
    for (tg, tr, mt, typ), g in T.groupby(["target", "transition", "model_type", "Type"], sort=False):
        for sa_, sb_ in (("spectra", "microwave"), ("microwave", "full")):
            a, b = g[g["input_set"] == sa_], g[g["input_set"] == sb_]
            if a.empty or b.empty:
                continue
            for cname in ("none", "slope_bias"):
                add("inputset", {"target": tg, "input_set": f"{sa_}->{sb_}", "transition": tr, "model_type": mt,
                                 "cultivar": typ, "correction": cname}, a, b, cmap[cname], cmap[cname], sa_, sb_)
    C = pd.DataFrame(rows)
    C["p_holm"] = np.nan
    for _, idx in C.groupby(["comparison", "target"]).groups.items():
        C.loc[idx, "p_holm"] = holm(C.loc[idx, "p_boot"].to_numpy())
    C["verdict"] = np.where(C["ci95_high"] < 0, "B better", np.where(C["ci95_low"] > 0, "B worse", "uncertain"))
    return C


def transfer_sensitivity(P, target, input_set, kmax=4, cap=300, seed=0):
    """Sensitivity to the number of transfer levels (k) and to which levels are chosen. NO training: from the saved raw
    predictions, the correction is refitted for all (or at most `cap` random) combinations of k transfer levels and the
    RMSEP is computed on the REMAINING levels. (Note: the test levels change with k; the main analysis uses k = 2.)"""
    rng = np.random.default_rng(seed)
    D = P[(P["target"] == target) & (P["input_set"] == input_set) & (P["model_type"].isin(SPEC.values()))]
    D = D[D["model_type"] == D["Type"].map(SPEC)]
    rows = []
    for (tr, typ), g in D.groupby(["transition", "Type"]):
        y, p, lv = g["y_true"].to_numpy(float), g["y_pred"].to_numpy(float), g["Level_Key"].to_numpy()
        levels = list(pd.unique(lv))
        L = len(levels)
        for k in range(0, min(kmax, L - 3) + 1):
            combos = list(itertools.combinations(levels, k)) if math.comb(L, k) <= cap else \
                [tuple(rng.choice(levels, size=k, replace=False)) for _ in range(cap)]
            for combo in combos:
                tm = np.isin(lv, combo)
                te = ~tm
                r_none = float(np.sqrt(np.mean((p[te] - y[te]) ** 2)))
                rows.append((tr, typ, k, "none", r_none, r_none))
                if k >= 1:
                    shift = float(np.mean(y[tm] - p[tm]))
                    rows.append((tr, typ, k, "bias_only", float(np.sqrt(np.mean((p[te] + shift - y[te]) ** 2))), r_none))
                if k >= 2 and np.std(p[tm]) > 0:
                    b1, b0 = np.polyfit(p[tm], y[tm], 1)
                    rows.append((tr, typ, k, "slope_bias", float(np.sqrt(np.mean((b0 + b1 * p[te] - y[te]) ** 2))), r_none))
    R = pd.DataFrame(rows, columns=["transition", "Type", "k", "method", "rmsep", "rmsep_none_same_test"])
    A = R.groupby(["transition", "Type", "k", "method"]).agg(
        mean=("rmsep", "mean"), p05=("rmsep", lambda s: np.percentile(s, 5)), p95=("rmsep", lambda s: np.percentile(s, 95)),
        n_combos=("rmsep", "size")).reset_index()
    return A


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
def fig_design(rec, tinfo, save):
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 4.3), sharex=True)
    for ax, typ in zip(axes, ["CEREN", "CIFTCI"]):
        for i, (train_years, test_year) in enumerate(TRANSITIONS):
            y0 = 2 - i
            for yr in (2014, 2015, 2016, 2017):
                n_all = int(((rec["Year"] == yr) & (rec["Type"] == typ)).sum())
                x0 = yr - 2014
                if yr in train_years:
                    ax.add_patch(Rectangle((x0 + 0.04, y0 + 0.08), 0.92, 0.7, color=C_TRAIN, alpha=0.9, lw=0))
                    ax.text(x0 + 0.5, y0 + 0.43, f"train\nn = {n_all}", ha="center", va="center", color="white", fontsize=7)
                elif yr == test_year:
                    r = tinfo[(tinfo["Year"] == yr) & (tinfo["Type"] == typ)].iloc[0]
                    frac = r["n_transfer_records"] / r["n_incoming_records"]
                    ax.add_patch(Rectangle((x0 + 0.04, y0 + 0.08), 0.92 * frac, 0.7, color=C_TRANSFER, lw=0))
                    ax.add_patch(Rectangle((x0 + 0.04 + 0.92 * frac, y0 + 0.08), 0.92 * (1 - frac), 0.7, color=C_TEST, lw=0))
                    ax.text(x0 + 0.5, y0 + 0.43, f"transfer {int(r['n_transfer_records'])}\ntest {int(r['n_test_records'])}",
                            ha="center", va="center", color="black", fontsize=6.5)
                else:
                    ax.add_patch(Rectangle((x0 + 0.04, y0 + 0.08), 0.92, 0.7, facecolor="#EEEEEE", edgecolor="#BBBBBB", lw=0.6))
                    ax.text(x0 + 0.5, y0 + 0.43, "—", ha="center", va="center", color="#999999")
            ax.text(-0.08, y0 + 0.43, TLAB[TRANS_ORDER[i]], ha="right", va="center", fontsize=8)
        ax.set_xlim(-0.05, 4.02)
        ax.set_ylim(-0.05, 3.0)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title(CLAB[typ], loc="left", fontweight="bold", color=C_CULT[typ])
    axes[-1].set_xticks([0.5, 1.5, 2.5, 3.5])
    axes[-1].set_xticklabels(["S1 (2014)", "S2 (2015)", "S3 (2016)", "S4 (2017)"])
    axes[-1].set_xlabel("Harvest season")
    axes[0].legend(handles=[Patch(color=C_TRAIN, label="Training (model development)"),
                            Patch(color=C_TRANSFER, label="Transfer subset (2 MC levels; correction fitted here)"),
                            Patch(color=C_TEST, label="Test (remaining MC levels)")],
                   loc="upper center", bbox_to_anchor=(0.5, 1.45), ncol=3, fontsize=7)
    fig.subplots_adjust(hspace=0.28, left=0.1)
    save(fig, "Fig01_study_design")


def fig_data_overview(rec, save):
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.7), gridspec_kw={"width_ratios": [1.1, 1.1, 1]})
    rng = np.random.default_rng(1)
    for ax, col, lab in ((axes[0], "MC", TARGET_LAB["moisture"]), (axes[1], "BD", TARGET_LAB["bulk_density"])):
        for j, typ in enumerate(["CEREN", "CIFTCI"]):
            for k, yr in enumerate((2014, 2015, 2016, 2017)):
                d = rec[(rec["Type"] == typ) & (rec["Year"] == yr)][col].to_numpy()
                x = k + (-0.19 if typ == "CEREN" else 0.19)
                ax.scatter(x + rng.uniform(-0.09, 0.09, len(d)), d, s=3, color=C_CULT[typ], alpha=0.45, lw=0, rasterized=True)
                ax.boxplot(d, positions=[x], widths=0.28, showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor="none", edgecolor="black", lw=0.7), medianprops=dict(color="black", lw=1),
                           whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7))
        ax.set_xticks(range(4))
        ax.set_xticklabels(["S1", "S2", "S3", "S4"])
        ax.set_ylabel(lab)
        ax.set_xlabel("Season")
    for typ in ("CEREN", "CIFTCI"):
        d = rec[rec["Type"] == typ]
        axes[2].scatter(d["MC"], d["BD"], s=4, color=C_CULT[typ], alpha=0.5, lw=0, label=CSHORT[typ], rasterized=True)
    axes[2].set_xlabel(TARGET_LAB["moisture"])
    axes[2].set_ylabel(TARGET_LAB["bulk_density"])
    axes[2].legend(markerscale=2.5, loc="upper right")
    _panel_labels(axes)
    fig.tight_layout()
    save(fig, "Fig02_data_overview")


def fig_spectra(rec, blocks, save, year=2016):
    labs = [("dA", "ΔA (dB)", "Attenuation"), ("dphi", "Δφ (°)", "Phase shift"), ("e1", "ε′", "Dielectric constant"),
            ("e2", "ε″", "Loss factor")]
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 4.1), sharex=True, layout="constrained")
    norm = plt.Normalize(rec["MC"].min(), rec["MC"].max())
    cmap = plt.get_cmap("viridis")
    f = FREQS / 1000.0
    for r, typ in enumerate(["CEREN", "CIFTCI"]):
        m = ((rec["Year"] == year) & (rec["Type"] == typ)).to_numpy()
        sub = rec[m]
        for c, (b, lab, ttl) in enumerate(labs):
            ax = axes[r, c]
            for lk, g in sub.groupby("Level_Key"):
                idx = g["rid"].to_numpy()
                ax.plot(f, blocks[b][idx].mean(0), color=cmap(norm(g["MC"].mean())), lw=1.0)
            if r == 1:
                ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel(f"{CSHORT[typ]}: {lab}" if c == 0 else lab)
            if r == 0:
                ax.set_title(ttl)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", pad=0.05, aspect=50, shrink=0.8)
    cb.set_label(f"Moisture content of MC level (% w.b.); level-mean spectra, S3 ({year})")
    _panel_labels(axes[0], x=-0.2, y=1.12)
    save(fig, "Fig03_example_spectra")


def _specific(P, target, input_set):
    D = P[(P["target"] == target) & (P["input_set"] == input_set)]
    return D[D["model_type"] == D["Type"].map(SPEC)]


def fig_scatter(P, M, target, input_set, save, name):
    D = _specific(P, target, input_set)
    rows = [(t, c) for t in ["CEREN", "CIFTCI"] for c in ("none", "slope_bias")]
    fig, axes = plt.subplots(4, 3, figsize=(7.1, 8.4))
    cols = {"none": "p_none", "slope_bias": "p_slope_bias"}
    stats = {}
    for i, (typ, cn) in enumerate(rows):
        for j, tr in enumerate(TRANS_ORDER):
            ax = axes[i, j]
            g = D[(D["Type"] == typ) & (D["transition"] == tr)]
            te, tf = g[~g["is_transfer"]], g[g["is_transfer"]]
            ax.scatter(te["y_true"], te[cols[cn]], s=6, color=C_CULT[typ], alpha=0.55, lw=0, label="Test", rasterized=True)
            ax.scatter(tf["y_true"], tf[cols[cn]], s=14, facecolors="none", edgecolors=C_TRANSFER, lw=0.8, label="Transfer")
            lo = min(g["y_true"].min(), g[cols[cn]].min())
            hi = max(g["y_true"].max(), g[cols[cn]].max())
            pad = 0.04 * (hi - lo)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.7)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_aspect("equal", adjustable="box")
            r = M[(M["target"] == target) & (M["input_set"] == input_set) & (M["transition"] == tr) & (M["cultivar"] == typ)
                  & (M["model_type"] == SPEC[typ]) & (M["correction"] == cn)].iloc[0]
            d = DEC[target]
            stats[(i, j)] = f"RMSEP {r['RMSEP']:.{d}f}\nMAE {r['MAE']:.{d}f}\nR² {r['R2']:.3f}\nBias {r['Bias']:+.{d}f}"
            if i == 0:
                ax.set_title(TLAB_LONG[tr].replace("\n", " "), fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{CSHORT[typ]}: {CORR_LAB[cn].lower()}\nPredicted ({UNIT[target]})", fontsize=7.5)
            if i == 3:
                ax.set_xlabel(f"Reference ({UNIT[target]})")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color="#555555", ms=3.5, label="Test records"),
                        Line2D([], [], marker="o", ls="", mfc="none", mec=C_TRANSFER, ms=5, label="Transfer records"),
                        Line2D([], [], color="k", ls="--", lw=0.8, label="1:1 line")],
               loc="lower center", ncol=3, fontsize=7.5)
    fig.tight_layout(rect=[0, 0.025, 1, 1])
    for (i, j), txt in stats.items():          # after the layout is fixed, into the emptiest corner
        corner_text(axes[i, j], txt)
    save(fig, name)


def fig_dotwhisker(M, target, save, name):
    D = M[(M["target"] == target) & (M["model_type"] == M["cultivar"].map(SPEC))]
    corrs = ["none", "bias_only", "slope_bias"]
    ins = [i for i in ("spectra", "microwave", "full") if i in D["input_set"].unique()]
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.5), sharey="row")
    off = np.linspace(-0.22, 0.22, len(ins))
    for r, typ in enumerate(["CEREN", "CIFTCI"]):
        for c, cn in enumerate(corrs):
            ax = axes[r, c]
            for k, iset in enumerate(ins):
                for x, tr in enumerate(TRANS_ORDER):
                    q = D[(D["cultivar"] == typ) & (D["correction"] == cn) & (D["input_set"] == iset) & (D["transition"] == tr)]
                    if q.empty:
                        continue
                    q = q.iloc[0]
                    ax.errorbar(x + off[k], q["RMSEP"], yerr=[[q["RMSEP"] - q["RMSEP_lo"]], [q["RMSEP_hi"] - q["RMSEP"]]],
                                fmt="o", ms=4.2, color=C_INPUT[iset], ecolor=C_INPUT[iset], elinewidth=1, capsize=1.8, mec="white", mew=0.4)
            ax.set_xticks(range(3))
            ax.set_xticklabels([TLAB[t] for t in TRANS_ORDER])
            ax.set_xlim(-0.5, 2.5)
            ax.set_ylim(bottom=0)
            if r == 0:
                ax.set_title(CORR_LAB[cn], color=C_CORR[cn], fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{CSHORT[typ]}\nRMSEP ({UNIT[target]})")
            ax.yaxis.grid(True, color="#E5E5E5", lw=0.6)
            ax.set_axisbelow(True)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=C_INPUT[i], label=INPUT_LAB[i] + " input") for i in ins],
               loc="lower center", ncol=len(ins), fontsize=7.5)
    _panel_labels(axes, x=-0.2)
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    save(fig, name)


def fig_residuals(P, target, input_set, save, name):
    D = _specific(P, target, input_set)
    fig, axes = plt.subplots(4, 3, figsize=(7.1, 7.6), sharex="col")
    cols = {"none": "p_none", "slope_bias": "p_slope_bias"}
    for i, (typ, cn) in enumerate([(t, c) for t in ["CEREN", "CIFTCI"] for c in ("none", "slope_bias")]):
        for j, tr in enumerate(TRANS_ORDER):
            ax = axes[i, j]
            g = D[(D["Type"] == typ) & (D["transition"] == tr)].copy()
            g["err"] = g[cols[cn]] - g["y_true"]
            te, tf = g[~g["is_transfer"]], g[g["is_transfer"]]
            ax.axhline(0, color="k", lw=0.7)
            ax.scatter(te["y_true"], te["err"], s=4, color=C_CULT[typ], alpha=0.35, lw=0, rasterized=True)
            lv = te.groupby("Level_Key").agg(x=("y_true", "mean"), m=("err", "mean"), s=("err", "std"))
            ax.errorbar(lv["x"], lv["m"], yerr=lv["s"].fillna(0), fmt="o", ms=3.5, color="black", ecolor="#555555",
                        elinewidth=0.7, capsize=1.2, zorder=3)
            ax.scatter(tf["y_true"], tf["err"], s=14, facecolors="none", edgecolors=C_TRANSFER, lw=0.8, zorder=4)
            if i == 0:
                ax.set_title(TLAB_LONG[tr].replace("\n", " "), fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{CSHORT[typ]}: {CORR_LAB[cn].lower()}\nError ({UNIT[target]})", fontsize=7.5)
            if i == 3:
                ax.set_xlabel(f"Reference ({UNIT[target]})")
            ax.yaxis.grid(True, color="#EEEEEE", lw=0.5)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color="black", ms=3.5, label="MC-level mean ± SD (test records)"),
                        Line2D([], [], marker="o", ls="", mfc="none", mec=C_TRANSFER, label="Transfer records"),
                        Line2D([], [], marker="o", ls="", color="#888888", ms=2.5, alpha=0.6, label="Individual test records")],
               loc="lower center", ncol=3, fontsize=7)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    save(fig, name)


def _forest_rows(C, target, comparison, **flt):
    D = C[(C["target"] == target) & (C["comparison"] == comparison)]
    for k, v in flt.items():
        D = D[D[k] == v]
    return D


def fig_forest(C, target, save, name):
    prim = PRIMARY[target]
    panels = [
        ("(a) Correction vs none (cultivar-specific models)", pd.concat([
            _forest_rows(C, target, "slope_bias_vs_none", input_set=prim["input_set"]).query("model_type in ['Ceren','Ciftci']"),
            _forest_rows(C, target, "bias_only_vs_none", input_set=prim["input_set"]).query("model_type in ['Ceren','Ciftci']")]),
         lambda r: f"{CSHORT[r['cultivar'] if 'cultivar' in r else r['Type']]} {TLAB[r['transition']]} — {r['B'].replace('_', '–')}"),
        ("(b) Pooled vs cultivar-specific (slope–bias corrected)", _forest_rows(C, target, "pooled_vs_specific",
                                                                                 input_set=prim["input_set"], correction="slope_bias"),
         lambda r: f"{CSHORT[r['cultivar']]} {TLAB[r['transition']]}"),
        ("(c) More inputs vs fewer (cultivar-specific, slope–bias corrected)", _forest_rows(C, target, "inputset", correction="slope_bias")
         .query("model_type in ['Ceren','Ciftci']"),
         lambda r: f"{CSHORT[r['cultivar']]} {TLAB[r['transition']]} — {r['input_set'].replace('->', '→')}"),
    ]
    heights = [max(len(p[1]), 1) for p in panels]
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 1.2 + 0.17 * sum(heights)), gridspec_kw={"height_ratios": heights})
    for ax, (title, D, labf) in zip(axes, panels):
        D = D.copy()
        if D.empty:
            ax.axis("off")
            continue
        D["lab"] = D.apply(labf, axis=1)
        D = D.sort_values(["cultivar", "transition", "lab"], ascending=[True, True, True]).reset_index(drop=True)
        for yv, r in D.iterrows():
            col = {"B better": "#0072B2", "B worse": "#D55E00", "uncertain": "#7A7A7A"}[r["verdict"]]
            ax.errorbar(r["diff_RMSEP_B_minus_A"], yv, xerr=[[r["diff_RMSEP_B_minus_A"] - r["ci95_low"]],
                                                           [r["ci95_high"] - r["diff_RMSEP_B_minus_A"]]],
                        fmt="o", ms=3.5, color=col, ecolor=col, elinewidth=1, capsize=1.5)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(range(len(D)))
        ax.set_yticklabels(D["lab"], fontsize=6.3)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=8, fontweight="bold")
        ax.xaxis.grid(True, color="#EEEEEE", lw=0.5)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel(f"Δ RMSEP, B − A ({UNIT[target]}); negative = B better (95% bootstrap CI over test MC levels)")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color="#0072B2", label="B better (CI excludes 0)"),
                        Line2D([], [], marker="o", ls="", color="#D55E00", label="B worse (CI excludes 0)"),
                        Line2D([], [], marker="o", ls="", color="#7A7A7A", label="Uncertain")],
               loc="lower center", ncol=3, fontsize=7)
    fig.tight_layout(rect=[0, 0.025, 1, 1])
    save(fig, name)


def fig_transfer(TS, target, save, name):
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.4), sharey="row")
    for r, typ in enumerate(["CEREN", "CIFTCI"]):
        for c, tr in enumerate(TRANS_ORDER):
            ax = axes[r, c]
            for meth in ("none", "bias_only", "slope_bias"):
                q = TS[(TS["Type"] == typ) & (TS["transition"] == tr) & (TS["method"] == meth)].sort_values("k")
                if q.empty:
                    continue
                ax.plot(q["k"], q["mean"], "-o", ms=3.5, color=C_CORR[meth], label=CORR_LAB[meth])
                ax.fill_between(q["k"], q["p05"], q["p95"], color=C_CORR[meth], alpha=0.15, lw=0)
            ax.axvline(2, color="#999999", lw=0.7, ls=":")
            ax.set_xticks(range(0, 5))
            ax.set_xlim(-0.2, 4.2)
            if r == 0:
                ax.set_title(TLAB[tr])
            if r == 1:
                ax.set_xlabel("Number of transfer MC levels (k)")
            if c == 0:
                ax.set_ylabel(f"{CSHORT[typ]}\nRMSEP ({UNIT[target]})")
            ax.yaxis.grid(True, color="#EEEEEE", lw=0.5)
    fig.legend(handles=[Line2D([], [], marker="o", color=C_CORR[m], label=CORR_LAB[m]) for m in ("none", "bias_only", "slope_bias")],
               loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 0.03))
    fig.text(0.5, 0.0, "Dotted line: k = 2 (main analysis). Bands: 5th–95th percentile over the choice of transfer levels.",
             ha="center", va="bottom", fontsize=6.5, color="#555555")
    _panel_labels(axes[0], x=-0.2)
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    save(fig, name)


def fig_model_overview(frozen, cvs, save):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
    cnt = frozen.groupby(["selected", "target"]).size().unstack(fill_value=0)
    cnt = cnt.loc[cnt.sum(1).sort_values().index]
    left = np.zeros(len(cnt))
    tcol = {"moisture": "#0072B2", "bulk_density": "#E69F00"}
    for tg in cnt.columns:
        axes[0].barh(range(len(cnt)), cnt[tg], left=left, color=tcol[tg], label=TARGET_SHORT[tg], height=0.7)
        left += cnt[tg].to_numpy()
    axes[0].set_yticks(range(len(cnt)))
    axes[0].set_yticklabels(cnt.index)
    axes[0].set_xlabel("Times selected (of 45 combinations)")
    _h, _l = axes[0].get_legend_handles_labels()
    fig.legend(_h, _l, title="Target", loc="lower center", ncol=2, fontsize=7.5)
    best = cvs.groupby("combo_id")["cv_rmse"].transform("min")
    cvs = cvs.assign(ratio=cvs["cv_rmse"] / best)
    order = cvs.groupby("model")["ratio"].median().sort_values().index.tolist()
    data = [cvs.loc[cvs["model"] == m, "ratio"].to_numpy() for m in order]
    bp = axes[1].boxplot(data, vert=False, showfliers=True, patch_artist=True, widths=0.6,
                         flierprops=dict(marker=".", ms=2, alpha=0.4), medianprops=dict(color="black", lw=1))
    for b in bp["boxes"]:
        b.set(facecolor="#BFD7EA", edgecolor="#333333", lw=0.6)
    axes[1].set_yticks(range(1, len(order) + 1))
    axes[1].set_yticklabels(order)
    axes[1].invert_yaxis()
    axes[1].set_xscale("log")
    axes[1].set_xticks([1, 1.25, 1.5, 2, 3, 5])
    axes[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[1].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes[1].axvline(1, color="k", lw=0.6)
    axes[1].set_xlabel("CV-RMSE relative to the best model in the combination")
    _panel_labels(axes, x=-0.25)
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    save(fig, "Fig09_model_selection_overview")


_TSHORT = {"S1toS2": "S1→S2", "S1+S2toS3": "S1–2→S3", "S1+S2+S3toS4": "S1–3→S4"}


def fig_model_heatmap(ALLM, frozen, target, input_set, save, name):
    D = ALLM[(ALLM["target"] == target) & (ALLM["input_set"] == input_set) & (ALLM["model_type"] == ALLM["cultivar"].map(SPEC))]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 5.0), sharey=True)
    colkeys = [(c, t) for c in ["CEREN", "CIFTCI"] for t in TRANS_ORDER]
    order = D[D["correction"] == "slope_bias"].groupby("model")["RMSEP"].median().sort_values().index.tolist()
    sel = frozen[(frozen["target"] == target) & (frozen["input_set"] == input_set)]
    for ax, cn in zip(axes, ("none", "slope_bias")):
        mat = np.full((len(order), len(colkeys)), np.nan)
        for j, (c, t) in enumerate(colkeys):
            q = D[(D["cultivar"] == c) & (D["transition"] == t) & (D["correction"] == cn)].set_index("model")["RMSEP"]
            for i, m in enumerate(order):
                if m in q.index:
                    mat[i, j] = q[m]
        im = ax.imshow(mat, aspect="auto", cmap="viridis_r", vmin=np.nanpercentile(mat, 2), vmax=np.nanpercentile(mat, 90))
        for i in range(len(order)):
            for j in range(len(colkeys)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.{2 if target == 'moisture' else 3}f}", ha="center", va="center", fontsize=5.4,
                            color="white" if mat[i, j] > np.nanmedian(mat) else "black")
        for j, (c, t) in enumerate(colkeys):
            s = sel[(sel["model_type"] == SPEC[c]) & (sel["transition"] == t)]
            if not s.empty and s.iloc[0]["selected"] in order:
                ax.add_patch(Rectangle((j - 0.46, order.index(s.iloc[0]["selected"]) - 0.46), 0.92, 0.92, fill=False,
                                       ec="#FF3B30", lw=1.5, zorder=5))
        ax.set_xticks(range(len(colkeys)))
        ax.set_xticklabels([f"{CSHORT[c]}\n{_TSHORT[t]}" for c, t in colkeys], fontsize=5.8, rotation=0)
        ax.set_title(CORR_LAB[cn], fontsize=8, color=C_CORR[cn], fontweight="bold")
        ax.axvline(2.5, color="white", lw=2)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order, fontsize=7)
    cb = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.035, pad=0.1)
    cb.set_label(f"Test RMSEP ({UNIT[target]}); red frame = model selected by CV (main analysis)")
    _panel_labels(axes, x=-0.02, y=1.03)
    save(fig, name)


def fig_selection_regret(ALLM, frozen, P, target, input_set, save, name):
    D = ALLM[(ALLM["target"] == target) & (ALLM["input_set"] == input_set) & (ALLM["model_type"] == ALLM["cultivar"].map(SPEC))
             & (ALLM["correction"] == "slope_bias")]
    cvs = None
    rows = []
    for (c, t), g in D.groupby(["cultivar", "transition"]):
        s = frozen[(frozen["target"] == target) & (frozen["input_set"] == input_set) & (frozen["model_type"] == SPEC[c])
                   & (frozen["transition"] == t)]
        if s.empty:
            continue
        sel = s.iloc[0]["selected"]
        gg = g.set_index("model")["RMSEP"]
        if sel not in gg.index:
            continue
        rows.append({"cultivar": c, "transition": t, "selected": gg[sel], "best": gg.min(), "median": gg.median(),
                     "worst": gg.max(), "rank": int((gg < gg[sel]).sum() + 1), "n": len(gg), "best_model": gg.idxmin()})
    R = pd.DataFrame(rows)
    if R.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.1, 3.1))
    R["lab"] = R.apply(lambda r: f"{CSHORT[r['cultivar']]}  {TLAB[r['transition']]}", axis=1)
    R = R.sort_values(["cultivar", "transition"]).reset_index(drop=True)
    for i, r in R.iterrows():
        ax.plot([r["best"], r["worst"]], [i, i], color="#CCCCCC", lw=3, solid_capstyle="round", zorder=1)
        ax.scatter(r["median"], i, marker="|", s=60, color="#555555", zorder=2)
        ax.scatter(r["best"], i, marker="D", s=22, color="#009E73", zorder=3)
        ax.scatter(r["selected"], i, marker="o", s=34, color="#D55E00", zorder=4)
        ax.annotate(f"rank {r['rank']}/{r['n']}", xy=(r["worst"], i), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=6.5, color="#333333")
    ax.set_yticks(range(len(R)))
    ax.set_yticklabels(R["lab"])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(R["best"].min() * 0.75, R["worst"].max() * 2.6)
    _tk = [v for v in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100) if R["best"].min() * 0.75 <= v <= R["worst"].max() * 2.6]
    ax.set_xticks(_tk)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel(f"Test RMSEP ({UNIT[target]}), slope–bias corrected (log scale)")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color="#D55E00", label="Selected by CV (main analysis)"),
                       Line2D([], [], marker="D", ls="", color="#009E73", label="Best model on test (oracle, not usable)"),
                       Line2D([], [], marker="|", ls="", color="#555555", label="Median of all models"),
                       Line2D([], [], color="#CCCCCC", lw=3, label="Range of all models")], loc="lower center", ncol=2, fontsize=7)
    ax.xaxis.grid(True, color="#EEEEEE", lw=0.5)
    fig.tight_layout(rect=[0, 0.16, 1, 1])
    save(fig, name)
    return R


def fig_ensemble(ALLM_pred, frozen, cvs, M, target, input_set, save, name, k_top=3, n_boot=1000):
    """Selected model vs the mean of the k best models by CV (ensemble) - SUPPLEMENTARY sensitivity analysis."""
    D = ALLM_pred[(ALLM_pred["target"] == target) & (ALLM_pred["input_set"] == input_set)]
    D = D[D["model_type"] == D["Type"].map(SPEC)]
    rows = []
    for (mt, tr), g in D.groupby(["model_type", "transition"]):
        combo = f"{target}__{input_set}__{trn_label(*_trans_years(tr))}__{mt}"
        cv = cvs[cvs["combo_id"] == combo].sort_values("cv_rmse")
        if cv.empty:
            continue
        top = cv["model"].head(k_top).tolist()
        sel = cv["model"].iloc[0]
        for typ in g["Type"].unique():
            gt = g[g["Type"] == typ]
            piv = gt.pivot_table(index=["rid", "Level_Key", "y_true", "is_transfer"], columns="model", values="y_pred").reset_index()
            if not set(top).issubset(piv.columns):
                continue
            piv["ens"] = piv[top].mean(axis=1)
            for lab, col in (("selected", sel), ("ensemble", "ens")):
                d = pd.DataFrame({"y_true": piv["y_true"], "y_pred": piv[col], "is_transfer": piv["is_transfer"], "Level_Key": piv["Level_Key"]})
                tr_ = d[d["is_transfer"]]
                if len(tr_) >= 2 and np.std(tr_["y_pred"]) > 0:
                    b1, b0 = np.polyfit(tr_["y_pred"], tr_["y_true"], 1)
                else:
                    b1, b0 = 1.0, 0.0
                te = d[~d["is_transfer"]]
                m = metrics_with_ci(te["y_true"], b0 + b1 * te["y_pred"], te["Level_Key"], n_boot)
                rows.append({"cultivar": typ, "transition": tr, "variant": lab, "RMSEP": m["RMSEP"], "lo": m["RMSEP_lo"], "hi": m["RMSEP_hi"],
                             "top_models": ",".join(top)})
    R = pd.DataFrame(rows)
    if R.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=False)
    for ax, typ in zip(axes, ["CEREN", "CIFTCI"]):
        for x, tr in enumerate(TRANS_ORDER):
            for off, lab, col in ((-0.17, "selected", "#D55E00"), (0.17, "ensemble", "#0072B2")):
                q = R[(R["cultivar"] == typ) & (R["transition"] == tr) & (R["variant"] == lab)]
                if q.empty:
                    continue
                q = q.iloc[0]
                ax.errorbar(x + off, q["RMSEP"], yerr=[[q["RMSEP"] - q["lo"]], [q["hi"] - q["RMSEP"]]], fmt="o", ms=4.5, color=col, capsize=2, elinewidth=1)
        ax.set_xticks(range(3))
        ax.set_xticklabels([TLAB[t] for t in TRANS_ORDER])
        ax.set_title(CLAB[typ], color=C_CULT[typ], fontweight="bold")
        ax.set_ylabel(f"RMSEP ({UNIT[target]})")
        ax.set_ylim(bottom=0)
        ax.yaxis.grid(True, color="#EEEEEE", lw=0.5)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color="#D55E00", label="CV-selected model (main analysis)"),
                            Line2D([], [], marker="o", ls="", color="#0072B2", label=f"Mean of top-{k_top} CV models")], fontsize=7.5, loc="lower center", ncol=2)
    _panel_labels(axes)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save(fig, name)
    return R


def _trans_years(tr):
    parts = tr.split("to")
    train = tuple(y for y, s in SEASONS.items() if s in parts[0].split("+"))
    test = [y for y, s in SEASONS.items() if s == parts[1]][0]
    return train, test


def fig_bias_drift(P, save):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9))
    for ax, tg in zip(axes, ["moisture", "bulk_density"]):
        D = _specific(P, tg, PRIMARY[tg]["input_set"])
        D = D[~D["is_transfer"]].copy()
        D["err"] = D["p_none"] - D["y_true"]
        for typ, off in (("CEREN", -0.18), ("CIFTCI", 0.18)):
            for x, tr in enumerate(TRANS_ORDER):
                g = D[(D["Type"] == typ) & (D["transition"] == tr)]
                if g.empty:
                    continue
                S = _level_sums(g["y_true"], g["p_none"], g["Level_Key"])
                idx = np.random.default_rng(0).integers(0, len(S), size=(2000, len(S)))
                bt = _metrics_from_sums(S[idx].sum(1))["Bias"]
                b = g["err"].mean()
                ax.bar(x + off, b, width=0.32, color=C_CULT[typ], alpha=0.85)
                lo, hi = np.percentile(bt, [2.5, 97.5])
                ax.errorbar(x + off, b, yerr=[[b - lo], [hi - b]], color="black", capsize=2, lw=0.8)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(3))
        ax.set_xticklabels([TLAB[t] for t in TRANS_ORDER])
        ax.set_ylabel(f"Mean error before correction ({UNIT[tg]})")
        ax.set_title(f"{TARGET_SHORT[tg]}, {INPUT_LAB[PRIMARY[tg]['input_set']].lower()} input")
        ax.yaxis.grid(True, color="#EEEEEE", lw=0.5)
        ax.set_axisbelow(True)
    fig.legend(handles=[Patch(color=C_CULT[t], label=CSHORT[t]) for t in ("CEREN", "CIFTCI")], fontsize=7.5,
               loc="lower center", ncol=2)
    _panel_labels(axes)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save(fig, "Fig13_bias_drift_between_seasons")


def fig_coverage(P, ckpt_dir, save, level=0.90):
    """90 % interval (+-q) from the training OOF residuals and its actual coverage in the incoming season - SUPPLEMENTARY / exploratory."""
    rows = []
    for tg in ("moisture", "bulk_density"):
        ins = PRIMARY[tg]["input_set"]
        for typ in ("CEREN", "CIFTCI"):
            for tr in TRANS_ORDER:
                combo = f"{tg}__{ins}__{tr}__{SPEC[typ]}"
                res = pickle_load_safe(Path(ckpt_dir) / f"{combo}.pkl")
                if res is None:
                    continue
                oof = res["oof_selected"]
                q = float(np.quantile(np.abs(oof["y_true"] - oof["oof_pred"]), level))
                g = P[(P["target"] == tg) & (P["input_set"] == ins) & (P["transition"] == tr) & (P["model_type"] == SPEC[typ]) & (P["Type"] == typ) & (~P["is_transfer"])]
                for cn, col in (("none", "p_none"), ("slope_bias", "p_slope_bias")):
                    cov = float((np.abs(g[col] - g["y_true"]) <= q).mean())
                    rows.append({"target": tg, "cultivar": typ, "transition": tr, "correction": cn, "q": q, "coverage": cov})
    R = pd.DataFrame(rows)
    if R.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), sharey=True)
    for ax, tg in zip(axes, ["moisture", "bulk_density"]):
        for typ, off in (("CEREN", -0.2), ("CIFTCI", 0.2)):
            for cn, hatch, alpha in (("none", None, 0.9), ("slope_bias", "///", 0.5)):
                xs, ys = [], []
                for x, tr in enumerate(TRANS_ORDER):
                    q = R[(R["target"] == tg) & (R["cultivar"] == typ) & (R["transition"] == tr) & (R["correction"] == cn)]
                    if not q.empty:
                        xs.append(x + off + (0.0 if cn == "none" else 0.0))
                        ys.append(q.iloc[0]["coverage"])
                w = 0.19
                ax.bar(np.array(xs) + (-w / 2 if cn == "none" else w / 2) * 1.0, ys, width=w, color=C_CULT[typ], alpha=alpha, hatch=hatch,
                       edgecolor="white" if hatch is None else C_CULT[typ], lw=0.5)
        ax.axhline(level, color="k", ls="--", lw=0.8)
        ax.set_xticks(range(3))
        ax.set_xticklabels([TLAB[t] for t in TRANS_ORDER])
        ax.set_ylim(0, 1.05)
        ax.set_title(TARGET_SHORT[tg] + f" — nominal {int(level * 100)}% interval")
    axes[0].set_ylabel("Empirical coverage in the test season")
    fig.legend(handles=[Patch(color=C_CULT["CEREN"], label="Ceren"), Patch(color=C_CULT["CIFTCI"], label="Çiftçi"),
                        Patch(facecolor="#888888", label="no correction"),
                        Patch(facecolor="#BBBBBB", hatch="///", edgecolor="#555555", label="slope–bias correction")],
               fontsize=7, loc="lower center", ncol=4)
    _panel_labels(axes)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    save(fig, "Fig14_interval_coverage")
    return R


# ----------------------------------------------------------------------------------------------
# Tables (Excel + LaTeX)
# ----------------------------------------------------------------------------------------------
def _fmt(v, d):
    return "" if pd.isna(v) else f"{v:.{d}f}"


def main_table(M, target):
    prim = PRIMARY[target]
    D = M[(M["target"] == target) & (M["input_set"] == prim["input_set"]) & (M["model_type"] == M["cultivar"].map(SPEC))].copy()
    d = DEC[target]
    D["cv"] = pd.Categorical(D["cultivar"], ["CEREN", "CIFTCI"], ordered=True)
    D["tr"] = pd.Categorical(D["transition"], TRANS_ORDER, ordered=True)
    D["co"] = pd.Categorical(D["correction"], ["none", "bias_only", "slope_bias"], ordered=True)
    D = D.sort_values(["cv", "tr", "co"])
    out = pd.DataFrame({
        "Cultivar": D["cultivar"].map(CSHORT), "Transition": D["transition"].map(TLAB), "Correction": D["correction"].map(CORR_LAB),
        "N (levels)": [f"{n} ({l})" for n, l in zip(D["N"], D["N_levels"])],
        f"RMSEP [95% CI] ({UNIT[target]})": [f"{_fmt(r, d)} [{_fmt(lo, d)}, {_fmt(hi, d)}]" for r, lo, hi in zip(D["RMSEP"], D["RMSEP_lo"], D["RMSEP_hi"])],
        "MAE": D["MAE"].map(lambda v: _fmt(v, d)), "R²": D["R2"].map(lambda v: _fmt(v, 3)),
        "Bias": D["Bias"].map(lambda v: f"{v:+.{d}f}"), "SEP": D["SEP"].map(lambda v: _fmt(v, d)),
        "RPD": D["RPD"].map(lambda v: _fmt(v, 2)), "RPIQ": D["RPIQ"].map(lambda v: _fmt(v, 2)),
        "Test outside train range (%)": D["pct_test_outside_train_range"].map(lambda v: _fmt(v, 0))})
    return out.reset_index(drop=True)


def latex_table(df, caption, label):
    def esc(s):
        return str(s).replace("\\", "").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
    cols = "l" * 3 + "r" * (len(df.columns) - 3)
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small", f"\\caption{{{esc(caption)}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{cols}}}", r"\toprule", " & ".join(esc(c) for c in df.columns) + r" \\", r"\midrule"]
    prev = None
    for _, r in df.iterrows():
        key = (r.iloc[0], r.iloc[1])
        if prev is not None and key != prev:
            lines.append(r"\addlinespace")
        prev = key
        lines.append(" & ".join(esc(v) for v in r.tolist()) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def data_summary_table(rec):
    rows = []
    for typ in ("CEREN", "CIFTCI"):
        for yr in (2014, 2015, 2016, 2017):
            d = rec[(rec["Type"] == typ) & (rec["Year"] == yr)]
            rows.append({"Cultivar": CSHORT[typ], "Season": f"{SEASONS[yr]} ({yr})", "MC levels": d["Level_Key"].nunique(),
                         "Packing states per level": int(round(len(d) / d["Level_Key"].nunique())), "Records": len(d),
                         "MC min–max (% w.b.)": f"{d['MC'].min():.2f}–{d['MC'].max():.2f}",
                         "BD min–max (g cm⁻³)": f"{d['BD'].min():.3f}–{d['BD'].max():.3f}",
                         "MC mean ± SD": f"{d['MC'].mean():.2f} ± {d['MC'].std():.2f}"})
    return pd.DataFrame(rows)


def design_table(tinfo, M, rec):
    rows = []
    for tr, (ty, te) in zip(TRANS_ORDER, [(t, y) for t, y in TRANSITIONS]):
        for typ in ("CEREN", "CIFTCI"):
            r = tinfo[(tinfo["Year"] == te) & (tinfo["Type"] == typ)].iloc[0]
            ntrain = int((rec["Year"].isin(ty) & (rec["Type"] == typ)).sum())
            q = M[(M["transition"] == tr) & (M["cultivar"] == typ) & (M["model_type"] == SPEC[typ]) & (M["target"] == "moisture")]
            rows.append({"Transition": TLAB[tr], "Cultivar": CSHORT[typ], "Training records (specific)": ntrain,
                         "Transfer levels": r["transfer_levels"].replace(";", "; "), "Transfer records": int(r["n_transfer_records"]),
                         "Test records": int(r["n_test_records"]),
                         "Test MC levels": int(q["N_levels"].iloc[0]) if not q.empty else ""})
    return pd.DataFrame(rows)


def write_excel(path, sheets, readme):
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame({"Sheet": [s[0] for s in readme], "Content": [s[1] for s in readme]}).to_excel(xw, sheet_name="README", index=False)
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
        thin = Side(style="thin", color="BBBBBB")
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
            for c in ws[1]:
                c.font = Font(bold=True, color="FFFFFF")
                c.fill = PatternFill("solid", start_color="1F4E79")
                c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
            ws.row_dimensions[1].height = 32
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:60])
                ws.column_dimensions[get_column_letter(col[0].column)].width = min(max(10, width + 2), 48)
            for row in ws.iter_rows(min_row=2):
                for c in row:
                    c.border = Border(bottom=thin)
                    if isinstance(c.value, float):
                        c.number_format = "0.0000"


# ----------------------------------------------------------------------------------------------
# Main workflow
# ----------------------------------------------------------------------------------------------
def build_paper_package(out_root=RESULTS_ROOT / "chronological", data_root=DATA_ROOT, n_boot=2000, formats=("pdf", "png"), dpi=600):
    setup_style()
    out_root = Path(out_root)
    T = out_root / "tables"
    mk = out_root / "paper"
    (mk / "tables").mkdir(parents=True, exist_ok=True)
    save = _Saver(mk / "figures", formats, dpi)
    print(f"Paper outputs -> {mk}")
    P = pd.read_csv(T / "predictions_all.csv")
    frozen = pd.read_csv(T / "frozen_models.csv")
    cvs = pd.read_csv(T / "model_cv_scores.csv")
    tinfo = pd.read_csv(T / "transfer_subsets.csv")
    rec, blocks = build_records(load_dataset("Pooled", data_root))
    allpred = pd.read_csv(T / "all_model_predictions.csv") if (T / "all_model_predictions.csv").exists() else None
    allm_scores = pd.read_csv(T / "all_models_test_scores.csv") if (T / "all_models_test_scores.csv").exists() else None
    if allm_scores is not None and "model" not in allm_scores.columns:
        allm_scores = allm_scores.rename(columns={"family": "model"})
    if allm_scores is not None and "correction" not in allm_scores.columns:   # older (v1) format: a subset only, no correction column
        allm_scores = None

    print("  Statistics (bootstrap over levels) ...")
    M = build_metric_table(P, rec, n_boot)
    C = build_comparisons(P, n_boot)
    TSm = transfer_sensitivity(P, "moisture", PRIMARY["moisture"]["input_set"])
    TSb = transfer_sensitivity(P, "bulk_density", PRIMARY["bulk_density"]["input_set"])

    print("  Figures ...")
    fig_design(rec, tinfo, save)
    fig_data_overview(rec, save)
    fig_spectra(rec, blocks, save)
    for tg, nm in (("moisture", ["Fig04_moisture_predicted_vs_reference", "Fig05_moisture_RMSEP_configurations",
                                 "Fig06_moisture_error_vs_reference", "Fig07_moisture_paired_comparisons", "Fig08_moisture_transfer_sensitivity"]),
                   ("bulk_density", ["Fig15_bd_predicted_vs_reference", "Fig16_bd_RMSEP_configurations", "Fig17_bd_error_vs_reference",
                                     "Fig18_bd_paired_comparisons", "Fig19_bd_transfer_sensitivity"])):
        ins = PRIMARY[tg]["input_set"]
        fig_scatter(P, M, tg, ins, save, nm[0])
        fig_dotwhisker(M, tg, save, nm[1])
        fig_residuals(P, tg, ins, save, nm[2])
        fig_forest(C, tg, save, nm[3])
        fig_transfer(TSm if tg == "moisture" else TSb, tg, save, nm[4])
    fig_model_overview(frozen, cvs, save)
    regret = ens = None
    if allm_scores is not None:
        fig_model_heatmap(allm_scores, frozen, "moisture", PRIMARY["moisture"]["input_set"], save, "Fig10_moisture_all_models_heatmap")
        fig_model_heatmap(allm_scores, frozen, "bulk_density", PRIMARY["bulk_density"]["input_set"], save, "Fig20_bd_all_models_heatmap")
        regret = fig_selection_regret(allm_scores, frozen, P, "moisture", PRIMARY["moisture"]["input_set"], save, "Fig11_moisture_selection_regret")
    if allpred is not None:
        ens = fig_ensemble(allpred, frozen, cvs, M, "moisture", PRIMARY["moisture"]["input_set"], save, "Fig12_moisture_ensemble_vs_selected")
    fig_bias_drift(P, save)
    cov = fig_coverage(P, out_root / "checkpoints", save)

    print("  Tables ...")
    t1 = data_summary_table(rec)
    t2 = design_table(tinfo, M, rec)
    t3 = main_table(M, "moisture")
    t4 = main_table(M, "bulk_density")
    prim_m = PRIMARY["moisture"]
    cmp_summary = C.groupby(["comparison", "target", "verdict"]).size().unstack(fill_value=0).reset_index()
    t5 = C[(C["input_set"].isin([prim_m["input_set"], "spectra->microwave", "microwave->full", PRIMARY["bulk_density"]["input_set"]]))].copy()
    t_models = frozen[["combo_id", "target", "input_set", "transition", "model_type", "selected", "selected_params", "selected_cv_rmse", "n_train", "n_features"]]
    keepM = M.copy()
    sheets = {"T1_Data": t1, "T2_Design": t2, "T3_Moisture_main": t3, "T4_BulkDensity_main": t4,
              "T5_Comparison_summary": cmp_summary, "T6_Transfer_sensitivity_MC": TSm, "T7_Frozen_models": t_models,
              "S1_All_configs_full": keepM, "S2_Comparisons_p_Holm": C, "S3_Model_CV_scores": cvs,
              "S4_Transfer_sensitivity_BD": TSb}
    if allm_scores is not None:
        sheets["S5_All_models_test"] = allm_scores
    if regret is not None:
        sheets["S6_Selection_regret_MC"] = regret
    if ens is not None:
        sheets["S7_Ensemble_vs_selected"] = ens
    if cov is not None:
        sheets["S8_Interval_coverage"] = cov
    readme = [("T1_Data", "Dataset summary by cultivar and season (levels, records, MC and BD ranges)."),
              ("T2_Design", "Chronological design: training records, transfer levels, test records per transition and cultivar."),
              ("T3_Moisture_main", "Moisture: primary configuration (cultivar-specific, full input) for all three corrections; RMSEP with 95% bootstrap CI over test MC levels."),
              ("T4_BulkDensity_main", "Bulk density: primary configuration (cultivar-specific, microwave input)."),
              ("T5_Comparison_summary", "Counts of paired comparisons by verdict (CI excludes 0 → better/worse; otherwise uncertain)."),
              ("T6_Transfer_sensitivity_MC", "RMSEP versus number of transfer MC levels (post hoc, recomputed from saved predictions)."),
              ("T7_Frozen_models", "Model selected by CV in each combination and its hyperparameters."),
              ("S1–S8", "Supplementary: all configurations with CIs; all comparisons with bootstrap p (Holm-adjusted, exploratory); model CV scores; "
                        "all models' test scores; selection regret; ensemble sensitivity; interval coverage."),
              ("Note", "Primary configuration was fixed before inspecting test results (see paper_outputs.PRIMARY). Everything else is "
                       "supplementary/sensitivity analysis. Bootstrap CIs resample test MC levels (the independent unit); with 5–17 test levels "
                       "per cultivar they are approximate. RPD = SD(reference)/RMSEP; RPIQ = IQR(reference)/RMSEP; R² = 1 − SSE/SST.")]
    write_excel(mk / "tables" / "Paper_Tables.xlsx", sheets, readme)
    for n, df in (("T3_Moisture_main", t3), ("T4_BulkDensity_main", t4), ("T2_Design", t2), ("T1_Data", t1)):
        df.to_csv(mk / "tables" / f"{n}.csv", index=False, encoding="utf-8-sig")
    (mk / "tables" / "Table3_Moisture_main.tex").write_text(latex_table(
        t3, "Moisture content prediction in the test seasons for cultivar-specific models (full input set). RMSEP with 95% bootstrap CI over test MC levels.", "tab:mc_main"), encoding="utf-8")
    (mk / "tables" / "Table4_BulkDensity_main.tex").write_text(latex_table(
        t4, "Bulk density prediction in the test seasons for cultivar-specific models (microwave input set).", "tab:bd_main"), encoding="utf-8")

    captions = """# Figure captions (draft; edit before submission)

Fig01. Chronological design. Models were developed on earlier seasons and evaluated on the next season; in each incoming season two MC levels per cultivar (25th and 75th percentile of level MC) formed the transfer subset used only to fit the correction, and the remaining levels formed the test set.
Fig02. Data overview: (a) moisture content and (b) bulk density by season and cultivar (points: records; boxes: median and IQR); (c) bulk density versus moisture content.
Fig03. Level-mean transmission spectra of the S3 (2016) season for each cultivar, coloured by the moisture content of the MC level: attenuation, phase shift, dielectric constant and loss factor.
Fig04. Predicted versus reference moisture content for the cultivar-specific models (full input set), without correction and after slope–bias correction. Filled points: test records; open circles: transfer records; dashed line: 1:1. Metrics refer to test records.
Fig05. RMSEP (± 95% bootstrap CI over test MC levels) of cultivar-specific models by input set, correction and transition.
Fig06. Prediction error versus reference moisture content; black markers: mean ± SD within each MC level (test records).
Fig07. Paired RMSEP differences (95% bootstrap CI) for (a) correction versus none, (b) pooled versus cultivar-specific models, (c) richer versus poorer input sets. Negative values favour B.
Fig08. Sensitivity of RMSEP to the number of transfer MC levels (k); bands: 5th–95th percentile over choices of the transfer levels. k = 2 is the main analysis. Post hoc, recomputed from saved predictions.
Fig09. Model selection overview: (a) how often each model was selected; (b) CV-RMSE of every model relative to the best model in the same combination.
Fig10. Test RMSEP of all models (cultivar-specific, full input), without and with slope–bias correction; red circles mark the model selected by CV. Not used for selection.
Fig11. Selection regret: test RMSEP of the CV-selected model versus the best, median and worst model on the test season.
Fig12. Mean of the top-3 CV models versus the CV-selected model (slope–bias corrected; 95% bootstrap CI). Supplementary sensitivity analysis.
Fig13. Mean prediction error before correction by season and cultivar (95% bootstrap CI over MC levels): between-season drift.
Fig14. Empirical coverage in the test season of 90% intervals derived from out-of-fold training residuals (exploratory).
Fig15–Fig20. Bulk-density counterparts of Fig04–Fig08 and Fig10.
"""
    (mk / "Figure_Captions_Draft.md").write_text(captions, encoding="utf-8")
    (mk / "README_paper.txt").write_text(
        "PAPER OUTPUTS (generated automatically)\n\nfigures/  : PDF (vector) and 600 dpi PNG\ntables/   : Paper_Tables.xlsx (explanations on the README sheet), "
        "LaTeX (Table3, Table4) and CSV\nFigure_Captions_Draft.md : figure captions (draft)\n\nPrimary (pre-specified) configuration: "
        f"{PRIMARY}\nEverything is computed from the saved predictions; there is no retraining.\nFigure sizes: single column ~3.5 in, double column 7.1 in "
        "(adjust to the journal's guide for authors).\n", encoding="utf-8")
    n_iss = sum(len(v) for v in save.issues.values())
    print(f"  Readability check: {len(save.saved)} figures, {n_iss} warning(s)" + ("" if n_iss == 0 else f" ({list(save.issues)})"))
    print(f"  Done: {len(save.saved)} figures, {len(sheets) + 1} Excel sheets -> {mk}")
    return {"M": M, "C": C, "TS_moisture": TSm, "TS_bd": TSb}
