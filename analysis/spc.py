# -*- coding: utf-8 -*-
"""
Consistency monitoring (Section 2.3.1):  r = rho_g - rho_d,  z = (r - mu0) / sigma0

    Shewhart : alarm |z| > 3, warning |z| > 2 (logged only; triggers no command)
    EWMA     : E_t = lam z_t + (1 - lam) E_{t-1},  E_0 = 0,  alarm |E_t| > L sigma_E(t)          lam = 0.20, L = 3.0
               sigma_E(t) = sqrt(lam / (2 - lam) * (1 - (1 - lam)^(2t)))   (exact, time-dependent limit)
    CUSUM    : C+_t = max(0, C+_{t-1} + z_t - k),  C-_t = max(0, C-_{t-1} - z_t - k),  alarm C+ > h or C- > h
               k = 0.5, h = 5
    Combined : alarm if any of the three limits is exceeded.

SpcState is vectorised: every element is an independent monitoring stream. The evaluation processes thousands of
sequences at once and the twin a single stream (n = 1) with the same code; there is only one implementation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

CHARTS = ("shewhart", "ewma", "cusum", "combined")


@dataclass(frozen=True)
class SpcSettings:
    warn: float = 2.0
    shewhart: float = 3.0
    lam: float = 0.20
    L: float = 3.0
    k: float = 0.5
    h: float = 5.0
    ewma_exact_limits: bool = True

    def as_dict(self):
        return asdict(self)


class SpcState:
    def __init__(self, n=1, settings: SpcSettings = SpcSettings()):
        self.s = settings
        self.n = int(n)
        self.ewma = np.zeros(self.n)
        self.cpos = np.zeros(self.n)
        self.cneg = np.zeros(self.n)
        self.t = np.zeros(self.n, dtype=int)

    def reset(self, mask=None):
        """Resets EWMA and CUSUM (in live operation only after a recorded physical removal of the fault)."""
        m = np.ones(self.n, bool) if mask is None else np.asarray(mask, bool)
        self.ewma[m] = 0.0
        self.cpos[m] = 0.0
        self.cneg[m] = 0.0
        self.t[m] = 0

    def ewma_limit(self):
        s = self.s
        if s.ewma_exact_limits:
            return s.L * np.sqrt(s.lam / (2.0 - s.lam) * (1.0 - (1.0 - s.lam) ** (2 * self.t)))
        return np.full(self.n, s.L * np.sqrt(s.lam / (2.0 - s.lam)))

    def update(self, z):
        s = self.s
        z = np.broadcast_to(np.asarray(z, float), (self.n,))
        self.t = self.t + 1
        self.ewma = s.lam * z + (1.0 - s.lam) * self.ewma
        self.cpos = np.maximum(0.0, self.cpos + z - s.k)
        self.cneg = np.maximum(0.0, self.cneg - z - s.k)
        lim = self.ewma_limit()
        out = {"z": z.copy(), "ewma": self.ewma.copy(), "ewma_limit": lim, "cpos": self.cpos.copy(),
               "cneg": self.cneg.copy(), "warn": np.abs(z) > s.warn, "shewhart": np.abs(z) > s.shewhart,
               "ewma_alarm": np.abs(self.ewma) > lim, "cusum_alarm": (self.cpos > s.h) | (self.cneg > s.h)}
        out["combined"] = out["shewhart"] | out["ewma_alarm"] | out["cusum_alarm"]
        return out


def run_sequences(Z, settings: SpcSettings = SpcSettings()):
    """Z: (n_sequences, T) matrix of z; every row is an independent monitoring stream starting from zero.
    Returns: (n_sequences, T) alarm matrices for every chart + intermediate statistics."""
    Z = np.atleast_2d(np.asarray(Z, float))
    st = SpcState(Z.shape[0], settings)
    keys = ("z", "ewma", "ewma_limit", "cpos", "cneg", "warn", "shewhart", "ewma_alarm", "cusum_alarm", "combined")
    cols = {k: [] for k in keys}
    for t in range(Z.shape[1]):
        o = st.update(Z[:, t])
        for k in keys:
            cols[k].append(o[k])
    out = {k: np.column_stack(v) for k, v in cols.items()}
    return {"shewhart": out["shewhart"], "ewma": out["ewma_alarm"], "cusum": out["cusum_alarm"],
            "combined": out["combined"], "warn": out["warn"], "stats": out}
