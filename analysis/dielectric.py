# -*- coding: utf-8 -*-
"""
Dielectric relations (Section 2.2) and synthetic offsets (Section 2.3.2).

    eps'(f)  ~ (1 + dphi(f) * lambda0(f) / (360 d))^2
    eps''(f) ~ |dA(f)| * lambda0(f) * sqrt(eps'(f)) / (8.686 pi d)          d = 200 mm

SIGN CONVENTION: the 'Attenuation' column of the archive stores dA = 20 log10(S21,s / S21,0) (mostly negative for a
filled box; range -22.9 ... +0.8 dB). The paper defines it with the opposite sign (S21,0 / S21,s). The stored eps''
values are reproduced with |dA| (max. difference 0.006; 0.003 for eps', because the values are stored with 3 decimals).
This module uses the archive convention.

The offsets are applied with their physical meaning:
    "dA"    : +x dB extra attenuation (attenuator)      -> the archive dA decreases by x (more negative)
    "dphi"  : +x degrees extra phase delay (extension)  -> dphi increases by x
    "rho_g" : +x g cm-3 (BD equivalent of a mass offset) -> only rho_g and the rho_g-normalised terms change
eps', eps'', X and the rho_g-normalised terms are recomputed from the perturbed values. So that the rounding of the
stored values does not mix with the offset effect, the CHANGE computed from the relations is added to the stored eps
values; a zero offset leaves the record unchanged.
"""
from __future__ import annotations

import numpy as np

from common import FREQS  # noqa: E402

C0 = 299_792_458.0
D_SAMPLE_M = 0.200            # sample thickness d (inner box length along the antenna axis), Section 2.2
LAMBDA0_M = C0 / (FREQS * 1e6)

OFFSET_KINDS = ("dA", "dphi", "rho_g")
OFFSET_UNITS = {"dA": "dB", "dphi": "deg", "rho_g": "g cm-3"}
# test conditions of Section 2.3.2 (test conditions, NOT detection limits)
OFFSET_GRID = {"dA": (0.25, 0.5, 1.0), "dphi": (2.0, 5.0, 10.0), "rho_g": (0.005, 0.010, 0.020)}


def eps_prime(dphi):
    return (1.0 + np.asarray(dphi, float) * LAMBDA0_M / (360.0 * D_SAMPLE_M)) ** 2


def eps_double_prime(dA, e1):
    return np.abs(np.asarray(dA, float)) * LAMBDA0_M * np.sqrt(e1) / (8.686 * np.pi * D_SAMPLE_M)


def make_blocks(dA, dphi, e1, e2, rho_g):
    """The same 7 blocks as build_records of the chronological analysis (dA, dphi, e1, e2, X, e1n, e2n)."""
    rho = np.asarray(rho_g, float)[:, None]
    return {"dA": dA, "dphi": dphi, "e1": e1, "e2": e2, "X": dA / dphi, "e1n": (e1 - 1.0) / rho, "e2n": e2 / rho}


def perturb(blocks, rho_g, kind, value):
    """Returns an offset copy of (blocks, rho_g). kind: 'dA' | 'dphi' | 'rho_g'; value in physical units (>= 0)."""
    if kind not in OFFSET_KINDS:
        raise ValueError(f"unknown offset type: {kind!r}")
    dA, dphi = np.array(blocks["dA"], float), np.array(blocks["dphi"], float)
    e1, e2 = np.array(blocks["e1"], float), np.array(blocks["e2"], float)
    rho = np.array(rho_g, float)
    if value == 0:
        return make_blocks(dA, dphi, e1, e2, rho), rho
    if kind == "dA":
        dA_new = dA - value
        e1_calc = eps_prime(dphi)
        e2 = e2 + (eps_double_prime(dA_new, e1_calc) - eps_double_prime(dA, e1_calc))
        dA = dA_new
    elif kind == "dphi":
        dphi_new = dphi + value
        e1_old, e1_new = eps_prime(dphi), eps_prime(dphi_new)
        e1 = e1 + (e1_new - e1_old)
        e2 = e2 + (eps_double_prime(dA, e1_new) - eps_double_prime(dA, e1_old))
        dphi = dphi_new
    else:
        rho = rho + value
    return make_blocks(dA, dphi, e1, e2, rho), rho


def offset_label(kind, value):
    return "clean" if value == 0 else f"{kind}+{value:g}"
