# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

import logging

import numpy as np

logger = logging.getLogger("DecayCore.dsp")

# Try to import Rust DSP extension
try:
    from decaycore_dsp import slope_passes_rs as _slope_passes_rs
    from decaycore_dsp import slope_passes_asym_rs as _slope_passes_asym_rs

    _DSP_RUST_AVAILABLE = True
except ImportError:
    _DSP_RUST_AVAILABLE = False


def _slope_passes(g, x, max_db_per_oct):
    n = x.size
    for k in range(1, n):
        dx = x[k] - x[k - 1]
        if dx <= 0.0:
            continue
        lim = max_db_per_oct * dx
        dg = g[k] - g[k - 1]
        if dg > lim:
            g[k] = g[k - 1] + lim
        elif dg < -lim:
            g[k] = g[k - 1] - lim
    for k in range(n - 2, -1, -1):
        dx = x[k + 1] - x[k]
        if dx <= 0.0:
            continue
        lim = max_db_per_oct * dx
        dg = g[k] - g[k + 1]
        if dg > lim:
            g[k] = g[k + 1] + lim
        elif dg < -lim:
            g[k] = g[k + 1] - lim
    return g


def _slope_passes_asym(g, x, boost, cut):
    _slope_passes_asym_forward(g, x, boost, cut)
    _slope_passes_asym_backward(g, x, boost, cut)
    return g


def _slope_passes_asym_forward(g, x, boost, cut):
    n = x.size
    for k in range(1, n):
        dx = x[k] - x[k - 1]
        if dx <= 0.0:
            continue
        dg = g[k] - g[k - 1]
        lim = (boost if dg > 0.0 else cut) * dx
        if (dg > 0.0 and boost <= 0.0) or (dg < 0.0 and cut <= 0.0):
            continue
        if dg > lim:
            g[k] = g[k - 1] + lim
        elif dg < -lim:
            g[k] = g[k - 1] - lim


def _slope_passes_asym_backward(g, x, boost, cut):
    n = x.size
    for k in range(n - 2, -1, -1):
        dx = x[k + 1] - x[k]
        if dx <= 0.0:
            continue
        dg = g[k] - g[k + 1]
        lim = (boost if dg > 0.0 else cut) * dx
        if (dg > 0.0 and boost <= 0.0) or (dg < 0.0 and cut <= 0.0):
            continue
        if dg > lim:
            g[k] = g[k + 1] + lim
        elif dg < -lim:
            g[k] = g[k + 1] - lim


def _apply_slope_passes(g, x, max_db_per_oct):
    """Dispatch to Rust slope limiter if available, fallback to pure Python."""
    if _DSP_RUST_AVAILABLE:
        try:
            g_arr = np.ascontiguousarray(g, dtype=np.float64)
            x_arr = np.ascontiguousarray(x, dtype=np.float64)
            return _slope_passes_rs(g_arr, x_arr, float(max_db_per_oct))
        except (TypeError, ValueError) as exc:
            logger.warning("Rust slope limiter rejected its input; using Python fallback: %s", exc)
    return _slope_passes(g, x, max_db_per_oct)


def _apply_slope_passes_asym(g, x, boost, cut):
    """Dispatch to Rust asym slope limiter if available, fallback to pure Python."""
    if _DSP_RUST_AVAILABLE:
        try:
            g_arr = np.ascontiguousarray(g, dtype=np.float64)
            x_arr = np.ascontiguousarray(x, dtype=np.float64)
            return _slope_passes_asym_rs(g_arr, x_arr, float(boost), float(cut))
        except (TypeError, ValueError) as exc:
            logger.warning("Rust asymmetric slope limiter rejected its input; using Python fallback: %s", exc)
    return _slope_passes_asym(g, x, boost, cut)


def soft_clip_gain(gain_db, max_boost_db, max_cut_db):
    """Rajaa korjauskayran pehmeasti tanh-funktiolla.

    Positiiviset arvot paastetaan muuttumattomina boost-kattoon asti.
    Katon ylittava osa pehmennetaan, ja myohemmat hard clamp -vaiheet
    pitavat lopullisen arvon asetetussa katossa.

    Negatiiviset arvot paastetaan muuttumattomina cut-kattoon asti.
    Rajaa ylittava osa pehmennetaan samalla mekanismilla kuin boost.
    """
    g = np.asarray(gain_db, dtype=float)
    out = np.empty_like(g)
    pos = g > 0
    neg = ~pos
    if np.any(pos):
        mb = float(max_boost_db) if max_boost_db > 0 else 0.0
        if mb > 0:
            pos_vals = g[pos]
            over = np.maximum(0.0, pos_vals - mb)
            out[pos] = np.where(
                pos_vals <= mb,
                pos_vals,
                mb + mb * np.tanh(over / (mb + 1e-12)),
            )
        else:
            out[pos] = 0.0
    if np.any(neg):
        mc = float(max_cut_db) if max_cut_db > 0 else 0.0
        if mc > 0:
            cut_vals = -g[neg]
            over_c = np.maximum(0.0, cut_vals - mc)
            out[neg] = -np.where(
                cut_vals <= mc,
                cut_vals,
                mc + mc * np.tanh(over_c / (mc + 1e-12)),
            )
        else:
            out[neg] = g[neg]
    return out


def limit_slope_per_octave(freq_axis, gain_db, max_db_per_oct=12.0):
    """Rajoittaa gain-kayran muutosnopeuden (dB/oktaavi) symmetrisesti.

    Toteutus tekee ensin eteenpain- ja sitten taaksepain-passit, jotta raja
    toteutuu molempiin suuntiin koko taajuusakselilla.
    """
    f = np.asarray(freq_axis, dtype=float)
    g = np.asarray(gain_db, dtype=float).copy()
    max_db_per_oct = float(max_db_per_oct)
    if max_db_per_oct <= 0:
        return g

    idx = np.where(f > 0)[0]
    if idx.size < 3:
        return g

    ii = idx
    x = np.log2(f[ii])
    g_sub = g[ii].copy()
    g[ii] = _apply_slope_passes(g_sub, x, max_db_per_oct)
    return g


def limit_slope_per_octave_asym(freq_axis, gain_db, max_db_per_oct_boost, max_db_per_oct_cut):
    """Rajoittaa gain-kayran muutosnopeuden epasymmetrisesti (dB/oktaavi).

    Nouseville kohdille kaytetaan boost-rajaa ja laskeville kohdille cut-rajaa.
    Etu- ja takapassi pitavat rajoituksen vakaana koko kayralla.
    """
    f = np.asarray(freq_axis, dtype=float)
    g = np.asarray(gain_db, dtype=float).copy()

    b = float(max_db_per_oct_boost or 0.0)
    c = float(max_db_per_oct_cut or 0.0)
    if b <= 0 and c <= 0:
        return g

    idx = np.where(f > 0.0)[0]
    if idx.size < 2:
        return g

    lf = np.log2(f[idx])
    g_sub = g[idx].copy()
    g[idx] = _apply_slope_passes_asym(g_sub, lf, b, c)
    return g


def build_slope_limit_envelope(
    freq_axis,
    target_db,
    *,
    mag_c_min: float,
    mag_c_max: float,
    max_slope_boost_db_per_oct: float,
    max_slope_cut_db_per_oct: float,
):
    """Rakentaa UI:ta varten visuaalisen slope-rajoitusvaipan.

    Palauttaa tuple-arvon `(env_lo, env_hi, pivot_hz)`, jossa vaippa on
    laskettu korjausalueelle [mag_c_min, mag_c_max]. Jos syotedata ei ole
    kayttokelpoinen tai rajat ovat pois paalta, palauttaa `(None, None, None)`.
    """
    f = np.asarray(freq_axis, dtype=float)
    t = np.asarray(target_db, dtype=float)

    if f.size < 8 or t.size != f.size:
        return None, None, None

    b = float(max_slope_boost_db_per_oct or 0.0)
    c = float(max_slope_cut_db_per_oct or 0.0)
    if b <= 0.0 and c <= 0.0:
        return None, None, None

    try:
        cmin = float(mag_c_min or 0.0)
        cmax = float(mag_c_max or 0.0)
    except (TypeError, ValueError, OverflowError):
        cmin, cmax = 0.0, 0.0
    if not (np.isfinite(cmin) and np.isfinite(cmax) and cmin > 0 and cmax > cmin):
        return None, None, None

    pivot_hz = float(np.sqrt(cmin * cmax))
    pivot_hz = float(
        np.clip(
            pivot_hz,
            float(np.min(f[f > 0])) if np.any(f > 0) else 1.0,
            float(np.max(f)),
        )
    )
    pivot_idx = int(np.argmin(np.abs(f - pivot_hz)))

    upper_delta = np.zeros_like(t, dtype=float)
    lower_delta = np.zeros_like(t, dtype=float)

    logf = np.log2(np.maximum(f, 1e-12))
    dx = np.maximum(np.diff(logf), 0.0)  # octave step per bin pair, size = f.size - 1

    # Right side: pivot_idx+1 .. end — cumsum of dx starting at pivot_idx
    if pivot_idx < f.size - 1:
        right_cs = np.cumsum(dx[pivot_idx:])
        upper_delta[pivot_idx + 1 :] = (max(0.0, b)) * right_cs
        lower_delta[pivot_idx + 1 :] = (max(0.0, c)) * right_cs

    # Left side: 0 .. pivot_idx-1 — suffix sums of dx[:pivot_idx]
    if pivot_idx > 0:
        left_cs = np.cumsum(dx[:pivot_idx][::-1])[::-1]
        upper_delta[:pivot_idx] = (max(0.0, b)) * left_cs
        lower_delta[:pivot_idx] = (max(0.0, c)) * left_cs

    env_hi = t + upper_delta
    env_lo = t - lower_delta

    band = (f >= cmin) & (f <= cmax)
    env_hi = np.where(band, env_hi, np.nan)
    env_lo = np.where(band, env_lo, np.nan)

    return env_lo, env_hi, pivot_hz
