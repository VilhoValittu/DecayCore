# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("DecayCore")


def _comparison_stats_interp(freq_cmp, f, y):
    y = np.asarray(y, dtype=float)
    if y.shape != f.shape:
        return None
    return np.interp(freq_cmp, f, y)


def _comparison_stats_write_output(
    out: dict,
    *,
    stats: dict,
    freq_cmp,
    m_cmp,
    t_cmp,
    g_cmp,
    c_cmp,
    mm_cmp,
    bw_cmp,
    cmp_offset_db: float,
) -> dict:
    out["analysis_mode"] = "comparison"
    out["cmp_ref_fs"] = int(out.get("cmp_ref_fs", 44100))
    out["cmp_ref_taps"] = int(out.get("cmp_ref_taps", 65536))
    out["cmp_freq_axis"] = freq_cmp.tolist()
    out["cmp_measured_mags"] = m_cmp.tolist()
    out["cmp_target_mags"] = t_cmp.tolist()
    if g_cmp is not None:
        out["cmp_filter_mags"] = g_cmp.tolist()
    if c_cmp is not None:
        c_clip = np.clip(c_cmp, 0.0, 1.0)
        out["cmp_confidence_mask"] = c_clip.tolist()
        out["cmp_avg_confidence"] = float(np.mean(c_clip) * 100.0)
    if mm_cmp is not None:
        out["cmp_mag_mask"] = (np.asarray(mm_cmp, dtype=float) > 0.5).astype(float).tolist()
    if bw_cmp is not None:
        out["cmp_afdw_bw_oct"] = np.clip(bw_cmp, 1.0 / 96.0, 1.0 / 3.0).tolist()
        out["cmp_offset_db"] = float(cmp_offset_db)
    smart_scan_range = out.get("smart_scan_range")
    if isinstance(smart_scan_range, (list, tuple)) and len(smart_scan_range) == 2:
        out["cmp_smart_scan_range"] = [float(smart_scan_range[0]), float(smart_scan_range[1])]
    if "eff_target_db" in stats and stats.get("eff_target_db") is not None:
        try:
            v = float(stats.get("eff_target_db"))
            if np.isfinite(v):
                out["eff_target_db"] = v
                out["cmp_eff_target_db"] = v
        except (
            AttributeError,
            TypeError,
            ValueError,
            KeyError,
            IndexError,
            RuntimeError,
            OSError,
            ImportError,
            ModuleNotFoundError,
            NameError,
        ):
            logger.exception("eff_target_db parse in comparison stats")
    if "target_level_db_window" in stats:
        out["cmp_target_level_db_window"] = stats.get("target_level_db_window")
    return out


def _comparison_stats_float_array(v):
    try:
        arr = np.asarray(v, dtype=float)
    except (
        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        RuntimeError,
        OSError,
        ImportError,
        ModuleNotFoundError,
        NameError,
    ):
        return None
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _comparison_stats_prepare_freq_cmp(f, ref_fs: int, ref_taps: int):
    nfft = int(ref_taps)
    nfft = max(nfft, 1024)
    if (nfft % 2) != 0:
        nfft += 1
    fmax = min(float(ref_fs) / 2.0, float(np.max(f)))
    if fmax <= 10.0:
        return None
    return np.linspace(0.0, fmax, nfft // 2 + 1)


def _comparison_stats_get_raw_arrays(stats: dict):
    stats = stats or {}
    out = dict(stats)
    if str(out.get("analysis_mode", "native")).lower() == "comparison" and ("cmp_freq_axis" in out):
        return None
    f = out.get("freq_axis")
    m = out.get("measured_mags")
    t = out.get("target_mags")
    g = out.get("filter_mags")
    c = out.get("confidence_mask")
    mm = out.get("mag_mask", out.get("mask_c"))
    if f is None or m is None or t is None:
        return None
    return out, f, m, t, g, c, mm


def _comparison_stats_coerce_arrays(raw):
    if raw is None:
        return None
    out, f, m, t, g, c, mm = raw
    f = _comparison_stats_float_array(f)
    m = _comparison_stats_float_array(m)
    t = _comparison_stats_float_array(t)
    g = _comparison_stats_float_array(g) if g is not None else None
    c = _comparison_stats_float_array(c) if c is not None else None
    mm = _comparison_stats_float_array(mm) if mm is not None else None
    if f is None or m is None or t is None:
        return None
    if f.ndim != 1 or f.size < 32 or m.ndim != 1 or t.ndim != 1:
        return None
    if m.size != f.size or t.size != f.size:
        return None
    if (g is not None) and ((g.ndim != 1) or (g.size != f.size)):
        g = None
    if (c is not None) and ((c.ndim != 1) or (c.size != f.size)):
        c = None
    if (mm is not None) and ((mm.ndim != 1) or (mm.size != f.size)):
        mm = None
    return out, f, m, t, g, c, mm


def _comparison_stats_prepare_arrays(stats: dict, ref_fs: int, ref_taps: int):
    coerced = _comparison_stats_coerce_arrays(_comparison_stats_get_raw_arrays(stats))
    if coerced is None:
        return None
    out, f, m, t, g, c, mm = coerced
    freq_cmp = _comparison_stats_prepare_freq_cmp(f, ref_fs, ref_taps)
    if freq_cmp is None:
        return None
    return out, f, freq_cmp, m, t, g, c, mm


def _comparison_stats_extract_arrays(stats: dict):
    return _comparison_stats_coerce_arrays(_comparison_stats_get_raw_arrays(stats))


def _make_comparison_stats(stats: dict, ref_fs: int = 44100, ref_taps: int = 65536) -> dict:
    prep = _comparison_stats_prepare_arrays(stats, ref_fs, ref_taps)
    if prep is None:
        return dict(stats or {})
    out, f, freq_cmp, m, t, g, c, mm = prep
    m_cmp = _comparison_stats_interp(freq_cmp, f, m)
    t_cmp = _comparison_stats_interp(freq_cmp, f, t)
    g_cmp = _comparison_stats_interp(freq_cmp, f, g) if g is not None else None
    c_cmp = _comparison_stats_interp(freq_cmp, f, c) if c is not None else None
    mm_cmp = _comparison_stats_interp(freq_cmp, f, mm) if mm is not None else None
    if m_cmp is None or t_cmp is None:
        return out

    rng = out.get("smart_scan_range", None)
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        fmin, fmax_rng = float(rng[0]), float(rng[1])
    else:
        fmin, fmax_rng = 200.0, 5000.0
    mask = (freq_cmp >= fmin) & (freq_cmp <= fmax_rng)
    if np.count_nonzero(mask) >= 20:
        cmp_offset_db = float(np.median((m_cmp - t_cmp)[mask]))
    else:
        cmp_offset_db = 0.0
    m_cmp = m_cmp - cmp_offset_db
    bw = out.get("afdw_bw_oct", None)
    bw_cmp = _comparison_stats_interp(freq_cmp, f, bw) if bw is not None and np.asarray(bw).shape == f.shape else None
    if bw_cmp is not None:
        out["cmp_afdw_bw_oct"] = np.clip(bw_cmp, 1.0 / 96.0, 1.0 / 3.0).tolist()
        out["cmp_offset_db"] = float(cmp_offset_db)
    out["cmp_ref_fs"] = int(ref_fs)
    out["cmp_ref_taps"] = int(ref_taps)
    return _comparison_stats_write_output(
        out,
        stats=stats,
        freq_cmp=freq_cmp,
        m_cmp=m_cmp,
        t_cmp=t_cmp,
        g_cmp=g_cmp,
        c_cmp=c_cmp,
        mm_cmp=mm_cmp,
        bw_cmp=bw_cmp,
        cmp_offset_db=float(cmp_offset_db),
    )


__all__ = ["_make_comparison_stats"]
