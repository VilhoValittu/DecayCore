# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Shared helpers for measurement-derived features (RT60 and harmonics).

All functions are pure and unit-testable. No DSP decision-making lives here.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# RT60 normalize / serialize
# ---------------------------------------------------------------------------

def normalize_rt60_value(value) -> float | None:
    """Convert any RT60 value representation to float, or None if invalid."""
    try:
        v = float(value)
        if math.isfinite(v) and v > 0.0:
            return v
    except (TypeError, ValueError):
        pass
    return None


def normalize_rt60_bands(bands) -> dict[float, float] | None:
    """Accept string- or float-keyed RT60 band dicts; return sorted float-keyed dict.

    JSON serialisation uses string keys like "31.5" or "6400.4". This helper
    converts them deterministically to float keys and filters invalid entries.
    Returns None if the result is empty.
    """
    if not isinstance(bands, dict) or not bands:
        return None
    result: dict[float, float] = {}
    for key, value in bands.items():
        try:
            freq = float(key)
            rt = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(freq) and freq > 0.0 and math.isfinite(rt) and rt > 0.0:
            result[float(freq)] = float(rt)
    if not result:
        return None
    return dict(sorted(result.items()))


def serialize_rt60_bands(bands) -> dict[str, float] | None:
    """Convert float-keyed RT60 bands to JSON-safe string-keyed dict.

    Use this shared serialiser everywhere instead of hand-rolling float
    key formatting in multiple places.
    """
    norm = normalize_rt60_bands(bands)
    if norm is None:
        return None
    return {f"{freq:g}": rt for freq, rt in norm.items()}


def median_rt60_mid_band(bands) -> float:
    """Return median RT60 in the 125–4000 Hz mid band, or overall median if empty.

    Used for control flow decisions. Returns 0.0 if bands is empty or invalid.
    """
    norm = normalize_rt60_bands(bands)
    if not norm:
        return 0.0
    ks = np.array(sorted(norm.keys()), dtype=float)
    vs = np.array([norm[k] for k in ks], dtype=float)
    mid = (ks >= 125.0) & (ks <= 4000.0) & (vs > 0.05) & (vs < 5.0)
    if np.any(mid):
        return float(np.median(vs[mid]))
    return float(np.median(vs)) if vs.size > 0 else 0.0


# ---------------------------------------------------------------------------
# RT60 summary
# ---------------------------------------------------------------------------

def summarize_rt60_bands(
    bands,
    wideband: float | None = None,
) -> dict:
    """Summarise RT60 bands into frequency-range group averages.

    Returns a dict with keys:
      valid, wideband_s, lf_20_120_s, lowmid_120_400_s,
      mid_400_2000_s, hf_2000_8000_s, source_band_count.
    """
    result: dict = {
        "valid": False,
        "wideband_s": None,
        "lf_20_120_s": None,
        "lowmid_120_400_s": None,
        "mid_400_2000_s": None,
        "hf_2000_8000_s": None,
        "source_band_count": 0,
    }

    wb = normalize_rt60_value(wideband)
    if wb is not None:
        result["wideband_s"] = wb

    norm = normalize_rt60_bands(bands)
    if norm is None:
        return result

    freqs = np.array(sorted(norm.keys()), dtype=float)
    vals = np.array([norm[f] for f in freqs], dtype=float)
    result["source_band_count"] = int(len(freqs))

    def _band_avg(lo: float, hi: float) -> float | None:
        mask = (freqs >= lo) & (freqs <= hi) & (vals > 0.05) & (vals < 5.0)
        if np.any(mask):
            return float(np.median(vals[mask]))
        # Log-frequency interpolation fallback when no bands fall directly in range.
        if freqs.size >= 2:
            try:
                lf = np.log(np.maximum(freqs, 1e-6))
                lv = np.log(np.maximum(vals, 1e-9))
                mid_log = 0.5 * (np.log(max(lo, 1e-6)) + np.log(max(hi, 1e-6)))
                interp_lv = float(np.interp(mid_log, lf, lv))
                v = float(np.exp(interp_lv))
                if math.isfinite(v) and 0.05 < v < 5.0:
                    return v
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
                pass
        return None

    result["lf_20_120_s"] = _band_avg(20.0, 120.0)
    result["lowmid_120_400_s"] = _band_avg(120.0, 400.0)
    result["mid_400_2000_s"] = _band_avg(400.0, 2000.0)
    result["hf_2000_8000_s"] = _band_avg(2000.0, 8000.0)

    if result["wideband_s"] is None:
        valid_vals = vals[(vals > 0.05) & (vals < 5.0)]
        if valid_vals.size > 0:
            result["wideband_s"] = float(np.median(valid_vals))

    result["valid"] = any(
        result[k] is not None
        for k in ("lf_20_120_s", "lowmid_120_400_s", "mid_400_2000_s", "hf_2000_8000_s")
    )
    return result


# ---------------------------------------------------------------------------
# Harmonic boost-risk curve
# ---------------------------------------------------------------------------


def _harmonic_boost_risk_empty_summary() -> dict:
    return {
        "mean_risk_20_80": 0.0,
        "mean_risk_20_120": 0.0,
        "mean_risk_20_200": 0.0,
        "peak_risk": 0.0,
        "peak_risk_hz": float("nan"),
        "valid": False,
    }


def _harmonic_boost_risk_collect_inputs(
    harmonic_freq_hz,
    harmonic_magnitudes_db: dict,
    fundamental_freq_hz=None,
    fundamental_mag_db=None,
):
    if harmonic_freq_hz is None or not isinstance(harmonic_magnitudes_db, dict):
        return None, None, False, None

    freq = np.asarray(harmonic_freq_hz, dtype=float).reshape(-1)
    if freq.size < 4:
        return None, None, False, None

    use_relative = fundamental_freq_hz is not None and fundamental_mag_db is not None
    fund_at_harm: np.ndarray | None = None
    if use_relative:
        ff = np.asarray(fundamental_freq_hz, dtype=float).reshape(-1)
        fm = np.asarray(fundamental_mag_db, dtype=float).reshape(-1)
        valid_f = np.isfinite(ff) & np.isfinite(fm) & (ff > 0.0)
        if valid_f.sum() >= 4:
            fund_at_harm = np.interp(freq, ff[valid_f], fm[valid_f], left=fm[valid_f][0], right=fm[valid_f][-1])
        else:
            use_relative = False
    return freq, harmonic_magnitudes_db, use_relative, fund_at_harm


def _harmonic_boost_risk_collect_orders(freq: np.ndarray, harmonic_magnitudes_db: dict) -> list[np.ndarray]:
    h_arrays: list[np.ndarray] = []
    for order in (2, 3, 4):
        arr = harmonic_magnitudes_db.get(order)
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float).reshape(-1)
        if a.size != freq.size:
            continue
        if not np.any(np.isfinite(a)):
            continue
        # Require H2 before accepting H3 and later
        if order > 2 and not h_arrays:
            break
        h_arrays.append(a)
    return h_arrays


def _harmonic_boost_risk_raw_curve(
    freq: np.ndarray,
    h_arrays: list[np.ndarray],
    *,
    use_relative: bool,
    fund_at_harm: np.ndarray | None,
) -> np.ndarray | None:
    if not h_arrays:
        return None
    if use_relative and fund_at_harm is not None:
        threshold_db = -30.0
        range_db = 15.0
        effective = [h - fund_at_harm for h in h_arrays]
    else:
        threshold_db = -40.0
        range_db = 15.0
        effective = h_arrays

    worst = np.full(freq.size, float("-inf"), dtype=float)
    for e in effective:
        finite_mask = np.isfinite(e)
        worst[finite_mask] = np.maximum(worst[finite_mask], e[finite_mask])
    nonfinite = ~np.isfinite(worst)
    if np.all(nonfinite):
        return None
    worst[nonfinite] = float(threshold_db) - 1.0
    raw_risk = np.clip((worst - threshold_db) / max(range_db, 1e-6), 0.0, 1.0)
    return np.asarray(raw_risk, dtype=float)


def _harmonic_boost_risk_smooth_curve(
    freq: np.ndarray,
    raw_risk: np.ndarray,
    *,
    lo_hz: float,
    hi_hz: float,
) -> np.ndarray:
    band_mask = (freq >= lo_hz) & (freq <= hi_hz) & np.isfinite(freq)
    if not np.any(band_mask) or freq[band_mask].size < 4:
        return raw_risk.copy()

    lf = np.log2(np.maximum(freq[band_mask], 1e-6))
    dlf = np.diff(lf)
    dlf = dlf[dlf > 0]
    if dlf.size < 2:
        return raw_risk.copy()

    step = float(np.median(dlf))
    sigma_bins = max(1.0, 0.5 / max(step, 1e-6))
    half = int(np.ceil(3.0 * sigma_bins))
    x = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    k /= k.sum()
    from scipy.signal import fftconvolve  # noqa: PLC0415

    smoothed = fftconvolve(raw_risk[band_mask], k, mode="same")
    risk = np.zeros(freq.size, dtype=float)
    risk[band_mask] = np.clip(smoothed, 0.0, 1.0)
    return risk


def _harmonic_boost_risk_summary(freq: np.ndarray, risk: np.ndarray) -> dict:
    def _mean_in_band(lo: float, hi: float) -> float:
        m = (freq >= lo) & (freq <= hi) & np.isfinite(risk)
        if np.any(m):
            return float(np.mean(risk[m]))
        return 0.0

    if np.any(np.isfinite(risk) & (risk > 0.0)):
        peak_idx = int(np.argmax(risk))
        peak_risk = float(risk[peak_idx])
        peak_risk_hz = float(freq[peak_idx])
    else:
        peak_risk = 0.0
        peak_risk_hz = float("nan")

    return {
        "mean_risk_20_80": _mean_in_band(20.0, 80.0),
        "mean_risk_20_120": _mean_in_band(20.0, 120.0),
        "mean_risk_20_200": _mean_in_band(20.0, 200.0),
        "peak_risk": peak_risk,
        "peak_risk_hz": peak_risk_hz,
        "valid": bool(peak_risk > 0.0),
    }


def _harmonic_boost_risk_build(
    harmonic_freq_hz,
    harmonic_magnitudes_db: dict,
    *,
    fundamental_freq_hz=None,
    fundamental_mag_db=None,
    lo_hz: float,
    hi_hz: float,
) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    empty_summary = _harmonic_boost_risk_empty_summary()
    try:
        freq, harmonic_magnitudes_db, use_relative, fund_at_harm = _harmonic_boost_risk_collect_inputs(
            harmonic_freq_hz,
            harmonic_magnitudes_db,
            fundamental_freq_hz,
            fundamental_mag_db,
        )
        if freq is None or harmonic_magnitudes_db is None:
            return None, None, empty_summary

        h_arrays = _harmonic_boost_risk_collect_orders(freq, harmonic_magnitudes_db)
        if not h_arrays:
            return None, None, empty_summary

        raw_risk = _harmonic_boost_risk_raw_curve(
            freq,
            h_arrays,
            use_relative=use_relative,
            fund_at_harm=fund_at_harm,
        )
        if raw_risk is None:
            return None, None, empty_summary

        risk = _harmonic_boost_risk_smooth_curve(freq, raw_risk, lo_hz=lo_hz, hi_hz=hi_hz)
        summary = _harmonic_boost_risk_summary(freq, risk)
        return np.asarray(freq, dtype=float), np.asarray(risk, dtype=float), summary

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
        return None, None, empty_summary

def build_harmonic_boost_risk_curve(
    harmonic_freq_hz,
    harmonic_magnitudes_db: dict,
    *,
    fundamental_freq_hz=None,
    fundamental_mag_db=None,
    lo_hz: float = 20.0,
    hi_hz: float = 800.0,
) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """Build a frequency-dependent 0..1 harmonic boost-risk curve.

    Uses H2 + H3 (and H4 if available). When fundamental data is provided,
    uses relative comparison (dBr); otherwise falls back to absolute dB logic.

    Returns (freq_hz, risk_curve_0_1, summary_dict).
    summary keys: mean_risk_20_80, mean_risk_20_120, mean_risk_20_200,
                  peak_risk, peak_risk_hz, valid.
    """
    return _harmonic_boost_risk_build(
        harmonic_freq_hz,
        harmonic_magnitudes_db,
        fundamental_freq_hz=fundamental_freq_hz,
        fundamental_mag_db=fundamental_mag_db,
        lo_hz=lo_hz,
        hi_hz=hi_hz,
    )


def build_target_decay_hint(
    *,
    rt60_values: list[float | None] | None = None,
    rt60_bands_list: list[dict[float, float] | None] | None = None,
    harmonic_summaries: list[dict | None] | None = None,
) -> dict[str, object]:
    """Build a conservative target-adjustment hint from measurement-derived data.

    Returns a small inspectable dict with:
      status: unavailable | ok | caution | strong
      has_data: bool
      reason: none | rt60 | harmonic | mixed
      advice_codes: tuple[str, ...]
      low_band_rt60_s, lowmid_rt60_s, harmonic_peak_risk, harmonic_mean_risk_20_120
    """
    rt60_values = list(rt60_values or [])
    rt60_bands_list = list(rt60_bands_list or [])
    harmonic_summaries = list(harmonic_summaries or [])

    low_band_vals: list[float] = []
    lowmid_vals: list[float] = []
    wideband_vals: list[float] = []
    harmonic_peak_vals: list[float] = []
    harmonic_mean_vals: list[float] = []

    for value in rt60_values:
        norm = normalize_rt60_value(value)
        if norm is not None:
            wideband_vals.append(float(norm))

    for bands in rt60_bands_list:
        summary = summarize_rt60_bands(bands)
        if summary.get("valid"):
            lf = normalize_rt60_value(summary.get("lf_20_120_s"))
            lowmid = normalize_rt60_value(summary.get("lowmid_120_400_s"))
            wide = normalize_rt60_value(summary.get("wideband_s"))
            if lf is not None:
                low_band_vals.append(float(lf))
            if lowmid is not None:
                lowmid_vals.append(float(lowmid))
            if wide is not None:
                wideband_vals.append(float(wide))

    for item in harmonic_summaries:
        if not isinstance(item, dict) or not bool(item.get("valid")):
            continue
        peak = item.get("peak_risk")
        mean_20_120 = item.get("mean_risk_20_120")
        try:
            peak_f = float(peak)
        except (TypeError, ValueError):
            peak_f = float("nan")
        try:
            mean_f = float(mean_20_120)
        except (TypeError, ValueError):
            mean_f = float("nan")
        if math.isfinite(peak_f):
            harmonic_peak_vals.append(float(np.clip(peak_f, 0.0, 1.0)))
        if math.isfinite(mean_f):
            harmonic_mean_vals.append(float(np.clip(mean_f, 0.0, 1.0)))

    low_band_rt60 = max(low_band_vals) if low_band_vals else None
    lowmid_rt60 = max(lowmid_vals) if lowmid_vals else None
    wideband_rt60 = max(wideband_vals) if wideband_vals else None
    harmonic_peak = max(harmonic_peak_vals) if harmonic_peak_vals else None
    harmonic_mean = max(harmonic_mean_vals) if harmonic_mean_vals else None

    has_rt60 = any(v is not None for v in (low_band_rt60, lowmid_rt60, wideband_rt60))
    has_harmonic = any(v is not None for v in (harmonic_peak, harmonic_mean))
    has_data = bool(has_rt60 or has_harmonic)

    rt60_strong = bool(
        (low_band_rt60 is not None and low_band_rt60 >= 0.75)
        or (lowmid_rt60 is not None and lowmid_rt60 >= 0.65)
        or (wideband_rt60 is not None and wideband_rt60 >= 0.60)
    )
    rt60_caution = bool(
        (low_band_rt60 is not None and low_band_rt60 >= 0.50)
        or (lowmid_rt60 is not None and lowmid_rt60 >= 0.45)
        or (wideband_rt60 is not None and wideband_rt60 >= 0.45)
    )
    harmonic_strong = bool(
        (harmonic_peak is not None and harmonic_peak >= 0.75)
        or (harmonic_mean is not None and harmonic_mean >= 0.50)
    )
    harmonic_caution = bool(
        (harmonic_peak is not None and harmonic_peak >= 0.45)
        or (harmonic_mean is not None and harmonic_mean >= 0.28)
    )

    if not has_data:
        status = "unavailable"
    elif rt60_strong or harmonic_strong:
        status = "strong"
    elif rt60_caution or harmonic_caution:
        status = "caution"
    else:
        status = "ok"

    if status == "unavailable":
        reason = "none"
    else:
        rt60_flag = rt60_strong or rt60_caution
        harmonic_flag = harmonic_strong or harmonic_caution
        if rt60_flag and harmonic_flag:
            reason = "mixed"
        elif rt60_flag:
            reason = "rt60"
        elif harmonic_flag:
            reason = "harmonic"
        else:
            reason = "none"

    advice_codes: tuple[str, ...]
    if status == "unavailable":
        advice_codes = ("no_data",)
    elif status == "ok":
        advice_codes = ("keep_changes_measured",)
    elif reason == "harmonic":
        advice_codes = ("avoid_deep_null_boost", "prefer_conservative_bass_boost")
    elif reason == "rt60":
        advice_codes = ("prefer_conservative_bass_boost", "keep_correction_band_limited")
    else:
        advice_codes = ("avoid_deep_null_boost", "keep_correction_band_limited")

    return {
        "status": status,
        "has_data": has_data,
        "reason": reason,
        "advice_codes": advice_codes,
        "low_band_rt60_s": low_band_rt60,
        "lowmid_rt60_s": lowmid_rt60,
        "wideband_rt60_s": wideband_rt60,
        "harmonic_peak_risk": harmonic_peak,
        "harmonic_mean_risk_20_120": harmonic_mean,
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_rt60_bands(
    bands_list: list,
    *,
    mode: str = "median",
) -> dict[float, float] | None:
    """Aggregate a list of RT60 band dicts into one, using median or mean per frequency.

    Returns None if no valid inputs exist.
    """
    if not bands_list:
        return None

    normalised = [normalize_rt60_bands(b) for b in bands_list]
    normalised = [n for n in normalised if n is not None]
    if not normalised:
        return None

    # Collect all unique frequencies.
    all_freqs: set[float] = set()
    for n in normalised:
        all_freqs.update(n.keys())
    if not all_freqs:
        return None

    result: dict[float, float] = {}
    for freq in sorted(all_freqs):
        vals = [n[freq] for n in normalised if freq in n]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 0.0)]
        if arr.size == 0:
            continue
        if mode == "mean":
            result[float(freq)] = float(np.mean(arr))
        else:
            result[float(freq)] = float(np.median(arr))

    return result or None


def aggregate_rt60_value(
    values: list,
    fallback_bands: list | None = None,
) -> float | None:
    """Aggregate a list of RT60 wideband values; fall back to estimating from bands."""
    clean = [normalize_rt60_value(v) for v in (values or [])]
    clean = [v for v in clean if v is not None]
    if clean:
        return float(np.median(np.array(clean, dtype=float)))

    # Fallback: estimate from the aggregated bands.
    if fallback_bands:
        agg = aggregate_rt60_bands(fallback_bands)
        if agg:
            s = summarize_rt60_bands(agg)
            if s.get("wideband_s") is not None:
                return float(s["wideband_s"])
    return None


def aggregate_harmonic_curves(
    freqs_list: list,
    mags_list: list,
    *,
    mode: str = "linear_mean",
) -> tuple[np.ndarray | None, dict[int, np.ndarray] | None]:
    """Average harmonic curves across takes/positions in the linear amplitude domain.

    Inputs:
      freqs_list: list of freq_hz arrays (one per take)
      mags_list: list of {order: mag_db array} dicts

    Returns (common_freq_hz, {order: averaged_mag_db}) or (None, None).
    """
    valid_pairs = _harmonic_valid_pairs(freqs_list, mags_list)
    if not valid_pairs:
        return None, None

    # Use the first take's freq axis as common grid (all should match within a session).
    ref_freq = valid_pairs[0][0]
    if ref_freq.size < 4:
        return None, None

    # Collect all harmonic orders present across takes.
    all_orders = _harmonic_collect_orders(valid_pairs)
    if not all_orders:
        return None, None

    averaged: dict[int, np.ndarray] = {}
    for order in sorted(all_orders):
        rows = _harmonic_rows_for_order(valid_pairs, order=order, ref_freq=ref_freq)
        if not rows:
            continue
        averaged[int(order)] = _harmonic_average_rows(rows, mode=mode)

    if not averaged:
        return None, None
    return np.asarray(ref_freq, dtype=float), averaged


def _harmonic_valid_pairs(freqs_list: list, mags_list: list) -> list[tuple[np.ndarray, dict]]:
    return [
        (np.asarray(freq, dtype=float), dict(mags))
        for freq, mags in zip(freqs_list or [], mags_list or [], strict=False)
        if freq is not None and mags is not None
    ]


def _harmonic_collect_orders(valid_pairs: list[tuple[np.ndarray, dict]]) -> set[int]:
    orders: set[int] = set()
    for _, mags in valid_pairs:
        orders.update(mags.keys())
    return orders


def _harmonic_rows_for_order(
    valid_pairs: list[tuple[np.ndarray, dict]],
    *,
    order: int,
    ref_freq: np.ndarray,
) -> list[np.ndarray]:
    rows: list[np.ndarray] = []
    for freq, mags in valid_pairs:
        arr = mags.get(order)
        if arr is None:
            continue
        vals = np.asarray(arr, dtype=float).reshape(-1)
        if vals.size == ref_freq.size:
            rows.append(vals)
        elif vals.size > 0 and freq.size == vals.size:
            rows.append(np.interp(ref_freq, freq, vals, left=vals[0], right=vals[-1]))
    return rows


def _harmonic_average_rows(rows: list[np.ndarray], *, mode: str) -> np.ndarray:
    mat = np.vstack(rows)
    if mode == "linear_mean":
        linear = np.power(10.0, mat / 20.0)
        averaged_linear = np.mean(linear, axis=0)
        return 20.0 * np.log10(np.maximum(averaged_linear, 1e-12))
    return np.median(mat, axis=0)


# ---------------------------------------------------------------------------
# Distance / outlier helpers
# ---------------------------------------------------------------------------

def rt60_distance_score(
    candidate_bands,
    reference_bands,
) -> float | None:
    """Return an RMS log-ratio distance between two RT60 band dicts.

    Returns None if comparison is not possible (too few common bands).
    Returns 0..inf where 0 = identical, higher = more different.
    """
    cand = normalize_rt60_bands(candidate_bands)
    ref = normalize_rt60_bands(reference_bands)
    if cand is None or ref is None:
        return None

    common_freqs = sorted(set(cand.keys()) & set(ref.keys()))
    if len(common_freqs) < 2:
        return None

    log_ratios = []
    for freq in common_freqs:
        c_val = cand[freq]
        r_val = ref[freq]
        if c_val > 0.0 and r_val > 0.0:
            log_ratios.append(math.log(c_val / r_val))

    if len(log_ratios) < 2:
        return None

    arr = np.array(log_ratios, dtype=float)
    return float(np.sqrt(np.mean(arr ** 2)))


def harmonic_distance_score(
    candidate_freq,
    candidate_mags: dict,
    ref_freq,
    ref_mags: dict,
) -> float | None:
    """RMS difference between two harmonic curve sets on a common frequency axis.

    Compares H2 and H3 (if both available). Returns None if comparison fails.
    Returns 0..inf where 0 = identical.
    """
    if candidate_freq is None or ref_freq is None:
        return None
    if not isinstance(candidate_mags, dict) or not isinstance(ref_mags, dict):
        return None

    try:
        cf = np.asarray(candidate_freq, dtype=float).reshape(-1)
        rf = np.asarray(ref_freq, dtype=float).reshape(-1)
        if cf.size < 4 or rf.size < 4:
            return None

        # Use log-spaced comparison grid over 20-800 Hz.
        grid = np.logspace(np.log10(20.0), np.log10(800.0), 64, dtype=float)

        diffs_sq = []
        for order in (2, 3):
            ca = candidate_mags.get(order)
            ra = ref_mags.get(order)
            if ca is None or ra is None:
                continue
            c_arr = np.asarray(ca, dtype=float).reshape(-1)
            r_arr = np.asarray(ra, dtype=float).reshape(-1)
            if c_arr.size != cf.size or r_arr.size != rf.size:
                continue
            c_on_grid = np.interp(grid, cf, c_arr, left=c_arr[0], right=c_arr[-1])
            r_on_grid = np.interp(grid, rf, r_arr, left=r_arr[0], right=r_arr[-1])
            diff = c_on_grid - r_on_grid
            diffs_sq.append(diff ** 2)

        if not diffs_sq:
            return None

        all_sq = np.concatenate(diffs_sq)
        return float(np.sqrt(np.mean(all_sq)))
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


def estimate_schroeder_hz(
    rt60_val: float | None,
    *,
    room_volume_m3: float = 40.0,
    min_hz: float = 80.0,
    max_hz: float = 500.0,
) -> float | None:
    """Estimate Schroeder frequency from RT60.

    Uses f_S = 2000 * sqrt(RT60 / V).  Room volume defaults to 40 m³ which is a
    reasonable middle estimate for a typical home listening room.  Returns None
    when RT60 is unavailable or non-positive.
    """
    import math
    if rt60_val is None:
        return None
    try:
        rt60 = float(rt60_val)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(rt60) and rt60 > 0.0):
        return None
    try:
        v = float(room_volume_m3)
    except (TypeError, ValueError):
        v = 40.0
    if not (math.isfinite(v) and v > 0.0):
        v = 40.0
    fs = 2000.0 * math.sqrt(rt60 / v)
    return float(max(float(min_hz), min(float(max_hz), fs)))


__all__ = [
    "normalize_rt60_value",
    "normalize_rt60_bands",
    "serialize_rt60_bands",
    "summarize_rt60_bands",
    "build_harmonic_boost_risk_curve",
    "build_target_decay_hint",
    "aggregate_rt60_bands",
    "aggregate_rt60_value",
    "aggregate_harmonic_curves",
    "rt60_distance_score",
    "harmonic_distance_score",
    "estimate_schroeder_hz",
]
