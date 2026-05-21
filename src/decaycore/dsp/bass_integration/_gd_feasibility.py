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

from typing import Any

import numpy as np
import scipy.signal

from ...auto_mode.shared import (
    AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
    AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
    _auto_bass_integration_profile_weights,
)
from ...io.measurement_bundle import BassIntegrationBundle, TransferData
from ._constants import (
    BASS_INTEGRATION_FEASIBILITY_THRESHOLDS,
    GD_CONTINUITY_GUARD_HI_RATIO,
    GD_CONTINUITY_GUARD_LO_RATIO,
)
from ._utils import _band_mask, _safe_float, normalize_sub_combine_mode


def _dominant_bass_channel(
    *,
    overlap_ripple_l: Any,
    overlap_ripple_r: Any,
    sub_dominance_l: Any,
    sub_dominance_r: Any,
    xo_gd_l: Any,
    xo_gd_r: Any,
) -> str:
    marginal = dict(BASS_INTEGRATION_FEASIBILITY_THRESHOLDS["marginal"])

    def _severity(ripple: Any, dominance: Any, gd: Any) -> float:
        vals: list[float] = []
        ripple_v = _safe_float(ripple, float("nan"))
        dom_v = _safe_float(dominance, float("nan"))
        gd_v = _safe_float(gd, float("nan"))
        if np.isfinite(ripple_v):
            vals.append(float(max(0.0, ripple_v)) / float(marginal["overlap_ripple_db"]))
        if np.isfinite(dom_v):
            vals.append(float(max(0.0, dom_v)) / float(marginal["sub_dominance_db"]))
        if np.isfinite(gd_v):
            vals.append(float(max(0.0, gd_v)) / float(marginal["xo_gd_rms_mismatch_ms"]))
        if not vals:
            return float("nan")
        return float(max(vals))

    left_score = _severity(overlap_ripple_l, sub_dominance_l, xo_gd_l)
    right_score = _severity(overlap_ripple_r, sub_dominance_r, xo_gd_r)
    if np.isfinite(left_score) and np.isfinite(right_score):
        if abs(float(left_score) - float(right_score)) < 0.05:
            return "balanced"
        return "left" if float(left_score) > float(right_score) else "right"
    if np.isfinite(left_score):
        return "left"
    if np.isfinite(right_score):
        return "right"
    return "unknown"


def _classify_bass_integration_feasibility(
    *,
    overlap_ripple_worst: Any,
    sub_dominance_worst: Any,
    xo_gd_rms_worst: Any,
    overlap_ripple_delta: Any,
    sub_dominance_delta: Any,
    xo_gd_delta: Any,
    dominant_channel: str,
    sub_combine_mode: str,
    sub_level_delta_db_20_120: Any,
    fc_hz: float = 80.0,
) -> tuple[str, str]:
    values = {
        "overlap_ripple_db": _safe_float(overlap_ripple_worst, float("nan")),
        "sub_dominance_db": _safe_float(sub_dominance_worst, float("nan")),
        "xo_gd_rms_mismatch_ms": _safe_float(xo_gd_rms_worst, float("nan")),
        "overlap_ripple_delta_db": _safe_float(overlap_ripple_delta, float("nan")),
        "sub_dominance_delta_db": _safe_float(sub_dominance_delta, float("nan")),
        "xo_gd_mismatch_delta_ms": _safe_float(xo_gd_delta, float("nan")),
    }
    # Scale GD thresholds by crossover frequency.
    # 12 ms at 80 Hz is 0.35 wavelengths; the same physical tolerance at 150 Hz is
    # only 0.18 wavelengths and causes severe comb filtering.  Clamp scale to [0.5, 2.0]
    # so extreme XO values don't produce unreasonable thresholds.
    _fc = float(max(_safe_float(fc_hz, 80.0), 1.0))
    _gd_scale = float(np.clip(80.0 / _fc, 0.5, 2.0))

    def _meets(limit_name: str) -> bool:
        limits = dict(BASS_INTEGRATION_FEASIBILITY_THRESHOLDS[limit_name])
        checked = False
        for key, limit in limits.items():
            value = values.get(key, float("nan"))
            if not np.isfinite(value):
                continue
            checked = True
            effective_limit = float(limit) * _gd_scale if key == "xo_gd_rms_mismatch_ms" else float(limit)
            if float(value) > effective_limit:
                return False
        return bool(checked)

    if _meets("good"):
        return "good", "Shared mono-sub integration meets balance and crossover guard targets."

    feasibility_class = "marginal" if _meets("marginal") else "infeasible"
    ref_limits = dict(
        BASS_INTEGRATION_FEASIBILITY_THRESHOLDS["good" if feasibility_class == "marginal" else "marginal"]
    )
    label = {
        "left": "Left channel remains limiting",
        "right": "Right channel remains limiting",
        "balanced": "Shared mono-sub integration remains balanced-but-limited",
    }.get(str(dominant_channel or "").strip().lower(), "Shared mono-sub integration remains system-limited")

    reasons = [label]
    metric_labels = (
        ("overlap_ripple_db", "overlap ripple", " dB"),
        ("sub_dominance_db", "sub dominance", " dB"),
        ("xo_gd_rms_mismatch_ms", "XO GD mismatch", " ms"),
        ("overlap_ripple_delta_db", "ripple delta", " dB"),
        ("sub_dominance_delta_db", "dominance delta", " dB"),
        ("xo_gd_mismatch_delta_ms", "GD delta", " ms"),
    )
    for key, metric_label, unit in metric_labels:
        value = values.get(key, float("nan"))
        limit = float(ref_limits.get(key, float("nan")))
        if np.isfinite(value) and np.isfinite(limit) and float(value) > float(limit):
            reasons.append(f"{metric_label} {float(value):.1f}{unit}")

    level_delta = _safe_float(sub_level_delta_db_20_120, float("nan"))
    combine_mode_norm = normalize_sub_combine_mode(sub_combine_mode)
    if (
        combine_mode_norm != "average"
        and np.isfinite(level_delta)
        and abs(float(level_delta)) >= 3.0
    ):
        reasons.append(f"{combine_mode_norm} combine adds {float(level_delta):+.1f} dB")

    if len(reasons) == 1:
        reasons.append(
            "usable but asymmetric" if feasibility_class == "marginal" else "no balanced shared mono-sub solution found"
        )
    return feasibility_class, ". ".join(str(part).strip() for part in reasons if str(part).strip()) + "."


def _gd_ms_from_transfer(transfer: TransferData) -> np.ndarray:
    """Compute group delay (ms) from unwrapped phase_deg in TransferData."""
    freqs = np.asarray(transfer.freqs_hz, dtype=float)
    phase_rad = np.deg2rad(np.asarray(transfer.phase_deg, dtype=float))
    omega = 2.0 * np.pi * np.maximum(freqs, 1e-9)
    # Smooth phase before differentiation to suppress np.gradient noise near DC and Nyquist.
    n = phase_rad.size
    if n >= 7:
        wlen = min(11, n if n % 2 == 1 else n - 1)
        phase_rad = scipy.signal.savgol_filter(phase_rad, window_length=wlen, polyorder=2)
    gd_s = -np.gradient(phase_rad, omega)
    return np.asarray(gd_s * 1000.0, dtype=float)


def _gd_ms_at_hz(transfer: TransferData, target_hz: float) -> float:
    """Interpolate group delay (ms) of a TransferData at a specific frequency."""
    freqs = np.asarray(transfer.freqs_hz, dtype=float)
    gd_ms = _gd_ms_from_transfer(transfer)
    if freqs.size < 2 or gd_ms.size != freqs.size:
        return float("nan")
    return float(np.interp(float(target_hz), freqs, gd_ms))


def _main_guard_band_drop_db(main: TransferData, fc_hz: float) -> float:
    """Return how far (dB) the main speaker has fallen in the XO guard band
    relative to its flat midrange reference.  Positive = rolling off.

    Reference band is fc-dependent so speakers with high XO (e.g. 150 Hz) that
    start rolling off at 200 Hz don't contaminate the reference measurement.
    ref_lo = max(fc*2, 300), ref_hi = max(fc*4, 600).
    """
    freqs = np.asarray(main.freqs_hz, dtype=float)
    mag_db = np.asarray(main.mag_db, dtype=float)
    fc = _safe_float(fc_hz, 80.0)
    ref_lo = max(float(fc) * 2.0, 300.0)
    ref_hi = max(float(fc) * 4.0, 600.0)
    ref_mask = _band_mask(freqs, ref_lo, ref_hi)
    if int(np.count_nonzero(ref_mask)) < 3:
        return float("nan")
    ref_db = float(np.mean(mag_db[ref_mask]))
    fc = _safe_float(fc_hz, 80.0)
    lo = max(5.0, 0.6 * fc)
    hi = max(lo + 1.0, 1.4 * fc)
    guard_mask = _band_mask(freqs, lo, hi)
    if int(np.count_nonzero(guard_mask)) < 3:
        return float("nan")
    guard_db = float(np.mean(mag_db[guard_mask]))
    return float(ref_db - guard_db)


def compute_xo_gd_continuity(
    bundle: BassIntegrationBundle,
    fc_hz: float,
    *,
    sub_combine_mode: str | None = None,
    guard_lo_ratio: float = GD_CONTINUITY_GUARD_LO_RATIO,
    guard_hi_ratio: float = GD_CONTINUITY_GUARD_HI_RATIO,
) -> dict[str, float]:
    from ._sub_combine import build_bundle_combined_sub_transfer

    fc = _safe_float(fc_hz, 80.0)
    lo_hz = max(5.0, fc * max(0.05, _safe_float(guard_lo_ratio, GD_CONTINUITY_GUARD_LO_RATIO)))
    hi_hz = max(lo_hz + 1.0, fc * max(0.05, _safe_float(guard_hi_ratio, GD_CONTINUITY_GUARD_HI_RATIO)))
    combine_mode_norm = normalize_sub_combine_mode(
        sub_combine_mode or dict(getattr(bundle, "diagnostics", {}) or {}).get("sub_combine_mode")
    )

    def _channel_gd_metrics(channel: str) -> dict[str, float]:
        main = bundle.r_main if channel == "r" else bundle.l_main
        combined_sub, _diag = build_bundle_combined_sub_transfer(
            bundle,
            channel=channel,
            mode=combine_mode_norm,
            label=f"{channel.upper()} GD combined sub",
        )
        freqs = np.asarray(main.freqs_hz, dtype=float)
        sub_freqs = np.asarray(combined_sub.freqs_hz, dtype=float)
        if freqs.size < 4 or sub_freqs.size < 4:
            return {
                "main_gd_at_fc_ms": float("nan"),
                "sub_gd_at_fc_ms": float("nan"),
                "gd_at_fc_delta_ms": float("nan"),
                "gd_rms_mismatch_ms": float("nan"),
                "gd_max_mismatch_ms": float("nan"),
                "gd_p90_mismatch_ms": float("nan"),
                "gd_max_mismatch_freq_hz": float("nan"),
            }
        main_gd = _gd_ms_from_transfer(main)
        sub_gd = _gd_ms_from_transfer(combined_sub)
        if sub_gd.size != freqs.size or not np.allclose(sub_freqs, freqs, rtol=0.0, atol=1e-9):
            sub_gd = np.interp(freqs, sub_freqs, sub_gd, left=sub_gd[0], right=sub_gd[-1])
        band_mask = _band_mask(freqs, lo_hz, hi_hz)
        if int(np.count_nonzero(band_mask)) < 4:
            rms_mm = float("nan")
            max_mm = float("nan")
            p90_mm = float("nan")
            max_mm_freq = float("nan")
        else:
            mismatch = np.abs(main_gd[band_mask] - sub_gd[band_mask])
            band_freqs = freqs[band_mask]
            rms_mm = float(np.sqrt(np.mean(np.square(mismatch))))
            max_idx = int(np.argmax(mismatch))
            max_mm = float(mismatch[max_idx])
            p90_mm = float(np.percentile(mismatch, 90.0))
            max_mm_freq = float(band_freqs[max_idx])
        main_gd_at_fc = float(np.interp(fc, freqs, main_gd))
        sub_gd_at_fc = float(np.interp(fc, freqs, sub_gd))
        return {
            "main_gd_at_fc_ms": float(main_gd_at_fc),
            "sub_gd_at_fc_ms": float(sub_gd_at_fc),
            "gd_at_fc_delta_ms": float(abs(main_gd_at_fc - sub_gd_at_fc)),
            "gd_rms_mismatch_ms": float(rms_mm),
            "gd_max_mismatch_ms": float(max_mm),
            "gd_p90_mismatch_ms": float(p90_mm),
            "gd_max_mismatch_freq_hz": float(max_mm_freq),
        }

    l_metrics = _channel_gd_metrics("l")
    r_metrics = _channel_gd_metrics("r")
    l_rms = _safe_float(l_metrics.get("gd_rms_mismatch_ms", float("nan")), float("nan"))
    r_rms = _safe_float(r_metrics.get("gd_rms_mismatch_ms", float("nan")), float("nan"))
    l_max = _safe_float(l_metrics.get("gd_max_mismatch_ms", float("nan")), float("nan"))
    r_max = _safe_float(r_metrics.get("gd_max_mismatch_ms", float("nan")), float("nan"))
    l_p90 = _safe_float(l_metrics.get("gd_p90_mismatch_ms", float("nan")), float("nan"))
    r_p90 = _safe_float(r_metrics.get("gd_p90_mismatch_ms", float("nan")), float("nan"))
    worst_channel = "r" if r_max >= l_max else "l"
    return {
        "fc_hz": float(fc),
        "gd_band_lo_hz": float(lo_hz),
        "gd_band_hi_hz": float(hi_hz),
        "l_main_gd_ms": float(l_metrics.get("main_gd_at_fc_ms", float("nan"))),
        "r_main_gd_ms": float(r_metrics.get("main_gd_at_fc_ms", float("nan"))),
        "sub_gd_ms": float(
            np.nanmean(
                np.asarray(
                    [
                        _safe_float(l_metrics.get("sub_gd_at_fc_ms", float("nan")), float("nan")),
                        _safe_float(r_metrics.get("sub_gd_at_fc_ms", float("nan")), float("nan")),
                    ],
                    dtype=float,
                )
            )
        ),
        "l_gd_mismatch_ms": float(l_rms),
        "r_gd_mismatch_ms": float(r_rms),
        "avg_gd_mismatch_ms": float(max(l_rms, r_rms)),
        "gd_rms_mismatch_ms_l": float(l_rms),
        "gd_rms_mismatch_ms_r": float(r_rms),
        "gd_rms_mismatch_ms_worst": float(max(l_rms, r_rms)),
        "gd_max_mismatch_ms_l": float(l_max),
        "gd_max_mismatch_ms_r": float(r_max),
        "gd_max_mismatch_ms_worst": float(max(l_max, r_max)),
        "gd_p90_mismatch_ms_l": float(l_p90),
        "gd_p90_mismatch_ms_r": float(r_p90),
        "gd_p90_mismatch_ms_worst": float(max(l_p90, r_p90)),
        "gd_max_mismatch_freq_hz_l": float(l_metrics.get("gd_max_mismatch_freq_hz", float("nan"))),
        "gd_max_mismatch_freq_hz_r": float(r_metrics.get("gd_max_mismatch_freq_hz", float("nan"))),
        "gd_max_mismatch_freq_hz": float(
            (r_metrics if worst_channel == "r" else l_metrics).get("gd_max_mismatch_freq_hz", float("nan"))
        ),
        "gd_at_fc_main_ms_l": float(l_metrics.get("main_gd_at_fc_ms", float("nan"))),
        "gd_at_fc_main_ms_r": float(r_metrics.get("main_gd_at_fc_ms", float("nan"))),
        "gd_at_fc_sub_ms_l": float(l_metrics.get("sub_gd_at_fc_ms", float("nan"))),
        "gd_at_fc_sub_ms_r": float(r_metrics.get("sub_gd_at_fc_ms", float("nan"))),
        "gd_at_fc_delta_ms_l": float(l_metrics.get("gd_at_fc_delta_ms", float("nan"))),
        "gd_at_fc_delta_ms_r": float(r_metrics.get("gd_at_fc_delta_ms", float("nan"))),
    }


def _direct_dac_metric_snapshot(
    diag: dict[str, Any] | None,
    gd_cont: dict[str, Any] | None,
    *,
    enabled: bool,
    freq_hz: float,
    q: float,
) -> dict[str, float | bool]:
    diag_obj = dict(diag or {})
    gd_obj = dict(gd_cont or {})
    return {
        "allpass_enabled": bool(enabled),
        "allpass_freq_hz": float(freq_hz),
        "allpass_q": float(q),
        "cancellation_risk": _safe_float(diag_obj.get("cancellation_risk", float("nan")), float("nan")),
        "overlap_ripple_db": _safe_float(diag_obj.get("overlap_ripple_db", float("nan")), float("nan")),
        "sub_dominance_db": _safe_float(diag_obj.get("sub_dominance_db", float("nan")), float("nan")),
        "xo_gd_mismatch_ms": _safe_float(gd_obj.get("avg_gd_mismatch_ms", float("nan")), float("nan")),
        "xo_l_gd_mismatch_ms": _safe_float(gd_obj.get("l_gd_mismatch_ms", float("nan")), float("nan")),
        "xo_r_gd_mismatch_ms": _safe_float(gd_obj.get("r_gd_mismatch_ms", float("nan")), float("nan")),
        "xo_main_gd_ms": _safe_float(
            (
                _safe_float(gd_obj.get("l_main_gd_ms", float("nan")), float("nan"))
                + _safe_float(gd_obj.get("r_main_gd_ms", float("nan")), float("nan"))
            )
            / 2.0,
            float("nan"),
        ),
        "xo_sub_gd_ms": _safe_float(gd_obj.get("sub_gd_ms", float("nan")), float("nan")),
    }


def _direct_dac_alignment_objective(
    diag: dict[str, Any] | None,
    gd_cont: dict[str, Any] | None,
    *,
    ap_freq_hz: float,
    ap_q: float,
    profile: str,
) -> float:
    weights = _auto_bass_integration_profile_weights(profile)
    cancel = _safe_float(dict(diag or {}).get("cancellation_risk", float("nan")), float("nan"))
    ripple = _safe_float(dict(diag or {}).get("overlap_ripple_db", float("nan")), float("nan"))
    dominance = _safe_float(dict(diag or {}).get("sub_dominance_db", float("nan")), float("nan"))
    gd_mm = _safe_float(dict(gd_cont or {}).get("avg_gd_mismatch_ms", float("nan")), float("nan"))
    if not (np.isfinite(cancel) and np.isfinite(ripple) and np.isfinite(dominance) and np.isfinite(gd_mm)):
        return float("nan")
    q_pen = max(0.0, _safe_float(ap_q, 0.0) - 0.90) / 1.30
    penalty = (
        float(weights.get("cancellation", 8.0)) * float(cancel)
        + float(weights.get("overlap_ripple", 1.8)) * (float(ripple) / 10.0)
        + float(weights.get("xo_gd_continuity", 0.8)) * (float(gd_mm) / 3.0)
        + float(weights.get("sub_dominance", 0.9)) * (abs(float(dominance)) / 8.0)
        + 0.18 * float(q_pen)
    )
    return float(-penalty)


def compute_direct_dac_bass_integration_analysis(
    bundle: BassIntegrationBundle,
    fc_hz: float,
    profile: str,
    *,
    main_hpf_order: int = 4,
    sub_lpf_order: int = 4,
    sub_hpf_hz: float = 20.0,
    sub_hpf_order: int = 2,
    sub_combine_mode: str = "average",
    sub_delay_ms: float = 0.0,
    sub_polarity_invert: bool = False,
    sub_gain_trim_db: float = 0.0,
    sub_lpf_hz: float | None = None,
    sub_allpass_freq_hz: float | None = None,
    sub_allpass_q: float | None = None,
    guard_lo_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
    guard_hi_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
) -> dict[str, Any]:
    from ._final_metrics import compute_final_bass_integration_metrics
    metrics = compute_final_bass_integration_metrics(
        bundle,
        fc_hz,
        profile,
        mode="direct_dac",
        main_hpf_order=int(main_hpf_order),
        sub_lpf_order=int(sub_lpf_order),
        sub_hpf_hz=float(sub_hpf_hz),
        sub_hpf_order=int(sub_hpf_order),
        sub_combine_mode=sub_combine_mode,
        sub_delay_ms=float(sub_delay_ms),
        sub_polarity_invert=bool(sub_polarity_invert),
        sub_gain_trim_db=float(sub_gain_trim_db),
        sub_lpf_hz=sub_lpf_hz,
        sub_allpass_freq_hz=sub_allpass_freq_hz,
        sub_allpass_q=sub_allpass_q,
        guard_lo_ratio=float(guard_lo_ratio),
        guard_hi_ratio=float(guard_hi_ratio),
    )
    return {
        "diagnostics": dict(metrics.get("diagnostics", {}) or {}),
        "gd_continuity": dict(metrics.get("gd_continuity", {}) or {}),
        "objective": float(_safe_float(metrics.get("objective", float("nan")), float("nan"))),
        "allpass_enabled": bool(metrics.get("bass_allpass_enabled", False)),
        "allpass_freq_hz": float(_safe_float(metrics.get("bass_allpass_freq_hz", 0.0), 0.0)),
        "allpass_q": float(_safe_float(metrics.get("bass_allpass_q", 0.707), 0.707)),
        "snapshot": dict(metrics.get("snapshot", {}) or {}),
    }


def compute_direct_dac_bass_integration_diagnostics(
    bundle: BassIntegrationBundle,
    fc_hz: float,
    profile: str,
    *,
    main_hpf_order: int = 4,
    sub_lpf_order: int = 4,
    sub_hpf_hz: float = 20.0,
    sub_hpf_order: int = 2,
    sub_combine_mode: str = "average",
    sub_delay_ms: float = 0.0,
    sub_polarity_invert: bool = False,
    sub_gain_trim_db: float = 0.0,
    sub_lpf_hz: float | None = None,
    sub_allpass_freq_hz: float | None = None,
    sub_allpass_q: float | None = None,
    guard_lo_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
    guard_hi_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
) -> dict[str, Any]:
    analysis = compute_direct_dac_bass_integration_analysis(
        bundle,
        fc_hz,
        profile,
        main_hpf_order=int(main_hpf_order),
        sub_lpf_order=int(sub_lpf_order),
        sub_hpf_hz=float(sub_hpf_hz),
        sub_hpf_order=int(sub_hpf_order),
        sub_combine_mode=sub_combine_mode,
        sub_delay_ms=float(sub_delay_ms),
        sub_polarity_invert=bool(sub_polarity_invert),
        sub_gain_trim_db=float(sub_gain_trim_db),
        sub_lpf_hz=sub_lpf_hz,
        sub_allpass_freq_hz=sub_allpass_freq_hz,
        sub_allpass_q=sub_allpass_q,
        guard_lo_ratio=float(guard_lo_ratio),
        guard_hi_ratio=float(guard_hi_ratio),
    )
    out = dict(analysis.get("diagnostics", {}) or {})
    out.update(dict(analysis.get("gd_continuity", {}) or {}))
    out.update(
        {
            "direct_dac_main_hpf_order": int(main_hpf_order),
            "direct_dac_sub_lpf_order": int(sub_lpf_order),
            "direct_dac_sub_hpf_hz": float(sub_hpf_hz),
            "direct_dac_sub_hpf_order": int(sub_hpf_order),
            "direct_dac_sub_allpass_enabled": bool(analysis.get("allpass_enabled", False)),
            "direct_dac_sub_allpass_freq_hz": float(analysis.get("allpass_freq_hz", 0.0)),
            "direct_dac_sub_allpass_q": float(analysis.get("allpass_q", 0.707)),
            "bass_integration_sub_delay_ms": float(_safe_float(sub_delay_ms, 0.0)),
            "bass_integration_sub_polarity_invert": bool(sub_polarity_invert),
            "bass_integration_sub_gain_trim_db": float(_safe_float(sub_gain_trim_db, 0.0)),
        }
    )
    return out
