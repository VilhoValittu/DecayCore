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
import scipy.ndimage

logger = logging.getLogger("DecayCore")

from ...common.acoustic_stats import _clamp, calc_acoustic_score, calc_ai_summary_from_stats
from ...common.comparison_stats import _make_comparison_stats
from ...dsp.phase_ir_metrics import format_pre_energy_status
from ...dsp.quality_metrics import _mag_error_db, _rms
from ...dsp.target_match import target_match_from_stats as _target_match_from_stats_ssot
from ...ui_i18n import LVL_MODE_AUTO, lvl_mode_legacy_name
















_calc_acoustic_score = calc_acoustic_score

__all__ = [
    'format_band_rt60_summary',
    '_float_allow_zero',
    '_clamp',
    'calc_acoustic_score',
    '_calc_acoustic_score',
    'calc_ai_summary_from_stats',
    '_calc_target_match',
    'calc_target_match_from_stats',
    'format_dsp_quality_report_block',
    'format_summary_content',
]

def _format_summary_content_legacy(settings, l_stats, r_stats):
    """Jasentaa tai muotoilee: format summary content."""
    from datetime import datetime
    import numpy as np

    settings = settings or {}
    l_stats = l_stats or {}
    r_stats = r_stats or {}
    program_version = str(settings.get("program_version", "") or "").strip()

    def _safe_float(v, default=0.0):
        try:
            x = float(v)
            if np.isfinite(x):
                return x
        except Exception:
            logger.exception("safe float parse in score summary")
        return float(default)

    def _fmt_score(v):
        return "n/a" if v is None else f"{float(v):.3f}/100"

    def _fmt_match(match_pct, rms_db):
        if match_pct is None or rms_db is None:
            return "n/a"
        return f"{float(match_pct):.1f}% (RMS {float(rms_db):.2f} dB)"

    def _fmt_range(rng):
        if not isinstance(rng, (list, tuple)) or len(rng) < 2:
            return "n/a"
        try:
            return f"{float(rng[0]):.0f}-{float(rng[1]):.0f} Hz"
        except Exception:
            return "n/a"

    def _report_offset_db(st: dict) -> float:
        try:
            if "cmp_offset_db" in st:
                return _safe_float(st.get("cmp_offset_db", 0.0), 0.0)
        except Exception:
            logger.exception("report offset_db fetch")
        return _safe_float(st.get("offset_db", 0.0), 0.0)

    def _large_offset_warning_line() -> str | None:
        off_l = abs(_report_offset_db(l_stats))
        off_r = abs(_report_offset_db(r_stats))
        max_off = max(off_l, off_r)
        if max_off < 30.0:
            return None
        return (
            "Warning: Very large leveling offset detected "
            f"(L {off_l:.2f} dB | R {off_r:.2f} dB). "
            "Measurement level appears relative/uncalibrated, so absolute-level report lines should be interpreted cautiously."
        )

    def _phase_clamp_line(side: str, st: dict) -> str:
        lim = st.get("phase_corr_clamp_deg", None)
        bef = st.get("phase_corr_max_before_deg", None)
        if lim is None or bef is None:
            return f"{side}: n/a"
        return f"{side}: max {float(bef):.1f} deg -> clamp {float(lim):.1f} deg"

    def _gd_grad_max_value(st: dict):
        keys = (
            "gd_limiter_max_grad_ms_per_oct",
            "gd_grad_limiter_max_grad_ms_per_oct",
            "gd_limiter_max_grad_after_ms_per_oct",
            "gd_grad_limiter_max_grad_after_ms_per_oct",
            "gd_limiter_max_grad_before_ms_per_oct",
            "gd_grad_limiter_max_grad_before_ms_per_oct",
        )
        for k in keys:
            try:
                v = float(st.get(k, None))
                if np.isfinite(v):
                    return float(v)
            except Exception:
                continue
        return None

    def _fmt_gd_grad_max(st: dict) -> str:
        v = _gd_grad_max_value(st)
        return "n/a" if v is None else f"{float(v):.2f} ms/oct"

    def _gd_limiter_line(side: str, st: dict) -> str:
        try:
            enabled = bool(st.get("gd_limiter_enabled", st.get("gd_grad_limiter_enabled", False)))
            reason = str(st.get("gd_limiter_reason", st.get("gd_grad_limiter_reason", "unknown")) or "unknown")
            limit_v = st.get("gd_limiter_limit_ms_per_oct", st.get("gd_grad_limit_ms_per_oct", None))
            grad_before = st.get(
                "gd_limiter_max_grad_before_ms_per_oct",
                st.get("gd_grad_limiter_max_grad_before_ms_per_oct", None),
            )
            grad_after = st.get(
                "gd_limiter_max_grad_after_ms_per_oct",
                st.get("gd_grad_limiter_max_grad_after_ms_per_oct", _gd_grad_max_value(st)),
            )
            before_hz = st.get(
                "gd_limiter_max_grad_before_hz",
                st.get("gd_grad_limiter_max_grad_before_hz", None),
            )

            lim_txt = "n/a"
            if limit_v is not None:
                try:
                    lim_txt = f"{float(limit_v):.2f} ms/oct"
                except Exception:
                    lim_txt = "n/a"

            grad_txt = "n/a"
            try:
                gb = float(grad_before) if grad_before is not None else None
            except Exception:
                gb = None
            try:
                ga = float(grad_after) if grad_after is not None else None
            except Exception:
                ga = None
            try:
                bh = float(before_hz) if before_hz is not None else None
            except Exception:
                bh = None
            if gb is not None and np.isfinite(gb) and ga is not None and np.isfinite(ga):
                hz_txt = f" @{bh:.0f}Hz" if bh is not None and np.isfinite(bh) else ""
                grad_txt = f"{gb:.2f}{hz_txt} -> {ga:.2f} ms/oct"
            elif ga is not None and np.isfinite(ga):
                grad_txt = f"{ga:.2f} ms/oct"

            return (
                f"{side}: {'ON' if enabled else 'OFF'} "
                f"(reason={reason}, limit={lim_txt}, GD-gradient max {grad_txt})"
            )
        except Exception:
            return f"{side}: n/a"

    def _afdw_line(side: str, st: dict) -> str:
        mode = str(st.get("fdw_mode", "") or "").strip().lower()
        if mode == "fixed":
            cyc = st.get("fdw_fixed_cycles", settings.get("fdw_cycles", None))
            bw = st.get("fdw_fixed_bw_oct", None)
            try:
                cyc_txt = f"{float(cyc):.2f}" if cyc is not None else "n/a"
            except Exception:
                cyc_txt = "n/a"
            try:
                bw_txt = f"{float(bw):.4f}" if bw is not None else "n/a"
            except Exception:
                bw_txt = "n/a"
            return f"{side}: FIXED | cycles={cyc_txt}, BW={bw_txt} oct (A-FDW OFF)"

        active = bool(st.get("afdw_active", False)) or bool(settings.get("enable_afdw", False))
        if not active:
            return f"{side}: OFF"
        mn = st.get("afdw_bw_min_oct", None)
        me = st.get("afdw_bw_mean_oct", None)
        mx = st.get("afdw_bw_max_oct", None)
        if mn is None or me is None or mx is None:
            return f"{side}: ON (effective bandwidth not available)"
        return f"{side}: ON | BW min/mean/max = {float(mn):.4f}/{float(me):.4f}/{float(mx):.4f} oct"

    def _fmt_bands(bands):
        return format_band_rt60_summary(bands)

    def _worst_event(st: dict) -> str:
        refs = st.get("reflections", []) or []
        if not refs:
            return "None"
        try:
            w = max(refs, key=lambda x: float(x.get("gd_error", 0.0) or 0.0))
        except Exception:
            return "None"
        freq = _safe_float(w.get("freq", 0.0), 0.0)
        gd_ms = _safe_float(w.get("gd_error", 0.0), 0.0)
        typ = str(w.get("type", "Event") or "Event")
        return f"{typ} at {freq:.0f} Hz ({gd_ms:.2f} ms)"

    def _calc_acoustic_score(conf_pct, match_pct, rt60_s=None, rt60_reliability=None):
        try:
            return globals()["calc_acoustic_score"](
                float(conf_pct),
                float(match_pct),
                rt60_s=rt60_s,
                rt60_rel=rt60_reliability,
            )
        except Exception:
            conf_pct = float(np.clip(float(conf_pct), 0.0, 100.0))
            match_pct = float(np.clip(float(match_pct), 0.0, 100.0))
            return float(np.clip(0.60 * match_pct + 0.40 * conf_pct, 0.0, 100.0))

    l_rt = _safe_float(l_stats.get("rt60_val", 0.0), 0.0)
    r_rt = _safe_float(r_stats.get("rt60_val", 0.0), 0.0)
    l_band_avg = _safe_float(l_stats.get("rt60_band_avg", 0.0), 0.0)
    r_band_avg = _safe_float(r_stats.get("rt60_band_avg", 0.0), 0.0)
    l_conf = _safe_float(l_stats.get("cmp_avg_confidence", l_stats.get("avg_confidence", 0.0)), 0.0)
    r_conf = _safe_float(r_stats.get("cmp_avg_confidence", r_stats.get("avg_confidence", 0.0)), 0.0)
    l_rms, l_match = _calc_target_match(l_stats)
    r_rms, r_match = _calc_target_match(r_stats)
    l_rms_raw, l_match_raw = _target_match_from_stats_ssot(
        l_stats or {},
        include_filter=False,
        use_confidence=True,
        use_smart_scan_range=True,
    )
    r_rms_raw, r_match_raw = _target_match_from_stats_ssot(
        r_stats or {},
        include_filter=False,
        use_confidence=True,
        use_smart_scan_range=True,
    )

    l_score = None
    if l_match is not None:
        l_score = _calc_acoustic_score(
            l_conf,
            l_match,
            l_stats.get("rt60_val", None),
            l_stats.get("rt60_reliability", None),
        )
    r_score = None
    if r_match is not None:
        r_score = _calc_acoustic_score(
            r_conf,
            r_match,
            r_stats.get("rt60_val", None),
            r_stats.get("rt60_reliability", None),
        )

    lines = [
        "=== DecayCore - Filter Generation Summary ===",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    if program_version:
        lines.append(f"Version: {program_version}")
    lines += [
        "",
        "--- Executive Summary ---",
        f"Acoustic Score: L {_fmt_score(l_score)} | R {_fmt_score(r_score)}",
        f"Target Match:   L {_fmt_match(l_match, l_rms)} | R {_fmt_match(r_match, r_rms)}",
        f"Confidence:     L {l_conf:.1f}% | R {r_conf:.1f}%",
        f"RT60 Wideband:  L {l_rt:.2f}s | R {r_rt:.2f}s",
        f"Worst Event:    L {_worst_event(l_stats)} | R {_worst_event(r_stats)}",
        "",
        "--- Core Settings ---",
    ]

    keys = [
        "mode",
        "fs",
        "taps",
        "filter_type",
        "mixed_freq",
        "mag_c_min",
        "mag_c_max",
        "max_boost",
        "max_cut_db",
        "max_slope_db_per_oct",
        "hpf_enable",
        "hpf_freq",
        "hpf_slope",
        "enable_tdc",
        "tdc_strength",
        "enable_afdw",
        "comparison_mode",
        "stereo_link",
        "bass_first_ai",
    ]
    for k in keys:
        if k in settings:
            lines.append(f"{k}: {settings.get(k)}")
    try:
        lvl_mode_value = lvl_mode_legacy_name(settings.get("lvl_mode", LVL_MODE_AUTO))
    except Exception:
        lvl_mode_value = lvl_mode_legacy_name(LVL_MODE_AUTO)
    lines.append(f"Level match mode: {lvl_mode_value}")
    if lvl_mode_value.lower() == "manual":
        lines.append(f"Manual target level: {_safe_float(settings.get('lvl_manual_db', 0.0), 0.0):+.1f} dB")
        lines.append(
            "Manual target tilt: "
            f"{_safe_float(settings.get('manual_target_tilt_db_per_oct', 0.0), 0.0):+.1f} dB/oct @ 1 kHz"
        )

    lines.append("\n--- Analysis Mode ---")
    lines.append(
        f"Analysis mode: L {str(l_stats.get('analysis_mode', 'native'))} | "
        f"R {str(r_stats.get('analysis_mode', 'native'))}"
    )
    if str(l_stats.get("analysis_mode", "native")) == "comparison":
        lines.append(
            f"Comparison grid (L): fs={_safe_float(l_stats.get('cmp_ref_fs', 0), 0):.0f} "
            f"taps={_safe_float(l_stats.get('cmp_ref_taps', 0), 0):.0f}"
        )
    if str(r_stats.get("analysis_mode", "native")) == "comparison":
        lines.append(
            f"Comparison grid (R): fs={_safe_float(r_stats.get('cmp_ref_fs', 0), 0):.0f} "
            f"taps={_safe_float(r_stats.get('cmp_ref_taps', 0), 0):.0f}"
        )

    max_cut_db = _safe_float(settings.get("max_cut_db", 15.0), 15.0)
    max_slope = _safe_float(settings.get("max_slope_db_per_oct", 12.0), 12.0)
    max_slope_boost = _safe_float(settings.get("max_slope_boost_db_per_oct", 0.0), 0.0) or max_slope
    max_slope_cut = _safe_float(settings.get("max_slope_cut_db_per_oct", 0.0), 0.0) or max_slope
    low_bass_cut_hz = _float_allow_zero(settings.get("low_bass_cut_hz", 40.0), 40.0)

    lines.append("\n--- Correction Guards ---")
    lines.append(f"Max cut: -{max_cut_db:.1f} dB")
    if abs(max_slope_boost - max_slope_cut) > 1e-9:
        lines.append(f"Slope: boost {max_slope_boost:.1f} dB/oct | cut {max_slope_cut:.1f} dB/oct")
    else:
        lines.append(f"Max slope: {max_slope:.1f} dB/oct")
    lines.append(f"Low-bass cut policy: <{low_bass_cut_hz:.1f} Hz (cuts only)")

    lines.append("\n--- Temporal Decay Control (TDC) ---")
    tdc_enabled = bool(settings.get("enable_tdc", False))
    lines.append(f"TDC enabled: {'YES' if tdc_enabled else 'NO'}")
    if tdc_enabled:
        lines.append(f"TDC strength: {_safe_float(settings.get('tdc_strength', 0), 0):.0f} %")
        lines.append(f"TDC max reduction: {_safe_float(settings.get('tdc_max_reduction_db', 0), 0):.1f} dB")
        slope = _safe_float(settings.get("tdc_slope_db_per_oct", 0), 0)
        if slope > 0:
            lines.append(f"TDC slope limit: {slope:.1f} dB/oct")
        l_tdc_peak = _safe_float(l_stats.get("tdc_peak_reduction_db", 0.0), 0.0)
        r_tdc_peak = _safe_float(r_stats.get("tdc_peak_reduction_db", 0.0), 0.0)
        if l_tdc_peak > 0.0 or r_tdc_peak > 0.0:
            lines.append(
                "TDC applied peak: "
                f"L {l_tdc_peak:.2f} dB @ {_safe_float(l_stats.get('tdc_peak_reduction_hz', 0.0), 0.0):.1f} Hz | "
                f"R {r_tdc_peak:.2f} dB @ {_safe_float(r_stats.get('tdc_peak_reduction_hz', 0.0), 0.0):.1f} Hz"
            )
            lines.append(
                "TDC events used: "
                f"L {int(round(_safe_float(l_stats.get('tdc_events_used', 0.0), 0.0)))} | "
                f"R {int(round(_safe_float(r_stats.get('tdc_events_used', 0.0), 0.0)))}"
            )

    lines.append("\n--- A-FDW ---")
    lines.append(_afdw_line("Left", l_stats))
    lines.append(_afdw_line("Right", r_stats))

    lines.append("\n--- XO and Phase ---")
    lines.append(f"XO phase model: L {l_stats.get('xo_summary', '-')} | R {r_stats.get('xo_summary', '-')}")
    lines.append(_phase_clamp_line("L", l_stats))
    lines.append(_phase_clamp_line("R", r_stats))
    lines.append(_gd_limiter_line("L", l_stats))
    lines.append(_gd_limiter_line("R", r_stats))
    lines.append(f"A/B GD-gradient max: L {_fmt_gd_grad_max(l_stats)} | R {_fmt_gd_grad_max(r_stats)}")

    lines.append("\n--- RT60 and Confidence ---")
    lines.append(f"RT60 wideband: L {l_rt:.2f}s | R {r_rt:.2f}s")
    if (l_band_avg > 0.0) or (r_band_avg > 0.0):
        lines.append(f"RT60 band average (125-4kHz): L {l_band_avg:.2f}s | R {r_band_avg:.2f}s")
    l_bands = l_stats.get("rt60_bands", {}) or {}
    r_bands = r_stats.get("rt60_bands", {}) or {}
    if l_bands or r_bands:
        lines.append(f"Band RT60 L: {_fmt_bands(l_bands)}")
        lines.append(f"Band RT60 R: {_fmt_bands(r_bands)}")
    lines.append(f"Confidence: L {l_conf:.1f}% | R {r_conf:.1f}%")

    l_om = l_stats.get("cmp_offset_method", l_stats.get("offset_method", "")) or "-"
    r_om = r_stats.get("cmp_offset_method", r_stats.get("offset_method", "")) or "-"
    lines.append(f"Offset method: L {l_om} | R {r_om}")
    l_win = l_stats.get("cmp_smart_scan_range", l_stats.get("smart_scan_range", None))
    r_win = r_stats.get("cmp_smart_scan_range", r_stats.get("smart_scan_range", None))
    lines.append(f"Level window: L {_fmt_range(l_win)} | R {_fmt_range(r_win)}")

    lines.append("\n--- Target Curve Match ---")
    lines.append(f"Left:  {_fmt_match(l_match, l_rms)}")
    lines.append(f"Right: {_fmt_match(r_match, r_rms)}")
    lines.append(
        "Debug raw->pred: "
        f"L {_fmt_match(l_match_raw, l_rms_raw)} -> {_fmt_match(l_match, l_rms)} | "
        f"R {_fmt_match(r_match_raw, r_rms_raw)} -> {_fmt_match(r_match, r_rms)}"
    )

    lines += format_dsp_quality_report_block(settings, l_stats, r_stats)

    lines.append("\n--- Alignment and Peaks ---")
    lines.append(f"L peak (pre-norm): {_safe_float(l_stats.get('peak_before_norm', 0), 0):.2f} dB")
    lines.append(f"R peak (pre-norm): {_safe_float(r_stats.get('peak_before_norm', 0), 0):.2f} dB")
    lines.append(f"Global offset applied: {_safe_float(l_stats.get('offset_db', 0), 0):.2f} dB")
    offset_warn = _large_offset_warning_line()
    if offset_warn:
        lines.append(offset_warn)
    lines.append(f"Auto gain margin setting: {_safe_float(settings.get('gain', 0.0), 0.0):.2f} dB")
    lines.append(
        f"Applied auto gain: L {_safe_float(l_stats.get('auto_global_gain_db', 0.0), 0.0):.2f} dB | "
        f"R {_safe_float(r_stats.get('auto_global_gain_db', 0.0), 0.0):.2f} dB"
    )

    return "\n".join(lines)

def format_summary_content(settings, l_stats, r_stats):
    """Jasentaa tai muotoilee: format summary content."""
    settings = settings or {}
    l_stats = l_stats or {}
    r_stats = r_stats or {}

    if bool(settings.get("comparison_mode", False)):
        l_stats = _make_comparison_stats(l_stats, 44100, 65536)
        r_stats = _make_comparison_stats(r_stats, 44100, 65536)

    return _format_summary_content_legacy(settings, l_stats, r_stats)


__all__ = ['_format_summary_content_legacy', 'format_summary_content']


def _load_sibling_symbols() -> None:
    import importlib
    package = __package__
    for module_name in ['quality_report', 'legacy_summary']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_load_sibling_symbols()
