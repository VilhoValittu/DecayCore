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
import math


logger = logging.getLogger(__name__)
from ...dsp.lr_difference_metrics import compute_lr_difference_metrics

_AUTO_ASYM_PHASE1_SEARCH_SPACE_EST = 1877500016615829065655090169509480

_RECOVERABLE_EVENT_EXCEPTIONS = (
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
)


def _safe_lower(value) -> str:
    try:
        return str(value).strip().lower()
    except _RECOVERABLE_EVENT_EXCEPTIONS:
        return ""


def _append_reflections(summary_content: str, side: str, st: dict) -> str:
    reflections = st.get("reflections") or []
    if not reflections:
        return summary_content
    summary_content += f"\n=== ACOUSTIC EVENTS ({side}) ===\n"
    summary_content += (
        "Note: 'Path delta' is an equivalent path-length from dt.\n"
        "Reflections: time-of-flight equivalent extra path.\n"
        "Resonances: not a physical distance.\n"
    )
    summary_content += f"{'Freq (Hz)':<10} {'Type':<12} {'dt (ms)':<12} {'Path delta (m)':<14}\n"
    summary_content += "-" * 50 + "\n"
    for rev in reflections:
        freq = float(rev.get("freq", 0) or 0)
        ev_type = str(rev.get("type", "Event") or "Event")
        gd_error = float(rev.get("gd_error", 0) or 0)
        dist = float(rev.get("dist", 0) or 0)
        if "reson" in _safe_lower(ev_type):
            summary_content += f"{freq:<10} {ev_type:<12} {gd_error:<12} {'n/a':<14}\n"
        else:
            summary_content += f"{freq:<10} {ev_type:<12} {gd_error:<12} {dist:<14}\n"
    return summary_content


def _append_headroom(summary_content: str, side: str, st: dict) -> str:
    summary_content += f"\n=== HEADROOM MANAGEMENT ({side}) ===\n"
    summary_content += f"Auto Gain Margin: {float(st.get('gain_margin_db', 0.0)):.2f} dB\n"
    summary_content += f"Applied Auto Gain: {float(st.get('auto_global_gain_db', 0.0)):.2f} dB\n"
    summary_content += f"Normalize: {'ON' if bool(st.get('do_normalize', False)) else 'OFF'}\n"
    summary_content += f"Peak Gain (pre-headroom): {float(st.get('peak_gain_db', 0.0)):.2f} dB\n"
    summary_content += f"Applied Headroom: {float(st.get('auto_headroom_db', 0.0)):.2f} dB\n"
    summary_content += f"Final Max (filter+auto_gain+headroom): {float(st.get('final_max_db', 0.0)):.2f} dB\n"
    return summary_content


def _append_boost_cut(summary_content: str, side: str, st: dict) -> str:
    summary_content += f"\n=== BOOST/CUT DIAGNOSTICS ({side}) ===\n"
    mb_eff = float(st.get("max_boost_db_effective", st.get("max_boost_db", 0.0)) or 0.0)
    mb_user = float(st.get("max_boost_db_user", st.get("max_boost_db", 0.0)) or 0.0)
    mb_cap = float(st.get("max_safe_boost_db", 0.0) or 0.0)
    if mb_cap > 0.0 and mb_user > mb_eff + 1e-9:
        summary_content += f"Config: max_boost_db={mb_eff:.2f} dB (user={mb_user:.2f}, cap={mb_cap:.2f}), "
    else:
        summary_content += f"Config: max_boost_db={mb_eff:.2f} dB, "
    summary_content += f"max_cut_db={float(st.get('max_cut_db', 0.0)):.2f} dB\n"
    summary_content += f"Config: low_bass_cut_hz={float(st.get('low_bass_cut_hz', 0.0)):.1f} Hz, "
    summary_content += f"exc_prot={'ON' if bool(st.get('exc_prot', False)) else 'OFF'}, "
    summary_content += f"exc_freq={float(st.get('exc_freq', 0.0)):.1f} Hz, "
    summary_content += f"max_slope_db_per_oct={float(st.get('max_slope_db_per_oct', 0.0)):.1f}\n"
    summary_content += f"Result (post-clamp): boost_peak={float(st.get('boost_peak_db', 0.0)):.2f} dB, "
    summary_content += f"cut_peak={float(st.get('cut_peak_db', 0.0)):.2f} dB, "
    summary_content += f"boost_bins={int(st.get('boost_bins', 0))}\n"
    summary_content += f"Net boost peak (post global/headroom): {float(st.get('net_boost_peak_db', 0.0)):.2f} dB\n"
    candidate_boost_peak = float(st.get("boost_candidate_peak_db", 0.0) or 0.0)
    summary_content += f"Candidate (pre-softclip): boost_peak={candidate_boost_peak:.2f} dB, "
    summary_content += f"boost_bins={int(st.get('boost_candidate_bins', 0))}, "
    summary_content += f"lowbass_boost_bins={int(st.get('boost_candidate_bins_lowbass', 0))}, "
    summary_content += f"excprot_boost_bins={int(st.get('boost_candidate_bins_excprot', 0))}\n"
    if candidate_boost_peak >= 15.0:
        summary_content += (
            "Large boost candidate warning: "
            f"{candidate_boost_peak:.2f} dB requested before safety limits; "
            "treat this as uncorrectable-room/target pressure rather than safe boost headroom.\n"
        )
    summary_content += f"Boost blocked reason: {st.get('boost_blocked_reason', 'n/a')!s}\n"
    return summary_content


def _append_hybrid_iir(summary_content: str, side: str, st: dict) -> str:
    summary_content += f"\n=== HYBRID IIR MODAL CUTS ({side}) ===\n"
    enabled = bool(st.get("hybrid_iir_enabled", False))
    biquads = [dict(item) for item in list(st.get("hybrid_iir_biquads", []) or []) if isinstance(item, dict)]
    rejected = [dict(item) for item in list(st.get("hybrid_iir_rejected", []) or []) if isinstance(item, dict)]
    summary_content += f"State: {'ON' if enabled else 'OFF'}\n"
    summary_content += "Mode: magnitude preconditioning only\n"
    if biquads:
        summary_content += "IIR modal cuts reduce excitation of selected minimum-phase-like bass peaks.\n"
        for idx, biquad in enumerate(biquads, start=1):
            summary_content += (
                f"- #{idx}: {float(biquad.get('freq', 0.0) or 0.0):.1f} Hz, "
                f"Q {float(biquad.get('q', 0.0) or 0.0):.2f}, "
                f"gain {float(biquad.get('gain', 0.0) or 0.0):+.2f} dB, "
                f"confidence {float(biquad.get('confidence', 0.0) or 0.0):.2f}, "
                f"safe_cut {float(biquad.get('safe_cut_db', 0.0) or 0.0):.2f} dB\n"
            )
    elif enabled:
        summary_content += "Selected cuts: none\n"
        if rejected:
            reasons = ", ".join(sorted({str(item.get("reason", "unknown")) for item in rejected[:6]}))
            summary_content += f"Rejected candidate reasons: {reasons}\n"
    return summary_content


def _append_clamp_diagnostics(summary_content: str, side: str, st: dict) -> str:
    summary_content += f"\n=== CLAMP DIAGNOSTICS ({side}) ===\n"
    summary_content += f"{st.get('clamp_summary', 'n/a')!s}\n"
    summary_content += (
        f"soft_clip: boost_bins={int(st.get('softclip_boost_bins', 0))}, "
        f"cut_bins={int(st.get('softclip_cut_bins', 0))}, "
        f"worst_over_boost={float(st.get('softclip_worst_over_boost_db', 0.0)):.2f} dB, "
        f"worst_over_cut={float(st.get('softclip_worst_over_cut_db', 0.0)):.2f} dB\n"
    )
    summary_content += (
        f"hard_clamp: boost_bins={int(st.get('hardclamp_boost_bins', 0))}, "
        f"cut_bins={int(st.get('hardclamp_cut_bins', 0))}, "
        f"worst_over_boost={float(st.get('hardclamp_worst_over_boost_db', 0.0)):.2f} dB, "
        f"worst_over_cut={float(st.get('hardclamp_worst_over_cut_db', 0.0)):.2f} dB\n"
    )
    return summary_content


def _append_mag_authority_trace(summary_content: str, side: str, st: dict) -> str:
    trace = st.get("mag_authority_trace") or []
    if not (isinstance(trace, list) and trace):
        return summary_content
    summary_content += f"\n=== MAGNITUDE AUTHORITY TRACE ({side}) ===\n"
    hard_boost = any(
        isinstance(item, dict)
        and "hardclamp_boost" in (item.get("reason_codes") or [])
        and int(item.get("changed_bins", 0) or 0) > 0
        for item in trace
    )
    soft_boost = any(
        isinstance(item, dict)
        and "softclip_boost" in (item.get("reason_codes") or [])
        and int(item.get("changed_bins", 0) or 0) > 0
        for item in trace
    )
    final_boost = float(st.get("boost_peak_db", 0.0) or 0.0)
    max_boost = float(st.get("max_boost_db_effective", st.get("max_boost_db", 0.0)) or 0.0)
    if hard_boost or soft_boost:
        summary_content += (
            "Authority verdict: large boost candidate was safely reduced by "
            f"{'softclip + hardclamp' if hard_boost and soft_boost else ('hardclamp' if hard_boost else 'softclip')}; "
            f"final boost {float(final_boost):.2f} dB stayed within max_boost {float(max_boost):.2f} dB.\n"
        )
    summary_content += (
        f"{'Stage':<30} {'Reasons':<72} {'Changed':>8} "
        f"{'MaxDelta':>9} {'BoostPk':>8} {'CutPk':>8}\n"
    )
    summary_content += "-" * 139 + "\n"
    for item in trace:
        if not isinstance(item, dict):
            continue
        changed = int(item.get("changed_bins", 0) or 0)
        reasons = item.get("reason_codes") or []
        if not changed and not reasons:
            continue
        stage = str(item.get("stage", "stage") or "stage")[:30]
        reason_text = ",".join(str(r) for r in reasons if str(r))
        max_delta = float(item.get("max_delta_db", 0.0) or 0.0)
        boost_pk = float(item.get("boost_peak_after_db", 0.0) or 0.0)
        cut_pk = float(item.get("cut_peak_after_db", 0.0) or 0.0)
        summary_content += (
            f"{stage:<30} {reason_text:<72} {changed:>8d} "
            f"{max_delta:>9.2f} {boost_pk:>8.2f} {cut_pk:>8.2f}\n"
        )
    return summary_content


def _append_stage_checkpoints(summary_content: str, side: str, st: dict) -> str:
    probes = st.get("stage_probes") or {}
    if not (isinstance(probes, dict) and probes):
        return summary_content
    summary_content += f"\n=== STAGE CHECKPOINTS ({side}) ===\n"
    summary_content += f"{'Stage':<22} {'BoostPk':>8} {'CutPk':>8} {'BoostBins':>10} {'CutBins':>8} {'NetBoostPk':>11}\n"
    summary_content += "-" * 75 + "\n"
    order = [
        "after_gain_apply",
        "after_lowbass_policy",
        "after_slope",
        "after_fade",
        "pre_softclip",
        "post_softclip",
        "post_hardclamp",
    ]
    for key in order:
        p = probes.get(key)
        if not isinstance(p, dict):
            continue
        stage = str(p.get("stage", key))
        bpk = float(p.get("boost_peak_db", 0.0) or 0.0)
        cpk = float(p.get("cut_peak_db", 0.0) or 0.0)
        bb = int(p.get("boost_bins", 0) or 0)
        cb = int(p.get("cut_bins", 0) or 0)
        nbp = float(p.get("net_boost_peak_db", 0.0) or 0.0)
        summary_content += f"{stage:<22} {bpk:>8.2f} {cpk:>8.2f} {bb:>10d} {cb:>8d} {nbp:>11.2f}\n"
    return summary_content


def _append_bass_first_ai(summary_content: str, side: str, st: dict) -> str:
    probes = st.get("stage_probes") or {}
    if not (isinstance(probes, dict) and probes):
        return summary_content
    summary_content += f"\n=== BASS-FIRST AI ({side}) ===\n"
    summary_content += f"Bass-first AI active: {'YES' if bool(st.get('bass_first_ai', False)) else 'NO'}\n"

    pk_hz = st.get("bass_first_mode_peak_hz")
    pk_sc = st.get("bass_first_mode_peak_score")
    if (pk_hz is not None) and (pk_sc is not None):
        summary_content += f"Mode peak: {float(pk_hz):.1f} Hz (score {float(pk_sc):.2f})\n"
    else:
        summary_content += "Mode peak: n/a\n"

    summary_content += f"Smoothing conf floor applied: {'YES' if bool(st.get('bass_first_conf_floor_applied', False)) else 'NO'}\n"
    rm_max = st.get("bass_first_roommode_max_20_200")
    rel_mean = st.get("bass_first_rel_mean_20_200")
    rel_min = st.get("bass_first_rel_min_20_200")
    conf_eff_mean = st.get("bass_first_conf_eff_mean_20_200")
    conf_eff_min = st.get("bass_first_conf_eff_min_20_200")
    floor_applied = bool(st.get("bass_first_conf_floor_applied", False))
    if (rm_max is not None) or (rel_mean is not None) or (rel_min is not None) or (conf_eff_mean is not None):
        summary_content += (
            f"BF masks (20-200): "
            f"roommode_max={float(rm_max or 0.0):.3f}, "
            f"rel_mean(raw)={float(rel_mean or 0.0):.3f}, "
            f"rel_min(raw)={float(rel_min or 0.0):.3f}, "
            f"conf_eff_mean={float(conf_eff_mean or 0.0):.3f}, "
            f"conf_eff_min={float(conf_eff_min or 0.0):.3f}, "
            f"conf_floor_applied={'YES' if floor_applied else 'NO'}\n"
        )

    src = st.get("bass_first_source")
    if isinstance(src, str) and src.strip():
        summary_content += f"BassFirst source: {src.strip()}\n"
    return summary_content


def _append_side_acoustic_events(summary_content: str, side: str, st: dict) -> str:
    summary_content = _append_reflections(summary_content, side, st)
    summary_content = _append_headroom(summary_content, side, st)
    summary_content = _append_boost_cut(summary_content, side, st)
    summary_content = _append_hybrid_iir(summary_content, side, st)
    summary_content = _append_clamp_diagnostics(summary_content, side, st)
    summary_content = _append_mag_authority_trace(summary_content, side, st)
    summary_content = _append_stage_checkpoints(summary_content, side, st)
    summary_content = _append_bass_first_ai(summary_content, side, st)
    return summary_content


def _append_acoustic_events(summary_content, l_st, r_st):
    for side, st in [("LEFT", l_st), ("RIGHT", r_st)]:
        summary_content = _append_side_acoustic_events(summary_content, side, st)
    return summary_content

def _append_lr_difference_summary(summary_content: str, l_st: dict, r_st: dict) -> str:
    """Append L/R difference metrics section to summary text.

    Computed from measured/analyzed left-right response data.  Values reflect
    actual room asymmetry and change with measurement data.
    """
    try:
        lr = compute_lr_difference_metrics(l_st, r_st)

        def _fmt_db(v: float) -> str:
            return f"{v:.2f} dB" if math.isfinite(v) else "n/a"

        def _fmt_ms(v: float) -> str:
            return f"{v:.2f} ms" if math.isfinite(v) else "n/a"

        # Only append if we have at least one meaningful value.
        has_any = any(
            math.isfinite(v)
            for v in (
                lr.mag_rms_bass_db,
                lr.mag_rms_mid_db,
                lr.mag_rms_band_db,
                lr.mag_maxabs_band_db,
            )
        )
        if not has_any:
            return summary_content

        band_lo = lr.band_lo_hz if math.isfinite(lr.band_lo_hz) else 20.0
        band_hi = lr.band_hi_hz if math.isfinite(lr.band_hi_hz) else 2000.0

        summary_content += "\n=== L/R DIFFERENCE METRICS ===\n"
        summary_content += f"L/R Mag RMS  (20–200 Hz):  {_fmt_db(lr.mag_rms_bass_db)}\n"
        summary_content += f"L/R Mag RMS  (200–2000 Hz): {_fmt_db(lr.mag_rms_mid_db)}\n"
        summary_content += (
            f"L/R Mag RMS  ({band_lo:.0f}–{band_hi:.0f} Hz): {_fmt_db(lr.mag_rms_band_db)}\n"
        )
        summary_content += (
            f"L/R Mag MaxAbs ({band_lo:.0f}–{band_hi:.0f} Hz): {_fmt_db(lr.mag_maxabs_band_db)}\n"
        )
        if math.isfinite(lr.gd_rms_band_ms):
            summary_content += (
                f"L/R GD RMS ({band_lo:.0f}–{band_hi:.0f} Hz): {_fmt_ms(lr.gd_rms_band_ms)}\n"
            )
        summary_content += (
            "Computed from measured/analyzed left-right response data. "
            "Lower values indicate more symmetric channel behavior.\n"
        )
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
    ) as exc:
        logger.debug("_append_lr_difference_summary failed: %s", exc, exc_info=True)
    return summary_content


__all__ = ['_append_acoustic_events', '_append_lr_difference_summary']

