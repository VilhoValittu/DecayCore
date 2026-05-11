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
import sys

from ...config.decaycore_pipeline import filter_type_supports_xo_phase_model

logger = logging.getLogger(__name__)
from ...dsp.smoothing import AFDW_BW_MAX_OCT, AFDW_BW_MIN_OCT
from ...dsp.lr_difference_metrics import compute_lr_difference_metrics
from ...auto_mode.rank_score import attach_official_rank_score, calibrated_auto_quality, display_rank_score, official_rank_score, _quality_band
from ..export_scoring import _pick_metric, _safe_float

_AUTO_ASYM_PHASE1_SEARCH_SPACE_EST = 1877500016615829065655090169509480

def _append_dsp_effective_params(summary_content, data, fs_v):
    try:
        enable_afdw = bool(data.get("enable_afdw", False))
        enable_tdc = bool(data.get("enable_tdc", False))
        tdc_strength = float(data.get("tdc_strength", 0.0) or 0.0)
        fdw_cycles = float(data.get("fdw_cycles", 15.0) or 15.0)
        fdw_oct_width = (2.0 / fdw_cycles) if fdw_cycles > 0 else 0.0
        afdw_min = max(3.0, fdw_cycles / 3.0)
        afdw_min_oct_width = (2.0 / afdw_min) if afdw_min > 0 else 0.0
        fdw_oct_width = float(max(AFDW_BW_MIN_OCT, min(AFDW_BW_MAX_OCT, fdw_oct_width)))
        afdw_min_oct_width = float(max(AFDW_BW_MIN_OCT, min(AFDW_BW_MAX_OCT, afdw_min_oct_width)))

        df_on = bool(data.get("df_smoothing", False))
        df_ref = 44100.0 / 65536.0
        fsmooth = data.get("filter_smooth", data.get("smoothing_level", 96))
        base_sigma = 60 // (fsmooth / 12 if (fsmooth or 0) > 0 else 1)
        sigma_hz = float(base_sigma) * df_ref
        df_cur = float(fs_v) / float(data.get("taps", 65536) or 65536)
        sigma_bins = (sigma_hz / df_cur) if (df_cur and df_cur > 0) else float(base_sigma)

        summary_content += "\n=== DSP EFFECTIVE PARAMS (THIS SAMPLE RATE) ===\n"
        summary_content += f"Sample rate: {int(fs_v)} Hz\n"
        try:
            _otilt = float(data.get("output_tilt_db_per_oct", 0.0) or 0.0)
            if _otilt != 0.0:
                summary_content += f"Output tilt: {_otilt:+.1f} dB/oct @ 1 kHz\n"
        except Exception:
            logger.exception("output tilt summary line")
        try:
            psl = str(data.get("plot_smoothing_level", "Psychoacoustic") or "Psychoacoustic").strip()
            psl_display = "DecayCore Reference" if "psy" in psl.lower() else psl
            summary_content += f"Plot smoothing: {psl_display}\n"
        except Exception:
            logger.exception("plot smoothing summary line")

        if enable_afdw:
            summary_content += "FDW mode: Adaptive (A-FDW)\n"
            summary_content += f"FDW base cycles: {fdw_cycles:.2f}  (oct width -> {fdw_oct_width:.3f})\n"
            summary_content += f"FDW min cycles:  {afdw_min:.2f}  (oct width -> {afdw_min_oct_width:.3f})\n"
            summary_content += "Note: A-FDW adapts per frequency/confidence; values above are the configured baseline.\n"
        else:
            summary_content += "FDW mode: Fixed\n"
            summary_content += f"FDW cycles: {fdw_cycles:.2f}  (oct width -> {fdw_oct_width:.3f})\n"

        summary_content += f"TDC: {'ON' if enable_tdc else 'OFF'}\n"
        if enable_tdc:
            summary_content += f"TDC strength: {tdc_strength:.1f}% (base_strength = {tdc_strength / 100.0:.3f})\n"
            tdc_peak = _safe_float(data.get("tdc_peak_reduction_db", float("nan")), float("nan"))
            if math.isfinite(tdc_peak) and tdc_peak > 0.0:
                tdc_peak_hz = _safe_float(data.get("tdc_peak_reduction_hz", float("nan")), float("nan"))
                tdc_events = int(round(_safe_float(data.get("tdc_events_used", 0.0), 0.0)))
                if math.isfinite(tdc_peak_hz) and tdc_peak_hz > 0.0:
                    summary_content += f"TDC applied peak: {tdc_peak:.2f} dB @ {tdc_peak_hz:.1f} Hz ({tdc_events} events)\n"
                else:
                    summary_content += f"TDC applied peak: {tdc_peak:.2f} dB ({tdc_events} events)\n"

        summary_content += f"DF smoothing: {'ON' if df_on else 'OFF'}\n"
        if df_on:
            summary_content += f"DF smoothing sigma: {sigma_bins:.1f} bins -> {sigma_hz:.2f} Hz\n"

        summary_content = _append_main_speaker_xo_hpf_summary(summary_content, data)
        summary_content = _append_bass_integration_summary(summary_content, data)
        summary_content = _append_bass_integration_allpass_auto_summary(summary_content, data)

        try:
            auto_meta = data.get("_auto_mode_meta", None)
            if bool(data.get("camillafir_automatic_mode", False)) and isinstance(auto_meta, dict):
                bm = attach_official_rank_score(auto_meta.get("best_metrics", {}))
                bp = dict(auto_meta.get("best_preset", {}) or {})
                tc = dict(data.get("_auto_target_curve_meta", {}) or {})
                best_rank = official_rank_score(bm)
                summary_content += "\n=== CAMILLAFIR AUTOMATIC MODE ===\n"
                summary_content += (
                    f"Trials: {int(auto_meta.get('trials_ok', 0))}/{int(auto_meta.get('trials_total', 0))} "
                    f"(search grid: {int(auto_meta.get('search_fs', 0))} Hz, {int(auto_meta.get('search_taps', 0))} taps)\n"
                )
                optimizer_backend = str(auto_meta.get("optimizer_backend", "") or "").strip()
                if optimizer_backend:
                    summary_content += f"Optimizer backend: {optimizer_backend}\n"
                tdc_decay_pen = _safe_float(bm.get("tdc_decay_penalty", bm.get("decay_penalty", float("nan"))), float("nan"))
                if math.isfinite(tdc_decay_pen):
                    tdc_sel = _safe_float(bp.get("tdc_strength", data.get("tdc_strength", float("nan"))), float("nan"))
                    tdc_need = _safe_float(bm.get("tdc_decay_need", float("nan")), float("nan"))
                    tdc_opt = _safe_float(bm.get("tdc_decay_optimum_strength", float("nan")), float("nan"))
                    tdc_weak = _safe_float(bm.get("tdc_weak_penalty", float("nan")), float("nan"))
                    tdc_overdamp = _safe_float(bm.get("tdc_overdamping_penalty", float("nan")), float("nan"))
                    tdc_overreach = _safe_float(bm.get("tdc_overreach_penalty", float("nan")), float("nan"))
                    tdc_imp = _safe_float(bm.get("modal_decay_improvement_estimate", float("nan")), float("nan"))
                    tdc_modal_excess = _safe_float(bm.get("modal_decay_excess_s", float("nan")), float("nan"))
                    tdc_modal_peak_excess = _safe_float(bm.get("strongest_modal_decay_excess_s", float("nan")), float("nan"))
                    tdc_mag_cost = _safe_float(bm.get("realized_rms_20_200_db", float("nan")), float("nan"))
                    tdc_modal_hz = _safe_float(bm.get("strongest_modal_decay_hz", float("nan")), float("nan"))
                    tdc_decision = str(bm.get("tdc_decision", "") or "").replace("_", " ").strip()
                    tdc_hint = str(bm.get("tdc_action_hint", "") or "").replace("_", " ").strip()
                    tdc_extreme = bool(bm.get("tdc_extreme_overreach", False))
                    summary_content += "AUTO TDC optimization:\n"
                    if math.isfinite(tdc_sel):
                        summary_content += f"- Selected TDC strength: {tdc_sel:.1f}%\n"
                    summary_content += f"- Decay penalty: {tdc_decay_pen:.2f}\n"
                    if any(math.isfinite(v) for v in (tdc_weak, tdc_overdamp, tdc_overreach)):
                        weak_txt = f"{tdc_weak:.2f}" if math.isfinite(tdc_weak) else "n/a"
                        overdamp_txt = f"{tdc_overdamp:.2f}" if math.isfinite(tdc_overdamp) else "n/a"
                        overreach_txt = f"{tdc_overreach:.2f}" if math.isfinite(tdc_overreach) else "n/a"
                        summary_content += (
                            f"- Penalty split: weak {weak_txt}, overdamping {overdamp_txt}, overreach {overreach_txt}\n"
                        )
                    if math.isfinite(tdc_need):
                        summary_content += f"- Decay need: {tdc_need:.2f}\n"
                    if math.isfinite(tdc_opt):
                        summary_content += f"- Estimated optimum strength: {tdc_opt * 100.0:.0f}%\n"
                    if math.isfinite(tdc_modal_excess):
                        summary_content += f"- Modal decay excess: {tdc_modal_excess:.3f} s\n"
                    if math.isfinite(tdc_imp):
                        summary_content += f"- Modal decay improvement estimate: {tdc_imp * 100.0:.0f}%\n"
                    if math.isfinite(tdc_mag_cost):
                        summary_content += f"- Magnitude tradeoff estimate: {tdc_mag_cost:.2f} dB RMS (20-200 Hz)\n"
                    if math.isfinite(tdc_modal_hz) and tdc_modal_hz > 0.0:
                        summary_content += f"- Strongest modal decay frequency: {tdc_modal_hz:.1f} Hz\n"
                    if math.isfinite(tdc_modal_peak_excess):
                        summary_content += f"- Strongest modal decay excess: {tdc_modal_peak_excess:.3f} s\n"
                    if tdc_decision:
                        summary_content += f"- Decision: {tdc_decision}\n"
                    if tdc_hint:
                        summary_content += f"- Action hint: {tdc_hint}\n"
                    if tdc_extreme:
                        summary_content += "- Safety diagnostic: extreme TDC overreach\n"
                modal_count = int(bm.get("modal_mode_count", 0) or 0)
                modal_events = [dict(item) for item in list(bm.get("modal_events", []) or []) if isinstance(item, dict)]
                if modal_count > 0 or modal_events:
                    summary_content += "Modal intelligence:\n"
                    summary_content += f"- Detected modal events: {modal_count}\n"
                    residual_modal_support = _safe_float(bm.get("residual_peak_modal_support", 0.0), 0.0)
                    residual_modal_penalty = _safe_float(bm.get("residual_peak_modal_penalty", 0.0), 0.0)
                    residual_modal_hz = bm.get("residual_peak_modal_dominant_freq_hz", None)
                    if residual_modal_support > 0.0:
                        hz_txt = ""
                        if residual_modal_hz is not None:
                            hz_val = _safe_float(residual_modal_hz, float("nan"))
                            hz_txt = f" @ {hz_val:.1f} Hz" if math.isfinite(hz_val) else ""
                        summary_content += (
                            f"- Residual peak modal support: {residual_modal_support:.2f}{hz_txt}, "
                            f"penalty add {residual_modal_penalty:.2f}\n"
                        )
                    tdc_modal_reductions = [
                        dict(item)
                        for item in list(bm.get("tdc_modal_reductions", []) or [])
                        if isinstance(item, dict)
                    ]
                    for event in modal_events[:3]:
                        freq = _safe_float(event.get("freq_hz", float("nan")), float("nan"))
                        sev = _safe_float(event.get("severity", 0.0), 0.0)
                        conf = _safe_float(event.get("confidence", 0.0), 0.0)
                        peak = _safe_float(event.get("peak_db", 0.0), 0.0)
                        gd_ms = _safe_float(event.get("gd_excess_ms", 0.0), 0.0)
                        if math.isfinite(freq):
                            summary_content += (
                                f"- Mode {freq:.1f} Hz: severity {sev:.2f}, confidence {conf:.2f}, "
                                f"peak {peak:.1f} dB, GD excess {gd_ms:.0f} ms\n"
                            )
                    for reduction in tdc_modal_reductions[:3]:
                        freq = _safe_float(reduction.get("freq_hz", float("nan")), float("nan"))
                        applied = _safe_float(reduction.get("applied_reduction_db", float("nan")), float("nan"))
                        sev = _safe_float(reduction.get("modal_severity", float("nan")), float("nan"))
                        conf = _safe_float(reduction.get("confidence", float("nan")), float("nan"))
                        if math.isfinite(freq) and math.isfinite(applied) and applied > 0.05:
                            summary_content += (
                                f"- TDC modal reduction: {freq:.1f} Hz by -{applied:.1f} dB "
                                f"(severity {sev:.2f}, confidence {conf:.2f})\n"
                            )
                phase_limit_polish = dict(auto_meta.get("phase_limit_winner_polish", {}) or {})
                if bool(phase_limit_polish.get("applicable", False)):
                    polish_start = _safe_float(phase_limit_polish.get("start_phase_limit_hz", float("nan")), float("nan"))
                    polish_final = _safe_float(phase_limit_polish.get("final_phase_limit_hz", float("nan")), float("nan"))
                    polish_rank_before = _polish_display_rank(phase_limit_polish, "rank_before", "rank_before_official")
                    polish_rank_after = _polish_display_rank(phase_limit_polish, "rank_after", "rank_after_official")
                    polish_tested = [float(v) for v in list(phase_limit_polish.get("tested_phase_limits_hz", []) or [])]
                    polish_tested_txt = ", ".join([f"{float(v):.1f}" for v in polish_tested]) if polish_tested else "n/a"
                    if bool(phase_limit_polish.get("applied", False)):
                        summary_content += (
                            f"Phase-limit winner polish: applied "
                            f"({float(polish_start):.1f} -> {float(polish_final):.1f} Hz, "
                            f"rank {float(polish_rank_before):.3f} -> {float(polish_rank_after):.3f}, "
                            f"tested [{polish_tested_txt}] Hz)\n"
                        )
                    else:
                        summary_content += (
                            f"Phase-limit winner polish: tested, no change "
                            f"(kept {float(polish_final):.1f} Hz, tested [{polish_tested_txt}] Hz)\n"
                        )
                mag_c_min_polish = dict(auto_meta.get("mag_c_min_winner_polish", {}) or {})
                if bool(mag_c_min_polish.get("applicable", False)):
                    polish_start = _safe_float(mag_c_min_polish.get("start_mag_c_min_hz", float("nan")), float("nan"))
                    polish_final = _safe_float(mag_c_min_polish.get("final_mag_c_min_hz", float("nan")), float("nan"))
                    polish_rank_before = _polish_display_rank(mag_c_min_polish, "rank_before", "rank_before_official")
                    polish_rank_after = _polish_display_rank(mag_c_min_polish, "rank_after", "rank_after_official")
                    polish_tested = [float(v) for v in list(mag_c_min_polish.get("tested_mag_c_min_hz", []) or [])]
                    polish_tested_txt = ", ".join([f"{float(v):.1f}" for v in polish_tested]) if polish_tested else "n/a"
                    if bool(mag_c_min_polish.get("applied", False)):
                        summary_content += (
                            f"Mag-c-min winner polish: applied "
                            f"({float(polish_start):.1f} -> {float(polish_final):.1f} Hz, "
                            f"rank {float(polish_rank_before):.3f} -> {float(polish_rank_after):.3f}, "
                            f"tested [{polish_tested_txt}] Hz)\n"
                        )
                    else:
                        summary_content += (
                            f"Mag-c-min winner polish: tested, no change "
                            f"(kept {float(polish_final):.1f} Hz, tested [{polish_tested_txt}] Hz)\n"
                        )
                low_bass_cut_polish = dict(auto_meta.get("low_bass_cut_winner_polish", {}) or {})
                if bool(low_bass_cut_polish.get("applicable", False)):
                    polish_start = _safe_float(low_bass_cut_polish.get("start_low_bass_cut_hz", float("nan")), float("nan"))
                    polish_final = _safe_float(low_bass_cut_polish.get("final_low_bass_cut_hz", float("nan")), float("nan"))
                    polish_rank_before = _polish_display_rank(low_bass_cut_polish, "rank_before", "rank_before_official")
                    polish_rank_after = _polish_display_rank(low_bass_cut_polish, "rank_after", "rank_after_official")
                    polish_tested = [float(v) for v in list(low_bass_cut_polish.get("tested_low_bass_cut_hz", []) or [])]
                    polish_tested_txt = ", ".join([f"{float(v):.1f}" for v in polish_tested]) if polish_tested else "n/a"
                    if bool(low_bass_cut_polish.get("applied", False)):
                        summary_content += (
                            f"Low-bass-cut winner polish: applied "
                            f"({float(polish_start):.1f} -> {float(polish_final):.1f} Hz, "
                            f"rank {float(polish_rank_before):.3f} -> {float(polish_rank_after):.3f}, "
                            f"tested [{polish_tested_txt}] Hz)\n"
                        )
                    else:
                        summary_content += (
                            f"Low-bass-cut winner polish: tested, no change "
                            f"(kept {float(polish_final):.1f} Hz, tested [{polish_tested_txt}] Hz)\n"
                        )
                hpf_polish = dict(auto_meta.get("hpf_winner_polish", {}) or {})
                if bool(hpf_polish.get("applicable", False)):
                    start_enabled = bool(hpf_polish.get("start_enabled", False))
                    start_freq_hz = _safe_float(hpf_polish.get("start_freq_hz", float("nan")), float("nan"))
                    start_slope_db_oct = int(round(_safe_float(hpf_polish.get("start_slope_db_oct", 0.0), 0.0)))
                    final_enabled = bool(hpf_polish.get("final_enabled", False))
                    final_freq_hz = _safe_float(hpf_polish.get("final_freq_hz", float("nan")), float("nan"))
                    final_slope_db_oct = int(round(_safe_float(hpf_polish.get("final_slope_db_oct", 0.0), 0.0)))
                    polish_rank_before = _polish_display_rank(hpf_polish, "rank_before", "rank_before_official")
                    polish_rank_after = _polish_display_rank(hpf_polish, "rank_after", "rank_after_official")
                    tested_candidates = [
                        str(dict(item).get("label", "") or "").strip()
                        for item in list(hpf_polish.get("tested_candidates", []) or [])
                        if isinstance(item, dict)
                    ]
                    tested_candidates = [txt for txt in tested_candidates if txt]
                    tested_txt = ", ".join(tested_candidates) if tested_candidates else "n/a"

                    def _fmt_hpf(enabled_state: bool, freq_hz: float, slope_db_oct: int) -> str:
                        if not bool(enabled_state):
                            return "HPF off"
                        return f"HPF {float(freq_hz):.1f} Hz/{int(slope_db_oct)} dB/oct"

                    if bool(hpf_polish.get("applied", False)):
                        summary_content += (
                            f"HPF winner polish: applied "
                            f"({_fmt_hpf(start_enabled, start_freq_hz, start_slope_db_oct)} -> "
                            f"{_fmt_hpf(final_enabled, final_freq_hz, final_slope_db_oct)}, "
                            f"rank {float(polish_rank_before):.3f} -> {float(polish_rank_after):.3f}, "
                            f"tested [{tested_txt}])\n"
                        )
                    else:
                        summary_content += (
                            f"HPF winner polish: tested, no change "
                            f"(kept {_fmt_hpf(final_enabled, final_freq_hz, final_slope_db_oct)}, tested [{tested_txt}])\n"
                        )
                eps_polish = dict(auto_meta.get("excess_phase_strength_winner_polish", {}) or {})
                if bool(eps_polish.get("applicable", False)):
                    eps_start = _safe_float(eps_polish.get("start_value", float("nan")), float("nan"))
                    eps_final = _safe_float(eps_polish.get("final_value", float("nan")), float("nan"))
                    eps_rank_before = _polish_display_rank(eps_polish, "rank_before", "rank_before_official")
                    eps_rank_after = _polish_display_rank(eps_polish, "rank_after", "rank_after_official")
                    eps_tested = int(eps_polish.get("tested_count", 0) or 0)
                    if bool(eps_polish.get("applied", False)):
                        summary_content += (
                            f"Excess-phase-strength winner polish: applied "
                            f"({float(eps_start):.4f} -> {float(eps_final):.4f}, "
                            f"rank {float(eps_rank_before):.3f} -> {float(eps_rank_after):.3f}, "
                            f"tested {eps_tested})\n"
                        )
                    else:
                        summary_content += (
                            f"Excess-phase-strength winner polish: tested, no change "
                            f"(kept {float(eps_final):.4f}, tested {eps_tested})\n"
                        )
                residual_peak_polish = dict(auto_meta.get("residual_peak_winner_polish", {}) or {})
                if bool(residual_peak_polish.get("applicable", False)):
                    peak_before = _safe_float(residual_peak_polish.get("worst_peak_before_db", float("nan")), float("nan"))
                    peak_after = _safe_float(residual_peak_polish.get("worst_peak_after_db", float("nan")), float("nan"))
                    peak_hz = _safe_float(residual_peak_polish.get("worst_peak_freq_hz", float("nan")), float("nan"))
                    width_oct = _safe_float(residual_peak_polish.get("worst_peak_width_oct", float("nan")), float("nan"))
                    tested_count = int(residual_peak_polish.get("tested_count", 0) or 0)
                    peak_pos = f" @ {float(peak_hz):.1f} Hz" if math.isfinite(float(peak_hz)) else ""
                    width_txt = f", width={float(width_oct):.3f} oct" if math.isfinite(float(width_oct)) else ""
                    if bool(residual_peak_polish.get("applied", False)):
                        summary_content += (
                            f"Residual-peak winner polish: applied "
                            f"({float(peak_before):.2f} -> {float(peak_after):.2f} dB{peak_pos}{width_txt}, "
                            f"tested {tested_count})\n"
                        )
                    elif bool(residual_peak_polish.get("enabled", False)):
                        summary_content += (
                            f"Residual-peak winner polish: tested, no change "
                            f"(peak {float(peak_after):.2f} dB{peak_pos}{width_txt}, tested {tested_count})\n"
                        )
                tdc_strength_polish = dict(auto_meta.get("tdc_strength_winner_polish", {}) or {})
                if bool(tdc_strength_polish.get("applicable", False)):
                    tdc_start = _safe_float(tdc_strength_polish.get("start_strength", float("nan")), float("nan"))
                    tdc_final = _safe_float(tdc_strength_polish.get("final_strength", float("nan")), float("nan"))
                    tdc_hint = _safe_float(tdc_strength_polish.get("optimum_strength_hint", float("nan")), float("nan"))
                    tdc_rank_before = _polish_display_rank(tdc_strength_polish, "rank_before", "rank_before_official")
                    tdc_rank_after = _polish_display_rank(tdc_strength_polish, "rank_after", "rank_after_official")
                    tdc_tested = [float(v) for v in list(tdc_strength_polish.get("tested_strengths", []) or [])]
                    tdc_tested_txt = ", ".join(f"{v:.1f}" for v in tdc_tested) if tdc_tested else "n/a"
                    if bool(tdc_strength_polish.get("applied", False)):
                        summary_content += (
                            f"TDC-strength winner polish: applied "
                            f"({float(tdc_start):.1f}% -> {float(tdc_final):.1f}%, "
                            f"hint={float(tdc_hint):.1f}%, "
                            f"rank {float(tdc_rank_before):.3f} -> {float(tdc_rank_after):.3f}, "
                            f"tested [{tdc_tested_txt}])\n"
                        )
                    else:
                        summary_content += (
                            f"TDC-strength winner polish: tested, no change "
                            f"(kept {float(tdc_start):.1f}%, hint={float(tdc_hint):.1f}%, tested {len(tdc_tested)})\n"
                        )
                summary_content += _runtime_versions_text() + "\n"
                summary_content += _auto_search_space_summary(data) + "\n"
                if tc:
                    tc_method = str(tc.get("selection_method", "fit_rms"))
                    summary_content += (
                        f"Selected target curve: {str(tc.get('selected_hc_mode', 'n/a'))} "
                        f"(fit_rms={float(tc.get('fit_rms_db', 0.0)):.3f} dB, method={tc_method})\n"
                    )
                    if tc_method == "top3x10_trials":
                        ev = list(tc.get("evaluated", []) or [])
                        seed = dict(tc.get("best_preset", {}) or {})
                        top_n = int(tc.get("top_n", 0) or 0)
                        tr_n = int(tc.get("trials_per_curve", 0) or 0)
                        if top_n > 0 and tr_n > 0:
                            summary_content += f"Target selection grid: top-{top_n} x {tr_n} trials\n"
                        if seed:
                            summary_content += (
                                "Target seed preset: "
                                + ", ".join([f"{k}={seed[k]}" for k in sorted(seed.keys())])
                                + "\n"
                            )
                        for i, row in enumerate(ev[:3], start=1):
                            bm_t = attach_official_rank_score(row.get("best_metrics", {}))
                            summary_content += (
                                f"Target #{i}: {str(row.get('hc_mode', 'n/a'))} "
                                f"(best_rank={official_rank_score(bm_t):.3f}, "
                                f"avg_rank={display_rank_score(row.get('avg_rank_score', 0.0)):.3f}, "
                                f"ok={int(row.get('trials_ok', 0))}/{int(row.get('trials_total', 0))})\n"
                            )
                _bass_pen = float(bm.get("bass_integration_penalty", 0.0))
                _bass_pen_txt = (
                    f", bass_pen={_bass_pen:.2f}"
                    if bool(data.get("bass_integration_enable", False)) and _bass_pen > 0.0
                    else ""
                )
                _optimizer_backend = str(auto_meta.get("optimizer_backend", "builtin") or "builtin").strip().lower()
                _exc_pen_txt = (
                    ""
                    if str(_optimizer_backend) == "optuna"
                    else f", exc_pen={float(bm.get('exc_penalty', 0.0)):.2f}"
                )
                _bass_feas_pen = float(bm.get("bass_feasibility_penalty", 0.0))
                _bass_feas_pen_txt = (
                    f", bass_feas_pen={_bass_feas_pen:.2f}"
                    if bool(data.get("bass_integration_enable", False)) and _bass_feas_pen > 0.0
                    else ""
                )
                _bass_boost_db = _safe_float(bm.get("bass_boost_20_200_db", float("nan")), float("nan"))
                _bass_boost_txt = (
                    f"bass_boost={float(_bass_boost_db):.2f} dB, "
                    if math.isfinite(float(_bass_boost_db))
                    else ""
                )
                _hard_gates = list(bm.get("hard_gate_failures", bm.get("hard_gate_reasons", [])) or [])
                _bass_hard_gate = bool(
                    bm.get("bass_integration_hard_gate_failed", False)
                    or "bass_integration_infeasible_hard_gate" in _hard_gates
                )
                if _bass_hard_gate:
                    _bass_gate_reason = str(
                        bm.get(
                            "bass_integration_hard_gate_reason",
                            bm.get("bass_feasibility_reason", ""),
                        )
                        or ""
                    ).strip()
                    _bass_gate_suffix = f": {_bass_gate_reason}" if _bass_gate_reason else ""
                    summary_content += f"Auto-mode safety gate: Bass Integration infeasible{_bass_gate_suffix}\n"
                _display_q = _safe_float(bm.get("display_score_100", float("nan")), float("nan"))
                if not math.isfinite(_display_q):
                    _display_q = calibrated_auto_quality(best_rank, bm)
                summary_content += (
                    f"Automatic quality score: {_display_q:.0f} / 100 "
                    f"({_quality_band(_display_q)})\n"
                )
                summary_content += (
                    f"Debug rank score: {best_rank:.3f}/100 "
                    f"(avg={float(bm.get('avg_score', 0.0)):.3f}, "
                    f"dsp_pen={float(bm.get('dsp_penalty', 0.0)):.2f}"
                    f"{_exc_pen_txt}"
                    f"{_bass_pen_txt}{_bass_feas_pen_txt}, "
                    f"{_bass_boost_txt}"
                    f"max_net_boost={float(bm.get('max_net_boost_db', 0.0)):.2f} dB, "
                    f"events={int(bm.get('events_total', 0))}, "
                    f"event_sev={float(bm.get('events_severity', 0.0)):.2f})\n"
                )
                _tt_pen = _safe_float(bm.get("target_tracking_penalty", 0.0), 0.0)
                _tt_rms_20_200 = _safe_float(bm.get("target_tracking_rms_20_200_db", float("nan")), float("nan"))
                _tt_rms_100_500 = _safe_float(bm.get("target_tracking_rms_100_500_db", float("nan")), float("nan"))
                _tt_max_vals = [
                    _safe_float(bm.get("target_tracking_max_20_200_db", float("nan")), float("nan")),
                    _safe_float(bm.get("target_tracking_max_100_500_db", float("nan")), float("nan")),
                ]
                _tt_max_vals = [float(v) for v in _tt_max_vals if math.isfinite(float(v))]
                _tt_max = max(_tt_max_vals) if _tt_max_vals else float("nan")

                def _fmt_auto_tracking_db(value: float) -> str:
                    v = _safe_float(value, float("nan"))
                    return f"{float(v):.2f} dB" if math.isfinite(float(v)) else "n/a"

                summary_content += (
                    f"Target tracking penalty: {float(_tt_pen):.2f} "
                    f"(worst RMS 20-200={_fmt_auto_tracking_db(_tt_rms_20_200)}, "
                    f"100-500={_fmt_auto_tracking_db(_tt_rms_100_500)}, "
                    f"max={_fmt_auto_tracking_db(_tt_max)})\n"
                )
                _rp_raw = _safe_float(bm.get("worst_residual_peak_raw_db", float("nan")), float("nan"))
                _rp_sev = _safe_float(bm.get("residual_peak_severity", bm.get("worst_residual_peak_db", float("nan"))), float("nan"))
                _rp_hz = _safe_float(bm.get("worst_residual_peak_hz", float("nan")), float("nan"))
                _rp_width_oct = _safe_float(bm.get("worst_residual_peak_width_oct", float("nan")), float("nan"))
                _rp_thr = _safe_float(bm.get("residual_peak_threshold_db", 3.0), 3.0)
                _rp_pen = _safe_float(bm.get("residual_peak_penalty", 0.0), 0.0)
                _rp_count = int(bm.get("residual_peak_count", 0) or 0)
                if math.isfinite(float(_rp_raw)) or _rp_count > 0:
                    _rp_hz_txt = f" @ {float(_rp_hz):.1f} Hz" if math.isfinite(float(_rp_hz)) else ""
                    _rp_width_txt = f", width={float(_rp_width_oct):.3f} oct" if math.isfinite(float(_rp_width_oct)) else ""
                    summary_content += (
                        f"Residual peaks: worst={_fmt_auto_tracking_db(_rp_raw)}{_rp_hz_txt}, "
                        f"severity={float(_rp_sev):.2f}, threshold={float(_rp_thr):.2f} dB, "
                        f"penalty={float(_rp_pen):.2f}, count={_rp_count}{_rp_width_txt}\n"
                    )
                _rp_area = _safe_float(bm.get("residual_peak_area_db_oct", 0.0), 0.0)
                _sharp_pen = _safe_float(bm.get("correction_sharpness_penalty", 0.0), 0.0)
                _slope_max = _safe_float(bm.get("correction_max_abs_slope_db_per_oct", 0.0), 0.0)
                _notches = int(bm.get("narrow_notch_count", 0) or 0)
                _dip_pen = _safe_float(bm.get("dip_fill_risk_penalty", 0.0), 0.0)
                _dip_score = _safe_float(bm.get("dip_fill_risk_score", 0.0), 0.0)
                _lr_pen = _safe_float(bm.get("channel_overfit_penalty", 0.0), 0.0)
                _lr_narrow = int(bm.get("narrow_lr_divergence_count", 0) or 0)
                _polish_applied = bool(residual_peak_polish.get("applied", False))
                _polish_reason = str(
                    residual_peak_polish.get(
                        "accepted_reason",
                        residual_peak_polish.get("rejected_reason", "not_applicable"),
                    )
                    or "not_applicable"
                )
                summary_content += (
                    "Auto-mode anti-overfit / peak-awareness report:\n"
                    f"  Broad residual peaks: count={_rp_count}, worst={_fmt_auto_tracking_db(_rp_raw)}, "
                    f"area={float(_rp_area):.2f} dB*oct\n"
                    f"  Correction sharpness: penalty={float(_sharp_pen):.2f}, "
                    f"max_slope={float(_slope_max):.1f} dB/oct, narrow_notches={_notches}\n"
                    f"  Dip-fill risk: penalty={float(_dip_pen):.2f}, score={float(_dip_score):.2f}\n"
                    f"  L/R narrow divergence: penalty={float(_lr_pen):.2f}, count={_lr_narrow}\n"
                    f"  Winner polish applied: {'yes' if _polish_applied else 'no'} ({_polish_reason})\n"
                )
                summary_content = _append_auto_stereo_policy_summary(summary_content, data, auto_meta)
                exc_seed_hz = _safe_float(
                    auto_meta.get(
                        "auto_exc_seed_freq_hz",
                        data.get("_auto_exc_seed_freq_hz", data.get("_auto_exc_freq_hz", float("nan"))),
                    ),
                    float("nan"),
                )
                exc_final_hz = _safe_float(
                    auto_meta.get(
                        "best_auto_exc_freq_hz",
                        data.get("_auto_exc_freq_hz", data.get("exc_freq", float("nan"))),
                    ),
                    float("nan"),
                )
                seed_ok = (exc_seed_hz == exc_seed_hz) and (abs(exc_seed_hz) != float("inf"))
                final_ok = (exc_final_hz == exc_final_hz) and (abs(exc_final_hz) != float("inf"))
                if str(_optimizer_backend) != "optuna":
                    if seed_ok and final_ok:
                        summary_content += (
                            f"Excursion protection (AUTO): seed {float(exc_seed_hz):.1f} Hz -> "
                            f"final {float(exc_final_hz):.1f} Hz\n"
                        )
                    elif final_ok:
                        summary_content += f"Excursion protection (AUTO): final {float(exc_final_hz):.1f} Hz\n"
                    elif seed_ok:
                        summary_content += f"Excursion protection (AUTO): seed {float(exc_seed_hz):.1f} Hz\n"
                if bp:
                    filter_type = str(bp.get("filter_type", data.get("filter_type", "")) or "").strip().lower()
                    keys = [
                        "fdw_cycles",
                        "tdc_strength",
                        "tdc_max_reduction_db",
                        "tdc_slope_db_per_oct",
                        "reg_strength",
                        "max_slope_db_per_oct",
                        "max_slope_boost_db_per_oct",
                        "max_slope_cut_db_per_oct",
                        "max_boost",
                        "mag_c_min",
                        "mag_c_max",
                        "trans_width",
                        "filter_smooth",
                        "bass_first_mode_max_hz",
                        "conf_pull_max_hz",
                        "low_bass_cut_hz",
                        "synth_tilt_frac",
                    ]
                    if filter_type == "mixed":
                        keys.insert(0, "mixed_freq")
                    elif filter_type in ("linear", "asym", "asymmetric"):
                        keys.insert(0, "phase_limit")
                    picked = [f"{k}={bp[k]}" for k in keys if k in bp]
                    if picked:
                        summary_content += "Best preset: " + ", ".join(picked) + "\n"
        except Exception:
            logger.exception("best preset summary line")
    except Exception as e:
        summary_content += "\n=== DSP EFFECTIVE PARAMS (THIS SAMPLE RATE) ===\n"
        summary_content += f"Could not compute effective params: {type(e).__name__}: {e}\n"

    return summary_content

def _append_realized_phase_limit(summary_content: str, data, l_st, r_st) -> str:
    try:
        summary_content += "\n=== PHASE CORRECTION LIMIT (REALIZED) ===\n"
        cfg_lim = _safe_float((data or {}).get("phase_limit", float("nan")), float("nan"))
        if cfg_lim == cfg_lim and abs(cfg_lim) != float("inf") and cfg_lim > 0.0:
            summary_content += f"Configured phase_limit: {float(cfg_lim):.1f} Hz\n"

        for side, st in [("LEFT", l_st), ("RIGHT", r_st)]:
            st = st if isinstance(st, dict) else {}
            realized_hz = _pick_metric(
                st,
                (
                    "mixed_phase_no_correction_hz",
                    "linear_phase_blend_end_hz",
                    "phase_limit_hz",
                ),
                nonneg=True,
            )
            source = "n/a"
            for key in ("mixed_phase_no_correction_hz", "linear_phase_blend_end_hz", "phase_limit_hz"):
                v = _pick_metric(st, (key,), nonneg=True)
                if v is not None:
                    source = key
                    break

            if realized_hz is not None and float(realized_hz) > 0.0:
                summary_content += f"[{side}] Realized upper limit: {float(realized_hz):.1f} Hz ({source})\n"
            else:
                summary_content += f"[{side}] Realized upper limit: n/a\n"
    except Exception:
        logger.exception("realized phase limit summary section")
    return summary_content


__all__ = ['_append_dsp_effective_params', '_append_realized_phase_limit']


def _load_sibling_symbols() -> None:
    import importlib
    package = __package__
    for module_name in ['runtime', 'bass_integration', 'stereo_policy', 'dsp_effective', 'events']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_load_sibling_symbols()
