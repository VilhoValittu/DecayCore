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
import math
import time
import typing
from datetime import datetime

import numpy as np

from ...application.health_service import compute_health
from ...application.house_curve_service import load_house_curve
from ...application.run_request import RunRequest
from ...application.run_contracts import (
    PreparedRunInput,
    ResolvedRunConfig,
    copy_resolved_data,
    copy_source_ui_data,
)
from ...common.result_postprocess import _irwin_tag
from ...config.decaycore_config import save_config
from ...config.decaycore_pipeline import (
    build_xos_hpf,
    choose_dash_fs,
    choose_target_rates,
    detect_is_wav_source,
    filter_type_short,
    log_df_smoothing_toggle,
)
from ...io.generated_measurement_source import generated_source_matches_upload, parse_generated_source
from ...io.measurements_loader import (
    _try_load_harmonic_sidecar,
    _try_load_rt60_sidecar,
    load_bass_integration_measurements,
    load_measurements_lr,
    load_raw_irs_lr,
    load_raw_ir_sub,
)
from ...io.measurements_txt import parse_measurements_from_path
from ...resources.i8n.decaycore_i18n import t
from ..bridge_types import ProcessRunCallbacks

if typing.TYPE_CHECKING:
    from ..process_run_flow import ProcessRunSupport

logger = logging.getLogger("DecayCore")

def _prepare_target_curve_and_run_context(
    ctx: dict,
    *,
    support: ProcessRunSupport,
    callbacks: "ProcessRunCallbacks | None" = None,
):
    data = ctx.get("resolved_data", ctx["data"])
    ctx["resolved_data"] = data
    ctx["data"] = data
    taps_base = int(ctx["taps_base"])
    f_l = ctx["f_l"]
    m_l = ctx["m_l"]
    p_l = ctx["p_l"]
    f_r = ctx["f_r"]
    m_r = ctx["m_r"]
    p_r = ctx["p_r"]
    _harmonic_freq_hz_l = ctx.get("harmonic_freq_hz_l")
    _harmonic_mags_l = ctx.get("harmonic_magnitudes_db_l")
    _harmonic_freq_hz_r = ctx.get("harmonic_freq_hz_r")
    _harmonic_mags_r = ctx.get("harmonic_magnitudes_db_r")

    hc_f, hc_m, hc_source = load_house_curve(
        data,
        parse_measurements_from_path=parse_measurements_from_path,
    )
    data["hc_source"] = hc_source
    target_curve_name = support.pick_target_curve_label(data)
    target_curve_tag = support.slugify_filename_token(target_curve_name, default="target")
    data["target_curve_name"] = target_curve_name
    data["target_curve_tag"] = target_curve_tag
    logger.info(f"House curve source: {hc_source}")
    logger.info(f"Export target curve tag: {target_curve_tag} (from '{target_curve_name}')")

    bi_recommended_xo_hz = None
    bi_recommended_sub_lpf_hz = None
    bi_rec_xo_l = None
    bi_rec_xo_r = None
    bi_selected_diagnostics = {}
    bi_alignment_recommendation = {}
    bi_allpass_recommendation = {}
    if bool(data.get("bass_integration_enable", False)):
        bundle = ctx.get("bass_integration_bundle", None)
        bi_mode = str(
            data.get("bass_integration_mode", "avr_lfe_main_decomposed") or "avr_lfe_main_decomposed"
        ).strip().lower()
        _mode_u = str(data.get("mode", "BASIC") or "BASIC").strip().upper()
        _auto_active = bool(_mode_u == "AUTO" or data.get("camillafir_automatic_mode", False))
        _is_direct_dac = bool(bi_mode == "direct_dac")

        if _auto_active:
            _status(callbacks, "DecayCore automatic mode: bass integration prepare init")
        _bi_prepare_t0 = time.perf_counter()

        data["bass_integration_sub_combine_mode"] = str(
            data.get("bass_integration_sub_combine_mode", "average") or "average"
        )
        data["bass_integration_sub_delay_ms"] = float(data.get("bass_integration_sub_delay_ms", 0.0) or 0.0)
        data["bass_integration_sub_polarity_invert"] = bool(data.get("bass_integration_sub_polarity_invert", False))
        data["bass_integration_sub_gain_trim_db"] = float(data.get("bass_integration_sub_gain_trim_db", 0.0) or 0.0)
        data["bass_integration_alignment_auto_applied"] = False
        data["bass_integration_alignment_reason"] = ""
        data["bass_integration_allpass_auto_enable"] = bool(
            data.get("bass_integration_allpass_auto_enable", False)
        )
        data["bass_integration_allpass_auto_applied"] = False
        data["bass_integration_allpass_freq_hz"] = float(data.get("bass_integration_allpass_freq_hz", 0.0) or 0.0)
        data["bass_integration_allpass_q"] = float(data.get("bass_integration_allpass_q", 0.707) or 0.707)
        data["bass_integration_allpass_reason"] = ""
        if bi_mode == "direct_dac":
            data["bass_integration_allpass_auto_enable"] = False
            data["bass_integration_allpass_auto_applied"] = False
            data["bass_integration_allpass_freq_hz"] = 0.0
            data["bass_integration_allpass_q"] = 0.707
            data["bass_integration_allpass_reason"] = "Direct DAC rewrite uses polarity/delay/gain only."
            _bi_unified = {
                "applied": False,
                "sub_delay_ms": 0.0,
                "sub_polarity_invert": False,
                "sub_gain_trim_db": 0.0,
                "recommended_hz": float(data.get("avr_crossover_hz", 80.0) or 80.0),
                "recommended_sub_lpf_hz": float(data.get("direct_dac_sub_lpf_hz", data.get("avr_crossover_hz", 80.0)) or data.get("avr_crossover_hz", 80.0) or 80.0),
                "baseline": {},
                "optimized": {},
                "improvement_score": 0.0,
                "reason": "Direct DAC optimization runs after FIR prediction.",
                "allpass_enabled": False,
                "allpass_freq_hz": 0.0,
                "allpass_q": 0.707,
                "allpass_reason": "Direct DAC rewrite uses polarity/delay/gain only.",
            }
            bi_alignment_recommendation = {
                "applied": bool(_bi_unified.get("applied", False)),
                "sub_delay_ms": float(_bi_unified.get("sub_delay_ms", 0.0) or 0.0),
                "sub_polarity_invert": bool(_bi_unified.get("sub_polarity_invert", False)),
                "sub_gain_trim_db": float(_bi_unified.get("sub_gain_trim_db", 0.0) or 0.0),
                "reason": str(_bi_unified.get("reason", "") or ""),
                "improvement_score": float(_bi_unified.get("improvement_score", 0.0) or 0.0),
                "baseline": dict(_bi_unified.get("baseline", {}) or {}),
                "optimized": dict(_bi_unified.get("optimized", {}) or {}),
            }
            data["bass_integration_alignment_auto_applied"] = bool(bi_alignment_recommendation["applied"])
            data["bass_integration_sub_delay_ms"] = float(bi_alignment_recommendation["sub_delay_ms"])
            data["bass_integration_sub_polarity_invert"] = bool(bi_alignment_recommendation["sub_polarity_invert"])
            data["bass_integration_sub_gain_trim_db"] = float(bi_alignment_recommendation["sub_gain_trim_db"])
            data["bass_integration_alignment_reason"] = bi_alignment_recommendation["reason"]
            logger.info(
                "Bass Integration Direct-DAC alignment %s: delay %.2f ms, polarity %s, gain %+0.2f dB",
                "applied" if bool(data.get("bass_integration_alignment_auto_applied", False)) else "kept baseline",
                float(data.get("bass_integration_sub_delay_ms", 0.0) or 0.0),
                "invert" if bool(data.get("bass_integration_sub_polarity_invert", False)) else "normal",
                float(data.get("bass_integration_sub_gain_trim_db", 0.0) or 0.0),
            )
            bi_recommended_xo_hz = _bi_unified.get("recommended_hz", None)
            bi_recommended_sub_lpf_hz = _bi_unified.get("recommended_sub_lpf_hz", None)
            bi_rec_xo_l = None
            bi_rec_xo_r = None
            if bi_recommended_xo_hz is not None:
                data["sub_crossover_hz"] = float(bi_recommended_xo_hz)
                data["avr_crossover_hz"] = float(bi_recommended_xo_hz)
                _sub_lpf_store = float(bi_recommended_sub_lpf_hz) if bi_recommended_sub_lpf_hz is not None else float(bi_recommended_xo_hz) + 20.0
                _sub_lpf_store = max(float(bi_recommended_xo_hz) + 20.0, _sub_lpf_store)
                data["direct_dac_sub_lpf_hz"] = _sub_lpf_store
                if _sub_lpf_store > float(bi_recommended_xo_hz) + 0.5:
                    logger.info(
                        "Bass Integration Direct-DAC auto XO selected: "
                        f"main HPF {float(bi_recommended_xo_hz):.1f} Hz, "
                        f"sub LPF {_sub_lpf_store:.1f} Hz (overlap)"
                    )
                else:
                    logger.info(
                        "Bass Integration Direct-DAC auto XO selected: "
                        f"{float(bi_recommended_xo_hz):.1f} Hz"
                    )
            try:
                _current_main_hpf = float(data.get("sub_crossover_hz", 80.0) or 80.0)
            except Exception:
                _current_main_hpf = 80.0
            if not math.isfinite(_current_main_hpf) or _current_main_hpf <= 0.0:
                _current_main_hpf = 80.0
            try:
                _current_sub_lpf = float(data.get("direct_dac_sub_lpf_hz", _current_main_hpf) or _current_main_hpf)
            except Exception:
                _current_sub_lpf = _current_main_hpf
            if not math.isfinite(_current_sub_lpf) or _current_sub_lpf <= 0.0:
                _current_sub_lpf = _current_main_hpf
            data["direct_dac_sub_lpf_hz"] = float(max(_current_main_hpf + 20.0, _current_sub_lpf))
            bi_allpass_recommendation = {
                "enabled": bool(_bi_unified.get("allpass_enabled", False)),
                "freq_hz": float(_bi_unified.get("allpass_freq_hz", 0.0) or 0.0),
                "q": float(_bi_unified.get("allpass_q", 0.707) or 0.707),
                "improvement_score": 0.0,
                "reason": str(_bi_unified.get("allpass_reason", "") or ""),
                "baseline": {},
                "optimized": {},
            }
            data["bass_integration_allpass_auto_applied"] = bool(bi_allpass_recommendation["enabled"])
            data["bass_integration_allpass_freq_hz"] = float(bi_allpass_recommendation["freq_hz"])
            data["bass_integration_allpass_q"] = float(bi_allpass_recommendation["q"])
            data["bass_integration_allpass_reason"] = bi_allpass_recommendation["reason"]
            if bool(data.get("bass_integration_allpass_auto_applied", False)):
                logger.info(
                    "Bass Integration Direct-DAC auto allpass applied: "
                    f"{float(data.get('bass_integration_allpass_freq_hz', 0.0)):.1f} Hz, "
                    f"Q {float(data.get('bass_integration_allpass_q', 0.707)):.3f}"
                )
            else:
                logger.info(
                    "Bass Integration Direct-DAC auto allpass kept OFF: "
                    f"{str(data.get('bass_integration_allpass_reason', '') or 'No meaningful improvement found.')}"
                )
        else:
            if _auto_active:
                _status(callbacks, "DecayCore automatic mode: bass integration AVR alignment scan")
            bi_alignment_recommendation = _compute_avr_lfe_main_prepare_recommendation(bundle, data)
            data["bass_integration_alignment_auto_applied"] = bool(
                bi_alignment_recommendation.get("applied", False)
            )
            data["bass_integration_sub_delay_ms"] = float(
                bi_alignment_recommendation.get("sub_delay_ms", 0.0) or 0.0
            )
            data["bass_integration_sub_polarity_invert"] = bool(
                bi_alignment_recommendation.get("sub_polarity_invert", False)
            )
            data["bass_integration_sub_gain_trim_db"] = float(
                bi_alignment_recommendation.get("sub_gain_trim_db", 0.0) or 0.0
            )
            data["bass_integration_alignment_reason"] = str(
                bi_alignment_recommendation.get("reason", "") or "Baseline AVR LFE+Main alignment kept."
            )
            bi_recommended_xo_hz = bi_alignment_recommendation.get("recommended_hz", None)
            if bi_recommended_xo_hz is not None:
                try:
                    _avr_rec_hz = float(bi_recommended_xo_hz)
                except Exception:
                    _avr_rec_hz = float("nan")
                if math.isfinite(_avr_rec_hz) and _avr_rec_hz > 0.0:
                    data["avr_crossover_hz"] = float(_avr_rec_hz)
                    data["sub_crossover_hz"] = float(_avr_rec_hz)
            logger.info(
                "Bass Integration AVR LFE+Main alignment %s: delay %.2f ms, polarity %s, gain %+0.2f dB",
                "applied" if bool(data.get("bass_integration_alignment_auto_applied", False)) else "kept baseline",
                float(data.get("bass_integration_sub_delay_ms", 0.0) or 0.0),
                "invert" if bool(data.get("bass_integration_sub_polarity_invert", False)) else "normal",
                float(data.get("bass_integration_sub_gain_trim_db", 0.0) or 0.0),
            )
            data["bass_integration_allpass_reason"] = "Direct DAC only."

        if _auto_active:
            _status(callbacks, "DecayCore automatic mode: bass integration diagnostics refresh")
        bi_selected_diagnostics = _compute_selected_bass_integration_diagnostics(bundle, data)

        # Summary log
        _bi_elapsed_s = time.perf_counter() - _bi_prepare_t0
        try:
            _cache_hits = object.__getattribute__(bundle, "_camillafir_metrics_cache_hits") if bundle is not None else 0
            _cache_misses = object.__getattribute__(bundle, "_camillafir_metrics_cache_misses") if bundle is not None else 0
            _cache_str = f", cache hits {_cache_hits}, misses {_cache_misses}"
        except Exception:
            _cache_str = ""
        _align_str = "alignment applied" if bool(data.get("bass_integration_alignment_auto_applied", False)) else "alignment skipped"
        _xo_val = float(data.get("avr_crossover_hz", 0.0) or 0.0)
        _allpass_str = (
            f"allpass {float(data.get('bass_integration_allpass_freq_hz', 0.0)):.1f} Hz"
            if bool(data.get("bass_integration_allpass_auto_applied", False))
            else "allpass off"
        )
        logger.info(
            "Bass Integration prepare summary: %.1f s%s, %s, xo %.1f Hz, %s",
            _bi_elapsed_s,
            _cache_str,
            _align_str,
            _xo_val,
            _allpass_str,
        )

        if _auto_active:
            _status(callbacks, "DecayCore automatic mode: bass integration prepare done")

    try:
        if hc_f is not None and hc_m is not None:
            logger.info(
                f"HC: n={len(hc_f)} f=[{hc_f[0]:.2f}..{hc_f[-1]:.2f}] "
                f"m=[{float(np.min(hc_m)):.2f}..{float(np.max(hc_m)):.2f}] mean={float(np.mean(hc_m)):.2f}"
            )
    except Exception:
        logger.exception("house curve info log")
    xos, hpf = build_xos_hpf(data)
    try:
        if xos:
            xo_txt = ", ".join(
                [
                    f"{float(x.get('freq')):.1f}Hz/{int(x.get('slope', int(x.get('order', 1)) * 6))}dB/oct"
                    for x in xos
                ]
            )
            logger.info(f"XO (UI->CFG): {xo_txt}")
        else:
            logger.info("XO (UI->CFG): off")
        if isinstance(hpf, dict) and hpf.get("enabled"):
            hf = float(hpf.get("freq", 0.0) or 0.0)
            ho = int(hpf.get("order", 0) or 0)
            logger.info(f"HPF (UI->CFG): {hf:.1f}Hz/{int(ho * 6)}dB/oct")
        else:
            logger.info("HPF (UI->CFG): off")
    except Exception:
        logger.exception("XO/HPF info log")
    log_df_smoothing_toggle(data, logger)

    target_rates = choose_target_rates(data)
    multi_rate_on = bool(data.get("multi_rate_opt"))
    dash_fs = choose_dash_fs(
        target_rates,
        multi_rate_on=multi_rate_on,
        forced_plot_fs_hz=int(support.force_single_plot_fs_hz),
    )
    mode_u = str(data.get("mode", "BASIC") or "BASIC").strip().upper()
    auto_mode_enabled = bool(mode_u == "AUTO" or data.get("camillafir_automatic_mode", False))
    zip_dashboards_on = False

    ts = datetime.now().strftime("%d%m%y_%H%M")
    file_ts = datetime.now().strftime("%H%M_%d%m%y")
    ft_short = filter_type_short(data["filter_type"])
    logger.info(
        f"EXPORT IR (UI): shape={data.get('ir_export_window_shape')}, "
        f"alpha={data.get('ir_export_tukey_alpha')}"
    )

    val_raw = data.get("ir_export_window_mode", None)
    if not isinstance(val_raw, str) or val_raw.strip() == "":
        val_raw = data.get("ir_window_mode", "auto")
    irw_mode = str(val_raw or "auto").strip().lower()
    if irw_mode not in ("auto", "off", "rew_sym", "rew_asym"):
        irw_mode = "auto"
    data["ir_export_window_mode"] = irw_mode
    irw_tag = _irwin_tag(irw_mode)

    is_wav_source = detect_is_wav_source(data)
    data["_is_wav_source"] = bool(is_wav_source)

    measurements = {
        "f_l": np.asarray(f_l, dtype=float),
        "m_l": np.asarray(m_l, dtype=float),
        "p_l": np.asarray(p_l, dtype=float),
        "f_r": np.asarray(f_r, dtype=float),
        "m_r": np.asarray(m_r, dtype=float),
        "p_r": np.asarray(p_r, dtype=float),
        "hc_f": hc_f,
        "hc_m": hc_m,
        "ui_data": data,
        "is_wav_source": bool(is_wav_source),
        "raw_ir_l": ctx.get("raw_ir_l"),
        "raw_ir_fs_l": ctx.get("raw_ir_fs_l", 0),
        "raw_ir_r": ctx.get("raw_ir_r"),
        "raw_ir_fs_r": ctx.get("raw_ir_fs_r", 0),
        "raw_ir_sub": ctx.get("raw_ir_sub"),
        "raw_ir_fs_sub": ctx.get("raw_ir_fs_sub", 0),
        "measured_rt60_l": ctx.get("measured_rt60_l"),
        "measured_rt60_bands_l": ctx.get("measured_rt60_bands_l"),
        "measured_rt60_r": ctx.get("measured_rt60_r"),
        "measured_rt60_bands_r": ctx.get("measured_rt60_bands_r"),
        "harmonic_freq_hz_l": _harmonic_freq_hz_l,
        "harmonic_magnitudes_db_l": _harmonic_mags_l,
        "harmonic_freq_hz_r": _harmonic_freq_hz_r,
        "harmonic_magnitudes_db_r": _harmonic_mags_r,
    }
    if bool(data.get("bass_integration_enable", False)):
        bundle = ctx.get("bass_integration_bundle", None)
        data["_bass_integration_meta"] = {
            "enabled": True,
            "mode": str(data.get("bass_integration_mode", "avr_lfe_main_decomposed") or "avr_lfe_main_decomposed"),
            "profile": str(data.get("bass_integration_profile", "safe") or "safe"),
            "sub_combine_mode": str(data.get("bass_integration_sub_combine_mode", "average") or "average"),
            "avr_crossover_hz": float(data.get("avr_crossover_hz", 80.0) or 80.0),
            "direct_dac_sub_lpf_hz": float(
                data.get(
                    "direct_dac_sub_lpf_hz",
                    data.get("sub_crossover_hz", 80.0),
                )
                or data.get("sub_crossover_hz", 80.0)
                or 80.0
            ),
            "inputs": {
                "l_main": str(data.get("local_path_l_main", "") or "") or str(dict(data.get("file_l_main", {}) or {}).get("filename", "") or ""),
                "r_main": str(data.get("local_path_r_main", "") or "") or str(dict(data.get("file_r_main", {}) or {}).get("filename", "") or ""),
                "l_sub": str(data.get("local_path_l_sub", "") or "") or str(dict(data.get("file_l_sub", {}) or {}).get("filename", "") or ""),
                "r_sub": str(data.get("local_path_r_sub", "") or "") or str(dict(data.get("file_r_sub", {}) or {}).get("filename", "") or ""),
            },
            "diagnostics": dict(bi_selected_diagnostics or getattr(bundle, "diagnostics", {}) or {}),
            "recommended_crossover_hz": bi_recommended_xo_hz,
            "recommended_sub_lpf_hz": bi_recommended_sub_lpf_hz,
            "recommended_crossover_hz_l": bi_rec_xo_l,
            "recommended_crossover_hz_r": bi_rec_xo_r,
            "sub_crossover_manual_override": bool(data.get("sub_crossover_manual_override", False)),
            "alignment": {
                "applied": bool(data.get("bass_integration_alignment_auto_applied", False)),
                "delay_ms": float(data.get("bass_integration_sub_delay_ms", 0.0) or 0.0),
                "polarity_invert": bool(data.get("bass_integration_sub_polarity_invert", False)),
                "gain_trim_db": float(data.get("bass_integration_sub_gain_trim_db", 0.0) or 0.0),
                "reason": str(data.get("bass_integration_alignment_reason", "") or ""),
                "improvement_score": float(bi_alignment_recommendation.get("improvement_score", 0.0) or 0.0),
                "baseline_metrics": dict(bi_alignment_recommendation.get("baseline", {}) or {}),
                "optimized_metrics": dict(bi_alignment_recommendation.get("optimized", {}) or {}),
            },
            "recommended_allpass": {
                "enabled": bool(data.get("bass_integration_allpass_auto_applied", False)),
                "freq_hz": float(data.get("bass_integration_allpass_freq_hz", 0.0) or 0.0),
                "q": float(data.get("bass_integration_allpass_q", 0.707) or 0.707),
                "improvement_score": float(bi_allpass_recommendation.get("improvement_score", 0.0) or 0.0),
                "reason": str(data.get("bass_integration_allpass_reason", "") or ""),
            },
            "allpass_baseline_metrics": dict(bi_allpass_recommendation.get("baseline", {}) or {}),
            "allpass_optimized_metrics": dict(bi_allpass_recommendation.get("optimized", {}) or {}),
        }
        measurements["bass_integration_enabled"] = True
        measurements["bass_integration_bundle"] = bundle
        measurements["bass_integration_mode"] = str(
            data.get("bass_integration_mode", "avr_lfe_main_decomposed") or "avr_lfe_main_decomposed"
        )
        measurements["bass_integration_sub_combine_mode"] = str(
            data.get("bass_integration_sub_combine_mode", "average") or "average"
        )
        measurements["avr_crossover_hz"] = float(data.get("avr_crossover_hz", 80.0) or 80.0)
        measurements["bass_integration_profile"] = str(
            data.get("bass_integration_profile", "safe") or "safe"
        )
        measurements["direct_dac_sub_lpf_hz"] = float(
            data.get(
                "direct_dac_sub_lpf_hz",
                data.get("sub_crossover_hz", 80.0),
            )
            or data.get("sub_crossover_hz", 80.0)
            or 80.0
        )
        measurements["bass_integration_sub_delay_ms"] = float(
            data.get("bass_integration_sub_delay_ms", 0.0) or 0.0
        )
        measurements["bass_integration_sub_polarity_invert"] = bool(
            data.get("bass_integration_sub_polarity_invert", False)
        )
        measurements["bass_integration_sub_gain_trim_db"] = float(
            data.get("bass_integration_sub_gain_trim_db", 0.0) or 0.0
        )
        measurements["bass_integration_alignment_auto_applied"] = bool(
            data.get("bass_integration_alignment_auto_applied", False)
        )
        measurements["bass_integration_allpass_auto_enable"] = bool(
            data.get("bass_integration_allpass_auto_enable", False)
        )
        measurements["bass_integration_allpass_auto_applied"] = bool(
            data.get("bass_integration_allpass_auto_applied", False)
        )
        measurements["bass_integration_allpass_freq_hz"] = float(
            data.get("bass_integration_allpass_freq_hz", 0.0) or 0.0
        )
        measurements["bass_integration_allpass_q"] = float(
            data.get("bass_integration_allpass_q", 0.707) or 0.707
        )

    resolved_config = ResolvedRunConfig(
        source_ui_data=copy_source_ui_data(ctx.get("source_ui_data", {})),
        resolved_data=data,
        measurements=measurements,
        hc_f=hc_f,
        hc_m=hc_m,
        xos=list(xos or []),
        hpf=dict(hpf or {}) if isinstance(hpf, dict) else hpf,
        target_rates=list(target_rates or []),
        dash_fs=int(dash_fs),
        target_curve_tag=str(target_curve_tag),
    )

    ctx.update(
        {
            "hc_f": hc_f,
            "hc_m": hc_m,
            "xos": xos,
            "hpf": hpf,
            "target_curve_tag": target_curve_tag,
            "target_rates": target_rates,
            "dash_fs": dash_fs,
            "auto_mode_enabled": auto_mode_enabled,
            "zip_dashboards_on": zip_dashboards_on,
            "ts": ts,
            "file_ts": file_ts,
            "ft_short": ft_short,
            "irw_tag": irw_tag,
            "measurements": measurements,
            "resolved_config": resolved_config,
            "results_by_fs": [],
            "l_st_f": None,
            "r_st_f": None,
            "sub_ir_f": None,
            "sub_st_f": None,
            "sub_meas_f": {},
            "l_imp_f": None,
            "r_imp_f": None,
            "ui_dashboards": {},
        }
    )


__all__ = ['_prepare_target_curve_and_run_context']


def _load_sibling_symbols() -> None:
    import importlib
    package = __package__
    for module_name in ['bass_diagnostics', 'measurements', 'target_context']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_load_sibling_symbols()
