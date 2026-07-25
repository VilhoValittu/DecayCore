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

from ...application.house_curve_service import load_house_curve
from ...application.run_contracts import (
    ResolvedRunConfig,
    copy_source_ui_data,
)
from ...common.result_postprocess import _irwin_tag
from ...config.legacy_keys import is_auto_mode
from ...config.pipeline_parts import (
    build_xos_hpf,
    choose_dash_fs,
    choose_target_rates,
    detect_is_wav_source,
    filter_type_short,
    log_df_smoothing_toggle,
)
from ...io.measurements_txt import parse_measurements_from_path
from ..bridge_types import ProcessRunCallbacks

from .bass_diagnostics import (
    _compute_direct_dac_prepare_recommendation,
    _compute_selected_bass_integration_diagnostics,
    _status,
)

if typing.TYPE_CHECKING:
    from ..process_run_flow import ProcessRunSupport

logger = logging.getLogger("DecayCore")


def _safe_float(value: object, default: float, *, positive: bool = False) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    if positive and parsed <= 0.0:
        return float(default)
    return parsed


def _safe_float_from_dict(data: dict, key: str, default: float, *, positive: bool = False) -> float:
    return _safe_float(data.get(key, default), default, positive=positive)


def _safe_log_info(active_logger: logging.Logger | None, msg: str, *args) -> None:
    try:
        if active_logger is not None and hasattr(active_logger, "info"):
            active_logger.info(msg, *args)
    except Exception:
        pass


def _build_bi_alignment_recommendation(bi_unified: dict) -> dict:
    sub_delay_ms = _safe_float(bi_unified.get("sub_delay_ms", 0.0), 0.0)
    main_delay_default = max(0.0, -float(sub_delay_ms))
    return {
        "applied": bool(bi_unified.get("applied", False)),
        "sub_delay_ms": float(sub_delay_ms),
        "sub_array_delay_ms": _safe_float(bi_unified.get("sub_array_delay_ms", sub_delay_ms), sub_delay_ms),
        "main_l_delay_ms": _safe_float(bi_unified.get("main_l_delay_ms", main_delay_default), main_delay_default),
        "main_r_delay_ms": _safe_float(bi_unified.get("main_r_delay_ms", main_delay_default), main_delay_default),
        "sub_polarity_invert": bool(bi_unified.get("sub_polarity_invert", False)),
        "sub_gain_trim_db": _safe_float(bi_unified.get("sub_gain_trim_db", 0.0), 0.0),
        "reason": str(bi_unified.get("reason", "") or ""),
        "improvement_score": _safe_float(bi_unified.get("improvement_score", 0.0), 0.0),
        "baseline": dict(bi_unified.get("baseline", {}) or {}),
        "optimized": dict(bi_unified.get("optimized", {}) or {}),
    }


def _build_bi_allpass_recommendation(bi_unified: dict) -> dict:
    return {
        "enabled": bool(bi_unified.get("allpass_enabled", False)),
        "freq_hz": _safe_float(bi_unified.get("allpass_freq_hz", 0.0), 0.0),
        "q": _safe_float(bi_unified.get("allpass_q", 0.707), 0.707, positive=True),
        "improvement_score": 0.0,
        "reason": str(bi_unified.get("allpass_reason", "") or ""),
        "baseline": {},
        "optimized": {},
    }


def _log_house_curve_info(hc_f, hc_m) -> None:
    try:
        if hc_f is None or hc_m is None or len(hc_f) == 0 or len(hc_m) == 0:
            return
        _safe_log_info(
            logger,
            "HC: n=%s f=[%.2f..%.2f] m=[%.2f..%.2f] mean=%.2f",
            len(hc_f),
            float(hc_f[0]),
            float(hc_f[-1]),
            float(np.min(hc_m)),
            float(np.max(hc_m)),
            float(np.mean(hc_m)),
        )
    except (TypeError, ValueError, IndexError):
        pass


def _log_xo_hpf_info(xos, hpf) -> None:
    try:
        if xos:
            xo_txt = ", ".join(
                [
                    f"{float(x.get('freq')):.1f}Hz/{int(x.get('slope', int(x.get('order', 1)) * 6))}dB/oct"
                    for x in xos
                ]
            )
            _safe_log_info(logger, "XO (UI->CFG): %s", xo_txt)
        else:
            _safe_log_info(logger, "XO (UI->CFG): off")
        if isinstance(hpf, dict) and hpf.get("enabled"):
            hf = _safe_float(hpf.get("freq", 0.0), 0.0)
            ho = int(hpf.get("order", 0) or 0)
            _safe_log_info(logger, "HPF (UI->CFG): %.1fHz/%sdB/oct", hf, int(ho * 6))
        else:
            _safe_log_info(logger, "HPF (UI->CFG): off")
    except (TypeError, ValueError, KeyError, IndexError):
        pass


def _prepare_house_curve_context(data: dict, support: ProcessRunSupport) -> tuple[object, object, str, str]:
    hc_f, hc_m, hc_source = load_house_curve(
        data,
        parse_measurements_from_path=parse_measurements_from_path,
    )
    data["hc_source"] = hc_source
    target_curve_name = support.pick_target_curve_label(data)
    target_curve_tag = support.slugify_filename_token(target_curve_name, default="target")
    data["target_curve_name"] = target_curve_name
    data["target_curve_tag"] = target_curve_tag
    _safe_log_info(logger, "House curve source: %s", hc_source)
    _safe_log_info(
        logger,
        "Export target curve tag: %s (from %r)",
        target_curve_tag,
        target_curve_name,
    )
    return hc_f, hc_m, hc_source, target_curve_tag


def _prepare_bass_integration_state(
    *,
    ctx: dict,
    data: dict,
    callbacks: ProcessRunCallbacks | None,
) -> dict:
    state = {
        "bi_recommended_xo_hz": None,
        "bi_recommended_sub_lpf_hz": None,
        "bi_rec_xo_l": None,
        "bi_rec_xo_r": None,
        "bi_selected_diagnostics": {},
        "bi_alignment_recommendation": {},
        "bi_allpass_recommendation": {},
    }
    if bool(data.get("bass_integration_enable", False)):
        state.update(
            _prepare_target_curve_bass_integration_context(
                ctx=ctx,
                data=data,
                callbacks=callbacks,
            )
        )
    return state


def _prepare_xo_hpf(data: dict) -> tuple[object, object]:
    xos, hpf = build_xos_hpf(data)
    _log_xo_hpf_info(xos, hpf)
    log_df_smoothing_toggle(data, logger)
    return xos, hpf


def _prepare_export_parameters(data: dict, support: ProcessRunSupport) -> dict:
    target_rates = choose_target_rates(data)
    multi_rate_on = bool(data.get("multi_rate_opt"))
    dash_fs = choose_dash_fs(
        target_rates,
        multi_rate_on=multi_rate_on,
        forced_plot_fs_hz=int(support.force_single_plot_fs_hz),
    )
    mode_u = str(data.get("mode", "BASIC") or "BASIC").strip().upper()
    auto_mode_enabled = is_auto_mode(data, mode_u)
    ts = datetime.now().strftime("%d%m%y_%H%M")
    file_ts = datetime.now().strftime("%H%M_%d%m%y")
    ft_short = filter_type_short(data["filter_type"])
    _safe_log_info(
        logger,
        "EXPORT IR (UI): shape=%s, alpha=%s",
        data.get("ir_export_window_shape"),
        data.get("ir_export_tukey_alpha"),
    )

    val_raw = data.get("ir_export_window_mode")
    if not isinstance(val_raw, str) or val_raw.strip() == "":
        val_raw = data.get("ir_window_mode", "auto")
    irw_mode = str(val_raw or "auto").strip().lower()
    if irw_mode not in ("auto", "off", "rew_sym", "rew_asym"):
        irw_mode = "auto"
    data["ir_export_window_mode"] = irw_mode
    return {
        "target_rates": target_rates,
        "dash_fs": dash_fs,
        "auto_mode_enabled": auto_mode_enabled,
        "ts": ts,
        "file_ts": file_ts,
        "ft_short": ft_short,
        "irw_tag": _irwin_tag(irw_mode),
    }


def _build_bass_integration_metadata_unified(
    *,
    data: dict,
    bi_state: dict,
    bundle_diagnostics: dict,
) -> tuple[dict, dict]:
    alignment = {
        "applied": bool(data.get("bass_integration_alignment_auto_applied", False)),
        "delay_ms": _safe_float_from_dict(data, "bass_integration_sub_delay_ms", 0.0),
        "sub_array_delay_ms": _safe_float(
            data.get("bass_integration_sub_array_delay_ms", bundle_diagnostics.get("sub_array_delay_ms", 0.0)),
            0.0,
        ),
        "sub1_delay_ms": _safe_float(
            data.get("bass_integration_sub1_delay_ms", bundle_diagnostics.get("sub1_delay_ms", 0.0)),
            0.0,
        ),
        "sub2_delay_ms": _safe_float(
            data.get("bass_integration_sub2_delay_ms", bundle_diagnostics.get("sub2_delay_ms", 0.0)),
            0.0,
        ),
        "main_l_delay_ms": _safe_float_from_dict(data, "bass_integration_main_l_delay_ms", 0.0),
        "main_r_delay_ms": _safe_float_from_dict(data, "bass_integration_main_r_delay_ms", 0.0),
        "polarity_invert": bool(data.get("bass_integration_sub_polarity_invert", False)),
        "gain_trim_db": _safe_float_from_dict(data, "bass_integration_sub_gain_trim_db", 0.0),
        "reason": str(data.get("bass_integration_alignment_reason", "") or ""),
        "improvement_score": _safe_float(
            bi_state["bi_alignment_recommendation"].get("improvement_score", 0.0),
            0.0,
        ),
        "baseline_metrics": dict(bi_state["bi_alignment_recommendation"].get("baseline", {}) or {}),
        "optimized_metrics": dict(bi_state["bi_alignment_recommendation"].get("optimized", {}) or {}),
    }
    recommended_allpass = {
        "enabled": bool(data.get("bass_integration_allpass_auto_applied", False)),
        "freq_hz": _safe_float_from_dict(data, "bass_integration_allpass_freq_hz", 0.0),
        "q": _safe_float_from_dict(data, "bass_integration_allpass_q", 0.707, positive=True),
        "improvement_score": _safe_float(
            bi_state["bi_allpass_recommendation"].get("improvement_score", 0.0),
            0.0,
        ),
        "reason": str(data.get("bass_integration_allpass_reason", "") or ""),
    }
    direct_dac_sub_lpf_hz = _safe_float_from_dict(
        data,
        "direct_dac_sub_lpf_hz",
        _safe_float_from_dict(data, "sub_crossover_hz", 80.0, positive=True),
        positive=True,
    )
    meta_dict = {
        "enabled": True,
        "mode": "direct_dac",
        "profile": str(data.get("bass_integration_profile", "safe") or "safe"),
        "sub_combine_mode": str(
            bundle_diagnostics.get(
                "sub_combine_mode",
                data.get("bass_integration_sub_combine_mode", "average"),
            )
            or "average"
        ),
        "avr_crossover_hz": _safe_float_from_dict(data, "avr_crossover_hz", 80.0, positive=True),
        "direct_dac_sub_lpf_hz": direct_dac_sub_lpf_hz,
        "inputs": {
            "l_main": str(data.get("local_path_l_main", "") or "")
            or str(dict(data.get("file_l_main", {}) or {}).get("filename", "") or ""),
            "r_main": str(data.get("local_path_r_main", "") or "")
            or str(dict(data.get("file_r_main", {}) or {}).get("filename", "") or ""),
            "l_sub": str(data.get("local_path_l_sub", "") or "")
            or str(dict(data.get("file_l_sub", {}) or {}).get("filename", "") or ""),
            "r_sub": str(data.get("local_path_r_sub", "") or "")
            or str(dict(data.get("file_r_sub", {}) or {}).get("filename", "") or ""),
        },
        "diagnostics": {
            **bundle_diagnostics,
            **dict(bi_state["bi_selected_diagnostics"] or {}),
            "lf_rolloff": dict(bi_state.get("bi_lf_rolloff", {}) or {}),
        },
        "recommended_crossover_hz": bi_state["bi_recommended_xo_hz"],
        "recommended_sub_lpf_hz": bi_state["bi_recommended_sub_lpf_hz"],
        "recommended_crossover_hz_l": bi_state["bi_rec_xo_l"],
        "recommended_crossover_hz_r": bi_state["bi_rec_xo_r"],
        "sub_crossover_manual_override": bool(data.get("sub_crossover_manual_override", False)),
        "alignment": alignment,
        "recommended_allpass": recommended_allpass,
        "allpass_baseline_metrics": dict(bi_state["bi_allpass_recommendation"].get("baseline", {}) or {}),
        "allpass_optimized_metrics": dict(bi_state["bi_allpass_recommendation"].get("optimized", {}) or {}),
    }
    measurements_updates = {
        "bass_integration_enabled": True,
        "bass_integration_mode": "direct_dac",
        "bass_integration_sub_combine_mode": meta_dict["sub_combine_mode"],
        "avr_crossover_hz": meta_dict["avr_crossover_hz"],
        "bass_integration_profile": meta_dict["profile"],
        "direct_dac_sub_lpf_hz": meta_dict["direct_dac_sub_lpf_hz"],
        "bass_integration_sub_delay_ms": alignment["delay_ms"],
        "bass_integration_sub_array_delay_ms": alignment["sub_array_delay_ms"],
        "bass_integration_sub1_delay_ms": alignment["sub1_delay_ms"],
        "bass_integration_sub2_delay_ms": alignment["sub2_delay_ms"],
        "bass_integration_main_l_delay_ms": alignment["main_l_delay_ms"],
        "bass_integration_main_r_delay_ms": alignment["main_r_delay_ms"],
        "bass_integration_sub_polarity_invert": alignment["polarity_invert"],
        "bass_integration_sub_gain_trim_db": alignment["gain_trim_db"],
        "bass_integration_alignment_auto_applied": alignment["applied"],
        "bass_integration_allpass_auto_enable": bool(data.get("bass_integration_allpass_auto_enable", False)),
        "bass_integration_allpass_auto_applied": recommended_allpass["enabled"],
        "bass_integration_allpass_freq_hz": recommended_allpass["freq_hz"],
        "bass_integration_allpass_q": recommended_allpass["q"],
    }
    return meta_dict, measurements_updates


def _build_measurements_dict(
    *,
    ctx: dict,
    data: dict,
    hc_f,
    hc_m,
    bi_state: dict,
) -> dict:
    is_wav_source = detect_is_wav_source(data)
    data["_is_wav_source"] = bool(is_wav_source)
    measurements = {
        "f_l": np.asarray(ctx["f_l"], dtype=float),
        "m_l": np.asarray(ctx["m_l"], dtype=float),
        "p_l": np.asarray(ctx["p_l"], dtype=float),
        "f_r": np.asarray(ctx["f_r"], dtype=float),
        "m_r": np.asarray(ctx["m_r"], dtype=float),
        "p_r": np.asarray(ctx["p_r"], dtype=float),
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
        "harmonic_freq_hz_l": ctx.get("harmonic_freq_hz_l"),
        "harmonic_magnitudes_db_l": ctx.get("harmonic_magnitudes_db_l"),
        "harmonic_freq_hz_r": ctx.get("harmonic_freq_hz_r"),
        "harmonic_magnitudes_db_r": ctx.get("harmonic_magnitudes_db_r"),
    }
    if bool(data.get("bass_integration_enable", False)):
        bundle = ctx.get("bass_integration_bundle")
        bundle_diagnostics = dict(getattr(bundle, "diagnostics", {}) or {})
        meta_dict, measurements_updates = _build_bass_integration_metadata_unified(
            data=data,
            bi_state=bi_state,
            bundle_diagnostics=bundle_diagnostics,
        )
        data["_bass_integration_meta"] = meta_dict
        measurements["bass_integration_bundle"] = bundle
        measurements.update(measurements_updates)
    return measurements


def _finalize_target_context(
    *,
    ctx: dict,
    data: dict,
    hc_f,
    hc_m,
    xos,
    hpf,
    target_curve_tag: str,
    export_state: dict,
    measurements: dict,
) -> None:
    resolved_config = ResolvedRunConfig(
        source_ui_data=copy_source_ui_data(ctx.get("source_ui_data", {})),
        resolved_data=data,
        measurements=measurements,
        hc_f=hc_f,
        hc_m=hc_m,
        xos=list(xos or []),
        hpf=dict(hpf or {}) if isinstance(hpf, dict) else hpf,
        target_rates=list(export_state["target_rates"] or []),
        dash_fs=int(export_state["dash_fs"]),
        target_curve_tag=str(target_curve_tag),
    )

    ctx.update(
        {
            "hc_f": hc_f,
            "hc_m": hc_m,
            "xos": xos,
            "hpf": hpf,
            "target_curve_tag": target_curve_tag,
            "target_rates": export_state["target_rates"],
            "dash_fs": export_state["dash_fs"],
            "auto_mode_enabled": export_state["auto_mode_enabled"],
            "ts": export_state["ts"],
            "file_ts": export_state["file_ts"],
            "ft_short": export_state["ft_short"],
            "irw_tag": export_state["irw_tag"],
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
        }
    )


def _prepare_target_curve_bass_integration_context(
    *,
    ctx: dict,
    data: dict,
    callbacks: ProcessRunCallbacks | None,
) -> dict:
    bundle = ctx.get("bass_integration_bundle")
    bi_mode = "direct_dac"
    data["bass_integration_mode"] = bi_mode
    _mode_u = str(data.get("mode", "BASIC") or "BASIC").strip().upper()
    _auto_active = is_auto_mode(data, _mode_u)

    if _auto_active:
        _status(callbacks, "DecayCore automatic mode: bass integration prepare init")
    _bi_prepare_t0 = time.perf_counter()
    requested_allpass_auto_enable = bool(data.get("bass_integration_allpass_auto_enable", False))

    data["bass_integration_sub_combine_mode"] = str(
        data.get("bass_integration_sub_combine_mode", "average") or "average"
    )
    data["bass_integration_sub_delay_ms"] = float(data.get("bass_integration_sub_delay_ms", 0.0) or 0.0)
    data["bass_integration_sub_polarity_invert"] = bool(data.get("bass_integration_sub_polarity_invert", False))
    data["bass_integration_sub_gain_trim_db"] = float(data.get("bass_integration_sub_gain_trim_db", 0.0) or 0.0)
    data["bass_integration_alignment_auto_applied"] = False
    data["bass_integration_alignment_reason"] = ""
    data["bass_integration_allpass_auto_enable"] = bool(requested_allpass_auto_enable)
    data["bass_integration_allpass_auto_applied"] = False
    data["bass_integration_allpass_freq_hz"] = 0.0
    data["bass_integration_allpass_q"] = 0.707
    data["bass_integration_allpass_reason"] = "Direct DAC rewrite uses polarity/delay/gain only."
    _bi_unified = dict(_compute_direct_dac_prepare_recommendation(bundle, data, callbacks=callbacks) or {})
    if not _bi_unified:
        _bi_unified = {
            "applied": False,
            "sub_delay_ms": 0.0,
            "sub_array_delay_ms": 0.0,
            "main_l_delay_ms": 0.0,
            "main_r_delay_ms": 0.0,
            "sub_polarity_invert": False,
            "sub_gain_trim_db": 0.0,
            "recommended_hz": _safe_float_from_dict(data, "avr_crossover_hz", 80.0, positive=True),
            "recommended_sub_lpf_hz": _safe_float(
                data.get(
                    "direct_dac_sub_lpf_hz",
                    _safe_float_from_dict(data, "avr_crossover_hz", 80.0, positive=True),
                ),
                _safe_float_from_dict(data, "avr_crossover_hz", 80.0, positive=True),
                positive=True,
            ),
            "baseline": {},
            "optimized": {},
            "improvement_score": 0.0,
            "reason": "Direct DAC optimization runs after FIR prediction.",
            "allpass_enabled": False,
            "allpass_freq_hz": 0.0,
            "allpass_q": 0.707,
            "allpass_reason": "Direct DAC rewrite uses polarity/delay/gain only.",
        }
    bi_alignment_recommendation = _build_bi_alignment_recommendation(_bi_unified)
    data["bass_integration_alignment_auto_applied"] = bool(bi_alignment_recommendation["applied"])
    data["bass_integration_sub_delay_ms"] = _safe_float(
        bi_alignment_recommendation["sub_delay_ms"],
        0.0,
    )
    data["bass_integration_sub_array_delay_ms"] = _safe_float(
        bi_alignment_recommendation["sub_array_delay_ms"],
        0.0,
    )
    data["bass_integration_main_l_delay_ms"] = _safe_float(
        bi_alignment_recommendation["main_l_delay_ms"],
        0.0,
    )
    data["bass_integration_main_r_delay_ms"] = _safe_float(
        bi_alignment_recommendation["main_r_delay_ms"],
        0.0,
    )
    data["bass_integration_sub_polarity_invert"] = bool(bi_alignment_recommendation["sub_polarity_invert"])
    data["bass_integration_sub_gain_trim_db"] = _safe_float(
        bi_alignment_recommendation["sub_gain_trim_db"],
        0.0,
    )
    data["bass_integration_alignment_reason"] = bi_alignment_recommendation["reason"]
    logger.info(
        "Bass Integration Direct-DAC alignment %s: delay %.2f ms, polarity %s, gain %+0.2f dB",
        "applied" if bool(data.get("bass_integration_alignment_auto_applied", False)) else "kept baseline",
        float(data.get("bass_integration_sub_delay_ms", 0.0) or 0.0),
        "invert" if bool(data.get("bass_integration_sub_polarity_invert", False)) else "normal",
        float(data.get("bass_integration_sub_gain_trim_db", 0.0) or 0.0),
    )
    bi_recommended_xo_hz = _bi_unified.get("recommended_hz")
    bi_recommended_sub_lpf_hz = _bi_unified.get("recommended_sub_lpf_hz")
    bi_rec_xo_l = None
    bi_rec_xo_r = None
    if bi_recommended_xo_hz is not None:
        data["sub_crossover_hz"] = float(bi_recommended_xo_hz)
        data["avr_crossover_hz"] = float(bi_recommended_xo_hz)
        _sub_lpf_store = (
            float(bi_recommended_sub_lpf_hz)
            if bi_recommended_sub_lpf_hz is not None
            else float(bi_recommended_xo_hz)
        )
        _sub_lpf_store = max(float(bi_recommended_xo_hz), _sub_lpf_store)
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
    _current_main_hpf = _safe_float_from_dict(data, "sub_crossover_hz", 80.0, positive=True)
    _current_sub_lpf = _safe_float_from_dict(
        data,
        "direct_dac_sub_lpf_hz",
        _current_main_hpf,
        positive=True,
    )
    data["direct_dac_sub_lpf_hz"] = float(max(_current_main_hpf, _current_sub_lpf))
    bi_allpass_recommendation = _build_bi_allpass_recommendation(_bi_unified)
    data["bass_integration_allpass_auto_applied"] = bool(bi_allpass_recommendation["enabled"])
    data["bass_integration_allpass_freq_hz"] = _safe_float(bi_allpass_recommendation["freq_hz"], 0.0)
    data["bass_integration_allpass_q"] = _safe_float(
        bi_allpass_recommendation["q"],
        0.707,
        positive=True,
    )
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
            f"{data.get('bass_integration_allpass_reason', '') or 'No meaningful improvement found.'!s}"
        )

    bi_selected_diagnostics = _refresh_target_curve_bass_integration_diagnostics(
        callbacks=callbacks,
        auto_active=_auto_active,
        bundle=bundle,
        data=data,
    )

    _bi_elapsed_s = time.perf_counter() - _bi_prepare_t0
    _log_target_curve_bass_integration_summary(bundle, data, _bi_elapsed_s)

    if _auto_active:
        _status(callbacks, "DecayCore automatic mode: bass integration prepare done")

    return {
        "bi_recommended_xo_hz": bi_recommended_xo_hz,
        "bi_recommended_sub_lpf_hz": bi_recommended_sub_lpf_hz,
        "bi_rec_xo_l": bi_rec_xo_l,
        "bi_rec_xo_r": bi_rec_xo_r,
        "bi_selected_diagnostics": bi_selected_diagnostics,
        "bi_alignment_recommendation": bi_alignment_recommendation,
        "bi_allpass_recommendation": bi_allpass_recommendation,
        "bi_lf_rolloff": dict(_bi_unified.get("lf_rolloff", {}) or {}),
    }


def _log_target_curve_bass_integration_summary(bundle, data: dict, elapsed_s: float) -> None:
    try:
        _cache_hits = object.__getattribute__(bundle, "_camillafir_metrics_cache_hits") if bundle is not None else 0
        _cache_misses = object.__getattribute__(bundle, "_camillafir_metrics_cache_misses") if bundle is not None else 0
        _cache_str = f", cache hits {_cache_hits}, misses {_cache_misses}"
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
        float(elapsed_s),
        _cache_str,
        _align_str,
        _xo_val,
        _allpass_str,
    )


def _refresh_target_curve_bass_integration_diagnostics(
    *,
    callbacks: ProcessRunCallbacks | None,
    auto_active: bool,
    bundle,
    data: dict,
) -> dict:
    if auto_active:
        _status(callbacks, "DecayCore automatic mode: bass integration diagnostics refresh")
    return _compute_selected_bass_integration_diagnostics(bundle, data)


def _prepare_target_curve_and_run_context(
    ctx: dict,
    *,
    support: ProcessRunSupport,
    callbacks: ProcessRunCallbacks | None = None,
):
    data = ctx.get("resolved_data", ctx["data"])
    ctx["resolved_data"] = data
    ctx["data"] = data
    hc_f, hc_m, _hc_source, target_curve_tag = _prepare_house_curve_context(data, support)
    bi_state = _prepare_bass_integration_state(
        ctx=ctx,
        data=data,
        callbacks=callbacks,
    )
    _log_house_curve_info(hc_f, hc_m)
    xos, hpf = _prepare_xo_hpf(data)
    export_state = _prepare_export_parameters(data, support)
    measurements = _build_measurements_dict(
        ctx=ctx,
        data=data,
        hc_f=hc_f,
        hc_m=hc_m,
        bi_state=bi_state,
    )
    _finalize_target_context(
        ctx=ctx,
        data=data,
        hc_f=hc_f,
        hc_m=hc_m,
        xos=xos,
        hpf=hpf,
        target_curve_tag=target_curve_tag,
        export_state=export_state,
        measurements=measurements,
    )


__all__ = [
    '_build_bass_integration_metadata_unified',
    '_prepare_target_curve_and_run_context',
    '_prepare_target_curve_bass_integration_context',
    '_safe_float_from_dict',
    'build_xos_hpf',
    'choose_dash_fs',
    'choose_target_rates',
    'detect_is_wav_source',
    'filter_type_short',
    'load_house_curve',
    'log_df_smoothing_toggle',
]
