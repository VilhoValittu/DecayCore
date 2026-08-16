# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

import json
import logging
import math
import os
from typing import Any

from . import decaycore_plot as plots
from .export_scoring import _append_export_ranking
from .export_summary import (
    _append_acoustic_events,
    _append_dsp_effective_params,
    _append_export_decision_summary,
    _append_lr_difference_summary,
    _append_realized_phase_limit,
    _append_leveling_summary,
)
from ..app_paths import program_version_token
from ..common.result_postprocess import _irwin_tag
from ..config.pipeline_parts import build_xos_hpf
from ..config.decaycore_convolver_configs import generate_hlc_config, generate_raspberry_yaml
from ..config.legacy_keys import CAMILLAFIR_AUTO_MODE
from ..config.results import FilterResult
from ..dsp.hpf_policy import hpf_settings_should_use_iir
from ..common.auto_reporting import attach_official_rank_score, official_rank_score
from ..ui_i18n import layout_legacy_name

logger = logging.getLogger("DecayCore")

_PROGRAM_NAME = "DecayCore"
_RECOVERABLE_JSON_SAFE_EXCEPTIONS = (
    TypeError,
    ValueError,
    AttributeError,
    KeyError,
    IndexError,
    OverflowError,
    ImportError,
    ModuleNotFoundError,
    RecursionError,
)


def _json_safe_numpy(obj):
    try:
        import numpy as _np
    except _RECOVERABLE_JSON_SAFE_EXCEPTIONS:
        return None
    try:
        if isinstance(obj, _np.generic):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except _RECOVERABLE_JSON_SAFE_EXCEPTIONS:
        logger.exception("numpy json conversion")
    return None


def _json_safe_dict(obj, *, depth: int, max_depth: int):
    out = {}
    for key, value in obj.items():
        try:
            key_str = str(key)
        except _RECOVERABLE_JSON_SAFE_EXCEPTIONS:
            key_str = "key"
        out[key_str] = _json_safe(value, _depth=depth + 1, _max_depth=max_depth)
    return out


def _json_safe(obj, *, _depth=0, _max_depth=12):
    try:
        if _depth > _max_depth:
            return str(obj)
        if obj is None or isinstance(obj, (str, bool, int, float)):
            return obj
        numpy_value = _json_safe_numpy(obj)
        if numpy_value is not None:
            return numpy_value
        if isinstance(obj, dict):
            return _json_safe_dict(obj, depth=_depth, max_depth=_max_depth)
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v, _depth=_depth + 1, _max_depth=_max_depth) for v in obj]
        if isinstance(obj, (bytes, bytearray)):
            try:
                return obj.decode("utf-8", errors="replace")
            except _RECOVERABLE_JSON_SAFE_EXCEPTIONS:
                return str(obj)
        return str(obj)
    except _RECOVERABLE_JSON_SAFE_EXCEPTIONS:
        return str(obj)


def _build_diagnostics_dict(data, fs_v, l_st, r_st):
    from ..version import VERSION as _version  # noqa: PLC0415

    def _leveling_block(st):
        if not isinstance(st, dict):
            return {}
        win = st.get("smart_scan_range", None)
        try:
            if isinstance(win, (list, tuple)) and len(win) >= 2:
                win = [float(win[0]), float(win[1])]
            else:
                win = None
        except (
            TypeError,
            ValueError,
            AttributeError,
            KeyError,
            IndexError,
            OverflowError,
            ImportError,
            ModuleNotFoundError,
            RecursionError,
        ):
            win = None
        return {
            "method": st.get("offset_method", None),
            "window_hz": win,
            "offset_db": st.get("offset_db", None),
            "eff_target_db": st.get("eff_target_db", None),
            "tilt_slope_db_per_oct": st.get("tilt_slope_db_per_oct", None),
            "avg_confidence_pct": st.get("avg_confidence", None),
        }

    return {
        "schema_version": 1,
        "meta": {
            "program": _PROGRAM_NAME,
            "version": str(data.get("program_version", _version) or _version),
            "fs_hz": int(fs_v),
            "taps": int(float(data.get("taps", 0) or 0)),
            "filter_type": str(data.get("filter_type", "") or ""),
            "layout": layout_legacy_name(data.get("layout", "mono")),
            "multi_rate": bool(data.get("multi_rate_opt", False)),
            "ir_export_window_mode": str(data.get("ir_export_window_mode", "") or ""),
            "ir_export_window_tag": str(_irwin_tag(data.get("ir_export_window_mode"))),
        },
        "settings": _json_safe(data),
        "leveling": {
            "stereo_link": bool(data.get("stereo_link", False)),
            "left": _leveling_block(l_st),
            "right": _leveling_block(r_st),
        },
        "left": _json_safe(l_st),
        "right": _json_safe(r_st),
    }


TEST_MODE = os.environ.get("DECAYCORE_TEST", os.environ.get("CAMILLAFIR_TEST", "0")) == "1"


def _export_version_tag(data: dict | None, *, program_version: str | None = None) -> str:
    raw_version = program_version
    if raw_version is None:
        try:
            raw_version = str((data or {}).get("program_version", "") or "").strip()
        except (
            TypeError,
            ValueError,
            AttributeError,
            KeyError,
            IndexError,
            OverflowError,
            ImportError,
            ModuleNotFoundError,
            RecursionError,
        ):
            raw_version = ""
    return str(program_version_token(raw_version, default="v0"))


def _export_winner_rank_score(data: dict | None) -> float:
    try:
        auto_used = bool((data or {}).get(CAMILLAFIR_AUTO_MODE, False))
    except (
        TypeError,
        ValueError,
        AttributeError,
        KeyError,
        IndexError,
        OverflowError,
        ImportError,
        ModuleNotFoundError,
        RecursionError,
    ):
        auto_used = False
    if not auto_used:
        return float("nan")

    try:
        auto_meta = dict((data or {}).get("_auto_mode_meta", {}) or {})
        best_metrics = attach_official_rank_score(auto_meta.get("best_metrics", {}))
    except (
        TypeError,
        ValueError,
        AttributeError,
        KeyError,
        IndexError,
        OverflowError,
        ImportError,
        ModuleNotFoundError,
        RecursionError,
    ):
        return float("nan")

    return float(official_rank_score(best_metrics))


def _camilladsp_yaml_name(
    *,
    data: dict | None,
    ft_short: str,
    irw_tag: str,
    fs_v: int | None = None,
) -> str:
    parts = ["camilladsp", str(ft_short)]
    if fs_v is not None:
        parts.append(f"{int(fs_v)}Hz")
    parts.append(str(irw_tag))
    parts.append(_export_version_tag(data))
    return "_".join(parts) + ".yml"


def _extract_result_payload(result: FilterResult) -> tuple[Any, ...]:
    meas = dict(getattr(result, "measurements", {}) or {})
    return (
        int(result.fs),
        meas.get("f_l"),
        meas.get("m_l"),
        meas.get("p_l"),
        result.l_ir,
        result.l_st,
        meas.get("f_r"),
        meas.get("m_r"),
        meas.get("p_r"),
        result.r_ir,
        result.r_st,
    )


def _validated_device_format(data: dict | None) -> str:
    fmt = str((data or {}).get("device_audio_format", "S32_LE") or "S32_LE").upper()
    return fmt if fmt in ("S32_LE", "S16_LE") else "S32_LE"


def _direct_dac_default_settings() -> dict[str, Any]:
    return {
        "include_sub": False,
        "sub_allpass_freq_hz": None,
        "sub_allpass_q": None,
        "sub_delay_ms": 0.0,
        "sub_polarity_invert": False,
        "sub_gain_trim_db": 0.0,
        "main_hpf_hz": None,
        "sub_hpf_hz": None,
        "sub_lpf_hz": None,
        "main_hpf_order": 2,
        "sub_hpf_order": 2,
        "sub_lpf_order": 2,
    }


def _safe_export_float(data: dict | None, key: str, default: float) -> float:
    try:
        value = float((data or {}).get(key, default) or default)
    except _RECOVERABLE_JSON_SAFE_EXCEPTIONS:
        return float(default)
    return float(value) if math.isfinite(value) else float(default)


def _safe_export_xo_order(data: dict | None, key: str, default_slope: float = 12.0) -> int:
    try:
        slope = float((data or {}).get(key, default_slope) or default_slope)
    except _RECOVERABLE_JSON_SAFE_EXCEPTIONS:
        return 2
    return int(round(float(slope) / 6.0))


def _apply_direct_dac_sub_xo_settings(settings: dict[str, Any], data: dict | None) -> None:
    main_hpf_hz = _safe_export_float(data, "sub_crossover_hz", _safe_export_float(data, "avr_crossover_hz", 80.0))
    sub_hpf_hz = _safe_export_float(data, "sub_hpf_freq", 20.0)
    sub_lpf_hz = _safe_export_float(data, "direct_dac_sub_lpf_hz", float(main_hpf_hz) + 20.0)
    settings["main_hpf_hz"] = float(main_hpf_hz if main_hpf_hz > 0.0 else 80.0)
    settings["sub_hpf_hz"] = float(sub_hpf_hz if sub_hpf_hz > 0.0 else 20.0)
    if sub_lpf_hz <= 0.0:
        sub_lpf_hz = float(settings["main_hpf_hz"]) + 20.0
    settings["sub_lpf_hz"] = float(max(float(settings["main_hpf_hz"]), float(sub_lpf_hz)))
    xo_order = _safe_export_xo_order(data, "sub_crossover_slope", 12.0)
    sub_hpf_order = _safe_export_xo_order(data, "sub_hpf_slope", 12.0)
    settings["main_hpf_order"] = int(xo_order)
    settings["sub_lpf_order"] = int(xo_order)
    settings["sub_hpf_order"] = int(sub_hpf_order)


def _apply_direct_dac_sub_allpass_settings(settings: dict[str, Any], data: dict | None) -> None:
    if not bool((data or {}).get("bass_integration_allpass_auto_applied", False)):
        return
    freq_hz = _safe_export_float(data, "bass_integration_allpass_freq_hz", float("nan"))
    q = _safe_export_float(data, "bass_integration_allpass_q", float("nan"))
    if math.isfinite(freq_hz) and math.isfinite(q) and freq_hz > 0.0 and q > 0.0:
        settings["sub_allpass_freq_hz"] = float(freq_hz)
        settings["sub_allpass_q"] = float(q)


def _effective_main_hpf_iir_export_settings(
    data: dict | None, *, result_taps: int | None = None
) -> tuple[float, int] | None:
    try:
        _xos, hpf = build_xos_hpf(dict(data or {}))
    except _RECOVERABLE_JSON_SAFE_EXCEPTIONS:
        return None
    taps = result_taps
    if taps is None:
        try:
            taps = int((data or {}).get("taps", 0) or 0)
        except _RECOVERABLE_JSON_SAFE_EXCEPTIONS:
            taps = 0
    if not hpf_settings_should_use_iir(hpf, taps):
        return None
    try:
        freq_hz = float((hpf or {}).get("freq", 0.0) or 0.0)
        order = int((hpf or {}).get("order", 0) or 0)
    except _RECOVERABLE_JSON_SAFE_EXCEPTIONS:
        return None
    if not (math.isfinite(freq_hz) and freq_hz > 0.0 and order > 0):
        return None
    return float(freq_hz), int(order)


def _direct_dac_yaml_export_settings(
    data: dict | None,
    *,
    include_sub: bool,
    result_taps: int | None = None,
) -> dict[str, Any]:
    settings = _direct_dac_default_settings()
    if bool(include_sub):
        settings["include_sub"] = True
        settings["sub_delay_ms"] = float(_safe_export_float(data, "bass_integration_sub_delay_ms", 0.0))
        settings["sub_polarity_invert"] = bool((data or {}).get("bass_integration_sub_polarity_invert", False))
        settings["sub_gain_trim_db"] = float(_safe_export_float(data, "bass_integration_sub_gain_trim_db", 0.0))
        _apply_direct_dac_sub_xo_settings(settings, data)
        _apply_direct_dac_sub_allpass_settings(settings, data)
    main_iir_hpf = _effective_main_hpf_iir_export_settings(data, result_taps=result_taps)
    if main_iir_hpf is not None and settings.get("main_hpf_hz") is None:
        settings["main_hpf_hz"] = float(main_iir_hpf[0])
        settings["main_hpf_order"] = int(main_iir_hpf[1])
    return settings


def _hybrid_iir_biquads_from_result(result: Any, side: str) -> list[dict]:
    st_name = "l_st" if str(side).lower().startswith("l") else "r_st"
    try:
        st = dict(getattr(result, st_name, {}) or {})
    except (
        TypeError,
        ValueError,
        AttributeError,
        KeyError,
        IndexError,
        OverflowError,
        ImportError,
        ModuleNotFoundError,
        RecursionError,
    ):
        st = {}
    return [dict(item) for item in list(st.get("hybrid_iir_biquads", []) or []) if isinstance(item, dict)]


def _camilladsp_hpf_type_label(order: int) -> str:
    return "Biquad Highpass" if int(order) <= 2 else "BiquadCombo LinkwitzRileyHighpass"


def _camilladsp_lpf_type_label(order: int) -> str:
    return "Biquad Lowpass" if int(order) <= 2 else "BiquadCombo LinkwitzRileyLowpass"


def _iir_report_biquad_lines(name_prefix: str, biquads: list[dict]) -> list[str]:
    lines: list[str] = []
    for idx, biquad in enumerate(biquads, start=1):
        try:
            freq = float(biquad.get("freq", 0.0) or 0.0)
            q = float(biquad.get("q", 0.0) or 0.0)
            gain = float(biquad.get("gain", 0.0) or 0.0)
            confidence = float(biquad.get("confidence", 0.0) or 0.0)
            safe_cut = float(biquad.get("safe_cut_db", 0.0) or 0.0)
            transfer_cut = float(biquad.get("transfer_cut_db", 0.0) or 0.0)
            residual_cut = float(biquad.get("residual_cut_db", 0.0) or 0.0)
        except _RECOVERABLE_JSON_SAFE_EXCEPTIONS:
            continue
        lines.append(
            f"{name_prefix}_hybrid_iir_{idx}: Peaking, "
            f"freq={freq:.3f} Hz, q={q:.6f}, gain={gain:.3f} dB, "
            f"confidence={confidence:.3f}, safe_cut={safe_cut:.3f} dB, "
            f"fir_transfer={transfer_cut:.3f} dB, residual_extra={residual_cut:.3f} dB"
        )
    return lines


def _build_iir_report_text(
    *,
    data: dict | None,
    result: FilterResult | None,
    yaml_settings: dict[str, Any],
    fs_v: int,
    ft_short: str,
    irw_tag: str,
) -> str | None:
    left_iir = _hybrid_iir_biquads_from_result(result, "left")
    right_iir = _hybrid_iir_biquads_from_result(result, "right")
    lines: list[str] = [
        "DecayCore IIR filters",
        f"Filter: {ft_short}",
        f"Sample rate: {int(fs_v)} Hz",
        f"IR window tag: {irw_tag}",
        "",
        "CamillaDSP filter order: mastergain -> external IIR -> FIR convolver",
        "",
    ]
    has_iir = False
    main_hpf = yaml_settings.get("main_hpf_hz")
    main_order = int(yaml_settings.get("main_hpf_order", 2) or 2)
    try:
        main_hpf_f = float(main_hpf)
    except _RECOVERABLE_JSON_SAFE_EXCEPTIONS:
        main_hpf_f = 0.0
    if math.isfinite(main_hpf_f) and main_hpf_f > 0.0:
        has_iir = True
        lines.append(
            f"main_hpf: {_camilladsp_hpf_type_label(main_order)}, "
            f"freq={main_hpf_f:.3f} Hz, order={main_order:d}, slope={main_order * 6:d} dB/oct"
        )
    if bool(yaml_settings.get("include_sub", False)):
        sub_hpf = yaml_settings.get("sub_hpf_hz")
        sub_lpf = yaml_settings.get("sub_lpf_hz")
        sub_hpf_order = int(yaml_settings.get("sub_hpf_order", 2) or 2)
        sub_lpf_order = int(yaml_settings.get("sub_lpf_order", 2) or 2)
        try:
            sub_hpf_f = float(sub_hpf)
        except _RECOVERABLE_JSON_SAFE_EXCEPTIONS:
            sub_hpf_f = 0.0
        try:
            sub_lpf_f = float(sub_lpf)
        except _RECOVERABLE_JSON_SAFE_EXCEPTIONS:
            sub_lpf_f = 0.0
        if math.isfinite(sub_hpf_f) and sub_hpf_f > 0.0:
            has_iir = True
            lines.append(
                f"sub_hpf: {_camilladsp_hpf_type_label(sub_hpf_order)}, "
                f"freq={sub_hpf_f:.3f} Hz, order={sub_hpf_order:d}, slope={sub_hpf_order * 6:d} dB/oct"
            )
        if math.isfinite(sub_lpf_f) and sub_lpf_f > 0.0:
            has_iir = True
            lines.append(
                f"sub_lpf: {_camilladsp_lpf_type_label(sub_lpf_order)}, "
                f"freq={sub_lpf_f:.3f} Hz, order={sub_lpf_order:d}, slope={sub_lpf_order * 6:d} dB/oct"
            )
    if left_iir or right_iir:
        if lines[-1] != "":
            lines.append("")
        lines.append("Hybrid FIR-IIR modal cuts")
        left_lines = _iir_report_biquad_lines("l", left_iir)
        right_lines = _iir_report_biquad_lines("r", right_iir)
        lines.extend(left_lines or ["l_hybrid_iir: none"])
        lines.extend(right_lines or ["r_hybrid_iir: none"])
        has_iir = True
    if not has_iir:
        return None
    try:
        taps = int(getattr(result, "taps", 0) or (data or {}).get("taps", 0) or 0)
    except _RECOVERABLE_JSON_SAFE_EXCEPTIONS:
        taps = 0
    if taps > 0:
        lines.append("")
        lines.append(f"FIR taps: {taps:d}")
    return "\n".join(lines).strip() + "\n"


def _iir_report_name(*, ft_short: str, fs_v: int, irw_tag: str) -> str:
    return f"IIR_{ft_short}_{int(fs_v)}Hz_{irw_tag}.txt"


def _build_summary_text(
    data: dict,
    fs_v: int,
    ft_short: str,
    file_ts: str,
    l_st: dict | None,
    r_st: dict | None,
    *,
    ranking_context: dict | None = None,
) -> str:
    summary_content = plots.format_summary_content(data, l_st, r_st)
    summary_content = _append_export_ranking(summary_content, int(fs_v), ranking_context)
    try:
        hc_src = str(data.get("hc_source", "") or "").strip()
        if hc_src:
            summary_content = f"House curve: {hc_src}\n" + summary_content
    except (
        TypeError,
        ValueError,
        AttributeError,
        KeyError,
        IndexError,
        OverflowError,
        ImportError,
        ModuleNotFoundError,
        RecursionError,
    ):
        logger.exception("house curve summary prefix")
    summary_content = _append_export_decision_summary(summary_content, data, fs_v, l_st, r_st)
    summary_content = _append_dsp_effective_params(summary_content, data, fs_v)
    summary_content = _append_realized_phase_limit(summary_content, data, l_st, r_st)
    summary_content = _append_leveling_summary(summary_content, l_st, r_st)
    summary_content = _append_acoustic_events(summary_content, l_st, r_st)
    summary_content = _append_lr_difference_summary(summary_content, l_st, r_st)

    if TEST_MODE:
        try:
            diag = _build_diagnostics_dict(data, fs_v, l_st, r_st)
            summary_content += "\n\n--- DIAGNOSTICS_JSON_BEGIN ---\n"
            summary_content += json.dumps(_json_safe(diag), indent=2)
            summary_content += "\n--- DIAGNOSTICS_JSON_END ---\n"
        except (TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError) as e:
            summary_content += "\n\n--- DIAGNOSTICS_JSON_BEGIN ---\n"
            summary_content += json.dumps(
                {
                    "schema_version": 1,
                    "error": f"diagnostics_json_failed: {type(e).__name__}: {e}",
                },
                indent=2,
            )
            summary_content += "\n--- DIAGNOSTICS_JSON_END ---\n"

    return summary_content


def _write_fs_outputs(
    zf,
    data,
    fs_v,
    ft_short,
    file_ts,
    f_l,
    m_l,
    p_l,
    l_imp,
    l_st,
    f_r,
    m_r,
    p_r,
    r_imp,
    r_st,
    *,
    result: FilterResult | None = None,
    irw_tag: str = "auto",
    ranking_context: dict | None = None,
):
    if result is not None:
        (
            fs_v,
            f_l,
            m_l,
            p_l,
            l_imp,
            l_st,
            f_r,
            m_r,
            p_r,
            r_imp,
            r_st,
        ) = _extract_result_payload(result)

    sum_name = f"Summary_{ft_short}_{fs_v}Hz_{file_ts}.txt"
    summary_content = _build_summary_text(data, fs_v, ft_short, file_ts, l_st, r_st, ranking_context=ranking_context)
    zf.writestr(sum_name, summary_content)

    target_curve_tag = str(data.get("target_curve_tag", "") or "").strip()

    hlc_cfg = generate_hlc_config(
        fs_v,
        ft_short,
        file_ts,
        irw_tag=irw_tag,
        target_curve_tag=target_curve_tag,
        layout=data.get("layout", "Mono"),
    )
    zf.writestr(f"Config_{ft_short}_{fs_v}Hz_{irw_tag}.cfg", hlc_cfg)

    sub_ir = getattr(result, "sub_ir", None) if result is not None else None
    yaml_settings = _direct_dac_yaml_export_settings(
        data,
        include_sub=bool(sub_ir is not None and getattr(sub_ir, "size", 0) > 0),
        result_taps=int(getattr(result, "taps", 0) or 0) if result is not None else None,
    )
    iir_report = _build_iir_report_text(
        data=data,
        result=result,
        yaml_settings=yaml_settings,
        fs_v=int(fs_v),
        ft_short=ft_short,
        irw_tag=irw_tag,
    )
    if iir_report:
        zf.writestr(_iir_report_name(ft_short=ft_short, fs_v=int(fs_v), irw_tag=irw_tag), iir_report)

    if not bool(data.get("multi_rate_opt", False)):
        yaml_content = generate_raspberry_yaml(
            fs_v,
            ft_short,
            file_ts,
            master_gain_db=0.0,
            irw_tag=irw_tag,
            target_curve_tag=target_curve_tag,
            layout=data.get("layout", "Mono"),
            program_version=str(data.get("program_version", "") or "").strip(),
            include_sub=bool(yaml_settings.get("include_sub", False)),
            sub_allpass_freq_hz=yaml_settings.get("sub_allpass_freq_hz"),
            sub_allpass_q=yaml_settings.get("sub_allpass_q"),
            sub_delay_ms=yaml_settings.get("sub_delay_ms"),
            sub_polarity_invert=bool(yaml_settings.get("sub_polarity_invert", False)),
            sub_gain_trim_db=yaml_settings.get("sub_gain_trim_db"),
            main_hpf_hz=yaml_settings.get("main_hpf_hz"),
            sub_hpf_hz=yaml_settings.get("sub_hpf_hz"),
            sub_lpf_hz=yaml_settings.get("sub_lpf_hz"),
            main_hpf_order=yaml_settings.get("main_hpf_order"),
            sub_hpf_order=yaml_settings.get("sub_hpf_order"),
            sub_lpf_order=yaml_settings.get("sub_lpf_order"),
            device_format=_validated_device_format(data),
            left_iir_biquads=_hybrid_iir_biquads_from_result(result, "left"),
            right_iir_biquads=_hybrid_iir_biquads_from_result(result, "right"),
        )
        zf.writestr(
            _camilladsp_yaml_name(data=data, ft_short=ft_short, irw_tag=irw_tag, fs_v=int(fs_v)),
            yaml_content,
        )
