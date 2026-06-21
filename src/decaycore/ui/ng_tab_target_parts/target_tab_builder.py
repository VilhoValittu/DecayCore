# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""NiceGUI Target tab builder.

Replaces build_target_section() from layout_builders.py.
"""
from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger("DecayCore")

from .. import ng_controls as ctrl
from ..ng_sections import page_shell, section_card
from ..target_preview_common import apply_manual_target_preview_adjustments
from ..target_preview_interaction import (
    build_draggable_tilt_handle_shape,
    build_draggable_target_shape,
    build_level_window_trace,
    build_target_curve_path,
    build_tilt_handle_path,
    build_vertical_marker_trace,
    extract_target_tilt_from_shape_relayout,
    extract_vertical_shift_from_shape_relayout,
    parse_svg_path_points,
)
from ...common.measurement_features import build_harmonic_boost_risk_curve, build_target_decay_hint
from ...ui_i18n import (
    LVL_ALGO_MEDIAN,
    LVL_ALGO_OPTION_LABEL_KEYS,
    LVL_MODE_AUTO,
    LVL_MODE_MANUAL,
    LVL_MODE_OPTION_LABEL_KEYS,
    OUTPUT_TILT_SOURCE_OFF,
    OUTPUT_TILT_SOURCE_OPTION_LABEL_KEYS,
    normalize_lvl_algo_value,
    normalize_lvl_mode_value,
    normalize_output_tilt_source_value,
    tr_options,
)

_HC_OPTS = {
    "Harman6":   "Harman 6 dB",
    "Harman8":   "Harman 8 dB",
    "Harman4":   "Harman 4 dB",
    "Harman10":  "Harman 10 dB",
    "Harman12":  "Harman 12 dB",
    "Studio":    "Studio Tilt",
    "Nearfield": "Nearfield",
    "HiFi":      "HiFi Loudness",
    "Speech":    "Speech",
    "Toole":     "Toole",
    "BK_Light":  "BK Light",
    "BK_Medium": "BK Medium",
    "BK_Strong": "BK Strong",
    "Flat":      "Flat",
    "Cinema":    "Cinema",
    "Upload":    "Upload Custom",
}

_TARGET_PREVIEW_REFRESH_TOKEN = 0
_TARGET_PREVIEW_DRAG_BASE_POINTS: list[tuple[float, float]] = []
_TARGET_PREVIEW_TILT_HANDLE_POINTS: list[tuple[float, float]] = []
_TARGET_PREVIEW_DRAG_ACTIVE = False
_TARGET_PREVIEW_PLOT = None
_TARGET_TAB_TRANSLATE: Callable[[str], str] | None = None


def _decay_hint_status_color(status: str) -> str:
    return {
        "ok": "#2f855a",
        "caution": "#b7791f",
        "strong": "#c53030",
        "unavailable": "#4a5568",
    }.get(str(status or "").strip().lower(), "#4a5568")


def _target_hint_translate(key: str) -> str:
    from ...resources.i8n.decaycore_i18n import t as resource_t  # noqa: PLC0415

    translator = _TARGET_TAB_TRANSLATE or resource_t
    try:
        translated = str(translator(key))
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
        translated = key

    if translated != key:
        return translated
    return str(resource_t(key))


def _normalize_harmonic_plot_source(harmonic_freq_hz, harmonic_magnitudes_db):
    try:
        import numpy as np  # noqa: PLC0415

        if harmonic_freq_hz is None or not isinstance(harmonic_magnitudes_db, dict):
            return None, None
        freq_raw = np.asarray(harmonic_freq_hz, dtype=float).reshape(-1)
        if freq_raw.size < 4:
            return None, None
        freq_mask = np.isfinite(freq_raw) & (freq_raw > 0.0)
        if int(freq_mask.sum()) < 4:
            return None, None
        freq = freq_raw[freq_mask]
        mags_out: dict[int, object] = {}
        for order_raw, arr_raw in harmonic_magnitudes_db.items():
            try:
                order = int(order_raw)
                arr = np.asarray(arr_raw, dtype=float).reshape(-1)
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
                continue
            if arr.size != freq_raw.size:
                continue
            arr = arr[freq_mask]
            if arr.size != freq.size or not np.isfinite(arr).any():
                continue
            mags_out[order] = arr
        if not mags_out:
            return None, None
        return freq, mags_out
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
        return None, None


def _empty_target_metadata_channel(side: str) -> dict[str, object]:
    return {
        "side": side,
        "source_kind": "none",
        "source_path": "",
        "rt60_value": None,
        "rt60_bands": None,
        "harmonic_freq_hz": None,
        "harmonic_magnitudes_db": None,
        "harmonic_risk_freq_hz": None,
        "harmonic_risk_curve": None,
        "harmonic_risk_summary": None,
    }


def _build_target_preview_metadata_payload() -> dict[str, object]:
    from ...common.measurement_features import normalize_rt60_bands, normalize_rt60_value  # noqa: PLC0415
    from ...io.generated_measurement_source import generated_source_matches_upload  # noqa: PLC0415
    from ...io.measurements_loader import _try_load_harmonic_sidecar, _try_load_rt60_sidecar  # noqa: PLC0415
    from ..target_preview_cache import load_path_measurement_curve, load_upload_measurement_curve  # noqa: PLC0415

    def _cv(name, default=None):
        return ctrl.value(name, default)

    def _curve_loader_settings() -> tuple[float, float, int]:
        try:
            pre_ms = float(_cv("ir_window_left", 85.0) or 85.0)
        except (TypeError, ValueError):
            pre_ms = 85.0
        try:
            post_ms = float(_cv("ir_window_right", _cv("ir_window", 500.0)) or 500.0)
        except (TypeError, ValueError):
            post_ms = 500.0
        return pre_ms, post_ms, 96

    def _fundamental_curve(slot: str, *, bass_integration_enabled: bool):
        source = _cv(f"generated_measurement_{slot.lower()}", None)
        upload_value = _cv(f"file_{slot.lower()}", None)
        if not bass_integration_enabled and isinstance(source, dict) and generated_source_matches_upload(source, upload_value):
            analysis_freq = source.get("spatial_avg_analysis_freq_hz", source.get("analysis_freq_hz"))
            analysis_mag = source.get("spatial_avg_analysis_magnitude_db", source.get("analysis_magnitude_db"))
            if analysis_freq is not None and analysis_mag is not None:
                return analysis_freq, analysis_mag
        pre_ms, post_ms, smoothing_level = _curve_loader_settings()
        if not bass_integration_enabled:
            freq, mag = load_upload_measurement_curve(
                upload_value,
                pre_ms=pre_ms,
                post_ms=post_ms,
                smoothing_level=smoothing_level,
            )
            if freq is not None and mag is not None:
                return freq, mag
        path_key = f"local_path_{slot.lower()}_main" if bass_integration_enabled else f"local_path_{slot.lower()}"
        freq, mag = load_path_measurement_curve(
            _cv(path_key, ""),
            pre_ms=pre_ms,
            post_ms=post_ms,
            smoothing_level=smoothing_level,
        )
        return freq, mag

    def _collect_generated(slot: str) -> dict[str, object] | None:
        source = _cv(f"generated_measurement_{slot.lower()}", None)
        upload_value = _cv(f"file_{slot.lower()}", None)
        if not isinstance(source, dict) or not generated_source_matches_upload(source, upload_value):
            return None
        channel = _empty_target_metadata_channel(slot)
        channel["source_kind"] = "generated"
        channel["rt60_value"] = normalize_rt60_value(source.get("measured_rt60"))
        channel["rt60_bands"] = normalize_rt60_bands(source.get("measured_rt60_bands"))
        harmonic_freq, harmonic_mags = _normalize_harmonic_plot_source(
            source.get("harmonic_freq_hz"),
            source.get("harmonic_magnitudes_db"),
        )
        channel["harmonic_freq_hz"] = harmonic_freq
        channel["harmonic_magnitudes_db"] = harmonic_mags
        if harmonic_freq is not None and harmonic_mags:
            analysis_freq = source.get("spatial_avg_analysis_freq_hz", source.get("analysis_freq_hz"))
            analysis_mag = source.get("spatial_avg_analysis_magnitude_db", source.get("analysis_magnitude_db"))
            risk_freq, risk_curve, risk_summary = build_harmonic_boost_risk_curve(
                harmonic_freq,
                harmonic_mags,
                fundamental_freq_hz=analysis_freq,
                fundamental_mag_db=analysis_mag,
            )
            channel["harmonic_risk_freq_hz"] = risk_freq
            channel["harmonic_risk_curve"] = risk_curve
            channel["harmonic_risk_summary"] = risk_summary
        return channel

    def _collect_sidecar(
        slot: str,
        path_value: object,
        *,
        fundamental_freq_hz=None,
        fundamental_mag_db=None,
    ) -> dict[str, object] | None:
        path = str(path_value or "").strip()
        if not path:
            return None
        channel = _empty_target_metadata_channel(slot)
        channel["source_kind"] = "sidecar"
        channel["source_path"] = path
        rt60_val, rt60_bands = _try_load_rt60_sidecar(path)
        channel["rt60_value"] = normalize_rt60_value(rt60_val)
        channel["rt60_bands"] = normalize_rt60_bands(rt60_bands)
        harmonic_freq_raw, harmonic_mags_raw = _try_load_harmonic_sidecar(path)
        harmonic_freq, harmonic_mags = _normalize_harmonic_plot_source(
            harmonic_freq_raw,
            harmonic_mags_raw,
        )
        channel["harmonic_freq_hz"] = harmonic_freq
        channel["harmonic_magnitudes_db"] = harmonic_mags
        if harmonic_freq is not None and harmonic_mags:
            risk_freq, risk_curve, risk_summary = build_harmonic_boost_risk_curve(
                harmonic_freq,
                harmonic_mags,
                fundamental_freq_hz=fundamental_freq_hz,
                fundamental_mag_db=fundamental_mag_db,
            )
            channel["harmonic_risk_freq_hz"] = risk_freq
            channel["harmonic_risk_curve"] = risk_curve
            channel["harmonic_risk_summary"] = risk_summary
        return channel

    app_mode = str(_cv("mode", "BASIC") or "BASIC").upper()
    bass_integration_enabled = bool(_cv("bass_integration_enable", False))

    channels: dict[str, dict[str, object]] = {}
    rt60_values: list[float | None] = []
    rt60_bands_list: list[dict[float, float] | None] = []
    harmonic_summaries: list[dict | None] = []
    has_rt60 = False
    has_harmonics = False
    has_harmonic_risk = False

    for slot in ("L", "R"):
        channel = None
        if not bass_integration_enabled:
            channel = _collect_generated(slot)
        if channel is None:
            path_key = f"local_path_{slot.lower()}_main" if bass_integration_enabled else f"local_path_{slot.lower()}"
            fundamental_freq_hz, fundamental_mag_db = _fundamental_curve(
                slot,
                bass_integration_enabled=bass_integration_enabled,
            )
            channel = _collect_sidecar(
                slot,
                _cv(path_key, ""),
                fundamental_freq_hz=fundamental_freq_hz,
                fundamental_mag_db=fundamental_mag_db,
            )
        if channel is None:
            channel = _empty_target_metadata_channel(slot)

        channels[slot] = channel
        rt60_values.append(channel.get("rt60_value"))
        rt60_bands_list.append(channel.get("rt60_bands"))
        harmonic_summaries.append(channel.get("harmonic_risk_summary"))
        if isinstance(channel.get("rt60_bands"), dict) and channel["rt60_bands"]:
            has_rt60 = True
        if isinstance(channel.get("harmonic_magnitudes_db"), dict) and channel["harmonic_magnitudes_db"]:
            has_harmonics = True
        if channel.get("harmonic_risk_freq_hz") is not None and channel.get("harmonic_risk_curve") is not None:
            has_harmonic_risk = True

    return {
        "app_mode": app_mode,
        "channels": channels,
        "rt60_values": rt60_values,
        "rt60_bands_list": rt60_bands_list,
        "harmonic_summaries": harmonic_summaries,
        "has_rt60": has_rt60,
        "has_harmonics": has_harmonics,
        "has_harmonic_risk": has_harmonic_risk,
        "has_any_metadata": bool(has_rt60 or has_harmonics or has_harmonic_risk),
    }


def _build_target_decay_hint_payload() -> dict[str, object]:
    payload = _build_target_preview_metadata_payload()
    hint = build_target_decay_hint(
        rt60_values=payload.get("rt60_values"),
        rt60_bands_list=payload.get("rt60_bands_list"),
        harmonic_summaries=payload.get("harmonic_summaries"),
    )
    hint["app_mode"] = payload.get("app_mode", "BASIC")
    return hint


def _build_target_preview_rt60_fig(metadata_payload: dict[str, object]):
    try:
        import plotly.graph_objects as go  # noqa: PLC0415

        traces_added = 0
        fig = go.Figure()
        colors = {"L": "#2563eb", "R": "#ea580c"}
        channels = metadata_payload.get("channels", {})
        if not isinstance(channels, dict):
            return None
        for side in ("L", "R"):
            channel = channels.get(side, {})
            bands = channel.get("rt60_bands") if isinstance(channel, dict) else None
            if not isinstance(bands, dict) or not bands:
                continue
            freqs = sorted(float(freq) for freq in bands.keys())
            vals = [float(bands[freq]) for freq in freqs]
            fig.add_trace(go.Scatter(
                x=freqs,
                y=vals,
                mode="lines+markers",
                name=f"RT60 {side}",
                line=dict(color=colors[side], width=2.0),
                marker=dict(size=6),
            ))
            traces_added += 1
        if traces_added == 0:
            return None
        fig.update_xaxes(
            type="log",
            title_text="Hz",
            gridcolor="rgba(15,23,42,0.10)",
            linecolor="rgba(15,23,42,0.22)",
            zerolinecolor="rgba(15,23,42,0.10)",
        )
        fig.update_yaxes(
            title_text="s",
            rangemode="tozero",
            gridcolor="rgba(15,23,42,0.10)",
            linecolor="rgba(15,23,42,0.22)",
            zerolinecolor="rgba(15,23,42,0.10)",
        )
        fig.update_layout(
            height=220,
            margin=dict(l=40, r=20, t=20, b=35),
            showlegend=True,
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#1f2937"),
        )
        return fig.to_plotly_json()
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


def _build_target_preview_harmonics_fig(metadata_payload: dict[str, object]):
    try:
        import numpy as np  # noqa: PLC0415
        import plotly.graph_objects as go  # noqa: PLC0415

        fig = go.Figure()
        traces_added = 0
        colors = {
            2: {"L": "#60a5fa", "R": "#fb923c"},
            3: {"L": "#2563eb", "R": "#ea580c"},
            4: {"L": "#1d4ed8", "R": "#c2410c"},
            5: {"L": "#1e3a8a", "R": "#9a3412"},
        }
        channels = metadata_payload.get("channels", {})
        if not isinstance(channels, dict):
            return None
        for side in ("L", "R"):
            channel = channels.get(side, {})
            freq = channel.get("harmonic_freq_hz") if isinstance(channel, dict) else None
            mags = channel.get("harmonic_magnitudes_db") if isinstance(channel, dict) else None
            if freq is None or not isinstance(mags, dict) or not mags:
                continue
            freq_arr = np.asarray(freq, dtype=float).reshape(-1)
            for order, values in sorted(mags.items()):
                val_arr = np.asarray(values, dtype=float).reshape(-1)
                if val_arr.size != freq_arr.size:
                    continue
                mask = (freq_arr > 0.0) & np.isfinite(freq_arr) & np.isfinite(val_arr)
                if int(mask.sum()) < 4:
                    continue
                fig.add_trace(go.Scatter(
                    x=freq_arr[mask],
                    y=val_arr[mask],
                    mode="lines",
                    name=f"H{int(order)} {side}",
                    line=dict(color=colors.get(int(order), {}).get(side, "#6b7280"), width=1.8),
                ))
                traces_added += 1
        if traces_added == 0:
            return None
        fig.update_xaxes(
            type="log",
            title_text="Hz",
            gridcolor="rgba(15,23,42,0.10)",
            linecolor="rgba(15,23,42,0.22)",
            zerolinecolor="rgba(15,23,42,0.10)",
        )
        fig.update_yaxes(
            title_text="dB",
            gridcolor="rgba(15,23,42,0.10)",
            linecolor="rgba(15,23,42,0.22)",
            zerolinecolor="rgba(15,23,42,0.10)",
        )
        fig.update_layout(
            height=240,
            margin=dict(l=40, r=20, t=20, b=35),
            showlegend=True,
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#1f2937"),
        )
        return fig.to_plotly_json()
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


def _build_target_preview_harmonic_risk_fig(metadata_payload: dict[str, object]):
    try:
        import numpy as np  # noqa: PLC0415
        import plotly.graph_objects as go  # noqa: PLC0415

        fig = go.Figure()
        traces_added = 0
        colors = {"L": "#2563eb", "R": "#ea580c"}
        max_risk = 0.0
        channels = metadata_payload.get("channels", {})
        if not isinstance(channels, dict):
            return None
        for side in ("L", "R"):
            channel = channels.get(side, {})
            freq = channel.get("harmonic_risk_freq_hz") if isinstance(channel, dict) else None
            curve = channel.get("harmonic_risk_curve") if isinstance(channel, dict) else None
            if freq is None or curve is None:
                continue
            freq_arr = np.asarray(freq, dtype=float).reshape(-1)
            curve_arr = np.asarray(curve, dtype=float).reshape(-1)
            if curve_arr.size != freq_arr.size:
                continue
            mask = (freq_arr > 0.0) & np.isfinite(freq_arr) & np.isfinite(curve_arr)
            if int(mask.sum()) < 4:
                continue
            masked_risk = curve_arr[mask]
            if not np.any(masked_risk > 0.0):
                continue
            max_risk = max(max_risk, float(np.max(masked_risk)))
            fig.add_trace(go.Scatter(
                x=freq_arr[mask],
                y=masked_risk,
                mode="lines",
                name=f"Risk {side}",
                line=dict(color=colors[side], width=2.0),
            ))
            traces_added += 1
        if traces_added == 0:
            return None
        y_upper = min(1.05, max(0.03, max_risk * 1.20))
        fig.update_xaxes(
            type="log",
            title_text="Hz",
            range=[np.log10(20.0), np.log10(800.0)],
            gridcolor="rgba(15,23,42,0.10)",
            linecolor="rgba(15,23,42,0.22)",
            zerolinecolor="rgba(15,23,42,0.10)",
        )
        fig.update_yaxes(
            title_text="Risk",
            range=[0.0, y_upper],
            gridcolor="rgba(15,23,42,0.10)",
            linecolor="rgba(15,23,42,0.22)",
            zerolinecolor="rgba(15,23,42,0.10)",
        )
        fig.update_layout(
            height=220,
            margin=dict(l=40, r=20, t=20, b=35),
            showlegend=True,
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#1f2937"),
        )
        return fig.to_plotly_json()
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


def _render_target_preview_metadata() -> None:
    from nicegui import ui  # noqa: PLC0415

    metadata_col = ctrl.get_container("target_preview_metadata_scope")
    if metadata_col is None:
        return

    metadata_payload = _build_target_preview_metadata_payload()
    metadata_col.clear()
    if not bool(metadata_payload.get("has_any_metadata", False)):
        return

    rt60_fig = _build_target_preview_rt60_fig(metadata_payload)
    harmonics_fig = _build_target_preview_harmonics_fig(metadata_payload)
    risk_fig = _build_target_preview_harmonic_risk_fig(metadata_payload)
    if rt60_fig is None and harmonics_fig is None and risk_fig is None:
        return

    with metadata_col:
        with ui.expansion(_target_hint_translate("target_preview_metadata_title")).classes("w-full"):
            with ui.column().classes("w-full gap-3"):
                if rt60_fig is not None:
                    ui.label(_target_hint_translate("target_preview_metadata_rt60_title")).classes("text-sm font-semibold")
                    ui.plotly(rt60_fig).classes("w-full")
                if harmonics_fig is not None:
                    ui.label(_target_hint_translate("target_preview_metadata_harmonics_title")).classes("text-sm font-semibold")
                    ui.plotly(harmonics_fig).classes("w-full")
                if risk_fig is not None:
                    ui.label(_target_hint_translate("target_preview_metadata_risk_title")).classes("text-sm font-semibold")
                    ui.plotly(risk_fig).classes("w-full")
                elif bool(metadata_payload.get("has_harmonics", False)):
                    ui.label(_target_hint_translate("target_preview_metadata_risk_title")).classes("text-sm font-semibold")
                    ui.label(_target_hint_translate("target_preview_metadata_risk_none")).classes("text-xs text-gray-500")


def _render_target_decay_hint() -> None:
    from nicegui import ui  # noqa: PLC0415

    hint_col = ctrl.get_container("target_decay_hint_scope")
    if hint_col is None:
        return

    hint = _build_target_decay_hint_payload()
    status = str(hint.get("status", "unavailable") or "unavailable")
    reason = str(hint.get("reason", "none") or "none")
    app_mode = str(hint.get("app_mode", "BASIC") or "BASIC").upper()
    is_advanced = app_mode == "ADVANCED"

    summary_key = {
        "unavailable": "target_decay_hint_summary_no_data",
        "ok": "target_decay_hint_summary_ok",
        "caution": "target_decay_hint_summary_caution",
        "strong": "target_decay_hint_summary_strong",
    }.get(status, "target_decay_hint_summary_no_data")

    advice_map = {
        "no_data": "target_decay_hint_body_no_data",
        "keep_changes_measured": "target_decay_hint_body_ok",
        "avoid_deep_null_boost": "target_decay_hint_body_avoid_nulls",
        "prefer_conservative_bass_boost": "target_decay_hint_body_conservative_bass",
        "keep_correction_band_limited": "target_decay_hint_body_band_limited",
    }
    summary_detail_map = {
        ("caution", "rt60"): "target_decay_hint_body_rt60_caution",
        ("caution", "harmonic"): "target_decay_hint_body_harmonic_caution",
        ("caution", "mixed"): "target_decay_hint_body_mixed_caution",
        ("strong", "rt60"): "target_decay_hint_body_rt60_strong",
        ("strong", "harmonic"): "target_decay_hint_body_harmonic_strong",
        ("strong", "mixed"): "target_decay_hint_body_mixed_strong",
    }

    detail_keys: list[str] = []
    summary_detail_key = summary_detail_map.get((status, reason))
    if summary_detail_key:
        detail_keys.append(summary_detail_key)
    for code in hint.get("advice_codes", ()):
        key = advice_map.get(str(code))
        if key and key not in detail_keys:
            detail_keys.append(key)
    if not is_advanced and detail_keys:
        detail_keys = detail_keys[:1]
    elif is_advanced:
        detail_keys = detail_keys[:2]

    badge_key = {
        "unavailable": "target_decay_hint_badge_unavailable",
        "ok": "target_decay_hint_badge_ok",
        "caution": "target_decay_hint_badge_caution",
        "strong": "target_decay_hint_badge_strong",
    }.get(status, "target_decay_hint_badge_unavailable")
    badge_color = _decay_hint_status_color(status)

    hint_col.clear()
    with hint_col:
        ui.label(_target_hint_translate("target_decay_hint_title")).classes("text-sm font-semibold")
        with ui.row().classes("w-full items-center gap-2"):
            ui.html(
                (
                    '<span style="display:inline-block;padding:4px 10px;'
                    'border-radius:9999px;font-size:0.72rem;font-weight:700;'
                    f'background:{badge_color};color:#ffffff;">{_target_hint_translate(badge_key)}</span>'
                )
            )
            ui.label(_target_hint_translate(summary_key)).classes("text-xs cf-target-hint-summary")
        for key in detail_keys:
            ui.label(_target_hint_translate(key)).classes("text-xs cf-target-hint-detail")

def _step_manual_target(delta_db: float) -> None:
    try:
        cur = float(ctrl.value("lvl_manual_db", 0.0) or 0.0)
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
        cur = 0.0

    nxt = round((float(cur) + float(delta_db)) * 10.0) / 10.0
    ctrl.set_value("lvl_manual_db", float(nxt))
    refresh_target_preview()

def _step_manual_target_tilt(delta_db_per_oct: float) -> None:
    try:
        cur = float(ctrl.value("manual_target_tilt_db_per_oct", 0.0) or 0.0)
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
        cur = 0.0

    nxt = round((float(cur) + float(delta_db_per_oct)) * 10.0) / 10.0
    ctrl.set_value("manual_target_tilt_db_per_oct", float(nxt))
    refresh_target_preview()

def build_target_tab(*, t: Callable, get_val: Callable) -> None:
    global _TARGET_PREVIEW_DRAG_ACTIVE, _TARGET_PREVIEW_DRAG_BASE_POINTS, _TARGET_PREVIEW_TILT_HANDLE_POINTS
    global _TARGET_PREVIEW_PLOT, _TARGET_TAB_TRANSLATE

    from nicegui import ui
    from ...application.house_curve_service import _normalize_hc_mode_key  # noqa: PLC0415

    _TARGET_PREVIEW_DRAG_ACTIVE = False
    _TARGET_PREVIEW_DRAG_BASE_POINTS = []
    _TARGET_PREVIEW_TILT_HANDLE_POINTS = []
    _TARGET_PREVIEW_PLOT = None
    _TARGET_TAB_TRANSLATE = t

    with page_shell(title=t("tab_target"), intro=t("target_page_intro"), wide=True):
        with section_card(title=t("ui_target_preview"), intro=t("target_preview_legend_hint"), hero=True):
            hint_col = ui.column().classes("w-full gap-1")
            ctrl.register_container("target_decay_hint_scope", hint_col)

            preview_col = ui.column().classes("w-full")
            ctrl.register_container("target_preview_scope", preview_col)
            metadata_col = ui.column().classes("w-full")
            ctrl.register_container("target_preview_metadata_scope", metadata_col)
            _render_target_decay_hint()
            _render_target_preview_metadata()

        _hc_file = ctrl._ValueHolder(get_val("hc_custom_file", None))
        ctrl.register("hc_custom_file", _hc_file)

        hc_value = _normalize_hc_mode_key(get_val("hc_mode", "Harman6"))
        with section_card(title=t("hc_mode")):
            ctrl.register(
                "hc_mode",
                ui.select(
                    options=_HC_OPTS,
                    value=hc_value,
                    label=t("hc_mode"),
                ).props("dense outlined").classes("w-full"),
            )

            with ui.column().classes("w-full") as hc_upload_col:
                ui.label(t("hc_custom")).classes("text-sm font-medium")

                async def _on_hc_upload(e) -> None:
                    _hc_file.set_value({
                        "filename": e.file.name,
                        "content": await e.file.read(),
                        "mime_type": getattr(e.file, "content_type", ""),
                    })
                    refresh_target_preview()

                ui.upload(
                    label=t("hc_custom"),
                    on_upload=_on_hc_upload,
                    auto_upload=True,
                ).props('accept=".txt"').classes("w-full")
            ctrl.register_container("hc_custom_upload_col", hc_upload_col)
            hc_upload_col.set_visibility(hc_value == "Upload")

        with section_card(title=t("ui_leveling_gain")):
            with ui.row().classes("w-full gap-4"):
                ctrl.register(
                    "lvl_algo",
                    ui.select(
                        tr_options(t, LVL_ALGO_OPTION_LABEL_KEYS),
                        value=normalize_lvl_algo_value(get_val("lvl_algo", LVL_ALGO_MEDIAN), t),
                        label=t("lvl_algo"),
                    ).props("dense outlined").classes("flex-1"),
                )
                ctrl.register(
                    "gain",
                    ui.number(
                        label=t("gain"),
                        value=float(get_val("gain", 0.0) or 0.0),
                        format="%.1f",
                    ).props("dense outlined").classes("flex-1"),
                )

            with ui.row().classes("w-full gap-4"):
                ctrl.register(
                    "lvl_min",
                    ui.number(
                        label=t("lvl_min"),
                        value=float(get_val("lvl_min", 500.0) or 500.0),
                        format="%.0f",
                    ).props("dense outlined").classes("flex-1"),
                )
                ctrl.register(
                    "lvl_max",
                    ui.number(
                        label=t("lvl_max"),
                        value=float(get_val("lvl_max", 2000.0) or 2000.0),
                        format="%.0f",
                    ).props("dense outlined").classes("flex-1"),
                )

            with ui.row().classes("w-full gap-4 items-start"):
                ctrl.register(
                    "lvl_mode",
                    ui.select(
                        options=tr_options(t, LVL_MODE_OPTION_LABEL_KEYS),
                        value=normalize_lvl_mode_value(get_val("lvl_mode", LVL_MODE_AUTO), t),
                        label=t("lvl_mode"),
                    ).props("dense outlined").classes("flex-1"),
                )
                lvl_manual_col = ui.column().classes("flex-1 gap-1")
                ctrl.register_container("lvl_manual_scope", lvl_manual_col)
                with lvl_manual_col:
                    with ui.row().classes("w-full gap-2 items-end"):
                        ctrl.register(
                            "lvl_manual_db",
                            ui.number(
                                label=t("lvl_target_db"),
                                value=float(get_val("lvl_manual_db", 0.0) or 0.0),
                                format="%.1f",
                            ).props("dense outlined step=0.1").classes("flex-1"),
                        )
                        ui.button(
                            "+",
                            on_click=lambda: _step_manual_target(+0.1),
                        ).props('color="secondary" outline').style("min-width:34px;")
                        ui.button(
                            "-",
                            on_click=lambda: _step_manual_target(-0.1),
                        ).props('color="secondary" outline').style("min-width:34px;")
                    with ui.row().classes("w-full gap-2 items-end"):
                        ctrl.register(
                            "manual_target_tilt_db_per_oct",
                            ui.number(
                                label=t("manual_target_tilt"),
                                value=float(get_val("manual_target_tilt_db_per_oct", 0.0) or 0.0),
                                format="%.1f",
                            ).props("dense outlined step=0.1").classes("flex-1"),
                        )
                        ui.button(
                            "+",
                            on_click=lambda: _step_manual_target_tilt(+0.1),
                        ).props('color="secondary" outline').style("min-width:34px;")
                        ui.button(
                            "-",
                            on_click=lambda: _step_manual_target_tilt(-0.1),
                        ).props('color="secondary" outline').style("min-width:34px;")
                    ui.label(t("lvl_manual_help")).classes("text-xs text-gray-400")
                    ui.label(t("manual_target_tilt_help")).classes("text-xs text-gray-400")
                    ui.label(t("lvl_manual_bias_hint")).classes("text-xs text-gray-400")
                    ui.label(t("lvl_manual_drag_hint_curve")).classes("text-xs text-gray-400")
                    ui.label(t("manual_target_tilt_drag_hint")).classes("text-xs text-gray-400")
                lvl_manual_col.set_visibility(False)

            _initial_mode = str(get_val("mode", "BASIC") or "BASIC").upper()
            _initial_lvl_mode = normalize_lvl_mode_value(get_val("lvl_mode", LVL_MODE_AUTO), t)
            output_tilt_col = ui.column().classes("w-full gap-1")
            ctrl.register_container("output_tilt_scope", output_tilt_col)
            with output_tilt_col:
                ui.label(t("output_tilt")).classes("text-sm font-semibold mt-1")
                with ui.column().classes("w-full gap-1"):
                    ctrl.register(
                        "output_tilt_source",
                        ui.radio(
                            tr_options(t, OUTPUT_TILT_SOURCE_OPTION_LABEL_KEYS),
                            value=normalize_output_tilt_source_value(
                                get_val("output_tilt_source", OUTPUT_TILT_SOURCE_OFF),
                                t,
                            ),
                        ).classes("w-full"),
                    )
                ui.label(t("output_tilt_help")).classes("text-xs text-gray-400")
            output_tilt_col.set_visibility(_initial_mode == "ADVANCED" and _initial_lvl_mode == LVL_MODE_MANUAL)

        with section_card(title=t("magnitude_correction_limits")):
            ctrl.register(
                "mag_correct",
                ui.checkbox(t("enable_corr"), value=bool(get_val("mag_correct", True))),
            )

            with ui.row().classes("w-full gap-4"):
                ctrl.register(
                    "mag_c_min",
                    ui.number(
                        label=t("min_freq"),
                        value=float(get_val("mag_c_min", 10.0) or 10.0),
                        format="%.1f",
                    ).props("dense outlined").classes("flex-1"),
                )
                ctrl.register(
                    "mag_c_max",
                    ui.number(
                        label=t("max_freq"),
                        value=float(get_val("mag_c_max", 200.0) or 200.0),
                        format="%.1f",
                    ).props("dense outlined").classes("flex-1"),
                )

            ctrl.register(
                "max_boost",
                ui.number(
                    label=t("max_boost"),
                    value=float(get_val("max_boost", 5.0) or 5.0),
                    format="%.1f",
                ).props("dense outlined").classes("w-full"),
            )

def refresh_target_preview() -> None:
    """Regenerate the target curve preview plot (NiceGUI version).

    Reads values from ng_controls instead of PyWebIO pin.
    """
    global _TARGET_PREVIEW_DRAG_ACTIVE, _TARGET_PREVIEW_DRAG_BASE_POINTS, _TARGET_PREVIEW_TILT_HANDLE_POINTS
    global _TARGET_PREVIEW_PLOT

    preview_col = ctrl.get_container("target_preview_scope")
    if preview_col is None:
        return
    _render_target_decay_hint()
    _render_target_preview_metadata()

    fig, drag_base_points, tilt_handle_points = _build_target_preview_fig()
    _TARGET_PREVIEW_DRAG_BASE_POINTS = drag_base_points
    _TARGET_PREVIEW_TILT_HANDLE_POINTS = tilt_handle_points
    if fig is None:
        preview_col.clear()
        _TARGET_PREVIEW_PLOT = None
        _TARGET_PREVIEW_DRAG_ACTIVE = False
        return

    if _TARGET_PREVIEW_PLOT is None:
        _TARGET_PREVIEW_PLOT = _mount_target_preview_plot(preview_col, fig)
    else:
        try:
            _TARGET_PREVIEW_PLOT.update_figure(fig)
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
            _TARGET_PREVIEW_PLOT = _mount_target_preview_plot(preview_col, fig)
    _TARGET_PREVIEW_DRAG_ACTIVE = False

def _mount_target_preview_plot(preview_col, fig):
    from nicegui import ui  # noqa: PLC0415

    preview_col.clear()
    with preview_col:
        plot = ui.plotly(fig).classes("w-full")
        plot.on("plotly_relayout", _on_target_preview_relayout)
    return plot

def _schedule_target_preview_refresh(delay_s: float = 0.10) -> None:
    global _TARGET_PREVIEW_REFRESH_TOKEN

    preview_col = ctrl.get_container("target_preview_scope")
    if preview_col is None:
        return

    _TARGET_PREVIEW_REFRESH_TOKEN += 1
    token = _TARGET_PREVIEW_REFRESH_TOKEN

    from nicegui import ui  # noqa: PLC0415

    def _run() -> None:
        global _TARGET_PREVIEW_DRAG_ACTIVE

        if token != _TARGET_PREVIEW_REFRESH_TOKEN:
            return
        try:
            refresh_target_preview()
        finally:
            _TARGET_PREVIEW_DRAG_ACTIVE = False

    with preview_col:
        ui.timer(delay_s, _run, once=True, immediate=False)

def _on_target_preview_relayout(e) -> None:
    global _TARGET_PREVIEW_DRAG_ACTIVE

    app_mode = str(ctrl.value("mode", "BASIC") or "BASIC").upper()
    lvl_mode = normalize_lvl_mode_value(ctrl.value("lvl_mode", LVL_MODE_AUTO))
    if app_mode in ("BASIC", "AUTO") or lvl_mode != LVL_MODE_MANUAL:
        return

    payload = getattr(e, "args", None)
    if not isinstance(payload, dict):
        return

    new_tilt = extract_target_tilt_from_shape_relayout(
        payload,
        _TARGET_PREVIEW_TILT_HANDLE_POINTS,
    )
    if new_tilt is not None:
        _TARGET_PREVIEW_DRAG_ACTIVE = True
        ctrl.set_value("manual_target_tilt_db_per_oct", new_tilt, emit=False)
        _schedule_target_preview_refresh()
        return

    new_db = extract_vertical_shift_from_shape_relayout(
        payload,
        _TARGET_PREVIEW_DRAG_BASE_POINTS,
    )
    if new_db is None:
        return

    _TARGET_PREVIEW_DRAG_ACTIVE = True
    ctrl.set_value("lvl_manual_db", new_db, emit=False)
    _schedule_target_preview_refresh()

def _build_target_preview_fig():  # noqa: C901 - target preview figure is assembled from many UI states
    """Build the target curve preview Plotly figure from current ctrl values.

    Returns a Plotly dict plus the base points of the draggable target curve and tilt handle.
    """
    try:
        import math  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        import plotly.graph_objects as go  # noqa: PLC0415
        from ...auto_mode.shared import _auto_goal_forced_level_window  # noqa: PLC0415
        from ...dsp.bass_integration import (  # noqa: PLC0415
            build_combined_sub_transfer,
            normalize_sub_combine_mode,
            sum_complex_responses,
        )
        from ...io.generated_measurement_source import generated_source_matches_upload, parse_generated_source  # noqa: PLC0415
        from ...dsp.smoothing import psychoacoustic_smoothing as _psycho_smooth  # noqa: PLC0415
        from ...application.house_curve_service import (  # noqa: PLC0415
            _normalize_hc_mode_key,
            load_house_curve,
            load_target_curve,
        )
        from ..target_preview_cache import (  # noqa: PLC0415
            load_path_measurement_curve,
            load_path_measurement_transfer,
            load_upload_measurement_curve,
            load_upload_measurement_transfer,
        )

        global _TARGET_PREVIEW_DRAG_ACTIVE

        def _cv(name, default=None):
            return ctrl.value(name, default)

        def _to_float(v, default):
            try:
                x = float(v)
                if math.isfinite(x):
                    return x
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
                logger.exception("float parse in target preview")
            return float(default)

        def _smooth_for_preview(freq_axis, m):
            try:
                return _psycho_smooth(freq_axis, np.asarray(m, dtype=float))
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
                return m

        # --- collect ctrl values ---
        hc_mode_raw = str(_cv("hc_mode", "Harman6") or "Harman6")
        hc_mode = str(_normalize_hc_mode_key(hc_mode_raw))
        hc_file = _cv("hc_custom_file", None)
        lvl_min = _to_float(_cv("lvl_min", 500.0), 500.0)
        lvl_max = _to_float(_cv("lvl_max", 2000.0), 2000.0)
        mag_c_min = _to_float(_cv("mag_c_min", 10.0), 10.0)
        mag_c_max = _to_float(_cv("mag_c_max", 200.0), 200.0)
        auto_goal = str(_cv("auto_goal", "balanced") or "balanced")
        app_mode = str(_cv("mode", "BASIC") or "BASIC").upper()
        lvl_mode = normalize_lvl_mode_value(_cv("lvl_mode", LVL_MODE_AUTO))
        if app_mode in ("BASIC", "AUTO"):
            lvl_mode = LVL_MODE_AUTO
        is_manual_level = lvl_mode == LVL_MODE_MANUAL
        lvl_manual_db = _to_float(_cv("lvl_manual_db", 0.0), 0.0)
        manual_target_tilt_db_per_oct = _to_float(_cv("manual_target_tilt_db_per_oct", 0.0), 0.0)
        preview_level_shift_db = lvl_manual_db if is_manual_level else 0.0
        preview_manual_target_tilt_db_per_oct = manual_target_tilt_db_per_oct if is_manual_level else 0.0
        pre_ms = _to_float(_cv("ir_window_left", 85.0), 85.0)
        post_ms = _to_float(_cv("ir_window_right") or _cv("ir_window", 500.0), 500.0)
        smoothing_level = 96
        bass_integration_enabled = bool(_cv("bass_integration_enable", False))
        bass_integration_sub_combine_mode = normalize_sub_combine_mode(
            _cv("bass_integration_sub_combine_mode", "average")
        )

        freq_axis = np.logspace(math.log10(10.0), math.log10(20000.0), 400)

        # --- build target curve ---
        target_curve = None
        if hc_mode == "Upload" and isinstance(hc_file, dict) and hc_file.get("content"):
            try:
                tf_f, tf_m = load_target_curve(hc_file["content"])
                if tf_f is not None and tf_m is not None:
                    tf_f = np.asarray(tf_f, dtype=float)
                    tf_m = np.asarray(tf_m, dtype=float)
                    if tf_f.size >= 2 and tf_m.size == tf_f.size:
                        target_curve = np.interp(freq_axis, tf_f, tf_m, left=tf_m[0], right=tf_m[-1])
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
                logger.exception("custom target curve load")

        if target_curve is None:
            hc_f, hc_m, _ = load_house_curve({"hc_mode": hc_mode})
            if hc_f is not None and hc_m is not None:
                hc_f = np.asarray(hc_f, dtype=float)
                hc_m = np.asarray(hc_m, dtype=float)
                target_curve = np.interp(freq_axis, hc_f, hc_m, left=hc_m[0], right=hc_m[-1])

        if target_curve is None:
            return None, [], []

        # AUTO mode forced level window
        if app_mode in ("BASIC", "AUTO"):
            forced = _auto_goal_forced_level_window(auto_goal)
            if forced is not None:
                lvl_min, lvl_max = float(forced[0]), float(forced[1])

        # --- speaker measurements ---
        def _align(m_curve, t_curve, fa, fmin, fmax):
            try:
                mask = (fa >= fmin) & (fa <= fmax)
                if not mask.any():
                    return m_curve
                diff = np.nanmedian(t_curve[mask] - np.asarray(m_curve, dtype=float)[mask])
                return np.asarray(m_curve, dtype=float) + diff
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
                return m_curve

        speaker_interp = {}
        speaker_components = {}
        _speaker_align_offset: dict[str, float] = {}
        _harmonic_sources: dict[str, tuple] = {}
        if bass_integration_enabled:
            def _load_transfer(up_key: str, path_key: str, label: str):
                tr = load_upload_measurement_transfer(
                    _cv(up_key, None),
                    pre_ms=pre_ms,
                    post_ms=post_ms,
                    smoothing_level=smoothing_level,
                    label=label,
                )
                if tr is None:
                    tr = load_path_measurement_transfer(
                        _cv(path_key, ""),
                        pre_ms=pre_ms,
                        post_ms=post_ms,
                        smoothing_level=smoothing_level,
                        label=label,
                    )
                return tr

            bi_slots = {
                "L_main": _load_transfer("file_l_main", "local_path_l_main", "L main only"),
                "R_main": _load_transfer("file_r_main", "local_path_r_main", "R main only"),
                "L_sub": _load_transfer("file_l_sub", "local_path_l_sub", "L sub only"),
                "R_sub": _load_transfer("file_r_sub", "local_path_r_sub", "R sub only"),
            }

            def _interp_transfer(tr):
                if tr is None:
                    return None
                try:
                    ff = np.asarray(tr.freqs_hz, dtype=float)
                    mm = np.asarray(tr.mag_db, dtype=float)
                    if ff.size < 8 or mm.size != ff.size:
                        return None
                    return np.interp(freq_axis, ff, mm, left=mm[0], right=mm[-1])
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

            for ch in ("L", "R"):
                main_tr = bi_slots.get(f"{ch}_main")
                l_sub_tr = bi_slots.get("L_sub")
                r_sub_tr = bi_slots.get("R_sub")
                active_subs = [tr for tr in (l_sub_tr, r_sub_tr) if tr is not None]
                if main_tr is None or not active_subs:
                    continue
                combined_sub_tr, _combine_diag = build_combined_sub_transfer(
                    main_tr,
                    *active_subs,
                    mode=bass_integration_sub_combine_mode,
                    label=f"{ch} combined sub predicted",
                )
                total_tr = sum_complex_responses(
                    main_tr,
                    combined_sub_tr,
                    label=f"{ch} total predicted",
                )
                total_curve = _interp_transfer(total_tr)
                if total_curve is None:
                    continue
                total_aligned = _align(total_curve, target_curve, freq_axis, lvl_min, lvl_max)
                _speaker_align_offset[ch] = float(np.nanmedian(total_aligned - total_curve)) if np.isfinite(total_curve).any() else 0.0
                speaker_interp[ch] = _smooth_for_preview(freq_axis, total_aligned)

                main_curve = _interp_transfer(main_tr)
                if main_curve is not None:
                    speaker_components[f"{ch}_main"] = _smooth_for_preview(
                        freq_axis, main_curve + _speaker_align_offset[ch]
                    )
                _individual_sub_tr = bi_slots.get(f"{ch}_sub")
                _individual_sub_curve = _interp_transfer(_individual_sub_tr) if _individual_sub_tr is not None else None
                if _individual_sub_curve is not None:
                    speaker_components[f"{ch}_sub"] = _smooth_for_preview(
                        freq_axis, _individual_sub_curve + _speaker_align_offset[ch]
                    )
        else:
            generated_sources = {
                "L": _cv("generated_measurement_l", None),
                "R": _cv("generated_measurement_r", None),
            }
            for ch, up_key, path_key in (("L", "file_l", "local_path_l"), ("R", "file_r", "local_path_r")):
                ff = None
                mm = None
                generated = generated_sources.get(ch, None)
                upload_value = _cv(up_key, None)
                if generated_source_matches_upload(generated, upload_value):
                    ff_gen, mm_gen, _pp_gen, _raw_ir, _raw_ir_fs, _h_freq, _h_mags = parse_generated_source(
                        generated,
                        pre_ms=pre_ms,
                        post_ms=post_ms,
                        smoothing_level=smoothing_level,
                    )
                    if ff_gen is not None and mm_gen is not None:
                        ff = ff_gen
                        mm = mm_gen
                        if _h_freq is not None and _h_mags:
                            _harmonic_sources[ch] = (_h_freq, _h_mags)
                if ff is None:
                    ff, mm = load_upload_measurement_curve(
                        upload_value,
                        pre_ms=pre_ms,
                        post_ms=post_ms,
                        smoothing_level=smoothing_level,
                    )
                if ff is None:
                    ff, mm = load_path_measurement_curve(
                        _cv(path_key, ""),
                        pre_ms=pre_ms,
                        post_ms=post_ms,
                        smoothing_level=smoothing_level,
                    )
                if ff is not None and mm is not None:
                    m_interp = np.interp(freq_axis, ff, mm, left=mm[0], right=mm[-1])
                    m_aligned = _align(m_interp, target_curve, freq_axis, lvl_min, lvl_max)
                    _speaker_align_offset[ch] = float(np.nanmedian(m_aligned - m_interp))
                    speaker_interp[ch] = _smooth_for_preview(freq_axis, m_aligned)

        from ...io.measurements_loader import _try_load_harmonic_sidecar  # noqa: PLC0415
        for _hch in ("L", "R"):
            if _hch in _harmonic_sources:
                continue
            _hlp = str(_cv(
                ("local_path_l_main" if _hch == "L" else "local_path_r_main")
                if bass_integration_enabled else
                ("local_path_l" if _hch == "L" else "local_path_r"),
                "",
            ) or "")
            if not _hlp:
                continue
            _hf, _hm = _try_load_harmonic_sidecar(_hlp)
            if _hf is not None and _hm:
                _harmonic_sources[_hch] = (_hf, _hm)

        target_curve_display = apply_manual_target_preview_adjustments(
            freq_axis,
            target_curve,
            preview_level_shift_db,
            preview_manual_target_tilt_db_per_oct,
        )
        target_curve_path = build_target_curve_path(freq_axis, target_curve_display)
        drag_base_points = []
        tilt_handle_points = []
        if is_manual_level and target_curve_path:
            drag_base_points = [
                (float(x), float(y))
                for x, y in zip(freq_axis, target_curve_display)
                if math.isfinite(float(x)) and math.isfinite(float(y))
            ]
            tilt_handle_y = float(np.interp(16000.0, freq_axis, target_curve_display))
            tilt_handle_points = parse_svg_path_points(build_tilt_handle_path(tilt_handle_y))

        # --- build figure ---
        fig = go.Figure()
        fig.add_trace(build_level_window_trace(lvl_min, lvl_max))
        fig.add_trace(build_vertical_marker_trace(mag_c_min))
        fig.add_trace(build_vertical_marker_trace(mag_c_max))

        target_trace = dict(
            x=freq_axis,
            y=target_curve_display,
            mode="lines",
            name=f"Target ({hc_mode_raw})",
            line=dict(color="#4caf50", width=2.0),
        )
        if is_manual_level:
            target_trace["opacity"] = 0.45
            target_trace["hoverinfo"] = "skip"
        fig.add_trace(go.Scatter(**target_trace))
        if is_manual_level:
            fig.add_trace(go.Scatter(
                x=[freq_axis[0], freq_axis[-1]],
                y=[lvl_manual_db, lvl_manual_db],
                mode="lines",
                name=f"Manual level ({lvl_manual_db:+.1f} dB)",
                line=dict(color="rgba(15,23,42,0.55)", width=1.2, dash="dot"),
                hoverinfo="skip",
                visible="legendonly",
            ))
            fig.add_trace(go.Scatter(
                x=[freq_axis[0], freq_axis[-1]],
                y=[0.0, 0.0],
                mode="lines",
                name=f"Manual tilt ({manual_target_tilt_db_per_oct:+.1f} dB/oct @ 1 kHz)",
                line=dict(color="#f4a261", width=1.8, dash="dot"),
                hoverinfo="skip",
                visible="legendonly",
            ))
        if "L" in speaker_interp:
            fig.add_trace(go.Scatter(
                x=freq_axis, y=speaker_interp["L"], mode="lines",
                name="Predicted L total" if bass_integration_enabled else "Speaker L",
                line=dict(color="rgba(102,187,255,0.55)", width=1.2),
            ))
        if "R" in speaker_interp:
            fig.add_trace(go.Scatter(
                x=freq_axis, y=speaker_interp["R"], mode="lines",
                name="Predicted R total" if bass_integration_enabled else "Speaker R",
                line=dict(color="rgba(255,167,102,0.55)", width=1.2),
            ))
        for comp_key, trace_name, color in (
            ("L_main", "L main only", "rgba(102,187,255,0.35)"),
            ("L_sub", "L sub only", "rgba(30,64,175,0.35)"),
            ("R_main", "R main only", "rgba(255,167,102,0.35)"),
            ("R_sub", "R sub only", "rgba(185,28,28,0.35)"),
        ):
            if comp_key in speaker_components:
                fig.add_trace(go.Scatter(
                    x=freq_axis,
                    y=speaker_components[comp_key],
                    mode="lines",
                    name=trace_name,
                    line=dict(color=color, width=1.0, dash="dot"),
                    visible="legendonly",
                ))
        if len(speaker_interp) > 0:
            avg = np.mean(np.vstack([speaker_interp[k] for k in sorted(speaker_interp)]), axis=0)
            fig.add_trace(go.Scatter(
                x=freq_axis, y=avg, mode="lines",
                name="Predicted avg" if bass_integration_enabled else "Speaker avg",
                line=dict(color="#dc2626", width=2.0),
            ))
        _HARM_CLRS = {
            2: {"L": "rgba(40,120,255,0.65)", "R": "rgba(255,110,40,0.65)"},
            3: {"L": "rgba(0,80,200,0.60)",   "R": "rgba(200,60,10,0.60)"},
            4: {"L": "rgba(0,50,160,0.55)",   "R": "rgba(160,35,0,0.55)"},
            5: {"L": "rgba(0,20,120,0.50)",   "R": "rgba(120,15,0,0.50)"},
        }
        for _hch in ("L", "R"):
            if _hch not in _harmonic_sources:
                continue
            _hf_arr = np.asarray(_harmonic_sources[_hch][0], dtype=float)
            _hm_dict = _harmonic_sources[_hch][1]
            _h_off = _speaker_align_offset.get(_hch, 0.0)
            for _hord, _harr in sorted(_hm_dict.items()):
                _ha = np.asarray(_harr, dtype=float)
                _hmsk = (_hf_arr > 10.0) & np.isfinite(_hf_arr) & np.isfinite(_ha)
                if _hmsk.sum() < 4:
                    continue
                _hi = np.interp(freq_axis, _hf_arr[_hmsk], _ha[_hmsk],
                                left=_ha[_hmsk][0], right=_ha[_hmsk][-1])
                _hcolor = _HARM_CLRS.get(_hord, {}).get(_hch, "rgba(128,128,128,0.50)")
                fig.add_trace(go.Scatter(
                    x=freq_axis,
                    y=_hi + _h_off,
                    mode="lines",
                    name=f"H{_hord} {_hch}",
                    line=dict(color=_hcolor, width=1.0, dash="dot"),
                    visible="legendonly",
                ))
        fig.update_xaxes(
            type="log",
            title_text="Hz",
            range=[math.log10(10.0), math.log10(20000.0)],
            fixedrange=True,
            gridcolor="rgba(15,23,42,0.10)",
            linecolor="rgba(15,23,42,0.22)",
            zerolinecolor="rgba(15,23,42,0.10)",
        )
        _y_sources = [target_curve_display, *speaker_interp.values()]
        _all_y = [v for arr in _y_sources for v in arr if math.isfinite(float(v))]
        _y_max = (max(_all_y) + 3.0) if _all_y else 20.0
        fig.update_yaxes(
            title_text="dB",
            range=[-20.0, max(-19.0, _y_max)],
            fixedrange=False,
            gridcolor="rgba(15,23,42,0.10)",
            linecolor="rgba(15,23,42,0.22)",
            zerolinecolor="rgba(15,23,42,0.10)",
        )
        fig.update_layout(height=320, margin=dict(l=40, r=20, t=30, b=35),
                          showlegend=True, template="plotly_white",
                          paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                          font=dict(color="#1f2937"),
                          uirevision="target_preview_lock")
        if is_manual_level and target_curve_path:
            drag_shape = build_draggable_target_shape(freq_axis, target_curve_display)
            drag_shape["path"] = target_curve_path
            tilt_handle_shape = build_draggable_tilt_handle_shape(float(np.interp(16000.0, freq_axis, target_curve_display)))
            fig.update_layout(shapes=[drag_shape, tilt_handle_shape])

        fig_dict = fig.to_plotly_json()
        layout = fig_dict.setdefault("layout", {})
        layout["uirevision"] = "target_preview_lock"
        layout["editrevision"] = (
            f"target_preview_manual_{_TARGET_PREVIEW_REFRESH_TOKEN}_{int(_TARGET_PREVIEW_DRAG_ACTIVE)}"
            if is_manual_level else
            "target_preview_static"
        )
        fig_dict["config"] = {
            "responsive": True,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
            "editable": False,
            "edits": {"shapePosition": bool(is_manual_level)},
            "doubleClick": False,
            "scrollZoom": False,
        }
        return fig_dict, drag_base_points, tilt_handle_points

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
        import logging  # noqa: PLC0415
        logging.getLogger("DecayCore").warning("_build_target_preview_fig failed", exc_info=True)
        return None, [], []


__all__ = [
    '_normalize_harmonic_plot_source',
    '_build_target_preview_metadata_payload',
    '_build_target_decay_hint_payload',
    '_build_target_preview_rt60_fig',
    '_build_target_preview_harmonics_fig',
    '_build_target_preview_harmonic_risk_fig',
    '_render_target_preview_metadata',
    '_step_manual_target',
    '_step_manual_target_tilt',
    'build_target_tab',
    'refresh_target_preview',
    '_mount_target_preview_plot',
    '_schedule_target_preview_refresh',
    '_on_target_preview_relayout',
    '_build_target_preview_fig',
]


def _link_sibling_exports() -> None:
    import importlib
    package = __package__
    for module_name in ['target_tab_builder']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_link_sibling_exports()
