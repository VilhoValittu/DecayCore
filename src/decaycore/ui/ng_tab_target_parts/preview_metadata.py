# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Target tab preview metadata: RT60 / harmonics payloads, figures and renderers."""
from __future__ import annotations

from .. import ng_controls as ctrl
from ...common.measurement_features import build_harmonic_boost_risk_curve, build_target_decay_hint
from .preview_state import STATE

def _decay_hint_status_color(status: str) -> str:
    return {
        "ok": "#2f855a",
        "caution": "#b7791f",
        "strong": "#c53030",
        "unavailable": "#4a5568",
    }.get(str(status or "").strip().lower(), "#4a5568")


def _target_hint_translate(key: str) -> str:
    from ...resources.i8n.decaycore_i18n import t as resource_t  # noqa: PLC0415

    translator = STATE.translate or resource_t
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
    from ...io.measurements_loader_parts import _try_load_harmonic_sidecar, _try_load_rt60_sidecar  # noqa: PLC0415
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
                
                    '<span style="display:inline-block;padding:4px 10px;'
                    'border-radius:9999px;font-size:0.72rem;font-weight:700;'
                    f'background:{badge_color};color:#ffffff;">{_target_hint_translate(badge_key)}</span>'
                
            )
            ui.label(_target_hint_translate(summary_key)).classes("text-xs cf-target-hint-summary")
        for key in detail_keys:
            ui.label(_target_hint_translate(key)).classes("text-xs cf-target-hint-detail")
