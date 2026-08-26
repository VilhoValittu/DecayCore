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

import hashlib
import logging
import sys

logger = logging.getLogger("DecayCore")

import numpy as np

from ...measurement.devices import (
    check_measurement_audio_backend,
    is_measurement_device_wasapi,
)
from ...measurement.models import MeasurementBundle, MeasurementSessionAggregate
from ...measurement.routing import (
    DEFAULT_LINUX_SUBWOOFER_OUTPUT_CHANNEL_INDEX,
    measurement_role_output_channel_count,
)
from ..plot_common import _view_mags_for_plot

_SESSION_PREVIEW_CHANNEL_ORDER = ("left", "right", "sub1", "sub2")


def _build_upload_payload(*, filename: str, content: bytes, mime_type: str = "") -> dict[str, object]:
    content_bytes = bytes(content or b"")
    return {
        "filename": str(filename or ""),
        "content": content_bytes,
        "mime_type": str(mime_type or ""),
        "size_bytes": len(content_bytes),
        "content_sha256": hashlib.sha256(content_bytes).hexdigest(),
    }


def _device_option_label(device, *, samplerate_hz: int | None = None) -> str:
    caps: list[str] = []
    if int(device.max_input_channels) > 0:
        caps.append(f"In {int(device.max_input_channels)}")
    if int(device.max_output_channels) > 0:
        caps.append(f"Out {int(device.max_output_channels)}")
    hostapi_name = str(getattr(device, "hostapi_name", "") or "").strip()
    if hostapi_name:
        caps.append(hostapi_name)
    if samplerate_hz is not None and int(samplerate_hz) > 0:
        caps.append(f"{int(samplerate_hz)} Hz")
    elif device.default_samplerates:
        caps.append(f"{int(device.default_samplerates[0])} Hz")
    suffix = f" ({', '.join(caps)})" if caps else ""
    return f"{int(device.index)}: {device.name}{suffix}"


def _measurement_use_wasapi_value(value) -> bool:
    return bool(value) if sys.platform == "win32" else False


def _measurement_sub_output_channel_default() -> int:
    return int(DEFAULT_LINUX_SUBWOOFER_OUTPUT_CHANNEL_INDEX)


def _measurement_sub_output_channel_value(value) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return _measurement_sub_output_channel_default()
    return max(0, parsed)


def _measurement_sub_output_channel_visible(role_value: str | None) -> bool:
    if sys.platform == "win32":
        return False
    role = str(role_value or "").strip().lower()
    return role == "sub"


def _measurement_output_channel_for_role(
    role_value: str | None,
    *,
    sub_output_channel: int | None = None,
) -> int:
    role = str(role_value or "").strip().lower()
    if role != "sub" or sys.platform == "win32":
        return 0
    return _measurement_sub_output_channel_value(sub_output_channel)


def _measurement_required_output_channels(
    role_value: str | None,
    *,
    sub_output_channel: int | None = None,
) -> int:
    resolved_sub_channel = _measurement_output_channel_for_role(
        role_value,
        sub_output_channel=sub_output_channel,
    )
    return int(
        measurement_role_output_channel_count(
            role_value,
            sub_output_channel=resolved_sub_channel,
        )
    )


def _pick_measurement_device_value(
    current_value,
    saved_default,
    options: dict[object, str],
):
    def _coerce(candidate):
        if candidate in options:
            return candidate
        try:
            legacy_prefix = f"{int(candidate)}: "
        except (TypeError, ValueError, OverflowError):
            return None
        return next((key for key, label in options.items() if str(label).startswith(legacy_prefix)), None)

    current = _coerce(current_value)
    if current is not None:
        return current
    default = _coerce(saved_default)
    if default is not None:
        return default
    if options:
        return next(iter(options))
    return None


def _filter_measurement_devices_for_wasapi(devices, *, enabled: bool):
    device_list = list(devices or [])
    if not bool(enabled):
        return device_list
    return [device for device in device_list if is_measurement_device_wasapi(device)]


def _measurement_audio_backend_message() -> str | None:
    ok, reason = check_measurement_audio_backend()
    if ok:
        return None
    message = str(reason or "").strip()
    return message or "Measurement audio backend is unavailable."


def _preview_magnitude_for_plot(freq_hz, magnitude_db):
    freq = np.asarray(freq_hz, dtype=float)
    mag = np.asarray(magnitude_db, dtype=float)
    if freq.size == 0 or mag.size == 0 or freq.size != mag.size:
        return mag
    try:
        return _view_mags_for_plot(
            freq,
            mag,
            plot_smoothing_level="Psychoacoustic",
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
    ):
        return mag


def _build_preview_figure(bundle: MeasurementBundle):
    import plotly.graph_objects as go  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415

    recorded = np.asarray(bundle.capture.recorded_signal, dtype=float).reshape(-1)
    ir = np.asarray(bundle.ir.impulse_response, dtype=float).reshape(-1)
    freq = np.asarray(bundle.analysis_freq_hz if bundle.analysis_freq_hz is not None else [], dtype=float)
    mag = np.asarray(bundle.analysis_magnitude_db if bundle.analysis_magnitude_db is not None else [], dtype=float)
    cal_mag = (
        None
        if bundle.calibrated_analysis_magnitude_db is None
        else np.asarray(bundle.calibrated_analysis_magnitude_db, dtype=float)
    )
    preview_mag = _preview_magnitude_for_plot(freq, mag)
    preview_cal_mag = None if cal_mag is None else _preview_magnitude_for_plot(freq, cal_mag)

    rec_step = max(1, recorded.size // 4000) if recorded.size else 1
    ir_step = max(1, ir.size // 4000) if ir.size else 1

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=("Recorded Signal", "Impulse Response", "Magnitude Preview"),
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(0, recorded[::rec_step].size * rec_step, rec_step),
            y=recorded[::rec_step],
            mode="lines",
            name="Recorded",
            line=dict(color="#3a72c8", width=1.2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(0, ir[::ir_step].size * ir_step, ir_step),
            y=ir[::ir_step],
            mode="lines",
            name="IR",
            line=dict(color="#2f855a", width=1.2),
        ),
        row=2,
        col=1,
    )
    if freq.size > 1 and preview_mag.size == freq.size:
        fig.add_trace(
            go.Scatter(
                x=freq,
                y=preview_mag,
                mode="lines",
                name="Magnitude (DecayCore Reference)",
                line=dict(color="#4f46e5", width=1.6),
            ),
            row=3,
            col=1,
        )
        if preview_cal_mag is not None and preview_cal_mag.size == freq.size:
            fig.add_trace(
                go.Scatter(
                    x=freq,
                    y=preview_cal_mag,
                    mode="lines",
                    name="Calibrated magnitude (DecayCore Reference)",
                    line=dict(color="#dc2626", width=1.6),
                ),
                row=3,
                col=1,
            )

    _HARMONIC_COLORS = {2: "#f97316", 3: "#ec4899", 4: "#8b5cf6", 5: "#6b7280"}
    _h_freq = np.asarray(bundle.harmonic_freq_hz if bundle.harmonic_freq_hz is not None else [], dtype=float)
    _h_mags: dict = bundle.harmonic_magnitudes_db or {}
    for _hn, _hcolor in _HARMONIC_COLORS.items():
        if _hn not in _h_mags:
            continue
        _hm = np.asarray(_h_mags[_hn], dtype=float)
        _preview_hm = _preview_magnitude_for_plot(_h_freq, _hm)
        if _h_freq.size > 1 and _preview_hm.size == _h_freq.size:
            fig.add_trace(
                go.Scatter(
                    x=_h_freq,
                    y=_preview_hm,
                    mode="lines",
                    name=f"H{_hn} (DecayCore Reference)",
                    line=dict(color=_hcolor, width=1.0, dash="dash"),
                ),
                row=3,
                col=1,
            )

    fig.update_xaxes(title_text="Samples", row=1, col=1)
    fig.update_xaxes(title_text="Samples", row=2, col=1)
    fig.update_xaxes(type="log", title_text="Hz", row=3, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)
    fig.update_yaxes(title_text="dB", row=3, col=1)
    fig.update_layout(
        height=760,
        margin=dict(l=40, r=20, t=50, b=40),
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#1f2937"),
        showlegend=True,
    )
    return fig


def _session_preview_bundles(session: MeasurementSessionAggregate) -> dict[str, MeasurementBundle]:
    bundles: dict[str, MeasurementBundle] = {}
    if session.final_left_bundle is not None:
        bundles["left"] = session.final_left_bundle
    if session.final_right_bundle is not None:
        bundles["right"] = session.final_right_bundle
    if session.final_sub1_bundle is not None:
        bundles["sub1"] = session.final_sub1_bundle
    if session.final_sub2_bundle is not None:
        bundles["sub2"] = session.final_sub2_bundle
    return bundles


__all__ = [
    "_build_upload_payload",
    "_device_option_label",
    "_measurement_use_wasapi_value",
    "_measurement_sub_output_channel_default",
    "_measurement_sub_output_channel_value",
    "_measurement_sub_output_channel_visible",
    "_measurement_output_channel_for_role",
    "_measurement_required_output_channels",
    "_pick_measurement_device_value",
    "_filter_measurement_devices_for_wasapi",
    "_measurement_audio_backend_message",
    "_preview_magnitude_for_plot",
    "_build_preview_figure",
    "_session_preview_bundles",
    "sys",
    "check_measurement_audio_backend",
    "_view_mags_for_plot",
]
