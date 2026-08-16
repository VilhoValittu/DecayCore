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
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .measurement_tab_helpers import (
    _build_preview_figure,
    _build_upload_payload,
    _device_option_label,
    _measurement_audio_backend_message,
    _measurement_output_channel_for_role,
    _measurement_required_output_channels,
    _measurement_sub_output_channel_default,
    _measurement_sub_output_channel_value,
    _measurement_sub_output_channel_visible,
    _measurement_use_wasapi_value,
    _pick_measurement_device_value,
    _session_preview_bundles,
)

logger = logging.getLogger("DecayCore")

import numpy as np

from ...app_paths import default_measurements_dir, safe_measurements_dir
from ...measurement.adapter import bundle_to_generated_source, bundle_to_upload_payload
from ...measurement.devices import (
    get_default_input_device_index,
    get_default_output_device_index,
    list_measurement_input_devices,
    list_measurement_output_devices,
    list_wasapi_input_devices,
    list_wasapi_output_devices,
)
from ...measurement.io import save_measurement_bundle
from ...measurement.models import (
    MeasurementBundle,
    MeasurementRequest,
    MeasurementSessionAggregate,
)
from ...measurement.reference_signal_parts import _sanitize_measurement_dither_level_db
from ...measurement.sweep import (
    DEFAULT_MEASUREMENT_DITHER_LEVEL_DB,
    DEFAULT_MEASUREMENT_SAMPLE_RATE,
    DEFAULT_OUTPUT_GAIN_DB,
    DEFAULT_POST_SILENCE_S,
    DEFAULT_PRE_SILENCE_S,
    DEFAULT_SWEEP_END_HZ,
    DEFAULT_SWEEP_LENGTH_S,
    DEFAULT_SWEEP_START_HZ,
)
from ...measurement.capture import run_audibility_test
from ...measurement.workflow_parts import run_measurement_workflow
from .. import measurement_state, ng_controls as ctrl
from ..ng_sections import page_shell, section_card
from ..ng_timers import page_timer
from ..measurement_session_dialog_parts import build_measurement_session_dialog

_RECOVERABLE_MEASUREMENT_UI_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    KeyError,
    IndexError,
    RuntimeError,
    OSError,
)


_SESSION_PREVIEW_CHANNEL_ORDER = ("left", "right", "sub1", "sub2")


def _session_preview_channel_options(
    bundles: dict[str, MeasurementBundle],
    t: Callable[[str], str] | None = None,
) -> dict[str, str]:
    sub_label = t("measurement_role_sub") if callable(t) else "Subwoofer"
    labels = {
        "left": t("measurement_role_left") if callable(t) else "Left",
        "right": t("measurement_role_right") if callable(t) else "Right",
        "sub1": f"{sub_label} 1",
        "sub2": f"{sub_label} 2",
    }
    return {
        channel_key: labels[channel_key] for channel_key in _SESSION_PREVIEW_CHANNEL_ORDER if channel_key in bundles
    }


def _session_preview_default_channel_key(
    bundles: dict[str, MeasurementBundle],
    preferred_channel_key: str | None = None,
) -> str | None:
    preferred = str(preferred_channel_key or "").strip().lower()
    if preferred and preferred in bundles:
        return preferred
    for channel_key in _SESSION_PREVIEW_CHANNEL_ORDER:
        if channel_key in bundles:
            return channel_key
    return None


def _session_preview_bundle(
    session: MeasurementSessionAggregate,
    *,
    preferred_channel_key: str | None = None,
) -> MeasurementBundle | None:
    bundles = _session_preview_bundles(session)
    selected_key = _session_preview_default_channel_key(
        bundles,
        preferred_channel_key=preferred_channel_key,
    )
    if selected_key is None:
        return None
    return bundles.get(selected_key)


def _format_ms(value_ms: float | None, t: Callable[[str], str]) -> str:
    if value_ms is None or not np.isfinite(value_ms):
        return t("health_not_set")
    return f"{float(value_ms):.1f} ms"


def _measurement_timing_reference_text(
    timing,
    t: Callable[[str], str],
) -> str:
    if timing.reference_expected:
        return t("measurement_reference_detected") if timing.reference_detected else t("measurement_reference_missing")
    return t("measurement_reference_unused")


def _measurement_calibration_text(bundle: MeasurementBundle, t: Callable[[str], str]) -> str:
    if bundle.health.calibration_applied:
        return t("measurement_calibration_applied")
    if bundle.health.calibration_loaded:
        return t("measurement_calibration_loaded")
    return t("measurement_calibration_none")


def _measurement_level_hint(peak_dbfs: float, clipped: bool, t: Callable[[str], str]) -> str:
    if peak_dbfs >= -3.0 or clipped:
        return f" — {t('measurement_level_clip_risk')}"
    if peak_dbfs >= -20.0:
        return f" — {t('measurement_level_ok')}"
    if peak_dbfs > -40.0:
        return f" — {t('measurement_level_low')}"
    return f" — {t('measurement_level_very_low')}"


def _measurement_snr_text(snr, t: Callable[[str], str]) -> str | None:
    if snr is None or not math.isfinite(float(snr)):
        return None
    snr_float = float(snr)
    if snr_float >= 30.0:
        snr_label = t("measurement_snr_good")
    elif snr_float >= 15.0:
        snr_label = t("measurement_snr_fair")
    else:
        snr_label = t("measurement_snr_poor")
    return f"<div><b>{t('measurement_snr')}</b>: {snr_float:.1f} dB ({snr_label})</div>"


def _measurement_summary_html(
    bundle: MeasurementBundle,
    t: Callable[[str], str],
    timing_ref_latency_ms: float | None = None,
) -> str:
    peak_dbfs = -240.0
    if bundle.capture.recording_peak > 0.0:
        peak_dbfs = float(20.0 * np.log10(bundle.capture.recording_peak))
    fs = int(bundle.capture.samplerate)
    timing = bundle.timing
    if timing is not None and fs > 0:
        reference_arrival_ms = (
            float(timing.reference_arrival_index) * 1000.0 / float(fs)
            if timing.reference_arrival_index is not None
            else None
        )
        sweep_peak_ms = float(timing.sweep_peak_index) * 1000.0 / float(fs)
        fallback_latency_ms = (
            float(timing.fallback_latency_samples) * 1000.0 / float(fs)
            if timing.fallback_latency_samples is not None
            else None
        )
        timing_reference_text = _measurement_timing_reference_text(timing, t)
    else:
        reference_arrival_ms = None
        sweep_peak_ms = float(bundle.ir.peak_index) * 1000.0 / float(fs) if fs > 0 else None
        fallback_latency_ms = None
        timing_reference_text = t("measurement_reference_unused")
    calibration_text = _measurement_calibration_text(bundle, t)
    level_hint = _measurement_level_hint(peak_dbfs, bool(bundle.health.clipped), t)
    html = (
        f"<div><b>{t('measurement_recording_peak')}</b>: {peak_dbfs:+.1f} dBFS{level_hint}</div>"
        f"<div><b>{t('measurement_timing_reference')}</b>: {timing_reference_text}</div>"
        f"<div><b>{t('measurement_reference_arrival')}</b>: {_format_ms(reference_arrival_ms, t)}</div>"
        f"<div><b>{t('measurement_sweep_peak')}</b>: {_format_ms(sweep_peak_ms, t)}</div>"
        f"<div><b>{t('measurement_relative_latency')}</b>: {_format_ms(bundle.health.latency_ms, t)}</div>"
        f"<div><b>{t('measurement_calibration_status')}</b>: {calibration_text}</div>"
    )
    if (
        timing is not None
        and timing.reference_expected
        and not bool(bundle.health.latency_reliable)
        and fallback_latency_ms is not None
    ):
        html += f"<div><b>{t('measurement_latency_fallback')}</b>: " f"{_format_ms(fallback_latency_ms, t)}</div>"
    ref_text = (
        _format_ms(timing_ref_latency_ms, t) if timing_ref_latency_ms is not None else t("measurement_left_ref_not_set")
    )
    html += f"<div><b>{t('measurement_left_ref_label')}</b>: {ref_text}</div>"
    if timing_ref_latency_ms is not None and bundle.health.latency_ms is not None and bundle.health.latency_reliable:
        rel_ms = bundle.health.latency_ms - timing_ref_latency_ms
        sign = "+" if rel_ms >= 0 else ""
        html += f"<div><b>{t('measurement_left_ref_delay')}</b>: {sign}{rel_ms:.2f} ms</div>"
    snr_text = _measurement_snr_text(bundle.health.snr_db, t)
    if snr_text:
        html += snr_text
    if timing is not None and timing.reference_expected:
        mode = str(bundle.metadata.get("timing_mode", "") or "")
        corr_pre = bundle.metadata.get("pre_chirp_correlation_score")
        corr_post = bundle.metadata.get("post_chirp_correlation_score")
        corr_vals = [float(v) for v in (corr_pre, corr_post) if v is not None and math.isfinite(float(v))]
        corr_txt = f" (corr {max(corr_vals):.2f})" if corr_vals else ""
        mode_label = {
            "reference_anchored": t("measurement_timing_mode_anchored"),
            "source_anchored": t("measurement_timing_mode_source"),
            "reference_failed_fallback": t("measurement_timing_mode_fallback"),
        }.get(mode, mode)
        html += f"<div><b>{t('measurement_timing_mode')}</b>: {mode_label}{corr_txt}</div>"
    return html


def _measurement_role_value(value) -> str:
    role = str(value or "").strip().lower()
    if role in ("sub1", "sub2"):
        return "sub"
    return role if role in ("left", "right", "sub") else "left"


@dataclass
class _MeasurementTabContext:
    """Controller logic for the Measurement tab.

    Holds shared config/state and the former nested closures (now methods).
    Widget-bound rendering helpers stay nested inside ``build_measurement_tab``;
    only the few widgets needed by controller methods are kept as attributes
    (set during the widget tree build).
    """

    t: Callable
    ui: Any
    get_val: Callable
    measurement_samplerate: int
    input_default: Any
    output_default: Any
    sub_output_default: Any
    default_role: str
    default_use_wasapi: bool
    backend_message_state: dict[str, str | None] = field(default_factory=lambda: {"value": None})
    backend_message_label: dict[str, Any] = field(default_factory=lambda: {"widget": None})
    session_preview_state: dict[str, object] = field(default_factory=lambda: {"bundles": {}, "selected_key": None})
    sub_output_channel_row: Any = None
    upload_status: Any = None
    clear_calibration_btn: Any = None

    # --- backend message ---------------------------------------------------
    def set_backend_message(self, message: str | None) -> None:
        text = str(message or "").strip() or None
        self.backend_message_state["value"] = text
        widget = self.backend_message_label["widget"]
        if widget is None:
            return
        widget.set_text(text or "")
        widget.set_visibility(bool(text))

    def refresh_backend_message(self) -> bool:
        message = _measurement_audio_backend_message()
        self.set_backend_message(message)
        return message is None

    # --- device options ----------------------------------------------------
    def input_device_options(self, *, use_wasapi: bool | None = None) -> dict[int, str]:
        if not self.refresh_backend_message():
            return {}
        enabled = self.default_use_wasapi if use_wasapi is None else _measurement_use_wasapi_value(use_wasapi)
        if enabled:
            input_devices = list_wasapi_input_devices(self.measurement_samplerate, channels=1)
        else:
            input_devices = list_measurement_input_devices(self.measurement_samplerate, channels=1)
        return {
            int(device.index): _device_option_label(device, samplerate_hz=self.measurement_samplerate)
            for device in input_devices
        }

    def output_device_options_for_role(self, role_value: str, *, use_wasapi: bool | None = None) -> dict[int, str]:
        if not self.refresh_backend_message():
            return {}
        enabled = self.default_use_wasapi if use_wasapi is None else _measurement_use_wasapi_value(use_wasapi)
        required_channels = _measurement_required_output_channels(
            role_value,
            sub_output_channel=ctrl.value(
                "measurement_sub_output_channel",
                self.sub_output_default,
            ),
        )
        if enabled:
            output_devices = list_wasapi_output_devices(self.measurement_samplerate, channels=required_channels)
        else:
            output_devices = list_measurement_output_devices(self.measurement_samplerate, channels=required_channels)
        return {
            int(device.index): _device_option_label(device, samplerate_hz=self.measurement_samplerate)
            for device in output_devices
        }

    def refresh_input_device_options(self, use_wasapi: bool | None = None) -> None:
        options = self.input_device_options(
            use_wasapi=_measurement_use_wasapi_value(
                ctrl.value("measurement_use_wasapi", self.default_use_wasapi) if use_wasapi is None else use_wasapi
            )
        )
        ctrl.set_options("measurement_input_device", options)
        current_value = _pick_measurement_device_value(
            ctrl.value("measurement_input_device", None),
            self.input_default,
            options,
        )
        ctrl.set_value("measurement_input_device", current_value, emit=False)

    def refresh_output_device_options(self, role_value: str | None = None, use_wasapi: bool | None = None) -> None:
        normalized_role = _measurement_role_value(role_value or ctrl.value("measurement_role", self.default_role))
        resolved_use_wasapi = _measurement_use_wasapi_value(
            ctrl.value("measurement_use_wasapi", self.default_use_wasapi) if use_wasapi is None else use_wasapi
        )
        options = self.output_device_options_for_role(normalized_role, use_wasapi=resolved_use_wasapi)
        ctrl.set_options("measurement_output_device", options)
        current_value = _pick_measurement_device_value(
            ctrl.value("measurement_output_device", None),
            self.output_default,
            options,
        )
        ctrl.set_value("measurement_output_device", current_value, emit=False)

    def refresh_sub_output_channel_ui(self, role_value: str | None = None) -> None:
        visible = _measurement_sub_output_channel_visible(
            role_value or ctrl.value("measurement_role", self.default_role)
        )
        self.sub_output_channel_row.set_visibility(visible)
        ctrl.set_enabled("measurement_sub_output_channel", visible)

    # --- request building / capture ---------------------------------------
    def build_request(self, role_override: str | None = None) -> MeasurementRequest:
        def _int(name: str, default: int) -> int:
            try:
                return int(ctrl.value(name, default))
            except (TypeError, ValueError, OverflowError):
                return int(default)

        def _float_config(name: str, default: float) -> float:
            try:
                return float(self.get_val(name, default))
            except (TypeError, ValueError, OverflowError):
                return float(default)

        save_dir = Path(safe_measurements_dir(str(default_measurements_dir())))
        samplerate = _int("measurement_samplerate", DEFAULT_MEASUREMENT_SAMPLE_RATE)
        calibration_label = str(ctrl.value("measurement_mic_calibration_label", "") or "").strip()
        calibration_upload = ctrl.value("measurement_mic_calibration_upload", None)
        calibration_content = None
        calibration_filename = None
        if isinstance(calibration_upload, dict):
            raw_content = calibration_upload.get("content", b"")
            if isinstance(raw_content, (bytes, bytearray)) and len(raw_content) > 0:
                calibration_content = bytes(raw_content)
                upload_name = str(calibration_upload.get("filename", "") or "").strip()
                calibration_filename = upload_name or None
        if calibration_content is None:
            calibration_label = ""
        if not calibration_label and calibration_filename:
            calibration_label = Path(calibration_filename).stem
        requested_role = str(role_override or ctrl.value("measurement_role", "left") or "left").strip().lower()
        role = _measurement_role_value(requested_role)
        resolved_output_channel = _measurement_output_channel_for_role(
            role,
            sub_output_channel=ctrl.value(
                "measurement_sub_output_channel",
                self.sub_output_default,
            ),
        )
        session_channel_key = requested_role if requested_role in ("sub1", "sub2") else role
        return MeasurementRequest(
            input_device_index=_int("measurement_input_device", -1),
            output_device_index=_int("measurement_output_device", -1),
            samplerate=samplerate,
            sweep_start_hz=_float_config("measurement_sweep_start_hz", DEFAULT_SWEEP_START_HZ),
            sweep_end_hz=_float_config(
                "measurement_sweep_end_hz",
                min(float(samplerate) / 2.0, DEFAULT_SWEEP_END_HZ),
            ),
            sweep_length_s=_float_config("measurement_sweep_length_s", DEFAULT_SWEEP_LENGTH_S),
            pre_silence_s=DEFAULT_PRE_SILENCE_S,
            post_silence_s=DEFAULT_POST_SILENCE_S,
            input_channel=0,
            output_channel=int(resolved_output_channel),
            output_gain_db=_float_config("measurement_output_gain_db", DEFAULT_OUTPUT_GAIN_DB),
            measurement_dither_level_db=_sanitize_measurement_dither_level_db(
                ctrl.value(
                    "measurement_dither_level_db",
                    self.get_val(
                        "measurement_dither_level_db",
                        DEFAULT_MEASUREMENT_DITHER_LEVEL_DB,
                    ),
                )
            ),
            role=role,
            use_wasapi=_measurement_use_wasapi_value(ctrl.value("measurement_use_wasapi", False)),
            save_dir=save_dir,
            measurement_source_path=None,
            mic_calibration_path=None,
            mic_calibration_label=calibration_label or None,
            mic_calibration_content=calibration_content,
            mic_calibration_filename=calibration_filename,
            session_channel_key=session_channel_key,
        )

    def run_measurement_thread(self) -> None:
        request = self.build_request()
        self.clear_session_preview_state()
        measurement_state.set_busy(self.t("measurement_status_running"))

        def _run() -> None:
            try:
                bundle = run_measurement_workflow(request)
                measurement_state.set_result(
                    bundle,
                    self.t("measurement_status_done").format(path=str(request.save_dir)),
                )
                if (
                    str(bundle.capture.request.role or "") == "left"
                    and bundle.health.latency_ms is not None
                    and bundle.health.latency_reliable
                ):
                    measurement_state.set_timing_reference(bundle.health.latency_ms)
            except _RECOVERABLE_MEASUREMENT_UI_EXCEPTIONS as exc:
                measurement_state.set_error(str(exc))

        threading.Thread(target=_run, daemon=True).start()

    def run_audibility_test_thread(self) -> None:
        request = self.build_request()
        measurement_state.set_busy(self.t("measurement_audibility_test_running"))

        def _run() -> None:
            try:
                run_audibility_test(request)
                measurement_state.set_status(self.t("measurement_audibility_test_done"))
            except _RECOVERABLE_MEASUREMENT_UI_EXCEPTIONS as exc:
                measurement_state.set_error(str(exc))

        threading.Thread(target=_run, daemon=True).start()

    # --- bundle application ------------------------------------------------
    def refresh_target_preview(self) -> None:
        try:
            from ..ng_tab_target_parts import refresh_target_preview  # noqa: PLC0415

            refresh_target_preview()
        except (ImportError, AttributeError, RuntimeError, OSError):
            logger.exception("target preview refresh")

    def set_html_content(self, element, content: str) -> None:
        try:
            element.set_content(content)
        except _RECOVERABLE_MEASUREMENT_UI_EXCEPTIONS:
            try:
                element.content = content
                element.update()
            except _RECOVERABLE_MEASUREMENT_UI_EXCEPTIONS:
                logger.exception("html element content set in measurement tab")

    def current_bundle(self):
        bundles = dict(self.session_preview_state["bundles"])
        if bundles:
            selected_key = _session_preview_default_channel_key(
                bundles,
                preferred_channel_key=str(self.session_preview_state.get("selected_key", "") or ""),
            )
            self.session_preview_state["selected_key"] = selected_key
            if selected_key is not None:
                return bundles.get(selected_key)
        snap = measurement_state.get_snapshot()
        bundle = snap.get("bundle", None)
        return bundle if isinstance(bundle, MeasurementBundle) else None

    def save_current_bundle(self) -> None:
        bundle = self.current_bundle()
        if bundle is None:
            return
        save_dir = Path(safe_measurements_dir(str(bundle.capture.request.save_dir or default_measurements_dir())))
        role = str(bundle.capture.request.role or "generic").strip().lower() or "generic"
        stem = "measurement" if role == "generic" else f"measurement_{role}"
        bundle.saved_paths = save_measurement_bundle(bundle, save_dir, stem)
        measurement_state.set_result(bundle, self.t("measurement_status_saved").format(path=str(save_dir)))

    def apply_bundle(self, role: str) -> None:
        bundle = self.current_bundle()
        if bundle is None:
            return
        slot = "l" if role == "left" else "r"
        source = bundle_to_generated_source(bundle, role=role)
        ctrl.set_value(f"generated_measurement_{slot}", source)
        ctrl.set_value(f"file_{slot}", source["upload"])
        ctrl.set_value(f"local_path_{slot}", "")
        ctrl.set_value(f"local_path_{slot}__legacy", "")
        self.refresh_target_preview()
        status_key = "measurement_status_applied_left" if role == "left" else "measurement_status_applied_right"
        measurement_state.set_result(bundle, self.t(status_key))

    def apply_bundle_as_sub(self, slot: int) -> None:
        bundle = self.current_bundle()
        if bundle is None:
            return
        file_key = "file_l_sub" if slot == 1 else "file_r_sub"
        path_key = "local_path_l_sub" if slot == 1 else "local_path_r_sub"
        filename = f"measurement_sub{slot}_ir.wav"
        ctrl.set_value(file_key, bundle_to_upload_payload(bundle, filename=filename))
        ctrl.set_value(path_key, "")
        self.refresh_target_preview()
        status_key = "measurement_status_applied_sub1" if slot == 1 else "measurement_status_applied_sub2"
        measurement_state.set_result(bundle, self.t(status_key))

    # --- guided session ----------------------------------------------------
    def selected_session_preview_bundle(self) -> MeasurementBundle | None:
        bundles = dict(self.session_preview_state["bundles"])
        selected_key = _session_preview_default_channel_key(
            bundles,
            preferred_channel_key=str(self.session_preview_state.get("selected_key", "") or ""),
        )
        self.session_preview_state["selected_key"] = selected_key
        if selected_key is None:
            return None
        return bundles.get(selected_key)

    def set_session_preview_state(
        self,
        session: MeasurementSessionAggregate,
        *,
        preferred_channel_key: str | None = None,
    ) -> MeasurementBundle | None:
        bundles = _session_preview_bundles(session)
        self.session_preview_state["bundles"] = bundles
        self.session_preview_state["selected_key"] = _session_preview_default_channel_key(
            bundles,
            preferred_channel_key=preferred_channel_key,
        )
        return self.selected_session_preview_bundle()

    def clear_session_preview_state(self) -> None:
        self.session_preview_state["bundles"] = {}
        self.session_preview_state["selected_key"] = None

    def on_session_start(self, status_text: str) -> None:
        measurement_state.set_busy(status_text)

    def on_session_complete(self, session: MeasurementSessionAggregate) -> None:
        preview_bundle = self.set_session_preview_state(session)
        if preview_bundle is None:
            measurement_state.set_error("Guided measurement session completed without final results.")
            return
        saved_dir = str((session.summary or {}).get("saved_session_dir", "") or "").strip()
        status_text = (
            f"Guided measurement session complete. Saved to {saved_dir}."
            if saved_dir
            else "Guided measurement session complete."
        )
        measurement_state.set_result(preview_bundle, status_text)
        if (
            session.final_left_bundle is not None
            and session.final_left_bundle.health.latency_ms is not None
            and session.final_left_bundle.health.latency_reliable
        ):
            measurement_state.set_timing_reference(session.final_left_bundle.health.latency_ms)

    def on_session_error(self, error_text: str) -> None:
        measurement_state.set_error(error_text)

    def apply_session_results(self, session: MeasurementSessionAggregate) -> None:
        applied_roles: list[str] = []
        if session.final_left_bundle is not None:
            left_source = bundle_to_generated_source(session.final_left_bundle, role="left")
            ctrl.set_value("generated_measurement_l", left_source)
            ctrl.set_value("file_l", left_source["upload"])
            ctrl.set_value("local_path_l", "")
            ctrl.set_value("local_path_l__legacy", "")
            applied_roles.append("Left")
        if session.final_right_bundle is not None:
            right_source = bundle_to_generated_source(session.final_right_bundle, role="right")
            ctrl.set_value("generated_measurement_r", right_source)
            ctrl.set_value("file_r", right_source["upload"])
            ctrl.set_value("local_path_r", "")
            ctrl.set_value("local_path_r__legacy", "")
            applied_roles.append("Right")
        if session.final_sub1_bundle is not None:
            ctrl.set_value(
                "file_l_sub",
                bundle_to_upload_payload(session.final_sub1_bundle, filename="measurement_sub1_ir.wav"),
            )
            ctrl.set_value("local_path_l_sub", "")
            applied_roles.append("Sub 1")
        if session.final_sub2_bundle is not None:
            ctrl.set_value(
                "file_r_sub",
                bundle_to_upload_payload(session.final_sub2_bundle, filename="measurement_sub2_ir.wav"),
            )
            ctrl.set_value("local_path_r_sub", "")
            applied_roles.append("Sub 2")

        preview_bundle = self.set_session_preview_state(session)
        if preview_bundle is not None:
            measurement_state.set_result(
                preview_bundle,
                f"Guided measurement session applied: {', '.join(applied_roles)}.",
            )
        if (
            session.final_left_bundle is not None
            and session.final_left_bundle.health.latency_ms is not None
            and session.final_left_bundle.health.latency_reliable
        ):
            measurement_state.set_timing_reference(session.final_left_bundle.health.latency_ms)
        self.refresh_target_preview()

    # --- calibration upload ------------------------------------------------
    async def handle_calibration_upload(self, e) -> None:
        payload = _build_upload_payload(
            filename=e.file.name,
            content=await e.file.read(),
            mime_type=getattr(e.file, "content_type", ""),
        )
        ctrl.set_value("measurement_mic_calibration_upload", payload)
        filename = str(payload.get("filename", "") or "").strip()
        ctrl.set_value("measurement_mic_calibration_label", Path(filename).stem if filename else "")

    def refresh_calibration_upload_status(self) -> None:
        payload = ctrl.value("measurement_mic_calibration_upload", None)
        if isinstance(payload, dict) and payload.get("content"):
            self.upload_status.set_text(
                self.t("measurement_mic_calibration_selected").format(
                    name=str(payload.get("filename", "") or self.t("health_not_set"))
                )
            )
            self.clear_calibration_btn.enable()
            return
        self.upload_status.set_text(self.t("measurement_calibration_none"))
        self.clear_calibration_btn.disable()

    def clear_calibration_upload(self) -> None:
        ctrl.set_value("measurement_mic_calibration_upload", None)
        ctrl.set_value("measurement_mic_calibration_label", "")


def build_measurement_tab(*, t: Callable, get_val: Callable) -> None:
    from nicegui import ui

    measurement_samplerate = int(
        get_val("measurement_samplerate", DEFAULT_MEASUREMENT_SAMPLE_RATE) or DEFAULT_MEASUREMENT_SAMPLE_RATE
    )
    input_default = get_val("measurement_input_device", get_default_input_device_index())
    output_default = get_val("measurement_output_device", get_default_output_device_index())
    sub_output_default = _measurement_sub_output_channel_value(
        get_val(
            "measurement_sub_output_channel",
            _measurement_sub_output_channel_default(),
        )
    )
    default_role = _measurement_role_value(get_val("measurement_role", "left"))
    default_use_wasapi = _measurement_use_wasapi_value(get_val("measurement_use_wasapi", False))

    ctx = _MeasurementTabContext(
        t=t,
        ui=ui,
        get_val=get_val,
        measurement_samplerate=measurement_samplerate,
        input_default=input_default,
        output_default=output_default,
        sub_output_default=sub_output_default,
        default_role=default_role,
        default_use_wasapi=default_use_wasapi,
    )

    input_options = ctx.input_device_options(use_wasapi=default_use_wasapi)
    output_options = ctx.output_device_options_for_role(default_role, use_wasapi=default_use_wasapi)

    open_guided_session_dialog = build_measurement_session_dialog(
        t=t,
        build_request_for_role=ctx.build_request,
        apply_session_results=ctx.apply_session_results,
        on_session_start=ctx.on_session_start,
        on_session_complete=ctx.on_session_complete,
        on_session_error=ctx.on_session_error,
    )

    with page_shell(title=t("tab_measurement"), intro=t("measurement_page_intro")):
        with section_card(title=t("measurement_capture_section_title"), intro=t("measurement_intro")):
            ui.label(t("measurement_volume_attenuation_hint")).classes("text-xs text-gray-400")
            ctx.backend_message_label["widget"] = ui.label("").classes(
                "w-full whitespace-pre-wrap text-sm text-red-400"
            )
            ctx.set_backend_message(ctx.backend_message_state["value"])

            ctrl.register("generated_measurement_l", ctrl._ValueHolder(None))
            ctrl.register("generated_measurement_r", ctrl._ValueHolder(None))
            ctrl.register("measurement_mic_calibration_upload", ctrl._ValueHolder(None))
            ctrl.register(
                "measurement_mic_calibration_label",
                ctrl._ValueHolder(str(get_val("measurement_mic_calibration_label", "") or "")),
            )
            ctrl.register(
                "measurement_samplerate",
                ctrl._ValueHolder(
                    int(
                        get_val("measurement_samplerate", DEFAULT_MEASUREMENT_SAMPLE_RATE)
                        or DEFAULT_MEASUREMENT_SAMPLE_RATE
                    )
                ),
            )
            ctrl.register(
                "measurement_sub_output_channel",
                ctrl._ValueHolder(sub_output_default),
            )
            ctrl.register(
                "measurement_dither_level_db",
                ctrl._ValueHolder(
                    _sanitize_measurement_dither_level_db(
                        get_val(
                            "measurement_dither_level_db",
                            DEFAULT_MEASUREMENT_DITHER_LEVEL_DB,
                        )
                    )
                ),
            )
            if sys.platform == "win32":
                with ui.row().classes("w-full"):
                    ctrl.register(
                        "measurement_use_wasapi",
                        ui.checkbox(
                            t("measurement_use_wasapi"),
                            value=default_use_wasapi,
                        ).classes("text-sm"),
                    )
            else:
                ctrl.register("measurement_use_wasapi", ctrl._ValueHolder(False))

            with ui.row().classes("w-full gap-4"):
                ctrl.register(
                    "measurement_input_device",
                    ui.select(
                        options=input_options,
                        value=_pick_measurement_device_value(None, input_default, input_options),
                        label=t("measurement_input_device"),
                    )
                    .props("dense outlined")
                    .classes("flex-1"),
                )
                ctrl.register(
                    "measurement_output_device",
                    ui.select(
                        options=output_options,
                        value=_pick_measurement_device_value(None, output_default, output_options),
                        label=t("measurement_output_device"),
                    )
                    .props("dense outlined")
                    .classes("flex-1"),
                )
                ctrl.register(
                    "measurement_role",
                    ui.select(
                        options={
                            "left": t("measurement_role_left"),
                            "right": t("measurement_role_right"),
                            "sub": t("measurement_role_sub"),
                        },
                        value=default_role,
                        label=t("measurement_role"),
                    )
                    .props("dense outlined")
                    .classes("flex-1"),
                )
                ui.button(
                    t("measurement_refresh_devices"),
                    on_click=lambda: (
                        ctx.refresh_input_device_options(),
                        ctx.refresh_output_device_options(),
                    ),
                ).props('outline color="secondary"').classes("self-end")

            ctx.sub_output_channel_row = ui.row().classes("w-full gap-4")
            with ctx.sub_output_channel_row:
                ctrl.register(
                    "measurement_sub_output_channel",
                    ui.select(
                        options={idx: str(idx + 1) for idx in range(8)},
                        value=sub_output_default,
                        label=t("measurement_sub_output_channel"),
                    )
                    .props("dense outlined")
                    .classes("flex-1"),
                )
                ui.label(t("measurement_sub_output_channel_hint")).classes("flex-1 text-xs text-gray-400")

            with ui.row().classes("w-full gap-4"):
                with ui.column().classes("flex-1 gap-2"):
                    ctrl.register(
                        "measurement_dither_level_db",
                        ui.number(
                            label=t("measurement_dither_level_db"),
                            value=_sanitize_measurement_dither_level_db(
                                get_val(
                                    "measurement_dither_level_db",
                                    DEFAULT_MEASUREMENT_DITHER_LEVEL_DB,
                                )
                            ),
                            format="%.1f",
                        )
                        .props("dense outlined")
                        .classes("w-full"),
                    )
                    ui.label(t("measurement_dither_level_hint")).classes("text-xs text-gray-400")
                with ui.column().classes("flex-1 gap-2"):
                    ui.upload(
                        label=t("measurement_mic_calibration_upload"),
                        on_upload=ctx.handle_calibration_upload,
                        auto_upload=True,
                    ).props('accept=".txt,.cal,.csv"').classes("w-full")
                    ctx.upload_status = ui.label(t("measurement_calibration_none")).classes("text-xs text-gray-400")

                ctx.clear_calibration_btn = ui.button(
                    t("file_status_clear"),
                    on_click=ctx.clear_calibration_upload,
                ).props('outline color="secondary"')

        with section_card(
            title=t("measurement_actions_section_title"),
            intro=t("measurement_actions_section_intro"),
        ):
            start_btn = (
                ui.button(t("measurement_start_button"), on_click=ctx.run_measurement_thread)
                .props('color="positive" unelevated')
                .classes("w-full")
            )
            audibility_test_btn = (
                ui.button(
                    t("measurement_audibility_test_button"),
                    on_click=ctx.run_audibility_test_thread,
                )
                .props('outline color="secondary"')
                .classes("w-full")
            )
            guided_session_btn = (
                ui.button(
                    t("measurement_guided_session_button"),
                    on_click=open_guided_session_dialog,
                )
                .props('outline color="secondary"')
                .classes("w-full")
            )
            status_label = ui.label("").classes("text-sm text-gray-400")
            error_label = ui.label("").classes("text-sm text-red-400")

        with section_card(title=t("measurement_preview_section_title"), hero=True):
            summary_html = ui.html("")
            warnings_label = ui.label("").classes("whitespace-pre-wrap text-sm")
            with ui.row().classes("w-full gap-3"):
                save_btn = ui.button(t("measurement_save_ir"), on_click=ctx.save_current_bundle).props(
                    'outline color="secondary"'
                )
                apply_left_btn = ui.button(
                    t("measurement_apply_left"),
                    on_click=lambda: ctx.apply_bundle("left"),
                ).props('outline color="secondary"')
                apply_right_btn = ui.button(
                    t("measurement_apply_right"),
                    on_click=lambda: ctx.apply_bundle("right"),
                ).props('outline color="secondary"')
                clear_ref_btn = ui.button(
                    t("measurement_left_ref_clear"),
                    on_click=measurement_state.clear_timing_reference,
                ).props('outline color="secondary"')
            session_preview_selector_row = ui.row().classes("w-full gap-3 items-center")
            with session_preview_selector_row:
                session_preview_channel = (
                    ui.select(
                        {},
                        value=None,
                        label=t("measurement_role"),
                    )
                    .props("dense outlined")
                    .classes("w-56")
                )
            session_preview_selector_row.set_visibility(False)
            plot_scope = ui.column().classes("w-full")

        plot_ref = {"widget": None}
        last_nonce = {"value": -1}

        def _refresh_session_preview_selector() -> None:
            bundles = dict(ctx.session_preview_state["bundles"])
            options = _session_preview_channel_options(bundles, t)
            visible = len(options) > 1
            try:
                session_preview_channel.set_options(options)
            except _RECOVERABLE_MEASUREMENT_UI_EXCEPTIONS:
                session_preview_channel.options = options
                session_preview_channel.update()
            selected_key = _session_preview_default_channel_key(
                bundles,
                preferred_channel_key=str(ctx.session_preview_state.get("selected_key", "") or ""),
            )
            ctx.session_preview_state["selected_key"] = selected_key
            if selected_key != session_preview_channel.value:
                try:
                    session_preview_channel.set_value(selected_key)
                except _RECOVERABLE_MEASUREMENT_UI_EXCEPTIONS:
                    session_preview_channel.value = selected_key
                    session_preview_channel.update()
            session_preview_selector_row.set_visibility(visible)

        def _render_bundle(bundle: MeasurementBundle | None, timing_ref_latency_ms: float | None = None) -> None:
            _refresh_session_preview_selector()
            if bundle is None:
                ctx.set_html_content(summary_html, "")
                warnings_label.set_text("")
                plot_scope.clear()
                plot_ref["widget"] = None
                save_btn.disable()
                apply_left_btn.disable()
                apply_right_btn.disable()
                clear_ref_btn.disable()
                return

            ctx.set_html_content(
                summary_html,
                _measurement_summary_html(bundle, t, timing_ref_latency_ms),
            )
            warnings = list(bundle.health.warnings)
            warnings_label.set_text("\n".join(warnings) if warnings else t("measurement_no_warnings"))
            save_btn.enable()
            apply_left_btn.enable()
            apply_right_btn.enable()
            if timing_ref_latency_ms is not None:
                clear_ref_btn.enable()

            fig = _build_preview_figure(bundle)
            if plot_ref["widget"] is None:
                plot_scope.clear()
                with plot_scope:
                    plot_ref["widget"] = ui.plotly(fig).classes("w-full")
            else:
                try:
                    plot_ref["widget"].update_figure(fig)
                except _RECOVERABLE_MEASUREMENT_UI_EXCEPTIONS:
                    plot_scope.clear()
                    with plot_scope:
                        plot_ref["widget"] = ui.plotly(fig).classes("w-full")

        def _on_session_preview_channel_change(e) -> None:
            selected_key = str(getattr(e, "value", "") or "").strip().lower()
            ctx.session_preview_state["selected_key"] = selected_key or None
            snap = measurement_state.get_snapshot()
            ref_ms = snap.get("timing_ref_latency_ms", None)
            _render_bundle(ctx.selected_session_preview_bundle(), timing_ref_latency_ms=ref_ms)

        session_preview_channel.on_value_change(_on_session_preview_channel_change)

        def _refresh_view() -> None:
            snap = measurement_state.get_snapshot()
            is_busy = bool(snap.get("busy", False))
            # Check if devices are available
            input_opts = ctx.input_device_options()
            output_opts = ctx.output_device_options_for_role(ctrl.value("measurement_role", default_role))
            has_devices = bool(input_opts and output_opts)

            if is_busy or not has_devices:
                start_btn.disable()
                audibility_test_btn.disable()
                guided_session_btn.disable()
            else:
                start_btn.enable()
                audibility_test_btn.enable()
                guided_session_btn.enable()
            status_label.set_text(str(snap.get("status_text", "") or ""))
            error_text = str(snap.get("error_text", "") or "")
            error_label.set_text(error_text)
            error_label.set_visibility(bool(error_text))
            ref_ms = snap.get("timing_ref_latency_ms", None)
            if ref_ms is not None:
                clear_ref_btn.enable()
            else:
                clear_ref_btn.disable()
            if int(snap.get("nonce", -1) or -1) == int(last_nonce["value"]):
                return
            last_nonce["value"] = int(snap.get("nonce", -1) or -1)
            bundle = ctx.current_bundle()
            _render_bundle(
                bundle if isinstance(bundle, MeasurementBundle) else None,
                timing_ref_latency_ms=ref_ms,
            )

        ctrl.get("measurement_mic_calibration_upload").on_value_change(
            lambda _e: ctx.refresh_calibration_upload_status()
        )
        ctrl.get("measurement_role").on_value_change(
            lambda e: (
                ctx.refresh_sub_output_channel_ui(e.value),
                ctx.refresh_output_device_options(e.value),
            )
        )
        ctrl.get("measurement_use_wasapi").on_value_change(
            lambda e: (
                ctx.refresh_input_device_options(e.value),
                ctx.refresh_output_device_options(ctrl.value("measurement_role", default_role), e.value),
            )
        )
        ctrl.get("measurement_sub_output_channel").on_value_change(
            lambda e: ctx.refresh_output_device_options(
                ctrl.value("measurement_role", default_role),
            )
        )
        ctx.refresh_sub_output_channel_ui(default_role)
        ctx.refresh_input_device_options(default_use_wasapi)
        ctx.refresh_output_device_options(default_role, default_use_wasapi)
        ctx.refresh_calibration_upload_status()
        _refresh_session_preview_selector()
        _render_bundle(None)
        page_timer(0.4, _refresh_view)


__all__ = [
    "_session_preview_channel_options",
    "_session_preview_default_channel_key",
    "_session_preview_bundle",
    "_format_ms",
    "_measurement_summary_html",
    "build_measurement_tab",
    "sys",
    "ctrl",
    "_sanitize_measurement_dither_level_db",
]
