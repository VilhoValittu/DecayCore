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

import numpy as np

from ...application.health_service import compute_health
from ...application.run_request import RunRequest
from ...application.run_contracts import (
    PreparedRunInput,
    copy_resolved_data,
    copy_source_ui_data,
)
from ...common.measurement_features import (
    normalize_rt60_bands,
    normalize_rt60_value,
    prepare_rt60_analysis_ir,
)
from ...config.decaycore_config import save_config
from ...config.pipeline_parts import (
    filter_type_short,
)
from ...io.generated_measurement_source import generated_source_matches_upload, parse_generated_source
from ...dsp.decaycore_analysis import calculate_rt60, calculate_rt60_bands
from ...io.measurements_loader_parts import (
    _try_load_harmonic_sidecar,
    _try_load_rt60_sidecar,
    load_bass_integration_measurements,
    load_measurements_lr,
    load_raw_irs_lr,
    load_raw_ir_sub,
)
from ...resources.i8n.decaycore_i18n import t
from ..bridge_types import ProcessRunCallbacks

if typing.TYPE_CHECKING:
    from ..process_run_flow import ProcessRunSupport

logger = logging.getLogger("DecayCore")

def _prepare_measurement_ui_window_settings(data: dict) -> None:
    ir_export_window_mode = data.get("ir_export_window_mode")
    if not isinstance(ir_export_window_mode, str) or ir_export_window_mode.strip() == "":
        data["ir_export_window_mode"] = "auto"
    logger.info(f"UI ir_export_window_mode={data.get('ir_export_window_mode')}")

    try:
        sh = str(data.get("ir_export_window_shape", "hann") or "hann").strip().lower()
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
        sh = "hann"
    if sh not in ("hann", "tukey"):
        sh = "hann"
    data["ir_export_window_shape"] = sh

    try:
        alpha = float(data.get("ir_export_tukey_alpha", 0.25))
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
        alpha = 0.25
    if not math.isfinite(alpha):
        alpha = 0.25
    data["ir_export_tukey_alpha"] = float(np.clip(alpha, 0.0, 1.0))

    try:
        if filter_type_short(str(data.get("filter_type", "") or "")) == "Asymmetric":
            data["ir_export_window_mode"] = "rew_asym"
            data["ir_window_mode"] = "rew_asym"
            data["ir_export_window_shape"] = "tukey"
            data["ir_export_tukey_alpha"] = 0.25
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
        logger.exception("asymmetric filter window mode set")


def _prepare_measurement_read_paths(
    data: dict,
    *,
    bass_integration_enabled: bool,
) -> tuple[str, str, dict]:
    if bass_integration_enabled:
        return "local_path_l_main", "local_path_r_main", {
            "file_key_l": "file_l_main",
            "path_key_l": "local_path_l_main",
            "file_key_r": "file_r_main",
            "path_key_r": "local_path_r_main",
        }
    return "local_path_l", "local_path_r", {}


def _prepare_measurement_baseline_io(
    data: dict,
    *,
    bass_integration_enabled: bool,
    generated_pair,
    logger: logging.Logger,
):
    if bass_integration_enabled:
        bundle, f_l, m_l, p_l, f_r, m_r, p_r = load_bass_integration_measurements(data, logger=logger)
        if bundle is not None:
            logger.info(
                "Bass Integration: decomposed bass integration measurements loaded; "
                "predicted totals built; xo/hpf phase model retained"
            )
        return bundle, f_l, m_l, p_l, f_r, m_r, p_r, None, 0, None, 0, None, None, None, None

    if generated_pair is not None:
        return (
            None,
            generated_pair[0],
            generated_pair[1],
            generated_pair[2],
            generated_pair[3],
            generated_pair[4],
            generated_pair[5],
            generated_pair[6],
            generated_pair[7],
            generated_pair[8],
            generated_pair[9],
            generated_pair[10],
            generated_pair[11],
            generated_pair[12],
            generated_pair[13],
        )

    f_l, m_l, p_l, f_r, m_r, p_r = load_measurements_lr(data, logger=logger)
    return None, f_l, m_l, p_l, f_r, m_r, p_r, None, 0, None, 0, None, None, None, None


def _prepare_measurement_sidecar_metadata(
    data: dict,
    *,
    bass_integration_enabled: bool,
) -> tuple[
    str,
    str,
    float | None,
    dict[float, float] | None,
    float | None,
    dict[float, float] | None,
    float | None,
    float | None,
]:
    measured_rt60_l, measured_rt60_bands_l = _extract_generated_source_rt60(
        data.get("generated_measurement_l")
    )
    measured_rt60_r, measured_rt60_bands_r = _extract_generated_source_rt60(
        data.get("generated_measurement_r")
    )
    measured_snr_db_l = _extract_generated_source_snr(data.get("generated_measurement_l"))
    measured_snr_db_r = _extract_generated_source_snr(data.get("generated_measurement_r"))

    if bass_integration_enabled:
        _lp_l = str(data.get("local_path_l_main", "") or "").strip()
        _lp_r = str(data.get("local_path_r_main", "") or "").strip()
    else:
        _lp_l = str(data.get("local_path_l", "") or "").strip()
        _lp_r = str(data.get("local_path_r", "") or "").strip()
    sidecar_rt60_l, sidecar_rt60_bands_l = _try_load_rt60_sidecar(_lp_l)
    sidecar_rt60_r, sidecar_rt60_bands_r = _try_load_rt60_sidecar(_lp_r)
    if measured_rt60_l is None:
        measured_rt60_l = sidecar_rt60_l
    if measured_rt60_bands_l is None:
        measured_rt60_bands_l = sidecar_rt60_bands_l
    if measured_rt60_r is None:
        measured_rt60_r = sidecar_rt60_r
    if measured_rt60_bands_r is None:
        measured_rt60_bands_r = sidecar_rt60_bands_r

    return (
        _lp_l,
        _lp_r,
        measured_rt60_l,
        measured_rt60_bands_l,
        measured_rt60_r,
        measured_rt60_bands_r,
        measured_snr_db_l,
        measured_snr_db_r,
    )


def _compute_rt60_from_raw_ir(
    raw_ir,
    raw_ir_fs,
) -> tuple[float | None, dict[float, float] | None]:
    """RT60 from a raw impulse response, with the same peak-anchored bounded
    window as built-in measurement metadata (measurement.rt60)."""
    ir_arr = np.asarray(raw_ir if raw_ir is not None else [], dtype=float).reshape(-1)
    fs_i = int(raw_ir_fs or 0)
    if ir_arr.size == 0 or fs_i <= 0:
        return None, None
    try:
        analysis_ir = prepare_rt60_analysis_ir(ir_arr, fs_i, -1)
        rt60_val = normalize_rt60_value(calculate_rt60(analysis_ir, fs_i))
        rt60_bands = normalize_rt60_bands(calculate_rt60_bands(analysis_ir, fs_i))
        return rt60_val, rt60_bands
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
        logger.exception("RT60 computation from imported IR failed")
        return None, None


def _prepare_measurement_read_bundle(
    data: dict,
    *,
    bass_integration_enabled: bool,
    logger: logging.Logger,
) -> tuple:
    generated_pair = None if bass_integration_enabled else _load_generated_measurement_pair(data)
    if bass_integration_enabled:
        (
            bass_integration_bundle,
            f_l,
            m_l,
            p_l,
            f_r,
            m_r,
            p_r,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = _prepare_measurement_baseline_io(
            data,
            bass_integration_enabled=True,
            generated_pair=None,
            logger=logger,
        )
    else:
        (
            bass_integration_bundle,
            f_l,
            m_l,
            p_l,
            f_r,
            m_r,
            p_r,
            raw_ir_l,
            raw_ir_fs_l,
            raw_ir_r,
            raw_ir_fs_r,
            _harmonic_freq_hz_l,
            _harmonic_mags_l,
            _harmonic_freq_hz_r,
            _harmonic_mags_r,
        ) = _prepare_measurement_baseline_io(
            data,
            bass_integration_enabled=False,
            generated_pair=generated_pair,
            logger=logger,
        )

    (
        _lp_l,
        _lp_r,
        measured_rt60_l,
        measured_rt60_bands_l,
        measured_rt60_r,
        measured_rt60_bands_r,
        measured_snr_db_l,
        measured_snr_db_r,
    ) = _prepare_measurement_sidecar_metadata(
        data,
        bass_integration_enabled=bass_integration_enabled,
    )
    if generated_pair is None:
        _harmonic_freq_hz_l, _harmonic_mags_l = _try_load_harmonic_sidecar(_lp_l)
        _harmonic_freq_hz_r, _harmonic_mags_r = _try_load_harmonic_sidecar(_lp_r)

    raw_ir_slot_keys = {}
    if bass_integration_enabled:
        raw_ir_slot_keys = {
            "file_key_l": "file_l_main",
            "path_key_l": "local_path_l_main",
            "file_key_r": "file_r_main",
            "path_key_r": "local_path_r_main",
        }
    if bass_integration_enabled or generated_pair is None:
        raw_ir_l, raw_ir_fs_l, raw_ir_r, raw_ir_fs_r = load_raw_irs_lr(
            data,
            logger=logger,
            **raw_ir_slot_keys,
        )
    if bass_integration_enabled:
        raw_ir_sub, raw_ir_fs_sub = load_raw_ir_sub(data, logger=logger)
    else:
        raw_ir_sub, raw_ir_fs_sub = None, 0

    # External imports carry no RT60 metadata or sidecar; without it the adaptive
    # target, decay hints and TDC severity silently fall back to generic constants
    # even though the imported IR holds the real decay. Compute it here so the
    # measurement context carries measured RT60 for every source that has an IR.
    if (measured_rt60_l is None or measured_rt60_bands_l is None) and raw_ir_l is not None:
        computed_val, computed_bands = _compute_rt60_from_raw_ir(raw_ir_l, raw_ir_fs_l)
        if measured_rt60_l is None:
            measured_rt60_l = computed_val
        if measured_rt60_bands_l is None:
            measured_rt60_bands_l = computed_bands
        if computed_val is not None or computed_bands is not None:
            data["rt60_source_l"] = "computed_from_imported_ir"
            logger.info(f"RT60 (L) computed from imported IR: {measured_rt60_l}")
    if (measured_rt60_r is None or measured_rt60_bands_r is None) and raw_ir_r is not None:
        computed_val, computed_bands = _compute_rt60_from_raw_ir(raw_ir_r, raw_ir_fs_r)
        if measured_rt60_r is None:
            measured_rt60_r = computed_val
        if measured_rt60_bands_r is None:
            measured_rt60_bands_r = computed_bands
        if computed_val is not None or computed_bands is not None:
            data["rt60_source_r"] = "computed_from_imported_ir"
            logger.info(f"RT60 (R) computed from imported IR: {measured_rt60_r}")

    return (
        bass_integration_bundle,
        f_l,
        m_l,
        p_l,
        f_r,
        m_r,
        p_r,
        raw_ir_l,
        raw_ir_fs_l,
        raw_ir_r,
        raw_ir_fs_r,
        raw_ir_sub,
        raw_ir_fs_sub,
        measured_rt60_l,
        measured_rt60_bands_l,
        measured_rt60_r,
        measured_rt60_bands_r,
        _harmonic_freq_hz_l,
        _harmonic_mags_l,
        _harmonic_freq_hz_r,
        _harmonic_mags_r,
        measured_snr_db_l,
        measured_snr_db_r,
    )


def _get_wav_window_params(data: dict) -> tuple[float, float, int]:
    try:
        pre_ms = float(data.get("ir_window_left", 85.0) or 85.0)
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
        pre_ms = 85.0
    try:
        post_ms = float(data.get("ir_window_right", data.get("ir_window", 500.0)) or 500.0)
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
        post_ms = 500.0
    try:
        smoothing_level = int(data.get("smoothing_level", 0) or 0)
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
        smoothing_level = 0
    return float(pre_ms), float(post_ms), int(smoothing_level)

def _extract_generated_source_rt60(source: object) -> tuple[float | None, dict[float, float] | None]:
    if not isinstance(source, dict):
        return None, None

    try:
        rt60_val = float(source.get("measured_rt60", None))
    except (TypeError, ValueError):
        rt60_val = None
    if rt60_val is not None and (not np.isfinite(rt60_val) or rt60_val <= 0.0):
        rt60_val = None

    raw_bands = source.get("measured_rt60_bands", None)
    rt60_bands: dict[float, float] | None = None
    if isinstance(raw_bands, dict) and raw_bands:
        normalized: dict[float, float] = {}
        for key, value in raw_bands.items():
            try:
                freq_hz = float(key)
                band_rt60 = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(freq_hz) and np.isfinite(band_rt60) and band_rt60 > 0.0:
                normalized[float(freq_hz)] = float(band_rt60)
        if normalized:
            rt60_bands = normalized

    return rt60_val, rt60_bands

def _extract_generated_source_snr(source: object) -> float | None:
    if not isinstance(source, dict):
        return None
    try:
        v = float(source.get("snr_db", None))
    except (TypeError, ValueError):
        return None
    return float(v) if np.isfinite(v) else None


def _load_generated_measurement_pair(data: dict) -> tuple | None:
    generated_l = data.get("generated_measurement_l")
    generated_r = data.get("generated_measurement_r")
    upload_l = data.get("file_l")
    upload_r = data.get("file_r")
    if not generated_source_matches_upload(generated_l, upload_l):
        return None
    if not generated_source_matches_upload(generated_r, upload_r):
        return None

    pre_ms, post_ms, smoothing_level = _get_wav_window_params(data)
    f_l, m_l, p_l, raw_ir_l, raw_ir_fs_l, harmonic_freq_l, harmonic_mags_l = parse_generated_source(
        generated_l,
        pre_ms=pre_ms,
        post_ms=post_ms,
        smoothing_level=smoothing_level,
        logger=logger,
    )
    f_r, m_r, p_r, raw_ir_r, raw_ir_fs_r, harmonic_freq_r, harmonic_mags_r = parse_generated_source(
        generated_r,
        pre_ms=pre_ms,
        post_ms=post_ms,
        smoothing_level=smoothing_level,
        logger=logger,
    )
    if f_l is None or f_r is None:
        return None
    return (
        f_l, m_l, p_l, f_r, m_r, p_r,
        raw_ir_l, raw_ir_fs_l, raw_ir_r, raw_ir_fs_r,
        harmonic_freq_l, harmonic_mags_l, harmonic_freq_r, harmonic_mags_r,
    )

def _prepare_ui_and_measurements(
    *,
    request: RunRequest,
    callbacks: ProcessRunCallbacks,
    support: ProcessRunSupport,
) -> dict | None:
    perf_stats = {
        "read_s": 0.0,
        "dsp_s": 0.0,
        "zip_png_s": 0.0,
    }
    per_fs_stats: dict[int, dict[str, float]] = {}

    source_ui_data = copy_source_ui_data(request.raw_ui_data)
    data = copy_resolved_data(source_ui_data)
    run_started_at = float(request.run_started_at or time.perf_counter())
    callbacks.set_auto_selected_bar("")

    try:
        mode = str(data.get("mode") or "BASIC").strip().upper()
        hr = compute_health(data, mode)
        if support.ui_bridge.toast_health_gate_result(hr, mode):
            return None
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
        logger.exception("health gate check")

    _prepare_measurement_ui_window_settings(data)

    taps_base = int(float(data.get("taps", 65536) or 65536))
    if not bool(data.get("_headless", False)):
        save_config(data)

    support.ui_bridge.ensure_progress_bar()

    read_started_at = time.perf_counter()
    callbacks.status(t("stat_reading"))
    bass_integration_enabled = bool(data.get("bass_integration_enable", False))
    (
        bass_integration_bundle,
        f_l,
        m_l,
        p_l,
        f_r,
        m_r,
        p_r,
        raw_ir_l,
        raw_ir_fs_l,
        raw_ir_r,
        raw_ir_fs_r,
        raw_ir_sub,
        raw_ir_fs_sub,
        measured_rt60_l,
        measured_rt60_bands_l,
        measured_rt60_r,
        measured_rt60_bands_r,
        _harmonic_freq_hz_l,
        _harmonic_mags_l,
        _harmonic_freq_hz_r,
        _harmonic_mags_r,
        measured_snr_db_l,
        measured_snr_db_r,
    ) = _prepare_measurement_read_bundle(
        data,
        bass_integration_enabled=bass_integration_enabled,
        logger=logger,
    )
    perf_stats["read_s"] += max(0.0, float(time.perf_counter() - read_started_at))
    if f_l is None or f_r is None:
        return None

    prepared_input = PreparedRunInput(
        source_ui_data=source_ui_data,
        resolved_data=data,
        f_l=f_l,
        m_l=m_l,
        p_l=p_l,
        f_r=f_r,
        m_r=m_r,
        p_r=p_r,
        bass_integration_bundle=bass_integration_bundle,
        raw_ir_l=raw_ir_l,
        raw_ir_fs_l=raw_ir_fs_l,
        raw_ir_r=raw_ir_r,
        raw_ir_fs_r=raw_ir_fs_r,
        raw_ir_sub=raw_ir_sub,
        raw_ir_fs_sub=raw_ir_fs_sub,
        measured_rt60_l=measured_rt60_l,
        measured_rt60_bands_l=measured_rt60_bands_l,
        measured_rt60_r=measured_rt60_r,
        measured_rt60_bands_r=measured_rt60_bands_r,
        harmonic_freq_hz_l=_harmonic_freq_hz_l,
        harmonic_magnitudes_db_l=_harmonic_mags_l,
        harmonic_freq_hz_r=_harmonic_freq_hz_r,
        harmonic_magnitudes_db_r=_harmonic_mags_r,
    )

    return {
        "run_started_at": run_started_at,
        "perf_stats": perf_stats,
        "per_fs_stats": per_fs_stats,
        "source_ui_data": source_ui_data,
        "resolved_data": data,
        "prepared_input": prepared_input,
        "data": data,
        "taps_base": taps_base,
        "f_l": f_l,
        "m_l": m_l,
        "p_l": p_l,
        "f_r": f_r,
        "m_r": m_r,
        "p_r": p_r,
        "bass_integration_bundle": bass_integration_bundle,
        "raw_ir_l": raw_ir_l,
        "raw_ir_fs_l": raw_ir_fs_l,
        "raw_ir_r": raw_ir_r,
        "raw_ir_fs_r": raw_ir_fs_r,
        "raw_ir_sub": raw_ir_sub,
        "raw_ir_fs_sub": raw_ir_fs_sub,
        "measured_rt60_l": measured_rt60_l,
        "measured_rt60_bands_l": measured_rt60_bands_l,
        "measured_rt60_r": measured_rt60_r,
        "measured_rt60_bands_r": measured_rt60_bands_r,
        "harmonic_freq_hz_l": _harmonic_freq_hz_l,
        "harmonic_magnitudes_db_l": _harmonic_mags_l,
        "harmonic_freq_hz_r": _harmonic_freq_hz_r,
        "harmonic_magnitudes_db_r": _harmonic_mags_r,
        "measured_snr_db_l": measured_snr_db_l,
        "measured_snr_db_r": measured_snr_db_r,
    }


__all__ = [
    '_get_wav_window_params',
    '_extract_generated_source_rt60',
    '_extract_generated_source_snr',
    '_load_generated_measurement_pair',
    '_prepare_ui_and_measurements',
    '_try_load_harmonic_sidecar',
    '_try_load_rt60_sidecar',
    'compute_health',
    'filter_type_short',
    'load_bass_integration_measurements',
    'load_measurements_lr',
    'load_raw_ir_sub',
    'load_raw_irs_lr',
    'save_config',
]
