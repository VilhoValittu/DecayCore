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

import numpy as np

from .measurements_wav import parse_measurements_from_wav_bytes

_RECOVERABLE_PARSE_EXCEPTIONS = (
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


def generated_source_matches_upload(source: object, upload: object) -> bool:
    if not isinstance(source, dict) or not isinstance(upload, dict):
        return False
    source_sha = str(source.get("upload_sha256", "") or "")
    upload_sha = str(upload.get("content_sha256", "") or "")
    if source_sha and upload_sha:
        return source_sha == upload_sha
    return bytes(source.get("upload", {}).get("content", b"")) == bytes(upload.get("content", b""))


def _interpolate_mic_calibration(
    cal_freq_hz: np.ndarray,
    cal_db: np.ndarray,
    target_freq_hz: np.ndarray,
) -> np.ndarray:
    cal_f = np.asarray(cal_freq_hz, dtype=float)
    cal_v = np.asarray(cal_db, dtype=float)
    target_f = np.asarray(target_freq_hz, dtype=float)

    if cal_f.size < 2 or cal_v.size != cal_f.size:
        raise ValueError("Calibration arrays must contain at least two matching frequency points.")
    if target_f.size == 0:
        return np.asarray([], dtype=float)

    cal_f = np.clip(cal_f, np.finfo(float).tiny, None)
    target_safe = np.clip(target_f, np.finfo(float).tiny, None)

    return np.interp(
        np.log10(target_safe),
        np.log10(cal_f),
        cal_v,
        left=float(cal_v[0]),
        right=float(cal_v[-1]),
    ).astype(float)


def _apply_mic_calibration_to_magnitude(magnitude_db: np.ndarray, calibration_db: np.ndarray) -> np.ndarray:
    magnitude = np.asarray(magnitude_db, dtype=float)
    raw_response = np.asarray(calibration_db, dtype=float)
    if magnitude.shape != raw_response.shape:
        raise ValueError("Magnitude and calibration arrays must have the same shape.")
    return (magnitude - raw_response).astype(float)


def _empty_generated_source_result() -> tuple[None, None, None, None, int, None, None]:
    return None, None, None, None, 0, None, None


def _apply_preferred_analysis_magnitude(
    source: dict,
    f_hz: np.ndarray,
    mag_db: np.ndarray,
    phase_deg: np.ndarray,
    *,
    logger=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    analysis_freq_hz = source.get("analysis_freq_hz")
    analysis_mag_db = source.get("analysis_magnitude_db")
    spatial_avg_freq_hz = source.get("spatial_avg_analysis_freq_hz")
    spatial_avg_mag_db = source.get("spatial_avg_analysis_magnitude_db")
    preferred_freq_hz = spatial_avg_freq_hz if spatial_avg_freq_hz is not None else analysis_freq_hz
    preferred_mag_db = spatial_avg_mag_db if spatial_avg_mag_db is not None else analysis_mag_db
    if preferred_freq_hz is None or preferred_mag_db is None:
        return f_hz, mag_db, phase_deg
    try:
        preferred_freq = np.asarray(preferred_freq_hz, dtype=float).reshape(-1)
        preferred_mag = np.asarray(preferred_mag_db, dtype=float).reshape(-1)
        base_freq = np.asarray(f_hz, dtype=float).reshape(-1)
        base_phase = np.asarray(phase_deg, dtype=float).reshape(-1)
        if (
            preferred_freq.size >= 2
            and preferred_mag.size == preferred_freq.size
            and np.all(np.isfinite(preferred_freq))
            and np.all(np.isfinite(preferred_mag))
            and base_freq.size == base_phase.size
            and base_freq.size >= 2
        ):
            phase_deg = np.interp(preferred_freq, base_freq, base_phase, left=base_phase[0], right=base_phase[-1])
            return preferred_freq, preferred_mag, phase_deg
    except _RECOVERABLE_PARSE_EXCEPTIONS:
        if logger:
            logger.warning("Generated measurement analysis magnitude apply failed", exc_info=True)
    return f_hz, mag_db, phase_deg


def _apply_generated_source_calibration(
    source: dict,
    f_hz: np.ndarray,
    mag_db: np.ndarray,
    *,
    logger=None,
) -> np.ndarray:
    cal_freq_hz = source.get("calibration_freq_hz")
    cal_db = source.get("calibration_db")
    if cal_freq_hz is None or cal_db is None:
        return mag_db
    try:
        interpolated = _interpolate_mic_calibration(
            np.asarray(cal_freq_hz, dtype=float),
            np.asarray(cal_db, dtype=float),
            np.asarray(f_hz, dtype=float),
        )
        return _apply_mic_calibration_to_magnitude(np.asarray(mag_db, dtype=float), interpolated)
    except _RECOVERABLE_PARSE_EXCEPTIONS:
        if logger:
            logger.warning("Generated measurement calibration apply failed", exc_info=True)
    return mag_db


def _extract_harmonic_outputs(source: dict) -> tuple[np.ndarray | None, dict[int, np.ndarray] | None]:
    harmonic_freq_hz = source.get("harmonic_freq_hz")
    harmonic_magnitudes_db = source.get("harmonic_magnitudes_db")
    harmonic_freq_out = None if harmonic_freq_hz is None else np.asarray(harmonic_freq_hz, dtype=float)
    harmonic_mags_out: dict[int, np.ndarray] | None = None
    if isinstance(harmonic_magnitudes_db, dict) and harmonic_magnitudes_db:
        harmonic_mags_out = {int(k): np.asarray(v, dtype=float) for k, v in harmonic_magnitudes_db.items()}
    return harmonic_freq_out, harmonic_mags_out


def parse_generated_source(
    source: object,
    *,
    pre_ms: float,
    post_ms: float,
    smoothing_level: int,
    logger=None,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    int,
    np.ndarray | None,
    dict[int, np.ndarray] | None,
]:
    if not isinstance(source, dict):
        return _empty_generated_source_result()
    upload = source.get("upload", None)
    if not isinstance(upload, dict):
        return _empty_generated_source_result()
    content = upload.get("content", b"")
    if not isinstance(content, (bytes, bytearray)):
        return _empty_generated_source_result()

    f_hz, mag_db, phase_deg = parse_measurements_from_wav_bytes(
        bytes(content),
        pre_ms=pre_ms,
        post_ms=post_ms,
        smoothing_level=smoothing_level,
        logger=logger,
    )
    if f_hz is None or mag_db is None or phase_deg is None:
        return _empty_generated_source_result()

    f_hz, mag_db, phase_deg = _apply_preferred_analysis_magnitude(
        source,
        f_hz,
        mag_db,
        phase_deg,
        logger=logger,
    )
    mag_db = _apply_generated_source_calibration(source, f_hz, mag_db, logger=logger)

    raw_ir = source.get("raw_ir", None)
    raw_ir_fs = int(source.get("raw_ir_fs", 0) or 0)
    raw_ir_arr = None if raw_ir is None else np.asarray(raw_ir, dtype=np.float32)

    harmonic_freq_out, harmonic_mags_out = _extract_harmonic_outputs(source)

    return (
        np.asarray(f_hz, dtype=float),
        np.asarray(mag_db, dtype=float),
        np.asarray(phase_deg, dtype=float),
        raw_ir_arr,
        raw_ir_fs,
        harmonic_freq_out,
        harmonic_mags_out,
    )


__all__ = ["generated_source_matches_upload", "parse_generated_source"]
