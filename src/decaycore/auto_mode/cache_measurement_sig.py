# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Auto-mode measurement signature computation."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from .cache_io import _AUTO_CACHE_LOCK
from .shared import _auto_hash_array_full, _auto_safe_float, logger


def _auto_measurement_metadata_identity(measurements: dict) -> str:
    """Hash measurement-derived metadata that affects DSP policy or scoring.

    The full measurement signature already hashes magnitude/frequency arrays.
    This identity is intentionally metadata-focused so auto-mode signatures can
    expose why cache entries differ when source-of-truth measurement context
    changes without the visible magnitude grid changing.
    """
    m = dict(measurements or {})
    h = hashlib.sha256()
    h.update(b"measurement-metadata-v1:")

    scalar_keys = (
        "measured_rt60_l",
        "measured_rt60_r",
        "measurement_snr_db_l",
        "measurement_snr_db_r",
        "measured_snr_db_l",
        "measured_snr_db_r",
        "snr_db_l",
        "snr_db_r",
        "capture_samplerate_l",
        "capture_samplerate_r",
        "drift_ratio_l",
        "drift_ratio_r",
        "reference_latency_s_l",
        "reference_latency_s_r",
    )
    dict_keys = (
        "rt60_summary_l",
        "rt60_summary_r",
        "measured_rt60_bands_l",
        "measured_rt60_bands_r",
        "harmonic_risk_summary_l",
        "harmonic_risk_summary_r",
        "measurement_health_l",
        "measurement_health_r",
        "timing_analysis_l",
        "timing_analysis_r",
        "repeat_analysis_l",
        "repeat_analysis_r",
        "metadata_l",
        "metadata_r",
        "measurement_metadata_l",
        "measurement_metadata_r",
    )
    array_keys = (
        "harmonic_freq_hz_l",
        "harmonic_freq_hz_r",
        "harmonic_risk_freq_hz_l",
        "harmonic_risk_freq_hz_r",
        "harmonic_risk_curve_l",
        "harmonic_risk_curve_r",
    )

    payload: dict[str, object] = {}
    for key in scalar_keys:
        value = _auto_safe_float(m.get(key, float("nan")), float("nan"))
        if np.isfinite(value):
            payload[key] = float(value)
    for key in dict_keys:
        value = m.get(key)
        if isinstance(value, dict) and value:
            payload[key] = value
    try:
        h.update(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8", "ignore"))
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
        logger.exception("measurement metadata payload signature hash update")
        h.update(str(sorted(payload.items())).encode("utf-8", "ignore"))

    for key in array_keys:
        value = m.get(key)
        if value is not None:
            h.update(str(key).encode("utf-8", "ignore"))
            h.update(_auto_hash_array_full(np.asarray(value, dtype=float)).encode("ascii", "ignore"))

    for key in ("harmonic_magnitudes_db_l", "harmonic_magnitudes_db_r"):
        value = m.get(key)
        if isinstance(value, dict) and value:
            try:
                h.update(str(key).encode("utf-8", "ignore"))
                for order in sorted(value.keys()):
                    arr = value.get(order)
                    if arr is not None:
                        h.update(str(order).encode("utf-8", "ignore"))
                        h.update(_auto_hash_array_full(np.asarray(arr, dtype=float)).encode("ascii", "ignore"))
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
                logger.exception("measurement harmonic metadata identity hash update")

    return h.hexdigest()


def _auto_get_measurement_signature(measurements: dict) -> str:
    """Return memoized measurement signature for this dict.

    Computes once and stores under the private key "_auto_measurement_signature".
    Safe: that key is not included in the hash computation.
    """
    _MEMO_KEY = "_auto_measurement_signature"
    with _AUTO_CACHE_LOCK:
        sig = measurements.get(_MEMO_KEY)
        if isinstance(sig, str) and sig:
            return sig
        sig = _auto_measurement_signature(measurements)
        try:
            measurements[_MEMO_KEY] = sig
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
            logger.exception("measurement signature memo store")
        return sig


def _auto_measurement_signature(measurements: dict) -> str:
    fL = measurements.get("f_l")
    mL = measurements.get("m_l")
    pL = measurements.get("p_l")
    fR = measurements.get("f_r")
    mR = measurements.get("m_r")
    pR = measurements.get("p_r")
    h = hashlib.sha256()
    h.update(_auto_hash_array_full(np.asarray(fL) if fL is not None else np.asarray([])).encode("ascii", "ignore"))
    h.update(_auto_hash_array_full(np.asarray(mL) if mL is not None else np.asarray([])).encode("ascii", "ignore"))
    h.update(_auto_hash_array_full(np.asarray(pL) if pL is not None else np.asarray([])).encode("ascii", "ignore"))
    h.update(_auto_hash_array_full(np.asarray(fR) if fR is not None else np.asarray([])).encode("ascii", "ignore"))
    h.update(_auto_hash_array_full(np.asarray(mR) if mR is not None else np.asarray([])).encode("ascii", "ignore"))
    h.update(_auto_hash_array_full(np.asarray(pR) if pR is not None else np.asarray([])).encode("ascii", "ignore"))
    h.update(_auto_measurement_metadata_identity(measurements).encode("ascii", "ignore"))

    for _rt60_scalar_key in ("measured_rt60_l", "measured_rt60_r"):
        _rt60_scalar = _auto_safe_float(measurements.get(_rt60_scalar_key, float("nan")), float("nan"))
        if np.isfinite(_rt60_scalar):
            h.update(f"{_rt60_scalar_key}:{float(_rt60_scalar):.6g}".encode("ascii", "ignore"))

    for _rt60_summary_key in ("rt60_summary_l", "rt60_summary_r"):
        _rt60_summary = measurements.get(_rt60_summary_key)
        if isinstance(_rt60_summary, dict) and _rt60_summary:
            try:
                h.update(json.dumps(_rt60_summary, sort_keys=True, default=str).encode("utf-8", "ignore"))
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
                logger.exception("rt60 summary signature hash update")

    for _rt60_key in ("measured_rt60_bands_l", "measured_rt60_bands_r"):
        _rt60_bands = measurements.get(_rt60_key)
        if isinstance(_rt60_bands, dict) and _rt60_bands:
            try:
                _rt60_sorted = sorted(
                    (
                        (float(k), float(v))
                        for k, v in _rt60_bands.items()
                        if np.isfinite(float(k)) and np.isfinite(float(v))
                    ),
                    key=lambda kv: kv[0],
                )
                h.update(json.dumps(_rt60_sorted).encode("utf-8", "ignore"))
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
                logger.exception("rt60 bands signature hash update")

    for _hf_key in ("harmonic_freq_hz_l", "harmonic_freq_hz_r"):
        _hf = measurements.get(_hf_key)
        if _hf is not None:
            h.update(_auto_hash_array_full(np.asarray(_hf, dtype=float)).encode("ascii", "ignore"))

    for _hm_key in ("harmonic_magnitudes_db_l", "harmonic_magnitudes_db_r"):
        _hm = measurements.get(_hm_key)
        if isinstance(_hm, dict) and _hm:
            try:
                for _order in sorted(_hm.keys()):
                    _arr = _hm.get(_order)
                    if _arr is not None:
                        h.update(_auto_hash_array_full(np.asarray(_arr, dtype=float)).encode("ascii", "ignore"))
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
                logger.exception("harmonic magnitudes signature hash update")

    for _hr_key in ("harmonic_risk_summary_l", "harmonic_risk_summary_r"):
        _hr = measurements.get(_hr_key)
        if isinstance(_hr, dict) and _hr:
            try:
                h.update(json.dumps(_hr, sort_keys=True, default=str).encode("utf-8", "ignore"))
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
                logger.exception("harmonic risk summary signature hash update")

    if bool(measurements.get("bass_integration_enabled", False)):
        bundle = measurements.get("bass_integration_bundle", None)
        for attr_name in ("l_main", "r_main", "l_sub", "r_sub"):
            comp = getattr(bundle, attr_name, None)
            freqs = getattr(comp, "freqs_hz", None)
            spec = getattr(comp, "complex_spec", None)
            h.update(_auto_hash_array_full(np.asarray(freqs) if freqs is not None else np.asarray([])).encode("ascii", "ignore"))
            arr = np.asarray(spec, dtype=np.complex128).reshape(-1) if spec is not None else np.asarray([], dtype=np.complex128)
            h.update(_auto_hash_array_full(np.real(arr)).encode("ascii", "ignore"))
            h.update(_auto_hash_array_full(np.imag(arr)).encode("ascii", "ignore"))
        try:
            h.update(
                json.dumps(
                    {
                        "avr_crossover_hz": float(_auto_safe_float(measurements.get("avr_crossover_hz", float("nan")), float("nan"))),
                        "bass_integration_profile": str(measurements.get("bass_integration_profile", "") or ""),
                        "bass_integration_mode": "direct_dac",
                        "dual_sub_preprocessing": {
                            key: dict(getattr(bundle, "diagnostics", {}) or {}).get(key)
                            for key in (
                                "dual_sub_preprocessing_applied",
                                "dual_sub_preprocessing_version",
                                "dual_sub_combined_method",
                                "dual_sub_relative_delay_samples",
                                "dual_sub_original_sub_combine_mode",
                                "sub_combine_mode",
                            )
                        },
                    },
                    sort_keys=True,
                ).encode("utf-8", "ignore")
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
            logger.exception("bass integration signature hash update")
    return h.hexdigest()


def _auto_optuna_stable_study_sig(measurement_identity: str, filter_key: str) -> str:
    """Stable Optuna study signature keyed only on measurement + filter type.

    Does NOT include taps/fs/xos — ensures the same Optuna study is reused
    across runs even when filter length or sample rate changes.
    """
    h = hashlib.sha256()
    h.update(str(filter_key or "mixed").strip().lower().encode("utf-8"))
    h.update(b":")
    h.update(str(measurement_identity or "").encode("utf-8"))
    return h.hexdigest()


def _auto_target_study_sig(
    measurement_identity: str,
    goal: str,
    filter_key: str | None = None,
) -> str:
    """Stable target-study signature keyed on measurement + auto goal + filter type."""
    h = hashlib.sha256()
    h.update(b"target-v3:")
    h.update(str(goal or "balanced").strip().lower().encode("utf-8", "ignore"))
    h.update(b":")
    h.update(str(filter_key or "mixed").strip().lower().encode("utf-8", "ignore"))
    h.update(b":")
    h.update(str(measurement_identity or "").encode("utf-8", "ignore"))
    return h.hexdigest()
