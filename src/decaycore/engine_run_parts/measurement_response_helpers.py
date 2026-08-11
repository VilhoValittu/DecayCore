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
from typing import Any

import numpy as np

from ..common.measurement_features import (
    build_harmonic_boost_risk_curve,
    normalize_rt60_bands,
    normalize_rt60_value,
)
from ..dsp.correction_types import MeasurementSideContext

logger = logging.getLogger("DecayCore")


def _build_measurement_side_ctx(measurements: dict, side: str) -> MeasurementSideContext:
    """Build a MeasurementSideContext for one channel from the measurements dict."""
    rt60_val = normalize_rt60_value(measurements.get(f"measured_rt60_{side}"))
    rt60_bands = normalize_rt60_bands(measurements.get(f"measured_rt60_bands_{side}"))
    try:
        snr_raw = measurements.get(f"measured_snr_db_{side}")
        snr_db: float | None = float(snr_raw) if snr_raw is not None and np.isfinite(float(snr_raw)) else None
    except (TypeError, ValueError):
        snr_db = None
    h_freq = measurements.get(f"harmonic_freq_hz_{side}")
    h_mags = measurements.get(f"harmonic_magnitudes_db_{side}")
    h_freq_arr = None if h_freq is None else np.asarray(h_freq, dtype=float)
    h_mags_dict: dict[int, np.ndarray] | None = None
    if isinstance(h_mags, dict) and h_mags:
        h_mags_dict = {int(k): np.asarray(v, dtype=float) for k, v in h_mags.items()}

    risk_freq: np.ndarray | None = None
    risk_curve: np.ndarray | None = None
    risk_summary: dict = {}
    if h_freq_arr is not None and h_mags_dict:
        f_key = "f_l" if side == "l" else "f_r"
        m_key = "m_l" if side == "l" else "m_r"
        try:
            risk_freq, risk_curve, risk_summary = build_harmonic_boost_risk_curve(
                h_freq_arr,
                h_mags_dict,
                fundamental_freq_hz=measurements.get(f_key),
                fundamental_mag_db=measurements.get(m_key),
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
            pass

    return MeasurementSideContext(
        measured_rt60=rt60_val,
        measured_rt60_bands=rt60_bands,
        measurement_snr_db=snr_db,
        harmonic_freq_hz=h_freq_arr,
        harmonic_magnitudes_db=h_mags_dict,
        harmonic_risk_freq_hz=risk_freq,
        harmonic_risk_curve=risk_curve,
        harmonic_risk_summary=risk_summary,
        side=str(side).strip().lower() or None,
    )


def _phase_from_ir(ir: np.ndarray, fs: int, freq_axis: np.ndarray) -> np.ndarray:
    x = np.asarray(ir, dtype=float).ravel()
    f_axis = np.asarray(freq_axis, dtype=float).ravel()
    if x.size < 8 or f_axis.size == 0 or int(fs) <= 0:
        return np.zeros_like(f_axis, dtype=float)
    h = np.fft.rfft(x)
    f_fft = np.fft.rfftfreq(x.size, d=1.0 / float(fs))
    p_deg = np.rad2deg(np.unwrap(np.angle(h)))
    f_q = np.clip(f_axis, float(np.min(f_fft)), float(np.max(f_fft)))
    return np.interp(f_q, f_fft, p_deg).astype(float)


def _to_axis(arr: Any, fallback: np.ndarray) -> np.ndarray:
    out = np.asarray(arr if arr is not None else [], dtype=float).ravel()
    if out.size > 1:
        return out
    return np.asarray(fallback if fallback is not None else [], dtype=float).ravel()


def _extract_lr_measurement_axes(measurements: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    empty = np.asarray([], dtype=float)
    return (
        _to_axis(measurements.get("f_l"), empty),
        _to_axis(measurements.get("m_l"), empty),
        _to_axis(measurements.get("p_l"), empty),
        _to_axis(measurements.get("f_r"), empty),
        _to_axis(measurements.get("m_r"), empty),
        _to_axis(measurements.get("p_r"), empty),
    )


def _resample_to_axis(values: Any, f_src: np.ndarray, f_dst: np.ndarray) -> np.ndarray:
    v = np.asarray(values if values is not None else [], dtype=float).ravel()
    fs = np.asarray(f_src if f_src is not None else [], dtype=float).ravel()
    fd = np.asarray(f_dst if f_dst is not None else [], dtype=float).ravel()
    if fd.size == 0:
        return np.asarray([], dtype=float)
    if v.size == fd.size and fd.size > 0:
        return v.astype(float, copy=False)
    if v.size < 2 or fs.size < 2:
        return np.zeros_like(fd, dtype=float)
    f_q = np.clip(fd, float(np.min(fs)), float(np.max(fs)))
    return np.interp(f_q, fs, v).astype(float)

def _resolve_sub_measurement_for_filter(
    measurements: dict,
    *,
    f_l: np.ndarray,
    m_l: np.ndarray,
    p_l: np.ndarray,
    f_r: np.ndarray,
    m_r: np.ndarray,
    p_r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def _apply_direct_dac_sub_alignment(transfer):
        if transfer is None:
            return None
        try:
            delay_ms = float(measurements.get("bass_integration_sub_delay_ms", 0.0) or 0.0)
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
            delay_ms = 0.0
        try:
            gain_trim_db = float(measurements.get("bass_integration_sub_gain_trim_db", 0.0) or 0.0)
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
            gain_trim_db = 0.0
        aligned = _apply_polarity_to_transfer(
            transfer,
            invert=bool(measurements.get("bass_integration_sub_polarity_invert", False)),
            label="Direct-DAC sub polarity",
        )
        aligned = _apply_gain_trim_to_transfer(
            aligned,
            gain_trim_db=float(gain_trim_db),
            label="Direct-DAC sub gain",
        )
        aligned = _apply_delay_to_transfer(
            aligned,
            delay_ms=float(delay_ms),
            label="Direct-DAC sub delay",
        )
        return aligned

    bundle = measurements.get("bass_integration_bundle")
    if bundle is not None:
        from ..dsp.bass_integration import (
            _apply_delay_to_transfer,
            _apply_gain_trim_to_transfer,
            _apply_polarity_to_transfer,
            build_bundle_combined_sub_transfer,
        )

        try:
            sub_transfer, _diag = build_bundle_combined_sub_transfer(
                bundle,
                channel="l",
                label="Direct-DAC sub total",
            )
            sub_transfer = _apply_direct_dac_sub_alignment(sub_transfer)
            return (
                np.asarray(sub_transfer.freqs_hz, dtype=float),
                np.asarray(sub_transfer.mag_db, dtype=float),
                np.asarray(sub_transfer.phase_deg, dtype=float),
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
            logger.debug("Could not build Bass Integration sub-only transfer", exc_info=True)

    f_sub = (f_l + f_r) / 2.0 if f_l.size == f_r.size else f_l
    m_sub = (m_l + m_r) / 2.0 if m_l.size == m_r.size else m_l
    p_sub = (p_l + p_r) / 2.0 if p_l.size == p_r.size else p_l
    return (
        np.asarray(f_sub, dtype=float),
        np.asarray(m_sub, dtype=float),
        np.asarray(p_sub, dtype=float),
    )

def _interp_complex_to_axis(
    f_src: np.ndarray,
    h_src: np.ndarray,
    f_dst: np.ndarray,
) -> np.ndarray:
    fs = np.asarray(f_src if f_src is not None else [], dtype=float).reshape(-1)
    hs = np.asarray(h_src if h_src is not None else [], dtype=np.complex128).reshape(-1)
    fd = np.asarray(f_dst if f_dst is not None else [], dtype=float).reshape(-1)
    if fd.size == 0:
        return np.asarray([], dtype=np.complex128)
    if fs.size < 2 or hs.size != fs.size:
        return np.zeros(fd.shape, dtype=np.complex128)
    f_q = np.clip(fd, float(np.min(fs)), float(np.max(fs)))
    re = np.interp(f_q, fs, np.real(hs))
    im = np.interp(f_q, fs, np.imag(hs))
    return (re + 1j * im).astype(np.complex128, copy=False)

def _ir_fft_on_axis(ir: np.ndarray, fs: int, f_axis: np.ndarray) -> np.ndarray:
    x = np.asarray(ir if ir is not None else [], dtype=float).reshape(-1)
    fa = np.asarray(f_axis if f_axis is not None else [], dtype=float).reshape(-1)
    fs_i = int(fs) if fs else 0
    if x.size < 8 or fa.size == 0 or fs_i <= 0:
        return np.zeros(fa.shape, dtype=np.complex128)
    h_fft = np.fft.rfft(x)
    f_fft = np.fft.rfftfreq(x.size, d=1.0 / float(fs_i))
    return _interp_complex_to_axis(f_fft, h_fft, fa)


__all__ = [
    '_build_measurement_side_ctx',
    '_phase_from_ir',
    '_to_axis',
    '_extract_lr_measurement_axes',
    '_resample_to_axis',
    '_resolve_sub_measurement_for_filter',
    '_interp_complex_to_axis',
    '_ir_fft_on_axis',
]
