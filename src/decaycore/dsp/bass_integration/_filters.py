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

import sys

import numpy as np
import scipy.signal

from ...io.measurement_bundle import BassIntegrationBundle, TransferData
from ..bass_cache import _BUTTER_RESPONSE_CACHE, _get_filtered_branch_cache
from ._utils import _build_transfer_like, _interp_complex_response, _safe_float


def _get_pkg():
    """Return the bass_integration package module for patchable attribute lookup."""
    return sys.modules[__name__.rsplit(".", 1)[0]]


def _butterworth_complex_response(
    freqs_hz: np.ndarray,
    cutoff_hz: float,
    order: int,
    *,
    btype: str,
) -> np.ndarray:
    freqs = np.asarray(freqs_hz, dtype=float)
    cutoff = _safe_float(cutoff_hz, float("nan"))
    try:
        ord_i = int(order)
    except (TypeError, ValueError, OverflowError):
        ord_i = 0
    if freqs.size == 0 or (not np.isfinite(cutoff)) or cutoff <= 0.0 or ord_i <= 0:
        return np.ones(freqs.shape, dtype=np.complex128)
    _n = freqs.size
    _cache_key = (_n, float(freqs[0]), float(freqs[_n // 2]), float(freqs[-1]), float(cutoff), int(ord_i), str(btype))
    _hit = _BUTTER_RESPONSE_CACHE.get(_cache_key)
    if _hit is not None:
        return _hit
    try:
        b, a = scipy.signal.butter(
            max(1, int(ord_i)),
            2.0 * np.pi * float(cutoff),
            btype=str(btype),
            analog=True,
        )
        _, h = scipy.signal.freqs(b, a, worN=2.0 * np.pi * freqs)
        result = np.asarray(h, dtype=np.complex128)
        if len(_BUTTER_RESPONSE_CACHE) >= 512:
            _BUTTER_RESPONSE_CACHE.clear()
        _BUTTER_RESPONSE_CACHE[_cache_key] = result
        return result
    except (TypeError, ValueError, FloatingPointError, OverflowError):
        return np.ones(freqs.shape, dtype=np.complex128)


def _allpass2_complex_response(freqs_hz: np.ndarray, freq_hz: float, q: float) -> np.ndarray:
    freqs = np.asarray(freqs_hz, dtype=float)
    fc = _safe_float(freq_hz, float("nan"))
    q_v = _safe_float(q, float("nan"))
    if freqs.size == 0 or (not np.isfinite(fc)) or fc <= 0.0 or (not np.isfinite(q_v)) or q_v <= 0.0:
        return np.ones(freqs.shape, dtype=np.complex128)
    q_v = max(float(q_v), 0.1)  # guard: very low Q causes huge damping and near-zero denominator
    omega = 2.0 * np.pi * freqs
    omega_0 = 2.0 * np.pi * float(fc)
    s = 1j * omega
    damping = float(omega_0 / float(q_v))
    num = (s ** 2) - damping * s + (omega_0 ** 2)
    den = (s ** 2) + damping * s + (omega_0 ** 2)
    den = np.where(np.abs(den) < 1e-12, 1e-12 + 0j, den)
    return np.asarray(num / den, dtype=np.complex128)


def _apply_allpass_to_transfer(
    transfer: TransferData,
    *,
    freq_hz: float,
    q: float,
    label: str,
) -> TransferData:
    freqs = np.asarray(transfer.freqs_hz, dtype=float)
    spec = np.asarray(transfer.complex_spec, dtype=np.complex128).copy()
    if spec.size != freqs.size:
        spec = _interp_complex_response(transfer, freqs)
    spec *= _allpass2_complex_response(freqs, float(freq_hz), float(q))
    return _build_transfer_like(transfer, spec, label=label)


def _apply_gain_trim_to_transfer(
    transfer: TransferData,
    *,
    gain_trim_db: float,
    label: str,
) -> TransferData:
    gain_db = _safe_float(gain_trim_db, 0.0)
    freqs = np.asarray(transfer.freqs_hz, dtype=float)
    spec = np.asarray(transfer.complex_spec, dtype=np.complex128).copy()
    if spec.size != freqs.size:
        spec = _interp_complex_response(transfer, freqs)
    spec *= float(np.power(10.0, float(gain_db) / 20.0))
    return _build_transfer_like(transfer, spec, label=label)


def _apply_polarity_to_transfer(
    transfer: TransferData,
    *,
    invert: bool,
    label: str,
) -> TransferData:
    freqs = np.asarray(transfer.freqs_hz, dtype=float)
    spec = np.asarray(transfer.complex_spec, dtype=np.complex128).copy()
    if spec.size != freqs.size:
        spec = _interp_complex_response(transfer, freqs)
    if bool(invert):
        spec *= -1.0
    return _build_transfer_like(transfer, spec, label=label)


def _apply_delay_to_transfer(
    transfer: TransferData,
    *,
    delay_ms: float,
    label: str,
) -> TransferData:
    freqs = np.asarray(transfer.freqs_hz, dtype=float)
    spec = np.asarray(transfer.complex_spec, dtype=np.complex128).copy()
    if spec.size != freqs.size:
        spec = _interp_complex_response(transfer, freqs)
    delay_s = _safe_float(delay_ms, 0.0) / 1000.0
    spec *= np.exp(-1j * 2.0 * np.pi * freqs * float(delay_s))
    return _build_transfer_like(transfer, spec, label=label)


def _apply_branch_filters(
    transfer: TransferData,
    *,
    hpf_hz: float | None = None,
    hpf_order: int | None = None,
    lpf_hz: float | None = None,
    lpf_order: int | None = None,
    label: str,
) -> TransferData:
    freqs = np.asarray(transfer.freqs_hz, dtype=float)
    spec = np.asarray(transfer.complex_spec, dtype=np.complex128)
    if spec.size != freqs.size:
        spec = _interp_complex_response(transfer, freqs)
    else:
        spec = spec.copy()
    if hpf_hz is not None and hpf_order is not None:
        spec *= _butterworth_complex_response(
            freqs,
            float(hpf_hz),
            int(hpf_order),
            btype="high",
        )
    if lpf_hz is not None and lpf_order is not None:
        spec *= _butterworth_complex_response(
            freqs,
            float(lpf_hz),
            int(lpf_order),
            btype="low",
        )
    return _build_transfer_like(transfer, spec, label=label)


def _get_filtered_branches(
    bundle: BassIntegrationBundle,
    *,
    fc: float,
    xo_order: int,
    sub_hp_hz: float,
    sub_hp_order: int,
    sub_lpf: float,
    lpf_order: int,
) -> tuple:
    """Return (l_main_f, r_main_f, l_sub_f, r_sub_f) with branch filters applied.

    Results are cached per bundle instance and filter parameter set so that repeated
    scans over delay/gain/polarity/allpass reuse the filtered branches safely.
    """
    cache = _get_filtered_branch_cache(bundle)
    key = (
        float(fc), int(xo_order), float(sub_hp_hz), int(sub_hp_order),
        float(sub_lpf), int(lpf_order),
    )
    hit = cache.get(key)
    if hit is not None:
        return hit
    _abf = _get_pkg()._apply_branch_filters
    l_main_f = _abf(bundle.l_main, hpf_hz=fc, hpf_order=xo_order, label="L main + HPF trial")
    r_main_f = _abf(bundle.r_main, hpf_hz=fc, hpf_order=xo_order, label="R main + HPF trial")
    l_sub_f = _abf(
        bundle.l_sub,
        hpf_hz=sub_hp_hz,
        hpf_order=sub_hp_order,
        lpf_hz=sub_lpf,
        lpf_order=lpf_order,
        label="L sub + LPF/HPF trial",
    )
    r_sub_f = _abf(
        bundle.r_sub,
        hpf_hz=sub_hp_hz,
        hpf_order=sub_hp_order,
        lpf_hz=sub_lpf,
        lpf_order=lpf_order,
        label="R sub + LPF/HPF trial",
    )
    result = (l_main_f, r_main_f, l_sub_f, r_sub_f)
    if len(cache) >= 256:
        cache.clear()
    cache[key] = result
    return result
