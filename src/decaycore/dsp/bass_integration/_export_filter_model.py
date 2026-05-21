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
import scipy.signal

from ...io.measurement_bundle import TransferData
from ._filters import _allpass2_complex_response
from ._utils import _build_transfer_like, _interp_complex_response, _safe_float

_EXPORT_BIQUAD_Q = 0.707107


def _camilladsp_crossover_order_model(value: int | float | str | None, default: int = 2) -> int:
    try:
        order = int(round(float(value))) if value is not None else int(default)
    except Exception:
        order = int(default)
    if order <= 2:
        return 2
    if order % 2:
        order -= 1
    return max(2, order)


def _second_order_biquad_response(freqs_hz: np.ndarray, *, kind: str, freq_hz: float) -> np.ndarray:
    freqs = np.asarray(freqs_hz, dtype=float)
    fc = _safe_float(freq_hz, float("nan"))
    if freqs.size == 0 or not np.isfinite(fc) or fc <= 0.0:
        return np.ones(freqs.shape, dtype=np.complex128)
    w0 = 2.0 * np.pi * float(fc)
    q = float(_EXPORT_BIQUAD_Q)
    try:
        if str(kind).strip().lower() == "highpass":
            b = [1.0, 0.0, 0.0]
        else:
            b = [0.0, 0.0, w0 * w0]
        a = [1.0, w0 / q, w0 * w0]
        _, h = scipy.signal.freqs(b, a, worN=2.0 * np.pi * freqs)
        return np.asarray(h, dtype=np.complex128)
    except (TypeError, ValueError, FloatingPointError, OverflowError):
        return np.ones(freqs.shape, dtype=np.complex128)


def camilladsp_export_crossover_response(
    freqs_hz: np.ndarray,
    *,
    kind: str,
    freq_hz: float,
    order: int,
) -> np.ndarray:
    freqs = np.asarray(freqs_hz, dtype=float)
    kind_norm = str(kind or "").strip().lower()
    if kind_norm in ("high", "hp", "hpf"):
        kind_norm = "highpass"
    elif kind_norm in ("low", "lp", "lpf"):
        kind_norm = "lowpass"
    if kind_norm not in ("highpass", "lowpass"):
        return np.ones(freqs.shape, dtype=np.complex128)
    order_i = _camilladsp_crossover_order_model(order, 2)
    section = _second_order_biquad_response(freqs, kind=kind_norm, freq_hz=float(freq_hz))
    if order_i <= 2:
        return section
    cascades = max(1, int(order_i) // 2)
    return np.asarray(section ** cascades, dtype=np.complex128)


def apply_direct_dac_export_branch_model(
    transfer: TransferData,
    *,
    hpf_hz: float | None = None,
    hpf_order: int | None = None,
    lpf_hz: float | None = None,
    lpf_order: int | None = None,
    gain_trim_db: float = 0.0,
    delay_ms: float = 0.0,
    polarity_invert: bool = False,
    allpass_freq_hz: float | None = None,
    allpass_q: float | None = None,
    label: str = "",
) -> TransferData:
    freqs = np.asarray(transfer.freqs_hz, dtype=float)
    spec = np.asarray(transfer.complex_spec, dtype=np.complex128)
    spec = _interp_complex_response(transfer, freqs) if spec.size != freqs.size else spec.copy()
    if bool(polarity_invert):
        spec *= -1.0
    spec *= float(np.power(10.0, _safe_float(gain_trim_db, 0.0) / 20.0))
    delay_s = _safe_float(delay_ms, 0.0) / 1000.0
    if abs(float(delay_s)) > 1e-15:
        spec *= np.exp(-1j * 2.0 * np.pi * freqs * float(delay_s))
    if hpf_hz is not None and hpf_order is not None:
        spec *= camilladsp_export_crossover_response(
            freqs,
            kind="highpass",
            freq_hz=float(hpf_hz),
            order=int(hpf_order),
        )
    if lpf_hz is not None and lpf_order is not None:
        spec *= camilladsp_export_crossover_response(
            freqs,
            kind="lowpass",
            freq_hz=float(lpf_hz),
            order=int(lpf_order),
        )
    ap_freq = _safe_float(allpass_freq_hz, float("nan"))
    ap_q = _safe_float(allpass_q, float("nan"))
    if np.isfinite(ap_freq) and ap_freq > 0.0 and np.isfinite(ap_q) and ap_q > 0.0:
        spec *= _allpass2_complex_response(freqs, float(ap_freq), float(ap_q))
    return _build_transfer_like(transfer, spec, label=label)
