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

from typing import Any

import numpy as np

from ...io.measurement_bundle import TransferData
from ._constants import AVR_CROSSOVER_CANDIDATES, DIRECT_DAC_CROSSOVER_STEP_HZ


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return float(out)
    except (TypeError, ValueError, OverflowError):
        pass
    return float(default)


def _band_mask(freqs_hz: np.ndarray, lo_hz: float, hi_hz: float) -> np.ndarray:
    try:
        f = np.asarray(freqs_hz, dtype=float)
    except Exception:
        return np.zeros(0, dtype=bool)
    lo = _safe_float(lo_hz, float("nan"))
    hi = _safe_float(hi_hz, float("nan"))
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
        return np.zeros(f.shape, dtype=bool)
    return np.isfinite(f) & (f >= float(lo)) & (f <= float(hi))


def _normalize_candidate_frequencies(candidates: Any) -> tuple[float, ...]:
    if candidates is None:
        return ()
    out: list[float] = []
    seen: set[float] = set()
    try:
        iterator = tuple(candidates)
    except TypeError:
        return ()
    for candidate in iterator:
        fc = _safe_float(candidate, float("nan"))
        if not np.isfinite(fc) or fc <= 0.0:
            continue
        key = float(round(float(fc), 6))
        if key in seen:
            continue
        seen.add(key)
        out.append(float(fc))
    return tuple(out)


def _normalize_candidate_q_values(candidates: Any) -> tuple[float, ...]:
    if candidates is None:
        return ()
    out: list[float] = []
    seen: set[float] = set()
    try:
        iterator = tuple(candidates)
    except TypeError:
        return ()
    for candidate in iterator:
        q = _safe_float(candidate, float("nan"))
        if not np.isfinite(q) or q <= 0.0:
            continue
        key = float(round(float(q), 6))
        if key in seen:
            continue
        seen.add(key)
        out.append(float(q))
    return tuple(out)


def _default_direct_dac_crossover_candidates() -> tuple[float, ...]:
    lo_hz = float(AVR_CROSSOVER_CANDIDATES[0])
    hi_hz = float(AVR_CROSSOVER_CANDIDATES[-1])
    step_hz = float(DIRECT_DAC_CROSSOVER_STEP_HZ)
    steps = max(0, int(round((hi_hz - lo_hz) / step_hz)))
    return tuple(float(lo_hz + step_hz * idx) for idx in range(steps + 1))


def normalize_sub_combine_mode(mode: Any) -> str:
    value = str(mode or "average").strip().lower().replace("-", "_")
    aliases = {
        "avg": "average",
        "summed": "sum",
        "aligned": "aligned_sum",
        "alignedsum": "aligned_sum",
    }
    value = str(aliases.get(value, value))
    if value not in ("average", "sum", "aligned_sum"):
        value = "average"
    return str(value)


def _interp_complex_response(source: TransferData, target_freqs_hz: np.ndarray) -> np.ndarray:
    src_f = np.asarray(source.freqs_hz, dtype=float)
    src_c = np.asarray(source.complex_spec, dtype=np.complex128)
    dst_f = np.asarray(target_freqs_hz, dtype=float)
    if src_f.size < 2 or src_c.size != src_f.size or dst_f.size < 1:
        return np.zeros(dst_f.shape, dtype=np.complex128)
    if src_f.size == dst_f.size and np.allclose(src_f, dst_f, rtol=0.0, atol=1e-9):
        return src_c.astype(np.complex128, copy=False)
    re = np.interp(dst_f, src_f, np.real(src_c), left=np.real(src_c[0]), right=np.real(src_c[-1]))
    im = np.interp(dst_f, src_f, np.imag(src_c), left=np.imag(src_c[0]), right=np.imag(src_c[-1]))
    result = np.asarray(re + 1j * im, dtype=np.complex128)
    return result


def _build_transfer_like(
    template: TransferData,
    complex_spec: np.ndarray,
    *,
    label: str,
) -> TransferData:
    spec = np.asarray(complex_spec, dtype=np.complex128)
    freqs = np.asarray(template.freqs_hz, dtype=float)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(spec), 1e-12))
    phase_deg = np.rad2deg(np.unwrap(np.angle(spec)))
    return TransferData(
        freqs_hz=freqs,
        complex_spec=spec,
        mag_db=np.asarray(mag_db, dtype=float),
        phase_deg=np.asarray(phase_deg, dtype=float),
        sample_rate=int(template.sample_rate),
        label=str(label or ""),
    )


def _sum_component_specs(template: TransferData, components: tuple[TransferData, ...]) -> np.ndarray:
    freqs = np.asarray(template.freqs_hz, dtype=float)
    total_spec = np.asarray(template.complex_spec, dtype=np.complex128).copy()
    for component in components:
        total_spec += _interp_complex_response(component, freqs)
    return np.asarray(total_spec, dtype=np.complex128)
