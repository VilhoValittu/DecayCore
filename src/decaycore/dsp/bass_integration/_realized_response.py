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
from typing import Any

import numpy as np
import scipy.signal

from ...io.measurement_bundle import BassIntegrationBundle, TransferData
from ._sub_combine import build_bundle_combined_sub_transfer, sum_complex_responses
from ._utils import _build_transfer_like, normalize_sub_combine_mode

ROBUST_PERTURBATION_POLICY_VERSION = 1
ROBUST_GAIN_OFFSETS_DB: tuple[float, ...] = (-1.0, 0.0, 1.0)
ROBUST_DELAY_OFFSETS_MS: tuple[float, ...] = (-0.5, 0.0, 0.5)
ROBUST_P90_PERCENTILE = 90.0
ROBUST_NOMINAL_WEIGHT = 0.5


def _fir_hash(values: Any | None) -> str:
    if values is None:
        return "none"
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return "empty"
    finite = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return hashlib.sha256(np.ascontiguousarray(finite).view(np.uint8)).hexdigest()[:20]


def realized_fir_signature(
    l_fir: Any | None,
    r_fir: Any | None,
    sub_fir: Any | None,
    *,
    sample_rate: int,
) -> tuple[str, str, str, int, int]:
    return (
        _fir_hash(l_fir),
        _fir_hash(r_fir),
        _fir_hash(sub_fir),
        int(sample_rate),
        int(ROBUST_PERTURBATION_POLICY_VERSION),
    )


def _fir_response_on_grid(fir: Any | None, freqs_hz: np.ndarray, sample_rate: int) -> np.ndarray:
    freqs = np.asarray(freqs_hz, dtype=float).reshape(-1)
    if fir is None:
        return np.ones(freqs.shape, dtype=np.complex128)
    taps = np.asarray(fir, dtype=float).reshape(-1)
    if taps.size == 0 or not np.all(np.isfinite(taps)):
        raise ValueError("Bass integration FIR must be non-empty and finite")
    fs = int(sample_rate)
    if fs <= 0:
        raise ValueError("Bass integration FIR sample rate must be positive")
    if freqs.size >= 2:
        diffs = np.diff(freqs)
        df = float(np.median(diffs))
        if np.isfinite(df) and df > 0.0 and np.allclose(diffs, df, rtol=1e-8, atol=1e-10):
            n_fft = int(round(float(fs) / df))
            bins = np.rint(freqs / df).astype(np.int64)
            aligned = np.allclose(freqs, bins.astype(float) * df, rtol=0.0, atol=max(1e-9, df * 1e-7))
            if (
                aligned
                and n_fft >= taps.size
                and n_fft > 0
                and bins.size
                and int(np.min(bins)) >= 0
                and int(np.max(bins)) <= n_fft // 2
            ):
                response = np.fft.rfft(taps, n=n_fft)
                return np.asarray(response[bins], dtype=np.complex128)
    omega = 2.0 * np.pi * np.clip(freqs, 0.0, 0.5 * float(fs)) / float(fs)
    _, response = scipy.signal.freqz(taps, worN=omega)
    return np.asarray(response, dtype=np.complex128)


def _apply_fir(
    transfer: TransferData,
    fir: Any | None,
    *,
    sample_rate: int,
    label: str,
    response: np.ndarray | None = None,
) -> TransferData:
    freqs = np.asarray(transfer.freqs_hz, dtype=float)
    spec = np.asarray(transfer.complex_spec, dtype=np.complex128)
    if spec.size != freqs.size:
        raise ValueError("Bass integration transfer frequency and spectrum shapes differ")
    fir_response = (
        np.asarray(response, dtype=np.complex128)
        if response is not None
        else _fir_response_on_grid(fir, freqs, sample_rate)
    )
    if fir_response.shape != freqs.shape:
        raise ValueError("Bass integration FIR response shape differs from transfer grid")
    return _build_transfer_like(transfer, spec * fir_response, label=label)


def build_realized_bass_integration_bundle(
    bundle: BassIntegrationBundle,
    *,
    l_fir: Any | None,
    r_fir: Any | None,
    sub_fir: Any | None,
    sample_rate: int | None = None,
    sub_combine_mode: str = "average",
) -> BassIntegrationBundle:
    """Apply exported FIRs to measured branches before Direct-DAC IIR evaluation."""
    fs = int(sample_rate or bundle.l_main.sample_rate)
    signature = realized_fir_signature(l_fir, r_fir, sub_fir, sample_rate=fs)
    combine_mode = normalize_sub_combine_mode(sub_combine_mode)
    cache_key = (*signature, combine_mode)
    try:
        cache = object.__getattribute__(bundle, "_decaycore_realized_bass_cache")
    except AttributeError:
        cache = {}
        object.__setattr__(bundle, "_decaycore_realized_bass_cache", cache)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    l_response = _fir_response_on_grid(l_fir, np.asarray(bundle.l_main.freqs_hz, dtype=float), fs)
    r_response = _fir_response_on_grid(r_fir, np.asarray(bundle.r_main.freqs_hz, dtype=float), fs)
    sub_response_l = _fir_response_on_grid(sub_fir, np.asarray(bundle.l_sub.freqs_hz, dtype=float), fs)
    sub_freqs_r = np.asarray(bundle.r_sub.freqs_hz, dtype=float)
    sub_response_r = (
        sub_response_l
        if np.array_equal(sub_freqs_r, np.asarray(bundle.l_sub.freqs_hz, dtype=float))
        else _fir_response_on_grid(sub_fir, sub_freqs_r, fs)
    )
    l_main = _apply_fir(
        bundle.l_main,
        l_fir,
        sample_rate=fs,
        response=l_response,
        label="L main measured x exported FIR",
    )
    r_main = _apply_fir(
        bundle.r_main,
        r_fir,
        sample_rate=fs,
        response=r_response,
        label="R main measured x exported FIR",
    )
    l_sub = _apply_fir(
        bundle.l_sub,
        sub_fir,
        sample_rate=fs,
        response=sub_response_l,
        label="Sub bus measured x exported FIR",
    )
    r_sub = _apply_fir(
        bundle.r_sub,
        sub_fir,
        sample_rate=fs,
        response=sub_response_r,
        label="Sub bus slot 2 measured x exported FIR",
    )
    provisional = BassIntegrationBundle(
        l_main=l_main,
        r_main=r_main,
        l_sub=l_sub,
        r_sub=r_sub,
        l_total=bundle.l_total,
        r_total=bundle.r_total,
        avr_crossover_hz=float(bundle.avr_crossover_hz),
        profile=str(bundle.profile),
        diagnostics=dict(bundle.diagnostics or {}),
    )
    l_sub_bus, l_diag = build_bundle_combined_sub_transfer(
        provisional, channel="l", mode=combine_mode, label="L realized normalized sub bus"
    )
    r_sub_bus, r_diag = build_bundle_combined_sub_transfer(
        provisional, channel="r", mode=combine_mode, label="R realized normalized sub bus"
    )
    diagnostics = dict(bundle.diagnostics or {})
    diagnostics.update(
        {
            "realized_response": True,
            "realized_fir_signature": ":".join(str(part) for part in signature),
            "sub_scaling_assumption": "single_bus_average_normalized",
            "sub_coherence_assumption": "measured_complex_with_bounded_gain_delay_perturbation",
            "robust_perturbation_policy_v": int(ROBUST_PERTURBATION_POLICY_VERSION),
            **dict(l_diag or {}),
            **{f"r_{key}": value for key, value in dict(r_diag or {}).items()},
        }
    )
    realized = BassIntegrationBundle(
        l_main=l_main,
        r_main=r_main,
        l_sub=l_sub,
        r_sub=r_sub,
        l_total=sum_complex_responses(l_main, l_sub_bus, label="L realized pre-IIR total"),
        r_total=sum_complex_responses(r_main, r_sub_bus, label="R realized pre-IIR total"),
        avr_crossover_hz=float(bundle.avr_crossover_hz),
        profile=str(bundle.profile),
        diagnostics=diagnostics,
    )
    if len(cache) >= 16:
        cache.clear()
    cache[cache_key] = realized
    return realized


__all__ = [
    "ROBUST_DELAY_OFFSETS_MS",
    "ROBUST_GAIN_OFFSETS_DB",
    "ROBUST_NOMINAL_WEIGHT",
    "ROBUST_P90_PERCENTILE",
    "ROBUST_PERTURBATION_POLICY_VERSION",
    "build_realized_bass_integration_bundle",
    "realized_fir_signature",
]
