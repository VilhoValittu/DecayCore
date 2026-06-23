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

from ...io.measurement_bundle import BassIntegrationBundle, TransferData
from ..bass_cache import _combined_sub_cache_key, _get_combined_sub_cache
from ._constants import COMBINED_SUB_ALIGNMENT_MAX_LAG_MS
from ._utils import (
    _band_mask,
    _build_transfer_like,
    _get_pkg,
    _interp_complex_response,
    _sum_component_specs,
    normalize_sub_combine_mode,
)


def _transfer_is_effectively_silent(transfer: TransferData | None) -> bool:
    if transfer is None:
        return True
    try:
        spec = np.asarray(transfer.complex_spec, dtype=np.complex128)
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
        return True
    return spec.size == 0 or float(np.max(np.abs(spec))) <= 1e-12


def _bundle_sub_slot_names(bundle: BassIntegrationBundle) -> tuple[str, ...]:
    diagnostics = dict(getattr(bundle, "diagnostics", {}) or {})
    raw_slots = diagnostics.get("sub_slots_present", ())
    slots = [str(slot) for slot in tuple(raw_slots) if str(slot) in ("l_sub", "r_sub")]
    if slots:
        return tuple(slots)
    if not _transfer_is_effectively_silent(getattr(bundle, "r_sub", None)):
        return ("l_sub", "r_sub")
    return ("l_sub",)


def _bundle_active_sub_transfers(bundle: BassIntegrationBundle) -> tuple[TransferData, ...]:
    mapping = {
        "l_sub": getattr(bundle, "l_sub", None),
        "r_sub": getattr(bundle, "r_sub", None),
    }
    out: list[TransferData] = []
    for slot in _bundle_sub_slot_names(bundle):
        transfer = mapping.get(slot)
        if isinstance(transfer, TransferData):
            out.append(transfer)
    if out:
        return tuple(out)
    fallback = mapping.get("l_sub")
    return (fallback,) if isinstance(fallback, TransferData) else ()


def _band_level_delta_db(
    freqs_hz: np.ndarray,
    combined_spec: np.ndarray,
    average_spec: np.ndarray,
    *,
    lo_hz: float,
    hi_hz: float,
) -> float:
    freqs = np.asarray(freqs_hz, dtype=float)
    mask = _band_mask(freqs, lo_hz, hi_hz)
    if int(np.count_nonzero(mask)) < 3:
        return float("nan")
    combined_mag = np.abs(np.asarray(combined_spec, dtype=np.complex128)[mask])
    average_mag = np.abs(np.asarray(average_spec, dtype=np.complex128)[mask])
    finite = (
        np.isfinite(combined_mag)
        & np.isfinite(average_mag)
    )
    if int(np.count_nonzero(finite)) < 3:
        return float("nan")
    combined_mag = combined_mag[finite]
    average_mag = average_mag[finite]
    # Use band-average power instead of the median of per-bin dB deltas.
    # Median-of-deltas can collapse to ~0 dB even when alignment removes a large
    # cancellation notch and materially raises the average in-band level.
    combined_power = float(np.mean(np.square(np.maximum(combined_mag, 1e-12))))
    average_power = float(np.mean(np.square(np.maximum(average_mag, 1e-12))))
    return float(10.0 * np.log10(max(combined_power, 1e-24) / max(average_power, 1e-24)))


def _preserve_precombined_dual_sub_diagnostics(
    bundle: BassIntegrationBundle,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Keep dual-sub preprocessing deltas when the bundle already contains one virtual sub.

    After dual-sub peak alignment, the bundle intentionally collapses to one combined
    acoustic sub reference. Recombining that single virtual sub later would always
    report 0 dB delta, so preserve the original dual-sub alignment deltas instead.
    """
    bundle_diag = dict(getattr(bundle, "diagnostics", {}) or {})
    if not bool(bundle_diag.get("dual_sub_preprocessing_applied", False)):
        return diagnostics
    if len(_bundle_sub_slot_names(bundle)) != 1:
        return diagnostics

    out = dict(diagnostics or {})
    for key in (
        "sub_combined_level_delta_db_20_120",
        "sub_combined_level_delta_db_30_90",
        "predicted_sub_array_gain_db_30_100",
    ):
        try:
            value = float(bundle_diag.get(key, float("nan")))
        except (
            TypeError,
            ValueError,
            KeyError,
            AttributeError,
        ):
            continue
        if np.isfinite(value):
            out[key] = float(value)
    return out


def _phase_rms_deg(
    freqs_hz: np.ndarray,
    spec_a: np.ndarray,
    spec_b: np.ndarray,
    *,
    lo_hz: float = 30.0,
    hi_hz: float = 100.0,
) -> float:
    freqs = np.asarray(freqs_hz, dtype=float)
    mask = _band_mask(freqs, lo_hz, hi_hz)
    if int(np.count_nonzero(mask)) < 3:
        return float("nan")
    a = np.asarray(spec_a, dtype=np.complex128)[mask]
    b = np.asarray(spec_b, dtype=np.complex128)[mask]
    finite = np.isfinite(a.real) & np.isfinite(a.imag) & np.isfinite(b.real) & np.isfinite(b.imag)
    if int(np.count_nonzero(finite)) < 3:
        return float("nan")
    delta = np.angle(a[finite] * np.conj(b[finite]))
    return float(np.rad2deg(np.sqrt(np.mean(np.square(delta)))))


def _phase_aligned_sub_delay_ms(
    ref_spec: np.ndarray,
    moving_spec: np.ndarray,
    freqs_hz: np.ndarray,
    *,
    max_delay_ms: float = COMBINED_SUB_ALIGNMENT_MAX_LAG_MS,
    lo_hz: float = 30.0,
    hi_hz: float = 100.0,
) -> tuple[float, float]:
    freqs = np.asarray(freqs_hz, dtype=float)
    mask = _band_mask(freqs, lo_hz, hi_hz)
    if int(np.count_nonzero(mask)) < 3:
        return 0.0, float("nan")
    ref = np.asarray(ref_spec, dtype=np.complex128)[mask]
    moving = np.asarray(moving_spec, dtype=np.complex128)[mask]
    freq_band = freqs[mask]
    finite = np.isfinite(ref.real) & np.isfinite(ref.imag) & np.isfinite(moving.real) & np.isfinite(moving.imag)
    if int(np.count_nonzero(finite)) < 3:
        return 0.0, float("nan")
    ref = ref[finite]
    moving = moving[finite]
    freq_band = freq_band[finite]

    def _best_for_grid(delay_grid_ms: np.ndarray) -> tuple[float, float]:
        phase = np.exp(-1j * 2.0 * np.pi * freq_band[None, :] * (delay_grid_ms[:, None] / 1000.0))
        aligned = moving[None, :] * phase
        delta = np.angle(ref[None, :] * np.conj(aligned))
        rms = np.rad2deg(np.sqrt(np.mean(np.square(delta), axis=1)))
        finite_rms = np.isfinite(rms)
        if not bool(np.any(finite_rms)):
            return 0.0, float("nan")
        best_idx = int(np.argmin(np.where(finite_rms, rms, np.inf)))
        return float(delay_grid_ms[best_idx]), float(rms[best_idx])

    max_delay = float(max(0.0, max_delay_ms))
    coarse = np.arange(-max_delay, max_delay + 0.5, 1.0, dtype=float)
    best_delay, _best_rms = _best_for_grid(coarse)
    refine_lo = max(-max_delay, best_delay - 1.0)
    refine_hi = min(max_delay, best_delay + 1.0)
    refine = np.arange(refine_lo, refine_hi + 0.05, 0.1, dtype=float)
    return _best_for_grid(refine)


def build_combined_sub_transfer(
    template: TransferData,
    *subs: TransferData,
    mode: str = "average",
    max_lag_ms: float = COMBINED_SUB_ALIGNMENT_MAX_LAG_MS,
    min_confidence: float = 0.35,
    label: str = "",
) -> tuple[TransferData, dict[str, Any]]:
    freqs = np.asarray(template.freqs_hz, dtype=float)
    fs = int(template.sample_rate)
    mode_norm = normalize_sub_combine_mode(mode)
    active_subs = tuple(sub for sub in subs if isinstance(sub, TransferData))
    if not active_subs:
        zero_spec = np.zeros(freqs.shape, dtype=np.complex128)
        return (
            _build_transfer_like(template, zero_spec, label=label),
            {
                "sub_combine_mode": str(mode_norm),
                "sub_slot_count": 0,
                "sub_topology": "no_sub",
                "whether_alignment_applied": False,
                "alignment_offset_ms": 0.0,
                "alignment_confidence": 0.0,
                "sub_array_delay_ms": 0.0,
                "sub1_delay_ms": 0.0,
                "sub2_delay_ms": 0.0,
                "sub_combined_level_delta_db_20_120": 0.0,
                "sub_combined_level_delta_db_30_90": 0.0,
                "predicted_sub_array_gain_db_30_100": 0.0,
            },
        )

    specs = [_interp_complex_response(sub, freqs) for sub in active_subs]
    sum_spec = np.sum(np.asarray(specs, dtype=np.complex128), axis=0)
    avg_spec = np.asarray(sum_spec / float(len(specs)), dtype=np.complex128)

    alignment_offsets_ms: list[float] = []
    alignment_confidences: list[float] = []
    alignment_applied = False
    if mode_norm == "aligned_sum" and len(specs) > 1:
        ref_spec = np.asarray(specs[0], dtype=np.complex128)
        aligned_specs = [ref_spec]
        _xcorr_fn = _get_pkg()._xcorr_lag_from_spectra
        for spec in specs[1:]:
            offset_ms, confidence = _xcorr_fn(
                ref_spec,
                np.asarray(spec, dtype=np.complex128),
                freqs,
                fs,
                float(max_lag_ms),
            )
            should_apply = bool(
                np.isfinite(confidence)
                and confidence >= float(min_confidence)
                and np.isfinite(offset_ms)
                and 0.10 < abs(offset_ms) < float(max_lag_ms)
            )
            if should_apply:
                spec = _apply_phase_advance(spec, freqs, float(offset_ms) / 1000.0)
                alignment_applied = True
            aligned_specs.append(np.asarray(spec, dtype=np.complex128))
            alignment_offsets_ms.append(float(offset_ms))
            alignment_confidences.append(float(confidence))
        sum_spec = np.sum(np.asarray(aligned_specs, dtype=np.complex128), axis=0)

    if mode_norm == "average":
        combined_spec = avg_spec
    else:
        combined_spec = np.asarray(sum_spec, dtype=np.complex128)

    applied_offsets = [
        float(v)
        for v in alignment_offsets_ms
        if np.isfinite(v) and abs(float(v)) > 0.10
    ]
    alignment_offset_ms = (
        float(np.mean(np.asarray(applied_offsets, dtype=float)))
        if applied_offsets
        else (float(alignment_offsets_ms[0]) if alignment_offsets_ms else 0.0)
    )
    valid_confidences = [float(v) for v in alignment_confidences if np.isfinite(v)]
    alignment_confidence = float(max(valid_confidences)) if valid_confidences else 0.0
    diagnostics = {
        "sub_combine_mode": str(mode_norm),
        "sub_slot_count": int(len(active_subs)),
        "sub_topology": "single_sub_bus" if len(active_subs) == 1 else f"dual_sub_{mode_norm}",
        "whether_alignment_applied": bool(alignment_applied),
        "alignment_offset_ms": float(alignment_offset_ms),
        "alignment_confidence": float(alignment_confidence),
        "sub_array_delay_ms": 0.0,
        "sub1_delay_ms": 0.0,
        "sub2_delay_ms": float(alignment_offset_ms if alignment_applied else 0.0),
        "sub_array_phase_rms_deg_30_100": (
            _phase_rms_deg(freqs, specs[0], specs[1], lo_hz=30.0, hi_hz=100.0)
            if len(specs) > 1
            else 0.0
        ),
        "sub_combined_level_delta_db_20_120": _band_level_delta_db(
            freqs,
            combined_spec,
            avg_spec,
            lo_hz=20.0,
            hi_hz=120.0,
        ),
        "sub_combined_level_delta_db_30_90": _band_level_delta_db(
            freqs,
            combined_spec,
            avg_spec,
            lo_hz=30.0,
            hi_hz=90.0,
        ),
        "predicted_sub_array_gain_db_30_100": _band_level_delta_db(
            freqs,
            combined_spec,
            avg_spec,
            lo_hz=30.0,
            hi_hz=100.0,
        ),
    }
    return _build_transfer_like(template, combined_spec, label=label), diagnostics


def prepare_dual_sub_peak_aligned_average(
    sub1: TransferData,
    sub2: TransferData,
    *,
    sub1_peak_samples: int,
    sub2_peak_samples: int,
    label: str = "Direct-DAC dual-sub combined reference",
) -> tuple[TransferData, dict[str, Any]]:
    freqs = np.asarray(sub1.freqs_hz, dtype=float)
    fs = int(sub1.sample_rate)
    if fs <= 0:
        fs = int(sub2.sample_rate)
    if fs <= 0:
        fs = 48000

    spec1 = np.asarray(sub1.complex_spec, dtype=np.complex128)
    if spec1.size != freqs.size:
        spec1 = _interp_complex_response(sub1, freqs)
    else:
        spec1 = spec1.copy()
    spec2 = _interp_complex_response(sub2, freqs)

    peak1 = int(sub1_peak_samples)
    peak2 = int(sub2_peak_samples)
    delay_samples = int(peak1 - peak2)
    peak_delay_ms = float(delay_samples) / float(fs) * 1000.0
    phase_delay_ms, phase_rms = _phase_aligned_sub_delay_ms(
        spec1,
        spec2,
        freqs,
        max_delay_ms=COMBINED_SUB_ALIGNMENT_MAX_LAG_MS,
        lo_hz=30.0,
        hi_hz=100.0,
    )
    delay_ms = (
        float(phase_delay_ms)
        if np.isfinite(phase_rms)
        else float(peak_delay_ms)
    )
    delay_s = float(delay_ms) / 1000.0
    spec2_aligned = spec2 * np.exp(-1j * 2.0 * np.pi * freqs * float(delay_s))
    combined_spec = np.asarray((spec1 + spec2_aligned) / 2.0, dtype=np.complex128)
    raw_average_spec = np.asarray((spec1 + spec2) / 2.0, dtype=np.complex128)

    peak1_ms = float(peak1) / float(fs) * 1000.0
    peak2_ms = float(peak2) / float(fs) * 1000.0
    # Alignment is reliable when the inferred delay is small relative to the XO period.
    # > 10 ms direct-peak offset is suspicious in a typical room setup: the peak may be
    # a reflection rather than the direct sound, which would misalign the integration.
    alignment_reliable = abs(delay_ms) <= 10.0
    diagnostics = {
        "sub_topology": "dual_sub_vector_average_reference",
        "dual_sub_preprocessing_applied": True,
        "dual_sub_preprocessing_version": 1,
        "dual_sub_sub1_peak_samples": int(peak1),
        "dual_sub_sub2_peak_samples": int(peak2),
        "dual_sub_sub1_peak_ms": float(peak1_ms),
        "dual_sub_sub2_peak_ms": float(peak2_ms),
        "dual_sub_relative_delay_samples": int(delay_samples),
        "dual_sub_peak_relative_delay_ms": float(peak_delay_ms),
        "dual_sub_relative_delay_ms": float(delay_ms),
        "dual_sub_phase_refined": bool(np.isfinite(phase_rms)),
        "dual_sub_phase_refined_rms_deg_30_100": float(phase_rms),
        "dual_sub_sub1_delay_ms": 0.0,
        "dual_sub_sub2_delay_ms": float(delay_ms),
        "sub1_delay_ms": 0.0,
        "sub2_delay_ms": float(delay_ms),
        "sub_array_delay_ms": 0.0,
        "dual_sub_alignment_reliable": bool(alignment_reliable),
        "dual_sub_combined_method": "peak_aligned_complex_vector_average",
        "dual_sub_effective_sub_slot_count": 1,
        "dual_sub_per_sub_optimization": False,
        "dual_sub_topology_label": "dual-sub vector-average reference",
        "sub_array_phase_rms_deg_30_100": _phase_rms_deg(freqs, spec1, spec2_aligned, lo_hz=30.0, hi_hz=100.0),
        "sub_combined_level_delta_db_20_120": _band_level_delta_db(
            freqs,
            combined_spec,
            raw_average_spec,
            lo_hz=20.0,
            hi_hz=120.0,
        ),
        "sub_combined_level_delta_db_30_90": _band_level_delta_db(
            freqs,
            combined_spec,
            raw_average_spec,
            lo_hz=30.0,
            hi_hz=90.0,
        ),
        "predicted_sub_array_gain_db_30_100": _band_level_delta_db(
            freqs,
            combined_spec,
            raw_average_spec,
            lo_hz=30.0,
            hi_hz=100.0,
        ),
    }
    return _build_transfer_like(sub1, combined_spec, label=label), diagnostics


def build_bundle_combined_sub_transfer(
    bundle: BassIntegrationBundle,
    *,
    channel: str = "l",
    mode: str | None = None,
    label: str = "",
    max_lag_ms: float = COMBINED_SUB_ALIGNMENT_MAX_LAG_MS,
    min_confidence: float = 0.35,
) -> tuple[TransferData, dict[str, Any]]:
    ch = str(channel or "l").strip().lower()
    template = bundle.r_main if ch == "r" else bundle.l_main
    mode_norm = normalize_sub_combine_mode(mode or dict(getattr(bundle, "diagnostics", {}) or {}).get("sub_combine_mode"))
    # Per-bundle cache: same combined sub is needed by diagnostics AND gd_continuity
    _csub_cache = _get_combined_sub_cache(bundle)
    _csub_key = _combined_sub_cache_key(ch, mode_norm, float(max_lag_ms), float(min_confidence))
    _hit = _csub_cache.get(_csub_key)
    if _hit is not None:
        return _hit
    combined, diagnostics = build_combined_sub_transfer(
        template,
        *_bundle_active_sub_transfers(bundle),
        mode=mode_norm,
        max_lag_ms=float(max_lag_ms),
        min_confidence=float(min_confidence),
        label=label or ("R combined sub" if ch == "r" else "L combined sub"),
    )
    diagnostics = _preserve_precombined_dual_sub_diagnostics(bundle, diagnostics)
    result = (combined, diagnostics)
    if len(_csub_cache) >= 64:
        _csub_cache.clear()
    _csub_cache[_csub_key] = result
    return result


def sum_complex_responses(
    main: TransferData,
    *subs: TransferData,
    label: str = "",
) -> TransferData:
    total_spec = _sum_component_specs(main, tuple(subs))
    return _build_transfer_like(main, total_spec, label=label)


def avg_complex_responses(
    main: TransferData,
    *subs: TransferData,
    label: str = "",
) -> TransferData:
    """Like sum_complex_responses but averages sub contributions instead of summing.

    Use when multiple sub measurements each capture the full combined sub response
    (e.g. AVR mono sub routing) so that averaging avoids double-counting.
    """
    freqs = np.asarray(main.freqs_hz, dtype=float)
    sub_spec = np.zeros(freqs.shape, dtype=np.complex128)
    for sub in subs:
        sub_spec += _interp_complex_response(sub, freqs)
    if len(subs) > 1:
        sub_spec /= len(subs)
    main_spec = np.asarray(main.complex_spec, dtype=np.complex128)
    return _build_transfer_like(main, main_spec + sub_spec, label=label)


def _xcorr_lag_from_spectra(
    main_spec: np.ndarray,
    sub_spec: np.ndarray,
    freqs_hz: np.ndarray,
    fs: int,
    max_lag_ms: float,
) -> tuple[float, float]:
    """Laskee cross-power spektrin kautta xcorr-viiveen main vs sub välillä.
    Palauttaa (offset_ms, confidence).
    offset_ms > 0 tarkoittaa sub myöhässä mainista.
    """
    A = np.asarray(main_spec, dtype=np.complex128)
    B = np.asarray(sub_spec, dtype=np.complex128)
    n = A.size
    if n < 4:
        return 0.0, 0.0

    C = A * np.conj(B)
    xcorr = np.fft.irfft(C, n=2 * (n - 1))
    n_xcorr = len(xcorr)

    max_lag_samp = min(int(round(max_lag_ms * float(fs) / 1000.0)), n_xcorr // 2 - 1)
    if max_lag_samp < 1:
        return 0.0, 0.0

    search = np.concatenate([xcorr[:max_lag_samp + 1], xcorr[n_xcorr - max_lag_samp:]])
    best_idx = int(np.argmax(np.abs(search)))
    best_value = float(search[best_idx])

    if best_idx <= max_lag_samp:
        lag_samp = best_idx
    else:
        lag_samp = -(n_xcorr - best_idx)

    offset_ms = float(lag_samp) / float(fs) * 1000.0

    energy_a = float(np.sum(np.abs(A) ** 2))
    energy_b = float(np.sum(np.abs(B) ** 2))
    denom = float(np.sqrt(max(energy_a * energy_b, 1e-60)))
    confidence = float(np.clip(abs(best_value) / denom, 0.0, 1.0))

    return offset_ms, confidence


def _apply_phase_advance(
    spec: np.ndarray,
    freqs_hz: np.ndarray,
    advance_s: float,
) -> np.ndarray:
    """Siirtää spektrin signaalia advance_s sekuntia aiemmaksi (positiivinen = advance).
    spec * exp(+j * 2π * f * advance_s)
    """
    freqs = np.asarray(freqs_hz, dtype=float)
    return np.asarray(spec, dtype=np.complex128) * np.exp(1j * 2.0 * np.pi * freqs * float(advance_s))


def sum_complex_responses_aligned(
    main: TransferData,
    *subs: TransferData,
    max_lag_ms: float = COMBINED_SUB_ALIGNMENT_MAX_LAG_MS,
    min_confidence: float = 0.35,
    label: str = "",
) -> tuple[TransferData, dict[str, Any]]:
    """Kuten sum_complex_responses, mutta korjaa ensin sub-kokonaisuuden
    xcorr-viiveen suhteessa mainiin ennen summaa.

    Palauttaa (total_TransferData, diagnostics_dict).
    """
    freqs = np.asarray(main.freqs_hz, dtype=float)
    fs = int(main.sample_rate)

    combined_sub = _sum_sub_components(main, *subs, label="combined sub")
    sub_spec = _interp_complex_response(combined_sub, freqs)
    main_spec = np.asarray(main.complex_spec, dtype=np.complex128)

    offset_ms, confidence = _xcorr_lag_from_spectra(main_spec, sub_spec, freqs, fs, max_lag_ms)

    apply = (
        confidence >= float(min_confidence)
        and abs(offset_ms) > 0.1
        and abs(offset_ms) < float(max_lag_ms)
    )

    if apply:
        aligned_spec = _apply_phase_advance(sub_spec, freqs, offset_ms / 1000.0)
    else:
        aligned_spec = sub_spec

    total_spec = main_spec + aligned_spec
    total = _build_transfer_like(main, total_spec, label=label)

    diagnostics: dict[str, Any] = {
        "xcorr_offset_ms": float(offset_ms),
        "xcorr_confidence": float(confidence),
        "xcorr_applied": bool(apply),
    }
    return total, diagnostics


def _sum_sub_components(
    template: TransferData,
    *subs: TransferData,
    label: str = "",
) -> TransferData:
    combined, _diag = build_combined_sub_transfer(
        template,
        *subs,
        mode="sum",
        label=label,
    )
    return combined
