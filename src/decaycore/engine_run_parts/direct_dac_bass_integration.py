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

from ..dsp.bass_integration import (
    build_bundle_combined_sub_transfer,
    run_direct_dac_bass_integration,
)

logger = logging.getLogger("DecayCore")


def _fir_complex_response(ir: np.ndarray, fs: int, freq_axis: np.ndarray) -> np.ndarray:
    x = np.asarray(ir, dtype=float).reshape(-1)
    freqs = np.asarray(freq_axis, dtype=float).reshape(-1)
    if x.size < 8 or freqs.size == 0 or int(fs) <= 0:
        return np.ones(freqs.shape, dtype=np.complex128)
    h = np.fft.rfft(x)
    f_fft = np.fft.rfftfreq(x.size, d=1.0 / float(fs))
    real = np.interp(np.clip(freqs, f_fft[0], f_fft[-1]), f_fft, np.real(h))
    imag = np.interp(np.clip(freqs, f_fft[0], f_fft[-1]), f_fft, np.imag(h))
    return np.asarray(real + 1j * imag, dtype=np.complex128)


def _interp_complex_to_axis(freq_src: np.ndarray, spec_src: np.ndarray, freq_dst: np.ndarray) -> np.ndarray:
    fs = np.asarray(freq_src, dtype=float).reshape(-1)
    sp = np.asarray(spec_src, dtype=np.complex128).reshape(-1)
    fd = np.asarray(freq_dst, dtype=float).reshape(-1)
    if fd.size == 0:
        return np.asarray([], dtype=np.complex128)
    if fs.size == fd.size and np.allclose(fs, fd, rtol=0.0, atol=1e-9):
        return sp.copy()
    if fs.size < 2 or sp.size != fs.size:
        return np.ones(fd.shape, dtype=np.complex128)
    fq = np.clip(fd, float(np.min(fs)), float(np.max(fs)))
    return np.asarray(
        np.interp(fq, fs, np.real(sp)) + 1j * np.interp(fq, fs, np.imag(sp)),
        dtype=np.complex128,
    )


def apply_direct_dac_bass_integration_result(
    *,
    cfg: Any,
    measurements: dict[str, Any],
    data: dict[str, Any],
    l_imp: np.ndarray,
    r_imp: np.ndarray,
    sub_ir: np.ndarray | None,
    l_st: dict[str, Any] | None,
    r_st: dict[str, Any] | None,
    sub_st: dict[str, Any] | None,
) -> dict[str, Any]:
    if (
        sub_ir is None
        or not bool(getattr(cfg, "bass_integration_enable", False))
        or str(getattr(cfg, "bass_integration_mode", "") or "").strip().lower() != "direct_dac"
    ):
        return {}
    bundle = measurements.get("bass_integration_bundle", None)
    if bundle is None:
        return {}
    try:
        opt_freqs = np.asarray(bundle.l_main.freqs_hz, dtype=float).reshape(-1)
        l_main_corr = _interp_complex_to_axis(
            np.asarray(bundle.l_main.freqs_hz, dtype=float),
            np.asarray(bundle.l_main.complex_spec, dtype=np.complex128),
            opt_freqs,
        ) * _fir_complex_response(np.asarray(l_imp, dtype=float), int(cfg.fs), opt_freqs)
        r_main_corr = _interp_complex_to_axis(
            np.asarray(bundle.r_main.freqs_hz, dtype=float),
            np.asarray(bundle.r_main.complex_spec, dtype=np.complex128),
            opt_freqs,
        ) * _fir_complex_response(np.asarray(r_imp, dtype=float), int(cfg.fs), opt_freqs)
        sub_total_transfer, _diag = build_bundle_combined_sub_transfer(
            bundle,
            channel="l",
            label="Direct-DAC corrected sub optimization",
        )
        sub_corr = _interp_complex_to_axis(
            np.asarray(sub_total_transfer.freqs_hz, dtype=float),
            np.asarray(sub_total_transfer.complex_spec, dtype=np.complex128),
            opt_freqs,
        ) * _fir_complex_response(np.asarray(sub_ir, dtype=float), int(cfg.fs), opt_freqs)
        direct_dac_result = run_direct_dac_bass_integration(
            0.5 * (l_main_corr + r_main_corr),
            sub_corr,
            opt_freqs,
        )
        result_dict = direct_dac_result.as_dict()
        data["bass_integration_mode"] = "direct_dac"
        data["sub_crossover_hz"] = float(direct_dac_result.main_hpf_hz)
        data["avr_crossover_hz"] = float(direct_dac_result.main_hpf_hz)
        data["sub_hpf_freq"] = float(direct_dac_result.sub_hpf_hz)
        data["direct_dac_sub_lpf_hz"] = float(direct_dac_result.sub_lpf_hz)
        data["bass_integration_sub_delay_ms"] = float(direct_dac_result.sub_delay_ms)
        data["bass_integration_sub_gain_trim_db"] = float(direct_dac_result.sub_gain_db)
        data["bass_integration_sub_polarity_invert"] = bool(direct_dac_result.sub_polarity_invert)
        data["bass_integration_alignment_auto_applied"] = True
        data["bass_integration_alignment_reason"] = "Direct DAC corrected-response optimizer."
        data["bass_integration_allpass_auto_enable"] = False
        data["bass_integration_allpass_auto_applied"] = False
        data["bass_integration_allpass_freq_hz"] = 0.0
        data["bass_integration_allpass_q"] = 0.707
        sub_target_policy = str(data.get("sub_target_policy", "") or "").strip()
        sub_target_lpf_hz = data.get("sub_target_lpf_hz", None)
        sub_target_slope = data.get("sub_target_lpf_slope_db_per_oct", None)
        bi_meta = data.setdefault("_bass_integration_meta", {})
        if isinstance(bi_meta, dict):
            bi_meta.update(
                {
                    "enabled": True,
                    "mode": "direct_dac",
                    "optimization": "automatic",
                    "avr_crossover_hz": float(direct_dac_result.main_hpf_hz),
                    "direct_dac_sub_lpf_hz": float(direct_dac_result.sub_lpf_hz),
                    "recommended_crossover_hz": float(direct_dac_result.main_hpf_hz),
                    "recommended_sub_lpf_hz": float(direct_dac_result.sub_lpf_hz),
                    "direct_dac_result": dict(result_dict),
                    **(
                        {
                            "sub_target_policy": sub_target_policy,
                            "sub_target_lpf_hz": float(sub_target_lpf_hz),
                            "sub_target_lpf_slope_db_per_oct": float(sub_target_slope),
                        }
                        if sub_target_policy and sub_target_lpf_hz is not None and sub_target_slope is not None
                        else {}
                    ),
                }
            )
            diagnostics = dict(bi_meta.get("diagnostics", {}) or {})
            diagnostics.update(
                {
                    "direct_dac_main_hpf_hz": float(direct_dac_result.main_hpf_hz),
                    "direct_dac_sub_hpf_hz": float(direct_dac_result.sub_hpf_hz),
                    "direct_dac_sub_lpf_hz": float(direct_dac_result.sub_lpf_hz),
                    "direct_dac_sub_overlap_hz": float(direct_dac_result.sub_overlap_hz),
                    "direct_dac_phase_error_deg": float(direct_dac_result.phase_error_deg),
                    "direct_dac_gd_mismatch_ms": float(direct_dac_result.gd_mismatch_ms),
                    "direct_dac_cancellation_risk": str(direct_dac_result.cancellation_risk),
                    "direct_dac_cancellation_score": float(direct_dac_result.cancellation_score),
                    "direct_dac_magnitude_ripple_db": float(direct_dac_result.magnitude_ripple_db),
                }
            )
            bi_meta["diagnostics"] = diagnostics
            bi_meta["alignment"] = {
                "applied": True,
                "delay_ms": float(direct_dac_result.sub_delay_ms),
                "polarity_invert": bool(direct_dac_result.sub_polarity_invert),
                "gain_trim_db": float(direct_dac_result.sub_gain_db),
                "reason": "Direct DAC corrected-response optimizer.",
                "improvement_score": float(-direct_dac_result.score),
                "baseline_metrics": {},
                "optimized_metrics": dict(result_dict),
            }
        for st in (l_st, r_st, sub_st):
            if isinstance(st, dict):
                st["direct_dac_bass_integration"] = dict(result_dict)
        logger.info(
            "Direct-DAC Bass Integration selected: main HPF %.1f Hz, sub HPF %.1f Hz, "
            "sub LPF %.1f Hz, polarity %s, delay %.2f ms, gain %+0.2f dB, score %.3f",
            float(direct_dac_result.main_hpf_hz),
            float(direct_dac_result.sub_hpf_hz),
            float(direct_dac_result.sub_lpf_hz),
            "invert" if bool(direct_dac_result.sub_polarity_invert) else "normal",
            float(direct_dac_result.sub_delay_ms),
            float(direct_dac_result.sub_gain_db),
            float(direct_dac_result.score),
        )
        return dict(result_dict)
    except Exception:
        logger.debug("Direct-DAC corrected-response optimization failed", exc_info=True)
        return {}


__all__ = ["apply_direct_dac_bass_integration_result"]
