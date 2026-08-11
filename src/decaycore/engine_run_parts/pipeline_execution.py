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

import dataclasses
import logging
from typing import Any

import numpy as np

from ..common.profiling import profiled_section
from ..common.comparison_stats import _make_comparison_stats
from ..common.measurement_features import (
    estimate_schroeder_hz,
    median_rt60_mid_band,
)
from ..common.result_postprocess import (
    _ensure_scoring_keys,
    _inject_filter_gd_stats,
    _inject_filter_mags_for_ui,
    _irwin_tag,
    _postpolish_wav_filter_ir,
    _shift_zeropad_1d,
)
from ..config.models import FilterConfig
from ..config.results import FilterResult
from ..dsp import decaycore_dsp_parts as dsp
from ..dsp._measurement_ctx_local import measurement_ctx_scope
from ..dsp.dsp_telemetry import quiet_dsp_info_logging
from ..dsp.phase_realization_feedback import (
    assess_phase_feedback_candidate,
    build_phase_feedback_strengths,
    phase_feedback_stats,
    phase_feedback_strength_field,
    select_phase_feedback_candidate,
)
from ..dsp.hybrid_iir import peaking_eq_response as _peaking_eq_response
from ..engine_build import _as_float
from ..engine_summary import summarize_run
from .subwoofer_target import build_subwoofer_target_with_lpf, subwoofer_target_metadata

from .measurement_response_helpers import (
    _build_measurement_side_ctx,
    _extract_lr_measurement_axes,
    _interp_complex_to_axis,
    _ir_fft_on_axis,
    _phase_from_ir,
    _resample_to_axis,
    _resolve_sub_measurement_for_filter,
    _to_axis,
)

logger = logging.getLogger("DecayCore")


def _call_generate_filter(
    freqs,
    mags,
    phases,
    cfg,
    *,
    include_response_arrays: bool,
    phase_feedback_replay_cache: dict | None = None,
    phase_feedback_replay_key: str | None = None,
):
    try:
        return dsp.generate_filter(
            freqs,
            mags,
            phases,
            cfg,
            include_response_arrays=bool(include_response_arrays),
            phase_feedback_replay_cache=phase_feedback_replay_cache,
            phase_feedback_replay_key=phase_feedback_replay_key,
        )
    except TypeError as exc:
        if not any(
            key in str(exc)
            for key in ("include_response_arrays", "phase_feedback_replay_cache")
        ):
            raise
        try:
            return dsp.generate_filter(
                freqs,
                mags,
                phases,
                cfg,
                include_response_arrays=bool(include_response_arrays),
            )
        except TypeError as fallback_exc:
            if "include_response_arrays" not in str(fallback_exc):
                raise
            return dsp.generate_filter(freqs, mags, phases, cfg)


def _call_generate_filter_pair(
    f_l,
    m_l,
    p_l,
    f_r,
    m_r,
    p_r,
    cfg,
    *,
    measurement_ctx_l,
    measurement_ctx_r,
    include_response_arrays: bool,
    phase_feedback_replay_cache: dict | None = None,
):
    try:
        return dsp.generate_filter_pair(
            f_l,
            m_l,
            p_l,
            f_r,
            m_r,
            p_r,
            cfg,
            measurement_ctx_l=measurement_ctx_l,
            measurement_ctx_r=measurement_ctx_r,
            include_response_arrays=bool(include_response_arrays),
            phase_feedback_replay_cache=phase_feedback_replay_cache,
        )
    except TypeError as exc:
        if not any(
            key in str(exc)
            for key in ("include_response_arrays", "phase_feedback_replay_cache")
        ):
            raise
        try:
            return dsp.generate_filter_pair(
                f_l,
                m_l,
                p_l,
                f_r,
                m_r,
                p_r,
                cfg,
                measurement_ctx_l=measurement_ctx_l,
                measurement_ctx_r=measurement_ctx_r,
                include_response_arrays=bool(include_response_arrays),
            )
        except TypeError as fallback_exc:
            if "include_response_arrays" not in str(fallback_exc):
                raise
            return dsp.generate_filter_pair(
                f_l,
                m_l,
                p_l,
                f_r,
                m_r,
                p_r,
                cfg,
                measurement_ctx_l=measurement_ctx_l,
                measurement_ctx_r=measurement_ctx_r,
            )


def _stats_level_comp_factor(st: dict | None) -> float:
    try:
        ag_db = float((st or {}).get("auto_global_gain_db", 0.0) or 0.0)
    except (
        RuntimeError,
        OSError,
        ImportError,
        TypeError,
        ValueError,
        AttributeError,
        KeyError,
        IndexError,
        OverflowError,
        FloatingPointError,
    ):
        ag_db = 0.0
    try:
        ah_db = float((st or {}).get("auto_headroom_db", 0.0) or 0.0)
    except (
        RuntimeError,
        OSError,
        ImportError,
        TypeError,
        ValueError,
        AttributeError,
        KeyError,
        IndexError,
        OverflowError,
        FloatingPointError,
    ):
        ah_db = 0.0
    try:
        return float(np.power(10.0, -0.05 * (float(ag_db) + float(ah_db))))
    except (
        RuntimeError,
        OSError,
        ImportError,
        TypeError,
        ValueError,
        AttributeError,
        KeyError,
        IndexError,
        OverflowError,
        FloatingPointError,
    ):
        return 1.0


def _iir_response_on_axis(st: dict | None, f_axis: np.ndarray, fs: int) -> np.ndarray:
    """Combined IIR complex response from stored biquad stats, on f_axis."""
    biquads = (st or {}).get("hybrid_iir_biquads", [])
    h = np.ones(len(f_axis), dtype=complex)
    for b in biquads or []:
        try:
            h *= _peaking_eq_response(f_axis, float(fs), float(b["freq"]), float(b["q"]), float(b["gain"]))
        except (
            RuntimeError,
            OSError,
            ImportError,
            TypeError,
            ValueError,
            AttributeError,
            KeyError,
            IndexError,
            OverflowError,
            FloatingPointError,
        ):
            pass
    return h


def _apply_measured_rt60_override(
    st: dict | None,
    *,
    measured_rt60: object,
    measured_bands: object,
    room_volume_m3: object = 40.0,
) -> None:
    if not isinstance(st, dict):
        return

    try:
        rt60_val = float(measured_rt60)
    except (TypeError, ValueError):
        return
    if not np.isfinite(rt60_val) or rt60_val <= 0.0:
        return

    st["rt60_val"] = float(rt60_val)
    st["rt60_reliability"] = 1.0
    st["rt60_source"] = "measured"
    schroeder_hz = estimate_schroeder_hz(rt60_val, room_volume_m3=room_volume_m3)
    if schroeder_hz is not None:
        st["schroeder_hz_estimate"] = float(schroeder_hz)

    if not isinstance(measured_bands, dict) or not measured_bands:
        return

    band_map = _measured_rt60_band_map(measured_bands)
    if not band_map:
        return

    st["rt60_bands"] = band_map
    st["rt60_band_avg"] = median_rt60_mid_band(band_map)


def _measured_rt60_band_map(measured_bands: dict) -> dict[float, float]:
    band_map: dict[float, float] = {}
    for key, value in measured_bands.items():
        try:
            freq_hz = float(key)
            band_rt60 = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(freq_hz) and np.isfinite(band_rt60) and band_rt60 > 0.0:
            band_map[float(freq_hz)] = float(band_rt60)
    return band_map


def _inject_direct_dac_summed_prediction_for_plot(
    *,
    st: dict | None,
    main_transfer: Any,
    sub_transfer: Any,
    main_ir: np.ndarray,
    sub_ir: np.ndarray | None,
    sub_st: dict | None,
    fs: int,
) -> None:
    if not isinstance(st, dict) or sub_ir is None:
        return
    try:
        f_axis = np.asarray(st.get("freq_axis", None), dtype=float)
        if f_axis.size < 4:
            f_axis = np.asarray(getattr(main_transfer, "freqs_hz", []), dtype=float)
        if f_axis.size < 4:
            return

        main_meas = _interp_complex_to_axis(
            np.asarray(getattr(main_transfer, "freqs_hz", []), dtype=float),
            np.asarray(getattr(main_transfer, "complex_spec", []), dtype=np.complex128),
            f_axis,
        )
        sub_meas = _interp_complex_to_axis(
            np.asarray(getattr(sub_transfer, "freqs_hz", []), dtype=float),
            np.asarray(getattr(sub_transfer, "complex_spec", []), dtype=np.complex128),
            f_axis,
        )
        main_h = _ir_fft_on_axis(np.asarray(main_ir, dtype=float), int(fs), f_axis)
        sub_h = _ir_fft_on_axis(np.asarray(sub_ir, dtype=float), int(fs), f_axis)
        main_iir = _iir_response_on_axis(st, f_axis, int(fs))
        sub_iir = _iir_response_on_axis(sub_st, f_axis, int(fs))

        total_measured = main_meas + sub_meas
        st["direct_dac_sum_measured_mags"] = (
            (20.0 * np.log10(np.maximum(np.abs(total_measured), 1e-12))).astype(float).tolist()
        )

        total_export = main_meas * (main_h * main_iir) + sub_meas * (sub_h * sub_iir)
        st["direct_dac_sum_predicted_mags"] = (
            (20.0 * np.log10(np.maximum(np.abs(total_export), 1e-12))).astype(float).tolist()
        )

        main_comp = float(_stats_level_comp_factor(st))
        sub_comp = float(_stats_level_comp_factor(sub_st))
        total_comp = main_meas * (main_h * main_comp * main_iir) + sub_meas * (sub_h * sub_comp * sub_iir)
        st["direct_dac_sum_predicted_mags_comp"] = (
            (20.0 * np.log10(np.maximum(np.abs(total_comp), 1e-12))).astype(float).tolist()
        )

    except (
        RuntimeError,
        OSError,
        ImportError,
        TypeError,
        ValueError,
        AttributeError,
        KeyError,
        IndexError,
        OverflowError,
        FloatingPointError,
    ):
        logger.debug("Direct-DAC summed plot prediction injection failed", exc_info=True)


def _generate_main_filters(
    cfg,
    f_l,
    m_l,
    p_l,
    f_r,
    m_r,
    p_r,
    _mctx_l,
    _mctx_r,
    *,
    include_response_arrays: bool,
):
    field = phase_feedback_strength_field(getattr(cfg, "filter_type_str", ""))
    enabled = bool(getattr(cfg, "phase_realization_feedback_enable", True))
    phase_feedback_replay_cache: dict | None = {} if enabled and field is not None else None

    def _run_channel(freqs, mags, phases, cfg_run, *, replay_key: str):
        try:
            return _call_generate_filter(
                freqs,
                mags,
                phases,
                cfg_run,
                include_response_arrays=bool(include_response_arrays),
                phase_feedback_replay_cache=phase_feedback_replay_cache,
                phase_feedback_replay_key=replay_key,
            )
        except TypeError as exc:
            if "phase_feedback_replay_cache" not in str(exc):
                raise
            return _call_generate_filter(
                freqs,
                mags,
                phases,
                cfg_run,
                include_response_arrays=bool(include_response_arrays),
            )

    def _run_once(cfg_run):
        if bool(getattr(cfg_run, "stereo_link", False)):
            return _call_generate_filter_pair(
                f_l,
                m_l,
                p_l,
                f_r,
                m_r,
                p_r,
                cfg_run,
                measurement_ctx_l=_mctx_l,
                measurement_ctx_r=_mctx_r,
                include_response_arrays=bool(include_response_arrays),
                phase_feedback_replay_cache=phase_feedback_replay_cache,
            )
        with measurement_ctx_scope(_mctx_l):
            l_imp, l_st = _run_channel(
                f_l,
                m_l,
                p_l,
                cfg_run,
                replay_key="left",
            )
        with measurement_ctx_scope(_mctx_r):
            r_imp, r_st = _run_channel(
                f_r,
                m_r,
                p_r,
                cfg_run,
                replay_key="right",
            )
        return l_imp, l_st, r_imp, r_st

    with profiled_section("run_pipeline.generate_filters"):
        initial = _run_once(cfg)

    if not enabled or field is None:
        if enabled and field is None:
            minimum_assessment = assess_phase_feedback_candidate(
                strength=0.0,
                requested_strength=0.0,
                impulse_l=initial[0],
                stats_l=initial[1],
                impulse_r=initial[2],
                stats_r=initial[3],
                cfg=cfg,
                payload=initial,
            )
            status = phase_feedback_stats(
                field=None,
                candidates=[minimum_assessment],
                selected=minimum_assessment,
                reason="not_applicable_minimum_phase",
                applicable=False,
            )
        else:
            status = {
                "phase_realization_feedback_enabled": False,
                "phase_realization_feedback_applicable": bool(field is not None),
                "phase_realization_feedback_applied": False,
                "phase_realization_feedback_reason": "disabled",
            }
        for stats in (initial[1], initial[3]):
            if isinstance(stats, dict):
                stats.update(status)
        return initial

    requested = float(np.clip(float(getattr(cfg, field, 0.9) or 0.0), 0.0, 1.0))
    candidate_count = int(
        np.clip(int(getattr(cfg, "phase_realization_feedback_candidate_count", 5) or 5), 2, 5)
    )
    strengths = build_phase_feedback_strengths(requested, candidate_count)
    candidates = [
        assess_phase_feedback_candidate(
            strength=requested,
            requested_strength=requested,
            impulse_l=initial[0],
            stats_l=initial[1],
            impulse_r=initial[2],
            stats_r=initial[3],
            cfg=cfg,
            payload=initial,
        )
    ]
    if not all(
        np.isfinite(float(channel.get("gd_score", float("nan"))))
        for channel in candidates[0]["channels"]
    ):
        telemetry = phase_feedback_stats(
            field=field,
            candidates=candidates,
            selected=candidates[0],
            reason="missing_realized_gd_metric",
        )
        for stats in (initial[1], initial[3]):
            if isinstance(stats, dict):
                stats.update(telemetry)
        return initial

    for strength in strengths[1:]:
        cfg_candidate = dataclasses.replace(
            cfg,
            **{
                field: float(strength),
                "phase_realization_feedback_enable": False,
            },
        )
        with profiled_section("run_pipeline.phase_realization_feedback_candidate"):
            output = _run_once(cfg_candidate)
        candidates.append(
            assess_phase_feedback_candidate(
                strength=float(strength),
                requested_strength=requested,
                impulse_l=output[0],
                stats_l=output[1],
                impulse_r=output[2],
                stats_r=output[3],
                cfg=cfg_candidate,
                payload=output,
            )
        )

    selected, reason = select_phase_feedback_candidate(candidates)
    output = selected.get("payload") or initial
    telemetry = phase_feedback_stats(
        field=field,
        candidates=candidates,
        selected=selected,
        reason=reason,
    )
    for stats in (output[1], output[3]):
        if isinstance(stats, dict):
            stats.update(telemetry)
    logger.info(
        "Phase realization feedback: field=%s requested=%.3f selected=%.3f reason=%s",
        field,
        requested,
        float(selected.get("strength", requested)),
        reason,
    )
    return output


def _generate_sub_ir(  # noqa: C901 - sub path setup preserves the direct-DAC and normal branches together
    cfg,
    measurements,
    data,
    f_l,
    m_l,
    p_l,
    f_r,
    m_r,
    p_r,
    warnings: list,
    *,
    is_direct_dac_bi: bool,
    include_response_arrays: bool,
):
    sub_ir: np.ndarray | None = None
    sub_f: np.ndarray | None = None
    sub_m: np.ndarray | None = None
    sub_p: np.ndarray | None = None
    sub_st: dict | None = None
    sub_target_meta: dict[str, Any] = {}

    if not (bool(getattr(cfg, "sub_integration_enable", False)) and bool(getattr(cfg, "sub_generate_ir", False))):
        return sub_ir, sub_f, sub_m, sub_p, sub_st, sub_target_meta

    try:
        f_sub, m_sub, p_sub = _resolve_sub_measurement_for_filter(
            measurements,
            f_l=f_l,
            m_l=m_l,
            p_l=p_l,
            f_r=f_r,
            m_r=m_r,
            p_r=p_r,
        )

        sub_xo_hz = float(getattr(cfg, "sub_crossover_hz", 80.0))
        sub_xo_order = int(getattr(cfg, "sub_crossover_order", 4))
        try:
            sub_lpf_hz = float(getattr(cfg, "direct_dac_sub_lpf_hz", sub_xo_hz) or sub_xo_hz)
        except (
            RuntimeError,
            OSError,
            ImportError,
            TypeError,
            ValueError,
            AttributeError,
            KeyError,
            IndexError,
            OverflowError,
            FloatingPointError,
        ):
            sub_lpf_hz = sub_xo_hz
        if not np.isfinite(sub_lpf_hz) or sub_lpf_hz <= 0.0:
            sub_lpf_hz = sub_xo_hz
        if (
            bool(getattr(cfg, "bass_integration_enable", False))
            and str(getattr(cfg, "bass_integration_mode", "") or "").strip().lower() == "direct_dac"
        ):
            sub_lpf_hz = max(float(sub_xo_hz), float(sub_lpf_hz))
        sub_hpf_freq = float(getattr(cfg, "sub_hpf_freq", 20.0))
        sub_hpf_order = int(getattr(cfg, "sub_hpf_order", 2))

        if is_direct_dac_bi:
            sub_target = build_subwoofer_target_with_lpf(
                getattr(cfg, "house_freqs", None),
                getattr(cfg, "house_mags", None),
            )
            sub_target_meta = subwoofer_target_metadata(sub_target)
            sub_cfg = dataclasses.replace(
                cfg,
                lpf_settings=None,
                hpf_settings=None,
                crossovers=[],
                house_freqs=sub_target.house_freqs,
                house_mags=sub_target.house_mags,
                mag_c_min=20.0,
                mag_c_max=200.0,
                lvl_min=20.0,
                lvl_max=200.0,
                stereo_link=False,
                sub_integration_enable=False,
                sub_generate_ir=False,
            )
            if isinstance(data, dict):
                data.update(sub_target_meta)
                bi_meta = data.setdefault("_bass_integration_meta", {})
                if isinstance(bi_meta, dict):
                    bi_meta.update(sub_target_meta)
        else:
            sub_cfg = dataclasses.replace(
                cfg,
                lpf_settings={"enabled": True, "freq": sub_lpf_hz, "order": sub_xo_order},
                hpf_settings={"enabled": True, "freq": sub_hpf_freq, "order": sub_hpf_order},
                crossovers=[{"freq": sub_lpf_hz, "order": sub_xo_order, "slope": sub_xo_order * 6, "idx": 0}],
                mag_c_min=max(5.0, float(sub_hpf_freq)),
                mag_c_max=sub_lpf_hz,
                lvl_min=20.0,
                lvl_max=200.0,
                stereo_link=False,
                sub_integration_enable=False,
                sub_generate_ir=False,
            )
        sub_imp, sub_st = _call_generate_filter(
            f_sub,
            m_sub,
            p_sub,
            sub_cfg,
            include_response_arrays=bool(include_response_arrays),
        )
        sub_ir = np.asarray(sub_imp, dtype=float)
        sub_f = np.asarray(f_sub, dtype=float)
        sub_m = np.asarray(m_sub, dtype=float)
        sub_p = np.asarray(p_sub, dtype=float)
        if is_direct_dac_bi and isinstance(sub_st, dict):
            sub_st.update(sub_target_meta)
        if is_direct_dac_bi:
            logger.info(
                "Sub pass: Direct-DAC full-band FIR, sub target LPF %.0f Hz / %.0f dB/oct, "
                "correction scan 20-200 Hz; "
                "XO/HPF/LPF applied in CamillaDSP YAML"
                % (
                    float(sub_target_meta.get("sub_target_lpf_hz", 200.0) or 200.0),
                    float(sub_target_meta.get("sub_target_lpf_slope_db_per_oct", 96.0) or 96.0),
                )
            )
        elif abs(float(sub_lpf_hz) - float(sub_xo_hz)) >= 0.5:
            logger.info(
                f"Sub pass: main HPF={sub_xo_hz:.0f} Hz, "
                f"sub LPF={sub_lpf_hz:.0f} Hz (order {sub_xo_order}), "
                f"HPF={sub_hpf_freq:.0f} Hz (order {sub_hpf_order})"
            )
        else:
            logger.info(
                f"Sub pass: xo={sub_xo_hz:.0f} Hz (order {sub_xo_order}), "
                f"HPF={sub_hpf_freq:.0f} Hz (order {sub_hpf_order})"
            )
    except (
        RuntimeError,
        OSError,
        ImportError,
        TypeError,
        ValueError,
        AttributeError,
        KeyError,
        IndexError,
        OverflowError,
        FloatingPointError,
    ) as exc:
        warnings.append(f"sub_pass_failed: {exc}")
        logger.warning(f"Sub DSP pass failed: {exc}")
        sub_ir = None

    return sub_ir, sub_f, sub_m, sub_p, sub_st, sub_target_meta


def _apply_rt60_and_alignment_checks(  # noqa: C901 - measurement quality and alignment checks are intentionally grouped
    l_st, r_st, measurements, data, cfg, *, is_wav: bool
) -> None:
    rt_rel = 1.0 if is_wav else 0.25
    rt_src = "WAV" if is_wav else "TXT/REW"
    if isinstance(l_st, dict):
        l_st["rt60_reliability"] = float(rt_rel)
        l_st["rt60_source"] = rt_src
    if isinstance(r_st, dict):
        r_st["rt60_reliability"] = float(rt_rel)
        r_st["rt60_source"] = rt_src
    _apply_measured_rt60_override(
        l_st,
        measured_rt60=measurements.get("measured_rt60_l"),
        measured_bands=measurements.get("measured_rt60_bands_l"),
        room_volume_m3=getattr(cfg, "room_volume_m3", 40.0),
    )
    _apply_measured_rt60_override(
        r_st,
        measured_rt60=measurements.get("measured_rt60_r"),
        measured_bands=measurements.get("measured_rt60_bands_r"),
        room_volume_m3=getattr(cfg, "room_volume_m3", 40.0),
    )

    _ir_align_lr: dict = {}
    _ir_align_sub: dict = {}
    _raw_ir_l = measurements.get("raw_ir_l")
    _raw_ir_r = measurements.get("raw_ir_r")
    _raw_ir_fs_l = int(measurements.get("raw_ir_fs_l") or 0)
    _raw_ir_fs_r = int(measurements.get("raw_ir_fs_r") or 0)
    _raw_ir_sub = measurements.get("raw_ir_sub")
    _raw_ir_fs_sub = int(measurements.get("raw_ir_fs_sub") or 0)
    if _raw_ir_l is not None and _raw_ir_r is not None and _raw_ir_fs_l > 0 and _raw_ir_fs_r > 0:
        try:
            from ..dsp.ir_alignment_check import run_ir_alignment_check  # noqa: PLC0415

            _xo_lr = float(data.get("avr_crossover_hz", 80.0) or 80.0)
            if not (10.0 <= _xo_lr <= 500.0):
                _xo_lr = 80.0
            _ir_align_lr = run_ir_alignment_check(
                np.asarray(_raw_ir_l, dtype=float),
                _raw_ir_fs_l,
                np.asarray(_raw_ir_r, dtype=float),
                _raw_ir_fs_r,
                xo_hz=_xo_lr,
            )
        except (
            RuntimeError,
            OSError,
            ImportError,
            TypeError,
            ValueError,
            AttributeError,
            KeyError,
            IndexError,
            OverflowError,
            FloatingPointError,
        ):
            logger.debug("IR alignment check L/R failed", exc_info=True)
    if _raw_ir_l is not None and _raw_ir_sub is not None and _raw_ir_fs_l > 0 and _raw_ir_fs_sub > 0:
        try:
            from ..dsp.ir_alignment_check import run_ir_alignment_check  # noqa: PLC0415

            _xo_sub = float(data.get("avr_crossover_hz") or data.get("sub_crossover_hz", 80.0) or 80.0)
            if not (10.0 <= _xo_sub <= 500.0):
                _xo_sub = 80.0
            _ir_align_sub = run_ir_alignment_check(
                np.asarray(_raw_ir_l, dtype=float),
                _raw_ir_fs_l,
                np.asarray(_raw_ir_sub, dtype=float),
                _raw_ir_fs_sub,
                xo_hz=_xo_sub,
            )
        except (
            RuntimeError,
            OSError,
            ImportError,
            TypeError,
            ValueError,
            AttributeError,
            KeyError,
            IndexError,
            OverflowError,
            FloatingPointError,
        ):
            logger.debug("IR alignment check main/sub failed", exc_info=True)
    if isinstance(l_st, dict):
        if _ir_align_lr:
            l_st["ir_align"] = dict(_ir_align_lr)
        if _ir_align_sub:
            l_st["ir_align_sub"] = dict(_ir_align_sub)
    if isinstance(r_st, dict):
        if _ir_align_lr:
            r_st["ir_align"] = dict(_ir_align_lr)
        if _ir_align_sub:
            r_st["ir_align_sub"] = dict(_ir_align_sub)


def _align_lr_and_sub(  # noqa: C901 - alignment policy keeps LR and sub timing branches explicit
    l_imp, r_imp, sub_ir, l_st, r_st, data, cfg, *, is_direct_dac_bi: bool
):
    with profiled_section("run_pipeline.align"):
        align_method = "peak"
        d_peak = int(np.argmax(np.abs(l_imp)) - np.argmax(np.abs(r_imp)))
        d_delay: int | None = None
        try:
            dl = l_st.get("delay_samples", None) if isinstance(l_st, dict) else None
            dr = r_st.get("delay_samples", None) if isinstance(r_st, dict) else None
            if dl is not None and dr is not None:
                d_delay = int(round(float(dr))) - int(round(float(dl)))
        except (
            RuntimeError,
            OSError,
            ImportError,
            TypeError,
            ValueError,
            AttributeError,
            KeyError,
            IndexError,
            OverflowError,
            FloatingPointError,
        ):
            d_delay = None

        if d_delay is None:
            d_s = d_peak
            align_method = "peak"
        else:
            d_s = int(d_delay)
            align_method = "delay_samples"
            try:
                guard_samples = 8
                if abs(int(d_delay) - int(d_peak)) > int(guard_samples):
                    d_s = int(d_peak)
                    align_method = "peak_guard"
                    logger.info(
                        f"Alignment guard: delay_samples={int(d_delay)} vs peak={int(d_peak)} "
                        f"(>{guard_samples} samp) -> using peak"
                    )
            except (
                RuntimeError,
                OSError,
                ImportError,
                TypeError,
                ValueError,
                AttributeError,
                KeyError,
                IndexError,
                OverflowError,
                FloatingPointError,
            ):
                logger.exception("alignment guard comparison")

        if d_s > 0:
            r_imp = _shift_zeropad_1d(r_imp, d_s)
        elif d_s < 0:
            l_imp = _shift_zeropad_1d(l_imp, -d_s)

        if sub_ir is not None and np.asarray(sub_ir, dtype=float).size > 0:
            sub_imp = np.asarray(sub_ir, dtype=float)
            d_sub: int | None = None
            try:
                # Sub delay_samples is computed from the 1–10 kHz phase reference band,
                # which is outside the sub's 20–80 Hz working range → unreliable (returns ~0).
                # Always use peak-based alignment for the subwoofer IR.
                main_peak_ref = int(
                    round(
                        0.5
                        * (
                            int(np.argmax(np.abs(np.asarray(l_imp, dtype=float))))
                            + int(np.argmax(np.abs(np.asarray(r_imp, dtype=float))))
                        )
                    )
                )
                sub_peak = int(np.argmax(np.abs(sub_imp)))
                d_sub = int(main_peak_ref - sub_peak)
            except (
                RuntimeError,
                OSError,
                ImportError,
                TypeError,
                ValueError,
                AttributeError,
                KeyError,
                IndexError,
                OverflowError,
                FloatingPointError,
            ):
                d_sub = None
            if is_direct_dac_bi:
                bi_meta = data.setdefault("_bass_integration_meta", {})
                if isinstance(bi_meta, dict):
                    bi_meta["sub_peak_alignment_diagnostic"] = {
                        "main_peak_ref_sample": int(main_peak_ref) if "main_peak_ref" in locals() else None,
                        "sub_peak_sample": int(sub_peak) if "sub_peak" in locals() else None,
                        "suggested_shift_samples": int(d_sub or 0),
                        "applied_to_ir": False,
                        "reason": "Direct-DAC timing is handled by exported delay filters.",
                    }
            elif d_sub != 0:
                sub_ir = _shift_zeropad_1d(sub_imp, int(d_sub))
                logger.info(f"Sub alignment applied: {int(d_sub)} samples")

    return l_imp, r_imp, sub_ir, d_s, d_peak, d_delay, align_method


def _detect_wav_like_grid(f_l: np.ndarray) -> bool:
    try:
        fx = np.asarray(f_l if f_l is not None else [], dtype=float)
        if fx.size > 1024 and abs(float(fx[0])) < 1e-9:
            df = float(np.median(np.diff(fx[: min(int(fx.size), 4096)])))
            if np.isfinite(df) and (0.0 < df < 2.0):
                return True
    except (
        RuntimeError,
        OSError,
        ImportError,
        TypeError,
        ValueError,
        AttributeError,
        KeyError,
        IndexError,
        OverflowError,
        FloatingPointError,
    ):
        pass
    return False


def _apply_wav_postpolish(
    l_imp, r_imp, cfg, data, warnings: list, *, is_wav: bool, wav_like_fft_grid: bool
) -> tuple:  # noqa: C901 - alignment policy must keep LR/sub timing branches explicit
    if not (bool(is_wav) or bool(wav_like_fft_grid)):
        return l_imp, r_imp
    with profiled_section("run_pipeline.wav_postpolish"):
        try:
            mc_min = _as_float(getattr(cfg, "mag_c_min", data.get("mag_c_min", 10.0)), 10.0)
            mc_max = _as_float(getattr(cfg, "mag_c_max", data.get("mag_c_max", 230.0)), 230.0)
            tr_w = _as_float(getattr(cfg, "trans_width", data.get("trans_width", 100.0)), 100.0)

            l_imp = _postpolish_wav_filter_ir(
                l_imp,
                int(cfg.fs),
                mag_c_min=mc_min,
                mag_c_max=mc_max,
                trans_width=tr_w,
            )
            r_imp = _postpolish_wav_filter_ir(
                r_imp,
                int(cfg.fs),
                mag_c_min=mc_min,
                mag_c_max=mc_max,
                trans_width=tr_w,
            )
            logger.info(
                f"WAV final IR polish applied at {int(cfg.fs)} Hz "
                f"(zone approx {max(mc_min, mc_max - 0.95 * tr_w):.0f}-{mc_max + 1.45 * tr_w:.0f} Hz, "
                f"is_wav={bool(is_wav)}, wav_like_fft_grid={bool(wav_like_fft_grid)})"
            )
        except (
            RuntimeError,
            OSError,
            ImportError,
            TypeError,
            ValueError,
            AttributeError,
            KeyError,
            IndexError,
            OverflowError,
            FloatingPointError,
        ) as exc:
            warnings.append(f"wav_postpolish_failed: {exc}")
            logger.warning(f"WAV final IR polish failed: {exc}")
    return l_imp, r_imp


def _build_response_arrays(
    l_st,
    r_st,
    l_imp,
    r_imp,
    sub_ir,
    sub_st,
    f_l,
    f_r,
    cfg,
    measurements,
    data,
    *,
    include_response_arrays: bool,
):
    if not bool(include_response_arrays):
        return (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
        )

    with profiled_section("run_pipeline.response_arrays"):
        _inject_filter_mags_for_ui(l_st, l_imp, int(cfg.fs))
        _inject_filter_mags_for_ui(r_st, r_imp, int(cfg.fs))

        if (
            sub_ir is not None
            and bool(getattr(cfg, "sub_integration_enable", False))
            and bool(getattr(cfg, "sub_generate_ir", False))
            and str(getattr(cfg, "bass_integration_mode", "") or "").strip().lower() == "direct_dac"
        ):
            bundle = measurements.get("bass_integration_bundle", None)
            if bundle is not None:
                try:
                    from ..dsp.bass_integration import (
                        apply_direct_dac_export_branch_model,
                        build_bundle_combined_sub_transfer,
                        direct_dac_candidate_from_data,
                    )

                    sub_total_transfer, _diag = build_bundle_combined_sub_transfer(
                        bundle,
                        channel="l",
                        label="Direct-DAC sub total",
                    )
                    candidate = direct_dac_candidate_from_data(data, cfg)
                    sub_total_transfer = apply_direct_dac_export_branch_model(
                        sub_total_transfer,
                        hpf_hz=float(candidate.sub_hpf_hz),
                        hpf_order=int(candidate.sub_hpf_order),
                        lpf_hz=float(candidate.sub_lpf_hz),
                        lpf_order=int(candidate.sub_lpf_order),
                        gain_trim_db=float(candidate.sub_gain_trim_db),
                        delay_ms=float(candidate.sub_delay_ms),
                        polarity_invert=bool(candidate.sub_polarity_invert),
                        allpass_freq_hz=(
                            float(candidate.sub_allpass_freq_hz) if candidate.sub_allpass_enabled else None
                        ),
                        allpass_q=(float(candidate.sub_allpass_q) if candidate.sub_allpass_enabled else None),
                        label="Direct-DAC final sub branch for plot",
                    )
                    _inject_direct_dac_summed_prediction_for_plot(
                        st=l_st,
                        main_transfer=apply_direct_dac_export_branch_model(
                            bundle.l_main,
                            hpf_hz=float(candidate.main_hpf_hz),
                            hpf_order=int(candidate.main_hpf_order),
                            label="L Direct-DAC final main branch for plot",
                        ),
                        sub_transfer=sub_total_transfer,
                        main_ir=np.asarray(l_imp, dtype=float),
                        sub_ir=np.asarray(sub_ir, dtype=float),
                        sub_st=sub_st,
                        fs=int(cfg.fs),
                    )
                    _inject_direct_dac_summed_prediction_for_plot(
                        st=r_st,
                        main_transfer=apply_direct_dac_export_branch_model(
                            bundle.r_main,
                            hpf_hz=float(candidate.main_hpf_hz),
                            hpf_order=int(candidate.main_hpf_order),
                            label="R Direct-DAC final main branch for plot",
                        ),
                        sub_transfer=sub_total_transfer,
                        main_ir=np.asarray(r_imp, dtype=float),
                        sub_ir=np.asarray(sub_ir, dtype=float),
                        sub_st=sub_st,
                        fs=int(cfg.fs),
                    )
                except (
                    RuntimeError,
                    OSError,
                    ImportError,
                    TypeError,
                    ValueError,
                    AttributeError,
                    KeyError,
                    IndexError,
                    OverflowError,
                    FloatingPointError,
                ):
                    logger.debug("Direct-DAC summed prediction build failed", exc_info=True)

        l_mode = str((l_st or {}).get("analysis_mode", "native")).lower()
        r_mode = str((r_st or {}).get("analysis_mode", "native")).lower()
        l_fk = "cmp_freq_axis" if l_mode == "comparison" else "freq_axis"
        r_fk = "cmp_freq_axis" if r_mode == "comparison" else "freq_axis"
        l_gk = "cmp_filter_mags" if l_mode == "comparison" else "filter_mags"
        r_gk = "cmp_filter_mags" if r_mode == "comparison" else "filter_mags"

        l_freq = _to_axis((l_st or {}).get(l_fk), f_l)
        r_freq = _to_axis((r_st or {}).get(r_fk), f_r)
        freq_axis = l_freq if l_freq.size >= r_freq.size else r_freq
        if freq_axis.size == 0:
            freq_axis = f_l if f_l.size else f_r

        l_mag = _resample_to_axis((l_st or {}).get(l_gk), l_freq, freq_axis)
        r_mag = _resample_to_axis((r_st or {}).get(r_gk), r_freq, freq_axis)
        l_phase = _phase_from_ir(np.asarray(l_imp, dtype=float), int(cfg.fs), freq_axis)
        r_phase = _phase_from_ir(np.asarray(r_imp, dtype=float), int(cfg.fs), freq_axis)

    return freq_axis, l_mag, r_mag, l_phase, r_phase


def run_pipeline(
    cfg: FilterConfig,
    measurements: dict,
    *,
    debug: bool = False,
    include_response_arrays: bool = True,
) -> FilterResult:
    """Execute one DSP pipeline run and return normalized results.

    `measurements` must contain L/R frequency, magnitude and phase arrays:
    `f_l,m_l,p_l,f_r,m_r,p_r`.

    Score-only runs (include_response_arrays=False, i.e. auto-mode search
    trials) suppress DecayCore.dsp INFO/DEBUG log lines for this thread:
    the per-trial stage logging is a measurable share of the hot path and
    the lines never reach any report. WARNING+ always passes; the winner
    materialization (arrays on) logs in full.
    """
    with quiet_dsp_info_logging(enabled=not bool(include_response_arrays)):
        return _run_pipeline_impl(
            cfg,
            measurements,
            debug=debug,
            include_response_arrays=include_response_arrays,
        )


def _run_pipeline_impl(  # noqa: C901 - pipeline orchestration is intentionally centralized
    cfg: FilterConfig,
    measurements: dict,
    *,
    debug: bool = False,
    include_response_arrays: bool = True,
) -> FilterResult:
    if not isinstance(measurements, dict):
        raise TypeError("measurements must be a dict")

    f_l, m_l, p_l, f_r, m_r, p_r = _extract_lr_measurement_axes(measurements)
    if min(f_l.size, m_l.size, p_l.size, f_r.size, m_r.size, p_r.size) == 0:
        raise ValueError("Incomplete measurement data for pipeline run")

    data = dict(measurements.get("ui_data") or {})
    hc_f = measurements.get("hc_f")
    hc_m = measurements.get("hc_m")
    comparison_mode = bool(data.get("comparison_mode", getattr(cfg, "comparison_mode", False)))

    warnings: list[str] = []
    sub_st: dict | None = None
    sub_target_meta: dict[str, Any] = {}

    _mctx_l = _build_measurement_side_ctx(measurements, "l")
    _mctx_r = _build_measurement_side_ctx(measurements, "r")

    l_imp, l_st, r_imp, r_st = _generate_main_filters(
        cfg,
        f_l,
        m_l,
        p_l,
        f_r,
        m_r,
        p_r,
        _mctx_l,
        _mctx_r,
        include_response_arrays=bool(include_response_arrays),
    )

    is_direct_dac_bi = bool(
        getattr(cfg, "bass_integration_enable", False)
        and str(getattr(cfg, "bass_integration_mode", "") or "").strip().lower() == "direct_dac"
    )
    sub_ir, sub_f, sub_m, sub_p, sub_st, sub_target_meta = _generate_sub_ir(
        cfg,
        measurements,
        data,
        f_l,
        m_l,
        p_l,
        f_r,
        m_r,
        p_r,
        warnings,
        is_direct_dac_bi=is_direct_dac_bi,
        include_response_arrays=bool(include_response_arrays),
    )

    is_wav = bool(measurements.get("is_wav_source", getattr(cfg, "is_wav_source", False)))
    _apply_rt60_and_alignment_checks(l_st, r_st, measurements, data, cfg, is_wav=is_wav)

    with profiled_section("run_pipeline.ensure_scoring_keys"):
        l_st = _ensure_scoring_keys(l_st, f_l, m_l, hc_f, hc_m)
        r_st = _ensure_scoring_keys(r_st, f_r, m_r, hc_f, hc_m)

    if comparison_mode:
        with profiled_section("run_pipeline.comparison_stats"):
            try:
                l_st = _make_comparison_stats(l_st, int(cfg.fs), int(cfg.num_taps))
                r_st = _make_comparison_stats(r_st, int(cfg.fs), int(cfg.num_taps))
            except (
                RuntimeError,
                OSError,
                ImportError,
                TypeError,
                ValueError,
                AttributeError,
                KeyError,
                IndexError,
                OverflowError,
                FloatingPointError,
            ) as exc:
                warnings.append(f"comparison_stats_failed: {exc}")
                logger.warning(f"Comparison-mode stats failed: {exc}")

    l_imp, r_imp, sub_ir, d_s, d_peak, d_delay, align_method = _align_lr_and_sub(
        l_imp,
        r_imp,
        sub_ir,
        l_st,
        r_st,
        data,
        cfg,
        is_direct_dac_bi=is_direct_dac_bi,
    )

    wav_like_fft_grid = _detect_wav_like_grid(f_l)
    l_imp, r_imp = _apply_wav_postpolish(
        l_imp,
        r_imp,
        cfg,
        data,
        warnings,
        is_wav=is_wav,
        wav_like_fft_grid=wav_like_fft_grid,
    )

    if isinstance(l_st, dict) and isinstance(r_st, dict):
        try:
            delay_ms = round((float(d_s) / float(cfg.fs)) * 1000.0, 3)
            distance_cm = round((delay_ms / 1000.0) * 34300.0, 2)
            gain_diff = round(float(l_st.get("offset_db", 0.0)) - float(r_st.get("offset_db", 0.0)), 2)
            l_st["auto_align"] = {
                "delay_ms": delay_ms,
                "distance_cm": distance_cm,
                "gain_diff_db": gain_diff,
                "method": str(align_method),
            }
        except (
            RuntimeError,
            OSError,
            ImportError,
            TypeError,
            ValueError,
            AttributeError,
            KeyError,
            IndexError,
            OverflowError,
            FloatingPointError,
        ):
            logger.exception("auto_align stats inject")

    with profiled_section("run_pipeline.inject_filter_gd_stats"):
        _inject_filter_gd_stats(l_st, l_imp, int(cfg.fs))
        _inject_filter_gd_stats(r_st, r_imp, int(cfg.fs))

    direct_dac_result_dict = {}
    if is_direct_dac_bi:
        from .direct_dac_bass_integration import apply_direct_dac_bass_integration_result

        direct_dac_result_dict = apply_direct_dac_bass_integration_result(
            cfg=cfg,
            measurements=measurements,
            data=data,
            l_imp=np.asarray(l_imp, dtype=float),
            r_imp=np.asarray(r_imp, dtype=float),
            sub_ir=np.asarray(sub_ir, dtype=float) if sub_ir is not None else None,
            l_st=l_st,
            r_st=r_st,
            sub_st=sub_st if isinstance(sub_st, dict) else None,
        )
    bass_metric_payload = dict(direct_dac_result_dict.pop("_bass_metric_payload", {}) or {})
    if bool(direct_dac_result_dict.get("rejected", False)) and not bool(direct_dac_result_dict.get("enabled", False)):
        sub_ir = None
        sub_st = None
        data["sub_integration_enable"] = False
        data["sub_generate_ir"] = False
        try:
            cfg.sub_integration_enable = False
            cfg.sub_generate_ir = False
        except (AttributeError, TypeError, ValueError):
            logger.debug("FilterConfig sub output flags are not writable")

    freq_axis, l_mag, r_mag, l_phase, r_phase = _build_response_arrays(
        l_st,
        r_st,
        l_imp,
        r_imp,
        sub_ir,
        sub_st,
        f_l,
        f_r,
        cfg,
        measurements,
        data,
        include_response_arrays=bool(include_response_arrays),
    )

    if sub_target_meta and isinstance(data, dict):
        data.update(sub_target_meta)
        bi_meta = data.setdefault("_bass_integration_meta", {})
        if isinstance(bi_meta, dict):
            bi_meta.update(sub_target_meta)
    if sub_target_meta and isinstance(sub_st, dict):
        sub_st.update(sub_target_meta)

    metrics = {
        "alignment_samples": int(d_s),
        "alignment_method": str(align_method),
        "delay_samples_delta": (int(d_delay) if d_delay is not None else None),
        "peak_delta_samples": int(d_peak),
        "is_wav_source": bool(is_wav),
        "wav_like_fft_grid": bool(wav_like_fft_grid),
        "comparison_mode": bool(comparison_mode),
        "ir_export_window_tag": _irwin_tag(getattr(cfg, "ir_export_window_mode", "auto")),
        "l_max_boost_db_effective": _as_float(
            (l_st or {}).get("max_boost_db_effective", (l_st or {}).get("max_boost_db", 0.0)), 0.0
        ),
        "r_max_boost_db_effective": _as_float(
            (r_st or {}).get("max_boost_db_effective", (r_st or {}).get("max_boost_db", 0.0)), 0.0
        ),
        "l_max_cut_db": _as_float((l_st or {}).get("max_cut_db", 0.0), 0.0),
        "r_max_cut_db": _as_float((r_st or {}).get("max_cut_db", 0.0), 0.0),
        "direct_dac_bass_integration": dict(direct_dac_result_dict),
        "bass_integration_crossover_hz": _as_float(direct_dac_result_dict.get("main_hpf_hz"), 0.0),
        "bass_integration_delay_ms": _as_float(direct_dac_result_dict.get("sub_delay_ms"), 0.0),
        "bass_integration_gain_db": _as_float(direct_dac_result_dict.get("sub_gain_db"), 0.0),
        "bass_integration_polarity": (
            "invert" if bool(direct_dac_result_dict.get("sub_polarity_invert", False)) else "normal"
        ),
        "bass_integration_cancellation_risk": _as_float(direct_dac_result_dict.get("cancellation_risk"), 0.0),
        **bass_metric_payload,
        **({f"bass_integration_{k}": v for k, v in sub_target_meta.items()} if sub_target_meta else {}),
    }

    result = FilterResult(
        fs=int(cfg.fs),
        taps=int(cfg.num_taps),
        l_ir=np.asarray(l_imp, dtype=float),
        r_ir=np.asarray(r_imp, dtype=float),
        l_mag=np.asarray(l_mag, dtype=float),
        r_mag=np.asarray(r_mag, dtype=float),
        l_phase=np.asarray(l_phase, dtype=float),
        r_phase=np.asarray(r_phase, dtype=float),
        freq_axis=np.asarray(freq_axis, dtype=float),
        l_st=dict(l_st or {}),
        r_st=dict(r_st or {}),
        metrics=metrics,
        warnings=warnings,
        measurements={
            "f_l": np.asarray(f_l, dtype=float),
            "m_l": np.asarray(m_l, dtype=float),
            "p_l": np.asarray(p_l, dtype=float),
            "f_r": np.asarray(f_r, dtype=float),
            "m_r": np.asarray(m_r, dtype=float),
            "p_r": np.asarray(p_r, dtype=float),
            **(
                {
                    "f_sub": np.asarray(sub_f, dtype=float),
                    "m_sub": np.asarray(sub_m, dtype=float),
                    "p_sub": np.asarray(sub_p, dtype=float),
                }
                if sub_f is not None
                else {}
            ),
        },
        cfg=cfg,
        sub_ir=sub_ir,
        sub_st=dict(sub_st) if isinstance(sub_st, dict) else None,
    )
    if debug:
        result.metrics["summary"] = summarize_run(result)
    return result


__all__ = [
    "_call_generate_filter",
    "_call_generate_filter_pair",
    "_stats_level_comp_factor",
    "_apply_measured_rt60_override",
    "_inject_direct_dac_summed_prediction_for_plot",
    "dsp",
    "run_pipeline",
]
