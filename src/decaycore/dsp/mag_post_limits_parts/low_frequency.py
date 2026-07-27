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

from .._measurement_ctx_local import get_measurement_ctx
from ..dsp_config import CfgReader
from ..dsp_telemetry import safe_put_many
from ..gain_policy import build_low_frequency_guard_mask
from ..mag_telemetry import (
    _record_stage_probe,
)

def _apply_low_frequency_policy(
    *,
    freq_axis: np.ndarray,
    mask_c: np.ndarray,
    gain_apply: np.ndarray,
    raw_g: np.ndarray,
    final_g: np.ndarray,
    gain_db: np.ndarray,
    gain_policy,
    stage_probes: dict[str, object],
    stage_probe_fn,
    cfg,
    logger,
) -> np.ndarray | None:
    low_cut_enable = bool(gain_policy.low_cut_enable)
    low_cut_strength = float(gain_policy.low_cut_strength)
    low_cut_floor_ref = None
    low_mask = mask_c & build_low_frequency_guard_mask(
        freq_axis,
        gain_policy,
        include_low_cut=True,
        include_exc_soft=False,
    )
    if low_cut_enable and np.any(low_mask):
        low_cut = np.minimum(gain_apply[low_mask], 0.0)
        if low_cut_strength > 0.0:
            stronger_cut = np.minimum(final_g[low_mask], raw_g[low_mask])
            stronger_cut = np.minimum(stronger_cut, 0.0)
            low_cut = (1.0 - low_cut_strength) * low_cut + (low_cut_strength) * stronger_cut
        gain_apply[low_mask] = low_cut
        if low_cut_strength > 0.0:
            low_cut_floor_ref = np.full_like(gain_apply, np.nan, dtype=float)
            low_cut_floor_ref[low_mask] = np.asarray(low_cut, dtype=float)
    try:
        tmp_after_low = np.zeros_like(gain_db, dtype=float)
        tmp_after_low[mask_c] = gain_apply[mask_c]
        _record_stage_probe(
            stage_probes,
            "after_lowbass_policy",
            stage_probe_fn,
            freq_axis,
            tmp_after_low,
            mask_c,
            cfg,
            logger,
        )
    except (TypeError, ValueError, FloatingPointError, IndexError):
        pass
    return low_cut_floor_ref

def _prepare_boost_caps(
    *,
    cfg,
    cfg_reader: CfgReader,
    st,
    logger,
    freq_axis: np.ndarray,
    mask_c: np.ndarray,
    conf_mask: np.ndarray,
    gain_db: np.ndarray,
    gain_policy,
) -> dict[str, object]:
    max_cut_db = float(gain_policy.max_cut_db)
    max_boost_db_base = float(gain_policy.max_boost_db)
    boost_cap_db = np.full_like(gain_db, float(max_boost_db_base), dtype=float)
    bass_boost_cap_mask = np.zeros_like(mask_c, dtype=bool)
    bass_adaptive_isolation_mode = cfg_reader.bool("bass_adaptive_isolation_mode", False)
    cap_enable = cfg_reader.bool("bass_boost_cap_enable", True)
    bass_boost_cap_extra_db = float(max(0.0, cfg_reader.float_allow_zero("bass_boost_cap_extra_db", 5.0)))
    bass_boost_cap_hz = float(max(20.0, cfg_reader.float("bass_boost_cap_hz", 200.0)))
    bass_boost_cap_conf_min = float(np.clip(cfg_reader.float("bass_boost_cap_conf_min", 0.30), 0.0, 0.99))
    bass_boost_post_restore_enable = cfg_reader.bool("bass_boost_post_restore_enable", True)
    bass_boost_post_restore_strength = float(
        np.clip(cfg_reader.float_allow_zero("bass_boost_post_restore_strength", 1.05), 0.0, 1.0)
    )
    bass_boost_cap_enabled = False
    if bass_adaptive_isolation_mode:
        cap_enable = False
        bass_boost_post_restore_enable = False
    if (
        bool(cap_enable)
        and max_boost_db_base > 0.0
        and bass_boost_cap_extra_db > 0.0
        and isinstance(conf_mask, np.ndarray)
        and conf_mask.shape == gain_db.shape
    ):
        bass_boost_cap_mask, bass_boost_cap_enabled = _prepare_boost_caps_confidence_cap(
            conf_mask=conf_mask,
            mask_c=mask_c,
            freq_axis=freq_axis,
            gain_db=gain_db,
            boost_cap_db=boost_cap_db,
            max_boost_db_base=max_boost_db_base,
            bass_boost_cap_conf_min=bass_boost_cap_conf_min,
            bass_boost_cap_extra_db=bass_boost_cap_extra_db,
            bass_boost_cap_hz=bass_boost_cap_hz,
        )
    _prepare_boost_caps_write_stats(
        st,
        mask_c=mask_c,
        freq_axis=freq_axis,
        max_boost_db_base=max_boost_db_base,
        bass_adaptive_isolation_mode=bass_adaptive_isolation_mode,
        bass_boost_cap_enabled=bass_boost_cap_enabled,
        bass_boost_cap_hz=bass_boost_cap_hz,
        bass_boost_cap_conf_min=bass_boost_cap_conf_min,
        bass_boost_cap_extra_db=bass_boost_cap_extra_db,
        bass_boost_post_restore_enable=bass_boost_post_restore_enable,
        bass_boost_post_restore_strength=bass_boost_post_restore_strength,
        boost_cap_db=boost_cap_db,
        logger=logger,
        gain_policy=gain_policy,
        cfg_reader=cfg_reader,
        max_cut_db=max_cut_db,
    )
    return {
        "max_cut_db": float(max_cut_db),
        "max_boost_db_base": float(max_boost_db_base),
        "boost_cap_db": np.asarray(boost_cap_db, dtype=float),
        "bass_boost_cap_hz": float(bass_boost_cap_hz),
        "bass_boost_post_restore_enable": bool(bass_boost_post_restore_enable),
        "bass_boost_post_restore_strength": float(bass_boost_post_restore_strength),
    }


def _prepare_boost_caps_confidence_cap(
    *,
    conf_mask: np.ndarray,
    mask_c: np.ndarray,
    freq_axis: np.ndarray,
    gain_db: np.ndarray,
    boost_cap_db: np.ndarray,
    max_boost_db_base: float,
    bass_boost_cap_conf_min: float,
    bass_boost_cap_extra_db: float,
    bass_boost_cap_hz: float,
) -> tuple[np.ndarray, bool]:
    try:
        c = np.asarray(conf_mask, dtype=float)
        c = np.clip(np.nan_to_num(c, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        w = np.clip(
            (c - float(bass_boost_cap_conf_min)) / max(1e-9, 1.0 - float(bass_boost_cap_conf_min)),
            0.0,
            1.0,
        )
        bass_boost_cap_mask = mask_c & (freq_axis >= 20.0) & (freq_axis <= float(bass_boost_cap_hz))
        if np.any(bass_boost_cap_mask):
            local_cap = (
                float(max_boost_db_base)
                + float(bass_boost_cap_extra_db) * np.asarray(w[bass_boost_cap_mask], dtype=float)
            )
            boost_cap_db[bass_boost_cap_mask] = local_cap
            bass_boost_cap_enabled = bool(np.any(boost_cap_db[bass_boost_cap_mask] > (max_boost_db_base + 1e-6)))
            return bass_boost_cap_mask, bass_boost_cap_enabled
    except (TypeError, ValueError, FloatingPointError, IndexError):
        pass
    return np.zeros_like(mask_c, dtype=bool), False


def _prepare_boost_caps_write_stats(
    st,
    *,
    mask_c: np.ndarray,
    freq_axis: np.ndarray,
    max_boost_db_base: float,
    bass_adaptive_isolation_mode: bool,
    bass_boost_cap_enabled: bool,
    bass_boost_cap_hz: float,
    bass_boost_cap_conf_min: float,
    bass_boost_cap_extra_db: float,
    bass_boost_post_restore_enable: bool,
    bass_boost_post_restore_strength: float,
    boost_cap_db: np.ndarray,
    logger,
    gain_policy,
    cfg_reader: CfgReader,
    max_cut_db: float,
) -> None:
    _prepare_boost_caps_store_stats(
        st,
        mask_c=mask_c,
        freq_axis=freq_axis,
        max_boost_db_base=max_boost_db_base,
        bass_adaptive_isolation_mode=bass_adaptive_isolation_mode,
        bass_boost_cap_enabled=bass_boost_cap_enabled,
        bass_boost_cap_hz=bass_boost_cap_hz,
        bass_boost_cap_conf_min=bass_boost_cap_conf_min,
        bass_boost_cap_extra_db=bass_boost_cap_extra_db,
        bass_boost_post_restore_enable=bass_boost_post_restore_enable,
        bass_boost_post_restore_strength=bass_boost_post_restore_strength,
        boost_cap_db=boost_cap_db,
    )
    _prepare_boost_caps_log_diagnostics(
        logger=logger,
        gain_policy=gain_policy,
        cfg_reader=cfg_reader,
        max_cut_db=max_cut_db,
        max_boost_db_base=max_boost_db_base,
        bass_adaptive_isolation_mode=bass_adaptive_isolation_mode,
        bass_boost_cap_enabled=bass_boost_cap_enabled,
        bass_boost_cap_hz=bass_boost_cap_hz,
        bass_boost_cap_conf_min=bass_boost_cap_conf_min,
        bass_boost_cap_extra_db=bass_boost_cap_extra_db,
        bass_boost_post_restore_enable=bass_boost_post_restore_enable,
        bass_boost_post_restore_strength=bass_boost_post_restore_strength,
    )
    _prepare_boost_caps_harmonic_risk(
        st,
        freq_axis=freq_axis,
        mask_c=mask_c,
        boost_cap_db=boost_cap_db,
        unsafe_raw_dsp=cfg_reader.bool("unsafe_raw_dsp", False),
    )


def _prepare_boost_caps_store_stats(
    st,
    *,
    mask_c: np.ndarray,
    freq_axis: np.ndarray,
    max_boost_db_base: float,
    bass_adaptive_isolation_mode: bool,
    bass_boost_cap_enabled: bool,
    bass_boost_cap_hz: float,
    bass_boost_cap_conf_min: float,
    bass_boost_cap_extra_db: float,
    bass_boost_post_restore_enable: bool,
    bass_boost_post_restore_strength: float,
    boost_cap_db: np.ndarray,
) -> None:
    try:
        if isinstance(st, dict):
            b20 = mask_c & (freq_axis >= 20.0) & (freq_axis <= 200.0)
            extra_mean = 0.0
            extra_max = 0.0
            if np.any(b20):
                extra = np.maximum(0.0, np.asarray(boost_cap_db[b20], dtype=float) - float(max_boost_db_base))
                extra_mean = float(np.mean(extra))
                extra_max = float(np.max(extra))
            safe_put_many(
                st,
                {
                    "bass_adaptive_isolation_mode": bool(bass_adaptive_isolation_mode),
                    "bass_boost_cap_enabled": bool(bass_boost_cap_enabled),
                    "bass_boost_cap_hz": float(bass_boost_cap_hz),
                    "bass_boost_cap_conf_min": float(bass_boost_cap_conf_min),
                    "bass_boost_cap_extra_db": float(bass_boost_cap_extra_db),
                    "bass_boost_post_restore_enable": bool(bass_boost_post_restore_enable),
                    "bass_boost_post_restore_strength": float(bass_boost_post_restore_strength),
                    "bass_boost_cap_avg_extra_db_20_200": float(extra_mean),
                    "bass_boost_cap_max_extra_db_20_200": float(extra_max),
                },
            )
    except (TypeError, ValueError):
        pass


def _prepare_boost_caps_log_diagnostics(
    *,
    logger,
    gain_policy,
    cfg_reader: CfgReader,
    max_cut_db: float,
    max_boost_db_base: float,
    bass_adaptive_isolation_mode: bool,
    bass_boost_cap_enabled: bool,
    bass_boost_cap_hz: float,
    bass_boost_cap_conf_min: float,
    bass_boost_cap_extra_db: float,
    bass_boost_post_restore_enable: bool,
    bass_boost_post_restore_strength: float,
) -> None:
    try:
        logger.info(
            "Diagnostic: "
            f"max_boost_db={float(max_boost_db_base):.2f} dB, "
            f"bass_adaptive_isolation={'ON' if bool(bass_adaptive_isolation_mode) else 'OFF'}, "
            f"bass_boost_cap={'ON' if bass_boost_cap_enabled else 'OFF'} "
            f"(extra={float(bass_boost_cap_extra_db):.2f} dB, hz<={float(bass_boost_cap_hz):.1f}, conf_min={float(bass_boost_cap_conf_min):.2f}), "
            f"bass_boost_post_restore={'ON' if bool(bass_boost_post_restore_enable) else 'OFF'} "
            f"(strength={float(bass_boost_post_restore_strength):.2f}), "
            f"max_cut_db={float(max_cut_db):.2f} dB, "
            f"low_bass_cut_hz={float(gain_policy.low_cut_hz):.1f} Hz, "
            f"exc_prot={'ON' if bool(gain_policy.exc_prot) else 'OFF'}, "
            f"exc_freq={float(gain_policy.exc_freq):.1f} Hz, "
            f"do_normalize={'ON' if cfg_reader.bool('do_normalize', False) else 'OFF'}, "
            f"global_gain_db={float(cfg_reader.float_allow_zero('global_gain_db', 0.0)):.2f} dB, "
            f"max_slope_db_per_oct={float(cfg_reader.float_allow_zero('max_slope_db_per_oct', 0.0)):.1f}"
        )
    except (AttributeError, TypeError, ValueError):
        pass


def _prepare_boost_caps_harmonic_risk(
    st,
    *,
    freq_axis: np.ndarray,
    mask_c: np.ndarray,
    boost_cap_db: np.ndarray,
    unsafe_raw_dsp: bool = False,
) -> None:
    _hrisk_cap_enabled = False
    _hrisk_cap_bypassed = bool(unsafe_raw_dsp)
    _hrisk_peak = 0.0
    _hrisk_peak_hz = float("nan")
    _hrisk_avg_reduction_20_200 = 0.0
    _hrisk_max_reduction_20_200 = 0.0
    if not unsafe_raw_dsp:
        try:
            _mctx = get_measurement_ctx()
            if (
                _mctx is not None
                and _mctx.harmonic_risk_freq_hz is not None
                and _mctx.harmonic_risk_curve is not None
            ):
                _hrisk_f = np.asarray(_mctx.harmonic_risk_freq_hz, dtype=float)
                _hrisk_c = np.asarray(_mctx.harmonic_risk_curve, dtype=float)
                if _hrisk_f.size >= 4 and _hrisk_c.size == _hrisk_f.size:
                    _risk_on_axis = np.interp(
                        freq_axis,
                        _hrisk_f,
                        _hrisk_c,
                        left=0.0,
                        right=0.0,
                    )
                    _hrisk_band = mask_c & (freq_axis >= 20.0) & (freq_axis <= 800.0)
                    _max_local_reduction_db = 2.0
                    _cap_reduction = np.zeros_like(freq_axis, dtype=float)
                    if np.any(_hrisk_band):
                        _cap_reduction[_hrisk_band] = (
                            float(_max_local_reduction_db) * np.asarray(_risk_on_axis[_hrisk_band], dtype=float)
                        )
                        boost_cap_db[_hrisk_band] = np.maximum(
                            0.0,
                            boost_cap_db[_hrisk_band] - _cap_reduction[_hrisk_band],
                        )
                        _hrisk_cap_enabled = bool(np.any(_cap_reduction[_hrisk_band] > 0.01))
                        if np.any(np.isfinite(_risk_on_axis[_hrisk_band])):
                            _peak_idx = int(np.argmax(_risk_on_axis[_hrisk_band]))
                            _hrisk_peak = float(np.max(_risk_on_axis[_hrisk_band]))
                            _band_freqs = freq_axis[_hrisk_band]
                            if _peak_idx < len(_band_freqs):
                                _hrisk_peak_hz = float(_band_freqs[_peak_idx])
                        _b20_200 = mask_c & (freq_axis >= 20.0) & (freq_axis <= 200.0)
                        if np.any(_b20_200):
                            _hrisk_avg_reduction_20_200 = float(np.mean(_cap_reduction[_b20_200]))
                            _hrisk_max_reduction_20_200 = float(np.max(_cap_reduction[_b20_200]))
        except (TypeError, ValueError, FloatingPointError, IndexError):
            pass
    try:
        if isinstance(st, dict):
            safe_put_many(
                st,
                {
                    "harmonic_risk_cap_enabled": bool(_hrisk_cap_enabled),
                    "harmonic_risk_cap_bypassed_by_unsafe_raw": bool(_hrisk_cap_bypassed),
                    "harmonic_risk_peak": float(_hrisk_peak),
                    "harmonic_risk_peak_hz": float(_hrisk_peak_hz),
                    "harmonic_risk_cap_avg_reduction_20_200": float(_hrisk_avg_reduction_20_200),
                    "harmonic_risk_cap_max_reduction_20_200": float(_hrisk_max_reduction_20_200),
                },
            )
    except (TypeError, ValueError):
        pass

def _stats_array(st, keys: tuple[str, ...], shape: tuple[int, ...]) -> tuple[np.ndarray | None, str]:
    if not isinstance(st, dict):
        return None, "missing"
    for key in keys:
        if key not in st:
            continue
        try:
            arr = np.asarray(st.get(key), dtype=float)
        except (TypeError, ValueError):
            continue
        if arr.shape == shape:
            return arr, "stats"
    return None, "missing"

def _authority_band_metrics(
    *,
    freq_axis: np.ndarray,
    mask_c: np.ndarray,
    reduction_db: np.ndarray,
    cap_db: np.ndarray,
) -> tuple[float, float, float]:
    try:
        red = np.asarray(reduction_db, dtype=float)
        cap = np.asarray(cap_db, dtype=float)
        band = np.asarray(mask_c, dtype=bool) & (np.asarray(freq_axis, dtype=float) >= 20.0) & (freq_axis <= 300.0)
        max_reduction = float(np.max(red[mask_c])) if np.any(mask_c) else 0.0
        mean_reduction = float(np.mean(red[band])) if np.any(band) else 0.0
        min_cap = float(np.min(cap[band])) if np.any(band) else 0.0
        return max(0.0, max_reduction), max(0.0, mean_reduction), max(0.0, min_cap)
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return 0.0, 0.0, 0.0


__all__ = ['_apply_low_frequency_policy', '_prepare_boost_caps', '_stats_array', '_authority_band_metrics']
