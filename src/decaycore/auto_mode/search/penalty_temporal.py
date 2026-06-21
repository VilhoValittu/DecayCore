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

import numpy as np

_logger = logging.getLogger(__name__)

from .. import shared

_RECOVERABLE_TEMPORAL_PENALTY_EXCEPTIONS = (
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
)


def _harmonic_band_mask(
    harmonic_freq_hz: np.ndarray | None,
    *,
    lo_hz: float,
    hi_hz: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    if harmonic_freq_hz is None:
        return None
    freq = np.asarray(harmonic_freq_hz, dtype=float).reshape(-1)
    band_mask = (freq >= float(lo_hz)) & (freq <= float(hi_hz)) & np.isfinite(freq)
    if int(np.count_nonzero(band_mask)) < 4:
        return None
    return freq, band_mask


def _harmonic_fundamental_on_harmonic_grid(
    *,
    use_relative: bool,
    fundamental_freq_hz: np.ndarray | None,
    fundamental_mag_db: np.ndarray | None,
    freq: np.ndarray,
) -> tuple[bool, np.ndarray | None]:
    if not use_relative:
        return False, None
    fund_f = np.asarray(fundamental_freq_hz, dtype=float).reshape(-1)
    fund_m = np.asarray(fundamental_mag_db, dtype=float).reshape(-1)
    valid = np.isfinite(fund_f) & np.isfinite(fund_m)
    if valid.sum() < 4:
        return False, None
    return True, np.interp(freq, fund_f[valid], fund_m[valid])


def _harmonic_proxy_values(
    *,
    harmonic_magnitudes_db: dict | None,
    freq: np.ndarray,
    band_mask: np.ndarray,
    use_relative: bool,
    fund_at_harm: np.ndarray | None,
) -> np.ndarray:
    h_arrays = []
    for order in (2, 3):
        arr = (harmonic_magnitudes_db or {}).get(order)
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float).reshape(-1)
        if a.size != freq.size:
            continue
        band_vals = a[band_mask]
        if use_relative and fund_at_harm is not None:
            band_vals = band_vals - fund_at_harm[band_mask]
        h_arrays.append(band_vals)
    if not h_arrays:
        return np.asarray([], dtype=float)
    proxy = np.maximum.reduce(h_arrays)
    proxy = proxy[np.isfinite(proxy)]
    return np.asarray(proxy, dtype=float)


def _harmonic_severity_from_proxy(
    proxy: np.ndarray,
    *,
    threshold_db: float,
    thd_range_db: float,
) -> float:
    if proxy.size == 0:
        return 0.0
    severity_vals = np.clip(
        (proxy - float(threshold_db)) / max(float(thd_range_db), 1e-6),
        0.0,
        1.0,
    )
    return float(np.mean(severity_vals))


def _auto_harmonic_boost_penalty(
    st: dict | None,
    harmonic_freq_hz: np.ndarray | None,
    harmonic_magnitudes_db: dict | None,
    *,
    lo_hz: float = 20.0,
    hi_hz: float = 800.0,
    thd_threshold_db: float = -35.0,
    thd_range_db: float = 15.0,
    scale: float = 0.15,
    fundamental_freq_hz: np.ndarray | None = None,
    fundamental_mag_db: np.ndarray | None = None,
) -> float:
    """Penaltti boostille korkean harmonisen säröön alueilla.

    Jos ``fundamental_freq_hz`` ja ``fundamental_mag_db`` annetaan, harmoniat
    normalisoidaan suhteessa fundamentaaliin (dBr) ennen kynnysvertailua, ja
    käytetään suhteellista kynnystä -30 dBr. Muuten verrataan absoluuttiseen
    dBFS-kynnykseen ``thd_threshold_db``.
    Palauttaa 0.0 jos harmonidata puuttuu tai mikään ehto ei täyty.
    """
    if harmonic_magnitudes_db is None:
        return 0.0
    try:
        st = dict(st or {})
        net_boost = float(max(0.0, shared._auto_safe_float(st.get("net_boost_peak_db", 0.0), 0.0)))
        if net_boost <= 0.0:
            return 0.0

        prepared = _harmonic_band_mask(
            harmonic_freq_hz,
            lo_hz=float(lo_hz),
            hi_hz=float(hi_hz),
        )
        if prepared is None:
            return 0.0
        freq, band_mask = prepared

        use_relative = (
            fundamental_freq_hz is not None
            and fundamental_mag_db is not None
        )
        use_relative, fund_at_harm = _harmonic_fundamental_on_harmonic_grid(
            use_relative=bool(use_relative),
            fundamental_freq_hz=fundamental_freq_hz,
            fundamental_mag_db=fundamental_mag_db,
            freq=freq,
        )
        thd_proxy = _harmonic_proxy_values(
            harmonic_magnitudes_db=harmonic_magnitudes_db,
            freq=freq,
            band_mask=band_mask,
            use_relative=bool(use_relative),
            fund_at_harm=fund_at_harm,
        )
        if thd_proxy.size == 0:
            return 0.0

        effective_threshold = -30.0 if use_relative else float(thd_threshold_db)
        severity = _harmonic_severity_from_proxy(
            thd_proxy,
            threshold_db=float(effective_threshold),
            thd_range_db=float(thd_range_db),
        )
        if severity <= 0.0:
            return 0.0

        penalty = float(net_boost) * float(severity) * float(scale)
        return float(max(0.0, penalty))
    except _RECOVERABLE_TEMPORAL_PENALTY_EXCEPTIONS:
        return 0.0


def _auto_rt60_policy_penalty(
    base_data: dict | None,
    l_st: dict | None,
    r_st: dict | None,
    *,
    rt60_high_threshold_s: float = 0.65,
    rt60_extreme_threshold_s: float = 1.0,
    scale: float = 0.05,
) -> float:
    """Small penalty when high RT60 meets high net boost (reverberation risk).

    Conservative secondary signal - max contribution ~0.05 * net_boost.
    """
    try:
        bd = dict(base_data or {})
        rt60_l = shared._auto_safe_float(
            (bd.get("rt60_summary_l") or {}).get("rt60_val", float("nan")),
            float("nan"),
        )
        rt60_r = shared._auto_safe_float(
            (bd.get("rt60_summary_r") or {}).get("rt60_val", float("nan")),
            float("nan"),
        )
        rt60_vals = [v for v in (rt60_l, rt60_r) if np.isfinite(v) and v > 0.0]
        if not rt60_vals:
            return 0.0
        rt60_max = float(max(rt60_vals))
        excess = max(0.0, rt60_max - float(rt60_high_threshold_s))
        if excess <= 0.0:
            return 0.0
        net_boost_l = float(max(0.0, shared._auto_safe_float((l_st or {}).get("net_boost_peak_db", 0.0), 0.0)))
        net_boost_r = float(max(0.0, shared._auto_safe_float((r_st or {}).get("net_boost_peak_db", 0.0), 0.0)))
        net_boost = max(net_boost_l, net_boost_r)
        if net_boost <= 0.0:
            return 0.0
        severity = float(np.clip(
            excess / max(float(rt60_extreme_threshold_s) - float(rt60_high_threshold_s), 0.1),
            0.0,
            1.0,
        ))
        return float(max(0.0, float(net_boost) * float(severity) * float(scale)))
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
        return 0.0


AUTO_TDC_DECAY_SCORING_VERSION = 2


def _auto_rt60_band_pairs(base_data: dict | None) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    bd = dict(base_data or {})
    for key in ("measured_rt60_bands_l", "measured_rt60_bands_r"):
        bands = bd.get(key)
        if not isinstance(bands, dict):
            continue
        for freq, rt60 in bands.items():
            f = shared._auto_safe_float(freq, float("nan"))
            v = shared._auto_safe_float(rt60, float("nan"))
            if np.isfinite(f) and np.isfinite(v) and f > 0.0 and v > 0.0:
                pairs.append((float(f), float(v)))
    return pairs


def _auto_tdc_decay_config(base_data: dict | None) -> dict:
    bd = dict(base_data or {})
    return {
        "bd": bd,
        "penalty_cap": max(
            0.0,
            shared._auto_safe_float(
                bd.get("auto_mode_tdc_decay_penalty_cap", shared.AUTO_MODE_TDC_DECAY_PENALTY_CAP),
                shared.AUTO_MODE_TDC_DECAY_PENALTY_CAP,
            ),
        ),
        "penalty_weight": max(
            0.0,
            shared._auto_safe_float(
                bd.get("auto_mode_tdc_decay_penalty_weight", shared.AUTO_MODE_TDC_DECAY_PENALTY_WEIGHT),
                shared.AUTO_MODE_TDC_DECAY_PENALTY_WEIGHT,
            ),
        ),
        "extreme_peak_db": max(
            0.0,
            shared._auto_safe_float(
                bd.get("auto_mode_tdc_extreme_peak_reduction_db", shared.AUTO_MODE_TDC_EXTREME_PEAK_REDUCTION_DB),
                shared.AUTO_MODE_TDC_EXTREME_PEAK_REDUCTION_DB,
            ),
        ),
        "low_need_threshold": float(
            np.clip(
                shared._auto_safe_float(
                    bd.get("auto_mode_tdc_low_need_threshold", shared.AUTO_MODE_TDC_LOW_NEED_THRESHOLD),
                    shared.AUTO_MODE_TDC_LOW_NEED_THRESHOLD,
                ),
                0.0,
                1.0,
            )
        ),
        "target_low_s": max(
            0.01,
            shared._auto_safe_float(
                bd.get("auto_mode_tdc_rt60_target_low_s", shared.AUTO_MODE_TDC_RT60_TARGET_LOW_S),
                shared.AUTO_MODE_TDC_RT60_TARGET_LOW_S,
            ),
        ),
        "target_upper_s": max(
            0.01,
            shared._auto_safe_float(
                bd.get("auto_mode_tdc_rt60_target_upper_s", shared.AUTO_MODE_TDC_RT60_TARGET_UPPER_S),
                shared.AUTO_MODE_TDC_RT60_TARGET_UPPER_S,
            ),
        ),
        "low_max_hz": max(
            20.0,
            shared._auto_safe_float(
                bd.get("auto_mode_tdc_rt60_low_max_hz", shared.AUTO_MODE_TDC_RT60_LOW_MAX_HZ),
                shared.AUTO_MODE_TDC_RT60_LOW_MAX_HZ,
            ),
        ),
        "eval_max_hz": max(
            20.0,
            shared._auto_safe_float(
                bd.get("auto_mode_tdc_rt60_eval_max_hz", shared.AUTO_MODE_TDC_RT60_EVAL_MAX_HZ),
                shared.AUTO_MODE_TDC_RT60_EVAL_MAX_HZ,
            ),
        ),
    }


def _auto_tdc_no_metadata_payload(*, strength: float, decision: str, action_hint: str) -> dict:
    return {
        "decay_metadata_available": False,
        "tdc_strength": float(strength),
        "tdc_decay_need": 0.0,
        "tdc_weak_penalty": 0.0,
        "tdc_overdamping_penalty": 0.0,
        "tdc_overreach_penalty": 0.0,
        "tdc_decay_penalty_total": 0.0,
        "tdc_decay_penalty": 0.0,
        "tdc_decision": str(decision),
        "tdc_action_hint": str(action_hint),
        "tdc_extreme_overreach": False,
        "tdc_decay_scoring_version": int(AUTO_TDC_DECAY_SCORING_VERSION),
    }


def _auto_tdc_modal_decay_need(
    *,
    pairs: list[tuple[float, float]],
    eval_max_hz: float,
    low_max_hz: float,
    target_low_s: float,
    target_upper_s: float,
) -> tuple[float, bool, float, float, float]:
    weighted_excess = 0.0
    weight_sum = 0.0
    strongest_freq = float("nan")
    strongest_excess = 0.0
    for freq, rt60 in pairs:
        if freq < 20.0 or freq > float(eval_max_hz):
            continue
        if freq <= float(low_max_hz):
            weight = 1.0
            target = target_low_s
        else:
            weight = 0.40
            target = target_upper_s
        excess = max(0.0, float(rt60) - float(target))
        weighted_excess += float(weight) * float(excess)
        weight_sum += float(weight)
        if excess > strongest_excess:
            strongest_excess = float(excess)
            strongest_freq = float(freq)
    modal_excess_s = float(weighted_excess / weight_sum) if weight_sum > 0.0 else 0.0
    metadata_available = bool(weight_sum > 0.0)
    need = float(np.clip(modal_excess_s / 0.35, 0.0, 1.0))
    return (
        float(need),
        bool(metadata_available),
        float(modal_excess_s),
        float(strongest_freq if np.isfinite(strongest_freq) else float("nan")),
        float(strongest_excess),
    )


def _auto_tdc_channel_stats(l_st: dict | None, r_st: dict | None) -> dict:
    stats = [dict(l_st or {}), dict(r_st or {})]

    def _finite_max(keys: tuple[str, ...], *, minimum_positive: bool = True) -> float:
        vals = [
            shared._auto_safe_float(st.get(key, float("nan")), float("nan"))
            for st in stats
            for key in keys
        ]
        vals = [float(v) for v in vals if np.isfinite(v) and (v > 0.0 if minimum_positive else True)]
        return float(max(vals)) if vals else 0.0

    peak_hz_vals = [
        shared._auto_safe_float(st.get("tdc_peak_reduction_hz", float("nan")), float("nan"))
        for st in stats
    ]
    peak_hz_vals = [float(v) for v in peak_hz_vals if np.isfinite(v) and v > 0.0]
    band_los = [
        shared._auto_safe_float(st.get("tdc_reduction_band_low_hz", float("nan")), float("nan"))
        for st in stats
    ]
    band_his = [
        shared._auto_safe_float(st.get("tdc_reduction_band_high_hz", float("nan")), float("nan"))
        for st in stats
    ]
    band_los = [float(v) for v in band_los if np.isfinite(v) and v > 0.0]
    band_his = [float(v) for v in band_his if np.isfinite(v) and v > 0.0]
    event_vals = [
        shared._auto_safe_float(st.get("tdc_events_used", 0.0), 0.0)
        for st in stats
    ]
    return {
        "tdc_peak_db": _finite_max(("tdc_peak_reduction_db",)),
        "tdc_peak_hz": float(peak_hz_vals[0]) if peak_hz_vals else float("nan"),
        "tdc_area": _finite_max(("tdc_reduction_area_db_hz",)),
        "tdc_events": float(max([float(v) for v in event_vals if np.isfinite(v)] or [0.0])),
        "tdc_band_lo": float(min(band_los)) if band_los else float("nan"),
        "tdc_band_hi": float(max(band_his)) if band_his else float("nan"),
    }


def _auto_tdc_decay_decision(*, strength: float, optimum: float, need: float) -> tuple[str, str]:
    keep_tol = 0.08
    if need <= 0.12:
        decision = "reduced_by_optimizer" if strength <= 0.35 else "possible_overdamping_risk"
        action_hint = "keep_tdc" if strength <= 0.35 else "decrease_tdc"
        return decision, action_hint
    if abs(strength - optimum) <= keep_tol:
        return "tdc_helped", "keep_tdc"
    if strength < optimum:
        return "weak_tdc_left_decay", "increase_tdc"
    return "strong_tdc_overdamping_risk", "decrease_tdc"


def _auto_tdc_decay_tradeoff(
    base_data: dict | None,
    l_st: dict | None,
    r_st: dict | None,
) -> tuple[float, dict]:
    """Band-limited AUTO TDC decay tradeoff penalty.

    This uses cheap metadata already available during AUTO scoring. It does not
    estimate new impulse responses inside the optimization loop.
    """
    cfg = _auto_tdc_decay_config(base_data)
    bd = dict(cfg.get("bd", {}) or {})
    penalty_cap = float(cfg["penalty_cap"])
    penalty_weight = float(cfg["penalty_weight"])
    extreme_peak_db = float(cfg["extreme_peak_db"])
    low_need_threshold = float(cfg["low_need_threshold"])
    target_low_s = float(cfg["target_low_s"])
    target_upper_s = float(cfg["target_upper_s"])
    low_max_hz = float(cfg["low_max_hz"])
    eval_max_hz = float(cfg["eval_max_hz"])
    if not bool(bd.get("enable_tdc", True)):
        return 0.0, _auto_tdc_no_metadata_payload(
            strength=0.0,
            decision="tdc_disabled",
            action_hint="tdc_disabled",
        )

    strength_pct = shared._auto_safe_float(bd.get("tdc_strength", float("nan")), float("nan"))
    if not np.isfinite(strength_pct):
        strength_pct = 50.0
    strength = float(np.clip(float(strength_pct) / 100.0, 0.0, 1.0))
    pairs = _auto_rt60_band_pairs(bd)
    need, metadata_available, modal_excess_s, strongest_freq, strongest_excess = _auto_tdc_modal_decay_need(
        pairs=pairs,
        eval_max_hz=float(eval_max_hz),
        low_max_hz=float(low_max_hz),
        target_low_s=float(target_low_s),
        target_upper_s=float(target_upper_s),
    )
    tdc_stats = _auto_tdc_channel_stats(l_st, r_st)
    tdc_peak_db = float(tdc_stats["tdc_peak_db"])
    tdc_peak_hz = float(tdc_stats["tdc_peak_hz"])
    tdc_area = float(tdc_stats["tdc_area"])
    tdc_events = float(tdc_stats["tdc_events"])
    tdc_band_lo = float(tdc_stats["tdc_band_lo"])
    tdc_band_hi = float(tdc_stats["tdc_band_hi"])

    if tdc_events > 0.0 and not metadata_available:
        need = max(need, float(np.clip(tdc_peak_db / 5.0, 0.0, 0.65)))
    if not metadata_available and tdc_events <= 0.0:
        return 0.0, _auto_tdc_no_metadata_payload(
            strength=float(strength),
            decision="neutral_no_decay_metadata",
            action_hint="neutral_no_metadata",
        )

    optimum = float(np.clip(0.24 + 0.36 * need, 0.20, 0.75))
    weak_pen = 6.0 * need * max(0.0, optimum - strength) ** 2 / max(optimum * optimum, 1e-6)
    dry_need = max(0.0, strength - optimum)
    dry_pen = (1.0 - 0.55 * need) * 5.0 * dry_need * dry_need / max((1.0 - optimum) ** 2, 1e-6)
    overreach_pen = 0.0
    if tdc_peak_db > 0.0:
        expected_peak = 1.0 + 6.5 * max(need, 0.15)
        overreach_pen += 0.18 * max(0.0, tdc_peak_db - expected_peak)
    overreach_pen += 0.0015 * max(0.0, tdc_area - (120.0 + 380.0 * need))
    extreme_overreach = bool(extreme_peak_db > 0.0 and tdc_peak_db >= extreme_peak_db and need <= low_need_threshold)
    if extreme_overreach:
        overreach_pen += 4.0 + 0.25 * max(0.0, tdc_peak_db - extreme_peak_db)
    raw_penalty = float(max(0.0, weak_pen + dry_pen + overreach_pen))
    penalty = float(np.clip(raw_penalty * penalty_weight, 0.0, penalty_cap))

    improvement = 0.0
    if need > 0.0:
        improvement = float(np.clip((min(strength, optimum) / max(optimum, 1e-6)) * need, 0.0, 1.0))
    decision, action_hint = _auto_tdc_decay_decision(
        strength=float(strength),
        optimum=float(optimum),
        need=float(need),
    )

    return penalty, {
        "decay_metadata_available": bool(metadata_available),
        "tdc_strength": float(strength),
        "tdc_decay_need": float(need),
        "tdc_weak_penalty": float(weak_pen * penalty_weight),
        "tdc_overdamping_penalty": float(dry_pen * penalty_weight),
        "tdc_overreach_penalty": float(overreach_pen * penalty_weight),
        "tdc_decay_penalty_total": float(penalty),
        "tdc_decay_penalty": float(penalty),
        "tdc_decay_optimum_strength": float(optimum),
        "modal_decay_excess_s": float(modal_excess_s),
        "strongest_modal_decay_hz": float(strongest_freq) if np.isfinite(strongest_freq) else float("nan"),
        "strongest_modal_decay_excess_s": float(strongest_excess),
        "modal_decay_improvement_estimate": float(improvement),
        "tdc_peak_reduction_db": float(tdc_peak_db),
        "tdc_peak_reduction_hz": float(tdc_peak_hz) if np.isfinite(tdc_peak_hz) else float("nan"),
        "tdc_reduction_area_db_hz": float(tdc_area),
        "tdc_reduction_band_low_hz": float(tdc_band_lo) if np.isfinite(tdc_band_lo) else float("nan"),
        "tdc_reduction_band_high_hz": float(tdc_band_hi) if np.isfinite(tdc_band_hi) else float("nan"),
        "tdc_decision": str(decision),
        "tdc_action_hint": str(action_hint),
        "tdc_extreme_overreach": bool(extreme_overreach),
        "tdc_decay_penalty_cap": float(penalty_cap),
        "tdc_decay_penalty_weight": float(penalty_weight),
        "tdc_decay_scoring_version": int(AUTO_TDC_DECAY_SCORING_VERSION),
    }


def _auto_harmonic_local_boost_penalty(
    base_data: dict | None,
    l_st: dict | None,
    r_st: dict | None,
    *,
    risk_threshold: float = 0.5,
    scale: float = 0.04,
) -> float:
    """Secondary penalty from harmonic risk summary.

    Distinct from _auto_harmonic_boost_penalty which does per-bin analysis.
    Uses the compact risk summary aggregated from repeat/multi-position analysis.

    When bass boost (20-200 Hz) is ≥ 3 dB AND the bass-band harmonic risk
    (mean_risk_20_120) is elevated, the scale is doubled (0.08) to more
    aggressively penalise combinations that risk audible harmonic distortion in
    the low-frequency boost region.
    """
    try:
        bd = dict(base_data or {})
        risk_l = shared._auto_safe_float(
            (bd.get("harmonic_risk_summary_l") or {}).get("peak_risk", float("nan")),
            float("nan"),
        )
        risk_r = shared._auto_safe_float(
            (bd.get("harmonic_risk_summary_r") or {}).get("peak_risk", float("nan")),
            float("nan"),
        )
        risk_vals = [v for v in (risk_l, risk_r) if np.isfinite(v) and v > 0.0]
        if not risk_vals:
            return 0.0
        risk_max = float(max(risk_vals))
        excess = max(0.0, risk_max - float(risk_threshold))
        if excess <= 0.0:
            return 0.0

        # Check for bass-specific boost to scale penalty strength
        def _bass_boost(st: dict | None) -> float:
            s = dict(st or {})
            for key in ("bass_boost_20_200_db", "lf_boost_max_db"):
                v = shared._auto_safe_float(s.get(key, float("nan")), float("nan"))
                if np.isfinite(v) and v > 0.0:
                    return float(v)
            return 0.0

        bass_boost_l = _bass_boost(l_st)
        bass_boost_r = _bass_boost(r_st)
        bass_boost = max(bass_boost_l, bass_boost_r)

        net_boost_l = float(max(0.0, shared._auto_safe_float((l_st or {}).get("net_boost_peak_db", 0.0), 0.0)))
        net_boost_r = float(max(0.0, shared._auto_safe_float((r_st or {}).get("net_boost_peak_db", 0.0), 0.0)))
        net_boost = max(net_boost_l, net_boost_r)
        if net_boost <= 0.0:
            return 0.0

        severity = float(np.clip(excess / max(1.0 - float(risk_threshold), 0.1), 0.0, 1.0))
        eff_scale = float(scale)

        # Bass-specific strengthening: when bass boosting with elevated harmonic risk
        # in the 20–120 Hz band, double the penalty scale (0.08 vs 0.04).
        if bass_boost >= 3.0:
            bass_risk_l = shared._auto_safe_float(
                (bd.get("harmonic_risk_summary_l") or {}).get("mean_risk_20_120", float("nan")),
                float("nan"),
            )
            bass_risk_r = shared._auto_safe_float(
                (bd.get("harmonic_risk_summary_r") or {}).get("mean_risk_20_120", float("nan")),
                float("nan"),
            )
            bass_risk_vals = [v for v in (bass_risk_l, bass_risk_r) if np.isfinite(v) and v > 0.0]
            if bass_risk_vals and max(bass_risk_vals) > 0.35:
                eff_scale = float(scale) * 2.0

        return float(max(0.0, float(net_boost) * float(severity) * float(eff_scale)))
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
        return 0.0



__all__ = ["AUTO_TDC_DECAY_SCORING_VERSION", "_auto_harmonic_boost_penalty", "_auto_rt60_policy_penalty", "_auto_tdc_decay_tradeoff", "_auto_harmonic_local_boost_penalty"]
