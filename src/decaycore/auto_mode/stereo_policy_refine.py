from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Callable

import numpy as np

from ..config.models import (
    ResolvedChannelAutoPolicy,
    StereoAutoPolicyConfig,
    StereoResolvedAutoPolicies,
)
from ..dsp.quality_metrics import (
    band_lr_mismatch_rms_from_stats,
    band_target_error_rms_from_stats,
    worst_channel_relief_db,
)
from .rank_score import official_rank_score
from .shared_parts import _auto_safe_float

logger = logging.getLogger("DecayCore")

_SEVERITY_ASYMMETRY_SCALE = 1.5
_SEVERITY_GATE_THRESHOLD = 0.08


def _policy_from_data(data: dict | None) -> StereoAutoPolicyConfig:
    src = dict(data or {})
    nested = src.get("stereo_auto_policy")
    if isinstance(nested, StereoAutoPolicyConfig):
        return nested
    if isinstance(nested, dict):
        return StereoAutoPolicyConfig.from_dict(nested)
    return StereoAutoPolicyConfig.from_dict(src)


def _resolved_shared_policy(data: dict | None) -> ResolvedChannelAutoPolicy:
    src = dict(data or {})
    return ResolvedChannelAutoPolicy(
        conf_pull_floor=float(_auto_safe_float(src.get("conf_pull_floor", 0.05), 0.05)),
        tdc_strength=float(_auto_safe_float(src.get("tdc_strength", 50.0), 50.0)),
        tdc_max_reduction_db=float(_auto_safe_float(src.get("tdc_max_reduction_db", 9.0), 9.0)),
        bass_first_mode_max_hz=float(_auto_safe_float(src.get("bass_first_mode_max_hz", 200.0), 200.0)),
        low_bass_cut_strength=float(
            np.clip(_auto_safe_float(src.get("low_bass_cut_strength", 0.0), 0.0), 0.0, 1.0)
        ),
        excess_phase_strength=float(
            np.clip(_auto_safe_float(src.get("excess_phase_strength", 0.9), 0.9), 0.0, 1.0)
        ),
    )


def _band_confidence_mean(stats: dict | None, *, lo_hz: float, hi_hz: float) -> float:
    st = dict(stats or {})
    try:
        freq = np.asarray(st.get("freq_axis", []), dtype=float).reshape(-1)
        conf = np.asarray(st.get("confidence_mask", []), dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return float("nan")
    if freq.size < 8 or conf.size != freq.size:
        return float("nan")
    mask = np.isfinite(freq) & np.isfinite(conf) & (freq >= float(lo_hz)) & (freq <= float(hi_hz))
    if int(np.count_nonzero(mask)) < 8:
        return float("nan")
    return float(np.mean(np.asarray(conf[mask], dtype=float)))


def _lf_reflection_severity(stats: dict | None, *, max_hz: float) -> float:
    severity = 0.0
    for ref in list(dict(stats or {}).get("reflections", []) or []):
        if not isinstance(ref, dict):
            continue
        freq_hz = _auto_safe_float(ref.get("freq", float("nan")), float("nan"))
        gd_error = _auto_safe_float(ref.get("gd_error", 0.0), 0.0)
        if np.isfinite(freq_hz) and float(freq_hz) <= float(max_hz):
            severity += max(0.0, float(gd_error) - 2.5)
    return float(severity)


def _gd_grad_from_stats(stats: dict | None) -> float:
    st = dict(stats or {})
    for key in (
        "gd_grad_limiter_after_max_ms_per_oct",
        "gd_grad_limiter_before_max_ms_per_oct",
        "gd_limiter_max_grad_ms_per_oct",
        "gd_grad_limiter_max_grad_ms_per_oct",
        "gd_limiter_max_grad_after_ms_per_oct",
        "gd_grad_limiter_max_grad_after_ms_per_oct",
    ):
        v = _auto_safe_float(st.get(key, float("nan")), float("nan"))
        if np.isfinite(v):
            return float(abs(v))
    return 0.0


def _seed_side_metrics(
    stats: dict | None,
    *,
    split_hz: float,
) -> dict[str, float]:
    st = dict(stats or {})
    return {
        "bass_error_rms_db": float(
            band_target_error_rms_from_stats(
                stats,
                lo_hz=20.0,
                hi_hz=float(split_hz),
            )
        ),
        "bass_confidence_mean": float(_band_confidence_mean(stats, lo_hz=20.0, hi_hz=float(split_hz))),
        "lf_reflection_severity": float(_lf_reflection_severity(stats, max_hz=float(split_hz))),
        "lf_boost_max_db": float(_auto_safe_float(st.get("lf_boost_max_db", 0.0), 0.0)),
        "gd_grad_max_ms_per_oct": float(_gd_grad_from_stats(stats)),
        "gd_abs_max_ms": float(_auto_safe_float(st.get("gd_abs_max_20_500_ms", 0.0), 0.0)),
    }


def _seed_side_score(metrics: dict[str, float]) -> float:
    bass_error = float(_auto_safe_float(metrics.get("bass_error_rms_db", float("nan")), float("nan")))
    conf_mean = float(_auto_safe_float(metrics.get("bass_confidence_mean", float("nan")), float("nan")))
    refl = float(_auto_safe_float(metrics.get("lf_reflection_severity", 0.0), 0.0))
    boost = float(_auto_safe_float(metrics.get("lf_boost_max_db", 0.0), 0.0))
    gd_grad = float(_auto_safe_float(metrics.get("gd_grad_max_ms_per_oct", 0.0), 0.0))
    gd_abs = float(_auto_safe_float(metrics.get("gd_abs_max_ms", 0.0), 0.0))
    score = 0.0
    if np.isfinite(bass_error):
        score += float(bass_error)
    if np.isfinite(conf_mean):
        score += 0.75 * max(0.0, 0.55 - float(conf_mean))
    score += 0.06 * max(0.0, float(refl))
    score += 0.15 * max(0.0, float(boost) - 1.5)
    score += 0.04 * max(0.0, float(gd_grad) - 10.0)
    score += 0.015 * max(0.0, float(gd_abs) - 18.0)
    return float(score)


def _stable_rng_seed(best_preset: dict | None, split_hz: float, worse_side: str) -> int:
    payload = {
        "best_preset": dict(best_preset or {}),
        "split_hz": float(split_hz),
        "worse_side": str(worse_side),
    }
    token = json.dumps(payload, sort_keys=True, default=str).encode("utf-8", "ignore")
    return int(hashlib.sha256(token).hexdigest()[:8], 16) & 0xFFFFFFFF


def _clamp_policy_value(key: str, value: float, shared_value: float, policy: StereoAutoPolicyConfig) -> float:
    limits = {
        "conf_pull_floor": (float(policy.max_confidence_pull_delta), 0.0, 1.0),
        "tdc_strength": (float(policy.max_tdc_strength_delta), 0.0, 100.0),
        "tdc_max_reduction_db": (float(policy.max_tdc_max_reduction_delta_db), 0.0, None),
        "bass_first_mode_max_hz": (float(policy.max_bass_first_mode_max_hz_delta), 20.0, None),
        "low_bass_cut_strength": (float(policy.max_low_bass_cut_strength_delta), 0.0, 1.0),
        "excess_phase_strength": (float(policy.max_excess_phase_strength_delta), 0.0, 1.0),
    }
    max_delta, abs_lo, abs_hi = limits[str(key)]
    out = float(np.clip(value, float(shared_value) - float(max_delta), float(shared_value) + float(max_delta)))
    if abs_lo is not None:
        out = max(float(abs_lo), out)
    if abs_hi is not None:
        out = min(float(abs_hi), out)
    return float(out)


def _build_seed_policy(
    *,
    shared: ResolvedChannelAutoPolicy,
    policy: StereoAutoPolicyConfig,
    severity: float,
) -> ResolvedChannelAutoPolicy:
    sev = float(np.clip(severity, 0.0, 1.0))
    scale = 0.35 + 0.65 * sev
    out = ResolvedChannelAutoPolicy()
    if bool(policy.allow_per_channel_confidence_pull):
        out.conf_pull_floor = _clamp_policy_value(
            "conf_pull_floor",
            float(shared.conf_pull_floor or 0.05) + float(policy.max_confidence_pull_delta) * scale,
            float(shared.conf_pull_floor or 0.05),
            policy,
        )
    if bool(policy.allow_per_channel_tdc):
        out.tdc_strength = _clamp_policy_value(
            "tdc_strength",
            float(shared.tdc_strength or 50.0) + float(policy.max_tdc_strength_delta) * (0.30 + 0.50 * sev),
            float(shared.tdc_strength or 50.0),
            policy,
        )
        out.tdc_max_reduction_db = _clamp_policy_value(
            "tdc_max_reduction_db",
            float(shared.tdc_max_reduction_db or 9.0) + float(policy.max_tdc_max_reduction_delta_db) * (0.25 + 0.55 * sev),
            float(shared.tdc_max_reduction_db or 9.0),
            policy,
        )
    if bool(policy.allow_per_channel_bass_first_mode_max_hz):
        out.bass_first_mode_max_hz = _clamp_policy_value(
            "bass_first_mode_max_hz",
            float(shared.bass_first_mode_max_hz or 200.0) + float(policy.max_bass_first_mode_max_hz_delta) * (0.20 + 0.45 * sev),
            float(shared.bass_first_mode_max_hz or 200.0),
            policy,
        )
    if bool(policy.allow_per_channel_low_bass_cut_strength):
        out.low_bass_cut_strength = _clamp_policy_value(
            "low_bass_cut_strength",
            float(shared.low_bass_cut_strength or 0.0) + float(policy.max_low_bass_cut_strength_delta) * (0.20 + 0.50 * sev),
            float(shared.low_bass_cut_strength or 0.0),
            policy,
        )
    if bool(policy.allow_per_channel_lf_phase_restraint):
        # Reduce excess_phase_strength for the worse channel (more restraint = lower value).
        out.excess_phase_strength = _clamp_policy_value(
            "excess_phase_strength",
            float(shared.excess_phase_strength or 0.9) - float(policy.max_excess_phase_strength_delta) * (0.25 + 0.55 * sev),
            float(shared.excess_phase_strength or 0.9),
            policy,
        )
    return out


def _scaled_policy_variant(
    seed_policy: ResolvedChannelAutoPolicy,
    *,
    shared: ResolvedChannelAutoPolicy,
    policy: StereoAutoPolicyConfig,
    scale: float,
    rng: np.random.Generator | None = None,
) -> ResolvedChannelAutoPolicy:
    out = ResolvedChannelAutoPolicy()
    for key in (
        "conf_pull_floor",
        "tdc_strength",
        "tdc_max_reduction_db",
        "bass_first_mode_max_hz",
        "low_bass_cut_strength",
        "excess_phase_strength",
    ):
        seed_value = getattr(seed_policy, key, None)
        shared_value = getattr(shared, key, None)
        if seed_value is None or shared_value is None:
            continue
        delta = float(seed_value) - float(shared_value)
        jitter = 0.0
        if rng is not None and abs(delta) > 1e-9:
            jitter = float(rng.normal(0.0, abs(delta) * 0.12))
        candidate = float(shared_value) + float(delta) * float(scale) + float(jitter)
        setattr(out, key, _clamp_policy_value(key, candidate, float(shared_value), policy))
    return out


def _gate_reason(metrics: dict | None, policy: StereoAutoPolicyConfig) -> str | None:
    m = dict(metrics or {})
    if bool(m.get("stereo_policy_gate_failed", False)):
        return "stereo_safety_gate"
    relief = float(_auto_safe_float(m.get("stereo_worst_channel_relief_db", float("nan")), float("nan")))
    if np.isfinite(relief) and float(relief) + 1e-9 < float(policy.min_worst_channel_improvement_db):
        return "worst_channel_gain_too_small"
    return None


def _stereo_trial_scales(*, trial_count: int, rng: np.random.Generator) -> list[float]:
    out = [0.35, 0.65, 1.0]
    while len(out) < int(trial_count):
        out.append(float(np.clip(rng.normal(0.90, 0.22), 0.15, 1.20)))
    return list(out[: int(trial_count)])


def _build_stereo_candidate_overlay(
    *,
    shared_policy: ResolvedChannelAutoPolicy,
    variant: ResolvedChannelAutoPolicy,
    worse_side: str,
    split_hz: float,
) -> StereoResolvedAutoPolicies:
    return StereoResolvedAutoPolicies(
        split_hz=float(split_hz),
        shared=shared_policy,
        left=(variant if str(worse_side) == "left" else ResolvedChannelAutoPolicy()),
        right=(variant if str(worse_side) == "right" else ResolvedChannelAutoPolicy()),
    )


def _stereo_candidate_relief_db(
    *,
    shared_result,
    candidate_result,
    split_hz: float,
) -> float:
    return float(
        worst_channel_relief_db(
            dict(getattr(shared_result, "l_st", {}) or {}),
            dict(getattr(shared_result, "r_st", {}) or {}),
            dict(getattr(candidate_result, "l_st", {}) or {}),
            dict(getattr(candidate_result, "r_st", {}) or {}),
            lo_hz=20.0,
            hi_hz=float(split_hz),
        )
    )


def _stereo_refine_is_better(
    *,
    auto_is_better_refine: Callable[..., Any],
    candidate_metrics: dict,
    best_candidate_metrics: dict,
    goal: str,
) -> bool:
    try:
        return bool(
            auto_is_better_refine(
                candidate_metrics,
                best_candidate_metrics,
                goal=goal,
            )
        )
    except Exception:
        logger.exception("stereo policy refine compare")
        return False


def _run_stereo_refine_trials(
    *,
    trial_scales: list[float],
    seed_policy: ResolvedChannelAutoPolicy,
    shared_policy: ResolvedChannelAutoPolicy,
    policy: StereoAutoPolicyConfig,
    shared_preset: dict,
    shared_result,
    split_hz: float,
    worse_side: str,
    rng: np.random.Generator,
    materialize_preset_result: Callable[..., tuple[object, dict, dict]],
    refine_base_data: dict,
    auto_is_better_refine: Callable[..., Any],
    goal: str,
    best_candidate_preset: dict,
    best_candidate_metrics: dict,
) -> tuple[dict, dict, float, dict | None, str | None, float, int]:
    best_candidate_rank = float(official_rank_score(best_candidate_metrics))
    best_candidate_meta = None
    gate_reason = None
    best_failed_relief_db = float("nan")
    trials_executed = 0

    for idx, scale in enumerate(list(trial_scales), start=1):
        variant = _scaled_policy_variant(
            seed_policy,
            shared=shared_policy,
            policy=policy,
            scale=float(scale),
            rng=(rng if idx > 3 else None),
        )
        overlay = _build_stereo_candidate_overlay(
            shared_policy=shared_policy,
            variant=variant,
            worse_side=str(worse_side),
            split_hz=float(split_hz),
        )
        candidate_preset = dict(shared_preset)
        candidate_preset["_stereo_resolved_auto_policies"] = overlay.to_dict()
        candidate_result, candidate_metrics, _candidate_data = materialize_preset_result(
            candidate_preset,
            include_response_arrays=True,
            summarize=False,
            base_data_override=refine_base_data,
        )
        trials_executed = int(idx)
        current_relief = _stereo_candidate_relief_db(
            shared_result=shared_result,
            candidate_result=candidate_result,
            split_hz=float(split_hz),
        )
        candidate_metrics = dict(candidate_metrics or {})
        candidate_metrics["stereo_worst_channel_relief_db"] = (
            float(current_relief) if np.isfinite(current_relief) else float("nan")
        )
        current_gate_reason = _gate_reason(candidate_metrics, policy)
        if current_gate_reason is not None:
            gate_reason = str(current_gate_reason)
            if np.isfinite(current_relief):
                if (not np.isfinite(best_failed_relief_db)) or float(current_relief) > float(best_failed_relief_db):
                    best_failed_relief_db = float(current_relief)
            continue
        better = _stereo_refine_is_better(
            auto_is_better_refine=auto_is_better_refine,
            candidate_metrics=dict(candidate_metrics or {}),
            best_candidate_metrics=dict(best_candidate_metrics or {}),
            goal=str(goal),
        )
        if bool(better):
            best_candidate_preset = dict(candidate_preset)
            best_candidate_metrics = dict(candidate_metrics or {})
            best_candidate_rank = float(official_rank_score(best_candidate_metrics))
            best_candidate_meta = {
                "resolved": overlay.to_dict(),
                "rank": float(best_candidate_rank),
                "scale": float(scale),
                "worst_channel_relief_db": (
                    float(current_relief) if np.isfinite(current_relief) else float("nan")
                ),
            }
    return (
        dict(best_candidate_preset or {}),
        dict(best_candidate_metrics or {}),
        float(best_candidate_rank),
        (dict(best_candidate_meta) if isinstance(best_candidate_meta, dict) else None),
        gate_reason,
        float(best_failed_relief_db),
        int(trials_executed),
    )


def apply_stereo_policy_refine(
    *,
    best_preset: dict | None,
    best_metrics: dict | None,
    base_data_ref: dict | None,
    goal: str,
    phase_label: str,
    materialize_preset_result: Callable[..., tuple[object, dict, dict]],
    auto_is_better_refine: Callable[..., Any],
    status_cb: Callable[[str], None] | None = None,
) -> tuple[dict, dict, bool, dict]:
    policy = _policy_from_data(base_data_ref)
    meta = {
        "applicable": bool(policy.enable_channel_specific_auto_policy),
        "enabled": bool(policy.enable_channel_specific_auto_policy),
        "split_hz": float(policy.channel_specific_policy_max_hz),
        "phase_label": str(phase_label),
        "state": "disabled",
        "gate_passed": False,
        "trials_requested": int(max(0, policy.channel_specific_refine_trials)),
        "trials_executed": 0,
    }
    preset = dict(best_preset or {})
    metrics = dict(best_metrics or {})
    if not bool(policy.enable_channel_specific_auto_policy):
        return preset, metrics, False, meta

    shared_preset = dict(preset)
    shared_preset.pop("_stereo_resolved_auto_policies", None)
    shared_result, shared_metrics, shared_data = materialize_preset_result(
        shared_preset,
        include_response_arrays=True,
        summarize=False,
        base_data_override=base_data_ref,
    )
    shared_rank = float(official_rank_score(shared_metrics))
    meta["shared_rank"] = float(shared_rank)

    l_st = dict(getattr(shared_result, "l_st", {}) or {})
    r_st = dict(getattr(shared_result, "r_st", {}) or {})
    refine_base_data = dict(base_data_ref or {})
    refine_base_data["_stereo_shared_l_st"] = dict(l_st)
    refine_base_data["_stereo_shared_r_st"] = dict(r_st)
    meta["_shared_l_st"] = dict(l_st)
    meta["_shared_r_st"] = dict(r_st)
    split_hz = float(policy.channel_specific_policy_max_hz)
    left_seed_metrics = _seed_side_metrics(l_st, split_hz=split_hz)
    right_seed_metrics = _seed_side_metrics(r_st, split_hz=split_hz)
    left_score = _seed_side_score(left_seed_metrics)
    right_score = _seed_side_score(right_seed_metrics)
    raw_lr_mismatch = float(
        band_lr_mismatch_rms_from_stats(
            l_st,
            r_st,
            lo_hz=20.0,
            hi_hz=split_hz,
            corrected=False,
        )
    )
    score_delta = abs(float(left_score) - float(right_score))
    asymmetry_strength = float(score_delta)
    if np.isfinite(raw_lr_mismatch):
        asymmetry_strength += 0.35 * float(raw_lr_mismatch)
    severity = float(np.clip(asymmetry_strength / _SEVERITY_ASYMMETRY_SCALE, 0.0, 1.0))
    if severity < _SEVERITY_GATE_THRESHOLD:
        meta.update(
            {
                "state": "shared_fallback",
                "gate_reason": "asymmetry_too_small",
                "severity": float(severity),
            }
        )
        return shared_preset, dict(shared_metrics or metrics), False, meta

    worse_side = "left" if float(left_score) >= float(right_score) else "right"
    meta["worse_side"] = str(worse_side)
    meta["severity"] = float(severity)
    shared_policy = _resolved_shared_policy(shared_data)
    seed_policy = _build_seed_policy(
        shared=shared_policy,
        policy=policy,
        severity=severity,
    )
    rng = np.random.default_rng(_stable_rng_seed(shared_preset, split_hz, worse_side))
    trial_count = int(max(1, policy.channel_specific_refine_trials))
    trial_scales = _stereo_trial_scales(trial_count=int(trial_count), rng=rng)

    best_candidate_preset = dict(shared_preset)
    best_candidate_metrics = dict(shared_metrics or {})
    (
        best_candidate_preset,
        best_candidate_metrics,
        best_candidate_rank,
        best_candidate_meta,
        gate_reason,
        best_failed_relief_db,
        trials_executed,
    ) = _run_stereo_refine_trials(
        trial_scales=trial_scales,
        seed_policy=seed_policy,
        shared_policy=shared_policy,
        policy=policy,
        shared_preset=shared_preset,
        shared_result=shared_result,
        split_hz=float(split_hz),
        worse_side=str(worse_side),
        rng=rng,
        materialize_preset_result=materialize_preset_result,
        refine_base_data=refine_base_data,
        auto_is_better_refine=auto_is_better_refine,
        goal=str(goal),
        best_candidate_preset=best_candidate_preset,
        best_candidate_metrics=best_candidate_metrics,
    )
    meta["trials_executed"] = int(trials_executed)

    if best_candidate_meta is None:
        meta.update(
            {
                "state": "shared_fallback",
                "gate_reason": str(gate_reason or "no_refined_winner"),
                "refined_rank": float(shared_rank),
                "worst_channel_relief_db": (
                    float(best_failed_relief_db) if np.isfinite(best_failed_relief_db) else float("nan")
                ),
                "min_worst_channel_improvement_db": float(policy.min_worst_channel_improvement_db),
            }
        )
        return shared_preset, dict(shared_metrics or metrics), False, meta

    best_overlay = StereoResolvedAutoPolicies.from_dict(best_candidate_preset.get("_stereo_resolved_auto_policies"))
    refined_result, refined_metrics, _refined_data = materialize_preset_result(
        best_candidate_preset,
        include_response_arrays=True,
        summarize=False,
        base_data_override=refine_base_data,
    )
    relief = worst_channel_relief_db(
        dict(getattr(shared_result, "l_st", {}) or {}),
        dict(getattr(shared_result, "r_st", {}) or {}),
        dict(getattr(refined_result, "l_st", {}) or {}),
        dict(getattr(refined_result, "r_st", {}) or {}),
        lo_hz=20.0,
        hi_hz=split_hz,
    )
    refined_metrics["stereo_worst_channel_relief_db"] = float(relief) if np.isfinite(relief) else float("nan")
    if np.isfinite(relief) and float(relief) + 1e-9 < float(policy.min_worst_channel_improvement_db):
        meta.update(
            {
                "state": "shared_fallback",
                "gate_reason": "worst_channel_gain_too_small",
                "refined_rank": float(best_candidate_rank),
                "worst_channel_relief_db": float(relief) if np.isfinite(relief) else float("nan"),
                "min_worst_channel_improvement_db": float(policy.min_worst_channel_improvement_db),
            }
        )
        return shared_preset, dict(shared_metrics or metrics), False, meta

    meta.update(
        {
            "state": "applied",
            "gate_passed": True,
            "shared_rank": float(shared_rank),
            "refined_rank": float(official_rank_score(refined_metrics)),
            "resolved": best_overlay.to_dict() if best_overlay is not None else {},
            "stereo_coherence_penalty": float(
                _auto_safe_float(refined_metrics.get("stereo_coherence_penalty", 0.0), 0.0)
            ),
            "phantom_center_stability_penalty": float(
                _auto_safe_float(refined_metrics.get("phantom_center_stability_penalty", 0.0), 0.0)
            ),
            "policy_divergence_penalty": float(
                _auto_safe_float(refined_metrics.get("policy_divergence_penalty", 0.0), 0.0)
            ),
            "worst_channel_relief_db": float(relief) if np.isfinite(relief) else float("nan"),
            "min_worst_channel_improvement_db": float(policy.min_worst_channel_improvement_db),
        }
    )
    if callable(status_cb):
        status_cb(
            "DecayCore automatic mode: stereo LF policy refine applied "
            f"({worse_side} protected below {split_hz:.1f} Hz, rank {shared_rank:.3f} -> {meta['refined_rank']:.3f})"
        )
    return best_candidate_preset, dict(refined_metrics or best_candidate_metrics), True, meta
