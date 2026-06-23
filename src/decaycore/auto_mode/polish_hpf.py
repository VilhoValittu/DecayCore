# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""HPF winner-polish for auto-mode finalize."""

from __future__ import annotations

import logging

import numpy as np

from .auto_mode_profile import profiled_section
from .shared import AUTO_MODE_HPF_ALLOWED_SLOPES_DB_OCT, AUTO_MODE_HPF_MAX_HZ, AUTO_MODE_HPF_MIN_HZ, _auto_safe_float
from .winner_polish_utils import _polish_rank_status, _winner_polish_acceptance

logger = logging.getLogger("DecayCore")


def _hpf_init_meta(*, enabled: bool, phase_label: str) -> dict:
    return {
        "enabled": bool(enabled),
        "applicable": True,
        "phase_label": str(phase_label),
        "seed_freq_hz": float("nan"),
        "seed_slope_db_oct": float("nan"),
        "tested_candidates": [],
        "accepted_candidates": [],
        "tested_count": 0,
        "applied": False,
        "start_enabled": False,
        "start_freq_hz": float("nan"),
        "start_slope_db_oct": float("nan"),
        "final_enabled": False,
        "final_freq_hz": float("nan"),
        "final_slope_db_oct": float("nan"),
        "rank_before": float("nan"),
        "rank_after": float("nan"),
        "avg_before": float("nan"),
        "avg_after": float("nan"),
    }


def _hpf_is_direct_dac_mode(base_data: dict) -> bool:
    bi_mode = str(base_data.get("bass_integration_mode", "") or "").strip().lower()
    return bool(base_data.get("bass_integration_enable", False)) and bi_mode == "direct_dac"


def _hpf_allowed_slopes() -> list[int]:
    allowed = sorted({int(v) for v in tuple(AUTO_MODE_HPF_ALLOWED_SLOPES_DB_OCT) if int(v) > 0})
    return [24] if not allowed else list(allowed)


def _hpf_nearest_allowed_slope(value: float, *, default: int, allowed_slopes: list[int]) -> int:
    cand = _auto_safe_float(value, float("nan"))
    if not np.isfinite(cand) or float(cand) <= 0.0:
        return int(default)
    return int(min(allowed_slopes, key=lambda v: abs(float(v) - float(cand))))


def _hpf_coerce_freq(value, *, default: float) -> float:
    freq_hz = _auto_safe_float(value, default)
    if not np.isfinite(freq_hz) or float(freq_hz) <= 0.0:
        freq_hz = float(default)
    return float(
        round(
            float(
                np.clip(
                    float(freq_hz),
                    float(AUTO_MODE_HPF_MIN_HZ),
                    float(AUTO_MODE_HPF_MAX_HZ),
                )
            ),
            1,
        )
    )


def _build_hpf_candidate_preset(base_preset: dict, *, enabled_state: bool, freq_hz: float, slope_db_oct: int) -> dict:
    order = max(1, int(round(float(slope_db_oct) / 6.0)))
    out = dict(base_preset or {})
    out["_auto_hpf_runtime_override"] = {
        "enabled": bool(enabled_state),
        "freq": float(freq_hz),
        "order": int(order),
    }
    out["hpf_enable"] = bool(enabled_state)
    out["hpf_freq"] = float(freq_hz)
    out["hpf_slope"] = int(slope_db_oct)
    return out


def _hpf_candidate_signature(enabled_state: bool, freq_hz: float, slope_db_oct: int) -> str:
    if not bool(enabled_state):
        return "off"
    return f"on:{float(freq_hz):.1f}:{int(slope_db_oct)}"


def _resolve_hpf_seed_state(*, base_data: dict, allowed_slopes: list[int]) -> tuple[float, int]:
    base_meta = dict(base_data.get("_auto_hpf_meta", {}) or {})
    seed_freq_hz = _hpf_coerce_freq(
        base_meta.get("freq", base_data.get("hpf_freq", 20.0)),
        default=20.0,
    )
    default_slope = int(min(allowed_slopes, key=lambda v: abs(float(v) - 24.0)))
    seed_slope_db_oct = _hpf_nearest_allowed_slope(
        base_meta.get("slope_db_oct", base_data.get("hpf_slope", default_slope)),
        default=default_slope,
        allowed_slopes=allowed_slopes,
    )
    return float(seed_freq_hz), int(seed_slope_db_oct)


def _resolve_hpf_current_state(
    *,
    preset: dict,
    seed_freq_hz: float,
    seed_slope_db_oct: int,
    allowed_slopes: list[int],
) -> tuple[bool, float, int]:
    current_override = dict(preset.get("_auto_hpf_runtime_override", {}) or {})
    current_enabled = bool(current_override.get("enabled", preset.get("hpf_enable", True)))
    current_freq_hz = _hpf_coerce_freq(
        current_override.get("freq", preset.get("hpf_freq", seed_freq_hz)),
        default=seed_freq_hz,
    )
    current_slope_db_oct = _hpf_nearest_allowed_slope(
        preset.get(
            "hpf_slope",
            (float(current_override.get("order", 0) or 0) * 6.0) if current_override else seed_slope_db_oct,
        ),
        default=seed_slope_db_oct,
        allowed_slopes=allowed_slopes,
    )
    return bool(current_enabled), float(current_freq_hz), int(current_slope_db_oct)


def _build_hpf_candidate_defs(
    *,
    current_enabled: bool,
    current_freq_hz: float,
    current_slope_db_oct: int,
    seed_freq_hz: float,
    seed_slope_db_oct: int,
    allowed_slopes: list[int],
) -> tuple[list[dict], float, int]:
    freq_base_hz = float(current_freq_hz) if bool(current_enabled) else float(seed_freq_hz)
    slope_base_db_oct = int(current_slope_db_oct) if bool(current_enabled) else int(seed_slope_db_oct)
    seed_idx = int(allowed_slopes.index(slope_base_db_oct))
    candidate_states: list[tuple[bool, float, int]] = [
        (False, float(freq_base_hz), int(slope_base_db_oct)),
        (True, float(freq_base_hz), int(slope_base_db_oct)),
    ]
    if seed_idx > 0:
        candidate_states.append((True, float(freq_base_hz), int(allowed_slopes[seed_idx - 1])))
    if seed_idx < int(len(allowed_slopes) - 1):
        candidate_states.append((True, float(freq_base_hz), int(allowed_slopes[seed_idx + 1])))
    for freq_offset_hz in (-6.0, -3.0, 3.0, 6.0):
        candidate_states.append(
            (
                True,
                _hpf_coerce_freq(
                    float(freq_base_hz) + float(freq_offset_hz),
                    default=float(freq_base_hz),
                ),
                int(slope_base_db_oct),
            )
        )

    seen = {_hpf_candidate_signature(bool(current_enabled), float(current_freq_hz), int(current_slope_db_oct))}
    candidate_defs: list[dict] = []
    for cand_enabled, cand_freq_hz, cand_slope_db_oct in candidate_states:
        sig = _hpf_candidate_signature(cand_enabled, cand_freq_hz, cand_slope_db_oct)
        if sig in seen:
            continue
        seen.add(sig)
        candidate_defs.append(
            {
                "enabled": bool(cand_enabled),
                "freq_hz": float(cand_freq_hz),
                "slope_db_oct": int(cand_slope_db_oct),
                "label": _format_hpf_candidate(cand_enabled, cand_freq_hz, cand_slope_db_oct),
            }
        )
    return list(candidate_defs), float(freq_base_hz), int(slope_base_db_oct)


def _hpf_update_accept_meta(meta: dict, *, reason: str, accept_meta: dict, better: bool) -> None:
    meta.setdefault("reject_reasons", {})
    if not bool(better):
        reject_reasons = dict(meta.get("reject_reasons", {}) or {})
        reject_reasons[str(reason)] = int(reject_reasons.get(str(reason), 0) or 0) + 1
        meta["reject_reasons"] = reject_reasons
    meta["last_candidate_delta"] = dict(accept_meta.get("delta", {}) or {})
    meta["last_candidate_hard_gate"] = {
        "hard_gate_failed": bool(accept_meta.get("hard_gate_failed", False)),
        "hard_gate_reasons": list(accept_meta.get("hard_gate_reasons", []) or []),
        "residual_peak_db": float(_auto_safe_float(accept_meta.get("residual_peak_db"), float("nan"))),
        "residual_peak_hard_gate_db": float(_auto_safe_float(accept_meta.get("residual_peak_hard_gate_db"), float("nan"))),
    }


def _hpf_materialize_candidate(
    *,
    candidate: dict,
    cur_best_preset: dict,
    materialize_preset_result,
    base_data_ref: dict | None,
) -> dict | None:
    cand_test = _build_hpf_candidate_preset(
        dict(cur_best_preset or {}),
        enabled_state=bool(candidate["enabled"]),
        freq_hz=float(candidate["freq_hz"]),
        slope_db_oct=int(candidate["slope_db_oct"]),
    )
    try:
        _result, cand_metrics, _data = materialize_preset_result(
            cand_test,
            include_response_arrays=False,
            summarize=False,
            base_data_override=base_data_ref,
        )
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
        return None
    return {
        "cand_test": dict(cand_test or {}),
        "cand_metrics": dict(cand_metrics or {}),
    }


def _hpf_finalize_state(
    *,
    cur_best_preset: dict,
    cur_best_metrics: dict,
    current_enabled: bool,
    seed_freq_hz: float,
    seed_slope_db_oct: int,
    allowed_slopes: list[int],
    meta: dict,
    improved: bool,
) -> tuple[dict, dict, bool, dict]:
    final_override = dict(cur_best_preset.get("_auto_hpf_runtime_override", {}) or {})
    final_enabled = bool(final_override.get("enabled", cur_best_preset.get("hpf_enable", current_enabled)))
    final_freq_hz = _hpf_coerce_freq(
        final_override.get("freq", cur_best_preset.get("hpf_freq", seed_freq_hz)),
        default=seed_freq_hz,
    )
    final_slope_db_oct = _hpf_nearest_allowed_slope(
        cur_best_preset.get(
            "hpf_slope",
            (float(final_override.get("order", 0) or 0) * 6.0) if final_override else seed_slope_db_oct,
        ),
        default=seed_slope_db_oct,
        allowed_slopes=allowed_slopes,
    )
    if final_override:
        cur_best_preset = dict(cur_best_preset)
        cur_best_preset["hpf_enable"] = bool(final_enabled)
        cur_best_preset["hpf_freq"] = float(final_freq_hz)
        cur_best_preset["hpf_slope"] = int(final_slope_db_oct)
    meta["applied"] = bool(improved)
    meta["final_enabled"] = bool(final_enabled)
    meta["final_freq_hz"] = float(final_freq_hz)
    meta["final_slope_db_oct"] = int(final_slope_db_oct)
    meta["rank_after"] = float(_auto_safe_float(cur_best_metrics.get("rank_score"), float("nan")))
    meta["avg_after"] = float(_auto_safe_float(cur_best_metrics.get("avg_score"), float("nan")))
    return cur_best_preset, cur_best_metrics, bool(improved), dict(meta or {})


def _format_hpf_candidate(enabled: bool, freq_hz: float, slope_db_oct: int) -> str:
    if not bool(enabled):
        return "HPF off"
    return f"HPF {float(freq_hz):.1f} Hz/{int(slope_db_oct)} dB/oct"


def apply_hpf_winner_polish(
    *,
    best_preset: dict | None,
    best_metrics: dict | None,
    base_data_ref: dict | None,
    phase_label: str,
    goal: str,
    enabled: bool,
    status_cb,
    materialize_preset_result,
    cache_ready_preset,
    auto_is_better_refine,
) -> tuple[dict, dict, bool, dict]:
    cur_best_preset = dict(best_preset or {})
    cur_best_metrics = dict(best_metrics or {})
    meta = _hpf_init_meta(enabled=bool(enabled), phase_label=str(phase_label))
    if not bool(enabled):
        return cur_best_preset, cur_best_metrics, False, meta
    if not isinstance(cur_best_metrics, dict) or not cur_best_metrics:
        return cur_best_preset, cur_best_metrics, False, meta

    base_data = dict(base_data_ref or {})
    if _hpf_is_direct_dac_mode(base_data):
        meta["applicable"] = False
        return cur_best_preset, cur_best_metrics, False, meta

    allowed_slopes = _hpf_allowed_slopes()
    seed_freq_hz, seed_slope_db_oct = _resolve_hpf_seed_state(base_data=base_data, allowed_slopes=allowed_slopes)
    meta["seed_freq_hz"] = float(seed_freq_hz)
    meta["seed_slope_db_oct"] = int(seed_slope_db_oct)

    current_enabled, current_freq_hz, current_slope_db_oct = _resolve_hpf_current_state(
        preset=cur_best_preset,
        seed_freq_hz=seed_freq_hz,
        seed_slope_db_oct=seed_slope_db_oct,
        allowed_slopes=allowed_slopes,
    )
    meta["start_enabled"] = bool(current_enabled)
    meta["start_freq_hz"] = float(current_freq_hz)
    meta["start_slope_db_oct"] = int(current_slope_db_oct)
    meta["rank_before"] = float(_auto_safe_float(cur_best_metrics.get("rank_score"), float("nan")))
    meta["avg_before"] = float(_auto_safe_float(cur_best_metrics.get("avg_score"), float("nan")))

    candidate_defs, freq_base_hz, slope_base_db_oct = _build_hpf_candidate_defs(
        current_enabled=current_enabled,
        current_freq_hz=current_freq_hz,
        current_slope_db_oct=current_slope_db_oct,
        seed_freq_hz=seed_freq_hz,
        seed_slope_db_oct=seed_slope_db_oct,
        allowed_slopes=allowed_slopes,
    )
    meta["tested_candidates"] = [dict(item) for item in candidate_defs]
    meta["tested_count"] = int(len(candidate_defs))
    if not candidate_defs:
        return _hpf_finalize_state(
            cur_best_preset=cur_best_preset,
            cur_best_metrics=cur_best_metrics,
            current_enabled=current_enabled,
            seed_freq_hz=seed_freq_hz,
            seed_slope_db_oct=seed_slope_db_oct,
            allowed_slopes=allowed_slopes,
            meta=meta,
            improved=False,
        )

    improved = False
    with profiled_section("winner_polish.hpf"):
        logger.info(
            "Automatic mode %s: testing %d HPF winner-polish candidate(s) around %s.",
            str(phase_label),
            int(len(candidate_defs)),
            _format_hpf_candidate(True, freq_base_hz, slope_base_db_oct),
        )
        current_label = _format_hpf_candidate(current_enabled, current_freq_hz, current_slope_db_oct)
        for idx, candidate in enumerate(candidate_defs, start=1):
            candidate_payload = _hpf_materialize_candidate(
                candidate=dict(candidate or {}),
                cur_best_preset=cur_best_preset,
                materialize_preset_result=materialize_preset_result,
                base_data_ref=base_data_ref,
            )
            if not isinstance(candidate_payload, dict) or not candidate_payload:
                logger.warning(
                    "Automatic mode %s failed for candidate %d/%d (%s): %s",
                    str(phase_label),
                    int(idx),
                    int(len(candidate_defs)),
                    str(candidate["label"]),
                    "materialize_failed",
                )
                continue

            cand_test = dict(candidate_payload.get("cand_test", {}) or {})
            cand_metrics = dict(candidate_payload.get("cand_metrics", {}) or {})
            better, reason, accept_meta = _winner_polish_acceptance(
                candidate_metrics=cand_metrics,
                current_metrics=cur_best_metrics,
                goal=goal,
                auto_is_better_refine=auto_is_better_refine,
            )
            _hpf_update_accept_meta(
                meta,
                reason=str(reason),
                accept_meta=dict(accept_meta or {}),
                better=bool(better),
            )
            logger.info(
                "Automatic mode %s candidate %d/%d: %s, rank=%.3f, decision=%s (%s)",
                str(phase_label),
                int(idx),
                int(len(candidate_defs)),
                str(candidate["label"]),
                _auto_safe_float(cand_metrics.get("rank_score"), 0.0),
                "accept" if bool(better) else "reject",
                str(reason),
            )
            if not bool(better):
                continue

            prev_best = dict(cur_best_metrics or {})
            cur_best_metrics = dict(cand_metrics or {})
            cur_best_preset = cache_ready_preset(cand_test, best_metrics=cur_best_metrics)
            improved = True
            accepted = list(meta.get("accepted_candidates", []) or [])
            accepted.append(dict(candidate))
            meta["accepted_candidates"] = accepted
            logger.info(
                "Automatic mode %s accepted candidate %d/%d: %s -> %s, rank %.3f -> %.3f, avg %.3f -> %.3f",
                str(phase_label),
                int(idx),
                int(len(candidate_defs)),
                str(current_label),
                str(candidate["label"]),
                _auto_safe_float(prev_best.get("rank_score"), 0.0),
                _auto_safe_float(cur_best_metrics.get("rank_score"), 0.0),
                _auto_safe_float(prev_best.get("avg_score"), 0.0),
                _auto_safe_float(cur_best_metrics.get("avg_score"), 0.0),
            )
            current_label = str(candidate["label"])
            if callable(status_cb):
                status_cb(
                    "DecayCore automatic mode: hpf winner polish improved "
                    f"({candidate['label']!s}, "
                    f"rank {_polish_rank_status(cur_best_metrics)}, "
                    f"avg {_auto_safe_float(cur_best_metrics.get('avg_score'), 0.0):.3f})"
                )

    return _hpf_finalize_state(
        cur_best_preset=cur_best_preset,
        cur_best_metrics=cur_best_metrics,
        current_enabled=current_enabled,
        seed_freq_hz=seed_freq_hz,
        seed_slope_db_oct=seed_slope_db_oct,
        allowed_slopes=allowed_slopes,
        meta=meta,
        improved=improved,
    )
