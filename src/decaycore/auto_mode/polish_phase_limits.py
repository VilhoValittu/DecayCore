# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Phase-limit winner-polish for auto-mode finalize."""

from __future__ import annotations

import logging

import numpy as np

from .auto_mode_profile import profiled_section
from .rank_score import official_rank_score
from .shared import _auto_phase_limit_clip, _auto_safe_float
from .winner_polish_utils import _winner_polish_acceptance

logger = logging.getLogger("DecayCore")


def apply_phase_limit_winner_polish(
    *,
    best_preset: dict | None,
    best_metrics: dict | None,
    base_data_ref: dict | None,
    phase_label: str,
    goal: str,
    filter_key: str,
    enabled: bool,
    offsets_hz,
    status_cb,
    materialize_preset_result,
    cache_ready_preset,
    auto_is_better_refine,
) -> tuple[dict, dict, bool, dict]:
    cur_best_preset = dict(best_preset or {})
    cur_best_metrics = dict(best_metrics or {})
    phase_limit_meta = {
        "enabled": bool(enabled),
        "applicable": bool(str(filter_key) in ("linear", "asym")),
        "phase_label": str(phase_label),
        "tested_phase_limits_hz": [],
        "accepted_phase_limits_hz": [],
        "tested_count": 0,
        "applied": False,
        "start_phase_limit_hz": float("nan"),
        "final_phase_limit_hz": float("nan"),
        "rank_before": float("nan"),
        "rank_after": float("nan"),
        "avg_before": float("nan"),
        "avg_after": float("nan"),
        "accepted_reason": "",
        "rejected_reason": "",
    }
    if not bool(enabled):
        return cur_best_preset, cur_best_metrics, False, phase_limit_meta
    if str(filter_key) not in ("linear", "asym"):
        return cur_best_preset, cur_best_metrics, False, phase_limit_meta
    if not isinstance(cur_best_metrics, dict) or not cur_best_metrics:
        return cur_best_preset, cur_best_metrics, False, phase_limit_meta

    phase_limit_base = _auto_safe_float(
        cur_best_preset.get(
            "phase_limit",
            dict(base_data_ref or {}).get("phase_limit", float("nan")),
        ),
        float("nan"),
    )
    if not np.isfinite(phase_limit_base):
        return cur_best_preset, cur_best_metrics, False, phase_limit_meta
    initial_phase_limit = round(
        float(
            _auto_phase_limit_clip(
                phase_limit_base,
                default=400.0,
            )
        ),
        1,
    )
    phase_limit_meta["start_phase_limit_hz"] = float(initial_phase_limit)
    phase_limit_meta["rank_before"] = float(
        _auto_safe_float(cur_best_metrics.get("rank_score"), float("nan"))
    )
    phase_limit_meta["avg_before"] = float(
        _auto_safe_float(cur_best_metrics.get("avg_score"), float("nan"))
    )

    tested_phase_limits: set[float] = {float(initial_phase_limit)}
    candidate_phase_limits: list[float] = []
    for delta_hz in offsets_hz:
        cand_phase_limit = round(
            float(
                _auto_phase_limit_clip(
                    float(phase_limit_base) + float(_auto_safe_float(delta_hz, 0.0)),
                    default=400.0,
                )
            ),
            1,
        )
        if cand_phase_limit in tested_phase_limits:
            continue
        tested_phase_limits.add(cand_phase_limit)
        candidate_phase_limits.append(float(cand_phase_limit))

    phase_limit_meta["tested_phase_limits_hz"] = [float(v) for v in candidate_phase_limits]
    phase_limit_meta["tested_count"] = int(len(candidate_phase_limits))
    if not candidate_phase_limits:
        phase_limit_meta["final_phase_limit_hz"] = float(initial_phase_limit)
        phase_limit_meta["rank_after"] = float(
            _auto_safe_float(cur_best_metrics.get("rank_score"), float("nan"))
        )
        phase_limit_meta["avg_after"] = float(
            _auto_safe_float(cur_best_metrics.get("avg_score"), float("nan"))
        )
        return cur_best_preset, cur_best_metrics, False, phase_limit_meta

    improved = False
    with profiled_section("winner_polish.phase_limit"):
        logger.info(
            "Automatic mode %s: testing %d phase_limit winner-polish candidate(s) around %.1f Hz.",
            str(phase_label),
            int(len(candidate_phase_limits)),
            float(initial_phase_limit),
        )
        _best_rank_gain = 0.0
        for idx, cand_phase_limit in enumerate(candidate_phase_limits, start=1):
            cand_test = dict(cur_best_preset or {})
            cand_test["phase_limit"] = float(cand_phase_limit)
            try:
                _phase_result, phase_metrics, _phase_data = materialize_preset_result(
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
            ) as exc:
                logger.warning(
                    "Automatic mode %s failed for candidate %d/%d (phase_limit=%.1f Hz): %s",
                    str(phase_label),
                    int(idx),
                    int(len(candidate_phase_limits)),
                    float(cand_phase_limit),
                    f"{type(exc).__name__}: {exc}",
                )
                continue

            phase_metrics = dict(phase_metrics or {})
            better, reason, accept_meta = _winner_polish_acceptance(
                candidate_metrics=phase_metrics,
                current_metrics=cur_best_metrics,
                goal=goal,
                auto_is_better_refine=auto_is_better_refine,
            )
            phase_limit_meta.setdefault("reject_reasons", {})
            if not bool(better):
                reject_reasons = dict(phase_limit_meta.get("reject_reasons", {}) or {})
                reject_reasons[str(reason)] = int(reject_reasons.get(str(reason), 0) or 0) + 1
                phase_limit_meta["reject_reasons"] = reject_reasons
            phase_limit_meta["last_candidate_delta"] = dict(accept_meta.get("delta", {}) or {})
            phase_limit_meta["last_candidate_hard_gate"] = {
                "hard_gate_failed": bool(accept_meta.get("hard_gate_failed", False)),
                "hard_gate_reasons": list(accept_meta.get("hard_gate_reasons", []) or []),
                "residual_peak_db": float(_auto_safe_float(accept_meta.get("residual_peak_db"), float("nan"))),
                "residual_peak_hard_gate_db": float(_auto_safe_float(accept_meta.get("residual_peak_hard_gate_db"), float("nan"))),
            }
            logger.info(
                "Automatic mode %s candidate %d/%d: phase_limit=%.1f Hz, rank=%.3f, decision=%s (%s)",
                str(phase_label),
                int(idx),
                int(len(candidate_phase_limits)),
                float(cand_phase_limit),
                _auto_safe_float(phase_metrics.get("rank_score"), 0.0),
                "accept" if bool(better) else "reject",
                str(reason),
            )
            if not bool(better):
                # Early exit: if we've seen at least 3 rejections after a peak gain,
                # and the gain has plateaued (current improvement < 50% of best seen),
                # stop searching.
                if improved and idx >= 3:
                    _rank_gain = float(
                        _auto_safe_float(phase_metrics.get("rank_score"), 0.0)
                        - _auto_safe_float(cur_best_metrics.get("rank_score"), 0.0)
                    )
                    if _best_rank_gain > 0.0 and abs(_rank_gain) < 0.5 * _best_rank_gain:
                        logger.info(
                            "Automatic mode %s: early exit at candidate %d/%d — rank gain plateaued.",
                            str(phase_label), int(idx), int(len(candidate_phase_limits)),
                        )
                        break
                continue

            prev_best = dict(cur_best_metrics or {})
            cur_best_metrics = dict(phase_metrics or {})
            cur_best_preset = cache_ready_preset(cand_test, best_metrics=cur_best_metrics)
            improved = True
            _rank_gain = float(
                _auto_safe_float(cur_best_metrics.get("rank_score"), 0.0)
                - _auto_safe_float(prev_best.get("rank_score"), 0.0)
            )
            if _rank_gain > _best_rank_gain:
                _best_rank_gain = _rank_gain
            accepted = list(phase_limit_meta.get("accepted_phase_limits_hz", []) or [])
            accepted.append(float(cand_phase_limit))
            phase_limit_meta["accepted_phase_limits_hz"] = accepted
            logger.info(
                "Automatic mode %s accepted candidate %d/%d: phase_limit %.1f -> %.1f Hz, rank %.3f -> %.3f, avg %.3f -> %.3f",
                str(phase_label),
                int(idx),
                int(len(candidate_phase_limits)),
                float(phase_limit_base),
                float(cand_phase_limit),
                _auto_safe_float(prev_best.get("rank_score"), 0.0),
                _auto_safe_float(cur_best_metrics.get("rank_score"), 0.0),
                _auto_safe_float(prev_best.get("avg_score"), 0.0),
                _auto_safe_float(cur_best_metrics.get("avg_score"), 0.0),
            )
            phase_limit_base = float(cand_phase_limit)
            if callable(status_cb):
                status_cb(
                    "DecayCore automatic mode: phase_limit winner polish improved "
                    f"(phase_limit {float(cand_phase_limit):.1f} Hz, "
                    f"rank {official_rank_score(cur_best_metrics):.3f}, "
                    f"avg {_auto_safe_float(cur_best_metrics.get('avg_score'), 0.0):.3f})"
                )

    phase_limit_meta["applied"] = bool(improved)
    phase_limit_meta["final_phase_limit_hz"] = float(
        round(
            float(
                _auto_phase_limit_clip(
                    cur_best_preset.get(
                        "phase_limit",
                        dict(base_data_ref or {}).get("phase_limit", 400.0),
                    ),
                    default=400.0,
                )
            ),
            1,
        )
    )
    phase_limit_meta["rank_after"] = float(
        _auto_safe_float(cur_best_metrics.get("rank_score"), float("nan"))
    )
    phase_limit_meta["avg_after"] = float(
        _auto_safe_float(cur_best_metrics.get("avg_score"), float("nan"))
    )
    return cur_best_preset, cur_best_metrics, bool(improved), dict(phase_limit_meta or {})
