# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Low-bass-cut winner-polish for auto-mode finalize."""

from __future__ import annotations

import json
import logging

import numpy as np

from .auto_mode_profile import profiled_section
from .rank_score import official_rank_score
from .shared import AUTO_MODE_LOW_BASS_MAX_HZ, AUTO_MODE_LOW_BASS_MIN_HZ, _auto_safe_float
from .winner_polish_utils import _winner_polish_acceptance

logger = logging.getLogger("DecayCore")


def apply_low_bass_cut_winner_polish(
    *,
    best_preset: dict | None,
    best_metrics: dict | None,
    base_data_ref: dict | None,
    phase_label: str,
    goal: str,
    enabled: bool,
    step_hz,
    max_delta_hz,
    status_cb,
    materialize_preset_result,
    cache_ready_preset,
    auto_is_better_refine,
    candidate_items: list[dict] | None = None,
) -> tuple[dict, dict, bool, dict]:
    cur_best_preset = dict(best_preset or {})
    cur_best_metrics = dict(best_metrics or {})
    low_bass_cut_meta = {
        "enabled": bool(enabled),
        "applicable": True,
        "phase_label": str(phase_label),
        "tested_low_bass_cut_hz": [],
        "accepted_low_bass_cut_hz": [],
        "tested_count": 0,
        "applied": False,
        "start_low_bass_cut_hz": float("nan"),
        "final_low_bass_cut_hz": float("nan"),
        "rank_before": float("nan"),
        "rank_after": float("nan"),
        "avg_before": float("nan"),
        "avg_after": float("nan"),
    }
    if not bool(enabled):
        return cur_best_preset, cur_best_metrics, False, low_bass_cut_meta
    if not isinstance(cur_best_metrics, dict) or not cur_best_metrics:
        return cur_best_preset, cur_best_metrics, False, low_bass_cut_meta

    base_data = dict(base_data_ref or {})
    if not bool(base_data.get("low_bass_cut_enable", True)):
        low_bass_cut_meta["applicable"] = False
        return cur_best_preset, cur_best_metrics, False, low_bass_cut_meta

    low_bass_cut_base = _auto_safe_float(
        cur_best_preset.get(
            "low_bass_cut_hz",
            base_data.get("low_bass_cut_hz", float("nan")),
        ),
        float("nan"),
    )
    if not np.isfinite(low_bass_cut_base):
        return cur_best_preset, cur_best_metrics, False, low_bass_cut_meta
    initial_low_bass_cut = round(
        float(
            np.clip(
                float(low_bass_cut_base),
                float(AUTO_MODE_LOW_BASS_MIN_HZ),
                float(AUTO_MODE_LOW_BASS_MAX_HZ),
            )
        ),
        1,
    )
    low_bass_cut_meta["start_low_bass_cut_hz"] = float(initial_low_bass_cut)
    low_bass_cut_meta["rank_before"] = float(
        _auto_safe_float(cur_best_metrics.get("rank_score"), float("nan"))
    )
    low_bass_cut_meta["avg_before"] = float(
        _auto_safe_float(cur_best_metrics.get("avg_score"), float("nan"))
    )

    step_hz_eff = max(0.1, float(_auto_safe_float(step_hz, 2.0)))
    max_delta_hz_eff = max(0.0, float(_auto_safe_float(max_delta_hz, 8.0)))
    min_low_bass_cut = float(AUTO_MODE_LOW_BASS_MIN_HZ)
    min_candidate_hz = max(
        float(min_low_bass_cut),
        float(initial_low_bass_cut - max_delta_hz_eff),
    )
    max_candidate_hz = min(
        float(AUTO_MODE_LOW_BASS_MAX_HZ),
        float(initial_low_bass_cut + max_delta_hz_eff),
    )
    candidate_low_bass_cuts: list[float] = []
    tested_low_bass_cuts: set[float] = {float(initial_low_bass_cut)}
    max_steps = int(max(0, np.floor((float(max_delta_hz_eff) / float(step_hz_eff)) + 1e-9)))
    for step_idx in range(1, int(max_steps) + 1):
        down_value = float(initial_low_bass_cut - float(step_idx) * float(step_hz_eff))
        if down_value >= (float(min_candidate_hz) - 1e-9):
            cand_low_bass_cut = round(
                float(
                    np.clip(
                        max(float(min_low_bass_cut), down_value),
                        float(AUTO_MODE_LOW_BASS_MIN_HZ),
                        float(AUTO_MODE_LOW_BASS_MAX_HZ),
                    )
                ),
                1,
            )
            if cand_low_bass_cut not in tested_low_bass_cuts:
                tested_low_bass_cuts.add(cand_low_bass_cut)
                candidate_low_bass_cuts.append(float(cand_low_bass_cut))

        up_value = float(initial_low_bass_cut + float(step_idx) * float(step_hz_eff))
        if up_value <= (float(max_candidate_hz) + 1e-9):
            cand_low_bass_cut = round(
                float(
                    np.clip(
                        max(float(min_low_bass_cut), up_value),
                        float(AUTO_MODE_LOW_BASS_MIN_HZ),
                        float(AUTO_MODE_LOW_BASS_MAX_HZ),
                    )
                ),
                1,
            )
            if cand_low_bass_cut not in tested_low_bass_cuts:
                tested_low_bass_cuts.add(cand_low_bass_cut)
                candidate_low_bass_cuts.append(float(cand_low_bass_cut))

    low_bass_cut_meta["tested_low_bass_cut_hz"] = [float(v) for v in candidate_low_bass_cuts]
    low_bass_cut_meta["tested_count"] = int(len(candidate_low_bass_cuts))
    if not candidate_low_bass_cuts:
        low_bass_cut_meta["final_low_bass_cut_hz"] = float(initial_low_bass_cut)
        low_bass_cut_meta["rank_after"] = float(
            _auto_safe_float(cur_best_metrics.get("rank_score"), float("nan"))
        )
        low_bass_cut_meta["avg_after"] = float(
            _auto_safe_float(cur_best_metrics.get("avg_score"), float("nan"))
        )
        return cur_best_preset, cur_best_metrics, False, low_bass_cut_meta

    def _preset_signature(preset: dict | None, *, metrics: dict | None = None) -> str:
        ready = cache_ready_preset(
            dict(preset or {}),
            best_metrics=dict(metrics or {}),
        )
        try:
            return str(
                json.dumps(
                    dict(ready or {}),
                    sort_keys=True,
                    separators=(",", ":"),
                )
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
            return str(sorted(dict(ready or {}).items()))

    candidate_metrics_lookup: dict[str, dict] = {}
    for item in list(candidate_items or []):
        if not isinstance(item, dict):
            continue
        item_preset = dict(item.get("preset", {}) or {})
        item_metrics = dict(item.get("metrics", {}) or {})
        if not item_preset or not item_metrics:
            continue
        sig = _preset_signature(item_preset, metrics=item_metrics)
        prev = dict(candidate_metrics_lookup.get(sig, {}) or {})
        prev_rank = _auto_safe_float(prev.get("metrics", {}).get("rank_score"), float("-inf"))
        next_rank = _auto_safe_float(item_metrics.get("rank_score"), float("-inf"))
        if (sig not in candidate_metrics_lookup) or (float(next_rank) > float(prev_rank)):
            candidate_metrics_lookup[sig] = {
                "preset": dict(item_preset or {}),
                "metrics": dict(item_metrics or {}),
                "source": str(item.get("phase", item.get("source", "search_candidate")) or "search_candidate"),
            }

    improved = False
    with profiled_section("winner_polish.low_bass_cut"):
        logger.info(
            "Automatic mode %s: testing %d low_bass_cut winner-polish candidate(s) around %.1f Hz.",
            str(phase_label),
            int(len(candidate_low_bass_cuts)),
            float(initial_low_bass_cut),
        )
        for idx, cand_low_bass_cut in enumerate(candidate_low_bass_cuts, start=1):
            cand_test = dict(cur_best_preset or {})
            cand_test["low_bass_cut_hz"] = float(cand_low_bass_cut)
            reuse_entry = candidate_metrics_lookup.get(
                _preset_signature(cand_test, metrics=cur_best_metrics)
            )
            if isinstance(reuse_entry, dict) and reuse_entry:
                low_bass_cut_metrics = dict(reuse_entry.get("metrics", {}) or {})
            else:
                try:
                    _result, low_bass_cut_metrics, _data = materialize_preset_result(
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
                        "Automatic mode %s failed for candidate %d/%d (low_bass_cut_hz=%.1f Hz): %s",
                        str(phase_label),
                        int(idx),
                        int(len(candidate_low_bass_cuts)),
                        float(cand_low_bass_cut),
                        f"{type(exc).__name__}: {exc}",
                    )
                    continue

            low_bass_cut_metrics = dict(low_bass_cut_metrics or {})
            better, reason, accept_meta = _winner_polish_acceptance(
                candidate_metrics=low_bass_cut_metrics,
                current_metrics=cur_best_metrics,
                goal=goal,
                auto_is_better_refine=auto_is_better_refine,
            )
            low_bass_cut_meta.setdefault("reject_reasons", {})
            if not bool(better):
                reject_reasons = dict(low_bass_cut_meta.get("reject_reasons", {}) or {})
                reject_reasons[str(reason)] = int(reject_reasons.get(str(reason), 0) or 0) + 1
                low_bass_cut_meta["reject_reasons"] = reject_reasons
            low_bass_cut_meta["last_candidate_delta"] = dict(accept_meta.get("delta", {}) or {})
            low_bass_cut_meta["last_candidate_hard_gate"] = {
                "hard_gate_failed": bool(accept_meta.get("hard_gate_failed", False)),
                "hard_gate_reasons": list(accept_meta.get("hard_gate_reasons", []) or []),
                "residual_peak_db": float(_auto_safe_float(accept_meta.get("residual_peak_db"), float("nan"))),
                "residual_peak_hard_gate_db": float(_auto_safe_float(accept_meta.get("residual_peak_hard_gate_db"), float("nan"))),
            }
            logger.info(
                "Automatic mode %s candidate %d/%d: low_bass_cut_hz=%.1f Hz, rank=%.3f, decision=%s (%s)%s",
                str(phase_label),
                int(idx),
                int(len(candidate_low_bass_cuts)),
                float(cand_low_bass_cut),
                _auto_safe_float(low_bass_cut_metrics.get("rank_score"), 0.0),
                "accept" if bool(better) else "reject",
                str(reason),
                (
                    ""
                    if not isinstance(reuse_entry, dict)
                    else f", reused={str(reuse_entry.get('source', 'search_candidate'))}"
                ),
            )
            if not bool(better):
                continue

            prev_best = dict(cur_best_metrics or {})
            prev_low_bass_cut = float(
                _auto_safe_float(cur_best_preset.get("low_bass_cut_hz", initial_low_bass_cut), initial_low_bass_cut)
            )
            cur_best_metrics = dict(low_bass_cut_metrics or {})
            cur_best_preset = cache_ready_preset(cand_test, best_metrics=cur_best_metrics)
            improved = True
            accepted = list(low_bass_cut_meta.get("accepted_low_bass_cut_hz", []) or [])
            accepted.append(float(cand_low_bass_cut))
            low_bass_cut_meta["accepted_low_bass_cut_hz"] = accepted
            logger.info(
                "Automatic mode %s accepted candidate %d/%d: low_bass_cut_hz %.1f -> %.1f Hz, rank %.3f -> %.3f, avg %.3f -> %.3f",
                str(phase_label),
                int(idx),
                int(len(candidate_low_bass_cuts)),
                float(prev_low_bass_cut),
                float(cand_low_bass_cut),
                _auto_safe_float(prev_best.get("rank_score"), 0.0),
                _auto_safe_float(cur_best_metrics.get("rank_score"), 0.0),
                _auto_safe_float(prev_best.get("avg_score"), 0.0),
                _auto_safe_float(cur_best_metrics.get("avg_score"), 0.0),
            )
            if callable(status_cb):
                status_cb(
                    "DecayCore automatic mode: low_bass_cut winner polish improved "
                    f"(low_bass_cut_hz {float(prev_low_bass_cut):.1f} -> {float(cand_low_bass_cut):.1f} Hz, "
                    f"rank {official_rank_score(prev_best):.3f} -> {official_rank_score(cur_best_metrics):.3f}, "
                    f"avg {_auto_safe_float(prev_best.get('avg_score'), 0.0):.3f} -> {_auto_safe_float(cur_best_metrics.get('avg_score'), 0.0):.3f})"
                )

    low_bass_cut_meta["applied"] = bool(improved)
    final_low_bass_cut = _auto_safe_float(
        cur_best_preset.get(
            "low_bass_cut_hz",
            base_data.get("low_bass_cut_hz", initial_low_bass_cut),
        ),
        initial_low_bass_cut,
    )
    low_bass_cut_meta["final_low_bass_cut_hz"] = float(
        round(
            float(
                np.clip(
                    float(final_low_bass_cut),
                    float(AUTO_MODE_LOW_BASS_MIN_HZ),
                    float(AUTO_MODE_LOW_BASS_MAX_HZ),
                )
            ),
            1,
        )
    )
    low_bass_cut_meta["rank_after"] = float(
        _auto_safe_float(cur_best_metrics.get("rank_score"), float("nan"))
    )
    low_bass_cut_meta["avg_after"] = float(
        _auto_safe_float(cur_best_metrics.get("avg_score"), float("nan"))
    )
    return cur_best_preset, cur_best_metrics, bool(improved), dict(low_bass_cut_meta or {})
