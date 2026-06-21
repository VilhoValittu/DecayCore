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

from .search.ranking_gates import (
    _auto_candidate_metrics as _auto_candidate_metrics,
    _auto_hard_gate_diagnostic as _auto_hard_gate_diagnostic,
    _auto_hard_gate_reasons as _auto_hard_gate_reasons,
    _auto_pick_metric as _auto_pick_metric,
    _auto_rank_value as _auto_rank_value,
    filter_hard_failed_candidates as filter_hard_failed_candidates,
    get_residual_peak_db as get_residual_peak_db,
    get_residual_peak_hard_gate_db as get_residual_peak_hard_gate_db,
    is_hard_failed as is_hard_failed,
    maybe_override_hard_failed_winner as maybe_override_hard_failed_winner,
    select_best_safe_candidate as select_best_safe_candidate,
)
from .search.ranking_keys import (
    _auto_apply_goal_tiebreak_metrics as _auto_apply_goal_tiebreak_metrics,
    _auto_bass_boost_for_rank as _auto_bass_boost_for_rank,
    _auto_hybrid_mixed_freq_penalty as _auto_hybrid_mixed_freq_penalty,
    _auto_mode_ripple_for_pareto as _auto_mode_ripple_for_pareto,
    _auto_phase_benefit_for_rank as _auto_phase_benefit_for_rank,
    _auto_phase_net_for_rank as _auto_phase_net_for_rank,
    _auto_phase_risk_for_rank as _auto_phase_risk_for_rank,
    _auto_prefer_bass_adjusted_avg_for_rank as _auto_prefer_bass_adjusted_avg_for_rank,
    _auto_rank_key as _auto_rank_key,
    _auto_rank_key_acoustic as _auto_rank_key_acoustic,
    _auto_rank_key_flat as _auto_rank_key_flat,
    _auto_rank_key_hybrid as _auto_rank_key_hybrid,
    _auto_rank_key_low_ripple as _auto_rank_key_low_ripple,
    _auto_rank_key_prefer_bass as _auto_rank_key_prefer_bass,
    _auto_rank_key_room_safe as _auto_rank_key_room_safe,
    _auto_realized_rms_20_200_for_pareto as _auto_realized_rms_20_200_for_pareto,
    _auto_target_tracking_for_pareto as _auto_target_tracking_for_pareto,
)
from .search import ranking_phase2 as _ranking_phase2
from .search.ranking_phase2 import (
    _auto_gate_threshold as _auto_gate_threshold,
    _auto_peak_metric_for_gate as _auto_peak_metric_for_gate,
    _auto_phase2_hard_gate_pool as _auto_phase2_hard_gate_pool,
    _auto_phase2_pareto_vector as _auto_phase2_pareto_vector,
    _auto_phase2_pick_pareto_winner as _auto_phase2_pick_pareto_winner,
    _auto_prepost_for_pareto as _auto_prepost_for_pareto,
    _auto_prepost_lr_for_pareto as _auto_prepost_lr_for_pareto,
    _auto_ripple_metric_for_gate as _auto_ripple_metric_for_gate,
    _pareto_dominates as _pareto_dominates,
)
from .search.ranking_refine import (
    _auto_adaptive_shrink_factor as _auto_adaptive_shrink_factor,
    _auto_build_refine_profile as _auto_build_refine_profile,
    _auto_goal_uses_local_refine as _auto_goal_uses_local_refine,
    _auto_is_better_refine as _auto_is_better_refine,
    _auto_rank_key_goal as _auto_rank_key_goal,
)
from .search.ranking_selection import (
    _auto_reject as _auto_reject,
    _auto_select_best_scored as _auto_select_best_scored,
)
from .search.ranking_target import (
    _auto_target_bass_forward_candidates as _auto_target_bass_forward_candidates,
    _auto_target_ladder_info as _auto_target_ladder_info,
    _auto_target_mildness_index as _auto_target_mildness_index,
    _auto_target_result_mode_ripple as _auto_target_result_mode_ripple,
    _auto_target_result_rank_key as _auto_target_result_rank_key,
    _auto_target_result_tie_key as _auto_target_result_tie_key,
    _auto_target_strength_tie_value as _auto_target_strength_tie_value,
    _tc_score as _tc_score,
)
from .search.winner_explanation import _auto_build_winner_explanation as _auto_build_winner_explanation


def _auto_phase2_pareto_front(pool: list[dict]) -> list[dict]:
    _ranking_phase2._auto_phase2_pareto_vector = globals()["_auto_phase2_pareto_vector"]
    return _ranking_phase2._auto_phase2_pareto_front(pool)


__all__ = [name for name in globals() if name.startswith("_auto_") or name.startswith("get_") or name in {
    "filter_hard_failed_candidates", "is_hard_failed", "maybe_override_hard_failed_winner",
    "select_best_safe_candidate", "_pareto_dominates", "_tc_score",
}]
