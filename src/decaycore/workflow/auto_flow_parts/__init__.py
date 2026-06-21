# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

from .progress import (
    _progress_lerp,
    _auto_progress_fraction,
    _estimate_auto_progress_from_status,
    _set_auto_progress,
    _get_auto_status_callback,
    _AUTO_PROGRESS_INIT,
    _AUTO_PROGRESS_TARGET_MODE,
    _AUTO_PROGRESS_TARGET_PRESELECT,
    _AUTO_PROGRESS_TARGET_TRIALS_START,
    _AUTO_PROGRESS_TARGET_TRIALS_END,
    _AUTO_PROGRESS_PRESET_SEARCH_START,
    _AUTO_PROGRESS_PHASE1_START,
    _AUTO_PROGRESS_PHASE1_END,
    _AUTO_PROGRESS_PHASE2_START,
    _AUTO_PROGRESS_PHASE2_END,
    _AUTO_PROGRESS_PHASE3_START,
    _AUTO_PROGRESS_PHASE3_END,
    _AUTO_PROGRESS_FINALIZE,
)
from .search import (
    _run_auto_mode_search_if_needed,
    _run_auto_mode_search,
)
from .seed_phases import _run_auto_mode_seed_phases
from .status_text import (
    _build_auto_selected_text,
    _resolve_auto_hpf_seed_source,
    _auto_finalize_status_suffix,
    _build_auto_finalize_status,
)

__all__ = [
    '_AUTO_PROGRESS_FINALIZE',
    '_AUTO_PROGRESS_INIT',
    '_AUTO_PROGRESS_PHASE1_END',
    '_AUTO_PROGRESS_PHASE1_START',
    '_AUTO_PROGRESS_PHASE2_END',
    '_AUTO_PROGRESS_PHASE2_START',
    '_AUTO_PROGRESS_PHASE3_END',
    '_AUTO_PROGRESS_PHASE3_START',
    '_AUTO_PROGRESS_PRESET_SEARCH_START',
    '_AUTO_PROGRESS_TARGET_MODE',
    '_AUTO_PROGRESS_TARGET_PRESELECT',
    '_AUTO_PROGRESS_TARGET_TRIALS_END',
    '_AUTO_PROGRESS_TARGET_TRIALS_START',
    '_auto_finalize_status_suffix',
    '_auto_progress_fraction',
    '_build_auto_finalize_status',
    '_build_auto_selected_text',
    '_estimate_auto_progress_from_status',
    '_get_auto_status_callback',
    '_progress_lerp',
    '_resolve_auto_hpf_seed_source',
    '_run_auto_mode_search',
    '_run_auto_mode_search_if_needed',
    '_run_auto_mode_seed_phases',
    '_set_auto_progress',
]
