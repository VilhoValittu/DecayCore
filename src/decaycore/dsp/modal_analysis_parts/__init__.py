# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

from .modal_preparation import (
    RoomModeEvent,
    ModalAnalysisResult,
    _empty_result,
    _as_float_array,
    _smooth_log_box,
    _safe_confidence,
    _prepare_arrays,
    _width_bounds,
)
from .mode_detection import (
    _modal_candidate_geometry,
    _voice_weight,
    modal_support_for_band,
    _lr_consistency_at,
    _decay_severity_at,
    _classify_event,
    detect_room_modes,
)

__all__ = [
    'ModalAnalysisResult',
    'RoomModeEvent',
    '_as_float_array',
    '_classify_event',
    '_decay_severity_at',
    '_empty_result',
    '_lr_consistency_at',
    '_modal_candidate_geometry',
    '_prepare_arrays',
    '_safe_confidence',
    '_smooth_log_box',
    '_voice_weight',
    '_width_bounds',
    'detect_room_modes',
    'modal_support_for_band',
]
