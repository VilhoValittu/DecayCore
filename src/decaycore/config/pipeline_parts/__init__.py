# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

from .filter_config import build_filter_config
from .managed_settings import (
    _finite_float_or_default,
    _advanced_manual_output_tilt_enabled,
    _effective_output_tilt_source,
    _resolve_output_tilt_db_per_oct,
    _apply_auto_mode_managed_settings,
    get_auto_mode_filter_auto_defaults,
)
from .ui_data import (
    collect_ui_config,
    collect_ui_data,
    log_df_smoothing_toggle,
)
from .xo_hpf import (
    _apply_auto_hpf_runtime_override,
    build_xos_hpf,
    filter_type_short,
    filter_type_supports_xo_phase_model,
    multi_rate_target_rates,
    choose_target_rates,
    choose_dash_fs,
    detect_is_wav_source,
)

__all__ = [
    '_advanced_manual_output_tilt_enabled',
    '_apply_auto_hpf_runtime_override',
    '_apply_auto_mode_managed_settings',
    '_effective_output_tilt_source',
    '_finite_float_or_default',
    '_resolve_output_tilt_db_per_oct',
    'build_filter_config',
    'build_xos_hpf',
    'choose_dash_fs',
    'choose_target_rates',
    'collect_ui_config',
    'collect_ui_data',
    'detect_is_wav_source',
    'filter_type_short',
    'filter_type_supports_xo_phase_model',
    'get_auto_mode_filter_auto_defaults',
    'log_df_smoothing_toggle',
    'multi_rate_target_rates',
]
