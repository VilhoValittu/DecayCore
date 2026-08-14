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

from importlib import import_module
from typing import Any

__all__ = [
    "_build_bass_integration_metadata_unified",
    "_compute_direct_dac_prepare_recommendation",
    "_compute_selected_bass_integration_diagnostics",
    "_direct_dac_alignment_params",
    "_direct_dac_filter_params",
    "_extract_generated_source_rt60",
    "_extract_generated_source_snr",
    "_get_wav_window_params",
    "_load_generated_measurement_pair",
    "_prepare_target_curve_and_run_context",
    "_prepare_target_curve_bass_integration_context",
    "_prepare_ui_and_measurements",
    "_safe_float_from_dict",
    "_status",
    "_try_load_harmonic_sidecar",
    "_try_load_rt60_sidecar",
    "build_xos_hpf",
    "choose_dash_fs",
    "choose_target_rates",
    "compute_health",
    "detect_is_wav_source",
    "filter_type_short",
    "load_bass_integration_measurements",
    "load_house_curve",
    "load_measurements_lr",
    "load_raw_ir_sub",
    "load_raw_irs_lr",
    "log_df_smoothing_toggle",
    "save_config",
]

_BASS_DIAGNOSTIC_EXPORTS = {
    "_status",
    "_direct_dac_filter_params",
    "_direct_dac_alignment_params",
    "_compute_direct_dac_prepare_recommendation",
    "_compute_selected_bass_integration_diagnostics",
}
_MEASUREMENT_EXPORTS = {
    "_get_wav_window_params",
    "_extract_generated_source_rt60",
    "_extract_generated_source_snr",
    "_load_generated_measurement_pair",
    "_prepare_ui_and_measurements",
    "_try_load_harmonic_sidecar",
    "_try_load_rt60_sidecar",
    "compute_health",
    "filter_type_short",
    "load_bass_integration_measurements",
    "load_measurements_lr",
    "load_raw_ir_sub",
    "load_raw_irs_lr",
    "save_config",
}
_TARGET_CONTEXT_EXPORTS = set(__all__) - _BASS_DIAGNOSTIC_EXPORTS - _MEASUREMENT_EXPORTS


def __getattr__(name: str) -> Any:
    if name in _BASS_DIAGNOSTIC_EXPORTS:
        module_name = "bass_diagnostics"
    elif name in _MEASUREMENT_EXPORTS:
        module_name = "measurements"
    elif name in _TARGET_CONTEXT_EXPORTS:
        module_name = "target_context"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
