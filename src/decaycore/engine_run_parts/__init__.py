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
    'SUB_TARGET_LPF_HZ',
    'SUB_TARGET_LPF_MAX_ATTENUATION_DB',
    'SUB_TARGET_LPF_SLOPE_DB_PER_OCT',
    'SUB_TARGET_POLICY',
    'SubwooferTarget',
    '_apply_measured_rt60_override',
    '_build_measurement_side_ctx',
    '_call_generate_filter',
    '_call_generate_filter_pair',
    '_extract_lr_measurement_axes',
    '_inject_direct_dac_summed_prediction_for_plot',
    '_interp_complex_to_axis',
    '_ir_fft_on_axis',
    '_phase_from_ir',
    '_resample_to_axis',
    '_resolve_sub_measurement_for_filter',
    '_stats_level_comp_factor',
    '_to_axis',
    'apply_direct_dac_bass_integration_result',
    'build_subwoofer_target_with_lpf',
    'dsp',
    'run_pipeline',
    'subwoofer_target_metadata',
]

_DIRECT_DAC_EXPORTS = {"apply_direct_dac_bass_integration_result"}
_MEASUREMENT_EXPORTS = {
    "_build_measurement_side_ctx",
    "_phase_from_ir",
    "_to_axis",
    "_extract_lr_measurement_axes",
    "_resample_to_axis",
    "_resolve_sub_measurement_for_filter",
    "_interp_complex_to_axis",
    "_ir_fft_on_axis",
}
_PIPELINE_EXPORTS = {
    "_call_generate_filter",
    "_call_generate_filter_pair",
    "_stats_level_comp_factor",
    "_apply_measured_rt60_override",
    "_inject_direct_dac_summed_prediction_for_plot",
    "dsp",
    "run_pipeline",
}
_SUBWOOFER_EXPORTS = set(__all__) - _DIRECT_DAC_EXPORTS - _MEASUREMENT_EXPORTS - _PIPELINE_EXPORTS


def __getattr__(name: str) -> Any:
    if name in _DIRECT_DAC_EXPORTS:
        module_name = "direct_dac_bass_integration"
    elif name in _MEASUREMENT_EXPORTS:
        module_name = "measurement_response_helpers"
    elif name in _PIPELINE_EXPORTS:
        module_name = "pipeline_execution"
    elif name in _SUBWOOFER_EXPORTS:
        module_name = "subwoofer_target"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
