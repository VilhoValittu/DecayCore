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
    "_append_auto_polish_to_status_log",
    "_build_auto_polish_lines",
    "_build_auto_audit_markdown",
    "_build_p6_validation_block",
    "_esc",
    "_fmt_biquad",
    "_fmt_external_iir_hpf",
    "_download_filter_bundle",
    "_filter_download_filename",
    "_filter_download_payload",
    "_format_recommended_xo_hz",
    "_metric_table_html",
    "_rejected_reasons",
    "_render_auto_diagnostics",
    "_render_bass_integration",
    "_render_dsp_quality",
    "_render_hybrid_iir_cuts",
    "_render_ir_alignment",
    "_render_lr_difference",
    "_render_filter_download",
    "_render_plots_and_export",
    "clear_plot_render_cache",
    "_render_run_overview",
    "_section",
    "_update_crossover_recommendation_label",
    "render_results",
]

_BASS_EXPORTS = {"_render_bass_integration"}
_PLOT_EXPORTS = {"clear_plot_render_cache", "_render_plots_and_export"}
_SECTION_EXPORTS = {"_section"}
_QUALITY_EXPORTS = {
    "_render_ir_alignment",
    "_render_dsp_quality",
    "_render_lr_difference",
    "_render_hybrid_iir_cuts",
    "_fmt_biquad",
    "_fmt_external_iir_hpf",
    "_rejected_reasons",
}
_OVERVIEW_EXPORTS = set(__all__) - _BASS_EXPORTS - _PLOT_EXPORTS - _SECTION_EXPORTS - _QUALITY_EXPORTS


def __getattr__(name: str) -> Any:
    if name in _BASS_EXPORTS:
        module_name = "bass_integration"
    elif name in _PLOT_EXPORTS:
        module_name = "plots_export"
    elif name in _SECTION_EXPORTS:
        module_name = "section"
    elif name in _QUALITY_EXPORTS:
        module_name = "quality"
    elif name in _OVERVIEW_EXPORTS:
        module_name = "overview"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
