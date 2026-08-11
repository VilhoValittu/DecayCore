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

from dataclasses import dataclass, field
from typing import Any, Callable

from ..config.models import FilterConfig


@dataclass
class RunRequest:
    """Immutable-by-convention snapshot of raw UI form data for one run."""

    raw_ui_data: dict[str, Any] = field(default_factory=dict)
    run_started_at: float | None = None


@dataclass
class PreparedRunInput:
    source_ui_data: dict[str, Any]
    resolved_data: dict[str, Any]
    f_l: Any = None
    m_l: Any = None
    p_l: Any = None
    f_r: Any = None
    m_r: Any = None
    p_r: Any = None
    bass_integration_bundle: Any = None
    raw_ir_l: Any = None
    raw_ir_fs_l: int = 0
    raw_ir_r: Any = None
    raw_ir_fs_r: int = 0
    raw_ir_sub: Any = None
    raw_ir_fs_sub: int = 0
    measured_rt60_l: Any = None
    measured_rt60_bands_l: Any = None
    measured_rt60_r: Any = None
    measured_rt60_bands_r: Any = None
    harmonic_freq_hz_l: Any = None
    harmonic_magnitudes_db_l: Any = None
    harmonic_freq_hz_r: Any = None
    harmonic_magnitudes_db_r: Any = None
    measured_snr_db_l: float | None = None
    measured_snr_db_r: float | None = None


@dataclass
class ResolvedRunConfig:
    source_ui_data: dict[str, Any]
    resolved_data: dict[str, Any]
    measurements: dict[str, Any] = field(default_factory=dict)
    hc_f: Any = None
    hc_m: Any = None
    xos: list[dict[str, Any]] = field(default_factory=list)
    hpf: dict[str, Any] | None = None
    target_rates: list[int] = field(default_factory=list)
    dash_fs: int = 0
    target_curve_tag: str = ""
    auto_applied_preset: dict[str, Any] = field(default_factory=dict)
    auto_result_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunContext:
    """Typed mutable state carried through one workflow run."""

    prepared_input: PreparedRunInput
    run_started_at: float
    taps_base: int
    perf_stats: dict[str, float]
    per_fs_stats: dict[int, dict[str, float]]
    resolved_config: ResolvedRunConfig | None = None
    auto_mode_preview: bool = False
    auto_mode_enabled: bool = False
    auto_goal: str = "balanced"
    auto_basis: str = "preset_objective_score"
    auto_applied_preset: dict[str, Any] = field(default_factory=dict)
    auto_progress_value: float = 0.0
    auto_status_callback: Callable[[str], None] | None = None
    ts: str = ""
    file_ts: str = ""
    ft_short: str = ""
    irw_tag: str = ""
    results_by_fs: list[Any] = field(default_factory=list)
    l_st_f: Any = None
    r_st_f: Any = None
    l_imp_f: Any = None
    r_imp_f: Any = None
    sub_ir_f: Any = None
    sub_st_f: Any = None
    sub_meas_f: dict[str, Any] = field(default_factory=dict)
    export_filename: str | None = None
    saved_filters_dir: Any = None

    @property
    def source_ui_data(self) -> dict[str, Any]:
        return self.prepared_input.source_ui_data

    @property
    def data(self) -> dict[str, Any]:
        if self.resolved_config is not None:
            return self.resolved_config.resolved_data
        return self.prepared_input.resolved_data

    @property
    def measurements(self) -> dict[str, Any]:
        return self.require_resolved_config().measurements

    def require_resolved_config(self) -> ResolvedRunConfig:
        if self.resolved_config is None:
            raise RuntimeError("RunContext has not reached the resolved-config stage")
        return self.resolved_config


@dataclass
class DSPRunInput:
    config: FilterConfig
    measurements: dict[str, Any]
    fs_v: int
    taps_v: int


def copy_source_ui_data(raw: dict[str, Any] | None) -> dict[str, Any]:
    return dict(raw or {})


def copy_resolved_data(data: dict[str, Any] | None) -> dict[str, Any]:
    return dict(data or {})


def build_dsp_measurements(
    resolved_data: dict[str, Any],
    prepared_measurements: dict[str, Any],
) -> dict[str, Any]:
    measurements = dict(prepared_measurements or {})
    measurements["ui_data"] = resolved_data
    return measurements


def apply_auto_mode_result(
    resolved: ResolvedRunConfig,
    auto_res: dict[str, Any],
    *,
    version: str,
    auto_mode_compat_version: str,
) -> ResolvedRunConfig:
    best_preset = dict(auto_res.get("best_preset", {}) or {})
    best_applied_preset = dict(auto_res.get("best_applied_preset", best_preset) or {})
    next_data = copy_resolved_data(resolved.resolved_data)
    if best_applied_preset:
        next_data.update(best_applied_preset)
        next_data["program_version"] = str(version)
        next_data["auto_mode_compat_version"] = str(auto_mode_compat_version)
        search_v2_debug = dict(
            dict(auto_res.get("auto_mode_debug", {}) or {}).get("auto_search_v2", {}) or {}
        )
        identity_base_data = search_v2_debug.get("identity_base_data")
        if isinstance(identity_base_data, dict) and identity_base_data:
            next_data["_auto_search_identity_base_data"] = dict(identity_base_data)
    next_measurements = build_dsp_measurements(next_data, resolved.measurements)
    return ResolvedRunConfig(
        source_ui_data=copy_source_ui_data(resolved.source_ui_data),
        resolved_data=next_data,
        measurements=next_measurements,
        hc_f=resolved.hc_f,
        hc_m=resolved.hc_m,
        xos=list(resolved.xos or []),
        hpf=dict(resolved.hpf or {}) if isinstance(resolved.hpf, dict) else resolved.hpf,
        target_rates=list(resolved.target_rates or []),
        dash_fs=int(resolved.dash_fs or 0),
        target_curve_tag=str(resolved.target_curve_tag or ""),
        auto_applied_preset=best_applied_preset,
        auto_result_meta=dict(auto_res or {}),
    )


__all__ = [
    "DSPRunInput",
    "PreparedRunInput",
    "ResolvedRunConfig",
    "RunContext",
    "RunRequest",
    "apply_auto_mode_result",
    "build_dsp_measurements",
    "copy_resolved_data",
    "copy_source_ui_data",
]
