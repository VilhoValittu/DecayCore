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

import logging
import threading
from contextlib import contextmanager
from typing import Any

import numpy as np

from .correction_types import BaselineComparisonTelemetry, BaselineNativeTelemetry
from .phase_ir_types import ResidualTelemetry

_DSP_LOGGER_NAME = "DecayCore.dsp"
_QUIET_DSP_LOG_STATE = threading.local()
_QUIET_FILTER_LOCK = threading.Lock()
_QUIET_FILTER_INSTALLED = False


class _QuietInfoLogFilter(logging.Filter):
    """Pudottaa INFO/DEBUG-rivit saikeissa joissa quiet_dsp_info_logging on paalla.

    WARNING ja sita vakavammat menevat aina lapi.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return int(getattr(_QUIET_DSP_LOG_STATE, "depth", 0) or 0) <= 0


def _ensure_quiet_filter_installed() -> None:
    global _QUIET_FILTER_INSTALLED
    if _QUIET_FILTER_INSTALLED:
        return
    with _QUIET_FILTER_LOCK:
        if not _QUIET_FILTER_INSTALLED:
            logging.getLogger(_DSP_LOGGER_NAME).addFilter(_QuietInfoLogFilter())
            _QUIET_FILTER_INSTALLED = True


@contextmanager
def quiet_dsp_info_logging(*, enabled: bool = True):
    """Vaimentaa DecayCore.dsp-loggerin INFO/DEBUG-rivit nykyisessa saikeessa.

    Tarkoitettu auto-moden score-only-trialeille: per-trial-stagelokitus on
    merkittava osa mag_post-ajasta eika trialien run.log-rivit paady mihinkaan
    raporttiin. Saiekohtainen (thread-local), joten rinnakkainen voittajan
    materialisointi lokittaa normaalisti. WARNING+ menee aina lapi.
    """
    if not enabled:
        yield
        return
    _ensure_quiet_filter_installed()
    _QUIET_DSP_LOG_STATE.depth = int(getattr(_QUIET_DSP_LOG_STATE, "depth", 0) or 0) + 1
    try:
        yield
    finally:
        _QUIET_DSP_LOG_STATE.depth = int(getattr(_QUIET_DSP_LOG_STATE, "depth", 1) or 1) - 1


def telemetry_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    try:
        return np.asarray(value).tolist()
    except (TypeError, ValueError):
        return None


def safe_put(st: Any, key: str, value: Any) -> None:
    if isinstance(st, dict):
        st[key] = value


def safe_put_many(st: Any, mapping: dict[str, Any]) -> None:
    if not isinstance(st, dict):
        return
    for key, value in mapping.items():
        st[key] = value


def safe_put_array(st: Any, key: str, arr: Any) -> None:
    value = telemetry_list(arr)
    if value is not None:
        safe_put(st, key, value)


def safe_put_range(st: Any, key: str, lo: float, hi: float) -> None:
    safe_put(st, key, [float(lo), float(hi)])


def apply_baseline_native_telemetry(*, st: Any, native: BaselineNativeTelemetry | None) -> None:
    if native is None:
        return
    safe_put(st, "analysis_mode", str(native.analysis_mode))
    safe_put_array(st, "freq_axis", native.freq_axis)
    safe_put_array(st, "measured_mags", native.measured_mags)
    safe_put_array(st, "target_mags", native.target_mags)
    if native.target_env_lo is not None and native.target_env_hi is not None:
        safe_put_array(st, "target_env_lo", native.target_env_lo)
        safe_put_array(st, "target_env_hi", native.target_env_hi)
        safe_put(
            st,
            "target_env_pivot_hz",
            float(native.target_env_pivot_hz) if native.target_env_pivot_hz is not None else None,
        )
    safe_put_many(
        st,
        {
            "target_shift_db": float(native.target_shift_db),
            "eff_target_db": float(native.eff_target_db),
            "target_level_db_window": float(native.target_level_db_window),
            "meas_level_db_window": float(native.meas_level_db_window),
            "offset_db": float(native.offset_db),
            "offset_method": str(native.offset_method),
        },
    )
    safe_put_range(st, "smart_scan_range", native.smart_scan_range[0], native.smart_scan_range[1])


def apply_baseline_comparison_telemetry(
    *,
    cmp: Any,
    comparison: BaselineComparisonTelemetry | None,
) -> None:
    if comparison is None:
        return
    safe_put(cmp, "analysis_mode", str(comparison.analysis_mode))
    safe_put_array(cmp, "cmp_target_mags", comparison.target_mags)
    safe_put_array(cmp, "cmp_measured_mags", comparison.measured_mags)
    safe_put_array(cmp, "cmp_filter_mags", comparison.filter_mags)
    safe_put_many(
        cmp,
        {
            "cmp_eff_target_db": float(comparison.eff_target_db),
            "cmp_offset_db": float(comparison.offset_db),
            "cmp_meas_level_db_window": float(comparison.meas_level_db_window),
            "cmp_target_level_db_window": float(comparison.target_level_db_window),
            "cmp_offset_method": str(comparison.offset_method),
            "cmp_target_shift_db": float(comparison.target_shift_db),
        },
    )
    safe_put_range(
        cmp,
        "cmp_smart_scan_range",
        comparison.smart_scan_range[0],
        comparison.smart_scan_range[1],
    )


def apply_baseline_telemetry_to_stats(
    *,
    st: Any,
    cmp: Any,
    native: BaselineNativeTelemetry | None,
    comparison: BaselineComparisonTelemetry | None,
) -> None:
    apply_baseline_native_telemetry(st=st, native=native)
    apply_baseline_comparison_telemetry(cmp=cmp, comparison=comparison)


def apply_residual_telemetry(*, st: Any, telemetry: ResidualTelemetry | None) -> None:
    if telemetry is None:
        return
    safe_put_many(
        st,
        {
            "residual_pass_enabled": bool(telemetry.residual_pass_enabled),
            "residual_pass_mode": str(telemetry.residual_pass_mode),
            "residual_null_guard_enabled": bool(telemetry.residual_null_guard_enabled),
            "residual_strength": float(telemetry.residual_strength),
            "residual_smoothing_mult": float(telemetry.residual_smoothing_mult),
            "residual_conf_power": float(telemetry.residual_conf_power),
            "residual_max_delta_db": float(telemetry.residual_max_delta_db),
            "residual_band_min_hz": float(telemetry.residual_band_min_hz),
            "residual_band_max_hz": float(telemetry.residual_band_max_hz),
            "residual_band_edge_hz": float(telemetry.residual_band_edge_hz),
            "residual_band_bins": int(telemetry.residual_band_bins),
            "residual_lf_guard_bins": int(telemetry.residual_lf_guard_bins),
            "residual_lf_boost_blocked_bins": int(telemetry.residual_lf_boost_blocked_bins),
            "residual_right_transition_fade_skipped": bool(telemetry.residual_right_transition_fade_skipped),
            "residual_authority_boost_cap_mean_db": float(telemetry.residual_authority_boost_cap_mean_db),
            "residual_authority_boost_cap_min_db": float(telemetry.residual_authority_boost_cap_min_db),
            "residual_authority_boost_cap_max_db": float(telemetry.residual_authority_boost_cap_max_db),
            "residual_authority_cut_cap_mean_db": float(telemetry.residual_authority_cut_cap_mean_db),
            "residual_authority_cut_cap_min_db": float(telemetry.residual_authority_cut_cap_min_db),
            "residual_authority_cut_cap_max_db": float(telemetry.residual_authority_cut_cap_max_db),
            "residual_blocked_null_risk_bins": int(telemetry.residual_blocked_null_risk_bins),
            "residual_blocked_reflection_risk_bins": int(telemetry.residual_blocked_reflection_risk_bins),
            "residual_blocked_low_authority_bins": int(telemetry.residual_blocked_low_authority_bins),
            "residual_blocked_low_modal_support_bins": int(telemetry.residual_blocked_low_modal_support_bins),
            "residual_modal_polish_bins": int(telemetry.residual_modal_polish_bins),
            "residual_requested_boost_max_db": float(telemetry.residual_requested_boost_max_db),
            "residual_applied_boost_max_db": float(telemetry.residual_applied_boost_max_db),
            "residual_requested_cut_max_db": float(telemetry.residual_requested_cut_max_db),
            "residual_applied_cut_max_db": float(telemetry.residual_applied_cut_max_db),
        },
    )
