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
import re
import typing

import numpy as np

from ...auto_mode.api import (
    AUTO_MODE_GOAL_FLAT,
    AUTO_MODE_COMPAT_VERSION,
    AUTO_MODE_EXC_FROM_F6_ADD_HZ,
    AUTO_MODE_EXC_MAX_HZ,
    AUTO_MODE_EXC_MIN_HZ,
    AUTO_MODE_LOCAL_REFINE_ENABLED,
    AUTO_MODE_LOCAL_REFINE_TOP_K,
    AUTO_MODE_LOCAL_REFINE_TRIALS_PER_TOP,
    AUTO_MODE_LOW_BASS_FROM_F6_ADD_HZ,
    AUTO_MODE_LOW_BASS_MAX_HZ,
    AUTO_MODE_LOW_BASS_MIN_HZ,
    AUTO_MODE_PHASE3_MICRO_TRIALS,
    AUTO_MODE_REFINE_TRIALS,
    AUTO_MODE_TARGET_TOP_N,
    AUTO_MODE_TARGET_TRIALS_PER_CURVE,
    AUTO_MODE_TRIALS,
    _auto_goal_norm,
    _auto_optimizer_backend,
    _auto_safe_float,
    _auto_select_builtin_target_curve,
    _auto_select_target_curve_with_trials,
    _estimate_auto_hpf_from_response,
    _estimate_auto_mag_c_min_hz,
    _resolve_auto_hpf_application,
    _run_auto_mode_search,
)
from ...auto_mode.rank_score import attach_official_rank_score, official_rank_score
from ...auto_mode.shared import _auto_goal_forced_level_window
from ...application.run_contracts import apply_auto_mode_result
from ...config.decaycore_pipeline import (
    build_xos_hpf,
    choose_target_rates,
    detect_is_wav_source,
    filter_type_short,
)
from ...ui.decaycore_utils import scale_taps_with_fs
from ..bridge_types import ProcessRunCallbacks

if typing.TYPE_CHECKING:
    from ..process_run_flow import ProcessRunSupport

logger = logging.getLogger("DecayCore")

_AUTO_PROGRESS_INIT = 0.06
_AUTO_PROGRESS_TARGET_MODE = 0.12
_AUTO_PROGRESS_TARGET_PRESELECT = 0.15
_AUTO_PROGRESS_TARGET_TRIALS_START = 0.21
_AUTO_PROGRESS_TARGET_TRIALS_END = 0.33
_AUTO_PROGRESS_PRESET_SEARCH_START = 0.36
_AUTO_PROGRESS_PHASE1_START = 0.39
_AUTO_PROGRESS_PHASE1_END = 0.61
_AUTO_PROGRESS_PHASE2_START = 0.61
_AUTO_PROGRESS_PHASE2_END = 0.82
_AUTO_PROGRESS_PHASE3_START = 0.82
_AUTO_PROGRESS_PHASE3_END = 0.85
_AUTO_PROGRESS_FINALIZE = 0.88

def _build_auto_selected_text(run_data: dict) -> str:
    try:
        target_name = str(
            run_data.get(
                "target_curve_name",
                run_data.get("hc_mode", "n/a"),
            )
            or "n/a"
        ).strip()
    except Exception:
        target_name = "n/a"
    if not target_name:
        target_name = "n/a"

    f6_hz = _auto_safe_float(
        run_data.get("_auto_mag_c_min_hz", run_data.get("mag_c_min", float("nan"))),
        float("nan"),
    )
    f6_txt = f"{f6_hz:.1f} Hz" if np.isfinite(f6_hz) else "n/a"

    bi_mode = str(
        run_data.get("bass_integration_mode", "avr_lfe_main_decomposed") or "avr_lfe_main_decomposed"
    ).strip().lower()
    is_direct_dac = bool(run_data.get("bass_integration_enable", False) and bi_mode == "direct_dac")

    hpf_enabled = True if is_direct_dac else bool(run_data.get("hpf_enable", False))
    hpf_freq_key = "sub_hpf_freq" if is_direct_dac else "hpf_freq"
    hpf_slope_key = "sub_hpf_slope" if is_direct_dac else "hpf_slope"
    hpf_label = "Sub HPF" if is_direct_dac else "HPF"
    hpf_freq = _auto_safe_float(run_data.get(hpf_freq_key, float("nan")), float("nan"))
    hpf_slope = _auto_safe_float(run_data.get(hpf_slope_key, float("nan")), float("nan"))
    if hpf_enabled and np.isfinite(hpf_freq):
        if np.isfinite(hpf_slope) and float(hpf_slope) > 0.0:
            hpf_txt = f"{hpf_freq:.1f} Hz/{int(round(hpf_slope))} dB/oct"
        else:
            hpf_txt = f"{hpf_freq:.1f} Hz"
    else:
        hpf_txt = "off"

    ft_short = filter_type_short(str(run_data.get("filter_type", "") or ""))

    phase_hz = _auto_safe_float(run_data.get("phase_limit", float("nan")), float("nan"))
    phase_txt = f"{phase_hz:.1f} Hz" if np.isfinite(phase_hz) else "n/a"

    mixed_hz = _auto_safe_float(run_data.get("mixed_freq", float("nan")), float("nan"))
    mixed_txt = f"{mixed_hz:.1f} Hz" if np.isfinite(mixed_hz) else "n/a"

    best_metrics = attach_official_rank_score(run_data.get("best_metrics", {}))
    rank_score = official_rank_score(best_metrics)
    rank_txt = f", winner rank {rank_score:.3f}/100" if np.isfinite(rank_score) else ""

    detail_txt = ""
    if ft_short in ("Linear", "Asymmetric"):
        detail_txt = f", phase limit {phase_txt}"
    elif ft_short == "Mixed":
        detail_txt = f", mixed freq {mixed_txt}"

    return (
        "Chosen (Automatic mode): "
        f"target {target_name}, {hpf_label} {hpf_txt}, -6 dB {f6_txt}{detail_txt}{rank_txt}"
    )

def _resolve_auto_hpf_seed_source(
    ctx: dict,
    data: dict,
    f_l: np.ndarray,
    m_l: np.ndarray,
    f_r: np.ndarray,
    m_r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str, str, bool]:
    bi_mode = str(
        data.get("bass_integration_mode", "avr_lfe_main_decomposed") or "avr_lfe_main_decomposed"
    ).strip().lower()
    is_direct_dac = bool(data.get("bass_integration_enable", False) and bi_mode == "direct_dac")
    if is_direct_dac:
        bundle = ctx.get("bass_integration_bundle", None)
        try:
            if bundle is not None:
                sub_f_l = np.asarray(getattr(bundle.l_sub, "freqs_hz", []), dtype=float)
                sub_m_l = np.asarray(getattr(bundle.l_sub, "mag_db", []), dtype=float)
                sub_f_r = np.asarray(getattr(bundle.r_sub, "freqs_hz", []), dtype=float)
                sub_m_r = np.asarray(getattr(bundle.r_sub, "mag_db", []), dtype=float)
                if sub_f_l.size > 0 and sub_m_l.size == sub_f_l.size:
                    return (
                        sub_f_l,
                        sub_m_l,
                        sub_f_r,
                        sub_m_r,
                        "sub_hpf_freq",
                        "sub_hpf_slope",
                        "Sub HPF",
                        True,
                    )
        except Exception:
            logger.debug("Direct-DAC sub HPF seed source fallback to main data", exc_info=True)
    return (
        np.asarray(f_l, dtype=float),
        np.asarray(m_l, dtype=float),
        np.asarray(f_r, dtype=float),
        np.asarray(m_r, dtype=float),
        "hpf_freq",
        "hpf_slope",
        "HPF",
        bool(data.get("hpf_enable", False)),
    )

def _auto_finalize_status_suffix(winner_explanation: dict | None) -> str:
    phase_label = str(dict(winner_explanation or {}).get("phase_label", "") or "").strip()
    if not phase_label:
        return ""
    lower = phase_label.lower()
    if not any(token in lower for token in ("pareto", "tie-break", "winner polish", "cache")):
        return ""
    return f", via {phase_label}"

def _build_auto_finalize_status(
    best_metrics: dict | None,
    *,
    winner_explanation: dict | None = None,
) -> str:
    metrics = attach_official_rank_score(best_metrics or {})
    best_rank_official = official_rank_score(metrics)
    bass_boost_db = _auto_safe_float(
        metrics.get("bass_boost_20_200_db", metrics.get("max_net_boost_db")),
        0.0,
    )
    return (
        "DecayCore automatic mode: finalize "
        f"(winner rank {best_rank_official:.3f}/100, "
        f"avg {_auto_safe_float(metrics.get('avg_score'), 0.0):.3f}, "
        f"boost {bass_boost_db:.2f} dB, "
        f"events {int(metrics.get('events_total', 0) or 0)}"
        f"{_auto_finalize_status_suffix(winner_explanation)})"
    )


__all__ = ['_build_auto_selected_text', '_resolve_auto_hpf_seed_source', '_auto_finalize_status_suffix', '_build_auto_finalize_status']


def _load_sibling_symbols() -> None:
    import importlib
    package = __package__
    for module_name in ['progress', 'status_text', 'seed_phases', 'search']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_load_sibling_symbols()
