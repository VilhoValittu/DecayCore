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

def _progress_lerp(start: float, end: float, fraction: float) -> float:
    frac = float(np.clip(_auto_safe_float(fraction, 0.0), 0.0, 1.0))
    return float(start + (end - start) * frac)

def _auto_progress_fraction(
    done: int,
    total: int,
    *,
    base_done: int = 0,
    base_total: int = 1,
) -> float:
    total_i = max(1, int(total))
    base_total_i = max(1, int(base_total))
    done_i = int(np.clip(int(done), 0, total_i))
    base_done_i = int(np.clip(int(base_done), 0, base_total_i))
    overall = float(base_done_i) + (float(done_i) / float(total_i))
    return float(np.clip(overall / float(base_total_i), 0.0, 1.0))

def _estimate_auto_progress_from_status(msg: str) -> float | None:
    text = str(msg or "").strip()
    if not text:
        return None
    lower = text.lower()
    if "decaycore automatic mode" not in lower and "camillafir automatic mode" not in lower:
        return None

    if "automatic mode: finalize " in lower or "automatic mode: phase 4 finalize" in lower:
        return float(_AUTO_PROGRESS_FINALIZE)
    if (
        "phase2 summary" in lower
        or "phase 2 pareto comparison" in lower
        or "phase 2 pareto selected winner" in lower
        or "phase_limit winner polish" in lower
        or "mag_c_min winner polish" in lower
        or "hpf winner polish" in lower
    ):
        return float(_AUTO_PROGRESS_FINALIZE - 0.01)

    match = re.search(r"(?:phase 3/3 micro|micro refine)\s+(\d+)/(\d+)", text, flags=re.IGNORECASE)
    if match:
        done, total = (int(match.group(1)), int(match.group(2)))
        return _progress_lerp(
            _AUTO_PROGRESS_PHASE3_START,
            _AUTO_PROGRESS_PHASE3_END,
            _auto_progress_fraction(done, total),
        )
    if "phase3 micro summary" in lower or "micro refine summary" in lower:
        return float(_AUTO_PROGRESS_PHASE3_END)
    if "phase3 micro " in lower or "micro refine " in lower:
        return float(_AUTO_PROGRESS_PHASE3_START)

    match = re.search(
        r"phase 2/2 local center#(\d+)\s+(\d+)/(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        center_idx, done, total = (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        )
        return _progress_lerp(
            _AUTO_PROGRESS_PHASE2_START,
            _AUTO_PROGRESS_PHASE2_END,
            _auto_progress_fraction(
                done,
                total,
                base_done=max(0, center_idx - 1),
                base_total=max(1, int(AUTO_MODE_LOCAL_REFINE_TOP_K)),
            ),
        )
    match = re.search(r"local refine summary .*center #(\d+)", text, flags=re.IGNORECASE)
    if match:
        center_idx = int(match.group(1))
        return _progress_lerp(
            _AUTO_PROGRESS_PHASE2_START,
            _AUTO_PROGRESS_PHASE2_END,
            float(center_idx) / float(max(1, int(AUTO_MODE_LOCAL_REFINE_TOP_K))),
        )
    if "automatic mode: local refine " in lower:
        return float(_AUTO_PROGRESS_PHASE2_START)

    match = re.search(r"phase 1/2\s+(\d+)/(\d+)", text, flags=re.IGNORECASE)
    if match:
        done, total = (int(match.group(1)), int(match.group(2)))
        return _progress_lerp(
            _AUTO_PROGRESS_PHASE1_START,
            _AUTO_PROGRESS_PHASE1_END,
            _auto_progress_fraction(done, total),
        )
    if "phase1 done" in lower:
        return float(_AUTO_PROGRESS_PHASE1_END)

    match = re.search(
        r"target\s+(\d+)/(\d+).*?trial\s+(\d+)/(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        target_done, target_total, trial_done, trial_total = (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
        )
        return _progress_lerp(
            _AUTO_PROGRESS_TARGET_TRIALS_START,
            _AUTO_PROGRESS_TARGET_TRIALS_END,
            _auto_progress_fraction(
                trial_done,
                trial_total,
                base_done=max(0, target_done - 1),
                base_total=target_total,
            ),
        )
    match = re.search(r"best improved\s+(?:trial\s+)?(\d+)/(\d+)", text, flags=re.IGNORECASE)
    if match and "selecting target curve" in lower:
        done, total = (int(match.group(1)), int(match.group(2)))
        return _progress_lerp(
            _AUTO_PROGRESS_TARGET_TRIALS_START,
            _AUTO_PROGRESS_TARGET_TRIALS_END,
            _auto_progress_fraction(done, total),
        )
    match = re.search(
        r"selecting target curve\s+\(testing .*?(\d+)/(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        done, total = (int(match.group(1)), int(match.group(2)))
        return _progress_lerp(
            _AUTO_PROGRESS_TARGET_TRIALS_START,
            _AUTO_PROGRESS_TARGET_TRIALS_END,
            _auto_progress_fraction(max(0, done - 1), total),
        )

    if (
        "target finalize" in lower
        or "target preselect winner" in lower
        or "target search winner" in lower
        or "adaptive target selected" in lower
        or "target loaded directly from cache" in lower
        or "adaptive target synthesized from measurements" in lower
    ):
        return float(_AUTO_PROGRESS_TARGET_TRIALS_END)
    if "target shortlist" in lower:
        return float(_AUTO_PROGRESS_TARGET_TRIALS_START - 0.01)
    if "target preselect top-3" in lower or "target preselect cache seed" in lower:
        return float(_AUTO_PROGRESS_TARGET_PRESELECT)
    if "target preselect init" in lower or "target search init" in lower:
        return float(_AUTO_PROGRESS_TARGET_PRESELECT)
    if (
        "target curve mode=" in lower
        or "custom target selected but no file found" in lower
    ):
        return float(_AUTO_PROGRESS_TARGET_MODE)

    match = re.search(r"cache refine round\s+(\d+)/(\d+)", text, flags=re.IGNORECASE)
    if match:
        round_idx, round_total = (int(match.group(1)), int(match.group(2)))
        inner_match = re.search(r"(\d+)/(\d+)", text[match.end() :], flags=re.IGNORECASE)
        if inner_match:
            inner_done, inner_total = (int(inner_match.group(1)), int(inner_match.group(2)))
        else:
            inner_done, inner_total = (0, 1)
        return _progress_lerp(
            _AUTO_PROGRESS_PHASE1_START,
            _AUTO_PROGRESS_PHASE2_END,
            _auto_progress_fraction(
                inner_done,
                inner_total,
                base_done=max(0, round_idx - 1),
                base_total=round_total,
            ),
        )
    if "cache refine init" in lower or "preset loaded from cache" in lower:
        return float(_AUTO_PROGRESS_PRESET_SEARCH_START)
    if "cache refine summary" in lower:
        return float(_AUTO_PROGRESS_PHASE2_END)

    if "preset search init" in lower or "phase search init" in lower:
        return float(_AUTO_PROGRESS_PRESET_SEARCH_START)
    if "hpf auto-fit" in lower or "protection seed" in lower:
        return float(_AUTO_PROGRESS_TARGET_MODE - 0.01)
    if "automatic mode: init" in lower:
        return float(_AUTO_PROGRESS_INIT)
    return None

def _set_auto_progress(ctx: dict, *, support: ProcessRunSupport, value: float) -> None:
    state = ctx.setdefault("_auto_progress_state", {})
    current = _auto_safe_float(state.get("value", 0.0), 0.0)
    next_value = float(np.clip(_auto_safe_float(value, current), current, _AUTO_PROGRESS_FINALIZE))
    if next_value <= float(current) + 1e-9:
        return
    state["value"] = float(next_value)
    try:
        support.ui_bridge.set_progress(float(next_value))
    except Exception:
        logger.exception("auto progress bridge update failed")

def _get_auto_status_callback(
    ctx: dict,
    *,
    callbacks: ProcessRunCallbacks,
    support: ProcessRunSupport,
) -> typing.Callable[[str], None]:
    cb = ctx.get("_auto_status_callback")
    if callable(cb):
        return cb

    def _status(msg: str) -> None:
        progress = _estimate_auto_progress_from_status(msg)
        if progress is not None:
            _set_auto_progress(ctx, support=support, value=float(progress))
        try:
            callbacks.status(msg)
        except Exception:
            logger.exception("auto status callback failed")

    ctx["_auto_status_callback"] = _status
    return _status


__all__ = ['_progress_lerp', '_auto_progress_fraction', '_estimate_auto_progress_from_status', '_set_auto_progress', '_get_auto_status_callback']


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
