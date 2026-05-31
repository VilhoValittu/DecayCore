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
    _auto_cache_get_entry,
    _auto_cache_get_target_for_measurements,
    _auto_cache_get_target_for_measurements_global,
    _auto_filter_cache_key,
    _auto_optimizer_backend,
    _auto_safe_float,
    _auto_select_builtin_target_curve,
    _auto_select_target_curve_with_trials,
    _auto_signature,
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


def _target_seed_from_cache_entry(entry: dict | None) -> dict:
    payload = dict(entry or {}) if isinstance(entry, dict) else {}
    target_seed = payload.get("target_seed_preset")
    if isinstance(target_seed, dict) and target_seed:
        return dict(target_seed)
    best = payload.get("best_preset")
    if isinstance(best, dict) and best:
        return dict(best)
    return {}


def _target_cache_pick_from_entry(
    entry: dict | None,
    *,
    selection_method: str,
) -> dict | None:
    payload = dict(entry or {}) if isinstance(entry, dict) else {}
    hc = str(
        payload.get(
            "best_target_curve",
            payload.get("best_hc_mode", ""),
        )
        or ""
    ).strip()
    seed = _target_seed_from_cache_entry(payload)
    if not hc or not seed:
        return None
    return {
        "selected_hc_mode": str(hc),
        "fit_rms_db": float(
            _auto_safe_float(
                payload.get("fit_rms_db", payload.get("preselect_score", float("nan"))),
                float("nan"),
            )
        ),
        "offset_db": float(_auto_safe_float(payload.get("offset_db", 0.0), 0.0)),
        "selection_method": str(selection_method),
        "top_n": 0,
        "trials_per_curve": 0,
        "candidates": [],
        "evaluated": [],
        "best_preset": dict(seed),
        "best_metrics": dict(payload.get("best_metrics", {}) or {}),
    }


def _try_cached_target_pick_before_search(
    *,
    data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    goal: str,
) -> dict | None:
    filter_key = str(_auto_filter_cache_key(data) or "").strip()
    compat_version = str(
        data.get("auto_mode_compat_version", AUTO_MODE_COMPAT_VERSION)
        or AUTO_MODE_COMPAT_VERSION
    )

    def _normalize_filter_key(value: object) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        try:
            return str(_auto_filter_cache_key({"filter_type": raw}) or "").strip()
        except (

            AttributeError,
            TypeError,
            ValueError,
            KeyError,
            IndexError,
            RuntimeError,
            OSError,
            ImportError,
            ModuleNotFoundError,
            NameError,
        ):
            low = raw.lower()
            if "asym" in low:
                return "asym"
            if "mixed" in low:
                return "mixed"
            if "minimum" in low or "minphase" in low or low == "min" or ("min" in low and "phase" in low):
                return "minimum"
            if "linear" in low:
                return "linear"
            return low

    def _filter_map_get(mapping: object, filter_key: str) -> dict:
            if not isinstance(mapping, dict):
                return {}

            wanted = _normalize_filter_key(filter_key)

            direct = mapping.get(filter_key)
            if isinstance(direct, dict):
                return dict(direct)

            direct_norm = mapping.get(wanted)
            if isinstance(direct_norm, dict):
                return dict(direct_norm)

            for key, value in mapping.items():
                if not isinstance(value, dict):
                    continue
                if _normalize_filter_key(key) == wanted:
                    return dict(value)

            return {}

    def _dict_for_current_filter(mapping: object) -> dict:
        if not isinstance(mapping, dict):
            return {}

        direct = mapping.get(filter_key)
        if isinstance(direct, dict):
            return dict(direct)

        wanted = _normalize_filter_key(filter_key)
        for key, value in mapping.items():
            if not isinstance(value, dict):
                continue
            if _normalize_filter_key(key) == wanted:
                return dict(value)

        return {}

    def _entry_matches_current_filter(entry: object) -> bool:
        if not isinstance(entry, dict):
            return False

        # Newer target-cache entries may carry a filter-specific seed map.
        # If so, accept only when the current filter has its own seed.
        seed = _dict_for_current_filter(entry.get("filter_seed_presets", {}))
        if seed:
            return True

        # Filter-specific entries may carry one of these fields.
        for field in (
            "filter_key",
            "filter_type",
            "_auto_filter_key",
            "_optuna_filter_key",
            "decaycore_filter_key",
        ):
            value = entry.get(field)
            if value is None:
                continue
            if _normalize_filter_key(value) == _normalize_filter_key(filter_key):
                return True

        # If no filter metadata exists, do not trust the entry.
        # This prevents old Linear/global target-cache entries from leaking
        # into Minimum/Mixed/Asymmetric runs.
        return False

    # 1. Measurement-global cache:
    # usable only if it contains a seed for the current filter.
    try:
        global_entry = _auto_cache_get_target_for_measurements_global(
            measurements,
            goal=goal,
            compat_version=compat_version,
        )
    except (

        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        RuntimeError,
        OSError,
        ImportError,
        ModuleNotFoundError,
        NameError,
    ):
        global_entry = None

    if isinstance(global_entry, dict):
        seed = _dict_for_current_filter(global_entry.get("filter_seed_presets", {}))
        if seed:
            entry = dict(global_entry)
            entry["target_seed_preset"] = dict(seed)

            metrics = _dict_for_current_filter(global_entry.get("filter_seed_metrics", {}))
            if metrics:
                entry["best_metrics"] = dict(metrics)

            entry["filter_key"] = filter_key

            pick = _target_cache_pick_from_entry(
                entry,
                selection_method="cache_measurement_global_filter_seed_hit",
            )
            if isinstance(pick, dict):
                return pick

    # 2. Filter-specific measurement target cache:
    # reject legacy/global leakage unless it explicitly matches this filter.
    try:
        measurement_entry = _auto_cache_get_target_for_measurements(
            measurements,
            goal=goal,
            filter_key=filter_key,
            compat_version=compat_version,
        )
    except (

        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        RuntimeError,
        OSError,
        ImportError,
        ModuleNotFoundError,
        NameError,
    ):
        measurement_entry = None

    if _entry_matches_current_filter(measurement_entry):
        entry = dict(measurement_entry)

        seed = _dict_for_current_filter(entry.get("filter_seed_presets", {}))
        if seed and not isinstance(entry.get("target_seed_preset"), dict):
            entry["target_seed_preset"] = dict(seed)

        metrics = _dict_for_current_filter(entry.get("filter_seed_metrics", {}))
        if metrics and not isinstance(entry.get("best_metrics"), dict):
            entry["best_metrics"] = dict(metrics)

        entry["filter_key"] = filter_key

        pick = _target_cache_pick_from_entry(
            entry,
            selection_method="cache_measurement_hit",
        )
        if isinstance(pick, dict):
            return pick

    # 3. Exact signature cache:
    # this is already filter-specific because _auto_signature() uses base_data
    # and _auto_cache_get_entry() is called with filter_key.
    try:
        sig_target = _auto_signature(
            base_data=data,
            measurements=measurements,
            fs_v=int(fs_v),
            taps_v=int(taps_v),
            xos=xos,
            hpf=hpf,
            hc_mode=None,
            include_hc_mode=False,
        )
        signature_entry = _auto_cache_get_entry(
            sig_target,
            filter_key=filter_key,
            compat_version=compat_version,
        )
    except (

        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        RuntimeError,
        OSError,
        ImportError,
        ModuleNotFoundError,
        NameError,
    ):
        signature_entry = None

    pick = _target_cache_pick_from_entry(
        signature_entry,
        selection_method="cache_signature_hit",
    )
    return pick if isinstance(pick, dict) else None


def _run_auto_mode_seed_phases(
    ctx: dict,
    *,
    callbacks: ProcessRunCallbacks,
    support: ProcessRunSupport,
):
    data = ctx.get("resolved_data", ctx["data"])
    ctx["resolved_data"] = data
    ctx["data"] = data
    taps_base = int(ctx["taps_base"])
    f_l = ctx["f_l"]
    m_l = ctx["m_l"]
    f_r = ctx["f_r"]
    m_r = ctx["m_r"]
    p_l = ctx["p_l"]
    p_r = ctx["p_r"]

    try:
        auto_mode_preview = bool(
            str(data.get("mode", "BASIC") or "BASIC").strip().upper() == "AUTO"
            or data.get("camillafir_automatic_mode", False)
        )
    except (

        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        RuntimeError,
        OSError,
        ImportError,
        ModuleNotFoundError,
        NameError,
    ):
        auto_mode_preview = False
    auto_goal = _auto_goal_norm(str(data.get("auto_goal", "balanced") or "balanced"))
    data["auto_goal"] = str(auto_goal)
    if auto_mode_preview and str(auto_goal) == AUTO_MODE_GOAL_FLAT:
        data["unsafe_raw_dsp"] = True
    forced_level_window = _auto_goal_forced_level_window(auto_goal) if auto_mode_preview else None
    if forced_level_window is not None:
        data["lvl_min"] = float(forced_level_window[0])
        data["lvl_max"] = float(forced_level_window[1])
    bass_integration_active = bool(data.get("bass_integration_enable", False)) and auto_mode_preview
    if bass_integration_active and forced_level_window is None:
        data["lvl_min"] = 500.0
        data["lvl_max"] = 3000.0
    auto_basis = "preset_objective_score"
    logger.info(f"Automatic mode goal: {auto_goal} (basis: {auto_basis})")
    auto_status = (
        _get_auto_status_callback(ctx, callbacks=callbacks, support=support)
        if auto_mode_preview
        else callbacks.status
    )

    if auto_mode_preview:
        try:
            ft = str(data.get("filter_type", "mixed") or "mixed")
        except (

            AttributeError,
            TypeError,
            ValueError,
            KeyError,
            IndexError,
            RuntimeError,
            OSError,
            ImportError,
            ModuleNotFoundError,
            NameError,
        ):
            ft = "mixed"
        auto_status(
            "DecayCore automatic mode: init "
            f"(goal {auto_goal}, basis {auto_basis}, filter {ft}, taps {int(taps_base)})"
        )
        if forced_level_window is not None:
            auto_status(
                "DecayCore automatic mode: forced Smart Scan range "
                f"{float(forced_level_window[0]):.0f}-{float(forced_level_window[1]):.0f} Hz "
                f"(goal {auto_goal})"
            )
        elif bass_integration_active:
            auto_status(
                "DecayCore automatic mode: bass integration Smart Scan range "
                "500-3000 Hz (main), 20-200 Hz (sub)"
            )
    if auto_mode_preview:
        try:
            est_mag_c_min = _estimate_auto_mag_c_min_hz(
                f_l,
                m_l,
                f_r,
                m_r,
                default_hz=_auto_safe_float(data.get("mag_c_min", 25.0), 25.0),
            )
            data["mag_c_min"] = float(est_mag_c_min)
            data["_auto_mag_c_min_hz"] = float(est_mag_c_min)
            est_low_bass_cut = float(
                np.clip(
                    float(est_mag_c_min) + float(AUTO_MODE_LOW_BASS_FROM_F6_ADD_HZ),
                    float(AUTO_MODE_LOW_BASS_MIN_HZ),
                    float(AUTO_MODE_LOW_BASS_MAX_HZ),
                )
            )
            est_exc_freq = float(
                np.clip(
                    float(est_mag_c_min) + float(AUTO_MODE_EXC_FROM_F6_ADD_HZ),
                    float(AUTO_MODE_EXC_MIN_HZ),
                    float(AUTO_MODE_EXC_MAX_HZ),
                )
            )
            data["low_bass_cut_hz"] = float(round(est_low_bass_cut, 1))
            data["exc_freq"] = float(round(est_exc_freq, 1))
            data["_auto_low_bass_cut_hz"] = float(round(est_low_bass_cut, 1))
            data["_auto_exc_freq_hz"] = float(round(est_exc_freq, 1))
            data["_auto_exc_seed_freq_hz"] = float(round(est_exc_freq, 1))
            optimizer_backend = str(_auto_optimizer_backend(data, default_optuna_enabled=False) or "builtin").strip().lower()
            if str(optimizer_backend) == "optuna":
                auto_status(
                    "DecayCore automatic mode: protection seed "
                    f"(smoothed -6 dB point {float(est_mag_c_min):.1f} Hz -> "
                    f"mag_c_min {float(data['mag_c_min']):.1f} Hz, "
                    f"low-cut {float(data['low_bass_cut_hz']):.1f} Hz, "
                    f"exc seed {float(data['exc_freq']):.1f} Hz)"
                )
            else:
                auto_status(
                    "DecayCore automatic mode: protection seed "
                    f"(smoothed -6 dB point {float(est_mag_c_min):.1f} Hz -> "
                    f"mag_c_min {float(data['mag_c_min']):.1f} Hz, "
                    f"low-cut {float(data['low_bass_cut_hz']):.1f} Hz, "
                    f"exc seed {float(data['exc_freq']):.1f} Hz, "
                    "exc seed preserved; mag_c_min and low_bass_cut values auto-tuned in preset search)"
                )
            (
                hpf_f_l,
                hpf_m_l,
                hpf_f_r,
                hpf_m_r,
                hpf_freq_key,
                hpf_slope_key,
                hpf_status_label,
                user_hpf_enabled,
            ) = _resolve_auto_hpf_seed_source(ctx, data, f_l, m_l, f_r, m_r)
            auto_hpf = _estimate_auto_hpf_from_response(
                hpf_f_l,
                hpf_m_l,
                hpf_f_r,
                hpf_m_r,
                default_freq_hz=_auto_safe_float(data.get(hpf_freq_key, 20.0), 20.0),
                default_slope_db_oct=int(_auto_safe_float(data.get(hpf_slope_key, 24), 24.0)),
            )
            if isinstance(auto_hpf, dict):
                auto_hpf = _resolve_auto_hpf_application(
                    auto_hpf,
                    user_hpf_enabled=bool(user_hpf_enabled),
                )
                auto_hpf_conf = _auto_safe_float(auto_hpf.get("confidence", 0.0), 0.0)
                auto_hpf_method = str(auto_hpf.get("method", "") or "").strip().lower()
                auto_hpf_apply = bool(auto_hpf.get("applied", False))
                auto_hpf_decision = str(auto_hpf.get("decision", "") or "").strip().lower()
                data["_auto_hpf_meta"] = dict(auto_hpf)
                if auto_hpf_apply:
                    auto_hpf_freq = _auto_safe_float(
                        auto_hpf.get("freq", data.get(hpf_freq_key, 20.0)),
                        _auto_safe_float(data.get(hpf_freq_key, 20.0), 20.0),
                    )
                    auto_hpf_slope = int(
                        round(
                            _auto_safe_float(
                                auto_hpf.get("slope_db_oct", data.get(hpf_slope_key, 24)),
                                _auto_safe_float(data.get(hpf_slope_key, 24), 24.0),
                            )
                        )
                    )
                    data[hpf_freq_key] = float(round(auto_hpf_freq, 1))
                    data[hpf_slope_key] = int(max(6, auto_hpf_slope))
                    auto_status(
                        f"DecayCore automatic mode: {hpf_status_label} auto-fit seed "
                        f"{float(data[hpf_freq_key]):.1f} Hz/{int(data[hpf_slope_key])} dB/oct "
                        f"(method {auto_hpf_method or 'n/a'}, confidence {auto_hpf_conf:.2f})"
                    )
                elif auto_hpf_decision in ("keep_ui_seed", "keep_user_seed"):
                    auto_hpf_freq = _auto_safe_float(
                        auto_hpf.get("freq", data.get(hpf_freq_key, 20.0)),
                        _auto_safe_float(data.get(hpf_freq_key, 20.0), 20.0),
                    )
                    auto_hpf_slope = int(
                        round(
                            _auto_safe_float(
                                auto_hpf.get("slope_db_oct", data.get(hpf_slope_key, 24)),
                                _auto_safe_float(data.get(hpf_slope_key, 24), 24.0),
                            )
                        )
                    )
                    auto_status(
                        f"DecayCore automatic mode: {hpf_status_label} auto-fit fallback "
                        f"{float(auto_hpf_freq):.1f} Hz/{int(max(6, auto_hpf_slope))} dB/oct "
                        f"(method {auto_hpf_method or 'n/a'}, confidence {auto_hpf_conf:.2f}) -> "
                        f"keeping {'user' if auto_hpf_decision == 'keep_user_seed' else 'UI'} seed"
                    )
                else:
                    auto_status(
                        f"DecayCore automatic mode: {hpf_status_label} auto-fit skipped "
                        f"(method {auto_hpf_method or 'n/a'}, confidence {auto_hpf_conf:.2f})"
                    )

            auto_target_mode = support.auto_target_mode_norm(data.get("auto_target_mode", "auto"))
            data["auto_target_mode"] = str(auto_target_mode)
            hc_mode_raw = str(data.get("hc_mode", "") or "").strip().lower()
            has_local_target = bool(str(data.get("local_path_house", "") or "").strip())
            has_upload_target = support.has_uploaded_target_file(data)
            wants_custom_target = bool(("upload" in hc_mode_raw) or ("custom" in hc_mode_raw))
            use_user_target = bool(auto_target_mode == "selected")
            if not use_user_target:
                use_user_target = bool(
                    has_local_target
                    or (wants_custom_target and has_upload_target)
                )
            if use_user_target:
                data.pop("_auto_target_seed_preset", None)
                selected_hc = str(data.get("hc_mode", "n/a") or "n/a")
                if auto_target_mode == "selected":
                    auto_status(
                        "DecayCore automatic mode: target curve mode=selected "
                        f"(using {selected_hc}, auto target comparison disabled)"
                    )
                else:
                    auto_status(
                        "DecayCore automatic mode: target curve mode=user "
                        f"(using {selected_hc}, skip built-in target comparison)"
                    )
            else:
                if wants_custom_target and not has_local_target:
                    auto_status(
                        "DecayCore automatic mode: custom target selected but no file found, "
                        "using built-in target comparison"
                    )
                pre_target_rates = choose_target_rates(data)
                pre_fs = int(pre_target_rates[0]) if pre_target_rates else int(data.get("fs", 44100) or 44100)
                if bool(data.get("multi_rate_opt", False)):
                    pre_taps = int(scale_taps_with_fs(pre_fs, base_taps=taps_base))
                else:
                    pre_taps = int(taps_base)
                pre_xos, pre_hpf = build_xos_hpf(data)
                # For direct_dac bass integration, target preselection should score
                # against the summed (main+sub) response, not the main-only response
                # that ctx["f_l"]/ctx["m_l"] carry in this mode.
                pre_f_l, pre_m_l, pre_f_r, pre_m_r = f_l, m_l, f_r, m_r
                bi_mode_pre = "direct_dac"
                if bool(data.get("bass_integration_enable", False)) and bi_mode_pre == "direct_dac":
                    try:
                        _bi_bundle = ctx.get("bass_integration_bundle", None)
                        if _bi_bundle is not None:
                            _lt = getattr(_bi_bundle, "l_total", None)
                            _rt = getattr(_bi_bundle, "r_total", None)
                            if _lt is not None and _rt is not None:
                                _tfl = np.asarray(getattr(_lt, "freqs_hz", []), dtype=float)
                                _tml = np.asarray(getattr(_lt, "mag_db", []), dtype=float)
                                _tfr = np.asarray(getattr(_rt, "freqs_hz", []), dtype=float)
                                _tmr = np.asarray(getattr(_rt, "mag_db", []), dtype=float)
                                if _tfl.size >= 32 and _tml.size == _tfl.size:
                                    pre_f_l, pre_m_l = _tfl, _tml
                                if _tfr.size >= 32 and _tmr.size == _tfr.size:
                                    pre_f_r, pre_m_r = _tfr, _tmr
                    except (

                        AttributeError,
                        TypeError,
                        ValueError,
                        KeyError,
                        IndexError,
                        RuntimeError,
                        OSError,
                        ImportError,
                        ModuleNotFoundError,
                        NameError,
                    ):
                        logger.exception("bass integration pre-measurement extract")
                pre_measurements = dict(ctx.get("measurements", {}) or {})
                pre_measurements.update(
                    {
                        "f_l": np.asarray(pre_f_l, dtype=float),
                        "m_l": np.asarray(pre_m_l, dtype=float),
                        "p_l": np.asarray(p_l, dtype=float),
                        "f_r": np.asarray(pre_f_r, dtype=float),
                        "m_r": np.asarray(pre_m_r, dtype=float),
                        "p_r": np.asarray(p_r, dtype=float),
                        "ui_data": data,
                        "is_wav_source": bool(detect_is_wav_source(data)),
                    }
                )
                tc_pick = _try_cached_target_pick_before_search(
                    data=data,
                    measurements=pre_measurements,
                    fs_v=int(pre_fs),
                    taps_v=int(pre_taps),
                    xos=pre_xos,
                    hpf=pre_hpf,
                    goal=str(auto_goal),
                )
                if not isinstance(tc_pick, dict):
                    tc_pick = _auto_select_target_curve_with_trials(
                        base_data=data,
                        measurements=pre_measurements,
                        fs_v=int(pre_fs),
                        taps_v=int(pre_taps),
                        xos=pre_xos,
                        hpf=pre_hpf,
                        status_cb=auto_status,
                        top_n=int(AUTO_MODE_TARGET_TOP_N),
                        trials_per_curve=int(AUTO_MODE_TARGET_TRIALS_PER_CURVE),
                    )
                if not isinstance(tc_pick, dict):
                    auto_status(
                        "DecayCore automatic mode: target search init "
                        f"(top-{AUTO_MODE_TARGET_TOP_N}, {AUTO_MODE_TARGET_TRIALS_PER_CURVE} trials/curve, "
                        f"fs {int(pre_fs)} Hz, taps {int(pre_taps)}, "
                        f"-6 dB point {float(est_mag_c_min):.1f} Hz, goal {auto_goal})"
                    )
                    tc_pick = _auto_select_builtin_target_curve(
                        data,
                        f_l=pre_f_l,
                        m_l=pre_m_l,
                        f_r=pre_f_r,
                        m_r=pre_m_r,
                        measurements=pre_measurements,
                    )
                if isinstance(tc_pick, dict):
                    chosen_hc = str(tc_pick.get("selected_hc_mode", "Harman6") or "Harman6")
                    prev_hc = str(data.get("hc_mode", "") or "")
                    method_raw = str(tc_pick.get("selection_method", "fit_rms") or "fit_rms")
                    method_txt = support.auto_target_selection_method_text(method_raw)
                    data["hc_mode"] = chosen_hc
                    target_seed_preset = dict(tc_pick.get("best_preset", {}) or {})
                    selection_method_raw = str(tc_pick.get("selection_method", "") or "").strip().lower()
                    cached_target_methods = {
                        "cache_signature_hit",
                        "cache_measurement_hit",
                        "cache_measurement_global_hit",
                        "cache_measurement_global_filter_seed_hit",
                        "cache_optuna_target_hit",
                    }
                    if target_seed_preset:
                        data["_auto_target_seed_preset"] = dict(target_seed_preset)
                        if selection_method_raw in cached_target_methods:
                            data["_auto_target_seed_source"] = selection_method_raw
                        else:
                            data["_auto_target_seed_source"] = "fresh_target_search"
                        target_seed_metrics = dict(tc_pick.get("best_metrics", {}) or {})
                        if target_seed_metrics:
                            data["_auto_target_seed_metrics"] = dict(target_seed_metrics)
                        else:
                            data.pop("_auto_target_seed_metrics", None)
                    else:
                        data.pop("_auto_target_seed_preset", None)
                        data.pop("_auto_target_seed_metrics", None)
                        data.pop("_auto_target_seed_source", None)
                    if chosen_hc == "Adaptive" and "_synth_hc_f" in tc_pick:
                        data["_synth_hc_f"] = tc_pick["_synth_hc_f"]
                        data["_synth_hc_m"] = tc_pick["_synth_hc_m"]
                    else:
                        data.pop("_synth_hc_f", None)
                        data.pop("_synth_hc_m", None)
                    data["local_path_house"] = ""
                    data["_auto_target_curve_meta"] = dict(tc_pick)
                    if chosen_hc == "Adaptive":
                        auto_status(
                            "DecayCore automatic mode: adaptive target selected "
                            "(synthesized from room measurements, bass buildup, tilt and HF roll-off)"
                        )
                    elif selection_method_raw in cached_target_methods:
                        auto_status(
                            "DecayCore automatic mode: target cache hit "
                            f"{chosen_hc} (method {method_txt}, "
                            f"fit_rms {float(tc_pick.get('fit_rms_db', 0.0)):.3f} dB, "
                            "skipping target comparison trials)"
                        )
                    else:
                        auto_status(
                            "DecayCore automatic mode: target search winner "
                            f"{chosen_hc} (method {method_txt}, "
                            f"fit_rms {float(tc_pick.get('fit_rms_db', 0.0)):.3f} dB, "
                            f"tested {int(tc_pick.get('top_n', 0) or 0)} curves x "
                            f"{int(tc_pick.get('trials_per_curve', 0) or 0)} trials)"
                        )
                    logger.info(
                        f"Automatic mode target select: {chosen_hc} "
                        f"(fit_rms={float(tc_pick.get('fit_rms_db', 0.0)):.3f} dB, "
                        f"method={method_txt}, "
                        f"prev={prev_hc or 'n/a'})"
                    )
                    if target_seed_preset:
                        logger.info(
                            "Automatic mode target seed preset: "
                            + ", ".join([f"{k}={v}" for k, v in target_seed_preset.items()])
                        )
        except (

            AttributeError,
            TypeError,
            ValueError,
            KeyError,
            IndexError,
            RuntimeError,
            OSError,
            ImportError,
            ModuleNotFoundError,
            NameError,
        ) as exc:
            logger.warning(f"Automatic mode target preselect failed: {type(exc).__name__}: {exc}")

    ctx["auto_mode_preview"] = auto_mode_preview
    ctx["auto_goal"] = auto_goal
    ctx["auto_basis"] = auto_basis


__all__ = ['_run_auto_mode_seed_phases']


def _link_sibling_exports() -> None:
    import importlib
    package = __package__
    for module_name in ['progress', 'status_text', 'seed_phases', 'search']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_link_sibling_exports()
