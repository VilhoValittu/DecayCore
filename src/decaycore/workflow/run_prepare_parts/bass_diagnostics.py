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
import math
import time
import typing
from datetime import datetime

import numpy as np

from ...application.health_service import compute_health
from ...application.house_curve_service import load_house_curve
from ...application.run_request import RunRequest
from ...application.run_contracts import (
    PreparedRunInput,
    ResolvedRunConfig,
    copy_resolved_data,
    copy_source_ui_data,
)
from ...common.result_postprocess import _irwin_tag
from ...config.decaycore_config import save_config
from ...config.decaycore_pipeline import (
    build_xos_hpf,
    choose_dash_fs,
    choose_target_rates,
    detect_is_wav_source,
    filter_type_short,
    log_df_smoothing_toggle,
)
from ...io.generated_measurement_source import generated_source_matches_upload, parse_generated_source
from ...io.measurements_loader import (
    _try_load_harmonic_sidecar,
    _try_load_rt60_sidecar,
    load_bass_integration_measurements,
    load_measurements_lr,
    load_raw_irs_lr,
    load_raw_ir_sub,
)
from ...io.measurements_txt import parse_measurements_from_path
from ...resources.i8n.decaycore_i18n import t
from ..bridge_types import ProcessRunCallbacks

if typing.TYPE_CHECKING:
    from ..process_run_flow import ProcessRunSupport

logger = logging.getLogger("DecayCore")

def _status(callbacks, msg: str) -> None:
    """Send a status message to UI callbacks, silently ignoring any errors."""
    if callbacks is not None:
        try:
            callbacks.status(msg)
        except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
            logger.exception("status callback")

def _direct_dac_filter_params(data: dict) -> tuple[int, float, int]:
    try:
        xo_order = max(1, int(round(float(data.get("sub_crossover_slope", 24) or 24.0))) // 6)
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
        xo_order = 4
    try:
        sub_hpf_hz = float(data.get("sub_hpf_freq", 20.0) or 20.0)
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
        sub_hpf_hz = 20.0
    try:
        sub_hpf_order = max(1, int(round(float(data.get("sub_hpf_slope", 12) or 12.0))) // 6)
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
        sub_hpf_order = 2
    return int(xo_order), float(sub_hpf_hz), int(sub_hpf_order)

def _direct_dac_alignment_params(data: dict) -> tuple[float, bool, float]:
    try:
        delay_ms = float(data.get("bass_integration_sub_delay_ms", 0.0) or 0.0)
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
        delay_ms = 0.0
    try:
        gain_trim_db = float(data.get("bass_integration_sub_gain_trim_db", 0.0) or 0.0)
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
        gain_trim_db = 0.0
    return (
        float(delay_ms if math.isfinite(delay_ms) else 0.0),
        bool(data.get("bass_integration_sub_polarity_invert", False)),
        float(gain_trim_db if math.isfinite(gain_trim_db) else 0.0),
    )

def _compute_direct_dac_prepare_recommendation(bundle: object, data: dict, callbacks=None) -> dict:
    """Unified direct-DAC prepare: alignment + crossover + allpass in one call.

    Backend selected by ``data["bass_integration_prepare_backend"]`` (default ``"auto"``):
    - ``"auto"``: uses Optuna when in AUTO mode and optuna is importable; else builtin.
    - ``"optuna"``: forces Optuna with builtin fallback on import/runtime error.
    - ``"builtin"``: always uses the staged builtin approach.
    """
    _EMPTY: dict = {
        "applied": False,
        "backend": "builtin",
        "sub_delay_ms": 0.0,
        "sub_polarity_invert": False,
        "sub_gain_trim_db": 0.0,
        "recommended_hz": None,
        "recommended_sub_lpf_hz": None,
        "allpass_enabled": False,
        "allpass_freq_hz": 0.0,
        "allpass_q": 0.707,
        "baseline": {},
        "optimized": {},
        "improvement_score": 0.0,
        "reason": "No meaningful improvement found.",
        "study_trials": 0,
        "allpass_reason": "No meaningful improvement found.",
    }
    if bundle is None:
        return dict(_EMPTY)

    backend_key = str(data.get("bass_integration_prepare_backend", "auto") or "auto").strip().lower()
    _mode_u = str(data.get("mode", "BASIC") or "BASIC").strip().upper()
    _auto_active = bool(_mode_u == "AUTO" or data.get("camillafir_automatic_mode", False))

    profile = str(data.get("bass_integration_profile", "safe") or "safe")
    xo_order, sub_hpf_hz, sub_hpf_order = _direct_dac_filter_params(data)
    sub_combine_mode = str(data.get("bass_integration_sub_combine_mode", "average") or "average")
    allpass_auto_enable = bool(data.get("bass_integration_allpass_auto_enable", False))

    use_optuna = False
    if backend_key == "optuna":
        use_optuna = True
    elif backend_key == "auto" and _auto_active:
        try:
            import optuna  # noqa: F401
            use_optuna = True
        except ImportError:
            use_optuna = False

    if use_optuna:
        try:
            from ...dsp.bass_integration import recommend_direct_dac_prepare_optuna  # noqa: PLC0415
            return dict(
                recommend_direct_dac_prepare_optuna(
                    bundle,
                    profile=profile,
                    main_hpf_order=int(xo_order),
                    sub_lpf_order=int(xo_order),
                    sub_hpf_hz=float(sub_hpf_hz),
                    sub_hpf_order=int(sub_hpf_order),
                    sub_combine_mode=sub_combine_mode,
                    allpass_auto_enable=allpass_auto_enable,
                    callbacks=callbacks,
                )
            )
        except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
            logger.debug("Direct-DAC prepare Optuna failed, falling back to builtin", exc_info=True)

    try:
        from ...dsp.bass_integration import _recommend_direct_dac_prepare_builtin_core  # noqa: PLC0415
        return dict(
            _recommend_direct_dac_prepare_builtin_core(
                bundle,
                profile=profile,
                main_hpf_order=int(xo_order),
                sub_lpf_order=int(xo_order),
                sub_hpf_hz=float(sub_hpf_hz),
                sub_hpf_order=int(sub_hpf_order),
                sub_combine_mode=sub_combine_mode,
                allpass_auto_enable=allpass_auto_enable,
                callbacks=callbacks,
            )
        )
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
        logger.debug("Direct-DAC prepare builtin recommendation failed", exc_info=True)
        return dict(_EMPTY)

def _compute_selected_bass_integration_diagnostics(bundle: object, data: dict) -> dict:
    try:
        if bundle is None:
            return {}
        profile = str(data.get("bass_integration_profile", "safe") or "safe")
        fc_hz = float(data.get("avr_crossover_hz", 80.0) or 80.0)
        if not math.isfinite(fc_hz) or fc_hz <= 0.0:
            fc_hz = 80.0
        from ...dsp.bass_integration import compute_final_bass_integration_metrics  # noqa: PLC0415

        xo_order, sub_hpf_hz, sub_hpf_order = _direct_dac_filter_params(data)
        sub_delay_ms, sub_polarity_invert, sub_gain_trim_db = _direct_dac_alignment_params(data)
        sub_allpass_freq_hz = None
        sub_allpass_q = None
        if bool(data.get("bass_integration_allpass_auto_applied", False)):
            try:
                sub_allpass_freq_hz = float(data.get("bass_integration_allpass_freq_hz", 0.0) or 0.0)
            except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
                sub_allpass_freq_hz = 0.0
            try:
                sub_allpass_q = float(data.get("bass_integration_allpass_q", 0.707) or 0.707)
            except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
                sub_allpass_q = 0.707
        try:
            _slpf: float | None = float(data.get("direct_dac_sub_lpf_hz") or fc_hz)
            if not math.isfinite(_slpf) or _slpf < fc_hz:
                _slpf = None
        except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
            _slpf = None
        metrics = dict(
            compute_final_bass_integration_metrics(
                bundle,
                float(fc_hz),
                profile,
                mode="direct_dac",
                main_hpf_order=int(xo_order),
                sub_lpf_order=int(xo_order),
                sub_hpf_hz=float(sub_hpf_hz),
                sub_hpf_order=int(sub_hpf_order),
                sub_combine_mode=str(data.get("bass_integration_sub_combine_mode", "average") or "average"),
                sub_delay_ms=float(sub_delay_ms),
                sub_polarity_invert=bool(sub_polarity_invert),
                sub_gain_trim_db=float(sub_gain_trim_db),
                sub_lpf_hz=_slpf,
                sub_allpass_freq_hz=sub_allpass_freq_hz,
                sub_allpass_q=sub_allpass_q,
            )
            or {}
        )
        out = dict(metrics.get("diagnostics", {}) or {})
        out.update(
            {
                "direct_dac_main_hpf_order": int(xo_order),
                "direct_dac_sub_lpf_order": int(xo_order),
                "direct_dac_sub_hpf_hz": float(sub_hpf_hz),
                "direct_dac_sub_hpf_order": int(sub_hpf_order),
                "direct_dac_sub_allpass_enabled": bool(metrics.get("bass_allpass_enabled", False)),
                "direct_dac_sub_allpass_freq_hz": float(metrics.get("bass_allpass_freq_hz", 0.0) or 0.0),
                "direct_dac_sub_allpass_q": float(metrics.get("bass_allpass_q", 0.707) or 0.707),
                "bass_integration_sub_delay_ms": float(sub_delay_ms),
                "bass_integration_sub_polarity_invert": bool(sub_polarity_invert),
                "bass_integration_sub_gain_trim_db": float(sub_gain_trim_db),
                "objective": float(metrics.get("objective", float("nan"))),
            }
        )
        return out
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
        logger.debug("Bass Integration diagnostics refresh failed", exc_info=True)
        return dict(getattr(bundle, "diagnostics", {}) or {}) if bundle is not None else {}


__all__ = ['_status', '_direct_dac_filter_params', '_direct_dac_alignment_params', '_compute_direct_dac_prepare_recommendation', '_compute_selected_bass_integration_diagnostics']


def _link_sibling_exports() -> None:
    import importlib
    package = __package__
    for module_name in ['bass_diagnostics', 'measurements', 'target_context']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_link_sibling_exports()
