# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

import logging
import math
import sys

from ...config.decaycore_pipeline import filter_type_supports_xo_phase_model

logger = logging.getLogger(__name__)
from ...auto_mode.rank_score import calibrated_auto_quality
from ..export_scoring import _safe_float

_AUTO_ASYM_PHASE1_SEARCH_SPACE_EST = 1877500016615829065655090169509480

_RECOVERABLE_SUMMARY_EXCEPTIONS = (
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
)


def _safe_float_or(value, default: float) -> float:
    try:
        return float(value)
    except _RECOVERABLE_SUMMARY_EXCEPTIONS:
        return float(default)


def _safe_int_or(value, default: int) -> int:
    try:
        return int(round(float(value)))
    except _RECOVERABLE_SUMMARY_EXCEPTIONS:
        return int(default)


def _polish_display_rank(polish: dict, raw_key: str, official_key: str) -> float:
    official = _safe_float(dict(polish or {}).get(official_key, float("nan")), float("nan"))
    if math.isfinite(official):
        return float(official)
    raw = _safe_float(dict(polish or {}).get(raw_key, float("nan")), float("nan"))
    return float(calibrated_auto_quality(raw))

def _format_recommended_xo_hz(value: float) -> str:
    hz = float(value)
    if math.isclose(hz, round(hz), abs_tol=1e-6):
        return f"{hz:.0f} Hz"
    return f"{hz:.1f} Hz"

def _auto_search_space_summary(data: dict | None) -> str:
    try:
        ft = str((data or {}).get("filter_type", "") or "").strip().lower()
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
        ft = ""

    if "asym" in ft:
        return (
            "Theoretical preset search space (discretized estimate, Asymmetric phase1): "
            f"~{_AUTO_ASYM_PHASE1_SEARCH_SPACE_EST:.3e} combinations. "
            "Automatic mode samples only a tiny subset and refines iteratively/cache-guided. "
            "Because the calculation count is still large, runtime depends strongly on computer performance."
        )
    if "linear" in ft:
        return (
            "Theoretical preset search space is extremely large. "
            "Automatic mode samples only a tiny subset and refines iteratively/cache-guided. "
            "Because the calculation count is still large, runtime depends strongly on computer performance."
        )
    if "mixed" in ft or "minimum" in ft or "min" in ft:
        return (
            "Theoretical preset search space is extremely large. "
            "Automatic mode samples only a tiny subset and refines iteratively/cache-guided. "
            "Because the calculation count is still large, runtime depends strongly on computer performance."
        )
    return (
        "Theoretical preset search space is extremely large. "
        "Automatic mode samples only a tiny subset and refines iteratively/cache-guided. "
        "Because the calculation count is still large, runtime depends strongly on computer performance."
    )

def _module_runtime_version(module_name: str) -> str:
    try:
        mod = __import__(str(module_name))
        ver = str(getattr(mod, "__version__", "") or "").strip()
        if ver:
            return ver
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
        logger.exception("module version import")
    try:
        from importlib.metadata import version

        ver = str(version(str(module_name)) or "").strip()
        if ver:
            return ver
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
        logger.exception("module version metadata")
    return "n/a"

def _runtime_versions_text() -> str:
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    parts = [
        f"Python {py_ver}",
        f"numpy {_module_runtime_version('numpy')}",
        f"scipy {_module_runtime_version('scipy')}",
        f"optuna {_module_runtime_version('optuna')}",
    ]
    return "Runtime: " + " | ".join(parts)


def _collect_xo_parts(ui_data: dict) -> list[str]:
    if not filter_type_supports_xo_phase_model(ui_data.get("filter_type", "")):
        return []
    xo_parts: list[str] = []
    for i in range(1, 6):
        f_raw = ui_data.get(f"xo{i}_f", None)
        if f_raw in (None, "", 0):
            continue
        freq_hz = _safe_float_or(f_raw, float("nan"))
        if not math.isfinite(freq_hz) or freq_hz <= 0.0:
            continue
        slope = _safe_int_or(ui_data.get(f"xo{i}_s", 12) or 12, 12)
        xo_parts.append(f"XO{i}: {freq_hz:.1f} Hz / {slope} dB/oct")
    return xo_parts


def _hpf_summary_line(ui_data: dict) -> str:
    if bool(ui_data.get("bass_integration_enable", False)):
        hpf_f = _safe_float_or(
            ui_data.get(
                "sub_crossover_hz",
                ui_data.get("avr_crossover_hz", 80.0),
            )
            or 80.0,
            80.0,
        )
        hpf_s = _safe_int_or(ui_data.get("sub_crossover_slope", 24) or 24, 24)
        return f"HPF: ON ({hpf_f:.1f} Hz / {hpf_s} dB/oct)\n"
    if bool(ui_data.get("hpf_enable", False)):
        hpf_f = _safe_float_or(ui_data.get("hpf_freq", 20.0) or 20.0, 20.0)
        hpf_s = _safe_int_or(ui_data.get("hpf_slope", 24) or 24, 24)
        return f"HPF: ON ({hpf_f:.1f} Hz / {hpf_s} dB/oct)\n"
    return "HPF: OFF\n"


def _append_main_speaker_xo_hpf_summary(summary_content: str, data: dict | None) -> str:
    try:
        ui_data = dict(data or {})
        summary_content += "\n=== MAIN SPEAKER XO / HPF PHASE MODEL ===\n"
        xo_parts = _collect_xo_parts(ui_data)
        if xo_parts:
            summary_content += "Crossovers: " + ", ".join(xo_parts) + "\n"
        else:
            summary_content += "Crossovers: OFF\n"

        summary_content += _hpf_summary_line(ui_data)
        summary_content += (
            "Note: this XO / HPF section models the main speaker crossover / HPF phase chain.\n"
        )
    except _RECOVERABLE_SUMMARY_EXCEPTIONS:
        logger.exception("XO/HPF summary section")
    return summary_content


__all__ = ['_polish_display_rank', '_format_recommended_xo_hz', '_auto_search_space_summary', '_module_runtime_version', '_runtime_versions_text', '_append_main_speaker_xo_hpf_summary']


def _link_sibling_exports() -> None:
    import importlib
    package = __package__
    for module_name in ['runtime', 'bass_integration', 'stereo_policy', 'dsp_effective', 'events']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_link_sibling_exports()
