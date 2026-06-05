# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Application-level health checks (DSP readiness validation)."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Literal, Optional

from ..resources.i8n.decaycore_i18n import t

Level = Literal["ok", "warn", "crit"]


@dataclass(frozen=True)
class Issue:
    level: Level
    title: str
    detail: str = ""


@dataclass(frozen=True)
class HealthResult:
    overall: Level
    blocked: bool
    issues: List[Issue]


def _tr(key: str, **kwargs: Any) -> str:
    text = t(key)
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError, TypeError, ValueError):
            return text
    return text


def _as_float(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_int(v: Any) -> Optional[int]:
    try:
        if v is None or v == "":
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def _clean_path_text(p: Any) -> str:
    if not isinstance(p, str):
        return ""
    return p.strip().strip('"').strip("'")


def _has_path_text(p: Any) -> bool:
    return bool(_clean_path_text(p))


def _valid_path(p: Any) -> bool:
    path = _clean_path_text(p)
    if not path:
        return False
    return os.path.isfile(path)


def _has_uploaded_file_name(v: Any) -> bool:
    try:
        if isinstance(v, dict):
            for key in ("filename", "name", "file_name"):
                name = str(v.get(key, "") or "").strip()
                if name:
                    return True
        if isinstance(v, list):
            return any(_has_uploaded_file_name(x) for x in v)
    except (TypeError, AttributeError):
        return False
    return False


def _uploaded_file_name(v: Any) -> str:
    try:
        if isinstance(v, dict):
            for key in ("filename", "name", "file_name"):
                name = str(v.get(key, "") or "").strip()
                if name:
                    return name
    except (TypeError, AttributeError):
        return ""
    return ""


def _has_uploaded_file(v: Any) -> bool:
    """Sisainen apufunktio: has uploaded file."""
    try:
        if isinstance(v, dict):
            if not _has_uploaded_file_name(v):
                return False
            content = v.get("content", None)
            if content is None:
                return False
            if isinstance(content, (bytes, bytearray, str)):
                return len(content) > 0
            return True
        if isinstance(v, list):
            return any(_has_uploaded_file(x) for x in v)
    except (TypeError, AttributeError):
        return False
    return False


def _has_wav_measurement_source(data: Dict[str, Any], *, file_key: str, path_key: str) -> bool:
    if _has_uploaded_file(data.get(file_key, None)):
        upload = data.get(file_key, None)
        name = _uploaded_file_name(upload).lower()
        content = (upload or {}).get("content", None) if isinstance(upload, dict) else None
        riff = isinstance(content, (bytes, bytearray)) and len(content) >= 4 and content[:4] == b"RIFF"
        return bool(name.endswith(".wav") or riff)
    if _has_path_text(data.get(path_key, None)):
        path = _clean_path_text(data.get(path_key, None))
        return bool(os.path.isfile(path) and path.lower().endswith(".wav"))
    return False


def _has_any_measurement_source(data: Dict[str, Any], *, file_key: str, path_key: str) -> bool:
    return bool(
        _has_uploaded_file_name(data.get(file_key, None))
        or _has_uploaded_file(data.get(file_key, None))
        or _has_path_text(data.get(path_key, None))
    )




def _health_bass_integration_issues(data: Dict[str, Any], mode_u: str) -> List[Issue]:
    issues: List[Issue] = []
    bi_mode = str(data.get("bass_integration_mode", "direct_dac") or "direct_dac")
    is_direct_dac = bi_mode == "direct_dac"

    if mode_u != "AUTO":
        issues.append(Issue("crit", _tr("health_bass_integration_auto_only")))

    bi_required_slots = [
        ("file_l_main", "local_path_l_main"),
        ("file_r_main", "local_path_r_main"),
        ("file_l_sub", "local_path_l_sub"),
    ]
    bi_optional_slots = [("file_r_sub", "local_path_r_sub")] if is_direct_dac else []
    if not is_direct_dac:
        bi_required_slots.append(("file_r_sub", "local_path_r_sub"))

    required_all_wav = all(
        _has_wav_measurement_source(data, file_key=file_key, path_key=path_key)
        for file_key, path_key in bi_required_slots
    )
    optional_valid_or_missing = all(
        (not _has_any_measurement_source(data, file_key=file_key, path_key=path_key))
        or _has_wav_measurement_source(data, file_key=file_key, path_key=path_key)
        for file_key, path_key in bi_optional_slots
    )
    any_source = any(
        _has_any_measurement_source(data, file_key=file_key, path_key=path_key)
        for file_key, path_key in [*bi_required_slots, *bi_optional_slots]
    )
    if required_all_wav and optional_valid_or_missing:
        issues.append(Issue("ok", _tr("health_bass_integration_measurements"), _tr("health_upload_source_provided")))
    elif any_source:
        issues.append(
            Issue(
                "crit",
                _tr("health_bass_integration_measurements"),
                _tr("health_bass_integration_wav_only_missing"),
            )
        )
    else:
        issues.append(
            Issue(
                "crit",
                _tr("health_bass_integration_measurements"),
                _tr("health_bass_integration_missing"),
            )
        )

    avr_fc = _as_float(data.get("avr_crossover_hz", None))
    if avr_fc is None or avr_fc <= 0.0:
        issues.append(Issue("crit", _tr("health_main_hpf_invalid")))
    elif avr_fc < 30.0 or avr_fc > 250.0:
        issues.append(Issue("warn", _tr("health_main_hpf_unusual"), f"{avr_fc:.1f} Hz"))
    else:
        issues.append(Issue("ok", _tr("health_main_hpf"), f"{avr_fc:.1f} Hz"))

    issues.append(
        Issue(
            "ok",
            _tr("health_bass_integration_playback_match_title"),
            _tr("health_bass_integration_playback_match"),
        )
    )
    return issues


def _health_standard_measurement_issues(data: Dict[str, Any]) -> List[Issue]:
    raw_up_l = _has_uploaded_file_name(data.get("file_l", None))
    raw_up_r = _has_uploaded_file_name(data.get("file_r", None))
    up_l = _has_uploaded_file(data.get("file_l", None))
    up_r = _has_uploaded_file(data.get("file_r", None))
    raw_lp_l = _has_path_text(data.get("local_path_l", None))
    raw_lp_r = _has_path_text(data.get("local_path_r", None))
    lp_l = _valid_path(data.get("local_path_l", None))
    lp_r = _valid_path(data.get("local_path_r", None))

    has_upload_pair = bool(up_l and up_r)
    has_local_pair = bool(lp_l and lp_r)

    if has_upload_pair or has_local_pair:
        return [Issue("ok", _tr("health_measurements"), _tr("health_upload_source_provided"))]
    if raw_up_l or raw_up_r or raw_lp_l or raw_lp_r:
        return [Issue("warn", _tr("health_measurements"), _tr("health_measurements_missing_pair"))]
    return [Issue("warn", _tr("health_measurements"), _tr("health_measurements_missing"))]


def _health_target_curve_issues(data: Dict[str, Any]) -> List[Issue]:
    issues: List[Issue] = []
    hc_mode = str(data.get("hc_mode") or "").strip()
    if hc_mode.lower() == "upload":
        upload_ok = _has_uploaded_file(data.get("hc_custom_file", None))
        local_ok = _valid_path(data.get("local_path_house", None))
        if upload_ok or local_ok:
            issues.append(Issue("ok", _tr("health_target_curve"), _tr("health_upload_source_provided")))
        else:
            issues.append(
                Issue(
                    "warn",
                    _tr("health_target_curve_upload_selected"),
                    _tr("health_target_curve_upload_missing"),
                )
            )
    elif hc_mode:
        issues.append(Issue("ok", _tr("health_target_curve"), hc_mode))
    else:
        issues.append(Issue("warn", _tr("health_target_curve"), _tr("health_not_set")))
    return issues


def _health_correction_issues(data: Dict[str, Any]) -> List[Issue]:
    issues: List[Issue] = []
    mag_on = bool(data.get("mag_correct", True))
    fmin = _as_float(data.get("mag_c_min", None))
    fmax = _as_float(data.get("mag_c_max", None))
    if not mag_on:
        issues.append(Issue("ok", _tr("health_magnitude_correction"), _tr("health_disabled")))
    elif (fmin is not None) and (fmax is not None):
        if fmin >= fmax:
            issues.append(Issue("crit", _tr("health_correction_range_invalid")))
        else:
            issues.append(Issue("ok", _tr("health_correction_range"), f"{fmin:.0f}-{fmax:.0f} Hz"))
            if fmax > 350.0:
                issues.append(Issue("warn", _tr("health_correction_max_very_high")))
    else:
        issues.append(Issue("warn", _tr("health_correction_range_set_bounds")))
    return issues


def _health_engine_issues(data: Dict[str, Any]) -> List[Issue]:
    issues: List[Issue] = []
    fs = _as_int(data.get("fs", None))
    taps = _as_int(data.get("taps", None))
    if fs and taps and fs > 0 and taps > 0:
        latency_ms = (taps / 2.0) / float(fs) * 1000.0
        bin_hz = float(fs) / float(taps)
        ir_window_left_ms = _as_float(data.get("ir_window_left", data.get("ir_window", None)))
        short_left_window = (ir_window_left_ms is not None) and (ir_window_left_ms < 120.0)
        ftype = str(data.get("filter_type") or "").lower()
        is_min = "min" in ftype
        is_asym = "asym" in ftype

        if is_min or is_asym:
            issues.append(Issue("ok", _tr("health_latency"), _tr("health_low_latency_mode")))
        else:
            issues.append(Issue("ok", _tr("health_latency"), f"{latency_ms:.0f} ms"))
        issues.append(Issue("ok", _tr("health_resolution"), f"{bin_hz:.2f} Hz/bin"))

        if (not is_min and not is_asym) and latency_ms > 150 and not short_left_window:
            issues.append(Issue("warn", _tr("health_taps_count_high")))
    else:
        issues.append(Issue("warn", _tr("health_engine_metrics_set")))
    return issues


def _health_leveling_issues(data: Dict[str, Any]) -> List[Issue]:
    issues: List[Issue] = []
    lvl_min = _as_float(data.get("lvl_min", None))
    lvl_max = _as_float(data.get("lvl_max", None))
    if (lvl_min is not None) and (lvl_max is not None):
        if lvl_min >= lvl_max:
            issues.append(Issue("crit", _tr("health_leveling_range_invalid")))
        elif (lvl_max - lvl_min) < 200.0:
            issues.append(Issue("warn", _tr("health_leveling_range_narrow")))
    return issues


def _health_boost_phase_issues(data: Dict[str, Any]) -> List[Issue]:
    issues: List[Issue] = []
    max_boost = _as_float(data.get("max_boost", None))
    if max_boost is None:
        issues.append(Issue("ok", _tr("health_max_boost"), _tr("health_default")))
    else:
        issues.append(Issue("ok", _tr("health_max_boost"), f"{max_boost:.1f} dB"))
        if max_boost > 10.0:
            issues.append(Issue("warn", _tr("health_max_boost_high")))

    phase_limit = _as_float(data.get("phase_limit", None))
    if (phase_limit is not None) and (phase_limit > 800.0):
        issues.append(Issue("warn", _tr("health_phase_limit_high"), f"{phase_limit:.0f} Hz"))
    return issues


def _health_protection_issues(data: Dict[str, Any]) -> List[Issue]:
    issues: List[Issue] = []
    exc_on = bool(data.get("exc_prot", False))
    if not exc_on:
        issues.append(Issue("warn", _tr("health_excursion_protection_off")))
    else:
        issues.append(Issue("ok", _tr("health_excursion_protection"), _tr("health_enabled")))

    hpf_on = bool(data.get("hpf_enable", False))
    if hpf_on:
        fs = _as_int(data.get("fs", None))
        hpf_f = _as_float(data.get("hpf_freq", None))
        hpf_s = _as_float(data.get("hpf_slope", None))
        if (hpf_f is None) or (hpf_f <= 0.0):
            issues.append(Issue("crit", _tr("health_hpf_freq_invalid")))
        elif (hpf_f is not None) and (fs is not None) and (fs > 0) and (hpf_f >= 0.45 * fs):
            issues.append(Issue("warn", _tr("health_hpf_freq_high")))

        if (hpf_s is None) or (hpf_s <= 0.0):
            issues.append(Issue("crit", _tr("health_hpf_slope_invalid")))
    return issues


def _health_mixed_issues(data: Dict[str, Any]) -> List[Issue]:
    issues: List[Issue] = []
    ftype = str(data.get("filter_type") or "").lower()
    if "mixed" in ftype:
        fs = _as_int(data.get("fs", None))
        mixed_f = _as_float(data.get("mixed_freq", None))
        trans_w = _as_float(data.get("trans_width", None))

        if (mixed_f is None) or (mixed_f <= 0.0):
            issues.append(Issue("crit", _tr("health_mixed_split_invalid")))
        else:
            if mixed_f < 40.0:
                issues.append(Issue("warn", _tr("health_mixed_split_low")))
            if (fs is not None) and (fs > 0) and (mixed_f > 0.45 * fs):
                issues.append(Issue("warn", _tr("health_mixed_split_high")))

        if (trans_w is not None) and (trans_w < 0.0):
            issues.append(Issue("crit", _tr("health_transition_width_nonnegative")))
        if (mixed_f is not None) and (mixed_f > 0.0) and (trans_w is not None) and (trans_w > mixed_f):
            issues.append(Issue("warn", _tr("health_transition_width_wider")))
    return issues


def compute_health(data: Dict[str, Any], mode: str) -> HealthResult:
    issues: List[Issue] = []
    mode_u = str(mode or "BASIC").strip().upper()
    bass_integration_on = bool(data.get("bass_integration_enable", False))

    if bass_integration_on:
        issues.extend(_health_bass_integration_issues(data, mode_u))
    else:
        issues.extend(_health_standard_measurement_issues(data))

    issues.extend(_health_target_curve_issues(data))
    issues.extend(_health_correction_issues(data))
    issues.extend(_health_engine_issues(data))
    issues.extend(_health_leveling_issues(data))
    issues.extend(_health_boost_phase_issues(data))
    issues.extend(_health_protection_issues(data))
    issues.extend(_health_mixed_issues(data))

    has_crit = any(i.level == "crit" for i in issues)
    has_warn = any(i.level == "warn" for i in issues)
    overall: Level = "crit" if has_crit else ("warn" if has_warn else "ok")

    blocked = (mode_u in ("BASIC", "AUTO")) and has_crit
    return HealthResult(overall=overall, blocked=blocked, issues=issues)




def format_health_summary(hr: HealthResult, max_items: int = 3) -> str:
    """Jasentaa tai muotoilee: format health summary."""
    crits = [i for i in hr.issues if i.level == "crit"]
    warns = [i for i in hr.issues if i.level == "warn"]

    if crits:
        head = _tr("health_summary_errors", count=len(crits))
        items = crits[:max_items]
        lines = [head] + [f"- {i.title}" for i in items]
        if len(crits) > max_items:
            lines.append(f"- {_tr('health_summary_more', count=len(crits) - max_items)}")
        return "\n".join(lines)

    if warns:
        head = _tr("health_summary_warnings", count=len(warns))
        items = warns[:max_items]
        lines = [head] + [f"- {i.title}" for i in items]
        if len(warns) > max_items:
            lines.append(f"- {_tr('health_summary_more', count=len(warns) - max_items)}")
        return "\n".join(lines)

    return ""
