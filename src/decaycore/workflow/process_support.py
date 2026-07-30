# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Helper utilities for process_run() wiring in decaycore.py.

These were previously defined inline in decaycore.py alongside
_process_run_legacy(). They belong to orchestration, not UI.
"""
from __future__ import annotations

import logging
import os
import re
import typing
import unicodedata

logger = logging.getLogger("DecayCore")

_RECOVERABLE_PROCESS_SUPPORT_EXCEPTIONS = (
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


def auto_target_mode_norm(mode: typing.Any) -> str:
    """Normalisoi AUTO-tilan target-kayran valintatavan.

    Palauttaa joko `auto` (etsi paras target-kayra),
    `adaptive` (johda mittauksista) tai `selected`
    (kayta kayttajan valitsemaa target-kayraa).
    """
    m = str(mode).strip().lower() if mode is not None else "auto"
    if m in ("selected", "manual", "fixed", "user"):
        return "selected"
    if m == "adaptive":
        return "adaptive"
    return "auto"


def auto_target_selection_method_text(method: typing.Any) -> str:
    """Muuntaa auto-target-valinnan metodit luettaviksi loki- ja UI-teksteiksi.
    """
    key = str(method).strip().lower() if method is not None else ""

    mapping = {
        "adaptive": "adaptive (synthesized from measurements)",
        "adaptive_guarded": "adaptive (confidence-weighted and safety-bounded)",
        "cache_measurement_hit": "cache hit (same measurements, no target re-evaluation)",
        "cache_optuna_target": "Optuna target study cache seed",
        "cache_optuna_target_hit": "cache hit (Optuna target study, no target re-evaluation)",
        "cache_signature_hit": "cache hit (same measurements + settings, no target re-evaluation)",
        "cache_measurement": "measurement cache seed",
        "cache_signature": "signature cache seed",
        "trial_with_cache_wildcard": "trial winner with cache wildcard",
        "top3x10_trials": "trial comparison",
        "top3x10_trials_rank_tie_composite": "trial comparison with rank tie-break",
        "fit_rms": "quick fit preselect",
    }
    return str(mapping.get(key, key or "unknown"))


def slugify_filename_token(value: typing.Any, *, default: str = "target", max_len: int = 48) -> str:
    """Muuntaa tekstin turvalliseksi tiedostonimiosaksi."""
    try:
        raw = str(value or "").strip()
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
        raw = ""
    if not raw:
        return default

    try:
        txt = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
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
        txt = raw
    txt = re.sub(r"[^A-Za-z0-9]+", "-", txt).strip("-").lower()
    if not txt:
        return default
    if len(txt) > int(max_len):
        txt = txt[: int(max_len)].rstrip("-")
    return txt or default


def _safe_data_str(data: dict, key: str) -> str:
    try:
        return str(data.get(key) or "").strip()
    except _RECOVERABLE_PROCESS_SUPPORT_EXCEPTIONS:
        return ""


def _target_label_from_upload(data: dict) -> str:
    try:
        uploaded = data.get("hc_custom_file")
    except _RECOVERABLE_PROCESS_SUPPORT_EXCEPTIONS:
        logger.exception("target curve label from upload")
        return ""
    if not isinstance(uploaded, dict):
        return ""
    for key in ("filename", "name", "file_name"):
        value = uploaded.get(key)
        if isinstance(value, str) and value.strip():
            return os.path.splitext(os.path.basename(value.strip()))[0]
    return ""


def _uploaded_file_has_name(uploaded: dict) -> bool:
    for key in ("filename", "name", "file_name"):
        value = _safe_data_str(uploaded, key)
        if value:
            return True
    return False


def _is_uploaded_file_obj(value: typing.Any) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        content = value.get("content", None)
    except _RECOVERABLE_PROCESS_SUPPORT_EXCEPTIONS:
        logger.exception("uploaded file content check")
        content = None
    if isinstance(content, (bytes, bytearray)) and len(content) > 0:
        return True
    return _uploaded_file_has_name(value)


def pick_target_curve_label(data: dict) -> str:
    """Valitsee target curven nimen vientitiedostonimiin."""
    uploaded_label = _target_label_from_upload(data)
    if uploaded_label:
        return uploaded_label

    local_path = _safe_data_str(data, "local_path_house")
    if local_path:
        return os.path.splitext(os.path.basename(local_path))[0]

    for key in ("hc_mode", "hc_source"):
        value = _safe_data_str(data, key)
        if value:
            return value
    return "Target"


def has_uploaded_target_file(data: dict) -> bool:
    """Checks if UI data contains an uploaded custom target file."""
    try:
        up = data.get("hc_custom_file")
    except _RECOVERABLE_PROCESS_SUPPORT_EXCEPTIONS:
        up = None

    if _is_uploaded_file_obj(up):
        return True
    if isinstance(up, list):
        for item in up:
            if _is_uploaded_file_obj(item):
                return True
    return False


def resolve_ui_stats_fs(ui_stats_fs: typing.Any, selected_fs: typing.Any) -> int:
    """Valitsee UI-statistiikalle oikean sample raten.

    Priorisoi dashboardille valitun analyysinopeuden (`ui_stats_fs`) ja
    kaatuu turvallisesti UI-valintaan (`selected_fs`) tai 44100 Hz:iin.
    """
    for cand in (ui_stats_fs, selected_fs, 44100):
        try:
            fs_i = int(cand)
            if fs_i > 0:
                return fs_i
        except (ValueError, TypeError):
            pass

    return 44100
