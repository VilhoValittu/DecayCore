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


logger = logging.getLogger("DecayCore")

from ..measurements_wav import (
    detect_coherent_anchor_sample_from_wav_bytes,
    detect_coherent_anchor_sample_from_wav_path,
    parse_coherent_transfer_from_wav_bytes,
    parse_coherent_transfer_from_wav_path,
)

def _clean_local_path(p) -> str:
    """Normalisoi kayttajan antaman paikallisen tiedostopolun merkkijonoksi."""
    try:
        return str(p or "").strip().strip('"').strip("'")
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
        return ""

def _get_uploaded_file(data: dict, key: str):
    """Palauttaa upload-dictionaryn datasta tai None."""
    try:
        v = data.get(key)
        if isinstance(v, dict) and v.get("content") is not None:
            return v
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
        logger.exception("uploaded file extract")
    return None

def _get_local_path(data: dict, key: str) -> str:
    """Palauttaa paikallisen tiedostopolun datasta tai tyhjän merkkijonon."""
    return _clean_local_path(data.get(key, ""))

def _get_wav_window_params(data: dict) -> tuple[float, float, int]:
    try:
        pre_ms = float(data.get("ir_window_left", 85.0) or 85.0)
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
        pre_ms = 10.0
    try:
        post_ms = float(data.get("ir_window_right", data.get("ir_window", 500.0)) or 500.0)
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
        post_ms = 500.0
    try:
        sl = int(data.get("smoothing_level", 0) or 0)
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
        sl = 0
    return float(pre_ms), float(post_ms), int(sl)

def _is_wav_upload(file_dict) -> bool:
    try:
        if not isinstance(file_dict, dict):
            return False
        name = str(file_dict.get("filename", "") or "").strip().lower()
        if name.endswith(".wav"):
            return True
        content = file_dict.get("content", None)
        return isinstance(content, (bytes, bytearray)) and len(content) >= 4 and content[:4] == b"RIFF"
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
        return False

def _load_coherent_transfer_slot(
    data: dict,
    *,
    file_key: str,
    path_key: str,
    label: str,
    pre_ms: float,
    post_ms: float,
    smoothing_level: int,
    anchor_sample: int | None,
    logger=None,
):
    up = _get_uploaded_file(data, file_key)
    if up is not None:
        if not _is_wav_upload(up):
            if logger:
                logger.error(f"Bass Integration requires WAV uploads: {label}")
            return None
        return parse_coherent_transfer_from_wav_bytes(
            up.get("content", b""),
            pre_ms=pre_ms,
            post_ms=post_ms,
            smoothing_level=smoothing_level,
            anchor_sample=anchor_sample,
            label=label,
            logger=logger,
        )

    lp = _get_local_path(data, path_key)
    if lp:
        if not str(lp).lower().endswith(".wav"):
            if logger:
                logger.error(f"Bass Integration requires WAV local files: {label}")
            return None
        return parse_coherent_transfer_from_wav_path(
            lp,
            pre_ms=pre_ms,
            post_ms=post_ms,
            smoothing_level=smoothing_level,
            anchor_sample=anchor_sample,
            label=label,
            logger=logger,
        )

    return None

def _detect_coherent_slot_anchor_sample(
    data: dict,
    *,
    file_key: str,
    path_key: str,
    logger=None,
) -> int | None:
    up = _get_uploaded_file(data, file_key)
    if up is not None:
        if not _is_wav_upload(up):
            return None
        return detect_coherent_anchor_sample_from_wav_bytes(
            up.get("content", b""),
            logger=logger,
        )

    lp = _get_local_path(data, path_key)
    if lp and str(lp).lower().endswith(".wav"):
        return detect_coherent_anchor_sample_from_wav_path(lp, logger=logger)

    return None

def _detect_shared_coherent_anchor_sample(data: dict, *, logger=None) -> int | None:
    peaks = []
    for file_key, path_key in (
        ("file_l_main", "local_path_l_main"),
        ("file_r_main", "local_path_r_main"),
        ("file_l_sub", "local_path_l_sub"),
        ("file_r_sub", "local_path_r_sub"),
    ):
        peak = _detect_coherent_slot_anchor_sample(
            data,
            file_key=file_key,
            path_key=path_key,
            logger=logger,
        )
        if peak is not None:
            peaks.append(int(peak))

    if not peaks:
        return None

    shared_anchor = int(round(sum(peaks) / float(len(peaks))))
    if logger:
        try:
            spread = int(max(peaks) - min(peaks))
            logger.info(f"Bass Integration shared WAV anchor sample: {shared_anchor} (spread {spread} samples)")
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
            logger.exception("anchor spread log")
    return shared_anchor


__all__ = [
    '_clean_local_path',
    '_get_uploaded_file',
    '_get_local_path',
    '_get_wav_window_params',
    '_is_wav_upload',
    '_load_coherent_transfer_slot',
    '_detect_coherent_slot_anchor_sample',
    '_detect_shared_coherent_anchor_sample',
]

