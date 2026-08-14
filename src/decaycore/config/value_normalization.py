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

from typing import Any, Callable

from ..resources.i8n.decaycore_i18n import TRANSLATIONS

LAYOUT_MONO = "mono"
LAYOUT_STEREO = "stereo"

LVL_MODE_AUTO = "auto"
LVL_MODE_MANUAL = "manual"

OUTPUT_TILT_SOURCE_OFF = "off"
OUTPUT_TILT_SOURCE_MANUAL_TARGET_TILT = "manual_target_tilt"

LVL_ALGO_MEDIAN = "median"
LVL_ALGO_AVERAGE = "average"

TDC_PRESET_SAFE = "safe"
TDC_PRESET_NORMAL = "normal"
TDC_PRESET_AGGRESSIVE = "aggressive"

AFDW_PRESET_TIGHT = "tight"
AFDW_PRESET_BALANCED = "balanced"
AFDW_PRESET_SAFE = "safe"
AFDW_PRESET_MINIMAL = "minimal"

FILTER_WAV_FORMAT_FLOAT32 = "FLOAT32"
FILTER_WAV_FORMAT_S32LE = "S32_LE"
FILTER_WAV_FORMAT_S16LE = "S16_LE"

DEVICE_AUDIO_FORMAT_S32LE = "S32_LE"
DEVICE_AUDIO_FORMAT_S16LE = "S16_LE"

BASS_INTEGRATION_PROFILES = ("safe", "normal", "assertive")

LAYOUT_OPTION_LABEL_KEYS = {
    LAYOUT_MONO: "layout_mono",
    LAYOUT_STEREO: "layout_stereo",
}

FILTER_WAV_FORMAT_OPTION_LABEL_KEYS = {
    FILTER_WAV_FORMAT_FLOAT32: "filter_wav_format_float32",
    FILTER_WAV_FORMAT_S32LE: "filter_wav_format_s32le",
    FILTER_WAV_FORMAT_S16LE: "filter_wav_format_s16le",
}

DEVICE_AUDIO_FORMAT_OPTION_LABEL_KEYS = {
    DEVICE_AUDIO_FORMAT_S32LE: "device_audio_format_s32le",
    DEVICE_AUDIO_FORMAT_S16LE: "device_audio_format_s16le",
}

LVL_MODE_OPTION_LABEL_KEYS = {
    LVL_MODE_AUTO: "lvl_mode_auto",
    LVL_MODE_MANUAL: "lvl_mode_manual",
}

OUTPUT_TILT_SOURCE_OPTION_LABEL_KEYS = {
    OUTPUT_TILT_SOURCE_OFF: "state_off",
    OUTPUT_TILT_SOURCE_MANUAL_TARGET_TILT: "output_tilt_use_manual_target_tilt",
}

LVL_ALGO_OPTION_LABEL_KEYS = {
    LVL_ALGO_MEDIAN: "lvl_algo_median",
    LVL_ALGO_AVERAGE: "lvl_algo_average",
}

TDC_PRESET_LABEL_KEYS = {
    TDC_PRESET_SAFE: "tdc_preset_safe",
    TDC_PRESET_NORMAL: "tdc_preset_normal",
    TDC_PRESET_AGGRESSIVE: "tdc_preset_aggressive",
}

AFDW_PRESET_LABEL_KEYS = {
    AFDW_PRESET_TIGHT: "afdw_preset_tight",
    AFDW_PRESET_BALANCED: "afdw_preset_balanced",
    AFDW_PRESET_SAFE: "afdw_preset_safe",
    AFDW_PRESET_MINIMAL: "afdw_preset_minimal",
}

_LAYOUT_LEGACY_LABELS = {
    LAYOUT_MONO: ("Mono",),
    LAYOUT_STEREO: ("Stereo",),
}

_LVL_MODE_LEGACY_LABELS = {
    LVL_MODE_AUTO: ("Auto",),
    LVL_MODE_MANUAL: ("Manual",),
}

_OUTPUT_TILT_SOURCE_LEGACY_LABELS = {
    OUTPUT_TILT_SOURCE_OFF: ("Off",),
    OUTPUT_TILT_SOURCE_MANUAL_TARGET_TILT: ("Use Manual Tilt value",),
}

_LVL_ALGO_LEGACY_LABELS = {
    LVL_ALGO_MEDIAN: ("Median",),
    LVL_ALGO_AVERAGE: ("Average",),
}

_TDC_PRESET_LEGACY_LABELS = {
    TDC_PRESET_SAFE: ("Safe",),
    TDC_PRESET_NORMAL: ("Normal",),
    TDC_PRESET_AGGRESSIVE: ("Aggressive",),
}

_AFDW_PRESET_LEGACY_LABELS = {
    AFDW_PRESET_TIGHT: ("Tight",),
    AFDW_PRESET_BALANCED: ("Balanced",),
    AFDW_PRESET_SAFE: ("Safe",),
    AFDW_PRESET_MINIMAL: ("Minimal",),
}


def tr_options(t: Callable[[str], str], spec: dict[str, str]) -> dict[str, str]:
    return {stable_key: t(label_key) for stable_key, label_key in spec.items()}


def normalize_bass_integration_profile(value: Any) -> str:
    profile = str(value or "safe").strip().lower()
    return profile if profile in BASS_INTEGRATION_PROFILES else "safe"


def normalize_sub_combine_mode(value: Any) -> str:
    mode = str(value or "average").strip().lower().replace("-", "_")
    aliases = {
        "avg": "average",
        "summed": "sum",
        "aligned": "aligned_sum",
        "alignedsum": "aligned_sum",
    }
    mode = str(aliases.get(mode, mode))
    return mode if mode in ("average", "sum", "aligned_sum") else "average"


def _as_text(value: Any) -> str:
    try:
        return str(value or "").strip()
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


def _translation_variants(label_key: str, t: Callable[[str], str] | None = None) -> set[str]:
    variants: set[str] = set()

    if callable(t):
        try:
            translated = _as_text(t(label_key))
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
            translated = ""
        if translated and translated != label_key:
            variants.add(translated)

    for catalog in TRANSLATIONS.values():
        try:
            translated = _as_text((catalog or {}).get(label_key))
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
            translated = ""
        if translated and translated != label_key:
            variants.add(translated)

    return variants


def _normalize_choice_value(
    value: Any,
    *,
    default: str | None,
    label_keys: dict[str, str],
    legacy_labels: dict[str, tuple[str, ...]],
    t: Callable[[str], str] | None = None,
    allow_none: bool = False,
) -> str | None:
    raw = _as_text(value)
    if not raw:
        return None if allow_none else default

    aliases: dict[str, str] = {}
    for stable_key, label_key in label_keys.items():
        aliases[_as_text(stable_key).casefold()] = stable_key
        for legacy in legacy_labels.get(stable_key, ()):
            aliases[_as_text(legacy).casefold()] = stable_key
        for translated in _translation_variants(label_key, t=t):
            aliases[_as_text(translated).casefold()] = stable_key

    return aliases.get(raw.casefold(), None if allow_none else default)


def normalize_layout_value(value: Any, t: Callable[[str], str] | None = None) -> str:
    return str(
        _normalize_choice_value(
            value,
            default=LAYOUT_MONO,
            label_keys=LAYOUT_OPTION_LABEL_KEYS,
            legacy_labels=_LAYOUT_LEGACY_LABELS,
            t=t,
        )
    )


def normalize_lvl_mode_value(value: Any, t: Callable[[str], str] | None = None) -> str:
    return str(
        _normalize_choice_value(
            value,
            default=LVL_MODE_AUTO,
            label_keys=LVL_MODE_OPTION_LABEL_KEYS,
            legacy_labels=_LVL_MODE_LEGACY_LABELS,
            t=t,
        )
    )


def normalize_output_tilt_source_value(value: Any, t: Callable[[str], str] | None = None) -> str:
    return str(
        _normalize_choice_value(
            value,
            default=OUTPUT_TILT_SOURCE_OFF,
            label_keys=OUTPUT_TILT_SOURCE_OPTION_LABEL_KEYS,
            legacy_labels=_OUTPUT_TILT_SOURCE_LEGACY_LABELS,
            t=t,
        )
    )


def normalize_lvl_algo_value(value: Any, t: Callable[[str], str] | None = None) -> str:
    return str(
        _normalize_choice_value(
            value,
            default=LVL_ALGO_MEDIAN,
            label_keys=LVL_ALGO_OPTION_LABEL_KEYS,
            legacy_labels=_LVL_ALGO_LEGACY_LABELS,
            t=t,
        )
    )


def normalize_tdc_preset_key(value: Any, t: Callable[[str], str] | None = None) -> str | None:
    return _normalize_choice_value(
        value,
        default=None,
        label_keys=TDC_PRESET_LABEL_KEYS,
        legacy_labels=_TDC_PRESET_LEGACY_LABELS,
        t=t,
        allow_none=True,
    )


def normalize_afdw_preset_key(value: Any, t: Callable[[str], str] | None = None) -> str | None:
    return _normalize_choice_value(
        value,
        default=None,
        label_keys=AFDW_PRESET_LABEL_KEYS,
        legacy_labels=_AFDW_PRESET_LEGACY_LABELS,
        t=t,
        allow_none=True,
    )


def layout_legacy_name(value: Any, t: Callable[[str], str] | None = None) -> str:
    return "Stereo" if normalize_layout_value(value, t=t) == LAYOUT_STEREO else "Mono"


def lvl_mode_legacy_name(value: Any, t: Callable[[str], str] | None = None) -> str:
    return "Manual" if normalize_lvl_mode_value(value, t=t) == LVL_MODE_MANUAL else "Auto"


def lvl_algo_legacy_name(value: Any, t: Callable[[str], str] | None = None) -> str:
    return "Average" if normalize_lvl_algo_value(value, t=t) == LVL_ALGO_AVERAGE else "Median"


__all__ = [
    "AFDW_PRESET_BALANCED",
    "AFDW_PRESET_LABEL_KEYS",
    "AFDW_PRESET_MINIMAL",
    "AFDW_PRESET_SAFE",
    "AFDW_PRESET_TIGHT",
    "BASS_INTEGRATION_PROFILES",
    "DEVICE_AUDIO_FORMAT_OPTION_LABEL_KEYS",
    "DEVICE_AUDIO_FORMAT_S16LE",
    "DEVICE_AUDIO_FORMAT_S32LE",
    "FILTER_WAV_FORMAT_FLOAT32",
    "FILTER_WAV_FORMAT_OPTION_LABEL_KEYS",
    "FILTER_WAV_FORMAT_S16LE",
    "FILTER_WAV_FORMAT_S32LE",
    "LAYOUT_MONO",
    "LAYOUT_OPTION_LABEL_KEYS",
    "LAYOUT_STEREO",
    "LVL_ALGO_AVERAGE",
    "LVL_ALGO_MEDIAN",
    "LVL_ALGO_OPTION_LABEL_KEYS",
    "LVL_MODE_AUTO",
    "LVL_MODE_MANUAL",
    "LVL_MODE_OPTION_LABEL_KEYS",
    "OUTPUT_TILT_SOURCE_MANUAL_TARGET_TILT",
    "OUTPUT_TILT_SOURCE_OFF",
    "OUTPUT_TILT_SOURCE_OPTION_LABEL_KEYS",
    "TDC_PRESET_AGGRESSIVE",
    "TDC_PRESET_LABEL_KEYS",
    "TDC_PRESET_NORMAL",
    "TDC_PRESET_SAFE",
    "layout_legacy_name",
    "lvl_algo_legacy_name",
    "lvl_mode_legacy_name",
    "normalize_afdw_preset_key",
    "normalize_bass_integration_profile",
    "normalize_layout_value",
    "normalize_lvl_algo_value",
    "normalize_lvl_mode_value",
    "normalize_output_tilt_source_value",
    "normalize_sub_combine_mode",
    "normalize_tdc_preset_key",
    "tr_options",
]
