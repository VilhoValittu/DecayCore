# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

import numpy as np

_FULL_HC_FREQS = np.array([
    0.0,
    20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0,
    160.0, 200.0, 250.0, 400.0, 1000.0, 2000.0, 4000.0,
    8000.0, 16000.0, 20000.0,
])

_TOOLE_FREQS = np.array([
    0.0,
    20.0, 63.0, 100.0, 200.0, 400.0,
    1000.0, 2000.0, 4000.0, 10000.0, 20000.0,
])

_CINEMA_FREQS = np.array([
    0.0, 20.0, 2000.0, 4000.0, 8000.0, 16000.0, 20000.0,
])

_CURVE_MAGS: dict[str, np.ndarray] = {
    "Harman6": np.array([
        6.0,
        6.0, 5.9, 5.8, 5.6, 5.3, 4.9, 4.3, 3.5, 2.5,
        1.4, 0.4, 0.0, -0.5, -1.0, -1.8, -2.8,
        -4.0, -5.5, -6.0,
    ]),
    "Harman8": np.array([
        8.0,
        8.0, 7.9, 7.8, 7.6, 7.3, 6.9, 6.3, 5.5, 4.5,
        3.4, 1.4, 0.0, -0.5, -1.0, -1.8, -2.8,
        -4.0, -5.5, -6.0,
    ]),
    "Harman4": np.array([
        4.0,
        4.0, 3.9, 3.8, 3.6, 3.3, 2.9, 2.3, 1.5, 0.8,
        0.2, 0.0, 0.0, -0.3, -0.6, -1.2, -2.0,
        -3.0, -4.5, -5.0,
    ]),
    "Harman10": np.array([
        10.0,
        10.0, 9.8, 9.5, 9.0, 8.2, 7.2, 6.0, 4.8, 3.5,
        2.2, 0.8, 0.0, -0.5, -1.0, -1.8, -2.8,
        -4.0, -5.5, -6.0,
    ]),
    "Harman12": np.array([
        12.0,
        12.0, 11.8, 11.4, 10.8, 9.8, 8.6, 7.2, 5.8, 4.2,
        2.8, 1.0, 0.0, -0.5, -1.0, -1.8, -2.8,
        -4.0, -5.5, -6.0,
    ]),
    "BK_Light": np.array([
        2.5,
        2.5, 2.4, 2.3, 2.1, 1.9, 1.6, 1.3, 1.0, 0.7,
        0.4, 0.1, -0.2, -0.6, -1.0, -1.6, -2.3,
        -3.1, -4.0, -4.4,
    ]),
    "BK_Strong": np.array([
        4.5,
        4.5, 4.4, 4.2, 3.9, 3.5, 3.0, 2.4, 1.8, 1.2,
        0.8, 0.3, -0.2, -0.8, -1.5, -2.3, -3.2,
        -4.3, -5.4, -6.0,
    ]),
    "BK_Medium": np.array([
        3.5,
        3.5, 3.4, 3.2, 3.0, 2.7, 2.3, 1.9, 1.4, 1.0,
        0.6, 0.2, -0.2, -0.7, -1.3, -2.0, -2.8,
        -3.8, -4.8, -5.3,
    ]),
    "Studio": np.array([
        3.0,
        3.0, 2.6, 2.2, 1.8, 1.4, 1.0, 0.6, 0.2, 0.0,
        -0.4, -0.8, -1.2, -1.8, -2.4, -3.0, -3.8,
        -4.8, -6.0, -6.5,
    ]),
    "Nearfield": np.array([
        2.5,
        2.5, 2.4, 2.2, 2.0, 1.8, 1.4, 1.0, 0.6, 0.2,
        0.0, 0.0, 0.0, -0.2, -0.5, -1.0, -1.8,
        -3.0, -4.5, -5.0,
    ]),
    "HiFi": np.array([
        6.0,
        6.0, 5.8, 5.5, 5.0, 4.3, 3.5, 2.6, 1.8, 1.0,
        0.4, 0.0, -0.2, -0.6, -1.0, -1.6, -2.6,
        -3.6, -5.0, -5.5,
    ]),
    "Speech": np.array([
        -2.0,
        -2.0, -1.8, -1.5, -1.2, -1.0, -0.6, -0.2, 0.4, 1.0,
        1.5, 1.8, 2.0, 2.0, 1.0, 0.0, -1.5,
        -3.5, -6.0, -8.0,
    ]),
    "Flat": np.zeros_like(_FULL_HC_FREQS),
    "Adaptive": np.zeros_like(_FULL_HC_FREQS),
}

_TOOLE_MAGS = np.array([
    2.5,
    2.5, 2.0, 1.5, 1.0, 0.5,
    0.0, -1.0, -2.0, -4.0, -6.0,
])

_CINEMA_MAGS = np.array([
    0.0, 0.0, 0.0, -3.0, -9.0, -15.0, -18.0,
])


def _normalize_hc_mode_key(v) -> str:
    """Sisainen apufunktio: normalize hc mode key."""
    try:
        s = str(v or "")
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
        s = ""

    known = set(_CURVE_MAGS.keys()) | {"Toole", "Cinema", "Custom", "Upload"}
    if s in known:
        return s

    n = s.lower().replace(" ", "")
    contains = lambda *tags: any(tag in n for tag in tags)
    alias = _normalize_hc_direct_alias(n, contains=contains)
    if alias is not None:
        return str(alias)
    bk = _normalize_hc_bk_alias(n, contains=contains)
    if bk is not None:
        return str(bk)
    harman = _normalize_hc_harman_alias(n, contains=contains)
    if harman is not None:
        return str(harman)
    return "Harman6"


def _normalize_hc_direct_alias(n: str, *, contains) -> str | None:
    direct_aliases = [
        (("adaptive",), "Adaptive"),
        (("custom", "lataa", "upload"), "Upload"),
        (("cinema",), "Cinema"),
        (("flat",), "Flat"),
        (("toole",), "Toole"),
        (("speech", "broadcast"), "Speech"),
        (("nearfield", "desk"), "Nearfield"),
        (("hifi", "loudness"), "HiFi"),
        (("studio", "tilt"), "Studio"),
    ]
    for tags, key in direct_aliases:
        if contains(*tags):
            return str(key)
    return None


def _normalize_hc_bk_alias(n: str, *, contains) -> str | None:
    _ = n
    if not contains("b&k", "bk"):
        return None
    if contains("light"):
        return "BK_Light"
    if contains("strong"):
        return "BK_Strong"
    return "BK_Medium"


def _normalize_hc_harman_alias(n: str, *, contains) -> str | None:
    if "harman" not in str(n):
        return None
    if contains("+12db", "12db"):
        return "Harman12"
    if contains("+10db", "10db", "subheavy"):
        return "Harman10"
    if contains("+8db", "8db"):
        return "Harman8"
    if contains("+4db", "4db"):
        return "Harman4"
    return "Harman6"


def get_house_curve_by_name(name):
    key = _normalize_hc_mode_key(name)
    if key == "Toole":
        return np.asarray(_TOOLE_FREQS, dtype=float), np.asarray(_TOOLE_MAGS, dtype=float)
    if key == "Cinema":
        return np.asarray(_CINEMA_FREQS, dtype=float), np.asarray(_CINEMA_MAGS, dtype=float)
    mags = _CURVE_MAGS.get(str(key), _CURVE_MAGS["Harman6"])
    return np.asarray(_FULL_HC_FREQS, dtype=float), np.asarray(mags, dtype=float)


def make_parametric_house_curve(
    bass_shelf_db: float,
    mid_lean_db_per_oct: float = 0.0,
    treble_rolloff_db_per_oct: float = 0.0,
    freqs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    full_freqs = np.array([
        0.0,
        20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0,
        160.0, 200.0, 250.0, 400.0, 1000.0, 2000.0, 4000.0,
        8000.0, 16000.0, 20000.0
    ])

    if freqs is None:
        freqs = full_freqs
    else:
        freqs = np.asarray(freqs, dtype=float)

    mags = np.zeros_like(freqs, dtype=float)

    bass_start_hz = 20.0
    bass_end_hz = 250.0
    mid_end_hz = 2000.0
    treble_start_hz = 2000.0

    for i, f in enumerate(freqs):
        if f <= 0.0:
            continue

        bass_contrib = 0.0
        if f <= bass_end_hz:
            if f < bass_start_hz:
                bass_contrib = bass_shelf_db
            else:
                frac = (f - bass_start_hz) / (bass_end_hz - bass_start_hz)
                bass_contrib = bass_shelf_db * (1.0 - frac)

        mid_contrib = 0.0
        if bass_end_hz < f < mid_end_hz:
            octaves_from_pivot = np.log2(f / bass_end_hz)
            mid_contrib = mid_lean_db_per_oct * octaves_from_pivot

        treble_contrib = 0.0
        if f > treble_start_hz:
            octaves_from_pivot = np.log2(f / treble_start_hz)
            treble_contrib = treble_rolloff_db_per_oct * octaves_from_pivot

        mags[i] = bass_contrib + mid_contrib + treble_contrib

    return freqs, mags


def adapt_house_curve_to_rt60(
    hc_f: np.ndarray,
    hc_m: np.ndarray,
    rt60_lf_s: float,
) -> np.ndarray:
    hc_m = np.asarray(hc_m, dtype=float).copy()

    if not np.isfinite(rt60_lf_s) or rt60_lf_s <= 0.0:
        return hc_m

    bass_reduction_db = float(np.clip((rt60_lf_s - 0.6) / 0.4, 0.0, 1.0) * 1.5)

    if bass_reduction_db < 0.01:
        return hc_m

    bass_floor_hz = 80.0
    bass_transition_hz = 250.0

    reduction_mask = np.ones_like(hc_f, dtype=float)
    reduction_mask[hc_f >= bass_transition_hz] = 0.0

    for i, f in enumerate(hc_f):
        if bass_floor_hz < f < bass_transition_hz:
            frac = (f - bass_floor_hz) / (bass_transition_hz - bass_floor_hz)
            cos_fade = 0.5 - 0.5 * np.cos(np.pi * frac)
            reduction_mask[i] = cos_fade

    hc_m = hc_m - bass_reduction_db * reduction_mask
    return hc_m
