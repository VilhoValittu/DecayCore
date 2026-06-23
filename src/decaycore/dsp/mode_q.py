# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0
"""Jaettu Q-estimaattori spektripiikeille.

Q maaritellaan geometrisesti: piikin keskitaajuus jaettuna puoliprominenssi-
kaistanleveydella (`scipy.signal.peak_widths`, rel_height=0.5). Sama
maaritelma on kaytossa seka heijastusnodejen (decaycore_analysis) etta
bassfirst-moodipainotuksen Q-laskennassa.

Huom: modal_analysis_parts/mode_detection.py kayttaa tarkoituksella eri
maaritelmaa (leveys kiintealla absoluuttisella tasolla min_peak_db * 0.5,
koska sama leveys syottaa myos pinta-alalaskennan) - sita ei yhtenaisteta
tahan.
"""

from __future__ import annotations

import numpy as np
import scipy.signal

__all__ = ["estimate_peak_q"]


def estimate_peak_q(
    freqs: np.ndarray,
    values: np.ndarray,
    peak_indices: np.ndarray,
    *,
    min_bw_ratio: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Laskee Q-arvot ja kaistanleveydet annetuille piikeille.

    `peak_indices` tulee olla `scipy.signal.find_peaks`-tuloksia samasta
    `values`-kayrasta. Kaistanleveys mitataan puoliprominenssitasolla
    interpoloiduista reunoista, ja sille asetetaan lattia
    `min_bw_ratio * f_peak` (estaa rajattomat Q-arvot yhden binin piikeille).

    Palauttaa `(q_values, bw_hz)` samassa jarjestyksessa kuin `peak_indices`.
    """
    f = np.asarray(freqs, dtype=float)
    v = np.asarray(values, dtype=float)
    peaks = np.asarray(peak_indices, dtype=int).reshape(-1)
    if peaks.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    if f.size != v.size or f.size < 3:
        return np.zeros(peaks.size, dtype=float), np.zeros(peaks.size, dtype=float)
    try:
        _widths, _heights, left_ips, right_ips = scipy.signal.peak_widths(
            v, peaks, rel_height=0.5
        )
    except (ValueError, TypeError, FloatingPointError):
        return np.zeros(peaks.size, dtype=float), np.zeros(peaks.size, dtype=float)
    x = np.arange(f.size, dtype=float)
    f_lo = np.interp(left_ips, x, f)
    f_hi = np.interp(right_ips, x, f)
    f_peak = f[peaks]
    bw_floor = np.maximum(float(min_bw_ratio) * np.abs(f_peak), 1e-9)
    bw_hz = np.maximum(f_hi - f_lo, bw_floor)
    q_values = f_peak / bw_hz
    return np.asarray(q_values, dtype=float), np.asarray(bw_hz, dtype=float)
