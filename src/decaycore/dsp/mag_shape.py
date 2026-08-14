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
from typing import Any

import numpy as np

_logger = logging.getLogger("DecayCore.dsp")

from .smoothing import smooth_meas_freq_dep

_smooth_meas_freq_dep = smooth_meas_freq_dep


def _active_correction_band_min(cfg: Any) -> float:
    """Palauttaa korjauskaistan alarajan.

    Mixed-filtterissä käytetään tarvittaessa hieman alempaa lähtökohtaa,
    jotta `mag_c_min`-seed ei leikkaa korjausta liian aikaisin juuri
    low-bass / transition-alueella.
    """
    try:
        fmin = float(getattr(cfg, "mag_c_min", 20.0) or 20.0)
    except (AttributeError, TypeError, ValueError):
        fmin = 20.0
    if not np.isfinite(fmin) or fmin <= 0.0:
        fmin = 20.0

    filter_type = str(getattr(cfg, "filter_type_str", "") or "").strip().lower()
    if "mixed" in filter_type:
        try:
            low_bass_cut = float(getattr(cfg, "low_bass_cut_hz", fmin) or fmin)
        except (AttributeError, TypeError, ValueError):
            low_bass_cut = fmin
        if np.isfinite(low_bass_cut) and low_bass_cut > 0.0:
            fmin = min(float(fmin), max(20.0, float(low_bass_cut)))

    return float(fmin)


def _select_active_band(freq_axis: np.ndarray, cfg: Any) -> tuple[np.ndarray, tuple[float, float]]:
    """Laskee nykylogiikkaa vastaavan aktiivisen korjauskaistan."""
    fmin = float(_active_correction_band_min(cfg))
    fmax = float(cfg.mag_c_max)
    mask = (freq_axis >= fmin) & (freq_axis <= fmax)
    return mask, (fmin, fmax)


def _compute_error_db(meas_mag_db: np.ndarray, target_mag_db: np.ndarray) -> np.ndarray:
    """Laskee tavoitteen ja mittauksen erotuksen (dB)."""
    return np.asarray(target_mag_db, dtype=float) - np.asarray(meas_mag_db, dtype=float)


def _apply_confidence_logic(
    err_db: np.ndarray,
    freq_axis: np.ndarray,
    cfg: Any,
    st: Any,
    conf_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Soveltaa nykyisen confidence-painotuksen ilman algoritmimuutosta."""
    if bool(getattr(cfg, "enable_afdw", False)):
        return np.asarray(err_db, dtype=float).copy(), {"mode": "afdw_bypass"}
    conf_floor_bass = float(getattr(cfg, "conf_floor_bass", 0.60))
    if not (0.0 <= conf_floor_bass <= 1.0):
        conf_floor_bass = 0.60
    eff_conf = np.where(freq_axis < 100, np.maximum(conf_mask, conf_floor_bass), conf_mask)
    n_floored = int(np.sum((freq_axis < 100) & (conf_mask < conf_floor_bass)))
    if n_floored > 0:
        _logger.debug(
            "mag_shape: conf_floor_bass=%.2f applied to %d bins below 100 Hz",
            conf_floor_bass,
            n_floored,
        )
    return (np.asarray(err_db, dtype=float) * eff_conf).copy(), {
        "mode": "weighted_conf_floor",
        "conf_floor_bass": conf_floor_bass,
        "conf_floor_bass_bins_floored": n_floored,
    }


_DEADBAND_DB = 0.35
_DEADBAND_RISE = 0.40


def _error_to_correction_mag(err_db_final: np.ndarray) -> np.ndarray:
    """Muuntaa virhekayran korjausmagnitudiksi soft-knee dead-bandilla."""
    e = np.asarray(err_db_final, dtype=float)
    abs_e = np.abs(e)
    scale = np.clip((abs_e - _DEADBAND_DB) / _DEADBAND_RISE, 0.0, 1.0)
    return np.where(abs_e < _DEADBAND_DB, 0.0, e * scale)


def _resolve_filter_smooth(cfg: Any) -> float:
    """Lukee filter smooth -asetuksen nykyisella fallback-logiikalla."""
    try:
        value = float(getattr(cfg, "filter_smooth", getattr(cfg, "smoothing_level", 96)) or 96)
    except (AttributeError, TypeError, ValueError):
        value = 12.0
    if not np.isfinite(value) or value <= 0:
        value = 12.0
    return value


def _apply_regularization(
    err_db: np.ndarray,
    freq_axis: np.ndarray,
    cfg: Any,
    st: Any,
    smoothed_err_db: np.ndarray,
) -> np.ndarray:
    """Sekoittaa raakavirheen ja smoothatun kayran nykyisella reg_strength-kaavalla."""
    return err_db - (err_db - smoothed_err_db) * (cfg.reg_strength / 100.0)
