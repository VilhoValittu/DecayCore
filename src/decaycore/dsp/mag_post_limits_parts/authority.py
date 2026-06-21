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

import numpy as np

from ..dsp_config import CfgReader
from ..dsp_telemetry import safe_put_many
from ..smoothing import smooth_gain_fractional_octave

from .low_frequency import (
    _authority_band_metrics,
    _stats_array,
)

def _authority_metrics_payload(
    *,
    st,
    freq_axis: np.ndarray,
    mask_c: np.ndarray,
    base_boost_cap: np.ndarray,
    base_cut_cap: np.ndarray,
    boost_out: np.ndarray,
    cut_out: np.ndarray,
    enabled: bool,
    source: str,
) -> dict[str, object]:
    boost_reduction = np.maximum(0.0, base_boost_cap - boost_out)
    cut_reduction = np.maximum(0.0, base_cut_cap - cut_out)
    boost_max, boost_mean, boost_min = _authority_band_metrics(
        freq_axis=freq_axis,
        mask_c=mask_c,
        reduction_db=boost_reduction,
        cap_db=boost_out,
    )
    cut_max, cut_mean, cut_min = _authority_band_metrics(
        freq_axis=freq_axis,
        mask_c=mask_c,
        reduction_db=cut_reduction,
        cap_db=cut_out,
    )
    try:
        m_bins = np.asarray(mask_c, dtype=bool)
        if m_bins.shape != boost_reduction.shape:
            m_bins = np.ones_like(boost_reduction, dtype=bool)
        boost_bins = int(np.count_nonzero(m_bins & (boost_reduction > 1e-9)))
        cut_bins = int(np.count_nonzero(m_bins & (cut_reduction > 1e-9)))
    except (TypeError, ValueError):
        boost_bins = 0
        cut_bins = 0
    try:
        if isinstance(st, dict):
            safe_put_many(
                st,
                {
                    "acoustic_authority_limits_enabled": bool(enabled),
                    "acoustic_authority_limits_source": str(source),
                    "authority_boost_cap_reduction_max_db": float(boost_max),
                    "authority_boost_cap_reduction_mean_db_20_300": float(boost_mean),
                    "authority_cut_cap_reduction_max_db": float(cut_max),
                    "authority_cut_cap_reduction_mean_db_20_300": float(cut_mean),
                    "authority_boost_cap_min_db_20_300": float(boost_min),
                    "authority_cut_cap_min_db_20_300": float(cut_min),
                    "authority_boost_cap_reduced_bins": int(boost_bins),
                    "authority_cut_cap_reduced_bins": int(cut_bins),
                },
            )
    except (TypeError, ValueError):
        pass
    return {
        "boost_cap_db": np.asarray(boost_out, dtype=float),
        "cut_cap_db": np.asarray(cut_out, dtype=float),
        "enabled": bool(enabled),
        "source": str(source),
        "boost_reduction_max_db": float(boost_max),
        "boost_reduction_mean_db_20_300": float(boost_mean),
        "cut_reduction_max_db": float(cut_max),
        "cut_reduction_mean_db_20_300": float(cut_mean),
        "boost_reduced_bins": int(boost_bins),
        "cut_reduced_bins": int(cut_bins),
    }


def _authority_limits_from_stats(
    *,
    cfg_reader: CfgReader,
    st,
    source: str,
    base_cut_cap: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    boost_authority, boost_source = _stats_array(st, ("authority_boost", "boost_authority"), base_cut_cap.shape)
    cut_authority, cut_source = _stats_array(st, ("authority_cut", "cut_authority"), base_cut_cap.shape)
    if boost_authority is None or cut_authority is None:
        return None, None, "missing"
    resolved_source = "stats" if boost_source == "stats" and cut_source == "stats" else str(source)
    return np.asarray(boost_authority, dtype=float), np.asarray(cut_authority, dtype=float), resolved_source


def _apply_acoustic_authority_caps(
    *,
    cfg,
    cfg_reader: CfgReader,
    st,
    logger,
    freq_axis: np.ndarray,
    mask_c: np.ndarray,
    boost_cap_db: np.ndarray,
    max_cut_db: float,
) -> dict[str, object]:
    base_boost_cap = np.asarray(boost_cap_db, dtype=float)
    base_cut_cap = np.full_like(np.asarray(freq_axis, dtype=float), float(max_cut_db), dtype=float)
    source = "missing"
    enabled = False
    boost_out = base_boost_cap.copy()
    cut_out = base_cut_cap.copy()

    try:
        if base_boost_cap.shape != base_cut_cap.shape or np.asarray(mask_c, dtype=bool).shape != base_cut_cap.shape:
            return _authority_metrics_payload(
                st=st,
                freq_axis=freq_axis,
                mask_c=mask_c,
                base_boost_cap=base_boost_cap,
                base_cut_cap=base_cut_cap,
                boost_out=boost_out,
                cut_out=cut_out,
                enabled=enabled,
                source=source,
            )
        if not cfg_reader.bool("acoustic_authority_limits_enable", True):
            source = "disabled"
            return _authority_metrics_payload(
                st=st,
                freq_axis=freq_axis,
                mask_c=mask_c,
                base_boost_cap=base_boost_cap,
                base_cut_cap=base_cut_cap,
                boost_out=boost_out,
                cut_out=cut_out,
                enabled=enabled,
                source=source,
            )

        boost_authority, cut_authority, source = _authority_limits_from_stats(
            cfg_reader=cfg_reader,
            st=st,
            source="missing",
            base_cut_cap=base_cut_cap,
        )
        if boost_authority is None or cut_authority is None:
            return _authority_metrics_payload(
                st=st,
                freq_axis=freq_axis,
                mask_c=mask_c,
                base_boost_cap=base_boost_cap,
                base_cut_cap=base_cut_cap,
                boost_out=boost_out,
                cut_out=cut_out,
                enabled=enabled,
                source=source,
            )

        boost_gamma = float(max(0.0, cfg_reader.float_allow_zero("authority_boost_gamma", 1.35)))
        boost_min_frac = float(np.clip(cfg_reader.float_allow_zero("authority_boost_min_frac", 0.05), 0.0, 1.0))
        boost_min_cap_db = float(max(0.0, cfg_reader.float_allow_zero("authority_boost_min_cap_db", 0.0)))
        cut_gamma = float(max(0.0, cfg_reader.float_allow_zero("authority_cut_gamma", 0.75)))
        cut_min_frac = float(np.clip(cfg_reader.float_allow_zero("authority_cut_min_frac", 0.35), 0.0, 1.0))
        cut_min_cap_db = float(max(0.0, cfg_reader.float_allow_zero("authority_cut_min_cap_db", 3.0)))
        caps_smooth_oct = cfg_reader.float_allow_zero("authority_caps_smooth_oct", 1.0 / 9.0)

        a_boost = np.clip(np.nan_to_num(boost_authority, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        a_cut = np.clip(np.nan_to_num(cut_authority, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        if np.isfinite(caps_smooth_oct) and caps_smooth_oct > 0.0:
            # Ilman tasoitusta per-bin rosoiset authority-käyrät leikkaavat
            # sileän gainin sahalaitaiseksi HF:llä (mag_c_max >> 230 Hz).
            smooth_value = 1.0 / float(np.clip(caps_smooth_oct, 1.0 / 1.0, 1.0))
            a_boost = np.clip(smooth_gain_fractional_octave(freq_axis, a_boost, smooth_value), 0.0, 1.0)
            a_cut = np.clip(smooth_gain_fractional_octave(freq_axis, a_cut, smooth_value), 0.0, 1.0)
        boost_factor = boost_min_frac + (1.0 - boost_min_frac) * np.power(a_boost, boost_gamma)
        cut_factor = cut_min_frac + (1.0 - cut_min_frac) * np.power(a_cut, cut_gamma)

        local_boost_cap = base_boost_cap * boost_factor
        local_boost_cap = np.maximum(local_boost_cap, boost_min_cap_db)
        local_boost_cap = np.minimum(local_boost_cap, base_boost_cap)
        local_cut_cap = base_cut_cap * cut_factor
        local_cut_cap = np.maximum(local_cut_cap, cut_min_cap_db)
        local_cut_cap = np.minimum(local_cut_cap, base_cut_cap)

        m = np.asarray(mask_c, dtype=bool)
        boost_out[m] = local_boost_cap[m]
        cut_out[m] = local_cut_cap[m]
        enabled = True
    except (TypeError, ValueError, FloatingPointError, IndexError):
        source = "missing"
        enabled = False
        boost_out = base_boost_cap.copy()
        cut_out = base_cut_cap.copy()

    result = _authority_metrics_payload(
        st=st,
        freq_axis=freq_axis,
        mask_c=mask_c,
        base_boost_cap=base_boost_cap,
        base_cut_cap=base_cut_cap,
        boost_out=boost_out,
        cut_out=cut_out,
        enabled=enabled,
        source=source,
    )
    if bool(result["enabled"]):
        try:
            logger.info(
                "Acoustic authority limits: ON "
                f"source={result['source']} "
                f"boost_cap_reduction_max={float(result['boost_reduction_max_db']):.2f} dB "
                f"cut_cap_reduction_max={float(result['cut_reduction_max_db']):.2f} dB"
            )
        except (AttributeError, TypeError, ValueError):
            pass
    return result

def _apply_candidate_metrics(
    *,
    gain_apply: np.ndarray,
    gain_db: np.ndarray,
    freq_axis: np.ndarray,
    mask_c: np.ndarray,
    gain_policy,
) -> tuple[float, float, int, int, int]:
    try:
        cand = np.zeros_like(gain_db, dtype=float)
        cand[mask_c] = gain_apply[mask_c]
        cand_boost_mask = (cand > 1e-6) & mask_c
        boost_cand_peak = float(np.max(cand[mask_c])) if np.any(mask_c) else 0.0
        n_boost_cand = int(np.sum(cand_boost_mask))
        n_boost_cand_low = int(np.sum((cand > 1e-6) & mask_c & (freq_axis <= float(gain_policy.low_cut_hz))))
        if bool(gain_policy.exc_prot):
            n_boost_cand_exc = int(
                np.sum((cand > 1e-6) & mask_c & (freq_axis < float(gain_policy.exc_freq)))
            )
        else:
            n_boost_cand_exc = 0
        cand_boost_pos_mask = cand_boost_mask & (freq_axis > 0.0)
        boost_cand_min_hz = float(np.min(freq_axis[cand_boost_pos_mask])) if np.any(cand_boost_pos_mask) else float("nan")
        return boost_cand_peak, boost_cand_min_hz, n_boost_cand, n_boost_cand_low, n_boost_cand_exc
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return 0.0, float("nan"), 0, 0, 0


__all__ = ['_apply_acoustic_authority_caps', '_apply_candidate_metrics']

