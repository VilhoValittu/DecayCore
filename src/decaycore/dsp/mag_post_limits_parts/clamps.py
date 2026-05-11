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
import scipy.ndimage

from ..decaycore_analysis import _sigma_bins_from_hz
from ..correction_types import _MagPostProcessInputs, _MagPostProcessOutputs
from .._measurement_ctx_local import get_measurement_ctx
from ..dsp_config import CfgReader
from ..dsp_telemetry import safe_put_many
from ..gain_policy import apply_cuts_only_guard, build_low_frequency_guard_mask, resolve_gain_policy
from ..mag_limits import (
    _apply_hard_boost_cut_clamp,
    _apply_max_boost_cut,
    _apply_slope_limits,
    _blend_masked_fractional_octave,
)
from ..mag_postprocess import apply_bass_boost_post_restore, apply_confpull_post_slope
from ..mag_telemetry import (
    _band_delta_metrics,
    _band_error_rms,
    _log_stage_stats,
    _record_stage_probe,
    _summarize_correction_metrics,
)
from ..phase_ir_utils import _cosine_fade_out_01
from ..smoothing import smooth_gain_fractional_octave

def _apply_soft_clamps(
    *,
    cfg,
    freq_axis: np.ndarray,
    mask_c: np.ndarray,
    gain_apply: np.ndarray,
    gain_db: np.ndarray,
    max_cut_db: float,
    max_boost_db_base: float,
    boost_cap_db: np.ndarray,
    cut_cap_db: np.ndarray,
    stage_probes: dict[str, object],
    stage_probe_fn,
    logger,
) -> tuple[np.ndarray, float, float, int, int]:
    tmp = np.zeros_like(gain_db, dtype=float)
    tmp[mask_c] = gain_apply[mask_c]
    _record_stage_probe(stage_probes, "pre_softclip", stage_probe_fn, freq_axis, tmp, mask_c, cfg, logger)
    try:
        pre_soft = tmp.copy()
        max_boost_local = np.asarray(boost_cap_db, dtype=float)
        max_cut_local = np.asarray(cut_cap_db, dtype=float)
        if max_cut_local.shape != tmp.shape:
            max_cut_local = np.full_like(tmp, float(max_cut_db), dtype=float)
        max_cut_local = np.clip(
            np.nan_to_num(max_cut_local, nan=float(max_cut_db), posinf=float(max_cut_db), neginf=0.0),
            0.0,
            float(max_cut_db),
        )
        if np.any(mask_c):
            over_boost = (
                float(np.max(pre_soft[mask_c] - max_boost_local[mask_c]))
                if float(max_boost_db_base) > 0.0
                else float(np.max(pre_soft[mask_c]))
            )
            over_boost = max(0.0, over_boost)
            over_cut = max(0.0, float(np.max((-pre_soft[mask_c]) - max_cut_local[mask_c])))
        else:
            over_boost, over_cut = 0.0, 0.0
    except (TypeError, ValueError, FloatingPointError, IndexError):
        pre_soft = tmp
        over_boost, over_cut = 0.0, 0.0
    try:
        mb = np.asarray(boost_cap_db, dtype=float)
        if mb.shape != tmp.shape:
            mb = np.full_like(tmp, float(max_boost_db_base), dtype=float)
        if not np.all(np.isfinite(mb)) or np.any(mb < 0.0):
            raise ValueError(f"boost_cap_db invalid (min={float(np.nanmin(mb)):.3g}); using hard clamp")
        mc = np.asarray(cut_cap_db, dtype=float)
        if mc.shape != tmp.shape:
            mc = np.full_like(tmp, float(max_cut_db), dtype=float)
        mc = np.clip(
            np.nan_to_num(mc, nan=float(max_cut_db), posinf=float(max_cut_db), neginf=0.0),
            0.0,
            float(max_cut_db),
        )
        post_soft = np.asarray(tmp, dtype=float).copy()
        pos = mask_c & (post_soft > 0.0)
        if np.any(pos):
            mbp = np.maximum(np.asarray(mb[pos], dtype=float), 0.0)
            pos_vals = post_soft[pos]
            over = np.maximum(0.0, pos_vals - mbp)
            post_soft[pos] = np.where(
                mbp > 0.0,
                np.where(
                    pos_vals <= mbp,
                    pos_vals,
                    mbp + mbp * np.tanh(over / (mbp + 1e-12)),
                ),
                0.0,
            )
        neg = mask_c & (~pos)
        if np.any(neg):
            capn = np.asarray(mc[neg], dtype=float)
            vals = np.asarray(post_soft[neg], dtype=float)
            post_soft[neg] = np.where(
                capn > 0.0,
                -capn * np.tanh((-vals) / (capn + 1e-12)),
                0.0,
            )
        tmp = post_soft
    except (TypeError, ValueError, FloatingPointError):
        tmp = _apply_max_boost_cut(tmp, cfg, max_cut_db)
    try:
        post_soft = np.asarray(tmp, dtype=float)
        if np.any(mask_c):
            mb = np.asarray(boost_cap_db, dtype=float)
            mc = np.asarray(cut_cap_db, dtype=float)
            if mc.shape != post_soft.shape:
                mc = np.full_like(post_soft, float(max_cut_db), dtype=float)
            mc = np.clip(
                np.nan_to_num(mc, nan=float(max_cut_db), posinf=float(max_cut_db), neginf=0.0),
                0.0,
                float(max_cut_db),
            )
            softclip_boost_bins = int(
                np.sum(
                    (pre_soft[mask_c] > (mb[mask_c] + 1e-9))
                    & (post_soft[mask_c] <= (mb[mask_c] + 1e-9))
                )
            )
            softclip_cut_bins = int(
                np.sum(
                    (pre_soft[mask_c] < (-mc[mask_c] - 1e-9))
                    & (post_soft[mask_c] >= (-mc[mask_c] - 1e-9))
                )
            )
        else:
            softclip_boost_bins, softclip_cut_bins = 0, 0
    except (TypeError, ValueError, FloatingPointError, IndexError):
        softclip_boost_bins, softclip_cut_bins = 0, 0
        over_boost, over_cut = 0.0, 0.0
    try:
        logger.info(
            "Clamp: soft_clip "
            f"(max_boost_base={float(max_boost_db_base):.2f} dB, max_cut={float(max_cut_db):.2f} dB) -> "
            f"boost_clipped_bins={softclip_boost_bins}, cut_clipped_bins={softclip_cut_bins}, "
            f"worst_over_boost={over_boost:.2f} dB, worst_over_cut={over_cut:.2f} dB"
        )
    except Exception:  # noqa: BLE001
        pass
    _record_stage_probe(stage_probes, "post_softclip", stage_probe_fn, freq_axis, tmp, mask_c, cfg, logger)
    return np.asarray(tmp, dtype=float), float(over_boost), float(over_cut), int(softclip_boost_bins), int(softclip_cut_bins)

def _apply_hard_clamps(
    *,
    cfg,
    freq_axis: np.ndarray,
    mask_c: np.ndarray,
    gain_db: np.ndarray,
    boost_cap_db: np.ndarray,
    cut_cap_db: np.ndarray,
    max_cut_db: float,
    max_boost_db_base: float,
    filter_smooth: float,
    debug_stage_stats: bool,
    stage_probes: dict[str, object],
    stage_probe_fn,
    logger,
    st,
) -> tuple[np.ndarray, int, int, float, float, str]:
    _record_stage_probe(stage_probes, "pre_hardclamp", stage_probe_fn, freq_axis, gain_db, mask_c, cfg, logger)
    try:
        pre_hard = gain_db.copy()
        max_boost_local = np.asarray(boost_cap_db, dtype=float)
        max_cut_local = np.asarray(cut_cap_db, dtype=float)
        if max_cut_local.shape != gain_db.shape:
            max_cut_local = np.full_like(gain_db, float(max_cut_db), dtype=float)
        max_cut_local = np.clip(
            np.nan_to_num(max_cut_local, nan=float(max_cut_db), posinf=float(max_cut_db), neginf=0.0),
            0.0,
            float(max_cut_db),
        )
    except (TypeError, ValueError, FloatingPointError, IndexError):
        pre_hard = gain_db
        max_boost_local = np.full_like(gain_db, float(max_boost_db_base), dtype=float)
        max_cut_local = np.full_like(gain_db, float(max_cut_db), dtype=float)
    gain_db = _apply_hard_boost_cut_clamp(
        gain_db,
        cfg,
        max_cut_db,
        boost_cap_db=boost_cap_db,
        cut_cap_db=cut_cap_db,
        mask=mask_c,
    )
    _record_stage_probe(stage_probes, "post_hardclamp", stage_probe_fn, freq_axis, gain_db, mask_c, cfg, logger)
    hardclamp_boost_bins = 0
    hardclamp_cut_bins = 0
    hard_over_boost = 0.0
    hard_over_cut = 0.0
    clamp_dominance_level = "NONE"
    try:
        if np.any(mask_c):
            hardclamp_boost_bins = int(
                np.sum(
                    (pre_hard[mask_c] > (max_boost_local[mask_c] + 1e-9))
                    & (gain_db[mask_c] <= (max_boost_local[mask_c] + 1e-9))
                )
            )
            hardclamp_cut_bins = int(
                np.sum(
                    (pre_hard[mask_c] < (-max_cut_local[mask_c] - 1e-9))
                    & (gain_db[mask_c] >= (-max_cut_local[mask_c] - 1e-9))
                )
            )
            hard_over_boost = max(0.0, float(np.max(pre_hard[mask_c] - max_boost_local[mask_c])))
            hard_over_cut = max(0.0, float(np.max((-pre_hard[mask_c]) - max_cut_local[mask_c])))
            band_bins = int(np.sum(mask_c))
        else:
            band_bins = 0
        logger.info(
            "Clamp: hard_clamp "
            f"(max_boost_base={float(max_boost_db_base):.2f} dB, max_cut={float(max_cut_db):.2f} dB) -> "
            f"boost_clipped_bins={hardclamp_boost_bins}, cut_clipped_bins={hardclamp_cut_bins}, "
            f"worst_over_boost={hard_over_boost:.2f} dB, worst_over_cut={hard_over_cut:.2f} dB"
        )
        clipped_total = int(hardclamp_boost_bins + hardclamp_cut_bins)
        clip_pct = 100.0 * clipped_total / float(max(1, band_bins))
        over_peak = float(max(hard_over_boost, hard_over_cut))
        if over_peak >= 12.0 or clip_pct >= 15.0:
            clamp_dominance_level = "HIGH"
        elif over_peak >= 6.0 or clip_pct >= 5.0:
            clamp_dominance_level = "MEDIUM"
        elif clipped_total > 0:
            clamp_dominance_level = "LOW"
        logger.info(
            "Clamp dominance: "
            f"{clamp_dominance_level} | "
            f"clipped={clipped_total}/{int(band_bins)} ({clip_pct:.2f}%), "
            f"over_boost={hard_over_boost:.2f} dB, over_cut={hard_over_cut:.2f} dB"
            + (" | smoothing impact may be masked" if clamp_dominance_level != "NONE" else "")
        )
        if isinstance(st, dict):
            safe_put_many(
                st,
                {
                    "clamp_dominance_level": str(clamp_dominance_level),
                    "clamp_dominance_clip_pct": float(clip_pct),
                    "clamp_dominance_clipped_bins": int(clipped_total),
                    "clamp_dominance_band_bins": int(band_bins),
                },
            )
    except (TypeError, ValueError, FloatingPointError, IndexError):
        hardclamp_boost_bins, hardclamp_cut_bins = 0, 0
        hard_over_boost, hard_over_cut = 0.0, 0.0
        clamp_dominance_level = "NONE"
    try:
        clamp_active = bool((hardclamp_boost_bins > 0) or (hardclamp_cut_bins > 0))
    except (TypeError, ValueError):
        clamp_active = False
    try:
        if clamp_active and np.any(mask_c):
            mix = float(np.clip(float(CfgReader(cfg).float("reg_strength", 30.0)) / 100.0, 0.0, 1.0))
            if mix > 0.0:
                pre = gain_db.copy()
                gain_db = _blend_masked_fractional_octave(
                    gain_db,
                    freq_axis,
                    mask_c,
                    smooth_value=filter_smooth,
                    mix=mix,
                )
                gain_db = _apply_hard_boost_cut_clamp(
                    gain_db,
                    cfg,
                    max_cut_db,
                    boost_cap_db=boost_cap_db,
                    cut_cap_db=cut_cap_db,
                    mask=mask_c,
                )
                _log_stage_stats(
                    "gain_db_post_final_clamp_smooth",
                    gain_db,
                    mask_c,
                    ref=pre,
                    logger=logger,
                    enabled=debug_stage_stats,
                )
    except (TypeError, ValueError, FloatingPointError):
        pass
    return (
        np.asarray(gain_db, dtype=float),
        int(hardclamp_boost_bins),
        int(hardclamp_cut_bins),
        float(hard_over_boost),
        float(hard_over_cut),
        str(clamp_dominance_level),
    )


__all__ = ['_apply_soft_clamps', '_apply_hard_clamps']


def _load_sibling_symbols() -> None:
    import importlib
    package = __package__
    for module_name in ['low_frequency', 'authority', 'clamps', 'metrics', 'pipeline']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_load_sibling_symbols()
