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
from ..mag_limits import (
    _apply_hard_boost_cut_clamp,
    _apply_max_boost_cut,
    _blend_masked_fractional_octave,
)
from ..mag_telemetry import (
    _log_stage_stats,
    _record_stage_probe,
)

def _safe_cut_cap(cap: np.ndarray, *, ref: np.ndarray, max_cut_db: float) -> np.ndarray:
    out = np.asarray(cap, dtype=float)
    if out.shape != ref.shape:
        out = np.full_like(ref, float(max_cut_db), dtype=float)
    return np.clip(
        np.nan_to_num(out, nan=float(max_cut_db), posinf=float(max_cut_db), neginf=0.0),
        0.0,
        float(max_cut_db),
    )


def _softclip_pre_metrics(
    *,
    tmp: np.ndarray,
    mask_c: np.ndarray,
    boost_cap_db: np.ndarray,
    cut_cap_db: np.ndarray,
    max_cut_db: float,
    max_boost_db_base: float,
) -> tuple[np.ndarray, float, float]:
    try:
        pre_soft = np.asarray(tmp, dtype=float).copy()
        max_boost_local = np.asarray(boost_cap_db, dtype=float)
        max_cut_local = _safe_cut_cap(np.asarray(cut_cap_db, dtype=float), ref=tmp, max_cut_db=max_cut_db)
        if np.any(mask_c):
            over_boost = (
                float(np.max(pre_soft[mask_c] - max_boost_local[mask_c]))
                if float(max_boost_db_base) > 0.0
                else float(np.max(pre_soft[mask_c]))
            )
            over_boost = max(0.0, over_boost)
            over_cut = max(0.0, float(np.max((-pre_soft[mask_c]) - max_cut_local[mask_c])))
            return pre_soft, float(over_boost), float(over_cut)
    except (TypeError, ValueError, FloatingPointError, IndexError):
        pass
    return np.asarray(tmp, dtype=float), 0.0, 0.0


def _softclip_apply(
    *,
    tmp: np.ndarray,
    mask_c: np.ndarray,
    cfg,
    boost_cap_db: np.ndarray,
    cut_cap_db: np.ndarray,
    max_boost_db_base: float,
    max_cut_db: float,
) -> np.ndarray:
    try:
        mb = np.asarray(boost_cap_db, dtype=float)
        if mb.shape != tmp.shape:
            mb = np.full_like(tmp, float(max_boost_db_base), dtype=float)
        if not np.all(np.isfinite(mb)) or np.any(mb < 0.0):
            raise ValueError(f"boost_cap_db invalid (min={float(np.nanmin(mb)):.3g}); using hard clamp")
        mc = _safe_cut_cap(np.asarray(cut_cap_db, dtype=float), ref=tmp, max_cut_db=max_cut_db)
        post_soft = np.asarray(tmp, dtype=float).copy()
        pos = mask_c & (post_soft > 0.0)
        if np.any(pos):
            mbp = np.maximum(np.asarray(mb[pos], dtype=float), 0.0)
            pos_vals = post_soft[pos]
            over = np.maximum(0.0, pos_vals - mbp)
            post_soft[pos] = np.where(
                mbp > 0.0,
                np.where(pos_vals <= mbp, pos_vals, mbp + mbp * np.tanh(over / (mbp + 1e-12))),
                0.0,
            )
        neg = mask_c & (~pos)
        if np.any(neg):
            capn = np.asarray(mc[neg], dtype=float)
            vals = np.asarray(post_soft[neg], dtype=float)
            post_soft[neg] = np.where(capn > 0.0, -capn * np.tanh((-vals) / (capn + 1e-12)), 0.0)
        return np.asarray(post_soft, dtype=float)
    except (TypeError, ValueError, FloatingPointError):
        return np.asarray(_apply_max_boost_cut(tmp, cfg, max_cut_db), dtype=float)


def _softclip_count_bins(
    *,
    pre_soft: np.ndarray,
    post_soft: np.ndarray,
    mask_c: np.ndarray,
    boost_cap_db: np.ndarray,
    cut_cap_db: np.ndarray,
    max_cut_db: float,
) -> tuple[int, int]:
    try:
        if not np.any(mask_c):
            return 0, 0
        mb = np.asarray(boost_cap_db, dtype=float)
        mc = _safe_cut_cap(np.asarray(cut_cap_db, dtype=float), ref=post_soft, max_cut_db=max_cut_db)
        softclip_boost_bins = int(
            np.sum((pre_soft[mask_c] > (mb[mask_c] + 1e-9)) & (post_soft[mask_c] <= (mb[mask_c] + 1e-9)))
        )
        softclip_cut_bins = int(
            np.sum((pre_soft[mask_c] < (-mc[mask_c] - 1e-9)) & (post_soft[mask_c] >= (-mc[mask_c] - 1e-9)))
        )
        return int(softclip_boost_bins), int(softclip_cut_bins)
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return 0, 0


def _prepare_hard_clamp_locals(
    *,
    gain_db: np.ndarray,
    boost_cap_db: np.ndarray,
    cut_cap_db: np.ndarray,
    max_cut_db: float,
    max_boost_db_base: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        pre_hard = np.asarray(gain_db, dtype=float).copy()
        max_boost_local = np.asarray(boost_cap_db, dtype=float)
        max_cut_local = _safe_cut_cap(np.asarray(cut_cap_db, dtype=float), ref=gain_db, max_cut_db=max_cut_db)
        return pre_hard, max_boost_local, max_cut_local
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return (
            np.asarray(gain_db, dtype=float),
            np.full_like(gain_db, float(max_boost_db_base), dtype=float),
            np.full_like(gain_db, float(max_cut_db), dtype=float),
        )


def _hard_clamp_dominance(*, clipped_total: int, band_bins: int, hard_over_boost: float, hard_over_cut: float) -> tuple[str, float]:
    clip_pct = 100.0 * int(clipped_total) / float(max(1, int(band_bins)))
    over_peak = float(max(hard_over_boost, hard_over_cut))
    if over_peak >= 12.0 or clip_pct >= 15.0:
        return "HIGH", float(clip_pct)
    if over_peak >= 6.0 or clip_pct >= 5.0:
        return "MEDIUM", float(clip_pct)
    if clipped_total > 0:
        return "LOW", float(clip_pct)
    return "NONE", float(clip_pct)


def _hard_clamp_metrics(
    *,
    pre_hard: np.ndarray,
    gain_db: np.ndarray,
    mask_c: np.ndarray,
    max_boost_local: np.ndarray,
    max_cut_local: np.ndarray,
) -> tuple[int, int, float, float, int]:
    if not np.any(mask_c):
        return 0, 0, 0.0, 0.0, 0
    hardclamp_boost_bins = int(
        np.sum((pre_hard[mask_c] > (max_boost_local[mask_c] + 1e-9)) & (gain_db[mask_c] <= (max_boost_local[mask_c] + 1e-9)))
    )
    hardclamp_cut_bins = int(
        np.sum((pre_hard[mask_c] < (-max_cut_local[mask_c] - 1e-9)) & (gain_db[mask_c] >= (-max_cut_local[mask_c] - 1e-9)))
    )
    hard_over_boost = max(0.0, float(np.max(pre_hard[mask_c] - max_boost_local[mask_c])))
    hard_over_cut = max(0.0, float(np.max((-pre_hard[mask_c]) - max_cut_local[mask_c])))
    band_bins = int(np.sum(mask_c))
    return int(hardclamp_boost_bins), int(hardclamp_cut_bins), float(hard_over_boost), float(hard_over_cut), int(band_bins)


def _hard_clamp_maybe_smooth(
    *,
    gain_db: np.ndarray,
    cfg,
    freq_axis: np.ndarray,
    mask_c: np.ndarray,
    filter_smooth: float,
    max_cut_db: float,
    boost_cap_db: np.ndarray,
    cut_cap_db: np.ndarray,
    logger,
    debug_stage_stats: bool,
) -> np.ndarray:
    try:
        clamp_active = bool(np.any(mask_c))
    except (TypeError, ValueError):
        clamp_active = False
    if not clamp_active:
        return np.asarray(gain_db, dtype=float)
    try:
        mix = float(np.clip(float(CfgReader(cfg).float("reg_strength", 30.0)) / 100.0, 0.0, 1.0))
        if mix <= 0.0:
            return np.asarray(gain_db, dtype=float)
        pre = np.asarray(gain_db, dtype=float).copy()
        out = _blend_masked_fractional_octave(
            np.asarray(gain_db, dtype=float),
            freq_axis,
            mask_c,
            smooth_value=filter_smooth,
            mix=mix,
        )
        out = _apply_hard_boost_cut_clamp(
            out,
            cfg,
            max_cut_db,
            boost_cap_db=boost_cap_db,
            cut_cap_db=cut_cap_db,
            mask=mask_c,
        )
        _log_stage_stats(
            "gain_db_post_final_clamp_smooth",
            out,
            mask_c,
            ref=pre,
            logger=logger,
            enabled=debug_stage_stats,
        )
        return np.asarray(out, dtype=float)
    except (TypeError, ValueError, FloatingPointError):
        return np.asarray(gain_db, dtype=float)


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
    pre_soft, over_boost, over_cut = _softclip_pre_metrics(
        tmp=tmp,
        mask_c=mask_c,
        boost_cap_db=boost_cap_db,
        cut_cap_db=cut_cap_db,
        max_cut_db=max_cut_db,
        max_boost_db_base=max_boost_db_base,
    )
    tmp = _softclip_apply(
        tmp=tmp,
        mask_c=mask_c,
        cfg=cfg,
        boost_cap_db=boost_cap_db,
        cut_cap_db=cut_cap_db,
        max_boost_db_base=max_boost_db_base,
        max_cut_db=max_cut_db,
    )
    softclip_boost_bins, softclip_cut_bins = _softclip_count_bins(
        pre_soft=pre_soft,
        post_soft=np.asarray(tmp, dtype=float),
        mask_c=mask_c,
        boost_cap_db=boost_cap_db,
        cut_cap_db=cut_cap_db,
        max_cut_db=max_cut_db,
    )
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
    pre_hard, max_boost_local, max_cut_local = _prepare_hard_clamp_locals(
        gain_db=gain_db,
        boost_cap_db=boost_cap_db,
        cut_cap_db=cut_cap_db,
        max_cut_db=max_cut_db,
        max_boost_db_base=max_boost_db_base,
    )
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
        hardclamp_boost_bins, hardclamp_cut_bins, hard_over_boost, hard_over_cut, band_bins = _hard_clamp_metrics(
            pre_hard=pre_hard,
            gain_db=gain_db,
            mask_c=mask_c,
            max_boost_local=max_boost_local,
            max_cut_local=max_cut_local,
        )
        logger.info(
            "Clamp: hard_clamp "
            f"(max_boost_base={float(max_boost_db_base):.2f} dB, max_cut={float(max_cut_db):.2f} dB) -> "
            f"boost_clipped_bins={hardclamp_boost_bins}, cut_clipped_bins={hardclamp_cut_bins}, "
            f"worst_over_boost={hard_over_boost:.2f} dB, worst_over_cut={hard_over_cut:.2f} dB"
        )
        clipped_total = int(hardclamp_boost_bins + hardclamp_cut_bins)
        clamp_dominance_level, clip_pct = _hard_clamp_dominance(
            clipped_total=int(clipped_total),
            band_bins=int(band_bins),
            hard_over_boost=float(hard_over_boost),
            hard_over_cut=float(hard_over_cut),
        )
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
    if bool((hardclamp_boost_bins > 0) or (hardclamp_cut_bins > 0)):
        gain_db = _hard_clamp_maybe_smooth(
            gain_db=gain_db,
            cfg=cfg,
            freq_axis=freq_axis,
            mask_c=mask_c,
            filter_smooth=filter_smooth,
            max_cut_db=max_cut_db,
            boost_cap_db=boost_cap_db,
            cut_cap_db=cut_cap_db,
            logger=logger,
            debug_stage_stats=debug_stage_stats,
        )
    return (
        np.asarray(gain_db, dtype=float),
        int(hardclamp_boost_bins),
        int(hardclamp_cut_bins),
        float(hard_over_boost),
        float(hard_over_cut),
        str(clamp_dominance_level),
    )


__all__ = ['_apply_soft_clamps', '_apply_hard_clamps']

