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


def _cfg_float(cfg, name: str, default: float) -> float:
    try:
        v = float(getattr(cfg, name, default))
    except (AttributeError, TypeError, ValueError):
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return float(v)


def _gain_values_for_peak(gain_db: np.ndarray, mask_c: np.ndarray) -> np.ndarray:
    g = np.asarray(gain_db, dtype=float)
    try:
        m = np.asarray(mask_c, dtype=bool)
        if m.shape == g.shape and np.any(m):
            g = g[m]
    except (TypeError, ValueError):
        pass
    g = g[np.isfinite(g)]
    return np.asarray(g, dtype=float)


def _compute_peak_metrics(cfg, gain_db: np.ndarray, mask_c: np.ndarray) -> tuple[float, float, float, bool, float]:
    peak_max_db = 0.0
    peak_percentile_db = 0.0
    peak_effective_db = 0.0
    spike_detected = False
    spike_delta_db = 0.0
    try:
        g_peak = _gain_values_for_peak(gain_db, mask_c)
        peak_max_db = float(np.max(g_peak)) if g_peak.size else 0.0
        q = _cfg_float(cfg, "auto_gain_peak_percentile", 99.5)
        q = float(np.clip(q, 50.0, 100.0))
        if g_peak.size:
            peak_percentile_db = float(peak_max_db) if q >= 100.0 else float(np.percentile(g_peak, q))
        spike_delta_db = float(peak_max_db - peak_percentile_db)
        spike_threshold_db = max(0.0, _cfg_float(cfg, "auto_gain_spike_threshold_db", 1.0))
        spike_guard_db = max(0.0, _cfg_float(cfg, "auto_gain_spike_guard_db", 0.5))
        peak_effective_db = float(peak_max_db)
        spike_detected = bool(spike_delta_db >= spike_threshold_db)
        if spike_detected:
            peak_effective_db = float(min(peak_max_db, peak_percentile_db + spike_guard_db))
        if not np.isfinite(peak_effective_db):
            peak_effective_db = float(peak_max_db)
    except (TypeError, ValueError, FloatingPointError):
        pass
    return (
        float(peak_max_db),
        float(peak_percentile_db),
        float(peak_effective_db),
        bool(spike_detected),
        float(spike_delta_db),
    )


def _resolve_gain_margin(cfg) -> float:
    try:
        gain_margin_db = float(
            getattr(cfg, "auto_gain_margin_db", getattr(cfg, "global_gain_db", 0.0)) or 0.0
        )
    except (AttributeError, TypeError, ValueError):
        gain_margin_db = 0.0
    if (not np.isfinite(gain_margin_db)) or (gain_margin_db < 0.0):
        gain_margin_db = 0.0
    return float(gain_margin_db)


def _log_pre_headroom_stage_stats(cfg, gain_db: np.ndarray, mask_c: np.ndarray, logger) -> None:
    try:
        if bool(getattr(cfg, "debug_stage_stats", True)):
            v = np.asarray(gain_db, dtype=float)
            m = np.asarray(mask_c, dtype=bool)
            vv = v[m] if (m.shape == v.shape and np.any(m)) else v
            vv = vv[np.isfinite(vv)]
            if vv.size >= 4:
                logger.info(
                    "StageStats: gain_db_pre_headroom: "
                    f"max={float(np.max(vv)):.3f} dB, "
                    f"min={float(np.min(vv)):.3f} dB, "
                    f"rms={float(np.sqrt(np.mean(vv * vv))):.3f} dB"
                )
    except (TypeError, ValueError):
        return


def _resolve_auto_global_gain(
    *,
    cfg,
    logger,
    peak_max_db: float,
    peak_percentile_db: float,
    peak_effective_db: float,
    gain_margin_db: float,
    spike_detected: bool,
    spike_delta_db: float,
) -> float:
    try:
        override = getattr(cfg, "auto_gain_db_override", None)
        if override is None:
            raise ValueError("no override")
        auto_global_gain_db = float(override)
        if not np.isfinite(auto_global_gain_db):
            raise ValueError("non-finite override")
        logger.info(
            f"Auto Level: using shared override {auto_global_gain_db:.2f} dB "
            f"(peak_max={peak_max_db:.2f} dB, peak_eff={peak_effective_db:.2f} dB, margin={gain_margin_db:.2f} dB)"
        )
        return float(auto_global_gain_db)
    except (AttributeError, TypeError, ValueError):
        auto_global_gain_db = -max(0.0, float(peak_effective_db)) - float(gain_margin_db)
        logger.info(
            f"Auto Level: peak_max={peak_max_db:.2f} dB, "
            f"peak_p={peak_percentile_db:.2f} dB, "
            f"peak_eff={peak_effective_db:.2f} dB "
            f"(spike={'yes' if spike_detected else 'no'}, delta={spike_delta_db:.2f} dB) "
            f"+ margin={gain_margin_db:.2f} dB "
            f"-> auto_global_gain={auto_global_gain_db:.2f} dB"
        )
        return float(auto_global_gain_db)


def _compute_normalize_headroom(cfg, *, peak_effective_db: float, auto_global_gain_db: float, logger) -> float:
    if not bool(getattr(cfg, "do_normalize", False)):
        return 0.0
    peak_after_auto = float(peak_effective_db + auto_global_gain_db)
    if peak_after_auto > -0.1:
        auto_headroom_db = -peak_after_auto - 0.1
        logger.info(
            f"Clip Prevention (Normalize ON): post-auto peak={peak_after_auto:.2f} dB "
            f"-> extra headroom={auto_headroom_db:.2f} dB"
        )
        return float(auto_headroom_db)
    logger.info(
        f"Clip Prevention (Normalize ON): no extra headroom needed "
        f"(post-auto peak={peak_after_auto:.2f} dB)"
    )
    return 0.0


def compute_auto_gain_and_headroom(
    *,
    cfg,
    gain_db: np.ndarray,
    mask_c: np.ndarray,
    logger,
) -> dict:
    """Contract:
      - This stage is gain-only.
      - It must not mutate IR/phase arrays.
      - It may compute gain terms and emit logs.

    Returns dict with:
      current_peak_gain: float
      gain_margin_db: float
      auto_global_gain_db: float
      auto_headroom_db: float
      final_gain_total: np.ndarray
    Side effects: logs StageStats and Auto Level exactly like before.
    """
    peak_max_db, peak_percentile_db, peak_effective_db, spike_detected, spike_delta_db = _compute_peak_metrics(
        cfg,
        gain_db,
        mask_c,
    )
    current_peak_gain = float(peak_max_db)
    gain_margin_db = _resolve_gain_margin(cfg)

    auto_headroom_db = 0.0
    _log_pre_headroom_stage_stats(cfg, gain_db, mask_c, logger)
    auto_global_gain_db = _resolve_auto_global_gain(
        cfg=cfg,
        logger=logger,
        peak_max_db=float(peak_max_db),
        peak_percentile_db=float(peak_percentile_db),
        peak_effective_db=float(peak_effective_db),
        gain_margin_db=float(gain_margin_db),
        spike_detected=bool(spike_detected),
        spike_delta_db=float(spike_delta_db),
    )
    auto_headroom_db = _compute_normalize_headroom(
        cfg,
        peak_effective_db=float(peak_effective_db),
        auto_global_gain_db=float(auto_global_gain_db),
        logger=logger,
    )

    final_gain_total = gain_db + auto_global_gain_db + auto_headroom_db

    return {
        "current_peak_gain": float(current_peak_gain),
        "gain_margin_db": float(gain_margin_db),
        "auto_global_gain_db": float(auto_global_gain_db),
        "auto_headroom_db": float(auto_headroom_db),
        "final_gain_total": np.asarray(final_gain_total, dtype=float),
    }
