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
from ..mag_telemetry import (
    _band_delta_metrics,
    _band_error_rms,
)


def _store_realized_pre_ir_metrics(
    *,
    st,
    cfg_reader: CfgReader,
    freq_axis: np.ndarray,
    mask_c: np.ndarray,
    gain_db: np.ndarray,
    pre_bass_adapt_g: np.ndarray | None,
    m_anal: np.ndarray,
    target_mags: np.ndarray,
    calc_offset_db: float,
) -> None:
    if not isinstance(st, dict):
        return
    try:
        d_rms_stage = float(st.get("bass_adaptive_smoothing_delta_rms_db_20_200_stage_local", 0.0) or 0.0)
        d_max_stage = float(st.get("bass_adaptive_smoothing_delta_max_db_20_200_stage_local", 0.0) or 0.0)
        safe_put_many(
            st,
            {
                "bass_adaptive_smoothing_delta_rms_db_20_200_core": float(d_rms_stage),
                "bass_adaptive_smoothing_delta_max_db_20_200_core": float(d_max_stage),
            },
        )
        if pre_bass_adapt_g is not None and np.asarray(pre_bass_adapt_g).shape == np.asarray(gain_db).shape:
            d_rms_realized, d_max_realized, d_hz_realized = _band_delta_metrics(
                np.asarray(gain_db, dtype=float),
                np.asarray(pre_bass_adapt_g, dtype=float),
                np.asarray(freq_axis, dtype=float),
                f_lo=20.0,
                f_hi=200.0,
            )
            safe_put_many(
                st,
                {
                    "bass_adaptive_smoothing_delta_rms_db_20_200_realized_pre_ir": float(d_rms_realized),
                    "bass_adaptive_smoothing_delta_max_db_20_200_realized_pre_ir": float(d_max_realized),
                    "bass_adaptive_smoothing_delta_max_hz_20_200_realized_pre_ir": (
                        float(d_hz_realized) if d_hz_realized is not None else None
                    ),
                    "bass_adaptive_smoothing_delta_rms_db_20_200": float(d_rms_realized),
                    "bass_adaptive_smoothing_delta_max_db_20_200": float(d_max_realized),
                    "bass_adaptive_smoothing_delta_max_hz_20_200": (
                        float(d_hz_realized) if d_hz_realized is not None else None
                    ),
                    "bass_adaptive_smoothing_delta_basis": "mag_post_limits_pre_ir",
                },
            )
    except (TypeError, ValueError, FloatingPointError, IndexError):
        safe_put_many(
            st,
            {
                "bass_adaptive_smoothing_delta_max_hz_20_200_realized_pre_ir": None,
                "bass_adaptive_smoothing_delta_basis": "core_stage_fallback",
            },
        )
    try:
        mid_lo = float(cfg_reader.float("mid_refit_hz_lo", 200.0))
        mid_hi = float(cfg_reader.float("mid_refit_hz_hi", 2000.0))
        before_rms = st.get("mid_refit_err_rms_before_stage_local")
        before_rms = float(before_rms) if before_rms is not None else None
        after_realized = _band_error_rms(
            gain_db=np.asarray(gain_db, dtype=float),
            measured_db=np.asarray(m_anal, dtype=float),
            target_db=np.asarray(target_mags, dtype=float),
            calc_offset_db=float(calc_offset_db),
            freq_axis=np.asarray(freq_axis, dtype=float),
            mask_c=np.asarray(mask_c, dtype=bool),
            f_lo=float(mid_lo),
            f_hi=float(mid_hi),
        )
        safe_put_many(
            st,
            {
                "mid_refit_err_rms_after_realized_pre_ir": after_realized,
                "mid_refit_delta_rms_realized_pre_ir": (
                    float(before_rms - after_realized)
                    if before_rms is not None and after_realized is not None
                    else None
                ),
            },
        )
    except (TypeError, ValueError, FloatingPointError, IndexError):
        pass


__all__ = ["_store_realized_pre_ir_metrics"]
