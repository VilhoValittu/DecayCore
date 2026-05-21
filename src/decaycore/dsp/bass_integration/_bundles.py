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

from ...io.measurement_bundle import BassIntegrationBundle
from ._filters import (
    _apply_allpass_to_transfer,
    _apply_delay_to_transfer,
    _apply_gain_trim_to_transfer,
    _apply_polarity_to_transfer,
    _get_filtered_branches,
)
from ._sub_combine import (
    _bundle_sub_slot_names,
    build_combined_sub_transfer,
    sum_complex_responses,
)
from ._utils import _safe_float, normalize_sub_combine_mode


def _build_direct_dac_trial_bundle(
    bundle: BassIntegrationBundle,
    *,
    fc_hz: float,
    main_hpf_order: int,
    sub_lpf_order: int,
    sub_hpf_hz: float,
    sub_hpf_order: int,
    sub_combine_mode: str = "average",
    sub_delay_ms: float = 0.0,
    sub_polarity_invert: bool = False,
    sub_gain_trim_db: float = 0.0,
    sub_lpf_hz: float | None = None,
    sub_allpass_freq_hz: float | None = None,
    sub_allpass_q: float | None = None,
) -> BassIntegrationBundle:
    fc = _safe_float(fc_hz, 80.0)
    # sub_lpf_hz allows overlap: sub LPF can be higher than main HPF (fc).
    # When not provided, defaults to fc (no overlap, current behaviour).
    _sub_lpf = _safe_float(sub_lpf_hz, fc) if sub_lpf_hz is not None else fc
    _sub_lpf = max(fc, _sub_lpf)  # never cut sub below main HPF
    xo_order = max(1, int(main_hpf_order))
    lpf_order = max(1, int(sub_lpf_order))
    sub_hp_hz = max(0.0, _safe_float(sub_hpf_hz, 0.0))
    sub_hp_order = max(1, int(sub_hpf_order))

    l_main_f, r_main_f, l_sub_f, r_sub_f = _get_filtered_branches(
        bundle,
        fc=fc,
        xo_order=xo_order,
        sub_hp_hz=sub_hp_hz,
        sub_hp_order=sub_hp_order,
        sub_lpf=_sub_lpf,
        lpf_order=lpf_order,
    )
    ap_freq_hz = _safe_float(sub_allpass_freq_hz, float("nan"))
    ap_q = _safe_float(sub_allpass_q, float("nan"))
    combine_mode_norm = normalize_sub_combine_mode(sub_combine_mode)
    sub_slot_names = _bundle_sub_slot_names(bundle)
    filtered_sub_map = {
        "l_sub": l_sub_f,
        "r_sub": r_sub_f,
    }
    filtered_active_subs = tuple(filtered_sub_map[name] for name in sub_slot_names if name in filtered_sub_map)
    combined_sub_f, combine_diag = build_combined_sub_transfer(
        l_main_f,
        *filtered_active_subs,
        mode=combine_mode_norm,
        label="Direct-DAC combined sub trial",
    )
    combined_sub_f = _apply_polarity_to_transfer(
        combined_sub_f,
        invert=bool(sub_polarity_invert),
        label="Direct-DAC combined sub polarity trial",
    )
    combined_sub_f = _apply_gain_trim_to_transfer(
        combined_sub_f,
        gain_trim_db=float(sub_gain_trim_db),
        label="Direct-DAC combined sub gain trial",
    )
    combined_sub_f = _apply_delay_to_transfer(
        combined_sub_f,
        delay_ms=float(sub_delay_ms),
        label="Direct-DAC combined sub delay trial",
    )
    if np.isfinite(ap_freq_hz) and ap_freq_hz > 0.0 and np.isfinite(ap_q) and ap_q > 0.0:
        combined_sub_f = _apply_allpass_to_transfer(
            combined_sub_f,
            freq_hz=float(ap_freq_hz),
            q=float(ap_q),
            label="Direct-DAC combined sub AP trial",
        )

    l_total_f = sum_complex_responses(l_main_f, combined_sub_f, label="L Direct-DAC trial total")
    r_total_f = sum_complex_responses(r_main_f, combined_sub_f, label="R Direct-DAC trial total")
    diagnostics = dict(getattr(bundle, "diagnostics", {}) or {})
    diagnostics.update(dict(combine_diag or {}))
    diagnostics.update(
        {
            "sub_slots_present": list(sub_slot_names),
            "sub_combine_mode": str(combine_mode_norm),
            "bass_integration_sub_delay_ms": float(_safe_float(sub_delay_ms, 0.0)),
            "bass_integration_sub_polarity_invert": bool(sub_polarity_invert),
            "bass_integration_sub_gain_trim_db": float(_safe_float(sub_gain_trim_db, 0.0)),
            "direct_dac_sub_allpass_enabled": bool(
                np.isfinite(ap_freq_hz) and ap_freq_hz > 0.0 and np.isfinite(ap_q) and ap_q > 0.0
            ),
            "direct_dac_sub_allpass_freq_hz": float(ap_freq_hz) if np.isfinite(ap_freq_hz) else 0.0,
            "direct_dac_sub_allpass_q": float(ap_q) if np.isfinite(ap_q) else 0.707,
        }
    )
    return BassIntegrationBundle(
        l_main=l_main_f,
        r_main=r_main_f,
        l_sub=l_sub_f,
        r_sub=r_sub_f,
        l_total=l_total_f,
        r_total=r_total_f,
        avr_crossover_hz=float(fc),
        profile=str(bundle.profile or "safe"),
        diagnostics=diagnostics,
    )
