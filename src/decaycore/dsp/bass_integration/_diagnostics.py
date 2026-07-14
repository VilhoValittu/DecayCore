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

from typing import Any

import numpy as np

from ...auto_mode.shared import (
    AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
    AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
    _auto_bass_integration_profile_norm,
)
from ...io.measurement_bundle import BassIntegrationBundle
from ._channel_metrics import (
    _channel_metric_summary,
    _channel_overlap_extension_metrics,
    _channel_overlap_metrics,
    _channel_predicted_sum_metrics,
)
from ._sub_combine import build_bundle_combined_sub_transfer
from ._utils import _safe_float, normalize_sub_combine_mode


def compute_bass_integration_diagnostics(
    bundle: BassIntegrationBundle,
    fc_hz: float,
    profile: str,
    *,
    sub_combine_mode: str | None = None,
    sub_lpf_hz: float | None = None,
    guard_lo_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
    guard_hi_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
) -> dict[str, Any]:
    fc = _safe_float(fc_hz, 80.0)
    _lpf = _safe_float(sub_lpf_hz, fc) if sub_lpf_hz is not None else fc
    _lpf = max(fc, _lpf)
    lo_ratio = max(0.05, _safe_float(guard_lo_ratio, 0.60))
    hi_ratio = max(lo_ratio + 0.05, _safe_float(guard_hi_ratio, 1.40))
    profile_name = _auto_bass_integration_profile_norm(profile)
    combine_mode_norm = normalize_sub_combine_mode(
        sub_combine_mode or dict(getattr(bundle, "diagnostics", {}) or {}).get("sub_combine_mode")
    )

    l_sub_total, l_combine_diag = build_bundle_combined_sub_transfer(
        bundle,
        channel="l",
        mode=combine_mode_norm,
        label="L combined sub",
    )
    r_sub_total, r_combine_diag = build_bundle_combined_sub_transfer(
        bundle,
        channel="r",
        mode=combine_mode_norm,
        label="R combined sub",
    )

    guard_lo_hz = max(5.0, fc * lo_ratio)
    guard_hi_hz = max(guard_lo_hz + 1.0, fc * hi_ratio)

    l_metrics = _channel_overlap_metrics(
        bundle.l_main,
        l_sub_total,
        bundle.l_total,
        lo_hz=guard_lo_hz,
        hi_hz=guard_hi_hz,
    )
    r_metrics = _channel_overlap_metrics(
        bundle.r_main,
        r_sub_total,
        bundle.r_total,
        lo_hz=guard_lo_hz,
        hi_hz=guard_hi_hz,
    )
    l_metrics = {
        **dict(l_metrics or {}),
        **_channel_predicted_sum_metrics(
            bundle.l_total,
            lo_hz=guard_lo_hz,
            hi_hz=guard_hi_hz,
            fc_hz=fc,
        ),
    }
    r_metrics = {
        **dict(r_metrics or {}),
        **_channel_predicted_sum_metrics(
            bundle.r_total,
            lo_hz=guard_lo_hz,
            hi_hz=guard_hi_hz,
            fc_hz=fc,
        ),
    }
    l_extension_metrics = _channel_overlap_extension_metrics(
        bundle.l_main,
        l_sub_total,
        bundle.l_total,
        fc_hz=float(fc),
        sub_lpf_hz=float(_lpf),
    )
    r_extension_metrics = _channel_overlap_extension_metrics(
        bundle.r_main,
        r_sub_total,
        bundle.r_total,
        fc_hz=float(fc),
        sub_lpf_hz=float(_lpf),
    )

    out = {
        "profile": profile_name,
        "avr_crossover_hz": float(fc),
        "sub_lpf_hz": float(_lpf),
        "sub_combine_mode": str(combine_mode_norm),
        "metric_channel_mode": "worst_case",
        "guard_lo_ratio": float(lo_ratio),
        "guard_hi_ratio": float(hi_ratio),
        "guard_lo_hz": float(guard_lo_hz),
        "guard_hi_hz": float(guard_hi_hz),
        "overlap_ratio": float(
            np.nanmean(
                np.asarray(
                    [
                        _safe_float((l_metrics or {}).get("overlap_ratio", float("nan")), float("nan")),
                        _safe_float((r_metrics or {}).get("overlap_ratio", float("nan")), float("nan")),
                    ],
                    dtype=float,
                )
            )
        ),
        "whether_alignment_applied": bool(
            (l_combine_diag or {}).get("whether_alignment_applied", False)
            or (r_combine_diag or {}).get("whether_alignment_applied", False)
        ),
        "alignment_offset_ms": _safe_float(
            (l_combine_diag or {}).get(
                "alignment_offset_ms",
                (r_combine_diag or {}).get("alignment_offset_ms", 0.0),
            ),
            0.0,
        ),
        "alignment_confidence": max(
            _safe_float((l_combine_diag or {}).get("alignment_confidence", float("nan")), float("nan")),
            _safe_float((r_combine_diag or {}).get("alignment_confidence", float("nan")), float("nan")),
        ),
        "sub_combined_level_delta_db_20_120": _safe_float(
            (l_combine_diag or {}).get(
                "sub_combined_level_delta_db_20_120",
                (r_combine_diag or {}).get("sub_combined_level_delta_db_20_120", float("nan")),
            ),
            float("nan"),
        ),
        "sub_combined_level_delta_db_30_90": _safe_float(
            (l_combine_diag or {}).get(
                "sub_combined_level_delta_db_30_90",
                (r_combine_diag or {}).get("sub_combined_level_delta_db_30_90", float("nan")),
            ),
            float("nan"),
        ),
        "overlap_extension_active": bool(
            bool((l_extension_metrics or {}).get("overlap_extension_active", False))
            or bool((r_extension_metrics or {}).get("overlap_extension_active", False))
        ),
        "overlap_extension_lo_hz": float(
            _safe_float(
                (l_extension_metrics or {}).get("overlap_extension_lo_hz", float("nan")),
                float("nan"),
            )
        ),
        "overlap_extension_hi_hz": float(
            _safe_float(
                (l_extension_metrics or {}).get("overlap_extension_hi_hz", float("nan")),
                float("nan"),
            )
        ),
        "channels": {
            "l": dict(l_metrics),
            "r": dict(r_metrics),
        },
        "channels_extension": {
            "l": dict(l_extension_metrics),
            "r": dict(r_extension_metrics),
        },
    }
    for key, absolute in (
        ("cancellation_risk", False),
        ("overlap_ripple_db", False),
        ("sub_dominance_db", True),
        ("null_severity", False),
        ("predicted_sum_flatness_db", False),
        ("predicted_sum_dip_depth_db", False),
        ("predicted_sum_peak_excess_db", False),
        ("null_depth_db", False),
        ("null_width_hz", False),
        ("dip_p10_db", False),
    ):
        out.update(_channel_metric_summary(l_metrics, r_metrics, key, absolute=absolute))
    for key, absolute in (
        ("overlap_extension_flatness_db", False),
        ("overlap_extension_cancellation_risk", False),
        ("overlap_extension_peak_excess_db", False),
        ("overlap_extension_sub_dominance_db", True),
    ):
        out.update(_channel_metric_summary(l_extension_metrics, r_extension_metrics, key, absolute=absolute))

    out["cancellation_risk"] = float(out.get("cancellation_risk_worst", float("nan")))
    out["overlap_ripple_db"] = float(out.get("overlap_ripple_db_worst", float("nan")))
    out["sub_dominance_db"] = float(out.get("sub_dominance_db_worst", float("nan")))
    out["null_severity"] = float(out.get("null_severity_worst", float("nan")))
    out["predicted_sum_flatness_db"] = float(out.get("predicted_sum_flatness_db_worst", float("nan")))
    out["predicted_sum_dip_depth_db"] = float(out.get("predicted_sum_dip_depth_db_worst", float("nan")))
    out["predicted_sum_peak_excess_db"] = float(out.get("predicted_sum_peak_excess_db_worst", float("nan")))
    out["null_depth_db"] = float(out.get("null_depth_db_worst", float("nan")))
    out["null_width_hz"] = float(out.get("null_width_hz_worst", float("nan")))
    out["overlap_extension_flatness_db"] = float(
        out.get("overlap_extension_flatness_db_worst", float("nan"))
    )
    out["overlap_extension_cancellation_risk"] = float(
        out.get("overlap_extension_cancellation_risk_worst", float("nan"))
    )
    out["overlap_extension_peak_excess_db"] = float(
        out.get("overlap_extension_peak_excess_db_worst", float("nan"))
    )
    out["overlap_extension_sub_dominance_db"] = float(
        out.get("overlap_extension_sub_dominance_db_worst", float("nan"))
    )
    worst_null_channel = "l"
    if _safe_float((r_metrics or {}).get("null_severity", float("nan")), float("nan")) >= _safe_float(
        (l_metrics or {}).get("null_severity", float("nan")),
        float("nan"),
    ):
        worst_null_channel = "r"
    out["null_center_hz"] = _safe_float(
        (r_metrics if worst_null_channel == "r" else l_metrics).get("null_center_hz", float("nan")),
        float("nan"),
    )
    return out


def compute_bass_integration_metric_payload(
    bundle: BassIntegrationBundle,
    fc_hz: float,
    profile: str,
    *,
    mode: str = "direct_dac",
    main_hpf_order: int = 4,
    sub_lpf_order: int = 4,
    sub_hpf_hz: float = 20.0,
    sub_hpf_order: int = 2,
    sub_combine_mode: str = "average",
    sub_delay_ms: float = 0.0,
    sub_polarity_invert: bool = False,
    sub_gain_trim_db: float = 0.0,
    sub_lpf_hz: float | None = None,
    sub_allpass_freq_hz: float | None = None,
    sub_allpass_q: float | None = None,
    guard_lo_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
    guard_hi_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
    l_fir: Any | None = None,
    r_fir: Any | None = None,
    sub_fir: Any | None = None,
    fir_sample_rate: int | None = None,
    robust: bool | None = None,
) -> dict[str, Any]:
    from ._final_metrics import compute_final_bass_integration_metrics
    metrics = compute_final_bass_integration_metrics(
        bundle,
        fc_hz,
        profile,
        mode=mode,
        main_hpf_order=int(main_hpf_order),
        sub_lpf_order=int(sub_lpf_order),
        sub_hpf_hz=float(sub_hpf_hz),
        sub_hpf_order=int(sub_hpf_order),
        sub_combine_mode=sub_combine_mode,
        sub_delay_ms=float(sub_delay_ms),
        sub_polarity_invert=bool(sub_polarity_invert),
        sub_gain_trim_db=float(sub_gain_trim_db),
        sub_lpf_hz=sub_lpf_hz,
        sub_allpass_freq_hz=sub_allpass_freq_hz,
        sub_allpass_q=sub_allpass_q,
        guard_lo_ratio=float(guard_lo_ratio),
        guard_hi_ratio=float(guard_hi_ratio),
        l_fir=l_fir,
        r_fir=r_fir,
        sub_fir=sub_fir,
        fir_sample_rate=fir_sample_rate,
        robust=robust,
    )
    return {key: value for key, value in metrics.items() if isinstance(key, str) and key.startswith("bass_")}
