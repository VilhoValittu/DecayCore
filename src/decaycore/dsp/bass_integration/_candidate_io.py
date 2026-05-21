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

import dataclasses
from typing import Any

import numpy as np

from ._candidate import DirectDacCandidate, DirectDacCandidateMetrics
from ._utils import _safe_float


def _cfg_get(cfg: Any | None, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    return getattr(cfg, key, default)


def _data_get(data: dict | None, cfg: Any | None, key: str, default: Any) -> Any:
    if isinstance(data, dict) and key in data:
        return data.get(key)
    return _cfg_get(cfg, key, default)


def _order_from_slope(value: Any, default_order: int) -> int:
    try:
        return max(1, int(round(float(value) / 6.0)))
    except Exception:
        return int(default_order)


def direct_dac_candidate_from_data(data: dict | None, cfg: Any | None = None) -> DirectDacCandidate:
    meta = dict((data or {}).get("_bass_integration_meta", {}) or {}) if isinstance(data, dict) else {}
    raw_candidate = dict(meta.get("direct_dac_candidate", {}) or {})

    def value(key: str, data_key: str, cfg_key: str, default: Any) -> Any:
        if key in raw_candidate:
            return raw_candidate.get(key)
        if data_key:
            return _data_get(data, cfg, data_key, default)
        return _cfg_get(cfg, cfg_key, default)

    main_hpf = _safe_float(
        value("main_hpf_hz", "sub_crossover_hz", "sub_crossover_hz", _data_get(data, cfg, "avr_crossover_hz", 80.0)),
        80.0,
    )
    sub_lpf = _safe_float(value("sub_lpf_hz", "direct_dac_sub_lpf_hz", "direct_dac_sub_lpf_hz", main_hpf), main_hpf)
    sub_hpf = _safe_float(value("sub_hpf_hz", "sub_hpf_freq", "sub_hpf_freq", 20.0), 20.0)
    main_order = int(raw_candidate.get("main_hpf_order", _order_from_slope(_data_get(data, cfg, "sub_crossover_slope", 24), 4)))
    sub_lpf_order = int(raw_candidate.get("sub_lpf_order", main_order))
    sub_hpf_order = int(raw_candidate.get("sub_hpf_order", _order_from_slope(_data_get(data, cfg, "sub_hpf_slope", 12), 2)))
    ap_freq = _safe_float(value("sub_allpass_freq_hz", "bass_integration_allpass_freq_hz", "bass_integration_allpass_freq_hz", 0.0), 0.0)
    ap_q = _safe_float(value("sub_allpass_q", "bass_integration_allpass_q", "bass_integration_allpass_q", 0.707), 0.707)
    topology = str(raw_candidate.get("topology", meta.get("sub_topology", "single_sub_bus")) or "single_sub_bus")
    return DirectDacCandidate(
        main_hpf_hz=float(max(1.0, main_hpf)),
        main_hpf_order=max(1, int(main_order)),
        sub_hpf_hz=float(max(0.0, sub_hpf)),
        sub_hpf_order=max(1, int(sub_hpf_order)),
        sub_lpf_hz=float(max(max(1.0, main_hpf), sub_lpf)),
        sub_lpf_order=max(1, int(sub_lpf_order)),
        sub_delay_ms=float(_safe_float(value("sub_delay_ms", "bass_integration_sub_delay_ms", "bass_integration_sub_delay_ms", 0.0), 0.0)),
        sub_gain_trim_db=float(_safe_float(value("sub_gain_trim_db", "bass_integration_sub_gain_trim_db", "bass_integration_sub_gain_trim_db", 0.0), 0.0)),
        sub_polarity_invert=bool(value("sub_polarity_invert", "bass_integration_sub_polarity_invert", "bass_integration_sub_polarity_invert", False)),
        sub_allpass_enabled=bool(raw_candidate.get("sub_allpass_enabled", _data_get(data, cfg, "bass_integration_allpass_auto_applied", False))) and ap_freq > 0.0,
        sub_allpass_freq_hz=float(ap_freq if np.isfinite(ap_freq) else 0.0),
        sub_allpass_q=float(ap_q if np.isfinite(ap_q) and ap_q > 0.0 else 0.707),
        topology=topology if topology in ("single_sub_bus", "dual_sub_vector_average_reference") else "single_sub_bus",
        source=str(raw_candidate.get("source", "data") or "data"),
    )


def _candidate_dict(candidate: DirectDacCandidate) -> dict[str, Any]:
    return dict(dataclasses.asdict(candidate))


def _metrics_dict(metrics: DirectDacCandidateMetrics | None) -> dict[str, Any]:
    if metrics is None:
        return {}
    return {
        "score": float(metrics.score),
        "objective": float(metrics.objective),
        "feasible": bool(metrics.feasible),
        "reject_reasons": list(metrics.reject_reasons),
        "worst_channel": str(metrics.dominant_channel),
        "left_score": float(metrics.left.score),
        "right_score": float(metrics.right.score),
        "feasibility_class": str(metrics.summary.get("feasibility_class", "unknown")),
        "feasibility_reason": str(metrics.summary.get("feasibility_reason", "")),
        "export_model": "camilladsp_yaml_compatible",
    }


def write_direct_dac_candidate_to_data(
    data: dict,
    candidate: DirectDacCandidate,
    metrics: DirectDacCandidateMetrics | None = None,
    *,
    reason: str,
) -> None:
    data["bass_integration_mode"] = "direct_dac"
    data["sub_crossover_hz"] = float(candidate.main_hpf_hz)
    data["avr_crossover_hz"] = float(candidate.main_hpf_hz)
    data["direct_dac_sub_lpf_hz"] = float(candidate.sub_lpf_hz)
    data["sub_hpf_freq"] = float(candidate.sub_hpf_hz)
    data["bass_integration_sub_delay_ms"] = float(candidate.sub_delay_ms)
    data["bass_integration_sub_gain_trim_db"] = float(candidate.sub_gain_trim_db)
    data["bass_integration_sub_polarity_invert"] = bool(candidate.sub_polarity_invert)
    data["bass_integration_allpass_auto_applied"] = bool(candidate.sub_allpass_enabled)
    data["bass_integration_allpass_freq_hz"] = float(candidate.sub_allpass_freq_hz if candidate.sub_allpass_enabled else 0.0)
    data["bass_integration_allpass_q"] = float(candidate.sub_allpass_q if candidate.sub_allpass_enabled else 0.707)
    bi_meta = data.setdefault("_bass_integration_meta", {})
    if not isinstance(bi_meta, dict):
        bi_meta = {}
        data["_bass_integration_meta"] = bi_meta
    bi_meta.update(
        {
            "enabled": True,
            "mode": "direct_dac",
            "optimization": "automatic",
            "metric_channel_mode": "worst_case",
            "export_model": "camilladsp_yaml_compatible",
            "sub_topology": str(candidate.topology),
            "direct_dac_candidate": _candidate_dict(candidate),
            "direct_dac_metrics": _metrics_dict(metrics),
            "reason": str(reason),
            "avr_crossover_hz": float(candidate.main_hpf_hz),
            "direct_dac_sub_lpf_hz": float(candidate.sub_lpf_hz),
            "recommended_crossover_hz": float(candidate.main_hpf_hz),
            "recommended_sub_lpf_hz": float(candidate.sub_lpf_hz),
        }
    )
    if metrics is not None:
        diagnostics = dict(bi_meta.get("diagnostics", {}) or {})
        diagnostics.update(dict(metrics.summary or {}))
        diagnostics.pop("eval_bundle", None)
        diagnostics.pop("left", None)
        diagnostics.pop("right", None)
        bi_meta["diagnostics"] = diagnostics
