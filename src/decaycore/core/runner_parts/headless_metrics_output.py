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

import json
import logging
import math
from pathlib import Path
from typing import Any


from ...auto_mode.audit_trail import build_auto_mode_audit_trail
from ...auto_mode.rank_score import attach_official_rank_score, official_rank_score
from ...version import VERSION

from .headless_progress import _git_commit

logger = logging.getLogger("DecayCore")

























































def _load_optional_metadata(config_dir: Path, config: dict) -> tuple[dict, dict, dict]:
    from .headless_export_bundle import _read_json

    metadata = {}
    if isinstance(config.get("measurement_metadata"), dict):
        metadata.update(config.get("measurement_metadata") or {})
    meta_path = config_dir / "measurement_metadata.json"
    if meta_path.exists():
        metadata.update(_read_json(meta_path))
    rt60 = dict(config.get("rt60", {}) or {}) if isinstance(config.get("rt60"), dict) else {}
    rt60_path = config_dir / "rt60.json"
    if rt60_path.exists():
        rt60.update(_read_json(rt60_path))
    harmonics = dict(config.get("harmonics", {}) or {}) if isinstance(config.get("harmonics"), dict) else {}
    harm_path = config_dir / "harmonics.json"
    if harm_path.exists():
        harmonics.update(_read_json(harm_path))
    return metadata, rt60, harmonics

def _f(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except (

        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        RuntimeError,
        OSError,
        ImportError,
        ModuleNotFoundError,
        NameError,
    ):
        return default
    return default

def _pick(stats: list[dict], keys: list[str], default: float | None = None) -> float | None:
    vals: list[float] = []
    for st in stats:
        for key in keys:
            val = _f(st.get(key), None)
            if val is not None:
                vals.append(float(val))
                break
    if not vals:
        return default
    return float(max(vals, key=abs)) if len(vals) > 1 else float(vals[0])

def _build_auto_audit_metrics(auto_meta: dict, best_metrics: dict) -> dict:
    if not auto_meta and not best_metrics:
        return {}
    audit = dict(auto_meta.get("audit_trail", {}) or {}) if isinstance(auto_meta, dict) else {}
    if not audit:
        audit = build_auto_mode_audit_trail(
            best_metrics=best_metrics,
            best_preset=dict(auto_meta.get("best_preset", {}) or {}),
            winner_explanation=dict(auto_meta.get("winner_explanation", {}) or {}),
            residual_peak_safety_override_meta=dict(auto_meta.get("residual_peak_safety_override", {}) or {}),
            optimizer_backend=str(auto_meta.get("optimizer_backend", "builtin") or "builtin"),
            goal=str(auto_meta.get("auto_goal", "balanced") or "balanced"),
            selection_basis=str(auto_meta.get("selection_basis", "rank_score") or "rank_score"),
            top=list(auto_meta.get("top", []) or []),
            phase1_ok=int(auto_meta.get("trials_phase1_ok", 0) or 0),
            phase2_ok=int(auto_meta.get("trials_phase2_ok", 0) or 0),
            phase1_tried=int(auto_meta.get("trials_phase1_total", 0) or 0),
            phase2_tried=int(auto_meta.get("trials_phase2_total", 0) or 0),
            phase3_total=int(auto_meta.get("trials_phase3_total", 0) or 0),
            phase3_ok=int(auto_meta.get("trials_phase3_ok", 0) or 0),
            phase4_steps=dict(auto_meta.get("phase4_steps", {}) or {}),
            fs_v=int(auto_meta.get("search_fs", 0) or 0),
            taps_v=int(auto_meta.get("search_taps", 0) or 0),
            trials_total=int(auto_meta.get("trials_total", 0) or 0),
            trials_ok=int(auto_meta.get("trials_ok", 0) or 0),
            source="headless_reconstructed",
        )
    selection = dict(audit.get("selection", {}) or {})
    winner = dict(audit.get("winner", {}) or {})
    hard_gates = dict(audit.get("hard_gates", {}) or {})
    search = dict(audit.get("search", {}) or {})
    return {
        "schema_version": int(audit.get("schema_version", 1) or 1),
        "winner_summary": str(winner.get("summary", "") or ""),
        "hard_gate_status": str(hard_gates.get("status", "passed") or "passed"),
        "hard_gate_failures": list(hard_gates.get("hard_gate_failures", []) or []),
        "rank_score_official": _f(winner.get("rank_score_official"), None),
        "avg_score": _f(winner.get("avg_score"), None),
        "trials_ok": int(search.get("trials_ok", auto_meta.get("trials_ok", 0)) or 0),
        "trials_total": int(search.get("trials_total", auto_meta.get("trials_total", 0)) or 0),
        "optimizer_backend": str(selection.get("optimizer_backend", auto_meta.get("optimizer_backend", "")) or ""),
    }

def _extract_rt60(rt60: dict, ctx: dict | None) -> dict[str, float]:
    out: dict[str, float] = {}
    src = rt60.get("rt60", rt60)
    if isinstance(src, dict):
        for k, v in src.items():
            key = str(k).lower().replace("hz", "").replace("k", "000").replace(".5", "").strip()
            val = _f(v, None)
            if val is not None:
                out[key] = float(val)
    for side in ("l", "r"):
        bands = (ctx or {}).get(f"measured_rt60_bands_{side}") if isinstance(ctx, dict) else None
        if isinstance(bands, dict):
            for k, v in bands.items():
                val = _f(v, None)
                if val is not None:
                    out[str(int(round(float(k))))] = float(val)
    return out

def _extract_harmonics(harmonics: dict) -> dict[str, float]:
    src = harmonics.get("harmonics", harmonics)
    risks: list[tuple[float, float]] = []
    if isinstance(src, dict):
        for item in src.values():
            if isinstance(item, dict):
                for freq, db in item.items():
                    f_hz = _f(str(freq).lower().replace("hz", "").replace("k", "000"), None)
                    db_val = _f(db, None)
                    if f_hz is not None and db_val is not None:
                        risks.append((float(f_hz), max(0.0, 60.0 + float(db_val))))
    if not risks:
        return {}
    def band(lo: float, hi: float) -> float:
        vals = [r for f, r in risks if lo <= f < hi]
        return float(max(vals)) if vals else 0.0
    max_pair = max(risks, key=lambda p: p[1], default=(0.0, 0.0))
    return {
        "risk_20_80": band(20.0, 80.0),
        "risk_80_200": band(80.0, 200.0),
        "risk_200_500": band(200.0, 500.0),
        "risk_max": float(max_pair[1]),
        "risk_freq_hz": float(max_pair[0]),
    }

def _build_metrics(status: str, data: dict, ctx: dict | None, metadata: dict, rt60_src: dict, harmonics_src: dict, *, started_at: str, finished_at: str, runtime_s: float, warnings: list[str], errors: list[str]) -> dict:
    results = list((ctx or {}).get("results_by_fs", []) or [])
    result = results[-1] if results else None
    stats = []
    if result is not None:
        stats = [dict(getattr(result, "l_st", {}) or {}), dict(getattr(result, "r_st", {}) or {})]
    auto_meta = dict(data.get("_auto_mode_meta", {}) or {})
    best_metrics = attach_official_rank_score(auto_meta.get("best_metrics", {}) or {})
    auto_score = _f(official_rank_score(best_metrics), None)
    avg_score = _f(best_metrics.get("avg_score"), None)
    metric_sources = list(stats)
    if result is not None and isinstance(getattr(result, "metrics", None), dict):
        metric_sources.append(dict(getattr(result, "metrics", {}) or {}))
    metric_sources.append(best_metrics)

    mag_error_rms_db = _pick(
        metric_sources,
        [
            "mag_error_rms_db",
            "magnitude_error_rms_db",
            "magnitude_rms_db",
            "error_rms_db",
            "response_error_rms_db",
            "fr_error_rms_db",
            "mag_rms",
            "rms_db",
            "real_mag_error_rms_20_200",
            "real_mag_error_rms",
            "pred_mag_error_rms_20_200",
            "pred_mag_error_rms",
            "target_tracking_rms_20_200_db",
            "realized_rms_20_200_db",
            "rms_error_db",
        ],
    )
    mag_error_max_db = _pick(
        metric_sources,
        [
            "mag_error_max_db",
            "magnitude_error_max_db",
            "magnitude_max_db",
            "error_max_db",
            "response_error_max_db",
            "fr_error_max_db",
            "mag_max",
            "max_error_db",
            "real_mag_error_max_20_200",
            "real_mag_error_max",
            "pred_mag_error_max_20_200",
            "pred_mag_error_max",
            "target_tracking_max_20_200_db",
            "target_tracking_peak_abs_20_200_db",
            "max_abs_error_db",
        ],
    )

    metrics: dict[str, Any] = {
        "auto_score": auto_score,
        "filter_score": auto_score if auto_score is not None else avg_score,
        "mag_error_rms_db": mag_error_rms_db,
        "mag_error_max_db": mag_error_max_db,
        "residual_peak_count": _pick(stats, ["residual_peak_count", "events_total"], 0.0),
        "residual_peak_max_db": _pick(stats, ["residual_peak_max_db", "events_severity"], 0.0),
        "gd_spike_count": _pick(stats, ["gd_spike_count"], 0.0),
        "gd_max_ms": _pick(stats, ["gd_abs_max_20_500_ms", "mixed_excess_delay_before_ms", "gd_grad_limiter_after_max_ms_per_oct"]),
        "gd_before_rms_ms": _pick(stats, ["phase_realized_gd_before_rms_ms"]),
        "gd_after_rms_ms": _pick(stats, ["phase_realized_gd_after_rms_ms"]),
        "gd_improvement_frac": _pick(stats, ["phase_realized_gd_improvement_score"]),
        "max_boost_db": _pick(stats, ["net_boost_peak_db", "max_net_boost_db", "max_boost_db"], _f(data.get("max_boost"), None)),
        "max_cut_db": -abs(float(_f(data.get("max_cut_db"), 0.0) or 0.0)),
        "bass_ripple_20_200_db": _pick(stats, ["bass_ripple_20_200_db", "ripple_rms"], None),
        "tdc_strength": _f(data.get("tdc_strength"), None),
        "confidence_pull": _f(data.get("conf_pull_floor"), None),
        "filter_taps": int(getattr(result, "taps", data.get("taps", 0)) or 0) if (result is not None or data.get("taps")) else None,
        "optuna_trials": int(auto_meta.get("trials_ok", auto_meta.get("trials_total", 0)) or 0),
        "schroeder_hz_estimate": _pick(stats, ["schroeder_hz_estimate"]),
    }

    bi_meta = dict(data.get("_bass_integration_meta", {}) or {})
    bi_diag = dict(bi_meta.get("diagnostics", {}) or {})
    align = dict(bi_meta.get("alignment", {}) or {})
    bass = {
        "enabled": bool(data.get("bass_integration_enable", False)),
        "mode": str(data.get("bass_integration_mode", "") or ""),
        "crossover_hz": _f(data.get("avr_crossover_hz", data.get("sub_crossover_hz")), None),
        "delay_ms": _f(align.get("delay_ms", data.get("bass_integration_sub_delay_ms")), None),
        "gain_db": _f(align.get("gain_trim_db", data.get("bass_integration_sub_gain_trim_db")), None),
        "polarity": "inverted" if bool(data.get("bass_integration_sub_polarity_invert", False)) else "normal",
        "overlap_ripple_db": _f(bi_diag.get("overlap_ripple_db", bi_diag.get("bass_overlap_ripple_db")), None),
        "cancellation_risk": _f(bi_diag.get("cancellation_risk", bi_diag.get("bass_cancellation_risk")), None),
        "sub_dominance_db": _f(bi_diag.get("sub_dominance_db", bi_diag.get("bass_sub_dominance_db")), None),
        "xo_gd_delta_ms": _f(bi_diag.get("xo_gd_delta_ms", bi_diag.get("bass_xo_gd_mismatch_delta_ms")), None),
        "phase_delta_deg": _f(bi_diag.get("phase_delta_deg", bi_diag.get("bass_phase_delta_deg")), None),
    }
    for key, value in bass.items():
        flat = "bi_enabled" if key == "enabled" else f"bi_{key}"
        metrics[flat] = value

    rt60 = _extract_rt60(rt60_src, ctx)
    harmonics = _extract_harmonics(harmonics_src)
    for key, value in rt60.items():
        metrics[f"rt60_{key}"] = value
    for key, value in harmonics.items():
        metrics[f"harmonic_{key}"] = value
        if key.startswith("risk_"):
            metrics[f"boost_{key}"] = value

    warnings_out = list(warnings or [])
    if status == "success" and (mag_error_rms_db is None or mag_error_max_db is None):
        missing = [
            key
            for key, value in (("mag_error_rms_db", mag_error_rms_db), ("mag_error_max_db", mag_error_max_db))
            if value is None
        ]
        warnings_out.append("Magnitude error metrics unavailable from DSP stats: " + ", ".join(missing))

    return {
        "status": status,
        "version": str(VERSION),
        "git_commit": _git_commit(),
        "started_at": started_at,
        "finished_at": finished_at,
        "runtime_s": float(runtime_s),
        "mode": str(data.get("mode", "") or "").lower(),
        "target_curve_mode": str(data.get("auto_target_mode", "") or ""),
        "selected_house_curve": str(data.get("hc_mode", "") or ""),
        "metrics": {k: v for k, v in metrics.items() if v is not None},
        "bass_integration": bass,
        "auto_audit": _build_auto_audit_metrics(auto_meta, best_metrics),
        "rt60": rt60,
        "harmonics": harmonics,
        "metadata": metadata,
        "warnings": warnings_out,
        "errors": list(errors or []),
    }

def _write_summary(path: Path, metrics_doc: dict) -> None:
    m = dict(metrics_doc.get("metrics", {}) or {})
    bi = dict(metrics_doc.get("bass_integration", {}) or {})
    lines = [
        f"Status: {metrics_doc.get('status')}",
        f"Runtime: {float(metrics_doc.get('runtime_s', 0.0) or 0.0):.2f} s",
        f"Auto score: {m.get('auto_score', 'n/a')}",
        f"Filter score: {m.get('filter_score', 'n/a')}",
        f"Mag error RMS/max: {m.get('mag_error_rms_db', 'n/a')} / {m.get('mag_error_max_db', 'n/a')} dB",
        f"Residual peak count/max: {m.get('residual_peak_count', 'n/a')} / {m.get('residual_peak_max_db', 'n/a')} dB",
        f"GD spike count/max: {m.get('gd_spike_count', 'n/a')} / {m.get('gd_max_ms', 'n/a')} ms",
        f"Realized GD RMS before/after: {m.get('gd_before_rms_ms', 'n/a')} / {m.get('gd_after_rms_ms', 'n/a')} ms",
        f"Realized GD improvement: {m.get('gd_improvement_frac', 'n/a')}",
        f"Max boost/cut: {m.get('max_boost_db', 'n/a')} / {m.get('max_cut_db', 'n/a')} dB",
        f"TDC strength: {m.get('tdc_strength', 'n/a')}",
        f"Confidence Pull: {m.get('confidence_pull', 'n/a')}",
        f"Filter taps: {m.get('filter_taps', 'n/a')}",
        f"Optuna trials: {m.get('optuna_trials', 'n/a')}",
    ]
    if bi.get("enabled"):
        lines.append(
            "Bass integration: "
            f"mode={bi.get('mode', 'n/a')}, crossover={bi.get('crossover_hz', 'n/a')} Hz, "
            f"delay={bi.get('delay_ms', 'n/a')} ms, gain={bi.get('gain_db', 'n/a')} dB, "
            f"polarity={bi.get('polarity', 'n/a')}"
        )
    schroeder = m.get("schroeder_hz_estimate")
    if schroeder is not None:
        lines.append(f"Schroeder frequency estimate: {schroeder:.1f} Hz")
    if metrics_doc.get("rt60"):
        lines.append("RT60 summary: " + ", ".join(f"{k}={v}" for k, v in dict(metrics_doc["rt60"]).items()))
    if metrics_doc.get("harmonics"):
        h = dict(metrics_doc["harmonics"])
        lines.append(f"Harmonic risk summary: max={h.get('risk_max', 0.0)} at {h.get('risk_freq_hz', 0.0)} Hz")
    if metrics_doc.get("warnings"):
        lines.append("Warnings: " + "; ".join(metrics_doc["warnings"]))
    if metrics_doc.get("errors"):
        lines.append("Errors: " + "; ".join(metrics_doc["errors"]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _write_outputs(output_dir: Path, metrics_doc: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics_doc, indent=2, sort_keys=True), encoding="utf-8")
    _write_summary(output_dir / "summary.txt", metrics_doc)


__all__ = [
    '_load_optional_metadata',
    '_f',
    '_pick',
    '_extract_rt60',
    '_extract_harmonics',
    '_build_metrics',
    '_write_summary',
    '_write_outputs',
]
