"""Lazy Automatic-mode reporting facade for shared manual/export paths."""

from __future__ import annotations

import math
from typing import Any


def _rank_score_module():
    try:
        from ..auto_mode import rank_score
    except ModuleNotFoundError:
        return None
    return rank_score


def attach_official_rank_score(
    metrics: dict[str, Any] | None,
    *,
    components: dict[str, Any] | None = None,
) -> dict[str, Any]:
    module = _rank_score_module()
    if module is not None:
        return module.attach_official_rank_score(metrics, components=components)
    return dict(metrics or {})


def official_rank_score(metrics: dict[str, Any] | None) -> float:
    module = _rank_score_module()
    if module is not None:
        return float(module.official_rank_score(metrics))
    values = dict(metrics or {})
    for value in (
        values.get("rank_score_official"),
        dict(values.get("rank_score_components", {}) or {}).get("rank_score"),
        values.get("rank_score"),
    ):
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            return parsed
    return float("nan")


def calibrated_auto_quality(
    rank_score_0_100: Any,
    metrics: dict[str, Any] | None = None,
) -> float:
    module = _rank_score_module()
    if module is not None:
        return float(module.calibrated_auto_quality(rank_score_0_100, metrics))
    try:
        raw = float(rank_score_0_100)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(raw):
        return float("nan")
    clipped = min(100.0, max(0.0, raw))
    return 100.0 * (1.0 - (1.0 - clipped / 100.0) ** 2.35)


def display_rank_score(score: Any) -> float:
    module = _rank_score_module()
    if module is not None:
        return float(module.display_rank_score(score))
    try:
        parsed = float(score)
    except (TypeError, ValueError):
        return float("nan")
    return min(100.0, max(0.0, parsed)) if math.isfinite(parsed) else float("nan")


def quality_band(score: float) -> str:
    module = _rank_score_module()
    if module is not None:
        return str(module._quality_band(score))
    if score >= 90.0:
        return "Excellent"
    if score >= 80.0:
        return "Good"
    if score >= 70.0:
        return "Usable"
    if score >= 60.0:
        return "Weak"
    return "Poor"


def compute_run_ranking_score_components(**kwargs: Any) -> dict[str, Any]:
    module = _rank_score_module()
    if module is not None:
        return dict(module.compute_run_ranking_score_components(**kwargs))
    avg = float(kwargs.get("avg_score", 0.0) or 0.0)
    penalties = {
        key: max(0.0, float(kwargs.get(key, 0.0) or 0.0))
        for key in ("boost_penalty", "event_penalty", "lr_delta_penalty", "dsp_penalty")
    }
    score = min(100.0, max(0.0, avg - sum(penalties.values())))
    components = {"rank_score": score, "avg_score": avg, **penalties}
    return {
        **components,
        "run_ranking_score": score,
        "run_ranking_score_components": dict(components),
    }


def build_auto_mode_audit_trail(**kwargs: Any) -> dict[str, Any]:
    try:
        from ..auto_mode.audit_trail import build_auto_mode_audit_trail as implementation
    except ModuleNotFoundError:
        return {}
    return dict(implementation(**kwargs) or {})


_quality_band = quality_band

__all__ = [
    "_quality_band",
    "attach_official_rank_score",
    "build_auto_mode_audit_trail",
    "calibrated_auto_quality",
    "compute_run_ranking_score_components",
    "display_rank_score",
    "official_rank_score",
    "quality_band",
]
