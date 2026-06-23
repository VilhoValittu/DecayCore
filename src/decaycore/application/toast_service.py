# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Toast notification service with deduplication and edge-triggered alerts."""

from __future__ import annotations

import time
from typing import Any

from .health_service import HealthResult
from ..resources.i8n.decaycore_i18n import t

_TOAST_LAST_SHOWN: dict[str, float] = {}
_TOAST_EDGE_STATE: dict[str, bool] = {}
_TOAST_CALLABLE: callable | None = None


def _prune_toast_cache(
    *,
    now: float,
    max_items: int = 256,
    max_age_s: float = 300.0,
) -> None:
    """Prune old entries from toast deduplication cache."""
    try:
        if not _TOAST_LAST_SHOWN:
            return

        if max_age_s > 0:
            old_keys = [k for k, ts in _TOAST_LAST_SHOWN.items() if (now - float(ts)) > float(max_age_s)]
            for k in old_keys:
                _TOAST_LAST_SHOWN.pop(k, None)

        if max_items > 0 and len(_TOAST_LAST_SHOWN) > int(max_items):
            items: list[tuple[str, float]] = [(k, float(v)) for k, v in _TOAST_LAST_SHOWN.items()]
            items.sort(key=lambda kv: kv[1], reverse=True)
            keep = set(k for k, _ in items[: int(max_items)])
            for k in list(_TOAST_LAST_SHOWN.keys()):
                if k not in keep:
                    _TOAST_LAST_SHOWN.pop(k, None)
    except (TypeError, ValueError):
        return


def _get_toast_callable():
    return _TOAST_CALLABLE


def set_toast_callable(fn: callable | None) -> None:
    global _TOAST_CALLABLE
    _TOAST_CALLABLE = fn


def show_toast(
    msg: str,
    *,
    duration: float = 5.0,
    color: str | None = None,
    dedupe_key: str | None = None,
    dedupe_window_s: float = 0.75,
) -> bool:
    key = str(dedupe_key or msg or "").strip()
    if key:
        now = time.monotonic()
        _prune_toast_cache(now=now)
        last = float(_TOAST_LAST_SHOWN.get(key, 0.0) or 0.0)
        if (now - last) < float(dedupe_window_s):
            return False
        _TOAST_LAST_SHOWN[key] = now

    fn = _get_toast_callable()
    if not callable(fn):
        return False
    try:
        if color is None:
            fn(msg, duration=duration)
        else:
            fn(msg, duration=duration, color=color)
        return True
    except Exception:
        return False


def _toast_on_edge(
    *,
    edge_key: str,
    active: bool,
    msg: str,
    duration: float,
    color: str | None = None,
    dedupe_key: str | None = None,
) -> bool:
    prev = bool(_TOAST_EDGE_STATE.get(edge_key, False))
    _TOAST_EDGE_STATE[edge_key] = bool(active)
    if active and not prev:
        return show_toast(
            msg,
            duration=duration,
            color=color,
            dedupe_key=dedupe_key or edge_key,
            dedupe_window_s=0.25,
        )
    return False


def _tr(key: str, **kwargs: Any) -> str:
    text = t(key)
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError, TypeError, ValueError):
            return text
    return text


def _health_toast_color(level: str, mode_u: str) -> str | None:
    if level == "warn":
        mode_s = str(mode_u or "").strip().upper()
        return "warn" if mode_s in ("BASIC", "AUTO") else "info"
    if level == "crit":
        return "error"
    return None


def toast_health_gate_result(hr: HealthResult, mode: str) -> bool:
    from .health_service import format_health_summary

    mode_u = str(mode or "BASIC").strip().upper()
    if hr.blocked:
        msg = format_health_summary(hr) or "Fix errors before running (blocked in BASIC/AUTO)."
        show_toast(
            msg,
            duration=15.0,
            color=_health_toast_color("crit", mode_u),
            dedupe_key=f"health_blocked:{msg}",
            dedupe_window_s=1.0,
        )
        return True

    if hr.overall in ("warn", "crit"):
        msg = format_health_summary(hr)
        if msg:
            show_toast(
                msg,
                duration=10.0,
                color=_health_toast_color(hr.overall, mode_u),
                dedupe_key=f"health_summary:{mode_u}:{hr.overall}:{msg}",
                dedupe_window_s=1.0,
            )
    return False


def toast_mode_defaults_applied(mode: str) -> None:
    try:
        msg = _tr("mode_defaults_applied_toast", mode=mode)
    except (KeyError, IndexError, TypeError, ValueError):
        msg = f"Mode defaults applied: {mode}"
    show_toast(
        msg,
        color="success",
        duration=2.0,
        dedupe_key=f"mode_defaults_applied:{mode}",
    )


def toast_tdc_preset_applied(name: str) -> None:
    show_toast(
        f"TDC preset applied: {name}",
        color="success",
        duration=1.5,
        dedupe_key=f"tdc_preset:{name}",
    )


def toast_afdw_preset_applied(name: str) -> None:
    show_toast(
        f"A-FDW preset applied: {name}",
        color="success",
        duration=1.5,
        dedupe_key=f"afdw_preset:{name}",
    )


def toast_max_boost_over_cap(value: Any, max_safe_boost: float) -> None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        _TOAST_EDGE_STATE["max_boost_over_cap"] = False
        return

    cap = float(max_safe_boost if max_safe_boost is not None else 0.0)
    over = (cap > 0.0) and (v > cap + 1e-9)
    if not over:
        _TOAST_EDGE_STATE["max_boost_over_cap"] = False
        return

    try:
        cap_suffix = _tr("max_boost_help_cap", value=f"{cap:.1f}")
    except (KeyError, IndexError, TypeError, ValueError):
        cap_suffix = f" (capped to {cap:.1f} dB)"
    msg = f"{_tr('max_boost')}: {v:.1f} dB > {cap:.1f} dB{cap_suffix}"
    _toast_on_edge(
        edge_key="max_boost_over_cap",
        active=True,
        msg=msg,
        duration=5.0,
        dedupe_key="max_boost_over_cap",
    )


def toast_taps_over_cap(value: Any, max_safe_taps: int) -> None:
    try:
        v = int(value)
    except (TypeError, ValueError):
        _TOAST_EDGE_STATE["taps_over_cap"] = False
        return

    cap = int(max_safe_taps if max_safe_taps is not None else 0)
    over = v > cap
    if not over:
        _TOAST_EDGE_STATE["taps_over_cap"] = False
        return

    try:
        msg = _tr("taps_warn_over", value=cap)
    except (KeyError, IndexError, TypeError, ValueError):
        msg = f"Taps > {cap}: very high latency and diminishing returns."
    _toast_on_edge(
        edge_key="taps_over_cap",
        active=True,
        msg=msg,
        duration=6.0,
        dedupe_key="taps_over_cap",
    )
