# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""NiceGUI element registry – replaces PyWebIO pin/pin_update/pin_on_change pattern.

All form elements created in tab builders call register() to store their
reference here.  Callbacks and cross-tab logic read/write via this module
instead of PyWebIO's global pin dict.

PyWebIO → NiceGUI equivalents
------------------------------
    pin["name"]              → ng_controls.value("name")
    pin.get("name", default) → ng_controls.value("name", default)
    pin_update("name", value=v)          → ng_controls.set_value("name", v)
    pin_update("name", options=[...])    → ng_controls.set_options("name", [...])
    pin_on_change("name", onchange=fn)   → ng_controls.on_change("name", fn)
    put_scope("s") / use_scope("s")      → ng_controls.get_container("s")
"""
from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger("DecayCore")

# name → NiceGUI element
_CONTROLS: dict[str, Any] = {}
# name → NiceGUI container (column/row/html) used as dynamic scope
_CONTAINERS: dict[str, Any] = {}
_SUPPRESSED_CALLBACKS: set[str] = set()


# ---------------------------------------------------------------------------
# Element registry
# ---------------------------------------------------------------------------

def register(name: str, element: Any) -> Any:
    """Register a NiceGUI element under *name*.  Returns the element."""
    _CONTROLS[name] = element
    return element


def get(name: str) -> Any | None:
    """Return the registered element, or None."""
    return _CONTROLS.get(name)


def value(name: str, default: Any = None) -> Any:
    """Return current value of the registered element."""
    el = _CONTROLS.get(name)
    if el is None:
        return default
    try:
        v = el.value
        return v if v is not None else default
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
        logger.debug("value(%r) read failed", name, exc_info=True)
        return default


# ---------------------------------------------------------------------------
# Value + option updates  (pin_update equivalent)
# ---------------------------------------------------------------------------

def set_value(name: str, v: Any, *, emit: bool = True) -> None:
    el = _CONTROLS.get(name)
    if el is None:
        return
    if not emit:
        _SUPPRESSED_CALLBACKS.add(name)
    try:
        try:
            el.set_value(v)
        except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
            try:
                el.value = v
                el.update()
            except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
                logger.debug("set_value(%r, %r) failed", name, v, exc_info=True)
    finally:
        if not emit:
            _SUPPRESSED_CALLBACKS.discard(name)


def set_options(name: str, options: list | dict) -> None:
    """Replace the options of a select/radio element."""
    el = _CONTROLS.get(name)
    if el is None:
        return
    try:
        el.set_options(options)
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
        logger.debug("set_options(%r) failed", name, exc_info=True)


def set_enabled(name: str, enabled: bool) -> None:
    """Enable or disable a form element."""
    el = _CONTROLS.get(name)
    if el is None:
        return
    try:
        if enabled:
            el.enable()
        else:
            el.disable()
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
        try:
            if enabled:
                el.props(remove="disable")
            else:
                el.props("disable")
        except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
            logger.debug("set_enabled(%r, %r) failed", name, enabled, exc_info=True)


def set_visibility(name: str, visible: bool) -> None:
    """Show or hide a registered element."""
    el = _CONTROLS.get(name)
    if el is None:
        el = _CONTAINERS.get(name)
    if el is None:
        return
    try:
        el.set_visibility(visible)
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
        logger.debug("set_visibility(%r, %r) failed", name, visible, exc_info=True)


# ---------------------------------------------------------------------------
# Change callbacks  (pin_on_change equivalent)
# ---------------------------------------------------------------------------

def on_change(name: str, callback: Callable) -> None:
    """Register a value-change callback on the named element.

    The callback receives the new value as its first argument.
    """
    el = _CONTROLS.get(name)
    if el is None:
        logger.debug("on_change: element %r not registered yet", name)
        return

    def _wrapped(e: Any) -> None:
        if name in _SUPPRESSED_CALLBACKS:
            return
        callback(e.value)

    try:
        el.on_value_change(_wrapped)
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
        try:
            el.on("change", _wrapped)
        except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
            logger.debug("on_change(%r) failed", name, exc_info=True)


def on_commit(name: str, callback: Callable) -> None:
    """Register a commit-style callback on the named element.

    Prefers the native ``change`` event so typing into number/text fields does
    not trigger writeback logic on every keystroke. Falls back to
    ``on_value_change`` for controls that do not expose ``change`` separately.
    """
    el = _CONTROLS.get(name)
    if el is None:
        logger.debug("on_commit: element %r not registered yet", name)
        return

    def _wrapped(e: Any) -> None:
        if name in _SUPPRESSED_CALLBACKS:
            return
        callback(e.value)

    try:
        el.on("change", _wrapped)
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
        try:
            el.on_value_change(_wrapped)
        except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
            logger.debug("on_commit(%r) failed", name, exc_info=True)


# ---------------------------------------------------------------------------
# Dynamic containers  (put_scope / use_scope equivalent)
# ---------------------------------------------------------------------------

def register_container(name: str, container: Any) -> Any:
    """Register a NiceGUI container (column/expansion/etc.) as a scope."""
    _CONTAINERS[name] = container
    return container


def get_container(name: str) -> Any | None:
    """Return the registered container, or None."""
    return _CONTAINERS.get(name)


def clear_container(name: str) -> None:
    """Clear the contents of a registered container."""
    c = _CONTAINERS.get(name)
    if c is None:
        return
    try:
        c.clear()
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
        logger.debug("clear_container(%r) failed", name, exc_info=True)


def reset() -> None:
    """Clear all registrations.  Call once per page load to avoid stale refs."""
    _CONTROLS.clear()
    _CONTAINERS.clear()
    _SUPPRESSED_CALLBACKS.clear()


# ---------------------------------------------------------------------------
# Value holder for non-element values (e.g. uploaded file data)
# ---------------------------------------------------------------------------

class _ValueChangeEvent:
    """Minimal event object for reactive holders."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value


class _ValueHolder:
    """Minimal element-compatible holder for values without a UI widget.

    Used for file uploads and logical proxy values. Supports reactive
    callbacks so hidden state can drive preview refreshes like normal controls.
    """

    __slots__ = ("_callbacks", "value")

    def __init__(self, value: Any = None) -> None:
        self.value = value
        self._callbacks: list[Callable[[Any], None]] = []

    def set_value(self, value: Any) -> None:
        self.value = value
        event = _ValueChangeEvent(value)
        for callback in list(self._callbacks):
            try:
                callback(event)
            except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
                logger.debug("_ValueHolder callback failed", exc_info=True)

    def update(self) -> None:
        """Compatibility no-op for controls updated via direct assignment."""

    def on_value_change(self, callback: Callable[[Any], None]) -> None:
        self._callbacks.append(callback)


# ---------------------------------------------------------------------------
# PyWebIO pin-compatible proxy  (passed to collect_ui_data as pin_obj)
# ---------------------------------------------------------------------------

class NgPinProxy:
    """Dict-like proxy over ng_controls, compatible with collect_ui_data(pin).

    PyWebIO pin access pattern:  pin[key]  or  pin.get(key, default)
    NiceGUI equivalent:           NgPinProxy()[key]
    """

    def __getitem__(self, key: str) -> Any:
        return value(key)

    def get(self, key: str, default: Any = None) -> Any:
        return value(key, default)

    def __contains__(self, key: str) -> bool:
        return key in _CONTROLS
