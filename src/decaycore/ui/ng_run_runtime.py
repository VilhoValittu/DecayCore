# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Runtime references shared by the Run tab and results renderer."""

from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger("DecayCore")

_results_container_ref: Any | None = None
_progress_ref: Any | None = None
_progress_overlay_refs: list[Any] = []
_progress_meta_refs: list[Any] = []


def get_results_container() -> Any | None:
    return _results_container_ref


def set_results_container(container: Any | None) -> None:
    global _results_container_ref
    _results_container_ref = container


def get_progress_element() -> Any | None:
    return _progress_ref


def set_progress_elements(
    progress: Any | None,
    *,
    overlay_refs: list[Any],
    meta_refs: list[Any],
) -> None:
    global _progress_ref, _progress_overlay_refs, _progress_meta_refs
    _progress_ref = progress
    _progress_overlay_refs = list(overlay_refs)
    _progress_meta_refs = list(meta_refs)
    _set_progress_overlay_text_dark(False)
    _set_progress_meta_completed(False)


def _set_progress_overlay_text_dark(enabled: bool) -> None:
    add_class = "text-black" if enabled else "text-white"
    remove_class = "text-white" if enabled else "text-black"
    for label in _progress_overlay_refs:
        try:
            label.classes(add=add_class, remove=remove_class)
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
            logger.debug("Failed to update progress overlay text color", exc_info=True)


def _set_progress_meta_completed(enabled: bool) -> None:
    add_class = "cf-progress-meta--complete" if enabled else "cf-progress-meta--running"
    remove_class = "cf-progress-meta--running" if enabled else "cf-progress-meta--complete"
    for meta in _progress_meta_refs:
        try:
            meta.classes(add=add_class, remove=remove_class)
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
            logger.debug("Failed to update progress meta visual state", exc_info=True)


def set_progress_visual_state(*, completed: bool) -> None:
    progress = get_progress_element()
    if progress is not None:
        try:
            progress.set_text_color("light-green-4" if completed else "primary")
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
            logger.debug("Failed to update progress bar color", exc_info=True)
    _set_progress_meta_completed(enabled=completed)
    _set_progress_overlay_text_dark(enabled=completed)


__all__ = [
    "get_progress_element",
    "get_results_container",
    "set_progress_elements",
    "set_progress_visual_state",
    "set_results_container",
]
