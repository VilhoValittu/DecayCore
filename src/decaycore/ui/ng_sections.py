# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Reusable page and section chrome for the NiceGUI UI."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


@contextmanager
def page_shell(
    *,
    title: str,
    intro: str | None = None,
    wide: bool = False,
    hero_class: str = "",
) -> Iterator[None]:
    """Wrap a tab in the shared Studio Soft page shell."""
    from nicegui import ui

    shell_classes = "cf-page-shell w-full"
    if wide:
        shell_classes += " cf-page-shell-wide"
    with ui.column().classes(shell_classes):
        with ui.column().classes(f"cf-page-hero w-full {hero_class}".strip()):
            ui.label(title).classes("cf-page-title")
            if intro:
                ui.label(intro).classes("cf-page-intro")
        yield


@contextmanager
def section_card(
    *,
    title: str,
    intro: str | None = None,
    hero: bool = False,
    section_classes: str = "",
    card_classes: str = "",
    body_classes: str = "w-full gap-4",
) -> Iterator[None]:
    """Render a titled section with a shared card body."""
    from nicegui import ui

    section_class_name = "cf-section w-full gap-3"
    if section_classes:
        section_class_name += f" {section_classes}"

    card_class_name = "cf-section-card w-full"
    if hero:
        card_class_name += " cf-section-card-hero"
    if card_classes:
        card_class_name += f" {card_classes}"

    with ui.column().classes(section_class_name):
        with ui.row().classes("w-full items-start justify-between gap-4"):
            with ui.column().classes("gap-1"):
                ui.label(title).classes("cf-section-title")
                if intro:
                    ui.label(intro).classes("cf-section-intro")
        with ui.card().classes(card_class_name):
            with ui.column().classes(body_classes):
                yield
