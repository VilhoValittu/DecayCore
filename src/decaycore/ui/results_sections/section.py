# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Shared rendering helpers for results metric sections."""
from __future__ import annotations

import html
from typing import Any

from ...resources.i8n.decaycore_i18n import t


def _esc(value: Any) -> str:
    return html.escape(str(value) if value is not None else "-")


def _metric_table_html(rows: list[dict]) -> str:
    """Build an HTML table from metric_row() / fmt_* dicts."""
    shared: list[tuple[str, str]] = []
    stereo: list[tuple[str, str, str]] = []

    for row in rows:
        left = dict(row.get("left", {}) or {})
        right = dict(row.get("right", {}) or {})
        label = str(row.get("label", "") or "")
        if str(left.get("compare", "")) == str(right.get("compare", "")):
            shared.append((label, left.get("render", "-")))
        else:
            stereo.append((label, left.get("render", "-"), right.get("render", "-")))

    parts = []
    if shared:
        parts.append(
            "<table style='width:100%;border-collapse:collapse;font-size:12px;'>"
            "<thead><tr>"
            f"<th style='text-align:left;padding:4px 8px;background:rgba(255,255,255,0.06);'>{_esc(t('results_table_metric'))}</th>"
            f"<th style='text-align:left;padding:4px 8px;background:rgba(255,255,255,0.06);'>{_esc(t('results_table_value'))}</th>"
            "</tr></thead><tbody>"
            + "".join(
                f"<tr><td style='padding:3px 8px;border-top:1px solid rgba(255,255,255,0.06);'>{_esc(label)}</td>"
                f"<td style='padding:3px 8px;border-top:1px solid rgba(255,255,255,0.06);'>{value}</td></tr>"
                for label, value in shared
            )
            + "</tbody></table>"
        )
    if stereo:
        parts.append(
            "<table style='width:100%;border-collapse:collapse;font-size:12px;margin-top:8px;'>"
            "<thead><tr>"
            f"<th style='text-align:left;padding:4px 8px;background:rgba(255,255,255,0.06);'>{_esc(t('results_table_metric'))}</th>"
            f"<th style='text-align:left;padding:4px 8px;background:rgba(255,255,255,0.06);'>{_esc(t('results_table_left_short'))}</th>"
            f"<th style='text-align:left;padding:4px 8px;background:rgba(255,255,255,0.06);'>{_esc(t('results_table_right_short'))}</th>"
            "</tr></thead><tbody>"
            + "".join(
                f"<tr><td style='padding:3px 8px;border-top:1px solid rgba(255,255,255,0.06);'>{_esc(label)}</td>"
                f"<td style='padding:3px 8px;border-top:1px solid rgba(255,255,255,0.06);'>{left}</td>"
                f"<td style='padding:3px 8px;border-top:1px solid rgba(255,255,255,0.06);'>{right}</td></tr>"
                for label, left, right in stereo
            )
            + "</tbody></table>"
        )
    return "\n".join(parts)


def _section(title: str, rows: list[dict], summary_lines: list[str] | None = None) -> None:
    """Render a collapsible metric section."""
    from nicegui import ui  # noqa: PLC0415

    with ui.expansion(title).classes("w-full"):
        for line in list(summary_lines or []):
            if line:
                ui.markdown(str(line))
        table_html = _metric_table_html(rows)
        if table_html:
            ui.html(table_html)
