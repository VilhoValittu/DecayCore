# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""NiceGUI dark theme for DecayCore.

Call apply_theme() once during page setup (before any components are created).
"""
from __future__ import annotations

_CF_CSS = """
@font-face {
    font-family: "Silkscreen";
    src: url("/static/Silkscreen-Regular.ttf") format("truetype");
    font-style: normal;
    font-weight: 400;
    font-display: swap;
}

@font-face {
    font-family: "Silkscreen";
    src: url("/static/Silkscreen-Bold.ttf") format("truetype");
    font-style: normal;
    font-weight: 700;
    font-display: swap;
}

:root {
    --cf-font-sans: "Avenir Next", "Segoe UI Variable", "Segoe UI", "Helvetica Neue", "Nimbus Sans", sans-serif;
    --cf-font-display: "Silkscreen", "Cascadia Mono", "Courier New", monospace;
    --cf-bg: #17133f;
    --cf-bg-elevated: #211a56;
    --cf-bg-soft: #292064;
    --cf-surface: rgba(198, 190, 255, 0.07);
    --cf-surface-2: rgba(198, 190, 255, 0.12);
    --cf-surface-3: rgba(198, 190, 255, 0.18);
    --cf-surface-accent: rgba(101, 215, 222, 0.15);
    --cf-border: rgba(174, 163, 235, 0.28);
    --cf-border-2: rgba(174, 163, 235, 0.42);
    --cf-border-strong: rgba(174, 163, 235, 0.62);
    --cf-text: #f4f1ff;
    --cf-text-strong: #ffffff;
    --cf-muted: #c9c1e7;
    --cf-faint: #9189b3;
    --cf-accent: #65d7de;
    --cf-accent-strong: #8be8ec;
    --cf-accent-warm: #cbb8ff;
    --cf-success: #72d69d;
    --cf-warning: #f0c36c;
    --cf-danger: #ee7a8a;
    --cf-focus: rgba(101, 215, 222, 0.30);
    --cf-radius-xs: 4px;
    --cf-radius-sm: 6px;
    --cf-radius: 8px;
    --cf-radius-lg: 12px;
    --cf-shadow-soft: 3px 3px 0 rgba(6, 4, 25, 0.38);
    --cf-shadow-card: 5px 5px 0 rgba(6, 4, 25, 0.42);
    --cf-shadow-hero: 8px 8px 0 rgba(6, 4, 25, 0.46);
}

body {
    background:
        radial-gradient(circle at top left, rgba(101, 215, 222, 0.12), transparent 27%),
        radial-gradient(circle at top right, rgba(158, 139, 230, 0.16), transparent 25%),
        linear-gradient(180deg, #21195a 0%, var(--cf-bg) 36%, #120f35 100%) !important;
    color: var(--cf-text) !important;
    font-family: var(--cf-font-sans);
}

.nicegui-content,
.q-layout,
.q-page,
.q-page-container {
    background: transparent !important;
    color: var(--cf-text) !important;
}

.nicegui-markdown,
.q-item__label,
.q-field__native,
.q-field__input,
.q-field__marginal,
.q-checkbox__label,
.q-radio__label,
.q-toggle__label,
.q-expansion-item__label,
.q-expansion-item__caption,
.q-card,
.q-dialog,
.q-table,
.q-btn {
    font-family: var(--cf-font-sans);
}

.q-card {
    background: linear-gradient(180deg, var(--cf-surface-2), var(--cf-surface)) !important;
    border: 1px solid var(--cf-border) !important;
    border-radius: var(--cf-radius) !important;
    box-shadow: var(--cf-shadow-card) !important;
    transition: transform 0.16s ease, box-shadow 0.22s ease, border-color 0.22s ease !important;
}

.q-card:hover {
    border-color: var(--cf-border-strong) !important;
    box-shadow: 6px 6px 0 rgba(6, 4, 25, 0.48) !important;
}

.q-expansion-item {
    border: 1px solid var(--cf-border) !important;
    margin-bottom: 8px;
    border-radius: var(--cf-radius-sm);
    overflow: hidden;
    background: var(--cf-surface) !important;
}

.q-expansion-item__header {
    background: var(--cf-surface) !important;
    transition: background 0.18s ease !important;
}

.q-expansion-item__header:hover {
    background: var(--cf-surface-2) !important;
}

.q-expansion-item__content {
    background: transparent !important;
}

.q-tabs {
    background: transparent !important;
    border-radius: var(--cf-radius-sm);
}

.q-tabs__content {
    gap: 4px;
    padding: 6px;
    border-radius: var(--cf-radius-sm);
    background: var(--cf-surface-2) !important;
    border: 1px solid var(--cf-border-2);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

.q-tab {
    min-height: 42px;
    padding: 0 14px;
    color: var(--cf-muted) !important;
    border-radius: var(--cf-radius-xs);
    font-family: var(--cf-font-display);
    font-size: 0.75rem;
    letter-spacing: 0.025em;
    transition: color 0.18s ease, background 0.18s ease, transform 0.18s ease !important;
}

.q-tab:hover {
    color: var(--cf-text) !important;
    background: var(--cf-surface-3) !important;
}

.q-tab--active {
    color: #17133f !important;
    background: var(--cf-accent-strong) !important;
    box-shadow: 3px 3px 0 rgba(6, 4, 25, 0.36);
}

.q-tabs__content .q-tab__indicator {
    display: none !important;
}

.q-field__control,
.q-field__native,
.q-select,
.q-input {
    border-radius: var(--cf-radius-sm) !important;
}

.q-field--outlined .q-field__control:before,
.q-field--outlined .q-field__control:after {
    border-color: var(--cf-border) !important;
    border-width: 1px !important;
}

.q-field__control {
    background: var(--cf-surface-2) !important;
    transition: border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease !important;
}

.q-field__label,
.q-select__dropdown-icon {
    color: var(--cf-muted) !important;
}

.q-field--focused .q-field__control {
    background: var(--cf-surface-3) !important;
    box-shadow: 0 0 0 2px var(--cf-focus) !important;
}

.q-field--disabled .q-field__control {
    opacity: 0.82 !important;
    background: var(--cf-surface) !important;
}

.q-field--disabled .q-field__label {
    color: var(--cf-muted) !important;
    opacity: 0.64 !important;
}

.q-btn {
    border-radius: var(--cf-radius-xs) !important;
    letter-spacing: 0.02em;
    text-transform: none !important;
    font-weight: 600;
    transition: transform 0.15s ease, box-shadow 0.15s ease, opacity 0.15s ease !important;
}

.q-btn--round {
    border-radius: 50% !important;
}

.q-btn:not([disabled]):hover {
    transform: translate(-1px, -1px);
    box-shadow: 3px 3px 0 rgba(6, 4, 25, 0.34);
}

.q-btn--outline {
    border-color: var(--cf-border-2) !important;
}

.q-btn.bg-primary,
.q-btn.bg-secondary {
    background: var(--cf-accent) !important;
    color: #17133f !important;
}

.q-btn.text-primary,
.q-btn.text-secondary {
    color: var(--cf-accent) !important;
}

.q-btn.bg-positive {
    background: var(--cf-success) !important;
    color: #102719 !important;
}

.q-btn.bg-negative {
    background: var(--cf-danger) !important;
    color: #351018 !important;
}

.q-table thead tr th {
    background: var(--cf-surface-3) !important;
    color: var(--cf-muted) !important;
    font-weight: 600;
}

.q-table tbody tr:nth-child(even) {
    background: var(--cf-surface) !important;
}

.q-table tbody tr:hover {
    background: var(--cf-surface-2) !important;
}

.cf-brand-shell {
    padding: 22px 22px 8px;
    background: transparent;
}

.cf-brand-hero {
    max-width: 1380px;
    margin: 0 auto;
    padding: 24px 28px;
    border-radius: var(--cf-radius-lg);
    border: 2px solid var(--cf-border-2);
    background:
        repeating-linear-gradient(
            0deg,
            rgba(203, 184, 255, 0.035) 0,
            rgba(203, 184, 255, 0.035) 1px,
            transparent 1px,
            transparent 4px
        ),
        radial-gradient(circle at top left, rgba(101, 215, 222, 0.16), transparent 35%),
        radial-gradient(circle at bottom right, rgba(158, 139, 230, 0.18), transparent 30%),
        linear-gradient(180deg, rgba(62, 49, 132, 0.82), rgba(35, 28, 91, 0.86));
    box-shadow: var(--cf-shadow-hero);
}

.cf-brand-block {
    min-width: 0;
}

.cf-brand-logo-frame {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 96px;
    height: 96px;
    padding: 0;
    border-radius: var(--cf-radius);
    overflow: hidden;
    background: linear-gradient(180deg, #0b0922, #151038);
    border: 2px solid var(--cf-border-2);
    box-shadow: 5px 5px 0 rgba(6, 4, 25, 0.48);
}

.cf-brand-logo-frame .nicegui-html {
    width: 100%;
    height: 100%;
}

.cf-brand-kicker {
    color: var(--cf-accent-warm);
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-family: var(--cf-font-display);
}

.cf-brand-title {
    color: var(--cf-text-strong);
    font-family: var(--cf-font-display);
    font-size: clamp(1.65rem, 2.2vw, 2.4rem);
    font-weight: 700;
    line-height: 1.2;
    letter-spacing: 0.02em;
}

.cf-brand-subtitle {
    color: var(--cf-muted);
    max-width: 44rem;
    font-size: 0.98rem;
    line-height: 1.6;
}

.cf-brand-version {
    color: var(--cf-faint);
    font-size: 0.86rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    font-family: var(--cf-font-display);
}

.cf-brand-actions {
    align-items: stretch;
}

.cf-tabs-shell {
    position: sticky;
    top: 0;
    z-index: 1000;
    padding: 10px 22px 0;
    background: transparent;
    backdrop-filter: none;
}

.cf-tabs-shell-inner {
    max-width: 1380px;
    margin: 0 auto;
    padding: 8px 10px 12px;
    border-radius: var(--cf-radius-lg);
    border: 2px solid var(--cf-border-2);
    background:
        repeating-linear-gradient(
            0deg,
            rgba(203, 184, 255, 0.03) 0,
            rgba(203, 184, 255, 0.03) 1px,
            transparent 1px,
            transparent 4px
        ),
        rgba(31, 24, 82, 0.94);
    box-shadow: var(--cf-shadow-card);
}

.cf-tab-panels-shell {
    background: transparent !important;
}

.cf-page-shell {
    max-width: 1200px;
    margin: 0 auto;
    padding: 24px 24px 72px;
    gap: 22px;
}

.cf-page-shell-wide {
    max-width: 1360px;
}

.cf-page-hero {
    padding: 6px 2px 4px;
    gap: 8px;
}

.cf-page-title {
    color: var(--cf-text-strong);
    font-family: var(--cf-font-display);
    font-size: clamp(1.15rem, 1.35vw, 1.55rem);
    font-weight: 700;
    line-height: 1.35;
}

.cf-page-intro {
    max-width: 56rem;
    color: var(--cf-muted);
    font-size: 0.98rem;
    line-height: 1.65;
}

.cf-section-title {
    color: var(--cf-text-strong);
    font-family: var(--cf-font-display);
    font-size: 0.9rem;
    font-weight: 700;
    line-height: 1.45;
    letter-spacing: 0.015em;
}

.cf-section-intro {
    max-width: 54rem;
    color: var(--cf-muted);
    font-size: 0.88rem;
    line-height: 1.55;
}

.cf-section-card {
    padding: 6px;
}

.cf-section-card-hero {
    background:
        radial-gradient(circle at top right, rgba(101, 215, 222, 0.14), transparent 34%),
        linear-gradient(180deg, var(--cf-surface-2), var(--cf-surface)) !important;
}

.cf-stack-tight {
    gap: 12px;
}

.cf-status-summary,
.cf-status-info,
.cf-auto-bar {
    padding: 12px 16px;
    border-radius: var(--cf-radius-sm);
    border: 1px solid transparent;
    animation: cf-fadein 0.22s ease both;
}

.cf-status-summary {
    background: color-mix(in srgb, var(--cf-success) 14%, transparent);
    border-color: color-mix(in srgb, var(--cf-success) 28%, transparent);
    color: var(--cf-text);
}

.cf-status-info {
    background: var(--cf-surface-accent);
    border-color: color-mix(in srgb, var(--cf-accent) 28%, transparent);
    color: var(--cf-text);
}

.cf-auto-bar {
    background: color-mix(in srgb, var(--cf-warning) 14%, transparent);
    border-color: color-mix(in srgb, var(--cf-warning) 28%, transparent);
    color: var(--cf-text);
}

.cf-progress-phase,
.cf-progress-meta-label {
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.28);
}

.cf-progress-meta {
    padding: 2px 10px;
    border-radius: var(--cf-radius-xs);
}

.cf-progress-meta--running {
    background: rgba(15, 11, 48, 0.52);
    border: 1px solid rgba(255, 255, 255, 0.16);
}

.cf-progress-meta--complete {
    background: rgba(244, 241, 255, 0.90);
    border: 1px solid var(--cf-border-2);
}

.cf-auto-details-scroll {
    width: 100%;
    max-height: min(40vh, 24rem);
    overflow-y: auto;
    overflow-x: auto;
    padding: 8px 10px 8px 8px;
}

.cf-modal-card {
    background:
        radial-gradient(circle at top right, rgba(101, 215, 222, 0.10), transparent 28%),
        linear-gradient(180deg, var(--cf-bg-elevated), var(--cf-bg-soft)) !important;
    border: 1px solid var(--cf-border-2) !important;
    box-shadow: 8px 8px 0 rgba(6, 4, 25, 0.48) !important;
}

.cf-modal-card .nicegui-markdown {
    color: var(--cf-text);
}

.cf-info-panel {
    background: var(--cf-surface);
    border: 1px solid var(--cf-border);
    border-radius: var(--cf-radius);
    padding: 12px 16px;
    min-width: 260px;
    max-width: 360px;
    text-align: right;
    font-size: 0.78rem;
    line-height: 1.7;
    color: var(--cf-text);
    white-space: nowrap;
    overflow: hidden;
    flex-shrink: 0;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
}

.cf-info-line-dim {
    color: var(--cf-muted);
}

.cf-info-line-score {
    color: var(--cf-accent-strong);
    font-weight: 700;
}

.cf-info-line-ok {
    color: var(--cf-success);
}

.cf-info-line-warn {
    color: var(--cf-warning);
}

.cf-adv-summary {
    background: var(--cf-surface-accent);
    border: 1px solid color-mix(in srgb, var(--cf-accent) 30%, transparent);
    border-radius: var(--cf-radius-sm);
    padding: 12px 14px;
}

.cf-adv-summary-text {
    color: var(--cf-text) !important;
}

.cf-target-hint-summary {
    color: var(--cf-text) !important;
}

.cf-target-hint-detail {
    color: var(--cf-muted) !important;
}

.cf-adv-preset-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 4px;
}

.q-dialog__backdrop {
    background: rgba(8, 6, 29, 0.74) !important;
    backdrop-filter: blur(4px) !important;
}

@keyframes cf-shimmer {
    0% { background-position: -400px 0; }
    100% { background-position: calc(400px + 100%) 0; }
}

.q-linear-progress {
    border-radius: var(--cf-radius-xs) !important;
    overflow: hidden;
    background: var(--cf-surface) !important;
}

.q-linear-progress__track {
    background: linear-gradient(
        90deg,
        var(--cf-surface-2) 25%,
        var(--cf-surface-accent) 50%,
        var(--cf-surface-2) 75%
    ) !important;
    background-size: 400px 100% !important;
    animation: cf-shimmer 1.8s linear infinite !important;
}

@keyframes cf-fadein {
    from { opacity: 0; transform: translateY(-4px); }
    to { opacity: 1; transform: none; }
}

::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: var(--cf-border-2);
    border-radius: var(--cf-radius-xs);
}

::-webkit-scrollbar-thumb:hover {
    background: var(--cf-border-strong);
}

@media (max-width: 760px) {
    .q-tab {
        padding: 0 9px;
        font-size: 0.68rem;
    }

    .cf-brand-title {
        font-size: 1.4rem;
    }

    .cf-page-title {
        font-size: 1.05rem;
    }

    .cf-section-title {
        font-size: 0.82rem;
    }
}

body.cf-light,
body.body--light {
    --cf-bg: #e9e4f5;
    --cf-bg-elevated: #f6f3fb;
    --cf-bg-soft: #ded6ed;
    --cf-surface: rgba(56, 39, 94, 0.05);
    --cf-surface-2: rgba(56, 39, 94, 0.09);
    --cf-surface-3: rgba(56, 39, 94, 0.14);
    --cf-surface-accent: rgba(25, 120, 137, 0.12);
    --cf-border: rgba(77, 58, 120, 0.24);
    --cf-border-2: rgba(77, 58, 120, 0.36);
    --cf-border-strong: rgba(77, 58, 120, 0.52);
    --cf-text: #2f2547;
    --cf-text-strong: #211833;
    --cf-muted: #64597a;
    --cf-faint: #8c809f;
    --cf-accent: #197889;
    --cf-accent-strong: #2a91a0;
    --cf-accent-warm: #6750a5;
    --cf-success: #267a4e;
    --cf-warning: #98631a;
    --cf-danger: #a53d51;
    --cf-focus: rgba(25, 120, 137, 0.20);
    --cf-shadow-soft: 3px 3px 0 rgba(55, 38, 91, 0.10);
    --cf-shadow-card: 5px 5px 0 rgba(55, 38, 91, 0.13);
    --cf-shadow-hero: 8px 8px 0 rgba(55, 38, 91, 0.16);
}

body.cf-light,
body.body--light {
    background:
        radial-gradient(circle at top left, rgba(25, 120, 137, 0.11), transparent 28%),
        radial-gradient(circle at top right, rgba(103, 80, 165, 0.12), transparent 25%),
        linear-gradient(180deg, #f4f1fa 0%, var(--cf-bg) 38%, #e1daef 100%) !important;
}

body.cf-light .cf-brand-hero,
body.body--light .cf-brand-hero,
body.cf-light .cf-tabs-shell-inner,
body.body--light .cf-tabs-shell-inner,
body.cf-light .cf-info-panel,
body.body--light .cf-info-panel,
body.cf-light .q-card,
body.body--light .q-card,
body.cf-light .cf-modal-card,
body.body--light .cf-modal-card {
    box-shadow: var(--cf-shadow-card) !important;
}

body.cf-light .cf-brand-hero,
body.body--light .cf-brand-hero {
    background:
        repeating-linear-gradient(
            0deg,
            rgba(77, 58, 120, 0.035) 0,
            rgba(77, 58, 120, 0.035) 1px,
            transparent 1px,
            transparent 4px
        ),
        radial-gradient(circle at top left, rgba(25, 120, 137, 0.13), transparent 35%),
        radial-gradient(circle at bottom right, rgba(103, 80, 165, 0.12), transparent 30%),
        linear-gradient(180deg, rgba(246, 243, 251, 0.96), rgba(225, 218, 239, 0.96));
}

body.cf-light .cf-tabs-shell-inner,
body.body--light .cf-tabs-shell-inner {
    background:
        repeating-linear-gradient(
            0deg,
            rgba(77, 58, 120, 0.03) 0,
            rgba(77, 58, 120, 0.03) 1px,
            transparent 1px,
            transparent 4px
        ),
        rgba(246, 243, 251, 0.96) !important;
}

body.cf-light .q-tabs__content,
body.body--light .q-tabs__content {
    background: var(--cf-surface-2) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.48);
}

body.cf-light .q-tab--active,
body.body--light .q-tab--active {
    color: #17133f !important;
    box-shadow: 3px 3px 0 rgba(55, 38, 91, 0.18);
}

body.cf-light .cf-progress-meta,
body.body--light .cf-progress-meta {
    box-shadow: var(--cf-shadow-soft);
}

body.cf-light .cf-progress-phase,
body.body--light .cf-progress-phase {
    padding: 2px 10px;
    border-radius: var(--cf-radius-xs);
    background: linear-gradient(90deg, rgba(47, 37, 71, 0.34), rgba(47, 37, 71, 0.18));
}

body.cf-light .cf-progress-meta--running,
body.body--light .cf-progress-meta--running {
    background: rgba(47, 37, 71, 0.34);
    border-color: rgba(255, 255, 255, 0.34);
}

body.cf-light .cf-progress-meta--complete,
body.body--light .cf-progress-meta--complete {
    background: rgba(246, 243, 251, 0.94);
    border-color: var(--cf-border);
}
"""


def apply_theme(*, dark: bool = True):
    """Inject theme CSS and enable NiceGUI dark mode.

    Must be called inside a @ui.page handler, before other components.
    Returns the DarkMode element so callers can toggle it later.
    """
    from nicegui import ui  # noqa: PLC0415
    dark_mode = ui.dark_mode()
    if dark:
        dark_mode.enable()
    else:
        dark_mode.disable()
    ui.add_css(_CF_CSS)
    return dark_mode
