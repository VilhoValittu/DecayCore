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
:root {
    --cf-font-sans: "Avenir Next", "Segoe UI Variable", "Segoe UI", "Helvetica Neue", "Nimbus Sans", sans-serif;
    --cf-font-display: "Avenir Next", "Trebuchet MS", "Segoe UI", sans-serif;
    --cf-bg: #121417;
    --cf-bg-elevated: #181b1f;
    --cf-bg-soft: #1f2328;
    --cf-surface: rgba(255, 248, 239, 0.045);
    --cf-surface-2: rgba(255, 248, 239, 0.075);
    --cf-surface-3: rgba(255, 248, 239, 0.11);
    --cf-surface-accent: rgba(87, 182, 173, 0.12);
    --cf-border: rgba(242, 228, 210, 0.10);
    --cf-border-2: rgba(242, 228, 210, 0.18);
    --cf-border-strong: rgba(242, 228, 210, 0.24);
    --cf-text: #f4ede3;
    --cf-text-strong: #fff8f1;
    --cf-muted: #c2b5a6;
    --cf-faint: #8f8277;
    --cf-accent: #57b6ad;
    --cf-accent-strong: #7ccdc5;
    --cf-accent-warm: #d49a63;
    --cf-success: #71c18f;
    --cf-warning: #d8a05a;
    --cf-danger: #cb6f6f;
    --cf-focus: rgba(87, 182, 173, 0.26);
    --cf-radius-xs: 10px;
    --cf-radius-sm: 14px;
    --cf-radius: 20px;
    --cf-radius-lg: 28px;
    --cf-shadow-soft: 0 18px 40px rgba(0, 0, 0, 0.22);
    --cf-shadow-card: 0 22px 50px rgba(0, 0, 0, 0.24);
    --cf-shadow-hero: 0 32px 80px rgba(0, 0, 0, 0.30);
}

body {
    background:
        radial-gradient(circle at top left, rgba(87, 182, 173, 0.12), transparent 26%),
        radial-gradient(circle at top right, rgba(212, 154, 99, 0.09), transparent 22%),
        linear-gradient(180deg, #16191d 0%, var(--cf-bg) 28%, #111315 100%) !important;
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
    background: linear-gradient(180deg, var(--cf-surface-2), rgba(255, 248, 239, 0.04)) !important;
    border: 1px solid var(--cf-border) !important;
    border-radius: var(--cf-radius) !important;
    box-shadow: var(--cf-shadow-card) !important;
    transition: transform 0.16s ease, box-shadow 0.22s ease, border-color 0.22s ease !important;
}

.q-card:hover {
    border-color: var(--cf-border-strong) !important;
    box-shadow: 0 26px 54px rgba(0, 0, 0, 0.28) !important;
}

.q-expansion-item {
    border: 1px solid var(--cf-border) !important;
    margin-bottom: 8px;
    border-radius: var(--cf-radius-sm);
    overflow: hidden;
    background: rgba(255, 248, 239, 0.025) !important;
}

.q-expansion-item__header {
    background: rgba(255, 248, 239, 0.035) !important;
    transition: background 0.18s ease !important;
}

.q-expansion-item__header:hover {
    background: rgba(255, 248, 239, 0.055) !important;
}

.q-expansion-item__content {
    background: transparent !important;
}

.q-tabs {
    background: transparent !important;
    border-radius: 999px;
}

.q-tabs__content {
    gap: 8px;
    padding: 8px;
    border-radius: 999px;
    background: rgba(255, 248, 239, 0.055) !important;
    border: 1px solid var(--cf-border);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
}

.q-tab {
    min-height: 42px;
    padding: 0 16px;
    color: var(--cf-muted) !important;
    border-radius: 999px;
    letter-spacing: 0.01em;
    transition: color 0.18s ease, background 0.18s ease, transform 0.18s ease !important;
}

.q-tab:hover {
    color: var(--cf-text) !important;
    background: rgba(255, 248, 239, 0.055) !important;
}

.q-tab--active {
    color: #0f1a19 !important;
    background: linear-gradient(135deg, var(--cf-accent) 0%, var(--cf-accent-strong) 100%) !important;
    box-shadow: 0 10px 24px rgba(87, 182, 173, 0.30);
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
    background: rgba(255, 248, 239, 0.055) !important;
    transition: border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease !important;
}

.q-field__label,
.q-select__dropdown-icon {
    color: var(--cf-muted) !important;
}

.q-field--focused .q-field__control {
    background: rgba(255, 248, 239, 0.07) !important;
    box-shadow: 0 0 0 3px var(--cf-focus) !important;
}

.q-field--disabled .q-field__control {
    opacity: 0.82 !important;
    background: rgba(255, 248, 239, 0.035) !important;
}

.q-field--disabled .q-field__label {
    color: var(--cf-muted) !important;
    opacity: 0.64 !important;
}

.q-btn {
    border-radius: 999px !important;
    letter-spacing: 0.02em;
    text-transform: none !important;
    font-weight: 600;
    transition: transform 0.15s ease, box-shadow 0.15s ease, opacity 0.15s ease !important;
}

.q-btn:not([disabled]):hover {
    transform: translateY(-1px);
}

.q-btn--outline {
    border-color: var(--cf-border-2) !important;
}

.q-btn.bg-positive {
    background: linear-gradient(135deg, #5fb987 0%, #7cc89a 100%) !important;
    color: #112217 !important;
}

.q-btn.bg-negative {
    background: linear-gradient(135deg, #cb6f6f 0%, #db8888 100%) !important;
    color: #2c1111 !important;
}

.q-table thead tr th {
    background: rgba(255, 248, 239, 0.08) !important;
    color: var(--cf-muted) !important;
    font-weight: 600;
}

.q-table tbody tr:nth-child(even) {
    background: rgba(255, 248, 239, 0.025) !important;
}

.q-table tbody tr:hover {
    background: rgba(255, 248, 239, 0.055) !important;
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
    border: 1px solid var(--cf-border);
    background:
        radial-gradient(circle at top left, rgba(87, 182, 173, 0.14), transparent 34%),
        radial-gradient(circle at bottom right, rgba(212, 154, 99, 0.10), transparent 26%),
        linear-gradient(180deg, rgba(255, 248, 239, 0.06), rgba(255, 248, 239, 0.03));
    box-shadow: var(--cf-shadow-hero);
    backdrop-filter: blur(10px);
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
    border-radius: 24px;
    overflow: hidden;
    background: linear-gradient(180deg, rgba(6, 8, 10, 0.96), rgba(13, 16, 19, 0.94));
    border: 1px solid rgba(242, 228, 210, 0.10);
    box-shadow:
        0 14px 30px rgba(0, 0, 0, 0.24),
        inset 0 1px 0 rgba(255, 255, 255, 0.04);
}

.cf-brand-kicker {
    color: var(--cf-accent-warm);
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.cf-brand-title {
    color: var(--cf-text-strong);
    font-family: var(--cf-font-display);
    font-size: clamp(2rem, 2.6vw, 3rem);
    font-weight: 700;
    line-height: 1;
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
    border: 1px solid var(--cf-border);
    background: rgba(20, 24, 28, 0.76);
    box-shadow: 0 18px 32px rgba(0, 0, 0, 0.16);
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
    font-size: clamp(1.4rem, 1.5vw, 1.85rem);
    font-weight: 650;
    line-height: 1.15;
}

.cf-page-intro {
    max-width: 56rem;
    color: var(--cf-muted);
    font-size: 0.98rem;
    line-height: 1.65;
}

.cf-section-title {
    color: var(--cf-text-strong);
    font-size: 1rem;
    font-weight: 650;
    letter-spacing: 0.01em;
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
        radial-gradient(circle at top right, rgba(87, 182, 173, 0.12), transparent 34%),
        linear-gradient(180deg, rgba(255, 248, 239, 0.075), rgba(255, 248, 239, 0.04)) !important;
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
    background: rgba(113, 193, 143, 0.12);
    border-color: rgba(113, 193, 143, 0.18);
    color: var(--cf-text);
}

.cf-status-info {
    background: rgba(87, 182, 173, 0.12);
    border-color: rgba(87, 182, 173, 0.16);
    color: var(--cf-text);
}

.cf-auto-bar {
    background: rgba(212, 154, 99, 0.12);
    border-color: rgba(212, 154, 99, 0.16);
    color: var(--cf-text);
}

.cf-progress-phase,
.cf-progress-meta-label {
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.28);
}

.cf-progress-meta {
    padding: 2px 10px;
    border-radius: 999px;
    backdrop-filter: blur(6px);
}

.cf-progress-meta--running {
    background: rgba(18, 26, 33, 0.18);
    border: 1px solid rgba(255, 255, 255, 0.16);
}

.cf-progress-meta--complete {
    background: rgba(255, 248, 239, 0.82);
    border: 1px solid rgba(62, 42, 27, 0.12);
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
        radial-gradient(circle at top right, rgba(87, 182, 173, 0.08), transparent 28%),
        linear-gradient(180deg, var(--cf-bg-elevated), var(--cf-bg-soft)) !important;
    border: 1px solid var(--cf-border-2) !important;
    box-shadow: 0 24px 64px rgba(0, 0, 0, 0.35) !important;
}

.cf-modal-card .nicegui-markdown {
    color: var(--cf-text);
}

.cf-info-panel {
    background: rgba(255, 248, 239, 0.045);
    border: 1px solid var(--cf-border);
    border-radius: 24px;
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
    background: rgba(87, 182, 173, 0.10);
    border: 1px solid rgba(87, 182, 173, 0.18);
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
    background: rgba(10, 11, 13, 0.70) !important;
    backdrop-filter: blur(8px) !important;
}

@keyframes cf-shimmer {
    0% { background-position: -400px 0; }
    100% { background-position: calc(400px + 100%) 0; }
}

.q-linear-progress {
    border-radius: 999px !important;
    overflow: hidden;
    background: rgba(255, 248, 239, 0.06) !important;
}

.q-linear-progress__track {
    background: linear-gradient(
        90deg,
        rgba(255, 248, 239, 0.10) 25%,
        rgba(87, 182, 173, 0.14) 50%,
        rgba(255, 248, 239, 0.10) 75%
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
    background: rgba(242, 228, 210, 0.16);
    border-radius: 999px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(242, 228, 210, 0.26);
}

body.cf-light,
body.body--light {
    --cf-bg: #f2ede6;
    --cf-bg-elevated: #f7f3ed;
    --cf-bg-soft: #ece4da;
    --cf-surface: rgba(62, 42, 27, 0.045);
    --cf-surface-2: rgba(62, 42, 27, 0.07);
    --cf-surface-3: rgba(62, 42, 27, 0.11);
    --cf-surface-accent: rgba(45, 129, 123, 0.10);
    --cf-border: rgba(86, 61, 39, 0.11);
    --cf-border-2: rgba(86, 61, 39, 0.17);
    --cf-border-strong: rgba(86, 61, 39, 0.22);
    --cf-text: #30261d;
    --cf-text-strong: #201812;
    --cf-muted: #6b5a4a;
    --cf-faint: #968271;
    --cf-accent: #2d817b;
    --cf-accent-strong: #4aa69f;
    --cf-accent-warm: #b46f32;
    --cf-success: #3c8d5d;
    --cf-warning: #b3722f;
    --cf-danger: #b85757;
    --cf-focus: rgba(45, 129, 123, 0.18);
    --cf-shadow-soft: 0 18px 40px rgba(76, 56, 38, 0.10);
    --cf-shadow-card: 0 16px 42px rgba(76, 56, 38, 0.12);
    --cf-shadow-hero: 0 24px 70px rgba(76, 56, 38, 0.14);
}

body.cf-light,
body.body--light {
    background:
        radial-gradient(circle at top left, rgba(45, 129, 123, 0.10), transparent 28%),
        radial-gradient(circle at top right, rgba(180, 111, 50, 0.08), transparent 24%),
        linear-gradient(180deg, #f6f1eb 0%, var(--cf-bg) 34%, #eee6db 100%) !important;
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

body.cf-light .cf-tabs-shell-inner,
body.body--light .cf-tabs-shell-inner {
    background: rgba(247, 243, 237, 0.86) !important;
}

body.cf-light .q-tabs__content,
body.body--light .q-tabs__content {
    background: rgba(62, 42, 27, 0.07) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.48);
}

body.cf-light .q-tab--active,
body.body--light .q-tab--active {
    color: #f7f3ed !important;
}

body.cf-light .cf-progress-meta,
body.body--light .cf-progress-meta {
    box-shadow: 0 4px 12px rgba(76, 56, 38, 0.10);
}

body.cf-light .cf-progress-phase,
body.body--light .cf-progress-phase {
    padding: 2px 10px;
    border-radius: 999px;
    background: linear-gradient(90deg, rgba(48, 38, 29, 0.34), rgba(48, 38, 29, 0.18));
}

body.cf-light .cf-progress-meta--running,
body.body--light .cf-progress-meta--running {
    background: rgba(48, 38, 29, 0.34);
    border-color: rgba(255, 255, 255, 0.34);
}

body.cf-light .cf-progress-meta--complete,
body.body--light .cf-progress-meta--complete {
    background: rgba(247, 243, 237, 0.92);
    border-color: rgba(86, 61, 39, 0.16);
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
