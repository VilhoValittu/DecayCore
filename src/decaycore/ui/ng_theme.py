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
    --cf-bg:        #0b0f14;
    --cf-surface:   rgba(255,255,255,0.04);
    --cf-surface-2: rgba(255,255,255,0.07);
    --cf-surface-3: rgba(255,255,255,0.11);
    --cf-border:    rgba(255,255,255,0.10);
    --cf-border-2:  rgba(255,255,255,0.18);
    --cf-text:      #cdd1d5;
    --cf-muted:     #9aa3ae;
    --cf-faint:     #5a6070;
    --cf-accent:    #6ba8f0;
    --cf-focus:     rgba(107,168,240,0.30);
    --cf-radius:    8px;
    --cf-radius-sm: 4px;
    --cf-shadow:    0 2px 12px rgba(0,0,0,0.45);
}

body { background: var(--cf-bg) !important; color: var(--cf-text) !important; }

/* Quasar card/surface overrides */
.q-card       { background: var(--cf-surface-2) !important; border: 1px solid var(--cf-border) !important; }
.q-expansion-item { border: 1px solid var(--cf-border) !important; margin-bottom: 4px; border-radius: var(--cf-radius-sm); }
.q-expansion-item__header { background: var(--cf-surface-2) !important; }
.q-expansion-item__content { background: var(--cf-bg) !important; }

/* Tabs */
.q-tabs      { background: var(--cf-surface-2) !important; }
.q-tab       { color: var(--cf-muted) !important; }
.q-tab--active { color: var(--cf-accent) !important; }
.q-tabs__content .q-tab__indicator { background: var(--cf-accent) !important; }

/* Header layout */
.cf-brand-shell {
    background: var(--cf-bg);
}
.cf-tabs-shell {
    position: sticky;
    top: 0;
    z-index: 1000;
    background: var(--cf-bg);
    box-shadow: 0 6px 14px rgba(0,0,0,0.24);
}
.cf-tabs-shell .q-separator { background: var(--cf-border) !important; }

/* Inputs */
.q-field__control { background: var(--cf-surface-2) !important; border-radius: var(--cf-radius-sm) !important; }
.q-field__label   { color: var(--cf-muted) !important; }
.q-field__native, .q-field__input { color: var(--cf-text) !important; }
.q-select__dropdown-icon { color: var(--cf-muted) !important; }

/* Buttons */
.q-btn:not(.bg-primary):not(.bg-positive):not(.bg-negative):not(.bg-warning) {
    border: 1px solid var(--cf-border-2) !important;
}

/* Table */
.q-table thead tr th { background: var(--cf-surface-3) !important; color: var(--cf-muted) !important; }
.q-table tbody tr:nth-child(even) { background: var(--cf-surface) !important; }
.q-table tbody tr:hover           { background: var(--cf-surface-2) !important; }

/* Scrollbar */
::-webkit-scrollbar            { width: 8px; height: 8px; }
::-webkit-scrollbar-track      { background: var(--cf-bg); }
::-webkit-scrollbar-thumb      { background: var(--cf-border-2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--cf-muted); }

/* Brand header */
.cf-brand-logo-wrap { display: flex; align-items: center; gap: 12px; padding: 8px 0; }
.cf-brand-logo      { font-size: 1.6rem; font-weight: 700; letter-spacing: 2px;
                      color: var(--cf-text); user-select: none; }
.cf-brand-version   { font-size: 0.75rem; color: var(--cf-muted); align-self: flex-end; padding-bottom: 3px; }
.cf-brand-title     { color: #ffffff; }

/* Status area */
.cf-status-summary { background: rgba(40,120,60,0.18); border-left: 3px solid #3a9a5c;
                     padding: 8px 12px; border-radius: var(--cf-radius-sm); margin-bottom: 6px; }
.cf-status-info    { background: rgba(40,90,160,0.18); border-left: 3px solid #3a72c8;
                     padding: 8px 12px; border-radius: var(--cf-radius-sm); margin-bottom: 6px; }
.cf-status-text    { color: var(--cf-muted); font-size: 0.85rem; }
.cf-auto-bar       { background: rgba(40,120,60,0.22); border-left: 3px solid #3a9a5c;
                     padding: 6px 12px; border-radius: var(--cf-radius-sm); font-size: 0.85rem; }
.cf-auto-details-scroll {
    width: 100%;
    max-height: min(40vh, 24rem);
    overflow-y: auto;
    overflow-x: auto;
    padding: 8px 10px 8px 8px;
}

/* Dialogs */
.q-dialog__backdrop {
    background: rgba(0,0,0,0.78) !important;
}
.cf-modal-card {
    background: #151a22 !important;
    border: 1px solid var(--cf-border-2) !important;
    box-shadow: 0 18px 48px rgba(0,0,0,0.62) !important;
}
.cf-modal-card .nicegui-markdown {
    color: var(--cf-text);
}

/* Info panel (top-right header summary) */
.cf-info-panel {
    background: var(--cf-surface-2);
    border: 1px solid var(--cf-border);
    border-radius: var(--cf-radius-sm);
    padding: 6px 12px;
    min-width: 220px;
    max-width: 340px;
    text-align: right;
    font-size: 0.78rem;
    line-height: 1.6;
    color: var(--cf-text);
    white-space: nowrap;
    overflow: hidden;
    flex-shrink: 0;
}
.cf-info-line-dim   { color: var(--cf-muted); }
.cf-info-line-score { color: var(--cf-accent); font-weight: 600; }
.cf-info-line-ok    { color: #3a9a5c; }
.cf-info-line-warn  { color: #c07a30; }

/* Advanced guided sections */
.cf-adv-summary {
    background: rgba(79,142,247,0.10);
    border: 1px solid var(--cf-border-2);
    border-radius: var(--cf-radius-sm);
    padding: 10px 12px;
}
.cf-adv-preset-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 2px;
}

/* Disabled field readability */
.q-field--disabled .q-field__control { opacity: 0.82 !important; }
.q-field--disabled .q-field__label   { color: var(--cf-text) !important; opacity: 0.55 !important; }
.q-field--disabled .q-field__native,
.q-field--disabled .q-field__input   { color: var(--cf-text) !important; opacity: 0.72 !important; }
.q-field--disabled .q-select__dropdown-icon { opacity: 0.45 !important; }

/* ── Micro-interactions: base transitions ─────────────────────────────────── */
.q-btn         { transition: transform 0.15s ease, box-shadow 0.15s ease, opacity 0.15s ease !important; }
.q-card        { transition: box-shadow 0.2s ease, border-color 0.2s ease !important; }
.q-field__control { transition: border-color 0.18s ease, box-shadow 0.18s ease !important; }
.q-tab         { transition: color 0.18s ease !important; }
.q-expansion-item__header { transition: background 0.18s ease !important; }

/* Hover effects */
.q-btn:not([disabled]):hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(107,168,240,0.18) !important;
}
.q-card:hover {
    border-color: var(--cf-border-2) !important;
    box-shadow: 0 6px 24px rgba(0,0,0,0.50) !important;
}
.q-tab:not(.q-tab--active):hover { color: var(--cf-text) !important; }
.q-expansion-item__header:hover  { background: var(--cf-surface-3) !important; }

/* Input focus glow */
.q-field--focused .q-field__control {
    border-color: var(--cf-accent) !important;
    box-shadow: 0 0 0 2px var(--cf-focus) !important;
}

/* ── Modal backdrop blur ──────────────────────────────────────────────────── */
.q-dialog__backdrop { backdrop-filter: blur(4px) !important; }

/* ── Header gradient ─────────────────────────────────────────────────────── */
.cf-brand-shell {
    background: linear-gradient(180deg, rgba(107,168,240,0.07) 0%, var(--cf-bg) 100%);
}

/* ── Progress bar shimmer ────────────────────────────────────────────────── */
@keyframes cf-shimmer {
    0%   { background-position: -400px 0; }
    100% { background-position: calc(400px + 100%) 0; }
}
.q-linear-progress__track {
    background: linear-gradient(
        90deg,
        var(--cf-surface-3) 25%,
        rgba(107,168,240,0.12) 50%,
        var(--cf-surface-3) 75%
    ) !important;
    background-size: 400px 100% !important;
    animation: cf-shimmer 1.8s linear infinite !important;
}

/* ── Status bars: fade-in when appearing ─────────────────────────────────── */
@keyframes cf-fadein {
    from { opacity: 0; transform: translateY(-4px); }
    to   { opacity: 1; transform: none; }
}
.cf-status-summary, .cf-status-info, .cf-auto-bar {
    animation: cf-fadein 0.22s ease both;
}

/* Light mode overrides (activated by body.cf-light) */
body.cf-light {
    --cf-bg:        #eef1f5;
    --cf-surface:   rgba(0,0,0,0.06);
    --cf-surface-2: rgba(0,0,0,0.10);
    --cf-surface-3: rgba(0,0,0,0.15);
    --cf-border:    rgba(0,0,0,0.20);
    --cf-border-2:  rgba(0,0,0,0.35);
    --cf-text:      #111620;
    --cf-muted:     #374151;
    --cf-faint:     #6b7280;
    --cf-accent:    #1048a0;
    --cf-focus:     rgba(16,72,160,0.28);
    --cf-shadow:    0 2px 12px rgba(0,0,0,0.18);
}
body.cf-light { background: var(--cf-bg) !important; color: var(--cf-text) !important; }
body.cf-light .cf-brand-title { color: #1c2028 !important; }
body.cf-light .cf-modal-card { background: #ffffff !important; }
body.cf-light .cf-tabs-shell { box-shadow: 0 4px 10px rgba(0,0,0,0.10); }
body.cf-light .cf-adv-summary .text-gray-300 { color: var(--cf-text) !important; }
body.cf-light .q-btn:not([disabled]):hover { box-shadow: 0 4px 14px rgba(16,72,160,0.16) !important; }
body.cf-light .q-card:hover { box-shadow: 0 6px 20px rgba(0,0,0,0.15) !important; }
body.cf-light .cf-brand-shell { background: linear-gradient(180deg, rgba(16,72,160,0.06) 0%, var(--cf-bg) 100%); }
"""


def apply_theme():
    """Inject theme CSS and enable NiceGUI dark mode.

    Must be called inside a @ui.page handler, before other components.
    Returns the DarkMode element so callers can toggle it later.
    """
    from nicegui import ui  # noqa: PLC0415
    dark = ui.dark_mode()
    dark.enable()
    ui.add_css(_CF_CSS)
    return dark
