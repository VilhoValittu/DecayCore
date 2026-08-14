# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""NiceGUI Advanced tab builder.

Replaces build_advanced_section() from layout_builders.py.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator

from . import ng_controls as ctrl
from .ng_advanced_presets import (
    AGGRESSIVE,
    NORMAL,
    SAFE,
    apply_bass_safety_preset,
    apply_conf_pull_preset,
    apply_shaping_preset,
    render_bass_safety_summary,
    render_conf_pull_summary,
    render_shaping_summary,
)
from .ng_sections import page_shell, section_card

_SLOPE_OPTS = [6, 12, 18, 24, 30, 36, 48]


@contextmanager
def _fine_tune_expansion(title: str) -> Iterator[None]:
    """Shared wrapper for the tab's collapsible sub-panels (fine-tune, guide,
    tuning, expert sections) so their styling is defined in one place."""
    from nicegui import ui

    with ui.expansion(title).classes("w-full"):
        yield


def _number_field(
    key: str,
    *,
    label_key: str,
    default: float,
    t: Callable,
    get_val: Callable,
    fmt: str = "%.1f",
    classes: str = "flex-1",
    **number_kwargs,
):
    """Register + build a ``ui.number`` field, replacing the repeated
    ``ctrl.register(key, ui.number(...).props(...).classes(...))`` block."""
    from nicegui import ui

    cast = int if fmt == "%d" else float
    raw = get_val(key, default)
    return ctrl.register(
        key,
        ui.number(
            label=t(label_key),
            value=cast(raw or default),
            format=fmt,
            **number_kwargs,
        )
        .props("dense outlined")
        .classes(classes),
    )


@dataclass
class _AdvancedTabContext:
    """Shared state for the Advanced tab.

    Groups ``t``/``get_val`` so the section builders below no longer repeat
    ``*, t: Callable, get_val: Callable`` in every function signature.
    """

    t: Callable
    get_val: Callable

    def build_shaping_section(self) -> None:
        from nicegui import ui

        t = self.t
        get_val = self.get_val

        with section_card(title=t("adv_shaping_title")):
            _build_preset_row(
                labels={
                    SAFE: t("preset_safe"),
                    NORMAL: t("preset_normal"),
                    AGGRESSIVE: t("preset_aggressive"),
                },
                on_pick=lambda key: _apply_guided_preset(
                    key,
                    t=t,
                    apply_fn=apply_shaping_preset,
                    render_fn=lambda: render_shaping_summary(t=t),
                ),
            )
            ctrl.register_container(
                "adv_shaping_summary_scope",
                ui.column().classes("cf-adv-summary w-full"),
            )

            with ui.row().classes("w-full gap-4 items-end"):
                _number_field(
                    "phase_limit",
                    label_key="phase_limit",
                    default=400.0,
                    t=t,
                    get_val=get_val,
                )

        with _fine_tune_expansion(t("adv_shaping_fine_tune_title")):
            with ui.column().classes("w-full gap-3"):
                with ui.row().classes("w-full gap-4"):
                    _number_field(
                        "max_slope_db_per_oct",
                        label_key="max_slope_db_per_oct",
                        default=12.0,
                        t=t,
                        get_val=get_val,
                    )
                    _number_field(
                        "max_cut_db",
                        label_key="max_cut_db",
                        default=30.0,
                        t=t,
                        get_val=get_val,
                    )

                with ui.row().classes("w-full gap-4"):
                    _number_field(
                        "max_slope_boost_db_per_oct",
                        label_key="max_slope_boost_db_per_oct",
                        default=0.0,
                        t=t,
                        get_val=get_val,
                    )
                    _number_field(
                        "max_slope_cut_db_per_oct",
                        label_key="max_slope_cut_db_per_oct",
                        default=0.0,
                        t=t,
                        get_val=get_val,
                    )

                with ui.row().classes("w-full gap-4"):
                    _number_field(
                        "min_boost_peak_db",
                        label_key="min_boost_peak_db",
                        default=2.0,
                        t=t,
                        get_val=get_val,
                        min=0.0,
                        max=3.0,
                        step=0.1,
                    )

                with ui.row().classes("w-full gap-4"):
                    _number_field(
                        "excess_phase_strength",
                        label_key="excess_phase_strength",
                        default=0.9,
                        t=t,
                        get_val=get_val,
                        min=0.0,
                        max=1.0,
                        step=0.05,
                    )
                    _number_field(
                        "low_freq_full_correction_hz",
                        label_key="low_freq_full_correction_hz",
                        default=140.0,
                        t=t,
                        get_val=get_val,
                        min=20.0,
                        max=1000.0,
                    )

                with ui.row().classes("w-full gap-4"):
                    _number_field(
                        "high_freq_no_correction_hz",
                        label_key="high_freq_no_correction_hz",
                        default=900.0,
                        t=t,
                        get_val=get_val,
                        min=40.0,
                        max=4000.0,
                    )
                    _number_field(
                        "gd_grad_limit_ms_per_oct",
                        label_key="gd_grad_limit_ms_per_oct",
                        default=30.0,
                        t=t,
                        get_val=get_val,
                        min=0.0,
                        max=100.0,
                    )

                with ui.row().classes("w-full gap-4"):
                    _number_field(
                        "trans_width",
                        label_key="trans_width_label",
                        default=100,
                        fmt="%d",
                        t=t,
                        get_val=get_val,
                    )
                    _number_field(
                        "reg_strength",
                        label_key="reg_strength",
                        default=30.0,
                        t=t,
                        get_val=get_val,
                    )

                ctrl.register(
                    "df_smoothing",
                    ui.checkbox(
                        f"{t('df_smoothing_label')} [EXPERIMENTAL]",
                        value=bool(get_val("df_smoothing", False)),
                    ),
                )

    def build_bass_safety_section(self) -> None:
        from nicegui import ui

        t = self.t
        get_val = self.get_val

        with section_card(title=t("adv_bass_safety_title")):
            _build_preset_row(
                labels={
                    SAFE: t("preset_safe"),
                    NORMAL: t("preset_normal"),
                    AGGRESSIVE: t("preset_aggressive"),
                },
                on_pick=lambda key: _apply_guided_preset(
                    key,
                    t=t,
                    apply_fn=apply_bass_safety_preset,
                    render_fn=lambda: render_bass_safety_summary(t=t),
                    extra_refreshers=(_refresh_bass_safety_ui,),
                ),
            )
            ctrl.register_container(
                "adv_bass_safety_summary_scope",
                ui.column().classes("cf-adv-summary w-full"),
            )

            with ui.row().classes("w-full gap-4 items-end"):
                ctrl.register(
                    "exc_prot",
                    ui.checkbox(
                        t("exc_prot_title"),
                        value=bool(get_val("exc_prot", False)),
                    ),
                )
                _number_field(
                    "exc_freq",
                    label_key="exc_freq",
                    default=25.0,
                    t=t,
                    get_val=get_val,
                )

            with ui.row().classes("w-full gap-4 items-end"):
                ctrl.register(
                    "hpf_enable",
                    ui.checkbox(
                        t("hpf_enable"),
                        value=bool(get_val("hpf_enable", False)),
                    ),
                )
                _number_field(
                    "hpf_freq",
                    label_key="hpf_freq",
                    default=20.0,
                    t=t,
                    get_val=get_val,
                )
                ctrl.register(
                    "hpf_slope",
                    ui.select(
                        _SLOPE_OPTS,
                        value=get_val("hpf_slope", 24),
                        label=t("hpf_slope"),
                    )
                    .props("dense outlined")
                    .classes("flex-1"),
                )

            with _fine_tune_expansion(t("adv_bass_safety_fine_tune_title")):
                with ui.column().classes("w-full gap-3"):
                    ctrl.register(
                        "low_bass_cut_enable",
                        ui.checkbox(
                            t("low_bass_cut_hz"),
                            value=bool(get_val("low_bass_cut_enable", True)),
                        ),
                    )
                    bass_cut_col = ui.column().classes("w-full gap-3")
                    ctrl.register_container("low_bass_cut_scope", bass_cut_col)
                    with bass_cut_col:
                        with ui.row().classes("w-full gap-4"):
                            _number_field(
                                "low_bass_cut_hz",
                                label_key="low_bass_cut_hz",
                                default=30.0,
                                t=t,
                                get_val=get_val,
                            )
                            _number_field(
                                "low_bass_cut_strength",
                                label_key="low_bass_cut_strength_label",
                                default=1.0,
                                fmt="%.2f",
                                t=t,
                                get_val=get_val,
                            )
                    bass_cut_col.set_visibility(bool(get_val("low_bass_cut_enable", True)))

                    ctrl.register(
                        "auto_optimize_low_bass_cut",
                        ui.checkbox(
                            t("auto_optimize_low_bass_cut"),
                            value=bool(get_val("auto_optimize_low_bass_cut", True)),
                        ),
                    )

                    ctrl.register(
                        "bass_first_ai",
                        ui.checkbox(
                            t("bass_first_enable_label"),
                            value=bool(get_val("bass_first_ai", False)),
                        ),
                    )
                    bass_first_col = ui.column().classes("w-full")
                    ctrl.register_container("bass_first_max_hz_scope", bass_first_col)
                    with bass_first_col:
                        _number_field(
                            "bass_first_mode_max_hz",
                            label_key="bass_first_max_hz_label",
                            default=200.0,
                            classes="w-full",
                            t=t,
                            get_val=get_val,
                        )
                    bass_first_col.set_visibility(bool(get_val("bass_first_ai", False)))

            with _fine_tune_expansion(t("guide_exc_prot_title")):
                ui.markdown(t("guide_exc_prot_body"))

            with _fine_tune_expansion(t("guide_low_bass_cut_title")):
                ui.markdown(t("guide_low_bass_cut_body"))

    def build_hybrid_iir_section(self) -> None:
        from nicegui import ui

        t = self.t
        get_val = self.get_val

        with section_card(title=t("hybrid_iir_title")):
            ui.label(t("hybrid_iir_help")).classes("text-xs text-gray-500")
            ctrl.register(
                "hybrid_iir_enabled",
                ui.checkbox(
                    t("hybrid_iir_enabled"),
                    value=bool(get_val("hybrid_iir_enabled", False)),
                ),
            )
            with _fine_tune_expansion(t("hybrid_iir_tuning_title")):
                with ui.column().classes("w-full gap-3"):
                    with ui.row().classes("w-full gap-4"):
                        _number_field(
                            "hybrid_iir_max_filters_per_channel",
                            label_key="hybrid_iir_max_filters_per_channel",
                            default=6,
                            fmt="%d",
                            t=t,
                            get_val=get_val,
                        )
                        _number_field(
                            "hybrid_iir_max_cut_db",
                            label_key="hybrid_iir_max_cut_db",
                            default=6.0,
                            t=t,
                            get_val=get_val,
                        )
                    with ui.row().classes("w-full gap-4"):
                        _number_field(
                            "hybrid_iir_min_freq_hz",
                            label_key="hybrid_iir_min_freq_hz",
                            default=20.0,
                            t=t,
                            get_val=get_val,
                        )
                        _number_field(
                            "hybrid_iir_max_freq_hz",
                            label_key="hybrid_iir_max_freq_hz",
                            default=200.0,
                            t=t,
                            get_val=get_val,
                        )
                    with ui.row().classes("w-full gap-4"):
                        _number_field(
                            "hybrid_iir_min_peak_db",
                            label_key="hybrid_iir_min_peak_db",
                            default=1.0,
                            t=t,
                            get_val=get_val,
                        )
                        _number_field(
                            "hybrid_iir_min_confidence",
                            label_key="hybrid_iir_min_confidence",
                            default=0.20,
                            fmt="%.2f",
                            t=t,
                            get_val=get_val,
                        )
                    with ui.row().classes("w-full gap-4"):
                        _number_field(
                            "hybrid_iir_min_q",
                            label_key="hybrid_iir_min_q",
                            default=3.0,
                            t=t,
                            get_val=get_val,
                        )
                        _number_field(
                            "hybrid_iir_max_q",
                            label_key="hybrid_iir_max_q",
                            default=12.0,
                            t=t,
                            get_val=get_val,
                        )
                        _number_field(
                            "hybrid_iir_min_gd_excess_ms",
                            label_key="hybrid_iir_min_gd_excess_ms",
                            default=1.0,
                            t=t,
                            get_val=get_val,
                        )
                        _number_field(
                            "hybrid_iir_min_cut_priority",
                            label_key="hybrid_iir_min_cut_priority",
                            default=0.0,
                            fmt="%.2f",
                            min=0.0,
                            max=1.0,
                            step=0.05,
                            t=t,
                            get_val=get_val,
                        )

    def build_conf_pull_section(self) -> None:
        from nicegui import ui

        t = self.t
        get_val = self.get_val

        conf_pull_col = ui.column().classes("w-full")
        ctrl.register_container("conf_pull_scope", conf_pull_col)
        with conf_pull_col:
            with ui.card().classes("w-full gap-3"):
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label(t("adv_conf_pull_title")).classes("text-sm font-semibold")
                    _notice_col = ui.column().classes("")
                    ctrl.register_container("conf_pull_notice_scope", _notice_col)
                    with _notice_col:
                        ui.label(t("ui_advanced_mode_only")).classes("text-xs text-gray-400 italic")
                _build_preset_row(
                    labels={
                        SAFE: t("preset_safe"),
                        NORMAL: t("preset_normal"),
                        AGGRESSIVE: t("preset_aggressive"),
                    },
                    on_pick=lambda key: _apply_guided_preset(
                        key,
                        t=t,
                        apply_fn=apply_conf_pull_preset,
                        render_fn=lambda: render_conf_pull_summary(t=t),
                    ),
                )
                ctrl.register_container(
                    "adv_conf_pull_summary_scope",
                    ui.column().classes("cf-adv-summary w-full"),
                )

                with _fine_tune_expansion(t("adv_conf_pull_tuning_title")):
                    with ui.row().classes("w-full gap-4"):
                        _number_field(
                            "conf_pull_floor",
                            label_key="conf_pull_floor_label",
                            default=0.0,
                            fmt="%.2f",
                            t=t,
                            get_val=get_val,
                        )
                        _number_field(
                            "conf_pull_ceil",
                            label_key="conf_pull_ceil_label",
                            default=1.0,
                            fmt="%.2f",
                            t=t,
                            get_val=get_val,
                        )
                        _number_field(
                            "conf_pull_max_hz",
                            label_key="conf_pull_max_hz_label",
                            default=200.0,
                            t=t,
                            get_val=get_val,
                        )
                    with ui.row().classes("w-full gap-4"):
                        _number_field(
                            "conf_pull_gamma_cut",
                            label_key="conf_pull_gamma_cut_label",
                            default=0.45,
                            fmt="%.2f",
                            min=0.05,
                            max=2.0,
                            step=0.05,
                            t=t,
                            get_val=get_val,
                        )
                        _number_field(
                            "conf_pull_gamma_boost",
                            label_key="conf_pull_gamma_boost_label",
                            default=0.35,
                            fmt="%.2f",
                            min=0.05,
                            max=2.0,
                            step=0.05,
                            t=t,
                            get_val=get_val,
                        )
                    with ui.row().classes("w-full gap-4"):
                        _number_field(
                            "conf_pull_bass_boost_floor_min",
                            label_key="conf_pull_bass_boost_floor_min_label",
                            default=0.55,
                            fmt="%.2f",
                            min=0.0,
                            max=1.0,
                            step=0.05,
                            t=t,
                            get_val=get_val,
                        )
                        _number_field(
                            "conf_pull_bass_boost_restore",
                            label_key="conf_pull_bass_boost_restore_label",
                            default=0.70,
                            fmt="%.2f",
                            min=0.0,
                            max=1.0,
                            step=0.05,
                            t=t,
                            get_val=get_val,
                        )
        # Hide notice label initially (BASIC is default; on_mode_change drives this).
        ctrl.get_container("conf_pull_notice_scope").set_visibility(False)
        ctrl.set_enabled("conf_pull_floor", False)
        ctrl.set_enabled("conf_pull_ceil", False)
        ctrl.set_enabled("conf_pull_max_hz", False)
        ctrl.set_enabled("conf_pull_gamma_cut", False)
        ctrl.set_enabled("conf_pull_gamma_boost", False)
        ctrl.set_enabled("conf_pull_bass_boost_floor_min", False)
        ctrl.set_enabled("conf_pull_bass_boost_restore", False)

    def build_stereo_link_section(self) -> None:
        from nicegui import ui

        t = self.t
        get_val = self.get_val

        # Hidden value holder – pipeline reads stereo_link bool directly from this.
        # Driven by stereo_link_strategy select; updated via callback.
        _sl_enabled = bool(get_val("stereo_link", True))
        _sl_strategy_raw = str(get_val("stereo_link_strategy", "auto") or "auto")
        _sl_init = _sl_strategy_raw if _sl_enabled else "off"
        ctrl.register("stereo_link", ctrl._ValueHolder(bool(_sl_enabled)))
        with ui.card().classes("w-full gap-3"):
            ctrl.register(
                "stereo_link_strategy",
                ui.select(
                    options={
                        "off": t("stereo_link_off"),
                        "auto": t("stereo_link_mode_auto"),
                        "hybrid": t("stereo_link_mode_hybrid"),
                        "shared": t("stereo_link_mode_shared_legacy"),
                    },
                    value=_sl_init,
                    label=t("stereo_link"),
                )
                .props("dense outlined")
                .classes("w-full"),
            )

    def build_raw_dsp_section(self) -> None:
        from nicegui import ui

        t = self.t
        get_val = self.get_val

        raw_dsp_col = ui.column().classes("w-full")
        ctrl.register_container("unsafe_raw_dsp_scope", raw_dsp_col)
        with raw_dsp_col:
            with ui.card().classes("w-full gap-3"):
                with _fine_tune_expansion(t("ui_expert_raw_dsp_title")):
                    with ui.row().classes("w-full items-center justify-between"):
                        _raw_notice_col = ui.column().classes("")
                        ctrl.register_container("raw_dsp_notice_scope", _raw_notice_col)
                        with _raw_notice_col:
                            ui.label(t("ui_advanced_mode_only_raw_dsp")).classes("text-xs text-gray-400 italic")
                    ctrl.register(
                        "unsafe_raw_dsp",
                        ui.checkbox(
                            t("unsafe_raw_dsp_enable_label"),
                            value=bool(get_val("unsafe_raw_dsp", False)),
                        ),
                    )
        ctrl.get_container("raw_dsp_notice_scope").set_visibility(False)
        ctrl.set_enabled("unsafe_raw_dsp", False)

    def build_plot_smoothing_section(self) -> None:
        from nicegui import ui

        t = self.t
        get_val = self.get_val

        with ui.card().classes("w-full gap-3"):
            with _fine_tune_expansion(t("ui_plots_visual_only")):
                ctrl.register(
                    "plot_smoothing_level",
                    ui.select(
                        options={
                            "Psychoacoustic": t("smooth_safe_reference"),
                            12: t("filter_smooth_12"),
                            24: t("filter_smooth_24"),
                            48: t("filter_smooth_48"),
                            96: t("filter_smooth_96"),
                        },
                        value=get_val("plot_smoothing_level", "Psychoacoustic"),
                        label=t("smooth_type"),
                    )
                    .props("dense outlined")
                    .classes("w-full"),
                )


def build_advanced_tab(*, t: Callable, get_val: Callable) -> None:
    ctx = _AdvancedTabContext(t=t, get_val=get_val)
    with page_shell(title=t("tab_adv"), intro=t("advanced_page_intro")):
        ctx.build_shaping_section()
        ctx.build_bass_safety_section()
        ctx.build_hybrid_iir_section()
        ctx.build_conf_pull_section()
        ctx.build_stereo_link_section()
        ctx.build_raw_dsp_section()
        ctx.build_plot_smoothing_section()


def _build_preset_row(*, labels: dict[str, str], on_pick: Callable[[str], None]) -> None:
    from nicegui import ui

    with ui.row().classes("cf-adv-preset-row"):
        for key in (SAFE, NORMAL, AGGRESSIVE):
            ui.button(
                labels[key],
                on_click=lambda preset_key=key: on_pick(preset_key),
            ).props('size="sm" outline no-caps')


def _apply_guided_preset(
    preset_key: str,
    *,
    t: Callable,
    apply_fn: Callable[..., None],
    render_fn: Callable[[], None],
    extra_refreshers: tuple[Callable[[], None], ...] = (),
) -> None:
    if str(ctrl.value("mode", "BASIC") or "BASIC").upper() == "AUTO":
        from .ng_health import show_toast  # noqa: PLC0415

        show_toast(t("toast_adv_preset_locked_auto"), color="info", duration=1.8)
        return

    apply_fn(preset_key, t=t)
    for refresher in extra_refreshers:
        refresher()
    render_fn()


def _refresh_bass_safety_ui() -> None:
    from .ng_mode_controls import (
        update_bass_first_ui,
        update_low_bass_cut_ui,
        update_stereo_auto_policy_ui,
    )  # noqa: PLC0415

    update_low_bass_cut_ui()
    update_bass_first_ui()
    update_stereo_auto_policy_ui()
