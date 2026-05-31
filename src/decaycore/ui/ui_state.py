# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

from __future__ import annotations

import logging
import re
from typing import Callable

logger = logging.getLogger("DecayCore")

_STATUS_BASE_MSG = ""
_STATUS_DOM_READY = False
_STATUS_LAST_TEXT = ""
_STATUS_SUMMARY_TEXT = ""
_STATUS_INFO_TEXT = ""
_AUTO_SELECTED_BAR_MSG = ""
_AUTO_STATUS_DETAILS: list[str] = []
_AUTO_STATUS_LAST_DETAIL = ""
_RUN_WALL_CLOCK_TEXT = ""
_LAST_RUN_INFO: dict = {}
_STATUS_RENDERER: Callable[..., None] | None = None


def set_status_renderer(renderer: Callable[..., None] | None) -> None:
    global _STATUS_RENDERER
    _STATUS_RENDERER = renderer if callable(renderer) else None


def mark_status_dom_ready(is_ready: bool = True) -> None:
    global _STATUS_DOM_READY
    _STATUS_DOM_READY = bool(is_ready)


def is_status_dom_ready() -> bool:
    return bool(_STATUS_DOM_READY)


def _status_split_elapsed_suffix(msg: str) -> tuple[str, str]:
    try:
        s = str(msg or "").strip()
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
        s = ""
    if not s:
        return "", ""
    try:
        match = re.match(r"^(.*?)(\|\s*\d+(?:\.\d+)?\s*s)\s*$", s, flags=re.IGNORECASE)
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
        match = None
    if not match:
        return str(s), ""
    return str(match.group(1) or "").strip(), str(match.group(2) or "").strip()


def _compact_auto_status_core(core: str) -> str:
    try:
        s = str(core or "").strip()
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
        s = ""

    # --- Bracket format: "DecayCore automatic mode [target] (...): <phase_text>" ---
    bracket_m = re.match(
        r"^DecayCore automatic mode \[([^\]]+)\][^:]*:\s*(.+)$",
        s,
        flags=re.IGNORECASE,
    )
    if bracket_m:
        target = bracket_m.group(1).strip()
        after_colon = bracket_m.group(2).strip()
        low_ac = after_colon.lower()

        # Main search: "phase 1/2 best improved trial N/M (...)"
        best_m = re.match(
            r"(phase\s+\d+/\d+)\s+best improved trial\s+(\d+/\d+)",
            after_colon, re.IGNORECASE,
        )
        if best_m:
            return f"Optimizing · {target} · {best_m.group(1)} · trial {best_m.group(2)} ↑"

        # Main search: "phase 1/2 N/M (...)"
        phase_m = re.match(r"(phase\s+\d+/\d+)\s+(\d+/\d+)", after_colon, re.IGNORECASE)
        if phase_m:
            return f"Optimizing · {target} · {phase_m.group(1)} · trial {phase_m.group(2)}"

        # Winner polish with trial counts: "<param> winner polish best improved trial N/M"
        polish_best_m = re.match(
            r"(\S+)\s+winner polish\s+best improved trial\s+(\d+/\d+)",
            after_colon, re.IGNORECASE,
        )
        if polish_best_m:
            label = _polish_param_label(polish_best_m.group(1))
            return f"Polishing · {target} · {label} · {polish_best_m.group(2)} ↑"

        # Winner polish with trial counts: "<param> winner polish N/M"
        polish_m = re.match(r"(\S+)\s+winner polish\s+(\d+/\d+)", after_colon, re.IGNORECASE)
        if polish_m:
            label = _polish_param_label(polish_m.group(1))
            return f"Polishing · {target} · {label} · {polish_m.group(2)}"

        # Residual tie-break / safety
        if low_ac.startswith("residual tie-break"):
            return f"Polishing · {target} · peaks"
        if low_ac.startswith("residual_peak safety override"):
            return f"Safety check · {target}"

        # Fallback: strip parenthetical suffix
        clean = re.sub(r"\s*\(.*\)\s*$", "", after_colon).strip()
        clean = re.sub(r"\s+\d+/\d+\s*$", "", clean).strip()
        return f"Optimizing · {target} · {clean}" if clean else f"Optimizing · {target}"

    # --- Old format: "DecayCore automatic mode: <phase_text>" ---
    prefix = "DecayCore automatic mode:"
    if not s.startswith(prefix):
        return s
    after = s[len(prefix):].strip()
    low = after.lower()

    # Target curve selection
    if low.startswith("adaptive target"):
        return "Synthesizing · target"
    if low.startswith("target shortlist"):
        return "Selecting · target"
    if low.startswith("target preselect winner"):
        return "Selecting · target"
    if low.startswith("target preselect cache seed") or low.startswith("target preselect seed loaded"):
        return "Loading · target"
    if low.startswith("target preselect"):
        return "Selecting · target"
    if low.startswith("target search"):
        return "Selecting · target"
    if low.startswith("selecting target curve"):
        return "Selecting · target"
    if low.startswith("target trials"):
        return "Searching · target"
    if low.startswith("target finalize"):
        return "Selecting · target"
    if low.startswith("target curve mode"):
        return "Selecting · target"
    if low.startswith("target loaded directly from cache") or low.startswith("target cache hit"):
        return "Loaded from cache"

    # Phase 1 search
    if low.startswith("phase1 done") or low.startswith("phase 1 done"):
        return "Searching · done"

    # Phase 2 local refine
    if low.startswith("local refine summary"):
        return "Refining · done"
    if low.startswith("local refine fallback"):
        return "Refining"
    if low.startswith("local refine"):
        return "Refining"
    if low.startswith("phase2 summary"):
        return "Refining · done"

    # Phase 3 micro refine
    if low.startswith("micro refine summary"):
        return "Micro-refining · done"
    if low.startswith("micro refine fallback"):
        return "Micro-refining"
    if low.startswith("micro refine"):
        return "Micro-refining"

    # Cache
    if low.startswith("cache refine best improved"):
        return "Refining · cache ↑"
    if low.startswith("cache refine round summary") or low.startswith("cache refine summary"):
        return "Refining · cache done"
    if low.startswith("cache refine init"):
        return "Refining · cache"
    if low.startswith("cache refine"):
        return "Refining · cache"
    if low.startswith("preset loaded from cache"):
        return "Loaded from cache"

    # Winner polish (old format "improved" messages)
    if low.startswith("tdc_strength winner polish"):
        return "Polishing · decay ↑"
    if low.startswith("mag_c_min winner polish"):
        return "Polishing · extension ↑"
    if low.startswith("low_bass_cut winner polish"):
        return "Polishing · bass ↑"
    if low.startswith("phase_limit winner polish"):
        return "Polishing · phase ↑"
    if low.startswith("hpf winner polish"):
        return "Polishing · HPF ↑"
    if low.startswith("excess_phase_strength winner polish"):
        return "Polishing · phase ↑"
    if low.startswith("residual-peak polish") or low.startswith("residual_peak winner polish"):
        return "Polishing · peaks ↑"
    if low.startswith("residual tie-break"):
        return "Polishing · peaks"
    if low.startswith("residual_peak safety override"):
        return "Safety check"

    # Finalize
    if low.startswith("phase 4 finalize") or low.startswith("finalize"):
        return "Finalizing"

    # Other search phases
    if low.startswith("preset search"):
        return "Searching · preset"
    if low.startswith("phase search"):
        return "Searching · phase"
    if low.startswith("protection model"):
        return "Safety check"
    if low.startswith("hpf auto-fit"):
        return "Fitting · HPF"
    if low.startswith("stereo lf policy"):
        return "Refining · stereo"
    if low.startswith("init"):
        return "Initializing"

    # Generic fallback
    try:
        clean = re.sub(r"\s*\(.*\)\s*$", "", after).strip()
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
        clean = after
    return f"Auto · {clean}" if clean else "Auto · running"


def _polish_param_label(param: str) -> str:
    """Map internal parameter name to a readable polish label."""
    p = str(param or "").lower()
    if p.startswith("tdc"):
        return "decay"
    if p.startswith("mag_c"):
        return "extension"
    if p.startswith("low_bass"):
        return "bass"
    if p.startswith("phase_limit"):
        return "phase"
    if p.startswith("hpf"):
        return "HPF"
    if p.startswith("excess_phase"):
        return "phase"
    if p.startswith("residual"):
        return "peaks"
    return p


def _humanize_auto_status_detail(msg: str) -> str:
    from ..resources.i8n.decaycore_i18n import t  # noqa: PLC0415

    def _f(key, **kw):
        return t(key).format(**kw)

    def _r(v, dec=1):
        return f"{float(v):.{dec}f}"

    prefix = "DecayCore automatic mode:"
    try:
        s = str(msg or "").strip()
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
        return msg
    if not s.startswith(prefix):
        return s
    after = s[len(prefix):].strip()
    low = after.lower()
    try:
        if low.startswith("init"):
            filt_m = re.search(r"filter (.+?)(?=,\s*taps)", after, re.IGNORECASE)
            taps_m = re.search(r"taps (\d+)", after)
            goal_m = re.match(r"init \(goal (\w+)", after, re.IGNORECASE)
            if goal_m and filt_m:
                return _f(
                    "auto_detail_init",
                    goal=goal_m.group(1),
                    filter=filt_m.group(1).strip(),
                    taps=taps_m.group(1) if taps_m else "?",
                )

        if low.startswith("protection seed"):
            hz_m = re.search(r"-6 dB point ([\d.]+) Hz", after, re.IGNORECASE)
            lc_m = re.search(r"low-cut ([\d.]+) Hz", after, re.IGNORECASE)
            exc_m = re.search(r"exc seed ([\d.]+) Hz", after, re.IGNORECASE)
            if hz_m and lc_m and exc_m:
                return _f("auto_detail_protection_seed", hz=hz_m.group(1), lc=lc_m.group(1), exc=exc_m.group(1))

        if low.startswith("hpf auto-fit applied"):
            m = re.match(r"HPF auto-fit applied ([\d.]+) Hz/([\d.]+) dB/oct.*?confidence ([\d.]+)", after, re.IGNORECASE)
            if m:
                return _f("auto_detail_hpf_applied", hz=m.group(1), slope=m.group(2), conf=int(round(float(m.group(3)) * 100)))

        if low.startswith("target loaded directly from cache"):
            m = re.search(r"->\s*(\S+?),", after, re.IGNORECASE)
            if m:
                return _f("auto_detail_target_from_cache", target=m.group(1))

        if low.startswith("target preselect top-"):
            n_m = re.match(r"target preselect top-(\d+)", after, re.IGNORECASE)
            candidates = re.findall(r"(\w+)\([^)]*fit=([\d.]+)", after)
            if n_m and candidates:
                item_tpl = t("auto_detail_target_preselect_topn_item")
                parts = ", ".join(item_tpl.format(name=name, fit=_r(fit)) for name, fit in candidates)
                return _f("auto_detail_target_preselect_topn", n=n_m.group(1), candidates=parts)

        if low.startswith("target preselect winner") or low.startswith("target search winner"):
            m = re.match(
                r"target (?:preselect|search) winner (\S+).*?fit_rms ([\d.]+) dB,\s*tested (\d+) curves x (\d+) trials",
                after, re.IGNORECASE,
            )
            if m:
                winner, fit, curves, trials = m.group(1), _r(m.group(2)), m.group(3), m.group(4)
                if m.group(3) == "0":
                    return _f("auto_detail_target_winner_cache", winner=winner, fit=fit)
                return _f("auto_detail_target_winner", winner=winner, fit=fit, curves=curves, trials=trials)

        if low.startswith("target preselect init") or low.startswith("target search init"):
            m = re.match(r"target (?:preselect|search) init \(top-(\d+),\s*(\d+) trials/curve,\s*fs ([\d.]+) Hz", after, re.IGNORECASE)
            if m:
                return _f("auto_detail_target_preselect_init", n=m.group(1), trials=m.group(2), fs=m.group(3))

        if low.startswith("target shortlist milder skipped"):
            m = re.match(r"target shortlist milder skipped \((\w+)\s*->\s*(\w+)", after, re.IGNORECASE)
            if m:
                return _f("auto_detail_target_shortlist_milder_skipped", a=m.group(1), b=m.group(2))

        if low.startswith("target shortlist"):
            m = re.match(r"target shortlist \(selected (\d+)/(\d+) by spread ([\d.]+) dB\)", after, re.IGNORECASE)
            if m:
                return _f("auto_detail_target_shortlist", sel=m.group(1), total=m.group(2), spread=m.group(3))

        if low.startswith("phase1 done target="):
            m = re.match(
                r"Phase1 done target=(\S+?),\s*rank=([\d.]+),\s*avg_score=([\d.]+),\s*mode_ripple=([\d.]+) dB,\s*boost=(-?[\d.]+) dB.*?tdc=([\d.]+)",
                after, re.IGNORECASE,
            )
            if m:
                return _f(
                    "auto_detail_phase1_done_target",
                    target=m.group(1), rank=_r(m.group(2)), avg=_r(m.group(3)),
                    ripple=_r(m.group(4)), tdc=_r(m.group(6)),
                )

        if low.startswith("selecting target curve"):
            n_m = re.search(r"best improved (\d+)/(\d+)", after, re.IGNORECASE)
            leader_m = re.search(r"leader (\S+?),", after, re.IGNORECASE)
            if n_m and leader_m:
                return _f("auto_detail_selecting_target", leader=leader_m.group(1), n=n_m.group(1), m=n_m.group(2))

        if low.startswith("target finalize"):
            m = re.match(
                r"target finalize \(winner (\S+?),\s*method \S+,\s*rank ([\d.]+),\s*avg ([\d.]+).*?fit ([\d.]+) dB\)",
                after, re.IGNORECASE,
            )
            if m:
                return _f("auto_detail_target_finalize", winner=m.group(1), fit=_r(m.group(4)))

        if low.startswith("preset loaded from cache"):
            target_m = re.search(r"target (\S+?),", after, re.IGNORECASE)
            extra_m = re.search(r"up to (\d+) x (\d+) extra micro-trials", after, re.IGNORECASE)
            if target_m and extra_m:
                return _f("auto_detail_preset_from_cache", target=target_m.group(1), rounds=extra_m.group(1), trials=extra_m.group(2))

        if low.startswith("preset search init") or low.startswith("phase search init"):
            m = re.match(
                r"(?:preset|phase) search init \(phase1 (\d+) \+ (?:phase2|refine) (\d+)"
                r"(?: \+ micro (\d+))? trials @ ([\d.]+) Hz.*?target (\S+)\)",
                after, re.IGNORECASE,
            )
            if m:
                if m.group(3) is None:
                    return _f(
                        "auto_detail_preset_search_init_legacy",
                        p1=m.group(1),
                        refine=m.group(2),
                        target=m.group(5),
                    )
                return _f(
                    "auto_detail_preset_search_init",
                    p1=m.group(1),
                    refine=m.group(2),
                    micro=m.group(3) or "?",
                    target=m.group(5),
                )

        if low.startswith("phase1 done rank="):
            m = re.match(
                r"Phase1 done rank=([\d.]+),\s*avg_score=([\d.]+),\s*mode_ripple=([\d.]+) dB"
                r".*?boost=(-?[\d.]+) dB.*?optuna run=(\d+), ok=(\d+)",
                after, re.IGNORECASE,
            )
            if m:
                return _f(
                    "auto_detail_phase1_done",
                    rank=_r(m.group(1)), avg=_r(m.group(2)), ripple=_r(m.group(3)),
                    run=m.group(5), ok=m.group(6),
                )

        if low.startswith("cache refine init"):
            m = re.match(r"cache refine init \(rounds up to (\d+), (\d+) trials/round", after, re.IGNORECASE)
            if m:
                return _f("auto_detail_cache_refine_init", rounds=m.group(1), trials=m.group(2))

        if low.startswith("cache refine round summary"):
            m = re.match(
                r"cache refine round summary \(round (\d+), executed \d+/\d+, improvements (\d+).*?optuna run=(\d+), ok=(\d+)",
                after, re.IGNORECASE,
            )
            if m:
                return _f("auto_detail_cache_refine_round_summary", n=m.group(1), improvements=m.group(2), run=m.group(3), ok=m.group(4))

        if low.startswith("cache refine best improved"):
            m = re.match(r"cache refine best improved \(round (\d+), \d+/\d+, rank ([\d.]+)", after, re.IGNORECASE)
            if m:
                return _f("auto_detail_cache_refine_improved", n=m.group(1), rank=_r(m.group(2)))

        if low.startswith("cache refine summary"):
            m = re.match(
                r"cache refine summary \(rounds (\d+)/(\d+), executed (\d+) trials, improvements (\d+)",
                after, re.IGNORECASE,
            )
            if m:
                return _f("auto_detail_cache_refine_summary", done=m.group(1), total=m.group(2), executed=m.group(3), improvements=m.group(4))

        if low.startswith("cache refine round"):
            m = re.match(r"cache refine round (\d+)/(\d+) \(optuna (\d+) trials\)", after, re.IGNORECASE)
            if m:
                return _f("auto_detail_cache_refine_round", n=m.group(1), total=m.group(2), trials=m.group(3))

        if "auto-fit seed" in low:
            m = re.match(
                r"(?:Sub )?HPF auto-fit seed ([\d.]+) Hz/([\d.]+) dB/oct.*?confidence ([\d.]+)",
                after, re.IGNORECASE,
            )
            if m:
                is_sub = bool(re.match(r"Sub HPF", after, re.IGNORECASE))
                key = "auto_detail_sub_hpf_autofit_seed" if is_sub else "auto_detail_hpf_autofit_seed"
                return _f(key, hz=m.group(1), slope=int(float(m.group(2))), conf=int(round(float(m.group(3)) * 100)))

        if "auto-fit fallback" in low:
            m = re.match(
                r"(?:Sub )?HPF auto-fit fallback ([\d.]+) Hz/([\d.]+) dB/oct.*?confidence ([\d.]+)",
                after, re.IGNORECASE,
            )
            if m:
                return _f("auto_detail_hpf_autofit_fallback", hz=m.group(1), slope=int(float(m.group(2))), conf=int(round(float(m.group(3)) * 100)))

        if "auto-fit skipped" in low:
            return _f("auto_detail_hpf_autofit_skipped")

        if low.startswith("local refine target="):
            m = re.match(r"Local refine target=(\S+) center #\d+ phase refine phase_limit=([\d.]+) Hz", after, re.IGNORECASE)
            if m:
                return _f("auto_detail_local_refine_target_start", name=m.group(1), hz=m.group(2))
            m2 = re.match(r"Local refine target=(\S+) center #\d+ mixed_freq=([\d.]+) Hz", after, re.IGNORECASE)
            if m2:
                return _f("auto_detail_local_refine_target_start_mixed", name=m2.group(1), hz=m2.group(2))

        if low.startswith("local refine center #"):
            m = re.match(r"Local refine center #(\d+) phase refine phase_limit=([\d.]+) Hz", after, re.IGNORECASE)
            if m:
                return _f("auto_detail_local_refine_center_start", n=m.group(1), hz=m.group(2))

        if low.startswith("local refine fallback"):
            m = re.match(r"Local refine fallback center #(\d+)", after, re.IGNORECASE)
            n = m.group(1) if m else "?"
            return _f("auto_detail_local_refine_fallback", n=n)

        if low.startswith("local refine summary"):
            m = re.match(
                r"Local refine summary center #(\d+),\s*current_best_rank=([\d.]+),\s*avg_score=([\d.]+)"
                r",\s*optuna run=(\d+), ok=(\d+)",
                after, re.IGNORECASE,
            )
            if m:
                return _f("auto_detail_local_refine", n=m.group(1), rank=_r(m.group(2)), avg=_r(m.group(3)), run=m.group(4), ok=m.group(5))

        if low.startswith("micro refine fallback"):
            return _f("auto_detail_micro_refine_fallback")

        if low.startswith("phase2 summary"):
            m = re.search(r"optuna run=(\d+), ok=(\d+)", after, re.IGNORECASE)
            if m:
                return _f("auto_detail_phase2_summary", run=m.group(1), ok=m.group(2))

        if low.startswith("hpf winner polish improved"):
            m = re.match(r"hpf winner polish improved \(([^,]+), rank ([\d.]+)", after, re.IGNORECASE)
            if m:
                return _f("auto_detail_hpf_polish_improved", label=m.group(1).strip(), rank=_r(m.group(2)))

        if low.startswith("micro refine summary"):
            m = re.match(
                r"micro refine summary current_best_rank=([\d.]+),\s*avg_score=([\d.]+)"
                r",\s*optuna run=(\d+), ok=(\d+)",
                after, re.IGNORECASE,
            )
            if m:
                return _f("auto_detail_micro_refine", rank=_r(m.group(1)), avg=_r(m.group(2)), run=m.group(3), ok=m.group(4))

        if low.startswith("micro refine"):
            m = re.match(r"micro refine (\d+) trials around current best", after, re.IGNORECASE)
            if m:
                return _f("auto_detail_micro_refine_start", trials=m.group(1))

        if low.startswith("mag_c_min winner polish improved"):
            m = re.match(
                r"mag_c_min winner polish improved \(mag_c_min ([\d.]+) -> ([\d.]+) Hz, rank ([\d.]+) -> ([\d.]+)",
                after, re.IGNORECASE,
            )
            if m:
                return _f("auto_detail_mag_c_min_improved", from_hz=m.group(1), to_hz=m.group(2), old_r=_r(m.group(3)), new_r=_r(m.group(4)))

        if low.startswith("phase 2 pareto comparison") or low.startswith("phase 2 pareto selected winner"):
            m = re.search(
                r"rank_best [\d.]+ -> pareto ([\d.]+).*?mode_ripple [\d.]+ dB -> ([\d.]+) dB",
                after, re.IGNORECASE,
            )
            if m:
                return _f("auto_detail_pareto_winner", rank=_r(m.group(1)), ripple=_r(m.group(2)))

        if low.startswith("finalize"):
            m = re.match(
                r"finalize \(winner rank ([\d.]+)/100,\s*avg ([\d.]+),\s*boost (-?[\d.]+) dB,\s*events (\d+)(?:,|\))",
                after, re.IGNORECASE,
            )
            if m:
                return _f("auto_detail_finalize", rank=_r(m.group(1)), avg=_r(m.group(2)), boost=_r(m.group(3)), events=m.group(4))
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
        logger.exception("auto status detail parse")
    return s


def _status_compact_with_detail(msg) -> tuple[str, str | None]:
    try:
        raw = str(msg or "").strip()
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
        raw = ""
    if not raw:
        return "DecayCore running", None
    core, elapsed = _status_split_elapsed_suffix(raw)
    compact_core = _compact_auto_status_core(core)
    out = compact_core
    if elapsed:
        out = f"{compact_core} {elapsed}"
    detail = None
    _is_auto_mode_text = isinstance(core, str) and (
        core.startswith("DecayCore automatic mode:")
        or bool(re.match(r"^DecayCore automatic mode \[", core))
    )
    if _is_auto_mode_text and str(core).strip() != str(compact_core).strip():
        detail = str(core).strip()
    return out, detail


def _status_base_from_text(msg) -> str:
    text, _detail = _status_compact_with_detail(msg)
    core, _elapsed = _status_split_elapsed_suffix(text)
    core = str(core or "").strip()
    return core or "DecayCore running"


def _normalize_auto_selected_text(msg) -> str:
    try:
        txt = str(msg or "").strip()
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
        txt = ""
    return txt


def _normalize_status_notice_text(msg) -> str:
    try:
        txt = str(msg or "").strip()
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
        txt = ""
    return txt


def get_status_base_message(default: str = "DecayCore running") -> str:
    try:
        value = str(_STATUS_BASE_MSG or "").strip()
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
        value = ""
    return value or str(default)


def set_run_wall_clock_text(value) -> None:
    global _RUN_WALL_CLOCK_TEXT
    try:
        _RUN_WALL_CLOCK_TEXT = str(value or "").strip()
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
        _RUN_WALL_CLOCK_TEXT = ""


def get_run_wall_clock_text(default: str = "") -> str:
    try:
        value = str(_RUN_WALL_CLOCK_TEXT or "").strip()
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
        value = ""
    return value or str(default or "")


def get_status_snapshot() -> dict:
    details = [str(item or "") for item in list(_AUTO_STATUS_DETAILS or [])]
    return {
        "status_base_message": str(_STATUS_BASE_MSG or ""),
        "status_dom_ready": bool(_STATUS_DOM_READY),
        "status_last_text": str(_STATUS_LAST_TEXT or ""),
        "status_summary_text": _normalize_status_notice_text(_STATUS_SUMMARY_TEXT),
        "status_info_text": _normalize_status_notice_text(_STATUS_INFO_TEXT),
        "auto_selected_bar_text": _normalize_auto_selected_text(_AUTO_SELECTED_BAR_MSG),
        "auto_status_details": details,
        "auto_status_detail_body": "\n".join(details),
        "run_wall_clock_text": get_run_wall_clock_text(""),
    }


def _notify_renderer(event: str) -> None:
    renderer = _STATUS_RENDERER
    if not callable(renderer):
        return
    try:
        renderer(event=str(event or ""), snapshot=get_status_snapshot())
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
        logger.debug("UI status renderer update failed", exc_info=True)


def update_status(msg) -> None:
    global _STATUS_BASE_MSG, _STATUS_LAST_TEXT, _AUTO_STATUS_DETAILS, _AUTO_STATUS_LAST_DETAIL
    text, detail = _status_compact_with_detail(msg)
    _STATUS_BASE_MSG = _status_base_from_text(text)
    _STATUS_LAST_TEXT = str(text or "")
    if isinstance(detail, str) and detail.strip():
        detail_txt = _humanize_auto_status_detail(str(detail).strip())
        if detail_txt != str(_AUTO_STATUS_LAST_DETAIL or ""):
            _AUTO_STATUS_LAST_DETAIL = detail_txt
            _AUTO_STATUS_DETAILS = list(_AUTO_STATUS_DETAILS or []) + [detail_txt]
    _notify_renderer("status")


def update_auto_selected_bar(msg) -> None:
    global _AUTO_SELECTED_BAR_MSG
    _AUTO_SELECTED_BAR_MSG = _normalize_auto_selected_text(msg)
    _notify_renderer("auto_selected_bar")


def update_status_notices(*, summary_text=None, info_text=None) -> None:
    global _STATUS_SUMMARY_TEXT, _STATUS_INFO_TEXT
    if summary_text is not None:
        _STATUS_SUMMARY_TEXT = _normalize_status_notice_text(summary_text)
    if info_text is not None:
        _STATUS_INFO_TEXT = _normalize_status_notice_text(info_text)
    _notify_renderer("status_notices")


def set_last_run_info(info: dict) -> None:
    global _LAST_RUN_INFO
    try:
        _LAST_RUN_INFO = dict(info) if isinstance(info, dict) else {}
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
        _LAST_RUN_INFO = {}


def get_last_run_info() -> dict:
    try:
        return dict(_LAST_RUN_INFO)
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError):
        return {}


def reset_auto_status_details() -> None:
    global _AUTO_STATUS_DETAILS, _AUTO_STATUS_LAST_DETAIL
    _AUTO_STATUS_DETAILS = []
    _AUTO_STATUS_LAST_DETAIL = ""
    _notify_renderer("reset_auto_status_details")


def append_auto_status_detail_raw(line: str) -> None:
    global _AUTO_STATUS_DETAILS, _AUTO_STATUS_LAST_DETAIL
    txt = str(line or "").strip()
    if not txt:
        return
    _AUTO_STATUS_LAST_DETAIL = txt
    _AUTO_STATUS_DETAILS = list(_AUTO_STATUS_DETAILS or []) + [txt]
    _notify_renderer("status")


__all__ = [
    "append_auto_status_detail_raw",
    "get_last_run_info",
    "get_run_wall_clock_text",
    "get_status_base_message",
    "get_status_snapshot",
    "is_status_dom_ready",
    "mark_status_dom_ready",
    "reset_auto_status_details",
    "set_last_run_info",
    "set_run_wall_clock_text",
    "set_status_renderer",
    "update_auto_selected_bar",
    "update_status",
    "update_status_notices",
]
