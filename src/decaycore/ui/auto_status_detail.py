# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Plain-language rendering for the "Automatic mode details" run panel.

The optimizer emits machine-readable English status lines that also go to the run
log. Those lines carry optimizer-internal comparison numbers (``rank``, ``avg``,
``candidate rank``, ``refine rank``). They are relative search numbers, not an
absolute quality measure, so they are hidden from the panel here while the raw
lines stay untouched for logs, audit trail and export.

Rows carry a ``group`` key. Consecutive rows that share a group replace each
other in the panel, so a phase of 40 trials stays on one self-updating line.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

BRACKET_RE = re.compile(r"^DecayCore automatic mode \[([^\]]+)\][^:]*:\s*(.+)$", re.IGNORECASE)

# The trailing "(...)" block is required: every per-trial emitter writes one, and
# demanding it keeps a bare phase label such as "phase 1/2" from being read as a
# trial counter.
_TRIAL_IMPROVED_RE = re.compile(
    r"^(?P<label>.+?)\s+best improved trial\s+(?P<i>\d+)/(?P<n>\d+)\s*(?P<tail>\(.*)$",
    re.IGNORECASE,
)
_TRIAL_RE = re.compile(
    r"^(?P<label>.+?)\s+(?P<i>\d+)/(?P<n>\d+)\s*(?P<tail>\(.*)$",
    re.IGNORECASE,
)

_OK_RE = re.compile(r"\bok (\d+)/(\d+)", re.IGNORECASE)
_DECISION_RE = re.compile(r"\bdecision ([^,)]+)", re.IGNORECASE)


def _t(key: str) -> str:
    # Late import mirrors ui_state: a partial module reload can update UI code
    # before the catalog module, and t() re-reads the catalog on a missing key.
    from ..resources.i8n.decaycore_i18n import t  # noqa: PLC0415

    return t(key)


def _f(key: str, **kw) -> str:
    return _t(key).format(**kw)


@dataclass(frozen=True)
class AutoDetailRow:
    """One rendered panel row.

    ``group`` is empty for rows that always stand on their own line.
    """

    text: str
    group: str = ""


@dataclass(frozen=True)
class AutoTrialLine:
    """A parsed per-trial status line."""

    group: str
    phase_text: str
    trial_index: int
    trial_total: int
    improved: bool
    reason: str | None = None


# ---------------------------------------------------------------------------
# phase names


def _param_label(param: str) -> str:
    """Localized label for a winner-polish parameter."""
    from .ui_state import _polish_param_label  # noqa: PLC0415

    plain = _polish_param_label(param)
    key = {
        "decay": "auto_param_decay",
        "extension": "auto_param_extension",
        "bass": "auto_param_bass",
        "phase": "auto_param_phase",
        "HPF": "auto_param_hpf",
        "peaks": "auto_param_peaks",
    }.get(plain)
    return _t(key) if key else plain


def _classify_phase(label: str) -> tuple[str, str]:
    """Map an internal phase label to ``(group, localized phase name)``."""
    s = str(label or "").strip()
    suffix = ""
    if s.lower().endswith(" rescue"):
        s = s[: -len(" rescue")].strip()
        suffix = ":rescue"

    m = re.match(r"^phase\s+\d+/\d+\s+local\s+center#(\d+)$", s, re.IGNORECASE)
    if m:
        return f"local:{m.group(1)}{suffix}", _f("auto_phase_local_refine", n=m.group(1))
    m = re.match(r"^cache refine round\s+(\d+)/(\d+)$", s, re.IGNORECASE)
    if m:
        return (
            f"cache:{m.group(1)}{suffix}",
            _f("auto_phase_cache_refine", n=m.group(1), total=m.group(2)),
        )
    m = re.match(r"^(\S+)\s+winner polish$", s, re.IGNORECASE)
    if m:
        return f"polish:{m.group(1).lower()}{suffix}", _f("auto_phase_polish", label=_param_label(m.group(1)))
    m = re.match(r"^target\s+(\S+)\s+\S+$", s, re.IGNORECASE)
    if m:
        return f"target:{m.group(1)}{suffix}", _f("auto_phase_target", name=m.group(1))
    if re.match(r"^phase\s+1\s+carry-forward$", s, re.IGNORECASE):
        return f"carry{suffix}", _t("auto_phase_carry_forward")
    if re.match(r"^(?:phase\s+\d+/\d+\s+)?micro(?:\s+refine)?$", s, re.IGNORECASE):
        return f"micro{suffix}", _t("auto_phase_micro")
    if re.match(r"^phase\s+1(?:/\d+)?$", s, re.IGNORECASE):
        return f"search{suffix}", _t("auto_phase_search")
    if re.match(r"^phase\s+\d+(?:/\d+)?$", s, re.IGNORECASE):
        return f"refine{suffix}", _t("auto_phase_refine")
    return f"other:{s.lower()}{suffix}", _t("auto_phase_generic")


# ---------------------------------------------------------------------------
# per-trial lines


def _to_int(value, fallback: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return fallback


def parse_trial_line(after_colon: str) -> AutoTrialLine | None:
    """Parse the per-trial part of a bracket-form automatic-mode status line."""
    s = str(after_colon or "").strip()
    if not s:
        return None
    improved = True
    m = _TRIAL_IMPROVED_RE.match(s)
    if m is None:
        improved = False
        m = _TRIAL_RE.match(s)
    if m is None:
        return None
    tail = m.group("tail") or ""
    # The first trial of a phase always "improves" because it seeds the
    # baseline; it is not evidence that the search found anything better.
    if improved and re.search(r"baseline initialized", tail, re.IGNORECASE):
        improved = False
    decision = _DECISION_RE.search(tail)
    group, phase_text = _classify_phase(m.group("label"))
    return AutoTrialLine(
        group=f"trial:{group}",
        phase_text=phase_text,
        trial_index=_to_int(m.group("i")),
        trial_total=_to_int(m.group("n")),
        improved=improved,
        reason=decision.group(1).strip() if decision else None,
    )


_REASON_KEYS = (
    (re.compile(r"^hard_gate:.*residual_peak", re.IGNORECASE), "auto_reason_boost_safety"),
    (re.compile(r"^(?:best_)?hard_gate:", re.IGNORECASE), "auto_reason_safety_limit"),
    (re.compile(r"^mode_ripple$", re.IGNORECASE), "auto_reason_ripple"),
    (re.compile(r"^(?:tracking|focus_ripple)$", re.IGNORECASE), "auto_reason_tracking"),
    (re.compile(r"^phase_(?:net|risk)$", re.IGNORECASE), "auto_reason_phase"),
)


def _reason_text(reason: str | None) -> str | None:
    """Plain-language explanation for a refine decision reason, if any."""
    s = str(reason or "").strip()
    if not s:
        return None
    for pattern, key in _REASON_KEYS:
        if pattern.match(s):
            return _t(key)
    # "rank" and anything unrecognized mean "simply not better" — no clause.
    return None


@dataclass
class TrialGroupTracker:
    """Tracks improvements and limiting reasons within one collapsed group."""

    group: str = ""
    improvements: int = 0
    _reasons: Counter = field(default_factory=Counter)

    def reset(self) -> None:
        self.group = ""
        self.improvements = 0
        self._reasons = Counter()

    def observe(self, line: AutoTrialLine) -> None:
        if line.group != self.group:
            self.group = line.group
            self.improvements = 0
            self._reasons = Counter()
        if line.improved:
            self.improvements += 1
        reason = _reason_text(line.reason)
        if reason:
            self._reasons[reason] += 1

    @property
    def dominant_reason(self) -> str | None:
        if not self._reasons:
            return None
        return self._reasons.most_common(1)[0][0]


def format_trial_row(line: AutoTrialLine, *, improvements: int, reason: str | None) -> str:
    """Render the collapsed panel row for a per-trial line."""
    text = _f(
        "auto_trial_row",
        phase=line.phase_text,
        i=line.trial_index,
        n=line.trial_total,
    )
    if improvements == 1:
        return text + _t("auto_trial_improvement_one")
    if improvements > 1:
        return text + _f("auto_trial_improvements", count=improvements)
    if line.trial_index < line.trial_total:
        return text
    if reason:
        return text + _f("auto_trial_no_improvement_reason", reason=reason)
    return text + _t("auto_trial_no_improvement")


def compact_running_text() -> str:
    """Neutral progress-bar label for status lines with no specific mapping."""
    return _t("auto_compact_running")


def format_trial_compact(line: AutoTrialLine, *, target: str) -> str:
    """Render the progress-bar label for a per-trial line."""
    key = "auto_compact_trial_improved" if line.improved else "auto_compact_trial"
    return _f(
        key,
        target=str(target or "").strip(),
        phase=line.phase_text,
        i=line.trial_index,
        n=line.trial_total,
    )


# ---------------------------------------------------------------------------
# legacy ("DecayCore automatic mode: ...") lines the older parsers do not cover

_BASS_INTEGRATION_PREFIXES = (
    "bass integration prepare init",
    "bass integration smart scan range",
    "bass integration optuna local refine",
    "bass integration optuna search",
    "bass integration allpass scan",
    "bass integration diagnostics refresh",
)

# Internal bookkeeping with no user-facing meaning, or an exact duplicate of a
# row that another parser already produces.
_HIDDEN_PREFIXES = (
    "adaptive target selected",
    "target shortlist cache wildcard",
    "target pilot complete",
)


def is_hidden_legacy(low: str) -> bool:
    """True for internal lines that should not reach the panel at all."""
    return any(low.startswith(prefix) for prefix in _HIDDEN_PREFIXES)


def _legacy_bass_integration(after: str, low: str) -> AutoDetailRow | None:
    if low.startswith("bass integration prepare done"):
        return AutoDetailRow(_t("auto_detail_bass_integration_done"))
    if any(low.startswith(prefix) for prefix in _BASS_INTEGRATION_PREFIXES):
        return AutoDetailRow(_t("auto_detail_bass_integration_start"), group="bass_integration")
    return None


def _legacy_target(after: str, low: str) -> AutoDetailRow | None:
    if low.startswith("adaptive target used conservative fallback"):
        m = re.search(r"fallback \(([^)]*)\)", after, re.IGNORECASE)
        return AutoDetailRow(_f("auto_detail_adaptive_target_fallback", reason=m.group(1) if m else "n/a"))
    if low.startswith("adaptive target synthesized"):
        return AutoDetailRow(_t("auto_detail_adaptive_target_synthesized"))
    if low.startswith("target cache hit"):
        m = re.match(r"target cache hit (\S+)", after, re.IGNORECASE)
        if m:
            return AutoDetailRow(_f("auto_detail_target_from_cache", target=m.group(1)))
    if low.startswith("target preselect cache seed") or low.startswith("target preselect seed loaded"):
        return AutoDetailRow(_t("auto_detail_target_cache_seed"))
    if low.startswith("target pilot evaluation"):
        m = re.match(r"target pilot evaluation \((\d+) curves", after, re.IGNORECASE)
        if m:
            return AutoDetailRow(_f("auto_detail_target_pilot", n=m.group(1)))
    if low.startswith("target shortlist milder included"):
        m = re.match(r"target shortlist milder included \((\S+)\s*->\s*(\S+?)\)", after, re.IGNORECASE)
        if m:
            return AutoDetailRow(_f("auto_detail_target_shortlist_milder_included", a=m.group(1), b=m.group(2)))
    if low.startswith("target curve mode="):
        m = re.match(r"target curve mode=\S+ \(using (\S+?),", after, re.IGNORECASE)
        if m:
            return AutoDetailRow(_f("auto_detail_target_curve_mode", name=m.group(1)))
    if low.startswith("custom target selected but no file found"):
        return AutoDetailRow(_t("auto_detail_custom_target_missing"))
    return None


def _legacy_target_trials(after: str, low: str) -> AutoDetailRow | None:
    """Room-curve comparison trials, which are as noisy as the bracket form."""
    if low.startswith("target trials"):
        m = re.search(r"target \d+/\d+ (\S+?),\s*trial (\d+)/(\d+)", after, re.IGNORECASE)
        if m:
            return AutoDetailRow(
                _f(
                    "auto_trial_row",
                    phase=_f("auto_phase_target", name=m.group(1)),
                    i=_to_int(m.group(2)),
                    n=_to_int(m.group(3)),
                ),
                group=f"trial:target:{m.group(1)}",
            )
    if low.startswith("selecting target curve (testing"):
        m = re.match(r"selecting target curve \(testing (\S+) (\d+)/(\d+)", after, re.IGNORECASE)
        if m:
            return AutoDetailRow(
                _f("auto_detail_selecting_target_testing", name=m.group(1), i=m.group(2), n=m.group(3)),
                group="target_select",
            )
    return None


def _legacy_refine(after: str, low: str) -> AutoDetailRow | None:
    if low.startswith("local refine center #"):
        m = re.match(r"Local refine center #(\d+) mixed_freq=([\d.]+) Hz", after, re.IGNORECASE)
        if m:
            return AutoDetailRow(_f("auto_detail_local_refine_center_start_mixed", n=m.group(1), hz=m.group(2)))
    if low.startswith("phase 4 finalize"):
        return AutoDetailRow(_t("auto_detail_finalize_start"))
    if low.startswith("phase 2 pareto"):
        return AutoDetailRow(_t("auto_detail_pareto_compare"))
    if low.startswith("stereo lf policy refine applied"):
        m = re.search(r"\((\S+) protected below ([\d.]+) Hz", after, re.IGNORECASE)
        if m:
            return AutoDetailRow(_f("auto_detail_stereo_lf_policy", side=m.group(1), hz=m.group(2)))
    if low.startswith("residual tie-break"):
        return AutoDetailRow(_t("auto_detail_residual_tiebreak"))
    if low.startswith("excursion protection kept at"):
        m = re.search(r"([\d.]+) Hz", after)
        if m:
            return AutoDetailRow(_f("auto_detail_excursion_protection", hz=m.group(1)))
    if low.startswith("forced smart scan range"):
        m = re.match(r"forced Smart Scan range ([\d.]+)-([\d.]+) Hz", after, re.IGNORECASE)
        if m:
            return AutoDetailRow(_f("auto_detail_smart_scan_forced", lo=m.group(1), hi=m.group(2)))
    m = re.match(r"^(\S+) (?:winner polish|polish) improved\b", after, re.IGNORECASE)
    if m:
        return AutoDetailRow(_f("auto_detail_polish_improved", label=_param_label(m.group(1))))
    return None


def render_legacy_extra(after: str, low: str) -> AutoDetailRow | None:
    """Render a legacy line that the older ui_state parsers do not cover."""
    for parser in (
        _legacy_bass_integration,
        _legacy_target_trials,
        _legacy_target,
        _legacy_refine,
    ):
        row = parser(after, low)
        if row is not None:
            return row
    return None
