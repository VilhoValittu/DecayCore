# DecayCore
# Copyright (c) 2026 Vilho Valittu
# All rights reserved.
#
# This file is part of the proprietary DecayCore codebase.
# No copying, redistribution, commercial reuse, or removal of attribution
# is permitted without prior written permission.

"""
DecayCore headless run analyser.

Two modes:

  Summary — show a metrics table for all runs in a directory:
      python tests/headless/compare_runs.py --summary /tmp/camillafir-headless-runs

  Compare — side-by-side diff between two run sets (before vs. after a code change):
      python tests/headless/compare_runs.py \
          --before /tmp/camillafir-runs-before \
          --after  /tmp/camillafir-runs-after

Exit codes:
  0 — OK (or compare: no regressions above threshold)
  1 — One or more regressions detected
  2 — No metrics.json files found
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Metric catalogue
# ---------------------------------------------------------------------------

# (key_in_metrics_dict, short_column_header, higher_is_better)
METRICS: list[tuple[str, str, bool]] = [
    ("auto_score",             "score",    True),
    ("filter_score",           "flt_scr",  True),
    ("mag_error_rms_db",       "mag_rms",  False),
    ("mag_error_max_db",       "mag_max",  False),
    ("residual_peak_count",    "res_pk#",  False),
    ("residual_peak_max_db",   "res_pkdb", False),
    ("gd_spike_count",         "gd_spk#",  False),
    ("gd_max_ms",              "gd_max",   False),
    ("max_boost_db",           "boost",    False),
    ("max_cut_db",             "cut",      False),
    ("bass_ripple_20_200_db",  "bass_rip", False),
    ("harmonic_risk_max",      "harm_rsk", False),
    ("runtime_s",              "time_s",   False),
]

METRIC_KEYS  = [m[0] for m in METRICS]
METRIC_HEADS = [m[1] for m in METRICS]
HIGHER_BETTER = {m[0]: m[2] for m in METRICS}

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR", "") == ""

GREEN  = "\033[32m" if _USE_COLOR else ""
RED    = "\033[31m" if _USE_COLOR else ""
YELLOW = "\033[33m" if _USE_COLOR else ""
RESET  = "\033[0m"  if _USE_COLOR else ""
BOLD   = "\033[1m"  if _USE_COLOR else ""


def _c(text: str, code: str) -> str:
    return f"{code}{text}{RESET}" if code else text


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_metrics(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
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
    ) as exc:
        return {"status": "load_error", "errors": [str(exc)], "metrics": {}}


def _collect(root: Path) -> dict[str, dict[str, Any]]:
    """Return {scenario_name: doc} for all metrics.json found under root."""
    found: dict[str, dict[str, Any]] = {}
    root = root.resolve()
    for p in sorted(root.glob("*/metrics.json")):
        name = p.parent.name
        found[name] = _load_metrics(p)
    return found


def _get(doc: dict, key: str) -> float | None:
    raw = (doc.get("metrics") or {}).get(key)
    if raw is None:
        raw = doc.get(key)
    if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
        return float(raw)
    return None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(v: float | None, width: int = 7) -> str:
    if v is None:
        return " " * width
    s = f"{v:.2f}" if abs(v) < 10_000 else f"{v:.0f}"
    return s.rjust(width)


def _status_col(status: str, width: int = 10) -> str:
    s = (status or "?")[:width].ljust(width)
    if status == "success":
        return _c(s, GREEN)
    if status in ("failed", "load_error"):
        return _c(s, RED)
    if status == "partial":
        return _c(s, YELLOW)
    return s


# ---------------------------------------------------------------------------
# Summary mode
# ---------------------------------------------------------------------------

def cmd_summary(root: Path) -> int:
    runs = _collect(root)
    if not runs:
        print(f"No metrics.json files found under {root}", file=sys.stderr)
        return 2

    name_w = max(len(n) for n in runs) + 2

    # Header
    head = "Scenario".ljust(name_w) + "status    " + "  ".join(h.rjust(7) for h in METRIC_HEADS)
    print(_c(head, BOLD))
    print("-" * len(head))

    for name, doc in sorted(runs.items()):
        status = str(doc.get("status") or "?")
        cols = [_fmt(_get(doc, k)) for k in METRIC_KEYS]
        row = name.ljust(name_w) + _status_col(status) + "  ".join(cols)
        print(row)

    print()
    n_ok   = sum(1 for d in runs.values() if d.get("status") == "success")
    n_part = sum(1 for d in runs.values() if d.get("status") == "partial")
    n_fail = sum(1 for d in runs.values() if d.get("status") not in ("success", "partial"))
    print(f"Scenarios: {len(runs)}  success: {n_ok}  partial: {n_part}  failed/error: {n_fail}")
    return 0


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------

def _delta_tag(key: str, before: float, after: float, threshold: float) -> str:
    """Return a coloured delta string."""
    if before == 0:
        return "  n/a   "
    pct = (after - before) / abs(before) * 100.0
    higher_better = HIGHER_BETTER.get(key, False)
    improved = (pct > 0) if higher_better else (pct < 0)
    if abs(pct) < threshold:
        tag = f"{pct:+.1f}%".rjust(7)
        return tag
    tag = f"{pct:+.1f}%".rjust(7)
    if improved:
        return _c(tag + " ✓", GREEN)
    return _c(tag + " ✗", RED)


def cmd_compare(before_root: Path, after_root: Path, threshold: float, json_out: Path | None) -> int:
    before_runs = _collect(before_root)
    after_runs  = _collect(after_root)

    if not before_runs and not after_runs:
        print("No metrics.json files found in either directory.", file=sys.stderr)
        return 2

    all_names = sorted(set(before_runs) | set(after_runs))
    regressions: list[tuple[str, str, float, float, float]] = []

    col_name  = max((len(n) for n in all_names), default=10) + 2
    col_mname = max(len(k) for k in METRIC_KEYS) + 2

    hdr = (
        "Scenario".ljust(col_name)
        + "Metric".ljust(col_mname)
        + "Before".rjust(9)
        + "  →  "
        + "After".rjust(9)
        + "  Delta"
    )
    print(_c(hdr, BOLD))
    print("-" * (len(hdr) + 4))

    report_rows: list[dict] = []

    for name in all_names:
        bdoc = before_runs.get(name, {})
        adoc = after_runs.get(name, {})
        bst  = bdoc.get("status", "missing")
        ast  = adoc.get("status", "missing")

        if bst == "missing" or ast == "missing":
            marker = _c("(only in before)", YELLOW) if ast == "missing" else _c("(only in after)", YELLOW)
            print(f"{name.ljust(col_name)}{marker}")
            continue

        printed_name = False
        for key in METRIC_KEYS:
            bv = _get(bdoc, key)
            av = _get(adoc, key)
            if bv is None and av is None:
                continue

            bstr = _fmt(bv, 9)
            astr = _fmt(av, 9)

            if bv is not None and av is not None:
                delta = _delta_tag(key, bv, av, threshold)
                pct = (av - bv) / abs(bv) * 100.0 if bv != 0 else 0.0
                higher_better = HIGHER_BETTER.get(key, False)
                is_regression = abs(pct) >= threshold and ((pct < 0) if higher_better else (pct > 0))
                if is_regression:
                    regressions.append((name, key, bv, av, pct))
            else:
                delta = "  n/a"
                is_regression = False

            prefix = name.ljust(col_name) if not printed_name else " " * col_name
            printed_name = True
            print(f"{prefix}{key.ljust(col_mname)}{bstr}  →  {astr}  {delta}")
            report_rows.append({
                "scenario": name,
                "metric": key,
                "before": bv,
                "after": av,
            })

        if not printed_name:
            print(f"{name.ljust(col_name)}(no numeric metrics)")

    print()
    if regressions:
        print(_c(f"Regressions detected ({len(regressions)}):", RED + BOLD))
        for sc, k, bv, av, pct in regressions:
            print(f"  {sc}  {k}  {bv:.3f} → {av:.3f}  ({pct:+.1f}%)")
    else:
        print(_c("No regressions above threshold.", GREEN))

    if json_out is not None:
        report = {
            "before": str(before_root),
            "after": str(after_root),
            "threshold_pct": threshold,
            "regressions": [
                {"scenario": s, "metric": m, "before": b, "after": a, "delta_pct": p}
                for s, m, b, a, p in regressions
            ],
            "rows": report_rows,
        }
        json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"JSON report written to {json_out}")

    return 1 if regressions else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="compare_runs",
        description="Analyse or compare DecayCore headless run metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd")

    # -- summary sub-command (also available as bare --summary PATH) ----------
    p_sum = sub.add_parser("summary", help="Show metrics table for one run directory.")
    p_sum.add_argument("path", type=Path, help="Directory containing <scenario>/metrics.json files.")

    p_cmp = sub.add_parser("compare", help="Diff two run directories.")
    p_cmp.add_argument("before", type=Path, help="Baseline run directory.")
    p_cmp.add_argument("after", type=Path, help="Current run directory.")
    p_cmp.add_argument("--threshold", type=float, default=2.0,
                       help="Minimum %% change to flag as significant (default: 2.0).")
    p_cmp.add_argument("--json", type=Path, dest="json_out", metavar="PATH",
                       help="Write JSON report to PATH.")

    # Legacy flat flags for convenience / shell script integration
    parser.add_argument("--summary", type=Path, metavar="DIR",
                        help="(shorthand) Show metrics table for DIR.")
    parser.add_argument("--before", type=Path, metavar="DIR")
    parser.add_argument("--after",  type=Path, metavar="DIR")
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="Minimum %% change to flag (default: 2.0).")
    parser.add_argument("--json", type=Path, dest="json_out", metavar="PATH",
                        help="Write JSON report to PATH.")

    args = parser.parse_args(argv)

    # Subcommand routing
    if args.cmd == "summary":
        return cmd_summary(args.path)

    if args.cmd == "compare":
        return cmd_compare(args.before, args.after, args.threshold, args.json_out)

    # Flat-flag routing
    if args.summary:
        return cmd_summary(args.summary)

    if args.before and args.after:
        return cmd_compare(args.before, args.after, args.threshold, args.json_out)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
