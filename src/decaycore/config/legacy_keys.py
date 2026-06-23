"""Legacy camillafir_* key constants and helpers.

All string literals for keys that originated under the old "camillafir"
namespace live here.  Import from this module instead of spelling the
strings inline so that call-sites never disagree on the exact key name.
"""

from __future__ import annotations

from typing import Any

# Data-dict key: whether the pipeline was requested to run in AUTO mode.
# Superseded by data["mode"] == "AUTO", but still written and read for
# backwards-compatible config files.
CAMILLAFIR_AUTO_MODE: str = "camillafir_automatic_mode"

# Summary / metrics key: wall-clock time of the DSP engine run.
CAMILLAFIR_RUNTIME_S: str = "camillafir_runtime_s"


def is_auto_mode(data: dict[str, Any], mode_u: str = "") -> bool:
    """Return True when AUTO mode is active.

    Checks both the canonical ``mode_u`` string and the legacy
    ``camillafir_automatic_mode`` flag so that callers do not need to
    duplicate this two-condition check.
    """
    return bool(mode_u == "AUTO" or data.get(CAMILLAFIR_AUTO_MODE, False))
