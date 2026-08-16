# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""In-app equivalents of the config_delete/ maintenance scripts.

Two independent operations:

- ``clear_auto_mode_disk_caches()`` removes the regenerable automatic-mode
  state (Optuna journals, the auto-mode result cache, filter priors) so the
  next run recomputes from scratch. User settings are kept.
- ``reset_user_settings()`` removes ``config.json`` so the next load falls
  back to pristine defaults. Caches are kept.

Neither touches user content: saved target presets, measurements under
``Documents/DecayCore/measurement`` and exports under
``Documents/DecayCore/filters`` all survive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..app_paths import decaycore_data_dir

logger = logging.getLogger("DecayCore")

# Mirrors the target patterns in config_delete/delete_decaycore_data_*.sh.
# "decaycore*optuna*" covers both the journal .log files and their .lock pairs.
_AUTO_CACHE_PATTERNS: tuple[str, ...] = (
    "decaycore*optuna*",
    "decaycore_auto_mode_cache_*.json",
    "decaycore_auto_mode_cache_*.json.tmp",
    "auto_mode_filter_priors.json",
)


@dataclass(frozen=True)
class ResetOutcome:
    """Result of a reset operation.

    Deletion continues past individual failures (a locked Optuna journal on
    Windows, for example), so both lists can be non-empty at once.
    """

    removed: list[Path] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def removed_count(self) -> int:
        return len(self.removed)

    @property
    def failed_count(self) -> int:
        return len(self.failed)


def _delete_files(paths: list[Path]) -> ResetOutcome:
    """Unlink each path, collecting successes and failures separately."""
    removed: list[Path] = []
    failed: list[tuple[Path, str]] = []
    for path in paths:
        try:
            # Guard against ever removing a directory such as target_presets/.
            if not path.is_file():
                continue
            path.unlink()
            removed.append(path)
        except OSError as exc:
            logger.exception("Reset failed to remove: %s", path)
            failed.append((path, f"{type(exc).__name__}: {exc}"))
    return ResetOutcome(removed=removed, failed=failed)


def auto_mode_disk_cache_files() -> list[Path]:
    """List the auto-mode cache files that a reset would remove."""
    data_dir = Path(decaycore_data_dir())
    matches: list[Path] = []
    seen: set[Path] = set()
    for pattern in _AUTO_CACHE_PATTERNS:
        try:
            candidates = sorted(data_dir.glob(pattern))
        except OSError:
            logger.exception("Reset failed to scan data dir with pattern: %s", pattern)
            continue
        for path in candidates:
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            matches.append(path)
    return matches


def clear_auto_mode_disk_caches() -> ResetOutcome:
    """Remove automatic-mode disk caches, then drop their in-memory mirrors.

    Order matters: the runtime cache memoizes the loaded auto-mode cache by
    path and writes it back on the next run, so clearing memory first would
    let the deleted files reappear.
    """
    outcome = _delete_files(auto_mode_disk_cache_files())

    from .runtime_cache import reset_runtime_caches  # noqa: PLC0415

    reset_runtime_caches()
    logger.info(
        "Auto-mode disk caches cleared: %d removed, %d failed",
        outcome.removed_count,
        outcome.failed_count,
    )
    return outcome


def user_settings_file() -> Path:
    """Return the config.json path that load/save actually use.

    ``decaycore_config.CONFIG_FILE`` is resolved once at import time and is
    the path every read and write goes through, so it -- not a fresh call to
    ``decaycore_config_path()`` -- is what a reset must remove.
    """
    from ..config.decaycore_config import CONFIG_FILE  # noqa: PLC0415

    return Path(CONFIG_FILE)


def reset_user_settings() -> ResetOutcome:
    """Remove config.json so the next load returns pristine defaults."""
    outcome = _delete_files([user_settings_file()])
    logger.info(
        "User settings reset: %d removed, %d failed",
        outcome.removed_count,
        outcome.failed_count,
    )
    return outcome
