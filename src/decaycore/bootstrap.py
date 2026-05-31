# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

import logging
import os
import sys

logger = logging.getLogger("DecayCore")


def _auto_thread_budget() -> tuple[int, int]:
    try:
        cores = int(os.cpu_count() or 1)
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
    ):
        cores = 1
    cores = max(1, int(cores))
    if sys.platform == "win32":
        use = 1
    else:
        use = max(1, int((cores * 2) // 4))
    return int(use), int(cores)


def _apply_auto_thread_env() -> tuple[int, int, list[str]]:
    use, cores = _auto_thread_budget()
    keys = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    applied: list[str] = []
    for k in keys:
        cur = str(os.environ.get(k, "") or "").strip()
        if cur:
            try:
                if int(float(cur)) > 0:
                    continue
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
            ):
                logger.exception("thread env var parse")
        os.environ[k] = str(use)
        applied.append(str(k))
    return int(use), int(cores), list(applied)


def _resolve_console_log_level() -> int:
    raw = str(os.environ.get("DECAYCORE_LOG_LEVEL", os.environ.get("CAMILLAFIR_LOG_LEVEL", "WARNING")) or "WARNING").strip().upper()
    return int(getattr(logging, raw, logging.WARNING))


def initialize_logging():
    auto_threads_use, auto_threads_cores, auto_threads_env_applied = _apply_auto_thread_env()
    console_log_level = _resolve_console_log_level()

    logging.basicConfig(
        level=console_log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger = logging.getLogger("DecayCore")
    logger.setLevel(console_log_level)

    try:
        _win_note = " (Windows: BLAS=1 to reduce scheduling overhead)" if sys.platform == "win32" else ""
        logger.info(
            "CPU thread budget: "
            f"{int(auto_threads_use)}/{int(auto_threads_cores)} cores "
            f"(auto for DSP/NumPy backends{_win_note})"
        )
        if auto_threads_env_applied:
            logger.info(
                "Applied thread env vars: "
                + ", ".join([f"{k}={os.environ.get(k, '')}" for k in auto_threads_env_applied])
            )
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
    ):
        logger.exception("thread budget log")

    return {
        "logger": logger,
        "console_log_level": console_log_level,
        "auto_threads_use": auto_threads_use,
        "auto_threads_cores": auto_threads_cores,
        "auto_threads_env_applied": auto_threads_env_applied,
    }
