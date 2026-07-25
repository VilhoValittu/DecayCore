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

from ._constants import *
from .safe_values import _auto_safe_bool, _auto_safe_int

logger = logging.getLogger("DecayCore")

def _auto_optimizer_backend(
    base_data: dict | None, *, default_optuna_enabled: bool = False
) -> str:
    env_raw = (
        str(os.environ.get("DECAYCORE_AUTO_MODE_OPTIMIZER", os.environ.get("CAMILLAFIR_AUTO_MODE_OPTIMIZER", "")) or "").strip().lower()
    )
    if env_raw in ("builtin", "optuna"):
        return str(env_raw)

    data = dict(base_data or {})
    raw = str(data.get("auto_mode_optimizer", "") or "").strip().lower()
    if raw in ("builtin", "optuna"):
        return str(raw)

    if _auto_safe_bool(
        data.get("auto_mode_optuna", default_optuna_enabled), default_optuna_enabled
    ):
        return "optuna"
    return "builtin"

def _auto_optuna_sampler_kwargs(base_data: dict | None, *, workers: int = 1) -> dict:
    data = dict(base_data or {})
    multivariate = _auto_safe_bool(
        data.get("auto_mode_optuna_multivariate", AUTO_MODE_OPTUNA_MULTIVARIATE),
        AUTO_MODE_OPTUNA_MULTIVARIATE,
    )
    group = bool(multivariate) and _auto_safe_bool(
        data.get("auto_mode_optuna_group", AUTO_MODE_OPTUNA_GROUP),
        AUTO_MODE_OPTUNA_GROUP,
    )
    constant_liar = bool(
        int(max(1, _auto_safe_int(workers, 1))) > 1
    ) and _auto_safe_bool(
        data.get("auto_mode_optuna_constant_liar", AUTO_MODE_OPTUNA_CONSTANT_LIAR),
        AUTO_MODE_OPTUNA_CONSTANT_LIAR,
    )
    return {
        "multivariate": bool(multivariate),
        "group": bool(group),
        "constant_liar": bool(constant_liar),
    }

def _auto_trial_workers(base_data: dict | None, n_trials: int) -> int:
    trials_n = int(max(1, _auto_safe_int(n_trials, 1)))
    min_trials = int(max(1, _auto_safe_int(AUTO_MODE_PARALLEL_MIN_TRIALS, 1)))
    if (not bool(AUTO_MODE_PARALLEL_ENABLED)) or trials_n < min_trials:
        return 1
    cpu_n = int(max(1, _auto_safe_int(os.cpu_count(), 1)))
    env_raw = os.environ.get("DECAYCORE_AUTO_MODE_WORKERS", os.environ.get("CAMILLAFIR_AUTO_MODE_WORKERS", "")).strip()
    req = _auto_safe_int((base_data or {}).get("auto_mode_workers", 0), 0)
    if env_raw:
        req = _auto_safe_int(env_raw, req)
    if req <= 0:
        # The current Optuna evaluator uses Python threads. DSP trials contend
        # on the GIL and large shared caches, so CPU-count auto sizing is slower
        # than the deterministic sequential path on representative workloads.
        req = 1
    hard_max = int(max(0, _auto_safe_int(AUTO_MODE_PARALLEL_MAX_WORKERS, 0)))
    if hard_max > 0:
        req = min(req, hard_max)
    # Keep at least `min_trials` work items per worker so tiny runs stay sequential.
    trial_budget_cap = int(max(1, trials_n // min_trials))
    req = int(max(1, min(int(req), int(cpu_n), trial_budget_cap)))
    return int(req)

def _auto_trial_chunk_size(workers: int) -> int:
    w = int(max(1, _auto_safe_int(workers, 1)))
    mul = int(max(1, _auto_safe_int(AUTO_MODE_PARALLEL_BATCH_MULTIPLIER, 2)))
    return int(max(w, w * mul))


__all__ = ['_auto_optimizer_backend', '_auto_optuna_sampler_kwargs', '_auto_trial_workers', '_auto_trial_chunk_size']
