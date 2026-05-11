# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Re-export shim: public API of candidate_generation.

Implementation is split across:
  - candidate_base.py            constants, helpers, builtin/optuna-coarse builders
  - candidate_optuna_coarse.py   coarse Optuna suggest + seed functions
  - candidate_optuna_local.py    local + micro Optuna suggest, seed, and builtin local builders
"""

# Public symbols used by orchestrator_finalize, orchestrator_refine,
# orchestrator_target, and api modules.
from .candidate_base import (
    _build_auto_mode_candidates,
    _build_auto_mode_candidates_optuna,
)
from .candidate_optuna_coarse import (
    _seed_auto_mode_candidate_optuna_params,
    _suggest_auto_mode_candidate_optuna,
)
from .candidate_optuna_local import (
    _seed_auto_mode_candidate_local_optuna_params,
    _suggest_auto_mode_candidate_local_optuna,
)
from .candidate_optuna_micro import (
    _seed_auto_mode_candidate_micro_optuna_params,
    _suggest_auto_mode_candidate_micro_optuna,
)
from .candidate_local_builders import (
    _build_auto_mode_candidates_local,
    _build_auto_mode_candidates_micro,
)

__all__ = [
    "_build_auto_mode_candidates",
    "_build_auto_mode_candidates_local",
    "_build_auto_mode_candidates_micro",
    "_build_auto_mode_candidates_optuna",
    "_seed_auto_mode_candidate_local_optuna_params",
    "_seed_auto_mode_candidate_micro_optuna_params",
    "_seed_auto_mode_candidate_optuna_params",
    "_suggest_auto_mode_candidate_local_optuna",
    "_suggest_auto_mode_candidate_micro_optuna",
    "_suggest_auto_mode_candidate_optuna",
]
