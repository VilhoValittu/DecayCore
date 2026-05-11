# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Backward-compatible alias for the relocated automatic-mode API.

This module exists only to preserve legacy imports. Do not add new business
logic here; extend `decaycore.auto_mode.api` instead.
"""

import sys

from ..auto_mode import api as _api

sys.modules[__name__] = _api
