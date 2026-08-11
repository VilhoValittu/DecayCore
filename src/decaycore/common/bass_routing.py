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

# Keep the electrical stereo-to-sub routing in one dependency-light module so
# prediction, cache identity, and CamillaDSP export cannot silently diverge.
DIRECT_DAC_BASS_ROUTING_POLICY_VERSION = 1
DIRECT_DAC_SUB_MIX_LEFT_GAIN_DB = 0.0
DIRECT_DAC_SUB_MIX_RIGHT_GAIN_DB = 0.0
__all__ = [
    "DIRECT_DAC_BASS_ROUTING_POLICY_VERSION",
    "DIRECT_DAC_SUB_MIX_LEFT_GAIN_DB",
    "DIRECT_DAC_SUB_MIX_RIGHT_GAIN_DB",
]
