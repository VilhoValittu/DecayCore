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

logger = logging.getLogger(__name__)


def _append_leveling_summary(
    summary_content: str,
    l_st: dict | None,
    r_st: dict | None,
) -> str:
    try:
        summary_content += "\n=== LEVELING ===\n"
        for side, st in [("LEFT", l_st), ("RIGHT", r_st)]:
            if not isinstance(st, dict):
                continue
            summary_content += f"[{side}]\n"
            summary_content += f"Method: {st.get('offset_method', 'n/a')}\n"
            win = st.get("smart_scan_range", None)
            if isinstance(win, (list, tuple)) and len(win) >= 2:
                try:
                    summary_content += f"Window: {float(win[0]):.0f}-{float(win[1]):.0f} Hz\n"
                except Exception:
                    logger.exception("leveling window range format")
            try:
                summary_content += f"Offset to measurement: {float(st.get('offset_db', 0.0) or 0.0):+.2f} dB\n"
            except Exception:
                logger.exception("leveling offset_db format")
            try:
                summary_content += f"Effective target level: {float(st.get('eff_target_db', 0.0) or 0.0):.2f} dB\n"
            except Exception:
                logger.exception("leveling eff_target_db format")
            tilt = st.get("tilt_slope_db_per_oct", None)
            if tilt is not None:
                try:
                    tilt_f = float(tilt)
                    summary_content += f"Tilt slope: {tilt_f:+.2f} dB/oct\n"
                    if abs(tilt_f) > 1.5:
                        summary_content += (
                            "Warning: Large broadband tilt detected. "
                            "May indicate measurement/target mismatch or strong room tilt.\n"
                        )
                except Exception:
                    logger.exception("leveling tilt slope format")
            summary_content += "\n"
    except Exception:
        logger.exception("leveling summary section")
    return summary_content


__all__ = ['_append_leveling_summary']
