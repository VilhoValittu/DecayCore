# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Auto-mode cache path helpers."""

from __future__ import annotations

import os

from ..app_paths import decaycore_data_dir, program_version_token
from .shared import AUTO_MODE_CACHE_FILENAME, logger


def _auto_cache_compat_token(compat_version: str | None = None) -> str:
    token = str(program_version_token(compat_version, default="") or "").strip()
    return str(token)


def _auto_cache_filename(*, compat_version: str | None = None) -> str:
    token = _auto_cache_compat_token(compat_version)
    if not token:
        return str(AUTO_MODE_CACHE_FILENAME)
    stem, ext = os.path.splitext(str(AUTO_MODE_CACHE_FILENAME))
    return f"{stem}_{token}{ext or '.json'}"


def _auto_cache_path(*, compat_version: str | None = None) -> str:
    filename = _auto_cache_filename(compat_version=compat_version)
    preferred_base = os.fspath(decaycore_data_dir())
    preferred_path = os.path.join(preferred_base, filename)
    legacy_base = os.path.join(os.path.expanduser("~"), ".camillafir")
    legacy_path = os.path.join(legacy_base, filename)

    try:
        os.makedirs(preferred_base, exist_ok=True)
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
        try:
            os.makedirs(legacy_base, exist_ok=True)
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
            logger.exception("cache dir create fallback")
        return legacy_path

    try:
        source_candidates = [legacy_path]
        if str(filename) != str(AUTO_MODE_CACHE_FILENAME):
            source_candidates.extend(
                (
                    os.path.join(preferred_base, AUTO_MODE_CACHE_FILENAME),
                    os.path.join(legacy_base, AUTO_MODE_CACHE_FILENAME),
                )
            )
        source_path = next(
            (
                path
                for path in source_candidates
                if path != preferred_path and os.path.isfile(path)
            ),
            None,
        )
        if (not os.path.isfile(preferred_path)) and source_path:
            try:
                os.replace(source_path, preferred_path)
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
                with open(source_path, "rb") as src_f:
                    payload = src_f.read()
                with open(preferred_path, "wb") as dst_f:
                    dst_f.write(payload)
                try:
                    os.remove(source_path)
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
                    logger.exception("cache source file remove after copy")
            logger.info(f"Automatic mode cache migrated to: {preferred_path}")
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
        return legacy_path

    return preferred_path


def get_auto_mode_cache_path(*, compat_version: str | None = None) -> str:
    return _auto_cache_path(compat_version=compat_version)
