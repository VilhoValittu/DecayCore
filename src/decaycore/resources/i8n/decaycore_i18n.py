# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

import json
import locale
import os
import sys
from pathlib import Path


def get_resource_path(relative_path: str) -> str:
    """Hakee tai ratkaisee: get resource path."""
    if hasattr(sys, "_MEIPASS"):
        base = os.path.join(sys._MEIPASS, relative_path)
        if os.path.exists(base):
            return base
        alt = os.path.join(sys._MEIPASS, "decaycore", "resources", relative_path)
        if os.path.exists(alt):
            return alt
        return os.path.join(sys._MEIPASS, "camillafir", "resources", relative_path)
    package_root = Path(__file__).resolve().parents[1]
    return str(package_root / relative_path)


TRANS_FILE = get_resource_path(os.path.join("i8n", "translations.json"))


def load_translations() -> tuple[dict, dict]:
    try:
        with open(TRANS_FILE, encoding="utf-8") as f:
            payload = json.load(f)
            if not isinstance(payload, dict):
                raise ValueError("translations.json must contain a top-level object")

            meta = payload.get("_meta", {})
            translations = {
                key: value for key, value in payload.items() if isinstance(key, str) and not key.startswith("_")
            }
            return translations, meta if isinstance(meta, dict) else {}
    except Exception as e:
        print("❌ Failed to load translations:", TRANS_FILE)
        print("Reason:", e)
        raise


TRANSLATIONS, TRANSLATIONS_META = load_translations()


def _reload_translations_in_place() -> None:
    global TRANSLATIONS, TRANSLATIONS_META
    translations, meta = load_translations()
    TRANSLATIONS = translations
    TRANSLATIONS_META = meta


def t(key: str) -> str:
    """Funktio: t."""
    lang = locale.getlocale()[0]
    lang = "fi" if lang and "fi" in lang.lower() else "en"

    if key == "zoom_hint":
        return "(Vinkki: Voit zoomata hiirellä kuvaajaa)" if lang == "fi" else "(Hint: Use mouse to zoom)"
    catalog = TRANSLATIONS.get(lang, TRANSLATIONS.get("en", {}))
    translated = catalog.get(key, None)
    if translated is not None:
        return translated

    # Hot-reload or partial module reload can update UI code before this module,
    # so retry once from disk when a key is missing.
    _reload_translations_in_place()
    catalog = TRANSLATIONS.get(lang, TRANSLATIONS.get("en", {}))
    return catalog.get(key, key)
