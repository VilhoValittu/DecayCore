from pathlib import Path

import decaycore.ui as _decaycore_ui

__path__ = [str(Path(__file__).resolve().parent), *list(getattr(_decaycore_ui, "__path__", []))]
