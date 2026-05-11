from pathlib import Path

import decaycore.resources as _decaycore_resources

__path__ = [str(Path(__file__).resolve().parent), *list(getattr(_decaycore_resources, "__path__", []))]
