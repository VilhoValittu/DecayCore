from pathlib import Path

import decaycore.resources.i8n as _decaycore_i8n

__path__ = [str(Path(__file__).resolve().parent), *list(getattr(_decaycore_i8n, "__path__", []))]
