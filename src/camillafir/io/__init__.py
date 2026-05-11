from pathlib import Path

import decaycore.io as _decaycore_io

__path__ = [str(Path(__file__).resolve().parent), *list(getattr(_decaycore_io, "__path__", []))]
