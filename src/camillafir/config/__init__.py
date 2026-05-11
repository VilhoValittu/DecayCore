from pathlib import Path

import decaycore.config as _decaycore_config

__path__ = [str(Path(__file__).resolve().parent), *list(getattr(_decaycore_config, "__path__", []))]
