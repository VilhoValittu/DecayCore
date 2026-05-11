from pathlib import Path

import decaycore.dsp as _decaycore_dsp

__path__ = [str(Path(__file__).resolve().parent), *list(getattr(_decaycore_dsp, "__path__", []))]
