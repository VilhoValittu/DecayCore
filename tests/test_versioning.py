import importlib
import sys
import types
from pathlib import Path

import decaycore.app_paths as app_paths
from decaycore.app_paths import (
    default_measurements_dir,
    program_version_token,
    safe_filters_dir,
    safe_measurements_dir,
)
from decaycore.version import normalize_version


def test_normalize_version_accepts_tag_and_ui_formats():
    assert normalize_version("3.6.0") == "v.3.6.0"
    assert normalize_version("v3.6.0") == "v.3.6.0"
    assert normalize_version("v.3.6.0") == "v.3.6.0"


def test_resolve_version_prefers_package_build_version(monkeypatch):
    monkeypatch.delenv("DECAYCORE_VERSION", raising=False)
    monkeypatch.delenv("CAMILLAFIR_VERSION", raising=False)
    build_version = types.ModuleType("decaycore.build_version")
    build_version.VERSION = "3.6.1"
    sys.modules["decaycore.build_version"] = build_version
    sys.modules.pop("decaycore.version", None)
    version_mod = importlib.import_module("decaycore.version")

    assert version_mod.resolve_version() == "v.3.6.1"


def test_resolve_version_prefers_decaycore_env(monkeypatch):
    monkeypatch.setenv("DECAYCORE_VERSION", "4.2.0")
    monkeypatch.setenv("CAMILLAFIR_VERSION", "1.0.0")
    sys.modules.pop("decaycore.version", None)
    version_mod = importlib.import_module("decaycore.version")

    assert version_mod.resolve_version() == "v.4.2.0"


def test_program_version_token_and_filters_dir_use_same_version_format(tmp_path):
    version = "v.3.6.0"
    assert program_version_token(version) == "v3.6.0"

    filters_dir = safe_filters_dir(str(tmp_path), program_version=version)

    assert filters_dir.endswith("v3.6.0")


def test_default_measurements_dir_uses_documents_folder():
    measurements_dir = default_measurements_dir()

    assert measurements_dir.parts[-3:] == ("Documents", "DecayCore", "measurement")


def test_safe_measurements_dir_falls_back_to_app_data_when_primary_path_is_unwritable(monkeypatch, tmp_path):
    blocked = tmp_path / "blocked" / "measurement"
    fallback_base = tmp_path / "appdata"
    monkeypatch.setattr(app_paths, "decaycore_data_dir", lambda: fallback_base)

    original_mkdir = Path.mkdir

    def _mkdir(self, *args, **kwargs):
        if self == blocked:
            raise OSError("permission denied")
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", _mkdir)

    result = Path(safe_measurements_dir(str(blocked)))

    assert result == fallback_base / "measurement"
    assert result.is_dir()
