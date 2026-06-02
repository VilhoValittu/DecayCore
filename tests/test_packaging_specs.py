from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_NAMES = (
    "DecayCore_standalone.spec",
    "DecayCore_linux.spec",
    "DecayCore_macos.spec",
)


def _execute_spec(spec_name: str):
    spec_path = REPO_ROOT / spec_name
    captured = {}

    def analysis(*args, **kwargs):
        captured["analysis"] = {"args": args, "kwargs": kwargs}
        return SimpleNamespace(
            pure=[],
            zipped_data=[],
            scripts=[],
            binaries=[],
            datas=[],
        )

    def pyz(*args, **kwargs):
        captured["pyz"] = {"args": args, "kwargs": kwargs}
        return SimpleNamespace()

    def exe(*args, **kwargs):
        captured["exe"] = {"args": args, "kwargs": kwargs}
        return SimpleNamespace()

    def collect(*args, **kwargs):
        captured["collect"] = {"args": args, "kwargs": kwargs}
        return SimpleNamespace()

    def bundle(*args, **kwargs):
        captured["bundle"] = {"args": args, "kwargs": kwargs}
        return SimpleNamespace()

    namespace = {
        "__file__": str(spec_path),
        "SPECPATH": str(spec_path.parent),
        "Analysis": analysis,
        "PYZ": pyz,
        "EXE": exe,
        "COLLECT": collect,
        "BUNDLE": bundle,
    }

    exec(spec_path.read_text(encoding="utf-8"), namespace)

    return namespace, captured


@pytest.mark.parametrize(
    "spec_name",
    SPEC_NAMES,
)
def test_pyinstaller_specs_resolve_project_paths(spec_name: str):
    namespace, captured = _execute_spec(spec_name)
    analysis_kwargs = captured["analysis"]["kwargs"]
    datas = analysis_kwargs["datas"]
    data_sources = [Path(source) for source, _ in datas]
    manual_sources = [Path(source) for source, target in datas if target == "docs"]

    assert namespace["PROJECT_ROOT"] == REPO_ROOT
    assert namespace["SRC_ROOT"] == REPO_ROOT / "src"
    assert namespace["DOCS_ROOT"] == REPO_ROOT / "docs"
    assert namespace["HOOKS_ROOT"] == REPO_ROOT / "pyinstaller_hooks"
    assert Path(namespace["ENTRYPOINT"]) == REPO_ROOT / "src" / "decaycore" / "__main__.py"

    assert analysis_kwargs["pathex"] == [str(REPO_ROOT), str(REPO_ROOT / "src")]
    assert analysis_kwargs["hookspath"] == [str(REPO_ROOT / "pyinstaller_hooks")]
    assert manual_sources == [REPO_ROOT / "docs" / "User_Manual.md"]
    assert all(path.is_absolute() for path in data_sources)
    assert all(path.exists() for path in data_sources)
    assert all(Path(path).exists() for path in analysis_kwargs["hookspath"])


@pytest.mark.parametrize("spec_name", SPEC_NAMES)
def test_packaging_specs_reference_expected_icons(spec_name: str):
    _, captured = _execute_spec(spec_name)

    if spec_name == "DecayCore_standalone.spec":
        assert Path(captured["exe"]["kwargs"]["icon"]) == REPO_ROOT / "src" / "decaycore" / "ui" / "assets" / "DecayCore.ico"
        assert Path(captured["exe"]["kwargs"]["icon"]).exists()

    if spec_name == "DecayCore_macos.spec":
        assert Path(captured["bundle"]["kwargs"]["icon"]) == REPO_ROOT / "src" / "decaycore" / "ui" / "assets" / "DecayCore.icns"
        assert Path(captured["bundle"]["kwargs"]["icon"]).exists()


@pytest.mark.parametrize("spec_name", SPEC_NAMES)
def test_packaging_specs_include_all_ui_assets(spec_name: str):
    _, captured = _execute_spec(spec_name)
    datas = captured["analysis"]["kwargs"]["datas"]

    assert (
        str(REPO_ROOT / "src" / "decaycore" / "ui" / "assets"),
        "decaycore/ui/assets",
    ) in datas


@pytest.mark.parametrize("spec_name", SPEC_NAMES)
def test_packaging_specs_include_lazy_runtime_dependencies(spec_name: str):
    _, captured = _execute_spec(spec_name)
    hiddenimports = captured["analysis"]["kwargs"]["hiddenimports"]

    assert "sounddevice" in hiddenimports
    assert "optuna" in hiddenimports
    assert "decaycore_scoring" in hiddenimports
    assert "decaycore_scoring.decaycore_scoring" in hiddenimports


@pytest.mark.parametrize("spec_name", SPEC_NAMES)
def test_packaging_specs_include_dynamic_part_modules(spec_name: str):
    _, captured = _execute_spec(spec_name)
    hiddenimports = captured["analysis"]["kwargs"]["hiddenimports"]

    assert "decaycore.config.pipeline_parts.managed_settings" in hiddenimports
    assert "decaycore.auto_mode.orchestrator_finalize_cache_parts.orchestrator_finalize_cache_01" in hiddenimports


@pytest.mark.parametrize("spec_name", SPEC_NAMES)
def test_packaging_specs_exclude_pywebio(spec_name: str):
    _, captured = _execute_spec(spec_name)
    analysis_kwargs = captured["analysis"]["kwargs"]
    hiddenimports = analysis_kwargs["hiddenimports"]
    excludes = analysis_kwargs["excludes"]

    assert "pywebio" in excludes
    assert all(not item.startswith("pywebio") for item in hiddenimports)


def test_plotly_hook_does_not_force_pandas_or_duplicate_plotly_js():
    hook_text = (REPO_ROOT / "pyinstaller_hooks" / "hook-plotly.py").read_text(encoding="utf-8")

    assert '"pandas"' not in hook_text
    assert "package_data/plotly.min.js" not in hook_text


def test_nicegui_hook_filters_source_maps():
    hook_text = (REPO_ROOT / "pyinstaller_hooks" / "hook-nicegui.py").read_text(encoding="utf-8")

    assert 'collect_data_files("nicegui")' in hook_text
    assert 'endswith(".map")' in hook_text


def test_release_workflow_verifies_manual_in_macos_app_bundle_layout():
    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release-build.yml"
    workflow_text = workflow_path.read_text(encoding="utf-8")

    assert 'find dist/DecayCore-${VERSION}.app -path "*docs/User_Manual.md" -print -quit | grep -q .' in workflow_text
    assert "test -f dist/DecayCore/docs/User_Manual.md" not in workflow_text
    assert 'Test-Path "dist\\DecayCore\\docs\\User_Manual.md"' not in workflow_text


def test_release_workflow_skips_matplotlib_warmup_when_dependency_is_missing():
    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release-build.yml"
    workflow_text = workflow_path.read_text(encoding="utf-8")

    assert workflow_text.count('import importlib.util') == 1
    assert workflow_text.count('importlib.util.find_spec("matplotlib") is None') == 1
    assert workflow_text.count('matplotlib not installed; skipping cache warm-up') == 1


def test_release_workflow_installs_rust_scoring_extension_before_packaging():
    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release-build.yml"
    workflow_text = workflow_path.read_text(encoding="utf-8")

    assert "pip install pyinstaller maturin" in workflow_text
    assert "python -m pip install pyinstaller maturin" in workflow_text
    assert workflow_text.count("python -m pip install ./decaycore-scoring") == 1


def test_installation_guide_mentions_linux_portaudio_runtime_dependency():
    installation_path = Path(__file__).resolve().parents[1] / "docs" / "Installation.md"
    installation_text = installation_path.read_text(encoding="utf-8")

    assert "libportaudio2" in installation_text
    assert "measurement audio" in installation_text.lower()
