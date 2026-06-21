import re
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_NAMES = (
    "DecayCore_standalone.spec",
    "DecayCore_linux.spec",
    "DecayCore_macos.spec",
)


def _load_spec_common():
    import importlib.util

    spec_common_path = REPO_ROOT / "decaycore_spec_common.py"
    module_spec = importlib.util.spec_from_file_location("decaycore_spec_common", spec_common_path)
    spec_common = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(spec_common)
    return spec_common


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


def test_all_repository_pyinstaller_specs_are_known_shared_specs():
    actual = tuple(sorted(path.name for path in REPO_ROOT.glob("*.spec")))
    assert actual == tuple(sorted(SPEC_NAMES))


def test_all_pyinstaller_specs_import_shared_spec_config():
    for spec_name in SPEC_NAMES:
        spec_text = (REPO_ROOT / spec_name).read_text(encoding="utf-8")
        assert "import decaycore_spec_common as spec_common" in spec_text
        assert "spec_common.build_datas(" in spec_text
        assert "spec_common.build_hiddenimports(" in spec_text
        assert "spec_common.build_excludes()" in spec_text


def test_all_pyinstaller_build_commands_use_shared_specs():
    files_to_check = (
        REPO_ROOT / "linux_package.sh",
        REPO_ROOT / "windows_package.ps1",
        REPO_ROOT / ".github" / "workflows" / "release-build.yml",
    )
    known_specs = set(SPEC_NAMES)
    build_command_pattern = re.compile(
        r"(?:python -m PyInstaller|\"\$PYTHON\" -m PyInstaller|pyinstaller --clean)\b[^\n]*"
    )
    for path in files_to_check:
        text = path.read_text(encoding="utf-8")
        for match in build_command_pattern.finditer(text):
            command = match.group(0)
            referenced_specs = set(re.findall(r"\bDecayCore_[A-Za-z0-9_]+\.spec\b", command))
            assert referenced_specs, f"{path.name} has PyInstaller command without a spec: {command}"
            assert referenced_specs <= known_specs, f"{path.name} uses unknown spec in: {command}"


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
        assert captured["bundle"]["kwargs"]["bundle_identifier"] == "com.github.vilhovalittu.decaycore"
        info_plist = captured["bundle"]["kwargs"]["info_plist"]
        assert "NSMicrophoneUsageDescription" in info_plist
        assert "NSDocumentsFolderUsageDescription" in info_plist


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
    assert "pkg_resources._vendor.platformdirs" in hiddenimports
    assert "platformdirs" in hiddenimports


@pytest.mark.parametrize("spec_name", SPEC_NAMES)
def test_packaging_specs_include_dynamic_part_modules(spec_name: str):
    _, captured = _execute_spec(spec_name)
    hiddenimports = captured["analysis"]["kwargs"]["hiddenimports"]

    assert "decaycore.config.pipeline_parts.managed_settings" in hiddenimports
    assert "decaycore.auto_mode.orchestrator_finalize_cache_parts.cache_finalize_status" in hiddenimports


@pytest.mark.parametrize("spec_name", SPEC_NAMES)
def test_packaging_specs_exclude_pywebio(spec_name: str):
    _, captured = _execute_spec(spec_name)
    analysis_kwargs = captured["analysis"]["kwargs"]
    hiddenimports = analysis_kwargs["hiddenimports"]
    excludes = analysis_kwargs["excludes"]

    assert "pywebio" in excludes
    assert all(not item.startswith("pywebio") for item in hiddenimports)


# Optional heavy subpackages that no platform build should bundle. Keeping this
# parity check stops the Windows/macOS specs from drifting back to pulling in
# pyarrow and unused Plotly/NiceGUI helpers (the drift this shared config fixed).
_REQUIRED_OPTIONAL_EXCLUDES = (
    "pyarrow",
    "plotly.express",
    "plotly.data",
    "plotly.figure_factory",
    "scipy.cluster",
    "uvloop",
    "watchfiles",
    "nicegui.elements.aggrid",
    "nicegui.elements.codemirror",
    "webview",
)


@pytest.mark.parametrize("spec_name", SPEC_NAMES)
def test_packaging_specs_share_optional_excludes(spec_name: str):
    _, captured = _execute_spec(spec_name)
    excludes = captured["analysis"]["kwargs"]["excludes"]

    for name in _REQUIRED_OPTIONAL_EXCLUDES:
        assert name in excludes, f"{spec_name} is missing exclude {name!r}"
    assert "nicegui.native" not in excludes, f"{spec_name} must keep NiceGUI native fallback package"


def test_all_specs_use_identical_excludes():
    exclude_sets = {
        spec_name: set(_execute_spec(spec_name)[1]["analysis"]["kwargs"]["excludes"])
        for spec_name in SPEC_NAMES
    }
    reference = exclude_sets["DecayCore_linux.spec"]
    for spec_name, excludes in exclude_sets.items():
        assert excludes == reference, f"{spec_name} excludes diverge from linux spec"


def test_report_prune_paths_match_optional_excludes():
    spec_common = _load_spec_common()
    prune_paths = spec_common.report_prune_paths()

    # No duplicates, and no path nested under another already in the list.
    assert len(prune_paths) == len(set(prune_paths))
    for path in prune_paths:
        assert not any(path != other and path.startswith(other + "/") for other in prune_paths)

    # Every optional exclude is covered by exactly one reported parent path.
    for name in spec_common._OPTIONAL_EXCLUDES:
        as_path = name.replace(".", "/")
        assert any(as_path == p or as_path.startswith(p + "/") for p in prune_paths)

    for path in spec_common.NICEGUI_DATA_EXCLUDE_PREFIXES:
        assert any(path == p or path.startswith(p + "/") for p in prune_paths)


def test_nicegui_data_filter_drops_unused_element_assets_but_keeps_plotly():
    spec_common = _load_spec_common()
    datas = [
        ("/venv/site-packages/nicegui/elements/aggrid/dist/index.js", "nicegui/elements/aggrid"),
        ("/venv/site-packages/nicegui/elements/codemirror/dist/index.js", "nicegui/elements/codemirror"),
        ("/venv/site-packages/nicegui/elements/plotly/dist/index.js", "nicegui/elements/plotly"),
        ("/venv/site-packages/nicegui/static/quasar.umd.js", "nicegui/static"),
    ]

    assert spec_common.filter_nicegui_datas(datas) == [
        ("/venv/site-packages/nicegui/elements/plotly/dist/index.js", "nicegui/elements/plotly"),
        ("/venv/site-packages/nicegui/static/quasar.umd.js", "nicegui/static"),
    ]


def test_package_scripts_derive_prune_report_from_shared_config():
    script_text = (REPO_ROOT / "linux_package.sh").read_text(encoding="utf-8")
    workflow_text = (REPO_ROOT / ".github" / "workflows" / "release-build.yml").read_text(
        encoding="utf-8"
    )

    # The size report must derive its prune list from the shared module rather
    # than carrying a hand-maintained copy that can drift from the spec excludes.
    assert "report_prune_paths()" in script_text
    assert "for rel in \\" not in script_text
    assert '(cd dist && 7z a -t7z -mx=9 -mmt=on -m0=lzma2 "../out/DecayCore_${VERSION}_linux.7z" "$PACK_NAME")' in script_text
    assert "| head -n 20" not in script_text
    assert "report_prune_paths()" in workflow_text
    assert "for rel in \\" not in workflow_text


def test_plotly_hook_does_not_force_pandas_or_duplicate_plotly_js():
    hook_text = (REPO_ROOT / "pyinstaller_hooks" / "hook-plotly.py").read_text(encoding="utf-8")

    assert '"pandas"' not in hook_text
    assert "package_data/plotly.min.js" not in hook_text


def test_nicegui_hook_filters_source_maps():
    hook_text = (REPO_ROOT / "pyinstaller_hooks" / "hook-nicegui.py").read_text(encoding="utf-8")

    assert 'collect_data_files("nicegui")' in hook_text
    assert 'endswith(".map")' in hook_text
    assert "filter_nicegui_datas" in hook_text
    assert '"webview"' in hook_text


def test_release_workflow_verifies_manual_in_macos_app_bundle_layout():
    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release-build.yml"
    workflow_text = workflow_path.read_text(encoding="utf-8")

    assert 'find dist/DecayCore-${VERSION}.app -path "*docs/User_Manual.md" -print -quit | grep -q .' in workflow_text
    assert "test -f dist/DecayCore/docs/User_Manual.md" not in workflow_text
    assert 'Test-Path "dist\\DecayCore\\docs\\User_Manual.md"' not in workflow_text
    assert 'APP_BUNDLE_NAME="DecayCore_${VERSION}.app"' in workflow_text
    assert 'APP_EXECUTABLE="${APP_BUNDLE_DIR}/Contents/MacOS/DecayCore_${VERSION}"' in workflow_text
    assert 'test -x "${APP_EXECUTABLE}"' in workflow_text
    assert 'cat > "dist/Start_Decay.command" << \'EOF\'' in workflow_text
    assert 'sed -i "" "s/__DECAYCORE_VERSION__/${VERSION}/g" "dist/Start_Decay.command"' in workflow_text
    assert 'cat > "dist/README.txt" << EOF' in workflow_text
    assert '(cd dist && 7z a -t7z -mx=9 -mmt=on -m0=lzma2 "../out/DecayCore_${VERSION}_macos_arm64.7z" "${APP_BUNDLE_NAME}" "Start_Decay.command" "README.txt")' in workflow_text
    assert '7z l "out/DecayCore_${VERSION}_macos_arm64.7z" | grep -q "${APP_BUNDLE_NAME}"' in workflow_text
    assert '7z l "out/DecayCore_${VERSION}_macos_arm64.7z" | grep -q "Start_Decay.command"' in workflow_text
    assert '7z l "out/DecayCore_${VERSION}_macos_arm64.7z" | grep -q "README.txt"' in workflow_text


def test_release_workflow_packages_linux_release_assets_like_local_script():
    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release-build.yml"
    workflow_text = workflow_path.read_text(encoding="utf-8")

    assert "build_linux_arm64:" in workflow_text
    assert "name: Build Linux ARM64" in workflow_text
    assert "runs-on: ubuntu-22.04-arm" in workflow_text
    assert "build_linux_x86_64:" not in workflow_text
    assert "name: Linux x86_64 build" not in workflow_text
    assert workflow_text.count('python-version: "3.12.3"') >= 2
    assert "libsndfile1" in workflow_text
    assert "libportaudio2" in workflow_text
    assert "portaudio19-dev" in workflow_text
    assert "p7zip-full" in workflow_text
    assert "python -m pip install -r requirements-measurement.txt" in workflow_text
    assert "pyinstaller --clean --noconfirm DecayCore_linux.spec" in workflow_text
    assert workflow_text.count("Generate build_version.py") == 2
    assert 'PACK_NAME="DecayCore_${VERSION}"' in workflow_text
    assert 'cat > "${DIST_DIR}/run.sh" << EOF' in workflow_text
    assert 'chmod +x "\\$DIR/DecayCore_${VERSION}" || true' in workflow_text
    assert 'exec "\\$DIR/DecayCore_${VERSION}" "\\$@"' in workflow_text
    assert 'cp src/decaycore/ui/assets/DecayCore_logo.png "${DIST_DIR}/DecayCore.png"' in workflow_text
    assert 'cat > "${DIST_DIR}/install-desktop-entry.sh" << \'EOF\'' in workflow_text
    assert 'DESKTOP_FILE="$DESKTOP_DIR/DecayCore.desktop"' in workflow_text
    assert 'Exec=$DIR/run.sh' in workflow_text
    assert "./install-desktop-entry.sh" in workflow_text
    assert '(cd dist && 7z a -t7z -mx=9 -mmt=on -m0=lzma2 "../out/DecayCore_${VERSION}_linux_arm64.7z" "${PACK_NAME}")' in workflow_text
    assert '7z l "out/DecayCore_${VERSION}_linux_arm64.7z" | grep -q "${PACK_NAME}/install-desktop-entry.sh"' in workflow_text
    assert '7z l "out/DecayCore_${VERSION}_linux_arm64.7z" | grep -q "${PACK_NAME}/DecayCore.png"' in workflow_text
    assert '7z l "out/DecayCore_${VERSION}_linux_arm64.7z" | grep -q "${PACK_NAME}/DecayCore_${VERSION}"' in workflow_text
    assert 'linux_x86_64.7z' not in workflow_text
    assert "Download Linux ARM64 build artifact" in workflow_text
    assert "name: DecayCore-linux-arm64" in workflow_text
    assert "path: out/*.7z" in workflow_text
    assert "needs: [build_macos_arm64, build_linux_arm64]" in workflow_text


def test_release_workflow_skips_matplotlib_warmup_when_dependency_is_missing():
    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release-build.yml"
    workflow_text = workflow_path.read_text(encoding="utf-8")

    assert workflow_text.count('import importlib.util') == 1
    assert workflow_text.count('importlib.util.find_spec("matplotlib") is None') == 1
    assert workflow_text.count('matplotlib not installed; skipping cache warm-up') == 1


def test_release_workflow_installs_measurement_and_rust_extensions_before_packaging():
    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release-build.yml"
    workflow_text = workflow_path.read_text(encoding="utf-8")

    assert "pip install pyinstaller maturin" in workflow_text
    assert "python -m pip install pyinstaller maturin" in workflow_text
    assert workflow_text.count("python -m pip install -r requirements-measurement.txt") == 2
    assert workflow_text.count("python -m pip install ./decaycore-scoring") == 2
    assert workflow_text.count("python -m pip install ./decaycore-dsp") == 2
    assert "build_windows_x86_64:" not in workflow_text
    assert "DecayCore-windows-x86_64" not in workflow_text


def test_installation_guide_mentions_linux_portaudio_runtime_dependency():
    installation_path = Path(__file__).resolve().parents[1] / "docs" / "Installation.md"
    installation_text = installation_path.read_text(encoding="utf-8")

    assert "libportaudio2" in installation_text
    assert "measurement audio" in installation_text.lower()
    assert "Start_Decay.command" in installation_text


def test_installation_guide_mentions_linux_arm64_release_package():
    installation_path = Path(__file__).resolve().parents[1] / "docs" / "Installation.md"
    installation_text = installation_path.read_text(encoding="utf-8")

    assert "DecayCore_<version>_linux_arm64.7z" in installation_text
    assert "Raspberry Pi / Linux ARM64" in installation_text
    assert "32-bit Raspberry Pi OS" in installation_text
