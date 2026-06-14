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
    assert "build_linux_x86_64:" in workflow_text
    assert "name: Linux x86_64 build" in workflow_text
    assert workflow_text.count('python-version: "3.12.3"') >= 3
    assert "libsndfile1" in workflow_text
    assert "libportaudio2" in workflow_text
    assert "portaudio19-dev" in workflow_text
    assert "p7zip-full" in workflow_text
    assert "python -m pip install -r requirements-measurement.txt" in workflow_text
    assert "pyinstaller --clean --noconfirm DecayCore_linux.spec" in workflow_text
    assert workflow_text.count("Generate build_version.py") >= 3
    assert 'PACK_NAME="DecayCore_${VERSION}"' in workflow_text
    assert 'cat > "${DIST_DIR}/run.sh" << EOF' in workflow_text
    assert 'chmod +x "$DIR/DecayCore_${VERSION}" || true' in workflow_text
    assert 'exec "$DIR/DecayCore_${VERSION}" "$@"' in workflow_text
    assert 'cp src/decaycore/ui/assets/DecayCore_logo.png "${DIST_DIR}/DecayCore.png"' in workflow_text
    assert 'cat > "${DIST_DIR}/install-desktop-entry.sh" << \'EOF\'' in workflow_text
    assert 'DESKTOP_FILE="$DESKTOP_DIR/DecayCore.desktop"' in workflow_text
    assert 'Exec=$DIR/run.sh' in workflow_text
    assert "./install-desktop-entry.sh" in workflow_text
    assert '(cd dist && 7z a -t7z -mx=9 -mmt=on -m0=lzma2 "../out/DecayCore_${VERSION}_linux_arm64.7z" "${PACK_NAME}")' in workflow_text
    assert '7z l "out/DecayCore_${VERSION}_linux_arm64.7z" | grep -q "${PACK_NAME}/install-desktop-entry.sh"' in workflow_text
    assert '7z l "out/DecayCore_${VERSION}_linux_arm64.7z" | grep -q "${PACK_NAME}/DecayCore.png"' in workflow_text
    assert '7z l "out/DecayCore_${VERSION}_linux_arm64.7z" | grep -q "${PACK_NAME}/DecayCore_${VERSION}"' in workflow_text
    assert '(cd dist && 7z a -t7z -mx=9 -mmt=on -m0=lzma2 "../out/DecayCore_${VERSION}_linux_x86_64.7z" "${PACK_NAME}")' in workflow_text
    assert '7z l "out/DecayCore_${VERSION}_linux_x86_64.7z" | grep -q "${PACK_NAME}/install-desktop-entry.sh"' in workflow_text
    assert '7z l "out/DecayCore_${VERSION}_linux_x86_64.7z" | grep -q "${PACK_NAME}/DecayCore.png"' in workflow_text
    assert '7z l "out/DecayCore_${VERSION}_linux_x86_64.7z" | grep -q "${PACK_NAME}/DecayCore_${VERSION}"' in workflow_text
    assert "Download Linux ARM64 build artifact" in workflow_text
    assert "name: DecayCore-linux-arm64" in workflow_text
    assert "path: out/*.7z" in workflow_text
    assert "needs: [build_macos_arm64, build_linux_arm64, build_windows_x86_64, build_linux_x86_64]" in workflow_text


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
    assert workflow_text.count("python -m pip install ./decaycore-scoring") == 3


def test_release_workflow_packages_windows_release_like_local_script():
    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release-build.yml"
    workflow_text = workflow_path.read_text(encoding="utf-8")

    assert "build_windows_x86_64:" in workflow_text
    assert "name: Windows x86_64 build" in workflow_text
    assert '$manual = Get-ChildItem -Path "dist\\DecayCore-$env:VERSION" -Recurse -Filter "User_Manual.md" -File -ErrorAction SilentlyContinue |' in workflow_text
    assert 'Where-Object { $_.Directory.Name -eq "docs" } |' in workflow_text
    assert 'Select-Object -First 1' in workflow_text
    assert '-RedirectStandardOutput "smoke_windows.out.log" `' in workflow_text
    assert '-RedirectStandardError "smoke_windows.err.log"' in workflow_text
    assert 'for ($i = 0; $i -lt 45; $i++) {' in workflow_text
    assert 'Write-Output "Startup smoke test FAILED."' in workflow_text
    assert 'Write-Output "Startup smoke test passed."' in workflow_text
    assert 'New-Item -ItemType Directory -Force -Path out | Out-Null' in workflow_text
    assert 'Rename-Item $SourceDir $PackName' in workflow_text
    assert '@"' in workflow_text
    assert 'DecayCore Windows portable build' in workflow_text
    assert 'Set-Content "$PackDir\\README.txt"' in workflow_text
    assert '7z a -t7z -mx=9 -mmt=on -m0=lzma2 "..\\out\\DecayCore_$($env:VERSION)_windows_x86_64.7z" $PackName' in workflow_text
    assert 'Select-String -Pattern "DecayCore_$($env:VERSION)\\\\README.txt" | Out-Null' in workflow_text
    assert 'Select-String -Pattern "DecayCore_$($env:VERSION)\\\\DecayCore_$($env:VERSION).exe" | Out-Null' in workflow_text


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
