import json
import os
import pathlib
import subprocess
import sys

from decaycore.core.runner import prepare_headless_config


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _env():
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC) if not existing else os.pathsep.join([str(SRC), existing])
    return env


def test_headless_cli_version():
    proc = subprocess.run(
        [sys.executable, "-m", "decaycore.headless", "--version"],
        cwd=str(ROOT),
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert proc.stdout.strip().startswith("v.")



def test_headless_import_does_not_import_gui_modules():
    script = """
import builtins
import importlib
import sys

blocked = ("nicegui", "pywebio")
orig_import = builtins.__import__

def guard(name, globals=None, locals=None, fromlist=(), level=0):
    if name in blocked or any(name.startswith(prefix + ".") for prefix in blocked):
        raise ModuleNotFoundError(name)
    return orig_import(name, globals, locals, fromlist, level)

builtins.__import__ = guard
try:
    importlib.import_module("decaycore.headless")
finally:
    builtins.__import__ = orig_import

assert "nicegui" not in sys.modules
assert "pywebio" not in sys.modules
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ROOT),
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr


def test_relative_input_paths_resolve(tmp_path):
    cfg = tmp_path / "config.json"
    (tmp_path / "L.wav").write_bytes(b"")
    (tmp_path / "R.wav").write_bytes(b"")
    cfg.write_text(json.dumps({"mode": "auto", "left_wav": "L.wav", "right_wav": "R.wav"}), encoding="utf-8")

    data = prepare_headless_config(cfg, tmp_path / "out")

    assert pathlib.Path(data["local_path_l"]) == (tmp_path / "L.wav").resolve()
    assert pathlib.Path(data["local_path_r"]) == (tmp_path / "R.wav").resolve()


def test_partial_run_writes_metrics_and_summary(tmp_path):
    cfg = tmp_path / "config.json"
    out = tmp_path / "out"
    cfg.write_text(
        json.dumps(
            {
                "mode": "auto",
                "left_wav": "missing_L.wav",
                "right_wav": "missing_R.wav",
                "sample_rate": 48000,
                "house_curve": "Harman10",
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "decaycore.headless",
            "--config",
            str(cfg),
            "--output",
            str(out),
            "--no-plots",
        ],
        cwd=str(ROOT),
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 2
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "partial"
    assert metrics["selected_house_curve"] == "Harman10"
    assert (out / "summary.txt").exists()
    assert (out / "run.log").exists()


def test_failure_still_writes_failed_metrics(tmp_path):
    cfg = tmp_path / "config.json"
    out = tmp_path / "out"
    cfg.write_text("{not valid", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "decaycore.headless", "--config", str(cfg), "--output", str(out)],
        cwd=str(ROOT),
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "failed"
    assert metrics["errors"]
    assert (out / "summary.txt").exists()
