# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

from .core.runner import ConsoleProgressSink, prepare_headless_config, run_batch
from .version import VERSION


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m decaycore.headless")
    parser.add_argument("--config", type=Path, help="Input JSON config path")
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--mode", default=None, choices=["auto"], help="Run mode")
    parser.add_argument("--language", default="en", help="Language code for logs/summaries")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    parser.add_argument("--no-plots", action="store_true", default=True, help="Disable plot/dashboard generation")
    parser.add_argument("--no-png", action="store_true", default=True, help="Accepted for worker compatibility")
    parser.add_argument("--write-summary", action="store_true", default=True, help="Write summary.txt")
    parser.add_argument("--write-metrics", action="store_true", default=True, help="Write metrics.json")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    return parser.parse_args(argv)


def _configure_logging(output_dir: Path, level: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level or "INFO").upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)


def _write_failed_minimal(output_dir: Path, *, config_path: Path | None, errors: list[str]) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "status": "failed",
        "version": str(VERSION),
        "git_commit": None,
        "started_at": "",
        "finished_at": "",
        "runtime_s": 0.0,
        "mode": "auto",
        "target_curve_mode": "",
        "selected_house_curve": "",
        "metrics": {},
        "bass_integration": {"enabled": False},
        "rt60": {},
        "harmonics": {},
        "warnings": [],
        "errors": list(errors),
    }
    (output_dir / "metrics.json").write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "summary.txt").write_text("Status: failed\nErrors: " + "; ".join(errors) + "\n", encoding="utf-8")
    logging.getLogger("DecayCore").error("Headless hard failure for %s: %s", config_path, "; ".join(errors))
    return doc


def run_headless(config_path: Path, output_dir: Path, args: argparse.Namespace) -> dict:
    output_dir = Path(output_dir).resolve()
    _configure_logging(output_dir, str(getattr(args, "log_level", "INFO") or "INFO"))
    logger = logging.getLogger("DecayCore")
    logger.info("Headless command args: %s", vars(args))
    logger.info("Config path: %s", Path(config_path).resolve())
    logger.info("Output path: %s", output_dir)

    try:
        data = prepare_headless_config(
            Path(config_path),
            output_dir,
            no_plots=bool(getattr(args, "no_plots", True) or getattr(args, "no_png", True)),
        )
        if getattr(args, "mode", None):
            data["mode"] = str(args.mode).upper()
            data["camillafir_automatic_mode"] = data["mode"] == "AUTO"
        data["language"] = str(getattr(args, "language", "en") or "en")
        data["_progress_sink"] = ConsoleProgressSink()
        logger.info("Selected mode: %s", data.get("mode"))
        logger.info("Resolved input L: %s", data.get("local_path_l"))
        logger.info("Resolved input R: %s", data.get("local_path_r"))
        if data.get("bass_integration_enable"):
            logger.info("Resolved BI main/sub: %s | %s | %s | %s", data.get("local_path_l_main"), data.get("local_path_r_main"), data.get("local_path_l_sub"), data.get("local_path_r_sub"))
        return run_batch(data, output_dir, headless=True)
    except (

        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        RuntimeError,
        OSError,
        ImportError,
        ModuleNotFoundError,
        NameError,
    ) as exc:
        logger.error("Headless hard failure: %s", traceback.format_exc())
        return _write_failed_minimal(output_dir, config_path=Path(config_path), errors=[f"{type(exc).__name__}: {exc}"])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if bool(args.version):
        print(VERSION)
        return 0
    if args.config is None or args.output is None:
        print("--config and --output are required unless --version is used", file=sys.stderr)
        return 1
    result = run_headless(Path(args.config), Path(args.output), args)
    status = str(result.get("status", "failed") or "failed")
    if status == "success":
        return 0
    if status == "partial":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
