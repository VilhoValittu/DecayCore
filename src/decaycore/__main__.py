# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    _pkg_root = os.path.dirname(os.path.abspath(__file__))
    _src_root = os.path.dirname(_pkg_root)
    if _src_root in sys.path:
        sys.path.remove(_src_root)
    sys.path.insert(0, _src_root)

from nicegui import app, ui
from nicegui.client import Client

from decaycore.decaycore import PROGRAM_NAME, configure_main_app

FAVICON_PATH = Path(__file__).resolve().parent / "ui" / "assets" / "favicon.ico"


def _register_single_client_guard() -> None:
    _active: list[str | None] = [None]

    @app.on_connect
    async def _on_connect(client: Client) -> None:
        prev_id = _active[0]
        _active[0] = client.id
        if prev_id and prev_id != client.id:
            old = Client.instances.get(prev_id)
            if old:
                await old.run_javascript('window.location.replace("about:blank")')

    @app.on_disconnect
    async def _on_disconnect(client: Client) -> None:
        if _active[0] == client.id:
            _active[0] = None


def main():
    os.environ.setdefault("DECAYCORE_AUTO_PROFILE", os.environ.get("CAMILLAFIR_AUTO_PROFILE", "0"))
    configure_main_app()
    _register_single_client_guard()

    @app.on_startup
    async def _print_shutdown_hint() -> None:
        print("To shut down the DecayCore, press Ctrl+C or close the terminal.", flush=True)

    ui.run(
        port=8080,
        show=True,
        dark=True,
        favicon=FAVICON_PATH,
        reload=False,
        title=PROGRAM_NAME,
    )


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
