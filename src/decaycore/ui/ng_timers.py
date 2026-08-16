# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Timer helper that survives orphaned page loads.

A NiceGUI element timer first blocks in ``client.connected()``. When the
client is deleted before its websocket ever arrives — a prefetched page, a
tab closed during load, a reload that orphans the previous page — that wait
ends anyway and the timer immediately dereferences its (already collected)
parent slot::

    RuntimeError: The parent slot of the element has been deleted.

NiceGUI prunes such clients 60 s after the page load, so every timer built
during the page build logs one traceback in a burst.

`page_timer` avoids the whole race: while the client has no socket the timer
is not created at all, only scheduled from the client's connect handler. An
orphaned client never runs that handler, so it never owns a timer.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("DecayCore")


def page_timer(interval: float, callback, *, once: bool = False, immediate: bool = True):
    """Create a ``ui.timer`` once the browser connection exists.

    Returns the timer when it could be created right away, otherwise ``None``
    (creation then happens in the client's connect handler).
    """
    from nicegui import ui  # noqa: PLC0415

    try:
        client = ui.context.client
    except (AttributeError, RuntimeError):
        logger.debug("page_timer: no client context; creating timer directly", exc_info=True)
        return ui.timer(interval, callback, once=once, immediate=immediate)

    if bool(getattr(client, "has_socket_connection", False)):
        return ui.timer(interval, callback, once=once, immediate=immediate)

    installed = {"done": False}

    def _install() -> None:
        # handle_handshake also fires on reconnect; the timer must stay unique.
        if installed["done"]:
            return
        installed["done"] = True
        ui.timer(interval, callback, once=once, immediate=immediate)

    client.on_connect(_install)
    return None


__all__ = ["page_timer"]
