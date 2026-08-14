from .measurement_source_helpers import (
    _get_local_path,
    _get_uploaded_file,
    _get_wav_window_params,
)

# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

import json
import logging
import os

import numpy as np

logger = logging.getLogger("DecayCore")

from ...common.measurement_features import normalize_rt60_bands, normalize_rt60_value
from ..measurement_bundle import TransferData
from ..measurements_txt import parse_measurements_from_path as parse_txt_path
from ..measurements_txt import parse_measurements_from_bytes as parse_txt_bytes
from ..measurements_wav import (
    parse_measurements_from_wav_bytes,
    parse_measurements_from_wav_path,
)


def _silent_transfer_like(template: TransferData, *, label: str) -> TransferData:
    freqs = np.asarray(template.freqs_hz, dtype=float)
    complex_spec = np.zeros(freqs.shape, dtype=np.complex128)
    return TransferData(
        freqs_hz=freqs,
        complex_spec=complex_spec,
        mag_db=np.full(freqs.shape, -240.0, dtype=float),
        phase_deg=np.zeros(freqs.shape, dtype=float),
        sample_rate=int(template.sample_rate),
        label=str(label or ""),
    )


def parse_measurements_from_upload(
    file_dict,
    *,
    channel_index: int = 0,
    pre_ms: float = 5.0,
    post_ms: float = 500.0,
    smoothing_level: int | None = None,
    logger=None,
):
    """Jasentaa selaimesta ladatun mittaustiedoston sisallon.

    Valitsee parserin tiedostopaateen tai RIFF-headerin perusteella:
    WAV -> WAV-parseri, muuten TXT-parseri.
    """
    try:
        if not file_dict:
            return None, None, None
        name = str(file_dict.get("filename", "") or "")
        content = file_dict.get("content", None)
        if content is None:
            return None, None, None
        ext = os.path.splitext(name)[1].lower()
        if ext == ".wav":
            return parse_measurements_from_wav_bytes(
                content,
                channel_index=channel_index,
                pre_ms=pre_ms,
                post_ms=post_ms,
                smoothing_level=smoothing_level,
                logger=logger,
            )
        if isinstance(content, (bytes, bytearray)) and len(content) >= 4 and content[:4] == b"RIFF":
            return parse_measurements_from_wav_bytes(
                content,
                channel_index=channel_index,
                pre_ms=pre_ms,
                post_ms=post_ms,
                smoothing_level=smoothing_level,
                logger=logger,
            )
        return parse_txt_bytes(content)
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
    ):
        return None, None, None


def _try_load_harmonic_sidecar(
    wav_path: str,
) -> tuple[np.ndarray | None, dict[int, np.ndarray] | None]:
    """Yrittää ladata harmonidata-sidecar-NPZ WAV-tiedoston vierestä.

    Sidecar-nimi: ``<stem>_harmonics.npz`` (väliviiva tai kaksoispiste riippuen
    tallennuslogiikasta: ``_save_bundle_files`` käyttää ``__harmonics.npz``).
    Ladataan molemmat variantit.

    Palauttaa ``(freq_hz, {order: mag_db_array, ...})`` tai ``(None, None)``
    jos sidecar puuttuu tai luku epäonnistuu.
    """
    if not wav_path:
        return None, None
    try:
        candidates = _measurement_sidecar_candidates(
            wav_path,
            ("__harmonics.npz", "_harmonics.npz"),
        )
        npz_path = next((p for p in candidates if os.path.isfile(p)), None)
        if npz_path is None:
            return None, None
        data_npz = np.load(npz_path, allow_pickle=False)
        freq_hz = np.asarray(data_npz["freq_hz"], dtype=np.float64)
        mags: dict[int, np.ndarray] = {}
        for key in data_npz.files:
            if key.startswith("order_") and key.endswith("_db"):
                try:
                    order = int(key.split("_")[1])
                    mags[order] = np.asarray(data_npz[key], dtype=np.float64)
                except (ValueError, IndexError):
                    pass
        if freq_hz.size == 0 or not mags:
            return None, None
        return freq_hz, mags
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
    ):
        return None, None


def _measurement_sidecar_stems(path: str) -> tuple[str, ...]:
    base = os.path.splitext(path)[0]
    stems = [base]
    if base.endswith("__ir"):
        stems.append(base[:-4])
    elif base.endswith("_ir"):
        stems.append(base[:-3])
    return tuple(dict.fromkeys(stems))


def _measurement_sidecar_candidates(path: str, suffixes: tuple[str, ...]) -> tuple[str, ...]:
    stems = _measurement_sidecar_stems(path)
    return tuple(f"{stem}{suffix}" for stem in stems for suffix in suffixes)


def _try_load_measurement_metadata_sidecar(wav_path: str) -> dict[str, object] | None:
    if not wav_path:
        return None
    try:
        candidates = _measurement_sidecar_candidates(
            wav_path,
            ("__metadata.json", "_metadata.json"),
        )
        metadata_path = next((p for p in candidates if os.path.isfile(p)), None)
        if metadata_path is None:
            return None
        with open(metadata_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else None
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
    ):
        return None


def _try_load_rt60_sidecar(wav_path: str) -> tuple[float | None, dict[float, float] | None]:
    metadata = _try_load_measurement_metadata_sidecar(wav_path)
    if not isinstance(metadata, dict):
        return None, None
    rt60_val = normalize_rt60_value(metadata.get("rt60_val", None))
    rt60_bands = normalize_rt60_bands(metadata.get("rt60_bands", None))
    return rt60_val, rt60_bands


def load_measurements_lr(data: dict, *, logger=None):
    """Lataa vasemman ja oikean kanavan mittaukset ensisijaisuusjarjestyksessa.

    Jarjestys:
    1) selainlataukset (`data["file_l"]`, `data["file_r"]`)
    2) paikalliset polut (`local_path_l`, `local_path_r`)

    Palauttaa aina 6-arvoisen tuplen:
    `(f_l, m_l, p_l, f_r, m_r, p_r)`.
    """
    pre_ms, post_ms, sl = _get_wav_window_params(data)

    up_l = _get_uploaded_file(data, "file_l")
    up_r = _get_uploaded_file(data, "file_r")

    if up_l is not None and up_r is not None:
        f_l, m_l, p_l = parse_measurements_from_upload(
            up_l, pre_ms=pre_ms, post_ms=post_ms, smoothing_level=sl, logger=logger
        )
        f_r, m_r, p_r = parse_measurements_from_upload(
            up_r, pre_ms=pre_ms, post_ms=post_ms, smoothing_level=sl, logger=logger
        )
        if f_l is not None and f_r is not None:
            return f_l, m_l, p_l, f_r, m_r, p_r

    lp_l = _get_local_path(data, "local_path_l")
    lp_r = _get_local_path(data, "local_path_r")

    if lp_l and lp_r:
        ext_l = os.path.splitext(lp_l)[1].lower()
        ext_r = os.path.splitext(lp_r)[1].lower()

        if ext_l == ".wav" and ext_r == ".wav":
            f_l, m_l, p_l = parse_measurements_from_wav_path(
                lp_l, pre_ms=pre_ms, post_ms=post_ms, smoothing_level=sl, logger=logger
            )
            f_r, m_r, p_r = parse_measurements_from_wav_path(
                lp_r, pre_ms=pre_ms, post_ms=post_ms, smoothing_level=sl, logger=logger
            )
            return f_l, m_l, p_l, f_r, m_r, p_r

        f_l, m_l, p_l = parse_txt_path(lp_l, logger=logger)
        f_r, m_r, p_r = parse_txt_path(lp_r, logger=logger)
        return f_l, m_l, p_l, f_r, m_r, p_r

    return None, None, None, None, None, None


__all__ = [
    "_silent_transfer_like",
    "parse_measurements_from_upload",
    "_try_load_harmonic_sidecar",
    "_measurement_sidecar_stems",
    "_measurement_sidecar_candidates",
    "_try_load_measurement_metadata_sidecar",
    "_try_load_rt60_sidecar",
    "load_measurements_lr",
]
