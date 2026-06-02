import json
import os

import numpy as np

from decaycore.ui.ng_tab_files import (
    _build_upload_payload,
    _build_measurement_library_options,
    _describe_local_path,
    _file_slot_input_name,
    _file_slot_scope_name,
    _format_upload_size,
    _guess_upload_format,
    _measurement_hint_tokens,
    _normalize_local_path_value,
    _normalize_layout_value,
    _scan_measurement_library,
    _suggest_measurement_library_matches,
)


def test_normalize_layout_value_accepts_legacy_and_stable_values():
    assert _normalize_layout_value("Mono") == "mono"
    assert _normalize_layout_value("mono") == "mono"
    assert _normalize_layout_value("Stereo") == "stereo"
    assert _normalize_layout_value("stereo") == "stereo"


def test_guess_upload_format_detects_txt_and_wav():
    assert _guess_upload_format({"filename": "left.txt", "content": b"20 0\n40 0\n"}) == "TXT"
    assert _guess_upload_format({"filename": "left.wav", "content": b"RIFF...."}) == "WAV"
    assert _guess_upload_format({"filename": "left.bin", "content": b"??"}) == "Unknown"


def test_normalize_local_path_value_strips_quotes():
    assert _normalize_local_path_value('"C:\\Temp\\left.txt"') == "C:\\Temp\\left.txt"


def test_describe_local_path_reports_existing_file(tmp_path):
    path = tmp_path / "left.txt"
    path.write_text("test", encoding="utf-8")

    info = _describe_local_path(str(path))

    assert info["entered"] is True
    assert info["exists"] is True
    assert info["filename"] == "left.txt"
    assert info["format"] == "TXT"
    assert info["size_bytes"] == 4
    assert info["has_harmonics"] is False
    assert info["rt60_val"] is None


def test_describe_local_path_reports_missing_file():
    info = _describe_local_path(r"C:\missing\left.wav")

    assert info["entered"] is True
    assert info["exists"] is False
    assert info["format"] == "WAV"


def test_describe_local_path_reports_saved_rt60_and_harmonics(tmp_path):
    wav_path = tmp_path / "left_final__ir.wav"
    wav_path.write_bytes(b"RIFF")
    (tmp_path / "left_final__metadata.json").write_text(
        json.dumps({"rt60_val": 0.47, "rt60_bands": {"125.0": 0.4, "250.0": 0.5}}),
        encoding="utf-8",
    )
    np.savez(
        str(tmp_path / "left_final__harmonics.npz"),
        freq_hz=np.asarray([100.0, 200.0], dtype=np.float32),
        order_2_db=np.asarray([-60.0, -62.0], dtype=np.float32),
    )

    info = _describe_local_path(str(wav_path))

    assert info["entered"] is True
    assert info["exists"] is True
    assert info["has_harmonics"] is True
    assert info["rt60_val"] == 0.47


def test_build_upload_payload_includes_digest_and_size():
    payload = _build_upload_payload(filename="left.txt", content=b"hello", mime_type="text/plain")

    assert payload["filename"] == "left.txt"
    assert payload["mime_type"] == "text/plain"
    assert payload["size_bytes"] == 5
    assert len(str(payload["content_sha256"])) == 64


def test_format_upload_size_uses_human_readable_units():
    assert _format_upload_size(1024) == "1.0 KB"
    assert _format_upload_size(2 * 1024 * 1024) == "2.00 MB"


def test_bass_integration_slot_names_are_unique_per_topology():
    assert _file_slot_scope_name("file_l_main", "bi") != _file_slot_scope_name("file_l_main", "direct")
    assert _file_slot_input_name("local_path_l_main", "bi") != _file_slot_input_name("local_path_l_main", "direct")


def test_measurement_hint_tokens_split_paths_and_names():
    tokens = _measurement_hint_tokens(r"session_01/final/left_final__ir.wav")

    assert tokens == ["session", "01", "final", "left", "final", "ir", "wav"]


def test_scan_measurement_library_lists_supported_files_recursively(tmp_path):
    newest = tmp_path / "FL0.wav"
    newest.write_bytes(b"RIFF")
    os.utime(newest, (200, 200))
    (tmp_path / "ignore.json").write_text("{}", encoding="utf-8")
    nested = tmp_path / "session_01" / "final"
    nested.mkdir(parents=True)
    older = nested / "right_final__ir.wav"
    older.write_bytes(b"RIFF")
    os.utime(older, (100, 100))

    entries = _scan_measurement_library(str(tmp_path))
    options = _build_measurement_library_options(entries)

    assert [entry["display_label"] for entry in entries] == [
        "FL0.wav [WAV]",
        "session_01/final/right_final__ir.wav [WAV]",
    ]
    assert len(options) == 2


def test_scan_measurement_library_prefers_newest_entries_first(tmp_path):
    older = tmp_path / "left_old.txt"
    older.write_text("20 0\n40 0\n", encoding="utf-8")
    os.utime(older, (100, 100))
    newer = tmp_path / "left_new.txt"
    newer.write_text("20 0\n40 0\n", encoding="utf-8")
    os.utime(newer, (200, 200))

    entries = _scan_measurement_library(str(tmp_path))

    assert [entry["display_label"] for entry in entries] == [
        "left_new.txt [TXT]",
        "left_old.txt [TXT]",
    ]


def test_suggest_measurement_library_matches_legacy_lr(tmp_path):
    (tmp_path / "pair_01_room_L.txt").write_text("20 0\n40 0\n", encoding="utf-8")
    (tmp_path / "pair_01_room_R.txt").write_text("20 0\n40 0\n", encoding="utf-8")

    entries = _scan_measurement_library(str(tmp_path))
    suggestions = _suggest_measurement_library_matches(
        entries,
        path_keys=["local_path_l", "local_path_r"],
    )

    assert suggestions["local_path_l"].endswith("pair_01_room_L.txt")
    assert suggestions["local_path_r"].endswith("pair_01_room_R.txt")


def test_suggest_measurement_library_matches_bass_integration_slots(tmp_path):
    for name in ("FL0.wav", "FR0.wav", "SW10.wav", "SW20.wav"):
        (tmp_path / name).write_bytes(b"RIFF")

    entries = _scan_measurement_library(str(tmp_path))
    suggestions = _suggest_measurement_library_matches(
        entries,
        path_keys=[
            "local_path_l_main",
            "local_path_r_main",
            "local_path_l_sub",
            "local_path_r_sub",
        ],
    )

    assert suggestions["local_path_l_main"].endswith("FL0.wav")
    assert suggestions["local_path_r_main"].endswith("FR0.wav")
    assert suggestions["local_path_l_sub"].endswith("SW10.wav")
    assert suggestions["local_path_r_sub"].endswith("SW20.wav")


def test_suggest_measurement_library_matches_prefers_newest_tied_candidate(tmp_path):
    older_dir = tmp_path / "session_old"
    older_dir.mkdir()
    older = older_dir / "pair_left.txt"
    older.write_text("20 0\n40 0\n", encoding="utf-8")
    os.utime(older, (100, 100))

    newer_dir = tmp_path / "session_new"
    newer_dir.mkdir()
    newer = newer_dir / "pair_left.txt"
    newer.write_text("20 0\n40 0\n", encoding="utf-8")
    os.utime(newer, (200, 200))

    entries = _scan_measurement_library(str(tmp_path))
    suggestions = _suggest_measurement_library_matches(
        entries,
        path_keys=["local_path_l"],
    )

    assert suggestions["local_path_l"].endswith(os.path.join("session_new", "pair_left.txt"))
