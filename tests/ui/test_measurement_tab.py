from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from decaycore.ui import measurement_tab


def test_device_option_label_prefers_active_measurement_samplerate() -> None:
    device = SimpleNamespace(
        index=3,
        name="USB Audio",
        max_input_channels=2,
        max_output_channels=2,
        default_samplerates=[44100],
    )

    label = measurement_tab._device_option_label(device, samplerate_hz=48000)

    assert label == "3: USB Audio (In 2, Out 2, 48000 Hz)"


def test_device_option_label_includes_hostapi_when_available() -> None:
    device = SimpleNamespace(
        index=4,
        name="HDMI AVR",
        max_input_channels=0,
        max_output_channels=6,
        default_samplerates=[48000],
        hostapi_name="Windows WASAPI",
    )

    label = measurement_tab._device_option_label(device, samplerate_hz=48000)

    assert label == "4: HDMI AVR (Out 6, Windows WASAPI, 48000 Hz)"


def test_filter_measurement_devices_for_wasapi_returns_only_wasapi_devices() -> None:
    devices = [
        SimpleNamespace(index=1, hostapi_name="Windows WASAPI"),
        SimpleNamespace(index=2, hostapi_name="MME"),
        SimpleNamespace(index=3, hostapi_name="DirectSound"),
    ]

    filtered = measurement_tab._filter_measurement_devices_for_wasapi(devices, enabled=True)

    assert [device.index for device in filtered] == [1]


def test_measurement_audio_backend_message_returns_backend_reason(monkeypatch) -> None:
    monkeypatch.setattr(
        measurement_tab,
        "check_measurement_audio_backend",
        lambda: (False, "Audio backend loaded, but device query failed: PortAudio host API not initialized"),
    )

    message = measurement_tab._measurement_audio_backend_message()

    assert message == "Audio backend loaded, but device query failed: PortAudio host API not initialized"


def test_preview_magnitude_for_plot_uses_camillafir_reference_smoothing(monkeypatch) -> None:
    calls: list[tuple[np.ndarray, np.ndarray, str]] = []

    def _fake_view_mags_for_plot(freqs, mags, *, plot_smoothing_level):
        freq_arr = np.asarray(freqs, dtype=float)
        mag_arr = np.asarray(mags, dtype=float)
        calls.append((freq_arr, mag_arr, str(plot_smoothing_level)))
        return mag_arr + 1.5

    monkeypatch.setattr(measurement_tab, "_view_mags_for_plot", _fake_view_mags_for_plot)

    freq = np.asarray([20.0, 100.0, 1000.0], dtype=float)
    mag = np.asarray([-6.0, -3.0, 0.0], dtype=float)

    out = measurement_tab._preview_magnitude_for_plot(freq, mag)

    assert len(calls) == 1
    np.testing.assert_allclose(calls[0][0], freq)
    np.testing.assert_allclose(calls[0][1], mag)
    assert calls[0][2] == "Psychoacoustic"
    np.testing.assert_allclose(out, mag + 1.5)


def test_build_preview_figure_uses_smoothed_magnitude_trace_labels(monkeypatch) -> None:
    def _fake_preview(freq_hz, magnitude_db):
        _ = freq_hz
        return np.asarray(magnitude_db, dtype=float) + 2.0

    monkeypatch.setattr(measurement_tab, "_preview_magnitude_for_plot", _fake_preview)

    bundle = SimpleNamespace(
        capture=SimpleNamespace(recorded_signal=np.linspace(-1.0, 1.0, 16, dtype=float)),
        ir=SimpleNamespace(impulse_response=np.linspace(1.0, -1.0, 16, dtype=float)),
        analysis_freq_hz=np.asarray([20.0, 100.0, 1000.0], dtype=float),
        analysis_magnitude_db=np.asarray([-6.0, -3.0, 0.0], dtype=float),
        calibrated_analysis_magnitude_db=np.asarray([-5.0, -2.0, 1.0], dtype=float),
        harmonic_freq_hz=np.asarray([20.0, 100.0, 1000.0], dtype=float),
        harmonic_magnitudes_db={
            2: np.asarray([-30.0, -24.0, -18.0], dtype=float),
            3: np.asarray([-40.0, -34.0, -28.0], dtype=float),
        },
    )

    fig = measurement_tab._build_preview_figure(bundle)

    trace_by_name = {str(trace.name): trace for trace in fig.data}
    assert "Magnitude (DecayCore Reference)" in trace_by_name
    assert "Calibrated magnitude (DecayCore Reference)" in trace_by_name
    assert "H2 (DecayCore Reference)" in trace_by_name
    assert "H3 (DecayCore Reference)" in trace_by_name
    np.testing.assert_allclose(
        np.asarray(trace_by_name["Magnitude (DecayCore Reference)"].y, dtype=float),
        bundle.analysis_magnitude_db + 2.0,
    )
    np.testing.assert_allclose(
        np.asarray(trace_by_name["Calibrated magnitude (DecayCore Reference)"].y, dtype=float),
        bundle.calibrated_analysis_magnitude_db + 2.0,
    )
    np.testing.assert_allclose(
        np.asarray(trace_by_name["H2 (DecayCore Reference)"].y, dtype=float),
        bundle.harmonic_magnitudes_db[2] + 2.0,
    )
    np.testing.assert_allclose(
        np.asarray(trace_by_name["H3 (DecayCore Reference)"].y, dtype=float),
        bundle.harmonic_magnitudes_db[3] + 2.0,
    )


def test_session_preview_bundles_collects_all_available_final_channels() -> None:
    left_bundle = object()
    right_bundle = object()
    sub2_bundle = object()
    session = SimpleNamespace(
        final_left_bundle=left_bundle,
        final_right_bundle=right_bundle,
        final_sub1_bundle=None,
        final_sub2_bundle=sub2_bundle,
    )

    bundles = measurement_tab._session_preview_bundles(session)

    assert bundles == {
        "left": left_bundle,
        "right": right_bundle,
        "sub2": sub2_bundle,
    }


def test_session_preview_default_channel_key_prefers_requested_channel() -> None:
    bundles = {
        "left": object(),
        "right": object(),
        "sub1": object(),
    }

    selected = measurement_tab._session_preview_default_channel_key(
        bundles,
        preferred_channel_key="sub1",
    )

    assert selected == "sub1"


def test_session_preview_bundle_falls_back_to_first_available_channel() -> None:
    right_bundle = object()
    sub1_bundle = object()
    session = SimpleNamespace(
        final_left_bundle=None,
        final_right_bundle=right_bundle,
        final_sub1_bundle=sub1_bundle,
        final_sub2_bundle=None,
    )

    preview_bundle = measurement_tab._session_preview_bundle(
        session,
        preferred_channel_key="left",
    )

    assert preview_bundle is right_bundle
