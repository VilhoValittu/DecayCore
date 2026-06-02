from __future__ import annotations

import time

from decaycore.measurement.models import MeasurementRequest, MeasurementSessionAggregate
from decaycore.ui.measurement_session_runner import MeasurementSessionRunner


def _request(role: str) -> MeasurementRequest:
    return MeasurementRequest(
        input_device_index=0,
        output_device_index=1,
        samplerate=48000,
        role=role,
    )


def _drain_until(runner: MeasurementSessionRunner, predicate, timeout_s: float = 2.0) -> list[dict]:
    deadline = time.time() + float(timeout_s)
    events: list[dict] = []
    while time.time() < deadline:
        batch = runner.poll_events()
        if batch:
            events.extend(batch)
            if predicate(events):
                return events
        time.sleep(0.01)
    return events


def test_measurement_session_runner_blocks_and_resumes_on_move_mic() -> None:
    def _fake_run_session(**kwargs) -> MeasurementSessionAggregate:
        progress_callback = kwargs["progress_callback"]
        progress_callback(
            {
                "stage": "capture_take",
                "position_index": 1,
                "position_count": 2,
                "role": "left",
                "take_index": 1,
                "repeats_per_channel": 2,
            }
        )
        progress_callback(
            {
                "stage": "move_mic",
                "position_index": 1,
                "next_position_index": 2,
                "position_count": 2,
            }
        )
        progress_callback(
            {
                "stage": "capture_take",
                "position_index": 2,
                "position_count": 2,
                "role": "left",
                "take_index": 1,
                "repeats_per_channel": 2,
            }
        )
        return MeasurementSessionAggregate(
            session_id="session_test",
            position_count=2,
            repeats_per_channel=2,
            primary_position_index=1,
            summary={"left": {"kept": 4, "total": 4, "rejected": []}},
        )

    runner = MeasurementSessionRunner(run_session_fn=_fake_run_session)
    runner.start(
        left_request=_request("left"),
        right_request=None,
        position_count=2,
        repeats_per_channel=2,
        primary_position_index=1,
        outlier_rejection_enabled=True,
        outlier_strictness="normal",
    )

    events = _drain_until(runner, lambda items: any(event.get("stage") == "move_mic" for event in items))
    assert any(event.get("stage") == "capture_take" for event in events)
    assert any(event.get("stage") == "move_mic" for event in events)
    assert runner.is_running() is True
    assert runner.get_result() is None

    runner.continue_after_move_mic()
    events = _drain_until(runner, lambda items: any(event.get("stage") == "session_complete" for event in items))

    assert any(event.get("stage") == "session_complete" for event in events)
    assert runner.get_result() is not None
    assert runner.get_result().session_id == "session_test"


def test_measurement_session_runner_cancel_reports_error() -> None:
    def _fake_run_session(**kwargs) -> MeasurementSessionAggregate:
        progress_callback = kwargs["progress_callback"]
        progress_callback(
            {
                "stage": "move_mic",
                "position_index": 1,
                "next_position_index": 2,
                "position_count": 2,
            }
        )
        return MeasurementSessionAggregate(
            session_id="session_test",
            position_count=2,
            repeats_per_channel=2,
            primary_position_index=1,
        )

    runner = MeasurementSessionRunner(run_session_fn=_fake_run_session)
    runner.start(
        left_request=_request("left"),
        right_request=None,
        position_count=2,
        repeats_per_channel=2,
        primary_position_index=1,
        outlier_rejection_enabled=True,
        outlier_strictness="normal",
    )

    _drain_until(runner, lambda items: any(event.get("stage") == "move_mic" for event in items))
    runner.cancel()
    events = _drain_until(runner, lambda items: any(event.get("stage") == "session_error" for event in items))

    assert any(event.get("stage") == "session_error" for event in events)
    assert "cancelled" in runner.get_error_text().lower()


def test_measurement_session_runner_blocks_and_resumes_on_sub_switch() -> None:
    def _fake_run_session(**kwargs) -> MeasurementSessionAggregate:
        progress_callback = kwargs["progress_callback"]
        progress_callback(
            {
                "stage": "prepare_subwoofer",
                "position_index": 1,
                "position_count": 1,
                "next_role": "sub1",
                "role": "sub1",
            }
        )
        progress_callback(
            {
                "stage": "capture_take",
                "position_index": 1,
                "position_count": 1,
                "role": "sub1",
                "take_index": 1,
                "repeats_per_channel": 1,
            }
        )
        progress_callback(
            {
                "stage": "switch_subwoofer",
                "position_index": 1,
                "position_count": 1,
                "previous_role": "sub1",
                "next_role": "sub2",
                "role": "sub2",
            }
        )
        progress_callback(
            {
                "stage": "capture_take",
                "position_index": 1,
                "position_count": 1,
                "role": "sub2",
                "take_index": 1,
                "repeats_per_channel": 1,
            }
        )
        return MeasurementSessionAggregate(
            session_id="session_test",
            position_count=1,
            repeats_per_channel=1,
            primary_position_index=1,
            summary={"sub1": {"kept": 1, "total": 1, "rejected": []}, "sub2": {"kept": 1, "total": 1, "rejected": []}},
        )

    runner = MeasurementSessionRunner(run_session_fn=_fake_run_session)
    runner.start(
        left_request=None,
        right_request=None,
        sub1_request=_request("sub"),
        sub2_request=_request("sub"),
        position_count=1,
        repeats_per_channel=1,
        primary_position_index=1,
        outlier_rejection_enabled=True,
        outlier_strictness="normal",
    )

    events = _drain_until(runner, lambda items: any(event.get("stage") == "prepare_subwoofer" for event in items))
    assert any(event.get("stage") == "prepare_subwoofer" for event in events)
    assert not any(event.get("stage") == "capture_take" and event.get("role") == "sub1" for event in events)
    assert runner.is_running() is True
    assert runner.get_result() is None

    runner.continue_after_pause()
    events = _drain_until(runner, lambda items: any(event.get("stage") == "switch_subwoofer" for event in items))
    assert any(event.get("stage") == "capture_take" and event.get("role") == "sub1" for event in events)
    assert any(event.get("stage") == "switch_subwoofer" for event in events)
    assert runner.is_running() is True
    assert runner.get_result() is None

    runner.continue_after_pause()
    events = _drain_until(runner, lambda items: any(event.get("stage") == "session_complete" for event in items))

    assert any(event.get("stage") == "capture_take" and event.get("role") == "sub2" for event in events)
    assert any(event.get("stage") == "session_complete" for event in events)
    assert runner.get_result() is not None
