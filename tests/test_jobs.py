import os
import time

import pytest

from src.jobs import JobManager, QueueFull
from src.errors import DataQualityError


def handler(task):
    if task.get("data_quality"):
        raise DataQualityError("Market history is stale. Wait for fresh provider data.")
    if task.get("crash"):
        os._exit(3)
    if task.get("fail"):
        raise RuntimeError("private error detail")
    time.sleep(task.get("delay", 0))
    return {"value": task.get("value"), "pid": os.getpid()}


def await_state(manager, job_id, states, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = manager.status(job_id)
        if status["state"] in states:
            return status
        time.sleep(.02)
    raise AssertionError(f"Job never reached {states}: {manager.status(job_id)}")


def test_bounded_queue_deduplicates_and_reuses_worker():
    manager = JobManager(handler, max_queued=1, timeout=5)
    try:
        first = manager.submit(("a",), {"value": 1, "delay": .3})
        assert manager.submit(("a",), {"value": 999}) == first
        second = manager.submit(("b",), {"value": 2})
        with pytest.raises(QueueFull, match="busy"):
            manager.submit(("c",), {})
        result1 = await_state(manager, first, {"completed"})["result"]
        result2 = await_state(manager, second, {"completed"})["result"]
        assert result1["value"] == 1 and result2["value"] == 2
        assert result1["pid"] == result2["pid"]  # persistent worker/model caches
        assert result1["pid"] != os.getpid()
    finally:
        manager.close()


def test_timeout_terminates_worker_and_next_job_recovers():
    manager = JobManager(handler, timeout=3)
    try:
        slow = manager.submit(("slow",), {"delay": 10})
        await_state(manager, slow, {"running"})
        old_process = manager.process
        fast = manager.submit(("fast",), {"value": 2})
        assert await_state(manager, slow, {"timed_out"})["result"]
        assert old_process._closed
        assert await_state(manager, fast, {"completed"})["result"]["value"] == 2
    finally:
        manager.close()


@pytest.mark.parametrize("failure", ["crash", "fail"])
def test_failure_and_worker_death_recover_without_leaking_errors(failure):
    manager = JobManager(handler, timeout=5)
    try:
        first = manager.submit(("bad",), {failure: True})
        failed = await_state(manager, first, {"failed"})
        assert "private error" not in failed["result"]
        second = manager.submit(("good",), {"value": 3})
        assert await_state(manager, second, {"completed"})["result"]["value"] == 3
    finally:
        manager.close()


def test_retained_results_are_bounded_and_expire():
    manager = JobManager(handler, max_results=1, result_ttl=.4)
    try:
        first = manager.submit(("a",), {})
        await_state(manager, first, {"completed"})
        second = manager.submit(("b",), {})
        await_state(manager, second, {"completed"})
        assert manager.status(first)["state"] == "expired"
        time.sleep(.45)
        assert manager.status(second)["state"] == "expired"
        assert manager.status("unknown")["state"] == "expired"
    finally:
        manager.close()


def test_close_rejects_new_work():
    manager = JobManager(handler)
    manager.close()
    with pytest.raises(QueueFull, match="restarting"):
        manager.submit(("a",), {})


def test_admission_rate_limit_does_not_charge_duplicates():
    manager = JobManager(handler, max_submissions=1)
    try:
        first = manager.submit(("a",), {"delay": .3})
        assert manager.submit(("a",), {}) == first
        with pytest.raises(QueueFull, match="request limit"):
            manager.submit(("b",), {})
    finally:
        manager.close()


def test_expected_data_quality_error_remains_actionable():
    manager = JobManager(handler, timeout=5)
    try:
        job_id = manager.submit(("stale",), {"data_quality": True})
        status = await_state(manager, job_id, {"failed"})
        assert status["result"] == "Market history is stale. Wait for fresh provider data."
    finally:
        manager.close()
