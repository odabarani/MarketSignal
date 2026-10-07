import logging
import sqlite3

from flask import Flask

from src.jobs import JobManager
from src.observability import register_observability
from src.snapshots import SnapshotStore


def make_server(tmp_path):
    server = Flask(__name__)
    store = SnapshotStore(tmp_path / "snapshots.sqlite")
    jobs = JobManager(None)
    register_observability(server, lambda: store, lambda: jobs)
    return server, store, jobs


def test_health_and_readiness_do_not_start_worker_or_provider(tmp_path):
    server, store, jobs = make_server(tmp_path)
    try:
        client = server.test_client()
        live = client.get("/health")
        ready = client.get("/ready")
        assert live.status_code == ready.status_code == 200
        assert live.get_json() == {"status": "alive"}
        assert ready.get_json()["storage"] == "writable"
        assert ready.get_json()["scheduler"] == {"healthy": True, "queued": 0, "running": 0}
        assert live.headers["Cache-Control"] == "no-store"
        assert len(ready.headers["X-Request-ID"]) == 32
        assert jobs.process is None and jobs.thread is None
        with sqlite3.connect(store.path) as connection:
            assert connection.execute("SELECT COUNT(*) FROM readiness_probe").fetchone()[0] == 0
            assert connection.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0] == 0
    finally:
        jobs.close()


def test_storage_failure_is_503_without_internal_details(tmp_path, monkeypatch):
    server, store, jobs = make_server(tmp_path)
    def fail():
        raise OSError("private internal path")
    monkeypatch.setattr(store, "check_writable", fail)
    try:
        response = server.test_client().get("/ready")
        assert response.status_code == 503
        assert b"private internal path" not in response.data
        assert response.get_json()["request_id"] == response.headers["X-Request-ID"]
    finally:
        jobs.close()


def test_closed_scheduler_is_not_ready_but_process_is_alive(tmp_path):
    server, store, jobs = make_server(tmp_path)
    jobs.close()
    client = server.test_client()
    assert client.get("/ready").status_code == 503
    assert client.get("/health").status_code == 200


def test_sqlite_write_lock_is_not_ready(tmp_path):
    server, store, jobs = make_server(tmp_path)
    store.check_writable()
    connection = sqlite3.connect(store.path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        response = server.test_client().get("/ready")
        assert response.status_code == 503
        connection.rollback()
        assert server.test_client().get("/ready").status_code == 200
    finally:
        connection.close()
        jobs.close()


def test_unexpected_error_and_logs_have_safe_request_ids(tmp_path, caplog):
    server, store, jobs = make_server(tmp_path)
    @server.get("/broken")
    def broken():
        raise RuntimeError("internal failure detail")
    try:
        with caplog.at_level(logging.INFO):
            response = server.test_client().get("/broken?token=private-query-value",
                                                headers={"X-Request-ID": "untrusted-input"})
        assert response.status_code == 500
        assert b"internal failure detail" not in response.data
        request_id = response.headers["X-Request-ID"]
        assert request_id != "untrusted-input"
        assert response.get_json()["request_id"] == request_id
        assert request_id in caplog.text
        assert "private-query-value" not in caplog.text
        assert "untrusted-input" not in caplog.text
        assert server.test_client().get("/missing").status_code == 404
    finally:
        jobs.close()
