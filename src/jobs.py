"""Bounded single-worker jobs for the initial, single-web-process beta."""

import atexit
from collections import OrderedDict, deque
from dataclasses import dataclass
import logging
import multiprocessing as mp
from threading import RLock, Thread
import time
from uuid import uuid4
from src.errors import DataQualityError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class QueueFull(ValueError):
    pass


def worker_loop(connection, handler):
    try:
        while True:
            task = connection.recv()
            try:
                result = handler(task)
                connection.send((True, result))
            except Exception as error:
                logger.exception("job_execution_failed job_id=%s request_id=%s",
                                 task.get("job_id"), task.get("request_id"))
                message = str(error) if isinstance(error, DataQualityError) else "Research could not finish. Check the ticker and data access, then try again."
                connection.send((False, message))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


@dataclass
class Job:
    key: tuple
    task: dict
    created: float
    state: str = "queued"
    result: object = None
    finished: float = None


class JobManager:
    def __init__(self, handler, max_queued=4, timeout=180, result_ttl=900, max_results=16,
                 max_submissions=12, submission_window=60):
        self.handler = handler
        self.max_queued = max_queued
        self.timeout = timeout
        self.result_ttl = result_ttl
        self.max_results = max_results
        self.max_submissions, self.submission_window = max_submissions, submission_window
        self.submissions = deque()
        self.jobs = OrderedDict()
        self.lock = RLock()
        self.process = self.connection = self.active = self.started = None
        self.thread = None
        self.closed = False
        self.last_tick = time.monotonic()
        atexit.register(self.close)

    def submit(self, key, task):
        with self.lock:
            if self.closed:
                raise QueueFull("Research service is restarting. Please try again.")
            self._prune()
            for job_id, job in self.jobs.items():
                if job.key == key and job.state in ("queued", "running"):
                    return job_id
            pending = sum(job.state in ("queued", "running") for job in self.jobs.values())
            if pending >= self.max_queued + 1:
                raise QueueFull("Research is busy. Please try again shortly.")
            now = time.monotonic()
            while self.submissions and now - self.submissions[0] >= self.submission_window:
                self.submissions.popleft()
            if len(self.submissions) >= self.max_submissions:
                raise QueueFull("Research request limit reached. Please try again in a minute.")
            self.submissions.append(now)
            job_id = uuid4().hex
            task = dict(task, job_id=job_id)
            self.jobs[job_id] = Job(key, task, time.monotonic())
            logger.info("job_queued job_id=%s request_id=%s kind=%s", job_id,
                        task.get("request_id"), task.get("kind"))
            if self.thread is None:
                self.thread = Thread(target=self._loop, daemon=True)
                self.thread.start()
            return job_id

    def overview(self):
        with self.lock:
            self._prune()
            healthy = not self.closed and (
                self.thread is None or (self.thread.is_alive() and time.monotonic() - self.last_tick < 10)
            )
            return {"healthy": healthy,
                    "running": sum(job.state == "running" for job in self.jobs.values()),
                    "queued": sum(job.state == "queued" for job in self.jobs.values())}

    def status(self, job_id):
        with self.lock:
            self._prune()
            job = self.jobs.get(job_id) if isinstance(job_id, str) else None
            if job is None:
                return {"state": "expired", "result": "This job expired or the server restarted. Run research again."}
            return {"state": job.state, "result": job.result}

    def _prune(self):
        now = time.monotonic()
        completed = [key for key, job in self.jobs.items() if job.finished is not None]
        for key in completed:
            if now - self.jobs[key].finished >= self.result_ttl:
                del self.jobs[key]
        completed = [key for key, job in self.jobs.items() if job.finished is not None]
        for key in completed[:-self.max_results]:
            del self.jobs[key]

    def _stop_worker(self):
        if self.process:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=1)
            self.process.close()
        if self.connection:
            self.connection.close()
        self.process = self.connection = None

    def _finish(self, state, result):
        job = self.jobs[self.active]
        logger.info("job_finished job_id=%s state=%s duration_ms=%.1f", self.active,
                    state, (time.monotonic() - self.started) * 1000)
        job.state, job.result, job.finished = state, result, time.monotonic()
        # Drop request arguments after completion; retained results are bounded.
        job.task = None
        self.active = self.started = None

    def _tick(self):
        if self.active:
            if self.connection.poll():
                try:
                    ok, result = self.connection.recv()
                    self._finish("completed" if ok else "failed", result)
                except (EOFError, OSError):
                    self._finish("failed", "Research worker stopped. Please try again.")
                    self._stop_worker()
            elif not self.process.is_alive():
                self._finish("failed", "Research worker stopped. Please try again.")
                self._stop_worker()
            elif time.monotonic() - self.started >= self.timeout:
                self._stop_worker()
                self._finish("timed_out", "Research exceeded its time limit. Try again with a longer retrain interval.")
        if self.active is None:
            next_job = next((key for key, job in self.jobs.items() if job.state == "queued"), None)
            if next_job is not None:
                if self.process is None or not self.process.is_alive():
                    self._stop_worker()
                    parent, child = mp.get_context("spawn").Pipe()
                    self.connection = parent
                    self.process = mp.get_context("spawn").Process(
                        target=worker_loop, args=(child, self.handler), daemon=True,
                    )
                    self.process.start()
                    child.close()
                self.active, self.started = next_job, time.monotonic()
                self.jobs[next_job].state = "running"
                logger.info("job_started job_id=%s", next_job)
                self.connection.send(self.jobs[next_job].task)

    def _loop(self):
        while True:
            with self.lock:
                if self.closed:
                    return
                try:
                    self._tick()
                    self._prune()
                    self.last_tick = time.monotonic()
                except Exception:
                    logging.exception("Background scheduler failed")
                    if self.active:
                        self._finish("failed", "Research service encountered a problem. Please try again.")
                    self._stop_worker()
            time.sleep(.05)

    def close(self):
        with self.lock:
            self.closed = True
            self._stop_worker()


def execute_research(task):
    # Import in the worker, preserving its model caches between tasks.
    import json
    from plotly.utils import PlotlyJSONEncoder
    import app
    from src.snapshots import SnapshotStore
    from threadpoolctl import threadpool_limits

    if str(app.MARKET_STORE.path) != task["store_path"]:
        app.MARKET_STORE = SnapshotStore(task["store_path"])
        app.fit_snapshot.cache_clear()
        app.run_model_comparison.cache_clear()
    with threadpool_limits(limits=2):
        if task["kind"] == "research":
            result = app.build_research_dashboard(*task["args"])
        elif task["kind"] == "study":
            from src.studies import StudyStore
            store = StudyStore(task["store_path"])
            if task["stage"] == "prepare":
                raw, _ = app.load_market(task["argument"])
                study = store.prepare_or_resume(raw.attrs["snapshot"]["id"])
                result = {"study_id": study["study_id"]}
            elif task["stage"] in ("validate", "finalize"):
                getattr(store, task["stage"])(task["argument"])
                result = {"study_id": task["argument"]}
            else:
                raise ValueError("Unknown study stage.")
        elif task["kind"] == "comparison":
            result = app.comparison_table(app.run_model_comparison(*task["args"]))
        else:
            raise ValueError("Unknown research task.")
    return json.loads(json.dumps(result, cls=PlotlyJSONEncoder))
