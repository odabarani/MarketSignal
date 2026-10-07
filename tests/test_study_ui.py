import json

from dash import Dash

from src import study_ui
from src.jobs import execute_research
from src.snapshots import SnapshotStore
from src.studies import StudyStore
from test_app import make_market_data


class Jobs:
    def __init__(self):
        self.tasks = []
        self.value = {"state": "expired", "result": None}

    def submit(self, key, task):
        self.tasks.append(task)
        return "job-1"

    def status(self, job_id):
        return self.value


def harness(tmp_path):
    market = SnapshotStore(tmp_path / "studies.sqlite")
    raw = market.get("AAPL", lambda _: make_market_data())
    study = StudyStore(market.path).prepare(raw.attrs["snapshot"]["id"])
    app = Dash(__name__)
    app.layout = study_ui.layout()
    jobs = Jobs()
    study_ui.register(app, lambda: market, jobs)
    return app, jobs, market, study


def post(app, prefix, inputs, states, changed):
    key = next(key for key in app.callback_map if key.startswith(prefix))
    outputs = app.callback_map[key]["output"]
    if isinstance(outputs, list):
        outputs = [{"id": x.component_id, "property": x.component_property} for x in outputs]
    else:
        outputs = {"id": outputs.component_id, "property": outputs.component_property}
    response = app.server.test_client().post("/_dash-update-component", json={
        "output": key, "outputs": outputs,
        "inputs": [{"id": i, "property": p, "value": v} for i, p, v in inputs],
        "state": [{"id": i, "property": p, "value": v} for i, p, v in states],
        "changedPropIds": [changed],
    })
    assert response.status_code == 200, response.data
    return response.get_json()["response"]


def test_reload_reads_retained_protocol_and_gates_final(tmp_path):
    app, jobs, market, study = harness(tmp_path)
    result = post(app, "..study-context", [("study-poll", "n_intervals", 1), ("study-job", "data", "expired-job")],
                  [("study-context", "data", study["study_id"])], "study-poll.n_intervals")
    assert result["study-context"]["data"] == study["study_id"]
    assert result["study-validate"]["disabled"] is False
    assert result["study-finalize"]["disabled"] is True
    assert "Training through" in json.dumps(result["study-view"])
    assert "Retained final evaluation" not in json.dumps(result["study-view"])


def test_final_submission_cannot_skip_validation(tmp_path):
    app, jobs, market, study = harness(tmp_path)
    result = post(app, "..study-job", [("study-prepare", "n_clicks", 0), ("study-validate", "n_clicks", 0),
                                       ("study-finalize", "n_clicks", 1)],
                  [("ticker", "value", "MSFT"), ("study-context", "data", study["study_id"])], "study-finalize.n_clicks")
    assert not jobs.tasks
    assert "already consumed" in result["study-submit-status"]["children"]
    assert StudyStore(market.path).read(study["study_id"])["state"] == "prepared"


def test_validation_submission_only_sends_server_study_id(tmp_path):
    app, jobs, market, study = harness(tmp_path)
    post(app, "..study-job", [("study-prepare", "n_clicks", 0), ("study-validate", "n_clicks", 1),
                              ("study-finalize", "n_clicks", 0)],
         [("ticker", "value", "MSFT"), ("study-context", "data", study["study_id"])], "study-validate.n_clicks")
    assert jobs.tasks[0]["argument"] == study["study_id"]
    assert jobs.tasks[0]["stage"] == "validate"
    assert StudyStore(market.path).read(study["study_id"])["state"] == "prepared"


def test_download_retained_record(tmp_path):
    app, jobs, market, study = harness(tmp_path)
    result = post(app, "study-download", [("study-export", "n_clicks", 1)],
                  [("study-context", "data", study["study_id"])], "study-export.n_clicks")
    record = json.loads(result["study-download"]["data"]["content"])
    assert record == StudyStore(market.path).read(study["study_id"])


def test_queued_prepare_resumes_without_exploratory_fit(tmp_path, monkeypatch):
    import app as dashboard
    app, jobs, market, study = harness(tmp_path)
    monkeypatch.setattr(dashboard, "MARKET_STORE", market)
    monkeypatch.setattr(dashboard, "load_market", lambda ticker: (market.load(study["protocol"]["dataset"]["id"]), None))
    def unexpected(*args):
        raise AssertionError("Preparation must not fit exploratory models")
    monkeypatch.setattr(dashboard, "build_research_dashboard", unexpected)
    task = {"kind": "study", "stage": "prepare", "argument": "AAPL", "store_path": str(market.path)}
    assert execute_research(task) == {"study_id": study["study_id"]}
    assert StudyStore(market.path).read(study["study_id"])["state"] == "prepared"
