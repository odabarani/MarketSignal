"""Probe a deployment; --research additionally runs a real provider-backed job."""

import argparse
import json
import time
from urllib.request import Request, urlopen


def smoke(base_url, research=False, timeout=240):
    def request(path, payload=None):
        body = json.dumps(payload).encode() if payload is not None else None
        with urlopen(Request(base_url.rstrip("/") + path, data=body,
                             headers={"Content-Type": "application/json"}), timeout=15) as response:
            assert response.status == 200
            assert response.headers.get("X-Request-ID")
            return response.read()

    assert json.loads(request("/health"))["status"] == "alive"
    assert json.loads(request("/ready"))["status"] == "ready"
    assert b"MarketSignal Research" in request("/")
    assert b"Explore the evidence" in request("/_dash-layout")
    assert json.loads(request("/_dash-dependencies"))
    print("Health, writable storage, scheduler, page and Dash endpoints passed.")
    if not research:
        return

    submission = {
        "output": "..research-job.data...research-submit-status.children..",
        "outputs": [{"id": "research-job", "property": "data"},
                    {"id": "research-submit-status", "property": "children"}],
        "inputs": [{"id": "run-button", "property": "n_clicks", "value": 1}],
        "state": [{"id": name, "property": "value", "value": value}
                  for name, value in [("ticker", "AAPL"), ("threshold", .6), ("retrain-days", 21),
                                     ("cost-bps", 10), ("slippage-bps", 5)]],
        "changedPropIds": ["run-button.n_clicks"],
    }
    response = json.loads(request("/_dash-update-component", submission))["response"]
    assert "research-job" in response, response
    job_id = response["research-job"]["data"]
    polling = {
        "output": "..research-results.children...research-job-status.children...research-poll.disabled...run-button.disabled..",
        "outputs": [{"id": name, "property": prop} for name, prop in [
            ("research-results", "children"), ("research-job-status", "children"),
            ("research-poll", "disabled"), ("run-button", "disabled")]],
        "inputs": [{"id": "research-job", "property": "data", "value": job_id},
                   {"id": "research-poll", "property": "n_intervals", "value": 1}],
        "state": [], "changedPropIds": ["research-poll.n_intervals"],
    }
    deadline = time.monotonic() + timeout
    while True:
        result = json.loads(request("/_dash-update-component", polling))["response"]
        if result["research-poll"]["disabled"]:
            assert "research-results" in result, result
            context = result["research-results"]["children"]["props"]["children"][0]["props"]["data"]
            break
        assert time.monotonic() < deadline, "Research smoke test timed out"
        time.sleep(1)
    export = {
        "output": "..report-download.data...export-status.children..",
        "outputs": [{"id": "report-download", "property": "data"},
                    {"id": "export-status", "property": "children"}],
        "inputs": [{"id": "export-button", "property": "n_clicks", "value": 1}],
        "state": [{"id": "research-context", "property": "data", "value": context}],
        "changedPropIds": ["export-button.n_clicks"],
    }
    download = json.loads(request("/_dash-update-component", export))["response"]["report-download"]["data"]
    report = json.loads(download["content"])["report"]
    assert report["settings"]["ticker"] == "AAPL"
    assert report["sample"]["evaluated_predictions"] >= 30
    assert report["metrics"]["classification"]["accuracy"] is not None
    assert len(report["equity_curves"]["strategy"]) > 1
    assert report["dataset"]["id"] == context["snapshot_id"]
    print(f"AAPL research and report download passed; latest history {report['sample']['history_end']}; "
          f"{report['sample']['evaluated_predictions']} evaluated predictions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--research", action="store_true", help="Submit an AAPL research job (uses provider access)")
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    smoke(args.base_url, args.research, args.timeout)
