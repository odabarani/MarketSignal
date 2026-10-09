import json
import sqlite3

import pytest
from dash import Dash

from src import library
from src.reports import ReportStore
from src.snapshots import SnapshotStore
from src.studies import StudyStore
from test_app import make_market_data
from test_study_ui import post


def report():
    return {"created_at": "2026-10-01T12:00:00+00:00", "settings": {"ticker": "AAPL", "threshold": .6, "retrain_days": 21},
            "latest_estimate": {"date": "2026-09-30", "up_probability": .61},
            "metrics": {"strategy": {"total_return": .1, "max_drawdown": -.2}},
            "equity_curves": {"dates": ["2025-01-01", "2026-01-01"], "strategy": [10000, 11000],
                             "stock_buy_hold": [10000, 12000], "spy_buy_hold": [10000, 11500]}}


def test_empty_history_then_records_survive_new_store_instance(tmp_path):
    path = tmp_path / 'data.sqlite'
    store = SnapshotStore(path)
    assert library.recent_records(store) == []
    key = ReportStore(store).save(report())
    raw = store.get('AAPL', lambda _: make_market_data())
    study = StudyStore(path).prepare(raw.attrs['snapshot']['id'])
    records = library.recent_records(SnapshotStore(path))
    assert {item['value'] for item in records} == {f'report:{key}', f"study:{study['study_id']}"}
    assert 'AAPL' in records[0]['label']


def test_history_is_bounded_and_detects_corruption(tmp_path):
    store = SnapshotStore(tmp_path / 'data.sqlite')
    for day in range(35):
        item = report()
        item['created_at'] = f'2026-10-01T12:00:{day:02d}+00:00'
        key = ReportStore(store).save(item)
    assert len(library.recent_records(store)) == 30
    with sqlite3.connect(store.path) as conn:
        conn.execute('UPDATE reports SET payload = ? WHERE id = ?', ('{}', key))
    with pytest.raises(ValueError, match='integrity'):
        library.load_record(store, f'report:{key}')
    for bad in [None, {}, 'report:1', 'study:../../x', "report:' OR 1=1"]:
        with pytest.raises(ValueError, match='Choose'):
            library.load_record(store, bad)


def test_saved_inspection_and_download_do_not_refit_or_fetch(tmp_path, monkeypatch):
    import app as dashboard
    store = SnapshotStore(tmp_path / 'data.sqlite')
    record = report()
    key = ReportStore(store).save(record)
    def unexpected(*args, **kwargs):
        raise AssertionError('Saved reports must not fetch or refit')
    monkeypatch.setattr(dashboard, 'get_stock_data', unexpected)
    monkeypatch.setattr(dashboard, 'run_model_research', unexpected)
    app = Dash(__name__)
    app.layout = library.layout()
    library.register(app, lambda: store)
    identity = f'report:{key}'
    result = post(app, '..library-view', [('library-record', 'value', identity), ('library-refresh', 'n_clicks', 0)], [], 'library-record.value')
    assert result['library-export']['disabled'] is False
    assert 'Archived results' in json.dumps(result['library-view'])
    result = post(app, '..library-download', [('library-export', 'n_clicks', 1)],
                  [('library-record', 'value', identity)], 'library-export.n_clicks')
    payload = json.loads(result['library-download']['data']['content'])
    assert payload == {'report_id': key, 'report': record}


def test_invalid_selection_disables_download(tmp_path):
    app = Dash(__name__)
    app.layout = library.layout()
    library.register(app, lambda: SnapshotStore(tmp_path / 'data.sqlite'))
    result = post(app, '..library-view', [('library-record', 'value', 'report:bad'), ('library-refresh', 'n_clicks', 0)], [], 'library-record.value')
    assert result['library-export']['disabled'] is True
