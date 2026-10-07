import json
import struct

import app


def test_install_metadata_and_icons_are_complete():
    client = app.server.test_client()
    page = client.get('/').data.decode()
    assert 'rel="manifest" href="/manifest.webmanifest"' in page
    assert 'apple-touch-icon' in page
    assert 'viewport-fit=cover' in page
    response = client.get('/manifest.webmanifest')
    assert response.mimetype == 'application/manifest+json'
    manifest = response.get_json()
    assert manifest['display'] == 'standalone'
    assert manifest['id'] == manifest['start_url'] == manifest['scope'] == '/'
    for icon in manifest['icons']:
        response = client.get(icon['src'])
        assert response.status_code == 200
        assert response.mimetype == 'image/png'
        assert response.data[:8] == b'\x89PNG\r\n\x1a\n'
        width, height = struct.unpack('>II', response.data[16:24])
        assert icon['sizes'] == f'{width}x{height}'
        assert width in (192, 512)
    assert client.get('/assets/icons/app-180.png').status_code == 200


def test_offline_route_and_worker_have_correct_scope_and_cache_policy():
    client = app.server.test_client()
    worker = client.get('/service-worker.js')
    assert worker.status_code == 200
    assert worker.mimetype == 'text/javascript'
    assert worker.headers['Service-Worker-Allowed'] == '/'
    assert worker.headers['Cache-Control'] == 'no-cache'
    assert b'__REVISION__' not in worker.data
    offline = client.get('/offline')
    assert offline.status_code == 200
    assert b'does not show cached prices or results' in offline.data
    assert b'href="/"' in offline.data


def test_workspace_mounts_both_workflows_for_navigation_without_losing_state():
    layout = app.server.test_client().get('/_dash-layout')
    assert layout.status_code == 200
    data = json.dumps(layout.get_json())
    for name in ('workspace-mode', 'research-workspace', 'study-workspace', 'connection-banner'):
        assert f'"id": "{name}"' in data
