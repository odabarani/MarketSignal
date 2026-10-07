"""Installable app metadata and an offline landing page; research is never cached."""
from hashlib import sha256
from pathlib import Path

from flask import jsonify, make_response, send_file


def register_installable(server, root):
    assets = Path(root) / "assets"

    @server.get("/manifest.webmanifest")
    def manifest():
        response = jsonify({
            "id": "/", "name": "MarketSignal", "short_name": "MarketSignal",
            "description": "Market research, strategy evaluation and retained study records.",
            "start_url": "/", "scope": "/", "display": "standalone",
            "background_color": "#07101f", "theme_color": "#07101f", "lang": "en",
            "icons": [
                {"src": f"/assets/icons/app-{size}.png", "sizes": f"{size}x{size}",
                 "type": "image/png", "purpose": "any maskable"}
                for size in (192, 512)
            ],
        })
        response.mimetype = "application/manifest+json"
        response.headers["Cache-Control"] = "no-cache"
        return response

    @server.get("/offline")
    def offline():
        response = make_response(send_file(assets / "offline.html"))
        response.headers["Cache-Control"] = "no-cache"
        return response

    @server.get("/service-worker.js")
    def service_worker():
        # Change the cache name whenever an offline asset changes. Only this
        # small, public landing page is cached; no API, research, or app HTML.
        revision = sha256(b"".join((assets / name).read_bytes() for name in
                                  ("offline.html", "icons/app-192.png", "icons/app-512.png"))).hexdigest()[:16]
        script = (assets / "service-worker.txt").read_text().replace("__REVISION__", revision)
        response = make_response(script)
        response.mimetype = "text/javascript"
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Service-Worker-Allowed"] = "/"
        return response
