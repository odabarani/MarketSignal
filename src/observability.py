"""Local readiness and correlated HTTP logs; probes never call market providers."""

import logging
import time
from uuid import uuid4

from flask import g, jsonify, request
from werkzeug.exceptions import HTTPException


def register_observability(server, get_store, get_jobs):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    server.logger.setLevel(logging.INFO)

    @server.before_request
    def begin_request():
        # Generate our own identifier; never trust arbitrary header contents in logs.
        g.request_id = uuid4().hex
        g.request_started = time.monotonic()

    @server.after_request
    def finish_request(response):
        response.headers["X-Request-ID"] = g.request_id
        if request.path in ("/health", "/ready"):
            response.headers["Cache-Control"] = "no-store"
        server.logger.info("http_request request_id=%s method=%s status=%s duration_ms=%.1f",
                           g.request_id, request.method, response.status_code,
                           (time.monotonic() - g.request_started) * 1000)
        return response

    @server.errorhandler(Exception)
    def unexpected_error(error):
        if isinstance(error, HTTPException):
            return error
        server.logger.exception("http_failure request_id=%s", g.request_id)
        return jsonify(error="The request could not finish. Please try again.",
                       request_id=g.request_id), 500

    @server.get("/health")
    def health():
        return jsonify(status="alive")

    @server.get("/ready")
    def ready():
        try:
            get_store().check_writable()
            scheduler = get_jobs().overview()
            healthy = scheduler["healthy"]
            return jsonify(status="ready" if healthy else "unavailable",
                           storage="writable", scheduler=scheduler), 200 if healthy else 503
        except Exception:
            server.logger.exception("readiness_failure request_id=%s", g.request_id)
            return jsonify(status="unavailable", storage="unavailable",
                           request_id=g.request_id), 503
