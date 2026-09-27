import socket
import threading
import time
import urllib.request

import webview
from waitress import serve

from app import server


def available_port():
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        return connection.getsockname()[1]


def wait_until_ready(url, attempts=60):
    for _ in range(attempts):
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("MarketSignal could not start its local server.")


def main():
    port = available_port()
    url = f"http://127.0.0.1:{port}"
    server_thread = threading.Thread(
        target=serve,
        kwargs={"app": server, "host": "127.0.0.1", "port": port, "threads": 4},
        daemon=True,
    )
    server_thread.start()
    wait_until_ready(url)
    webview.create_window(
        "MarketSignal", url, width=1380, height=900, min_size=(980, 680)
    )
    webview.start()


if __name__ == "__main__":
    main()
