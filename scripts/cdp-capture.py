import argparse
import base64
import json
import time
import urllib.request
from pathlib import Path

import websocket


def call(socket: websocket.WebSocket, request_id: int, method: str, params: dict | None = None) -> dict:
    message = {"id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    socket.send(json.dumps(message))
    while True:
        response = json.loads(socket.recv())
        if response.get("id") == request_id:
            return response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--nav-index", type=int)
    parser.add_argument("--expression")
    parser.add_argument("--port", type=int, default=9224)
    args = parser.parse_args()

    pages = json.load(urllib.request.urlopen(f"http://127.0.0.1:{args.port}/json"))
    page = next(item for item in pages if item.get("type") == "page")
    socket = websocket.create_connection(page["webSocketDebuggerUrl"], suppress_origin=True)
    try:
        if args.nav_index is not None:
            call(
                socket,
                1,
                "Runtime.evaluate",
                {"expression": f"document.querySelectorAll('.nav-item')[{args.nav_index}].click()"},
            )
            time.sleep(1)
        if args.expression:
            call(socket, 3, "Runtime.evaluate", {"expression": args.expression})
            time.sleep(0.5)
        response = call(
            socket,
            2,
            "Page.captureScreenshot",
            {"format": "png", "captureBeyondViewport": False},
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(base64.b64decode(response["result"]["data"]))
        print(args.output.resolve())
    finally:
        socket.close()


if __name__ == "__main__":
    main()
