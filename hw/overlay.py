"""Show a sim layout on top of the live camera, to place the blocks against.

    uv run python -m hw.overlay --index 0

Served as a page rather than drawn in a window, so the operator can stand at the arm and
work from whatever is in their hand. Blend towards the sim view to find where a block
belongs, towards the live view to see where it is. The edge mode draws the sim view's
outlines over the live frame, which is the easier one to nudge a block under.

One page lasts a whole session. `Console.place` puts a layout up and returns when the page
says the blocks are down, so a run of layouts costs no reloads; the stream holds its last
frame over whatever the caller does with the arm in between.
"""

from __future__ import annotations

import argparse
import socket
import threading
import time
from contextlib import contextmanager, suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from hw.robot import PARK_QPOS, ToSimCameras, follower, home
from sim.spec import CAMERAS

FPS = 15
BOUNDARY = "frame"

PAGE = """<!doctype html>
<title>sim over real</title>
<style>
 body { margin: 0; background: #111; color: #ddd; font: 14px system-ui; }
 img { width: 100%; max-width: 96vh; display: block; margin: 0 auto; }
 .bar { display: flex; gap: 1.5em; align-items: center; padding: .4em 1em; flex-wrap: wrap; }
 .bar > * { display: flex; gap: .4em; align-items: center; }
 button { font: inherit; padding: .3em .9em; }
</style>
<div class="bar">
  <button id=ready>blocks are down</button>
  <button id=skip>skip</button>
  <span id=note></span>
</div>
<div class="bar">
  <label>blend <input type=range min=0 max=100 value=50 id=blend></label>
  <label><input type=checkbox id=edges checked> sim edges</label>
  <label><input type=radio name=cam value=top checked> top</label>
  <label><input type=radio name=cam value=wrist> wrist</label>
  <button id=snap>save png</button>
  <span id=saved></span>
</div>
<img src="/stream">
<script>
 const send = () => fetch(`/set?blend=${blend.value / 100}&edges=${edges.checked ? 1 : 0}`
   + `&camera=${document.querySelector('input[name=cam]:checked').value}`);
 blend.oninput = send; edges.onchange = send;
 for (const r of document.querySelectorAll('input[name=cam]')) r.onchange = send;
 ready.onclick = () => fetch('/answer?say=ready');
 skip.onclick = () => fetch('/answer?say=skip');
 snap.onclick = async () => saved.textContent = await (await fetch('/snapshot')).text();
 setInterval(async () => note.textContent = await (await fetch('/note')).text(), 500);
</script>
"""


class Console:
    """What the page is composing, and what the operator has told it."""

    def __init__(self, robot, out):
        self.robot = robot
        self.out = out
        self.to_sim = ToSimCameras()
        self.blend, self.edges, self.camera = 0.5, True, CAMERAS[0]
        self.views, self.note, self.answer = None, "", None
        self.jpeg, self.frame, self.saved = None, None, 0
        self.lock = threading.Lock()
        # Composing costs a remap and a JPEG encode. The replay runs to a 50 Hz clock, so
        # this stands down while it has the arm and the stream sits on its last frame.
        self.live = threading.Event()

    def compose(self, stop):
        """Keep the latest frame composed, for as long as a layout is up."""
        while not stop.is_set():
            if not self.live.wait(timeout=0.2):
                continue
            # Ahead of the work rather than after it, so a frame this thread gives up on
            # costs the same wait as one it composes.
            time.sleep(1 / FPS)
            with self.lock:
                blend, edges, camera, views = (self.blend, self.edges, self.camera,
                                               self.views)
            try:
                frame = {camera: self.robot.cameras[camera].read_latest()}
            except (OSError, RuntimeError):
                # A camera that has stalled, or re-enumerated onto another device number,
                # holds the stream on its last frame, which shows as a freeze. Both arrive
                # as `OSError`, a stale one through `TimeoutError`.
                continue
            frame = compose(None if views is None else views[camera],
                            self.to_sim.observation(frame)[camera], blend, edges)
            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                with self.lock:
                    self.jpeg, self.frame = buffer.tobytes(), frame

    def say(self, note):
        """Put a line of status on the page."""
        with self.lock:
            self.note = note

    def place(self, views, note):
        """Hold until the page says the blocks are down. False to skip the layout."""
        with self.lock:
            self.views, self.note, self.answer = views, note, None
        self.live.set()
        answer = None
        while answer is None:
            time.sleep(0.1)
            with self.lock:
                answer = self.answer
        self.live.clear()
        return answer == "ready"

    @contextmanager
    def watch(self):
        """Compose the stream while the caller has the arm, so the operator sees it move.

        Composing costs a remap and an encode on the caller's clock, which is why `place`
        clears it.
        """
        self.live.set()
        try:
            yield
        finally:
            self.live.clear()


def compose(sim_view, live, blend, edges):
    """The two views as one frame, in the byte order the encoder wants.

    Without a sim view the live frame stands alone, which is what a caller wanting the
    page for its button rather than for a layout gets.

    The outline comes off the three channels rather than off their luminance, because the
    sim green block and the sim table sit within 2% of each other in luminance and a
    greyscale edge misses the block that is often the one being placed.
    """
    if sim_view is None:
        return cv2.cvtColor(live, cv2.COLOR_RGB2BGR)
    frame = cv2.addWeighted(live, 1 - blend, sim_view, blend, 0.0)
    if edges:
        outline = np.zeros(sim_view.shape[:2], np.uint8)
        for channel in cv2.split(sim_view):
            outline |= cv2.Canny(channel, 60, 160)
        frame[outline > 0] = (255, 255, 255)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def handler(console):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body, kind="text/plain"):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = urlparse(self.path)
            query = parse_qs(route.query)
            if route.path == "/":
                self._send(PAGE.encode(), "text/html")
            elif route.path == "/note":
                with console.lock:
                    self._send(console.note.encode())
            elif route.path == "/set":
                with console.lock:
                    console.blend = float(query["blend"][0])
                    console.edges = query["edges"][0] == "1"
                    console.camera = query["camera"][0]
                self._send(b"ok")
            elif route.path == "/answer":
                with console.lock:
                    console.answer = query["say"][0]
                self._send(b"ok")
            elif route.path == "/snapshot":
                with console.lock:
                    console.saved += 1
                    frame = console.frame
                    path = console.out / f"{console.camera}_{console.saved:02d}.png"
                if frame is None:
                    self._send(b"no frame yet")
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(path), frame)
                    self._send(f"wrote {path.name}".encode())
            elif route.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type",
                                 f"multipart/x-mixed-replace; boundary={BOUNDARY}")
                self.end_headers()
                sent = None
                # A reload leaves the old stream's socket closed under us, so the write
                # is where a page that has gone away is noticed.
                with suppress(ConnectionError):
                    while True:
                        with console.lock:
                            jpeg = console.jpeg
                        if jpeg is not None and jpeg is not sent:
                            self.wfile.write(
                                f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                                f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                            self.wfile.write(jpeg + b"\r\n")
                            sent = jpeg
                        time.sleep(1 / FPS)
            else:
                self.send_error(404)

    return Handler


def address():
    """The address of this machine that something else on the network can reach."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("8.8.8.8", 53))
        return probe.getsockname()[0]


@contextmanager
def serve(robot, out, port):
    """A console on ``port``, for as long as the block runs."""
    console = Console(robot, out)
    stop = threading.Event()
    server = ThreadingHTTPServer(("0.0.0.0", port), handler(console))
    print(f"http://{address()}:{port}/", flush=True)
    threading.Thread(target=console.compose, args=(stop,), daemon=True).start()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield console
    finally:
        stop.set()
        server.shutdown()


def main():
    # Imported here rather than at the top because it reaches the simulator, and the
    # console itself runs anywhere the cameras do.
    from hw.oracle import LAYOUTS, planner

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layouts", type=Path, default=LAYOUTS)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--out", type=Path, default=Path("."))
    args = parser.parse_args()

    with planner(args.layouts) as plan:
        laid_out = plan(args.index)
    print(f"layout {args.index}: {laid_out.prompt}")

    robot = follower()
    robot.connect()
    try:
        home(robot, PARK_QPOS)
        with serve(robot, args.out, args.port) as console:
            console.place(laid_out.views, f"layout {args.index}: {laid_out.prompt}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
