"""Publish what the bench is doing for a status page, without touching the plant.

Doug asked (2026-11-25) for a screen beside the arm that says what it is doing
("carrying the red block to (52, -7)") and shows the wrist and scene cameras,
to read failures and successes as they happen. The page is a separate,
low-priority process; this publisher is the only harness side of it.

It writes into a directory -- ``CAP_HARNESS_STATUS_DIR``, meant to be on tmpfs
(``/dev/shm/hebbot_status``) -- a few times a second:

    status.json        what the program last announced, the last plant
                       command, the run's identity, timestamps
    wrist.jpg          the latest frame of each camera, downscaled
    scene.jpg

Rules, since the arm is what matters:

* it never opens a device: frames come from the station's threaded cameras,
  which hand out their latest frame under a lock (the same call the episode
  recorder makes), and it never reads the robot (``libfranka`` lets one thread
  at a time do that); plant activity comes from ``env.events``, the command
  log the env already keeps;
* it runs on its own daemon thread at a low rate, and every write is replaced
  atomically (write a temporary file, rename), so a reader never sees half a
  file;
* any error is counted and swallowed; after repeated failures it stops itself.
  A run never waits on it and never fails because of it.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

STATUS_DIR_ENV = "CAP_HARNESS_STATUS_DIR"
#: Publishing rate. Enough to follow a carry, cheap next to the recorder's 20 Hz mp4.
DEFAULT_HZ = 3.0
#: Thumbnail width in pixels; 640x480 frames become 480x360.
THUMB_WIDTH = 480
JPEG_QUALITY = 70
#: Consecutive failed cycles before the publisher gives up for the run.
MAX_ERRORS = 50


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
    os.replace(tmp, path)


def _jpeg(rgb: np.ndarray, width: int = THUMB_WIDTH) -> bytes:
    from PIL import Image

    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    if image.width > width:
        image = image.resize((width, round(image.height * width / image.width)))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()


def describe_event(event: dict[str, Any] | None) -> str | None:
    """One line for the plant's last command, from the env's event log."""
    if not event:
        return None
    kind = event.get("kind")
    if kind in ("waypoints", "waypoints_retry"):
        return "moving"
    if kind == "waypoints_done":
        return "move done" if event.get("converged", True) else "move did not converge"
    if kind == "descend":
        return "descending until contact"
    if kind == "descend_done":
        stop = event.get("stop")
        z = event.get("contact_z")
        return f"descent stopped: {stop}" + (f" at z {z * 100:.1f} cm" if z is not None else "")
    if kind == "gripper":
        width = event.get("commanded_width_m")
        measured = event.get("measured_width_m")
        if width is None:
            return "gripper"
        text = "opening the gripper" if width > 0.05 else "closing the gripper"
        if measured is not None:
            text += f" (fingers {measured * 1000:.0f} mm apart)"
        return text
    if kind == "policy_chunk":
        return f"policy {event.get('policy')} moved {event.get('steps')} steps"
    if kind == "reflex_recovered":
        return "recovered from a reflex"
    return str(kind)


class StatusPublisher:
    """Daemon thread that mirrors the run's state into a status directory."""

    def __init__(self, env: Any, directory: str | Path, *, hz: float = DEFAULT_HZ) -> None:
        self._env = env
        self.directory = Path(directory)
        self.period_s = 1.0 / max(0.2, float(hz))
        self._lock = threading.Lock()
        self._fields: dict[str, Any] = {}
        self._announcements: list[dict[str, Any]] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self.errors = 0
        self._consecutive_errors = 0

    # -- what the harness tells it ------------------------------------------

    def update(self, **fields: Any) -> None:
        """Set run-level fields (program, seed, task, policy, instruction...)."""
        with self._lock:
            self._fields.update(fields)

    def announce(self, text: str) -> None:
        """The program's own words for what it is doing now."""
        entry = {"t": time.time(), "text": str(text)[:300]}
        with self._lock:
            self._fields["activity"] = entry["text"]
            self._fields["activity_t"] = entry["t"]
            self._announcements.append(entry)
            del self._announcements[:-12]

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> bool:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except Exception:
            return False
        with self._lock:
            self._fields.setdefault("started_t", time.time())
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="franka-status", daemon=True)
        self._thread.start()
        return True

    def stop(self, **final: Any) -> None:
        self._running = False
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._thread = None
        try:
            self.update(running=False, ended_t=time.time(), **final)
            self._write_status()
        except Exception:
            self.errors += 1

    # -- the thread -------------------------------------------------------------

    def _loop(self) -> None:
        while self._running:
            began = time.monotonic()
            try:
                self._publish()
                self._consecutive_errors = 0
            except Exception:
                self.errors += 1
                self._consecutive_errors += 1
                if self._consecutive_errors >= MAX_ERRORS:
                    self._running = False
                    return
            remaining = self.period_s - (time.monotonic() - began)
            time.sleep(max(remaining, 0.05))

    def _publish(self) -> None:
        env = self._env
        cameras: dict[str, float] = {}
        roles = [("wrist", getattr(env, "capture", None)), ("scene", getattr(env, "capture_scene", None))]
        for name, capture in roles:
            if capture is None:
                continue
            try:
                frame = capture()
            except Exception:
                self.errors += 1
                continue
            if frame is None or getattr(frame, "rgb", None) is None:
                continue
            _atomic_write(self.directory / f"{name}.jpg", _jpeg(frame.rgb))
            cameras[name] = float(getattr(frame, "timestamp_s", 0.0) or 0.0)
        self._write_status(cameras)

    def _write_status(self, cameras: dict[str, float] | None = None) -> None:
        events = getattr(self._env, "events", None) or []
        last = None
        try:
            last = dict(events[-1]) if events else None
        except Exception:
            last = None
        with self._lock:
            status = dict(self._fields)
            status["announcements"] = list(self._announcements)
        status.setdefault("running", True)
        status["updated_t"] = time.time()
        status["plant"] = describe_event(last)
        status["plant_event"] = None if last is None else last.get("kind")
        if cameras is not None:
            status["cameras"] = sorted(cameras)
        status["publisher_errors"] = self.errors
        _atomic_write(self.directory / "status.json", json.dumps(status, default=str).encode())


def from_environment(env: Any) -> StatusPublisher | None:
    """A publisher when ``CAP_HARNESS_STATUS_DIR`` is set, else None (the default)."""
    directory = os.environ.get(STATUS_DIR_ENV, "").strip()
    if not directory:
        return None
    return StatusPublisher(env, directory)
