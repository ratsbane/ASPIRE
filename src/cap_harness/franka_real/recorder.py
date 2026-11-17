"""Record one real-Franka episode at the plant's own rates, for training.

The harness trace is per call: a 3 s descent is one ``descend_to_contact``
record, a 7 s carry one ``move_to_pose``. A policy needs the trajectory
between calls. This recorder keeps three streams whose rates are set by what
produces them, and leaves resampling to the converter that builds a training
set from the raw layout:

* **arm** at the robot's 1 kHz, from panda_py's own state log, which the
  control loop feeds in its torque callback. That is the one source that does
  not race the robot: ``libfranka`` lets exactly one thread read or control
  the arm, and a second thread's ``get_state`` would contend with the move the
  program is in the middle of. While the arm idles the log only grows when the
  plant is read, so an idle stretch is a gap -- and the hand did not move in
  it, so hold-last is exact.
* **cameras** at their own frame rate, sampled by a daemon thread through the
  station's threaded cameras (which already hand out their latest frame under
  a lock): colour as mp4, depth as 16-bit PNG every few frames, and every
  frame's own timestamp so repeats can be dropped.
* **events**: what the plant was told, with host timestamps -- each waypoint
  batch and its target, each gripper command and what the fingers measured,
  each contact descent and how it stopped. The gripper is sampled by the
  camera thread too, under the env's gripper lock, because the trace only
  sees it at call boundaries.

Raw files written into ``out_dir`` (``episode/raw`` of a harness run):

    arm-q.npy, arm-dq.npy           (N, 7)   measured, 1 kHz during motion
    arm-tau_J.npy, arm-tau_ext.npy  (N, 7)
    arm-O_T_EE.npy                  (N, 16)  column-major 4x4, base frame
    arm-O_F_ext_hat_K.npy           (N, 6)   external wrench estimate
    arm-time_ms.npy                 (N,)     robot clock
    gripper-width_m.npy, gripper-t_s.npy     sampled with the cameras
    camera-<role>-t_s.npy, camera-<role>-frame_t_s.npy, -fresh.npy
    depth/<role>/NNNNNN.png                  uint16 millimetres
    events.json, metadata.json
    {role}-images-rgb.mp4 in ``video_dir``   one per camera

Sampling is best-effort by construction: every read is wrapped, failures are
counted, and the thread is a daemon. Recording must never stall or kill the
robot loop.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

#: Camera sampling rate. The wrist RealSense streams at 30 fps and the scene
#: camera at ~13 fps over USB2; 20 Hz keeps every scene frame and most wrist
#: frames, and ``*-fresh.npy`` marks the repeats.
DEFAULT_FPS = 20
#: Depth is kept every Nth sample: 5 Hz at 20 fps, since a 16-bit PNG per
#: frame is the bulk of an episode and the colour stream carries the motion.
DEPTH_EVERY_FRAMES = 4
#: panda_py log capacity in samples at 1 kHz; 1.2M is twenty minutes, far
#: beyond any one program run, so nothing is dropped mid-episode.
ARM_LOG_SAMPLES = 1_200_000
#: Arm log columns, as panda_py names them, and their widths.
ARM_CHANNELS: dict[str, int] = {
    "q": 7,
    "dq": 7,
    "tau_J": 7,
    "tau_ext_hat_filtered": 7,
    "O_T_EE": 16,
    "O_F_ext_hat_K": 6,
    "time": 1,
}


class FrankaEpisodeRecorder:
    """Arm log, camera sampler and command events for one episode."""

    def __init__(
        self,
        out_dir: str | Path,
        *,
        fps: int = DEFAULT_FPS,
        cameras: tuple[str, ...] | None = None,
        task: str = "",
        video_dir: str | Path | None = None,
        source: str = "cap_program",
    ) -> None:
        self.out_dir = Path(out_dir)
        self.video_dir = Path(video_dir) if video_dir is not None else self.out_dir
        self.fps = int(fps)
        self.task = task
        self.cameras = cameras
        self.current_source = source

        self._env: Any = None
        self._timestamps: list[float] = []
        self._gripper: list[float] = []
        self._source: list[str] = []
        self._frame_t: dict[str, list[float]] = {}
        self._fresh: dict[str, list[bool]] = {}
        self._last_frame_t: dict[str, float] = {}
        self._writers: dict[str, Any] = {}
        self._frames: dict[str, int] = {}
        self._depth_saved: dict[str, int] = {}
        self._metadata: dict[str, Any] = {}
        self._started_at: float | None = None
        self._clock_pairs: list[dict[str, float | None]] = []
        self._arm_logging = False
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._sample_errors = 0
        self._last_width: float = float("nan")

    # -- lifecycle ---------------------------------------------------------

    def start(self, env: Any, *, meta: dict[str, Any] | None = None) -> None:
        """Enable the arm log and begin sampling the cameras on a thread."""
        self._env = env
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        if self.cameras is None:
            roles = [env.config.camera.role]
            if getattr(env.config, "scene_camera", None) is not None:
                roles.append(env.config.scene_camera.role)
            self.cameras = tuple(roles)
        for role in self.cameras:
            self._frame_t[role] = []
            self._fresh[role] = []
        env.events.clear()
        self._metadata = {
            "robot": "franka_panda",
            "station": env.config.station,
            "task": self.task,
            "fps": self.fps,
            "cameras": list(self.cameras),
            "arm_log_hz": 1000,
            "depth_every_frames": DEPTH_EVERY_FRAMES,
            "control_contract": {
                "motion": "panda_py_motion_generator",
                "contact_descent": "panda_py_cartesian_impedance",
                "contact_stiffness": [400.0, 400.0, 300.0, 30.0, 30.0, 30.0],
                "speed_factor": float(env.config.speed_factor),
                "contact_descent_speed_m_s": float(env.config.contact_descent_speed_m_s),
            },
        }
        if meta:
            self._metadata.update(meta)
        self._started_at = time.monotonic()
        self._arm_logging = bool(env.start_arm_log(ARM_LOG_SAMPLES))
        self._clock_pairs.append(self._clock_pair("start"))
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="franka-episode-recorder", daemon=True
        )
        self._thread.start()

    def set_source(self, source: str) -> None:
        """Label subsequent frames, e.g. when a policy rather than a program drives."""
        with self._lock:
            self.current_source = str(source)

    def _clock_pair(self, label: str) -> dict[str, float | None]:
        """Host monotonic time against the robot clock, read on the caller's thread."""
        robot_ms = None
        try:
            robot_ms = self._env.robot_time_ms()
        except Exception:
            self._sample_errors += 1
        return {"label": label, "host_t_s": time.monotonic(), "robot_time_ms": robot_ms}

    # -- sampling ----------------------------------------------------------

    def _loop(self) -> None:
        period = 1.0 / max(1.0, float(self.fps))
        while self._running:
            began = time.monotonic()
            try:
                self._sample()
            except Exception:
                self._sample_errors += 1
            remaining = period - (time.monotonic() - began)
            if remaining > 0:
                time.sleep(remaining)

    def _sample(self) -> None:
        env = self._env
        now = time.monotonic()
        index = len(self._timestamps)
        # The gripper only when no command holds it: a wait here would queue
        # behind a grasp, and the width does not change between commands.
        width = None
        try:
            width = env.read_gripper_width(wait=False)
        except Exception:
            self._sample_errors += 1
        if width is not None:
            self._last_width = float(width)
        self._timestamps.append(0.0 if self._started_at is None else now - self._started_at)
        self._gripper.append(self._last_width)
        with self._lock:
            self._source.append(self.current_source)
        for role in self.cameras or ():
            frame = env.capture() if role == env.config.camera.role else env.capture_scene()
            stamp = float("nan") if frame is None else float(frame.timestamp_s)
            fresh = frame is not None and stamp != self._last_frame_t.get(role)
            self._frame_t[role].append(stamp)
            self._fresh[role].append(bool(fresh))
            if frame is None:
                continue
            self._last_frame_t[role] = stamp
            self._write_frame(role, np.asarray(frame.rgb, dtype=np.uint8))
            if index % DEPTH_EVERY_FRAMES == 0 and getattr(frame, "depth_m", None) is not None:
                self._write_depth(role, index, np.asarray(frame.depth_m, dtype=np.float64))

    def _write_frame(self, role: str, frame: np.ndarray) -> None:
        if role not in self._writers:
            import imageio.v2 as imageio

            self._writers[role] = imageio.get_writer(
                self.video_dir / f"{role}-images-rgb.mp4",
                fps=self.fps,
                codec="libx264",
                macro_block_size=None,
            )
            self._frames[role] = 0
        self._writers[role].append_data(frame)
        self._frames[role] += 1

    def _write_depth(self, role: str, index: int, depth_m: np.ndarray) -> None:
        import imageio.v2 as imageio

        directory = self.out_dir / "depth" / role
        directory.mkdir(parents=True, exist_ok=True)
        millimetres = np.clip(np.nan_to_num(depth_m, nan=0.0) * 1000.0, 0, 65535).astype(np.uint16)
        imageio.imwrite(directory / f"{index:06d}.png", millimetres)
        self._depth_saved[role] = self._depth_saved.get(role, 0) + 1

    # -- finish ------------------------------------------------------------

    def finalize(self, *, terminal_event: str | None = None, success: bool = False) -> Path:
        """Stop sampling, take the arm log, and write the episode. Safe to call twice."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        for writer in self._writers.values():
            try:
                writer.close()
            except Exception:
                pass
        self._writers.clear()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        arm_samples = 0
        if self._arm_logging:
            self._arm_logging = False
            self._clock_pairs.append(self._clock_pair("finalize"))
            try:
                log = self._env.take_arm_log()
            except Exception:
                log = {}
                self._sample_errors += 1
            for key, width in ARM_CHANNELS.items():
                rows = log.get(key)
                array = (
                    np.zeros((0, width), dtype=np.float64)
                    if rows is None or len(rows) == 0
                    else np.asarray(rows, dtype=np.float64).reshape(len(rows), -1)[:, :width]
                )
                name = "arm-time_ms.npy" if key == "time" else f"arm-{key}.npy"
                np.save(self.out_dir / name, array.reshape(-1) if key == "time" else array)
                arm_samples = max(arm_samples, int(array.shape[0]))

        count = len(self._timestamps)
        np.save(self.out_dir / "gripper-t_s.npy", np.asarray(self._timestamps[:count]))
        np.save(self.out_dir / "gripper-width_m.npy", np.asarray(self._gripper[:count]))
        for role in self.cameras or ():
            np.save(self.out_dir / f"camera-{role}-t_s.npy", np.asarray(self._timestamps[:count]))
            np.save(
                self.out_dir / f"camera-{role}-frame_t_s.npy",
                np.asarray(self._frame_t[role][:count], dtype=np.float64),
            )
            np.save(
                self.out_dir / f"camera-{role}-fresh.npy",
                np.asarray(self._fresh[role][:count], dtype=bool),
            )
        events = list(getattr(self._env, "events", []) or [])
        (self.out_dir / "events.json").write_text(
            json.dumps(events, indent=1, default=str), encoding="utf-8"
        )
        (self.out_dir / "source.json").write_text(
            json.dumps(self._source[:count]), encoding="utf-8"
        )
        self._metadata.update(
            {
                "camera_samples": count,
                "duration_s": float(self._timestamps[count - 1]) if count else 0.0,
                "video_frames": dict(self._frames),
                "depth_frames": dict(self._depth_saved),
                "arm_samples": arm_samples,
                "events": len(events),
                "clock_pairs": self._clock_pairs,
                "terminal_event": terminal_event,
                "success": bool(success),
                "sample_errors": self._sample_errors,
            }
        )
        (self.out_dir / "metadata.json").write_text(
            json.dumps(self._metadata, indent=2, default=str), encoding="utf-8"
        )
        return self.out_dir


__all__ = ["ARM_CHANNELS", "DEFAULT_FPS", "FrankaEpisodeRecorder"]
