"""EnvironmentAdapter for a single-arm Franka bench.

The shape follows :class:`cap_harness.yam_real.adapter.YamRealAdapter` with the
same two commitments and one simplification. The commitments: anything that can
command the arm is refused unless ``allow_physical_motion`` was set at
construction (one gate, every command path calls it first), and construction
performs no motion. The simplification: this plant does not stream per-tick
commands -- every motion is a guarded waypoint batch through the Franka motion
generator -- so ``step`` is a one-waypoint batch and a trajectory's ``dt_s``
is advisory, recorded as such.

``check_success`` is earned, not judged from the video: it re-observes the
scene after the program ends and runs the task's physical postcondition from
:mod:`cap_harness.franka_real.success` (top-face band centroids of the coloured
blocks, heights the bench verified by touch). The measured evidence behind the
verdict is written to the run's ``logs/success-report.json``; a predicate that
cannot measure what it needs reports unknown there and the adapter returns the
protocol's conservative ``False``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Optional

import numpy as np

from cap_harness.contracts import (
    CameraObservation,
    ExecutionResult,
    Observation,
    Pose,
    RobotAction,
    RobotState,
    StepResult,
    SynchronizedTrajectory,
    TaskContext,
    Trajectory,
)
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.franka_real.env import ARM_DOF, RealFrankaEnv
from cap_harness.franka_real.registry import (
    FrankaRealTaskMetadata,
    FrankaRealTaskRegistry,
)
from cap_harness.geometry import matrix_to_pose, pose_to_matrix

EMBODIMENT = "franka_real"
ARM = "primary"
BASE_FRAME = "base"
CAMERA_FRAME = "wrist_camera"
SCENE_FRAME = "scene_camera"
JOINT_NAMES = tuple(f"panda_joint{index + 1}" for index in range(ARM_DOF))
CAMERA_WAIT_S = 2.0
CAMERA_STALE_S = 1.0
"""A wrist frame older than this is no frame: the camera thread hands out its
last frame forever when the USB stream stalls (run 0242: the stream stopped
23 s before the final verify, which then saw the scene as it had been and
called a block lost that stood exactly where the hand had put it)."""
POLICY_SPEED_FACTOR = 0.10
"""Motion-generator speed for a learned policy's chunks: slower than the
program's 0.12, since nobody has yet watched this policy move the arm."""
POLICY_MAX_STEPS = 100
"""The longest chunk a program may ask the policy host to execute at once."""
POLICY_GRIPPER_DEADBAND_M = 0.005
DESCENT_REFLEX_RETRIES = 2  # a contact descent is resent this many times after reflexes
POLICY_MIN_Z = 0.024        # no policy waypoint below this: the program's lowest grasp is ~0.025
"""A chunk's final gripper width closer than this to the measured one is left alone."""
HAND_EYE_ALPHA = 0.15
"""Weight of one placement residual in the eye's running correction. At 0.5
the centimetre of noise in each residual walked the correction a few
millimetres per leg (the hand-frame y component crept from 2 to 5 cm over
an afternoon); touch anchors every batch now, so a placement only nudges."""
HAND_EYE_DRIFT_M = 0.02
"""A residual beyond this is reported as drift: re-probe before trusting a look."""
HAND_EYE_HISTORY = 100
PLANE_STRIDE = 8
"""Depth pixels are subsampled this much for the table-plane fit (~4.8k points)."""
PLANE_MIN_INLIERS = 0.35
"""Fraction of valid depth that must lie on the fitted plane for it to be the table."""
PLANE_MAX_TILT_DEG = 40.0
"""A fitted tilt beyond this is not a bracket, it is the wrong plane: ignore it."""
PLANE_MIN_HEIGHT_M = 0.15
"""Level against the table only when the camera is at least this high above it."""
PLANE_TOLERANCES_M = (0.03, 0.015, 0.008)
"""Inlier bands for the successive plane refits."""
PLANE_RANSAC_TRIALS = 120
"""Three-point samples tried for the seed plane before the refits."""
PLANE_MAX_JUMP_DEG = 8.0
WRIST_LEVEL_ENV = "CAP_HARNESS_WRIST_LEVEL"
"""Set to 0 to look with the camera pose as calibrated: no plane fit, nothing reused."""
"""A fitted tilt this far from the last accepted one is a wrong plane, not the mount moving."""


class FrankaRealAdapter:
    """Single-arm Franka embodiment behind the shared adapter protocol."""

    def __init__(
        self,
        env: RealFrankaEnv,
        *,
        registry: FrankaRealTaskRegistry | None = None,
        run_observer: Any = None,
        allow_physical_motion: bool = False,
        task_context: TaskContext | None = None,
        policy_client: Any = None,
    ) -> None:
        self._env = env
        self._registry = registry or FrankaRealTaskRegistry()
        self._run_observer = run_observer
        self._allow_physical_motion = bool(allow_physical_motion)
        self._task_context = task_context
        self._policy_client = policy_client
        """The learned-policy host; built lazily from the environment when None."""
        self._policy_reset_done: set[str] = set()
        self.last_success_report: dict[str, Any] = {}
        """Evidence behind the most recent ``check_success`` verdict."""
        self._hand_eye = pose_to_matrix(
            Pose(
                position=np.asarray(env.config.camera.hand_eye_position, dtype=np.float64),
                quaternion_wxyz=np.asarray(
                    env.config.camera.hand_eye_quaternion_wxyz, dtype=np.float64
                ),
                frame="ee",
            )
        )
        self._last_look_rotation: np.ndarray | None = None
        self._wrist_correction = self._load_wrist_correction()
        """The eye's running correction; see :meth:`report_hand_eye_residual`."""
        self._last_level: tuple[np.ndarray, float] | None = None
        """The last table-levelling rotation and height shift, reused when a
        look is too low to contain the table."""
        self.last_level_report: dict[str, Any] = {"leveled": False}

    # -- the eye levels itself against the table ---------------------------

    def _level_to_table(self, frame: Any, camera_matrix: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Rotate the camera pose until the table in its depth image is horizontal.

        The bracket tilts (2026-11-17: ~15 degrees, which put a block on the
        table 9 cm sideways and 6 cm up), and a tilt cannot be absorbed by a
        translation. But every look from a survey height contains the table,
        and the table is horizontal: fit the dominant plane in the depth
        image, rotate the camera pose about its own origin until that plane's
        normal is vertical, then shift it so the plane sits at the configured
        table height. A look too low to contain the table reuses the last
        levelling. The report says what was done.
        """
        table_z = float(getattr(self._env.config, "table_plane_z", 0.017))
        height = float(camera_matrix[2, 3]) - table_z
        report: dict[str, Any] = {"leveled": False, "camera_height_m": round(height, 3)}
        depth = getattr(frame, "depth_m", None)
        intrinsics = getattr(frame, "intrinsics", None)
        if height >= PLANE_MIN_HEIGHT_M and depth is not None and intrinsics is not None:
            fit = _fit_table_plane(np.asarray(depth, dtype=np.float64), np.asarray(intrinsics), camera_matrix)
            if fit is not None:
                normal, point, inliers = fit
                tilt = float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))
                below = float(camera_matrix[2, 3] - point[2])
                report.update(
                    {
                        "tilt_deg": round(tilt, 2),
                        "plane_inliers": round(inliers, 3),
                        "plane_below_camera_m": round(below, 3),
                    }
                )
                previous_tilt = self.last_level_report.get("tilt_deg")
                jump = (
                    previous_tilt is not None
                    and self.last_level_report.get("leveled")
                    and abs(tilt - float(previous_tilt)) > PLANE_MAX_JUMP_DEG
                )
                if jump:
                    # A mount does not change angle by this much between two
                    # looks; a plane this different is not the table (a look
                    # after a placement once read the block 24 cm from where
                    # the hand had released it, run 0185). Keep the last.
                    report["reason"] = "tilt jump"
                    report["jump_deg"] = round(abs(tilt - float(previous_tilt)), 2)
                elif (
                    inliers >= PLANE_MIN_INLIERS
                    and tilt <= PLANE_MAX_TILT_DEG
                    and below >= PLANE_MIN_HEIGHT_M
                ):
                    rotation = _rotation_taking(normal, np.array([0.0, 0.0, 1.0]))
                    origin = camera_matrix[:3, 3]
                    levelled_point = rotation @ (point - origin) + origin
                    self._last_level = (rotation, table_z - float(levelled_point[2]))
                    report["leveled"] = True
                    report["reused"] = False
                else:
                    report["reason"] = "not the table"
            else:
                report["reason"] = "too little depth"
        else:
            report["reason"] = "too low" if height < PLANE_MIN_HEIGHT_M else "no depth"
        if os.environ.get(WRIST_LEVEL_ENV, "1") in ("0", "off", "no"):
            # The experiment's control: the fit runs and is reported as usual,
            # but the look uses the camera pose as calibrated (2026-11-19: the
            # placement residuals were 1 cm on looks that leveled afresh and
            # 2-4 cm on looks that reused a levelling, and the fit failed on
            # most looks at 0.33 m). The fit still runs so that every code
            # path, and every lazy import under it, is the same as when the
            # levelling is applied: skipping it left a C-level import to
            # happen under the program's frame, whose builtins are a mapping
            # proxy, and that import fails (runs 9002, 9003).
            report["applied"] = False
            report["disabled"] = True
            return camera_matrix, report
        if self._last_level is None:
            return camera_matrix, report
        rotation, dz = self._last_level
        if not report["leveled"]:
            report["reused"] = True
        origin = camera_matrix[:3, 3].copy()
        levelled = camera_matrix.copy()
        levelled[:3, :3] = rotation @ camera_matrix[:3, :3]
        levelled[:3, 3] = origin  # rotated about the camera's own origin
        levelled[2, 3] += dz
        report["height_shift_m"] = round(dz, 4)
        return levelled, report

    # -- the eye's running correction --------------------------------------

    def _correction_path(self) -> Path:
        configured = getattr(self._env.config.camera, "correction_file", None)
        if configured:
            return Path(configured).expanduser()
        return (
            Path.home()
            / ".cap_harness"
            / "franka_real"
            / f"{self._env.config.station}-wrist-correction.json"
        )

    def _load_wrist_correction(self) -> dict[str, Any]:
        blank: dict[str, Any] = {
            "offset_hand_xyz": [0.0, 0.0, 0.0],
            "frame": "hand",
            "samples": 0,
            "history": [],
        }
        try:
            data = json.loads(self._correction_path().read_text(encoding="utf-8"))
        except Exception:
            return blank
        if not isinstance(data, dict):
            return blank
        if "offset_hand_xyz" not in data:
            if "offset_xyz" not in data:
                return blank
            # A file from when the correction was a base-frame shift, learned
            # almost entirely with the hand pointing down at yaw zero, where
            # hand x is base x and hand y is base -y.
            base = np.asarray(data["offset_xyz"], dtype=np.float64).reshape(3)
            data["offset_hand_xyz"] = [round(float(base[0]), 4), round(float(-base[1]), 4), 0.0]
            data["migrated_from_base_xyz"] = [float(v) for v in base]
            data.pop("offset_xyz", None)
        data["frame"] = "hand"
        data.setdefault("samples", 0)
        data.setdefault("history", [])
        return data

    def _save_wrist_correction(self) -> None:
        try:
            path = self._correction_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._wrist_correction, indent=1), encoding="utf-8")
        except Exception:
            pass  # the correction still applies for this run

    @property
    def wrist_correction_hand_xyz(self) -> np.ndarray:
        """The eye's running correction, a translation in the hand (EE) frame."""
        return np.asarray(self._wrist_correction["offset_hand_xyz"], dtype=np.float64).reshape(3)

    def wrist_correction_xyz_at(self, ee_rotation: np.ndarray) -> np.ndarray:
        """The correction as a base-frame shift for a hand with this rotation."""
        return np.asarray(ee_rotation, dtype=np.float64).reshape(3, 3) @ self.wrist_correction_hand_xyz

    def _look_rotation(self) -> np.ndarray:
        """The hand's rotation at the last look, or now if nothing has looked."""
        if self._last_look_rotation is not None:
            return self._last_look_rotation
        state = self.get_robot_state()
        return pose_to_matrix(state.end_effector_poses[ARM])[:3, :3]

    def report_hand_eye_residual(
        self,
        seen_xy: object,
        truth_xy: object,
        *,
        source: str = "placement",
        weight: float | None = None,
    ) -> dict[str, Any]:
        """Teach the eye from the hand: where the camera saw an object against
        where the hand or a touch probe found it, both in the base frame.

        Exposed to programs as ``franka_real.report_hand_eye_residual``. The
        wrist camera's bracket moves (2026-11-17: a few degrees, centimetres
        on the table, between two legs of one run), so the hand-eye numbers
        are a starting point and this is the rest. Every verified placement is
        a sample for free -- the hand's position at release is truth to a few
        millimetres -- and a touch-probed position is a stronger one. The
        correction is a translation of the camera pose in the hand frame,
        where a mount that has shifted or tilted actually puts the error, so
        a lesson learned at one wrist yaw holds at every other; it is updated
        by an exponential average and persisted between runs. A residual
        beyond ``HAND_EYE_DRIFT_M`` is reported as drift so the program can
        decide to re-probe rather than trust the next look.

        What a placement can teach is bounded: the fingers centre a held
        block along their travel, so the residual exposes the eye's error on
        that axis, but across the fingers (along the block's length, in the
        usual grasp) a biased grasp is carried from pick to place and back
        and reads as zero. Touch is the evidence on that axis.
        """
        seen = np.asarray(seen_xy, dtype=np.float64).reshape(-1)[:2]
        truth = np.asarray(truth_xy, dtype=np.float64).reshape(-1)[:2]
        if seen.shape != (2,) or truth.shape != (2,) or not np.all(np.isfinite(seen - truth)):
            raise ValueError("seen_xy and truth_xy must each be two finite numbers")
        residual = seen - truth
        magnitude = float(np.linalg.norm(residual))
        samples = int(self._wrist_correction.get("samples", 0))
        if weight is not None:
            alpha = float(np.clip(weight, 0.0, 1.0))
        elif samples == 0:
            alpha = 1.0  # the first sample bootstraps the correction whole
        else:
            # A touch probe is stronger than a look, but not flawless: the
            # pads shove a light block while finding it, so even touch is
            # averaged in once a correction exists.
            alpha = HAND_EYE_ALPHA
        rotation = self._look_rotation()
        residual_hand = rotation.T @ np.array([residual[0], residual[1], 0.0])
        offset = self.wrist_correction_hand_xyz.copy()
        offset -= alpha * residual_hand
        history = list(self._wrist_correction.get("history", []))[-(HAND_EYE_HISTORY - 1) :]
        history.append(
            {
                "t": time.time(),
                "source": str(source),
                "residual_xy": [round(float(v), 4) for v in residual],
                "residual_hand_xyz": [round(float(v), 4) for v in residual_hand],
                "alpha": alpha,
            }
        )
        self._wrist_correction = {
            "offset_hand_xyz": [round(float(v), 4) for v in offset],
            "frame": "hand",
            "samples": samples + 1,
            "history": history,
        }
        self._save_wrist_correction()
        return {
            "residual_xy": [round(float(v), 4) for v in residual],
            "residual_m": round(magnitude, 4),
            "drift": magnitude > HAND_EYE_DRIFT_M,
            "alpha": alpha,
            "offset_xyz": [round(float(v), 4) for v in (rotation @ offset)],
            "offset_hand_xyz": list(self._wrist_correction["offset_hand_xyz"]),
            "samples": samples + 1,
            "source": str(source),
        }

    # -- identity ----------------------------------------------------------

    @property
    def embodiment(self) -> str:
        return EMBODIMENT

    @property
    def native_env(self) -> RealFrankaEnv:
        return self._env

    @property
    def control_frequency(self) -> float:
        return float(self._env.config.control_frequency_hz)

    @property
    def control_period_s(self) -> float:
        return self._env.config.control_period_s

    @property
    def physical_motion_authorized(self) -> bool:
        return self._allow_physical_motion

    def resolve_arm(self, arm: object) -> str:
        if arm in (None, "", ARM):
            return ARM
        raise ValueError(f"franka_real has one arm, {ARM!r}; got {arm!r}")

    # -- the interlock -----------------------------------------------------

    def _motion_denied(self) -> ApiError | None:
        """The single gate. Every command path calls this first."""
        if self._allow_physical_motion:
            return None
        return ApiError(
            code=ErrorCode.SAFETY_INTERLOCK,
            message=(
                "physical motion is not authorized for this Franka adapter; "
                "construct it with allow_physical_motion=True "
                "(cap-harness run --allow-motion)"
            ),
            recoverable=False,
            details={"motions_executed": self._env.motions_executed},
        )

    # -- task identity -----------------------------------------------------

    def reset(self, task_ref: object = None, seed: int = 0) -> Observation:
        """Bind task identity and observe. The scene is physical: no motion.

        A hardware reset is a scene operation, not a simulator seed, and the
        registry records per task how the scene is restored (``manual`` or the
        bench's own ``skill`` reset). Homing is an explicit, authorized
        ``go_home``, never a side effect of binding a task.
        """
        metadata = task_ref
        if isinstance(task_ref, str):
            metadata = self._registry.resolve(task_ref, 0)
        if isinstance(metadata, FrankaRealTaskMetadata):
            self._task_context = TaskContext(
                suite=metadata.suite_name,
                task_id=metadata.task_id,
                task_name=metadata.task_name,
                language=metadata.language,
                family=metadata.family,
                metadata={
                    "scene_reset": metadata.scene_reset,
                    "seed": int(seed),
                    "goal": dict(metadata.goal),
                },
            )
        elif self._task_context is None:
            raise ValueError("reset needs a franka_real task name or metadata")
        self._start_episode_recording()
        return self.get_observation()

    # -- episode recording -------------------------------------------------

    def _start_episode_recording(self) -> None:
        """Record the plant at its own rates into the run's episode directory.

        The trace is per call and a policy needs the trajectory between calls
        (see :mod:`cap_harness.franka_real.recorder`). Needs a run to write
        into, so an ad-hoc adapter without a run observer records nothing.
        """
        root = getattr(self._run_observer, "root", None)
        if root is None:
            return
        self.finish_episode_recording()
        from .recorder import FrankaEpisodeRecorder

        task = "" if self._task_context is None else str(self._task_context.task_name or "")
        recorder = FrankaEpisodeRecorder(
            Path(root) / "episode" / "raw",
            video_dir=Path(root) / "media" / "videos",
            cameras=None if getattr(self._run_observer, "capture_videos", True) else (),
            task=task,
        )
        try:
            recorder.start(
                self._env,
                meta={
                    "physical_motion_authorized": self._allow_physical_motion,
                    "seed": None
                    if self._task_context is None
                    else self._task_context.metadata.get("seed"),
                },
            )
        except Exception:
            return
        self._episode_recorder = recorder

    def finish_episode_recording(self) -> None:
        """Stop the sampler and write the episode; the run teardown calls this
        before the artifacts are sealed, so the files are listed in the manifest."""
        recorder = getattr(self, "_episode_recorder", None)
        self._episode_recorder = None
        if recorder is None:
            return
        try:
            recorder.finalize(success=bool(self.last_success_report.get("verdict", False)))
        except Exception:
            pass

    def get_task_context(self) -> TaskContext:
        if self._task_context is None:
            raise ValueError("no task bound; call reset first")
        return self._task_context

    # -- observation -------------------------------------------------------

    def get_robot_state(self) -> RobotState:
        raw = self._env.get_observations()
        pose = matrix_to_pose(raw["ee_transform"], frame=BASE_FRAME)
        width = float(raw["gripper_width_m"]) / self._env.config.gripper_max_width_m
        return RobotState(
            joint_positions={ARM: raw["joint_pos"]},
            joint_velocities={ARM: raw["joint_vel"]},
            end_effector_poses={ARM: pose},
            gripper_positions={ARM: float(np.clip(width, 0.0, 1.0))},
            base_frame=BASE_FRAME,
            joint_names={ARM: JOINT_NAMES},
            timestamp_s=time.time(),
            embodiment=EMBODIMENT,
        )

    @staticmethod
    def _frame_age_s(frame: Any) -> float:
        """Seconds since the frame was acquired; infinite for no frame."""
        if frame is None:
            return float("inf")
        return max(0.0, time.monotonic() - float(frame.timestamp_s))

    def get_observation(self) -> Observation:
        state = self.get_robot_state()
        deadline = time.monotonic() + CAMERA_WAIT_S
        frame = self._env.capture()
        while self._frame_age_s(frame) > CAMERA_STALE_S and time.monotonic() < deadline:
            time.sleep(0.05)
            frame = self._env.capture()
        if frame is None:
            raise ValueError(
                "the wrist camera produced no frame; the Observation contract "
                "requires one, and a silent placeholder would be mistaken for a scene"
            )
        age = self._frame_age_s(frame)
        if age > CAMERA_STALE_S:
            raise ValueError(
                f"the wrist camera's last frame is {age:.1f} s old: the stream has "
                "stalled (USB), and a stale frame would be mistaken for the scene"
            )
        ee = pose_to_matrix(state.end_effector_poses[ARM])
        # `frame` names the camera's own optical frame; `camera_pose` maps it
        # into the base frame, which is what lets mask back-projection land in
        # world coordinates. Setting both to the base frame silently skips the
        # transform and hands back camera-frame points labelled "base".
        camera_matrix = ee @ self._hand_eye
        # First level the eye against the table it can see, then apply the
        # running correction learned from the hand: a hand-frame shift of the
        # camera pose, turned with the wrist, moves every back-projected
        # point by the same amount.
        camera_matrix, self.last_level_report = self._level_to_table(frame, camera_matrix)
        self._last_look_rotation = np.array(ee[:3, :3], dtype=np.float64)
        camera_matrix[:3, 3] += self.wrist_correction_xyz_at(ee[:3, :3])
        camera_pose = matrix_to_pose(camera_matrix, frame=BASE_FRAME)
        camera = CameraObservation(
            rgb=frame.rgb,
            depth_m=frame.depth_m,
            intrinsics=frame.intrinsics,
            frame=CAMERA_FRAME,
            camera_pose=camera_pose,
            timestamp_s=frame.timestamp_s,
        )
        cameras = {self._env.config.camera.role: camera}
        scene_config = self._env.config.scene_camera
        scene_frame = self._env.capture_scene()
        if scene_config is not None and scene_frame is not None:
            # The scene camera's pose is static in the base frame, solved once
            # by scene_calibration.py. Centimetre-grade by construction: good
            # for attribution and video, not for grasp targets.
            cameras[scene_config.role] = CameraObservation(
                rgb=scene_frame.rgb,
                depth_m=scene_frame.depth_m,
                intrinsics=scene_frame.intrinsics,
                frame=SCENE_FRAME,
                camera_pose=Pose(
                    position=np.asarray(scene_config.pose_position, dtype=np.float64),
                    quaternion_wxyz=np.asarray(
                        scene_config.pose_quaternion_wxyz, dtype=np.float64
                    ),
                    frame=BASE_FRAME,
                ),
                timestamp_s=scene_frame.timestamp_s,
            )
        return Observation(
            cameras=cameras,
            robot_state=state,
            task_context=self._task_context,
            timestamp_s=time.time(),
        )

    # -- actuation ---------------------------------------------------------

    def _execute(self, waypoints: np.ndarray) -> ApiError | None:
        """One guarded batch to the plant; the typed failure when refused."""
        denied = self._motion_denied()
        if denied is not None:
            return denied
        try:
            self._env.execute_waypoints(waypoints)
        except ValueError as exc:
            return ApiError(
                code=ErrorCode.SAFETY_INTERLOCK,
                message=f"workspace guard refused the motion: {exc}",
                recoverable=True,
            )
        except Exception as exc:  # libfranka reflexes, network, limits
            return ApiError(
                code=ErrorCode.CONTROLLER_FAILED,
                message=f"{type(exc).__name__}: {exc}",
            )
        return None

    def step(self, action: RobotAction) -> StepResult:
        """One waypoint batch of exactly one waypoint, plus any gripper command."""
        if action.embodiment != EMBODIMENT:
            return StepResult(
                ok=False,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message=f"action embodiment {action.embodiment!r} is not {EMBODIMENT!r}",
                ),
            )
        try:
            command = action.arms[self.resolve_arm(next(iter(action.arms)))]
        except ValueError as exc:
            return StepResult(
                ok=False,
                error=ApiError(code=ErrorCode.INVALID_REQUEST, message=str(exc)),
            )
        error = self._execute(np.asarray(command.target).reshape(1, ARM_DOF))
        if error is None and command.gripper_position is not None:
            width = command.gripper_position * self._env.config.gripper_max_width_m
            if not self._env.set_gripper_width(width)["ok"]:
                error = ApiError(
                    code=ErrorCode.CONTROLLER_FAILED, message="gripper command failed"
                )
        if error is not None:
            return StepResult(ok=False, error=error)
        return StepResult(ok=True, observation=self.get_observation())

    def execute_trajectory(
        self, trajectory: Trajectory | SynchronizedTrajectory
    ) -> ExecutionResult:
        if isinstance(trajectory, SynchronizedTrajectory):
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.UNSUPPORTED,
                    message="franka_real has one arm; synchronized trajectories do not apply",
                ),
            )
        if trajectory.embodiment != EMBODIMENT:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message=(
                        f"trajectory embodiment {trajectory.embodiment!r} "
                        f"is not {EMBODIMENT!r}"
                    ),
                ),
            )
        error = self._execute(trajectory.joint_positions)
        if error is not None:
            return ExecutionResult(ok=False, steps_executed=0, error=error)
        diagnostics: dict[str, object] = {
            "retimed_by": "panda_py_motion_generator",
            "requested_dt_s": float(trajectory.dt_s),
        }
        if trajectory.gripper_positions is not None:
            # The generator retimes the batch, so per-waypoint gripper values
            # cannot be synchronized mid-flight; the final value is honored
            # after the motion and the record says so.
            final = float(trajectory.gripper_positions[-1])
            width = final * self._env.config.gripper_max_width_m
            diagnostics["gripper_applied"] = "final_waypoint_value_after_motion"
            if not self._env.set_gripper_width(width)["ok"]:
                return ExecutionResult(
                    ok=False,
                    steps_executed=int(trajectory.joint_positions.shape[0]),
                    error=ApiError(
                        code=ErrorCode.CONTROLLER_FAILED,
                        message="trajectory motion succeeded but the gripper command failed",
                    ),
                    diagnostics=diagnostics,
                )
        return ExecutionResult(
            ok=True,
            steps_executed=int(trajectory.joint_positions.shape[0]),
            final_observation=self.get_observation(),
            diagnostics=diagnostics,
        )

    def set_gripper(self, position: float, *, arm: str = ARM) -> ExecutionResult:
        try:
            self.resolve_arm(arm)
        except ValueError as exc:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(code=ErrorCode.INVALID_REQUEST, message=str(exc)),
            )
        denied = self._motion_denied()
        if denied is not None:
            return ExecutionResult(ok=False, steps_executed=0, error=denied)
        width = float(position) * self._env.config.gripper_max_width_m
        result = self._env.set_gripper_width(width)
        if not result["ok"]:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.CONTROLLER_FAILED,
                    message=f"gripper did not reach {width:.3f} m",
                ),
                diagnostics=result,
            )
        return ExecutionResult(
            ok=True,
            steps_executed=1,
            final_observation=self.get_observation(),
            diagnostics=result,
        )

    def set_grippers(self, positions: Mapping[str, float]) -> ExecutionResult:
        names = set(positions)
        if names != {ARM}:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message=f"franka_real has one arm, {ARM!r}; got {sorted(names)}",
                ),
            )
        return self.set_gripper(float(positions[ARM]))

    def go_home(self, *, arm: str = ARM) -> ExecutionResult:
        """Open the jaws and return to the profile's home joints."""
        try:
            self.resolve_arm(arm)
        except ValueError as exc:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(code=ErrorCode.INVALID_REQUEST, message=str(exc)),
            )
        denied = self._motion_denied()
        if denied is not None:
            return ExecutionResult(ok=False, steps_executed=0, error=denied)
        self._env.set_gripper_width(self._env.config.gripper_max_width_m)
        error = self._execute(
            np.asarray(self._env.config.home_joints, dtype=np.float64).reshape(1, ARM_DOF)
        )
        if error is not None:
            return ExecutionResult(ok=False, steps_executed=0, error=error)
        return ExecutionResult(
            ok=True, steps_executed=1, final_observation=self.get_observation()
        )

    def descend_to_contact(
        self,
        *,
        floor_z: float,
        speed_m_s: float | None = None,
        contact_force_n: float | None = None,
    ) -> ExecutionResult:
        """Lower the hand straight down until it meets something, or the floor.

        Exposed to programs as ``franka_real.descend_to_contact``. A place that
        trusts a camera-measured height is a gamble against the sum of depth
        bias, grasp seating and the table plane; this primitive makes the last
        centimetre closed-loop instead (see
        :meth:`RealFrankaEnv.descend_until_contact`). ``ok`` says the descent
        ran as commanded; whether it ended on contact or at the floor is in the
        diagnostics, because a floor stop under a held block is the caller's
        news to act on, not an execution failure.
        """
        denied = self._motion_denied()
        if denied is not None:
            return ExecutionResult(ok=False, steps_executed=0, error=denied)
        def descend(retry: bool = False) -> dict[str, Any]:
            options: dict[str, Any] = {}
            if retry:
                options["min_descent_m"] = 0.0  # finish what the reflex cut short, from wherever the hand is
            return self._env.descend_until_contact(
                floor_z=floor_z, speed_m_s=speed_m_s, contact_force_n=contact_force_n, **options
            )

        try:
            attempts = 0
            while True:
                try:
                    report = descend(retry=attempts > 0)
                    if attempts:
                        report["retried_after_reflex"] = attempts
                    break
                except Exception as exc:  # libfranka reflexes, network, limits
                    # A reflex mid-descent is transient (2026-11-19: the
                    # communication reflex aborted probes and placements
                    # alike, and the robot recovers within a second): recover
                    # and descend again from wherever the hand stopped, up to
                    # DESCENT_REFLEX_RETRIES times (calibration 0434: two in a
                    # row on one probe). Anything else is reported as before.
                    text = str(exc).lower()
                    # Only the communication reflex is retried; a contact
                    # reflex means the hand met something (run 0476).
                    if (
                        isinstance(exc, ValueError)
                        or "communication_constraints_violation" not in text
                        or any(word in text for word in ("cartesian_reflex", "joint_reflex", "self_collision"))
                    ):
                        raise
                    recover = getattr(self._env, "recover_from_reflex", None)
                    if not callable(recover) or attempts >= DESCENT_REFLEX_RETRIES:
                        raise
                    attempts += 1
                    recover()
        except ValueError as exc:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.SAFETY_INTERLOCK,
                    message=f"contact descent refused: {exc}",
                    recoverable=True,
                ),
            )
        except Exception as exc:  # libfranka reflexes, network, limits
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.CONTROLLER_FAILED, message=f"{type(exc).__name__}: {exc}"
                ),
            )
        return ExecutionResult(
            ok=True,
            steps_executed=1,
            final_observation=self.get_observation(),
            diagnostics=report,
        )

    def _policy_host(self) -> Any:
        if self._policy_client is None:
            from cap_harness.franka_real.policy import HttpPolicyClient

            self._policy_client = HttpPolicyClient()
        return self._policy_client

    def policy_chunk(
        self,
        *,
        policy: str = "current",
        steps: int = 10,
        speed_factor: float | None = None,
    ) -> ExecutionResult:
        """Let a learned policy move the arm for one chunk of its actions.

        Exposed to programs as ``franka_real.policy_chunk``. The policy host
        (see :mod:`cap_harness.franka_real.policy`) gets the measured joints,
        gripper width, end-effector position, external wrench and both camera
        images, and answers with a chunk of joint targets plus a gripper
        width. The first ``steps`` of that chunk go through the same guarded
        waypoint path as every program motion -- joint limits, the workspace
        box, Franka's own reflexes -- and the chunk's final gripper width is
        applied after the motion when it differs from the measured one. The
        policy proposes; the bench's bounds dispose. The record names the
        policy on every chunk so an episode says who drove it.
        """
        denied = self._motion_denied()
        if denied is not None:
            return ExecutionResult(ok=False, steps_executed=0, error=denied)
        try:
            steps = int(steps)
        except (TypeError, ValueError):
            steps = 0
        if not 1 <= steps <= POLICY_MAX_STEPS:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message=f"steps must be between 1 and {POLICY_MAX_STEPS}",
                ),
            )
        policy = str(policy)
        started = time.monotonic()
        try:
            observation = self.get_observation()
        except ValueError as exc:  # no frame, or a stalled stream
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(code=ErrorCode.EXECUTION_FAILED, message=str(exc)),
            )
        raw = self._env.get_observations()
        q = np.asarray(raw["joint_pos"], dtype=np.float64).reshape(ARM_DOF)
        width_now = float(raw["gripper_width_m"])
        ee_position = np.asarray(raw["ee_transform"], dtype=np.float64)[:3, 3]
        wrench = self._env.external_wrench()
        from cap_harness.franka_real.policy import encode_image_png

        images = {
            role: encode_image_png(camera.rgb) for role, camera in observation.cameras.items()
        }
        request = {
            "policy": policy,
            "state": [float(v) for v in q] + [width_now],
            "ee_position": [float(v) for v in ee_position],
            "force": [float(v) for v in wrench],
            "images": images,
        }
        host = self._policy_host()
        try:
            if policy not in self._policy_reset_done:
                host.reset(policy)
                self._policy_reset_done.add(policy)
            actions = np.asarray(host.act(request), dtype=np.float64)
        except Exception as exc:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.EXECUTION_FAILED,
                    message=f"policy host: {type(exc).__name__}: {str(exc)[:200]}",
                    recoverable=True,
                ),
            )
        inference_s = time.monotonic() - started
        if actions.ndim != 2 or actions.shape[1] != ARM_DOF + 1 or not np.all(np.isfinite(actions)):
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.EXECUTION_FAILED,
                    message=(
                        f"policy host returned an action chunk of shape {actions.shape}, "
                        f"not (n, {ARM_DOF + 1}) finite"
                    ),
                    recoverable=True,
                ),
            )
        chunk = actions[:steps]
        joints = chunk[:, :ARM_DOF]
        max_width = self._env.config.gripper_max_width_m
        widths = np.clip(chunk[:, ARM_DOF], 0.0, max_width)
        width = float(widths[-1])
        factor = float(speed_factor if speed_factor is not None else POLICY_SPEED_FACTOR)
        # The chunk runs as segments split at every gripper change, each change
        # applied where the policy put it: a 100-step chunk can descend, close
        # and lift, and the fingers must close at the bottom, not after the lift.
        # The whole chunk is guarded before anything moves.
        # The policy has no touch: it commands positions, and the program's last
        # centimetre is a contact descent it does not have. No policy waypoint
        # goes below POLICY_MIN_Z (run 0476: a chunk to 1.9 cm, the table at 1.7,
        # stopped by the robot's own contact reflex). The floor is a stop, as the
        # program's descent is: the chunk runs down to the last row above it and
        # ends there; a gripper change the policy makes later in the chunk is
        # applied at that point, so a descend-and-close becomes a close at the
        # floor, and the next chunk decides what follows (run 0488: refusing the
        # whole chunk left the policy no way to grasp at all).
        floored_at = None
        for row, q in enumerate(joints):
            if float(self._env.fk_matrix(q)[2, 3]) < POLICY_MIN_Z:
                floored_at = row
                break
        if floored_at == 0:
            # The hand is already at the floor and the chunk would take it
            # lower: no motion, but the gripper change the policy makes in it
            # is applied (run 0490: down to 2.7 cm, then every chunk refused
            # outright, so it could neither close nor lift).
            later = [float(w) for w in widths if abs(float(w) - width_now) > POLICY_GRIPPER_DEADBAND_M]
            moved = bool(later)
            if moved and not self._env.set_gripper_width(later[0])["ok"]:
                return ExecutionResult(
                    ok=False,
                    steps_executed=0,
                    error=ApiError(
                        code=ErrorCode.CONTROLLER_FAILED,
                        message="the gripper command at the policy's floor failed",
                    ),
                )
            self._env.note_event(
                "policy_chunk", policy=policy, steps=0, chunk_size=int(actions.shape[0]),
                width_m=round(later[0], 4) if moved else round(width_now, 4),
                speed_factor=0.0, segments=0, floored_at_step=0,
            )
            after = self._env.get_observations()
            ee_after = np.asarray(after["ee_transform"], dtype=np.float64)[:3, 3]
            return ExecutionResult(
                ok=True,
                steps_executed=0,
                diagnostics={
                    "policy": policy,
                    "chunk_size": int(actions.shape[0]),
                    "steps": 0,
                    "speed_factor": 0.0,
                    "gripper_width_m": round(later[0], 4) if moved else round(width_now, 4),
                    "gripper_moved": moved,
                    "gripper_moves": int(moved),
                    "segments": 0,
                    "floored_at_step": 0,
                    "gripper_width_after_m": round(float(after["gripper_width_m"]), 4),
                    "ee_after": [round(float(v), 4) for v in ee_after],
                    "inference_s": round(inference_s, 3),
                    "elapsed_s": round(time.monotonic() - started, 3),
                },
            )
        if floored_at is not None:
            later = [float(w) for w in widths[floored_at:]
                     if abs(float(w) - float(widths[floored_at - 1])) > POLICY_GRIPPER_DEADBAND_M]
            joints = joints[:floored_at]
            widths = widths[:floored_at].copy()
            if later:
                widths[-1] = later[0]
            width = float(widths[-1])
        refusal = self._env.guard_waypoints(joints)
        if refusal is not None:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.SAFETY_INTERLOCK,
                    message=f"workspace guard refused the policy's chunk: {refusal}",
                    recoverable=True,
                ),
            )
        segments: list[tuple[int, int, float | None]] = []
        start, current = 0, width_now
        for row in range(joints.shape[0]):
            if abs(float(widths[row]) - current) > POLICY_GRIPPER_DEADBAND_M:
                segments.append((start, row + 1, float(widths[row])))
                start, current = row + 1, float(widths[row])
        if start < joints.shape[0]:
            segments.append((start, joints.shape[0], None))
        self._env.note_event(
            "policy_chunk",
            policy=policy,
            steps=int(joints.shape[0]),
            chunk_size=int(actions.shape[0]),
            width_m=round(width, 4),
            speed_factor=factor,
            segments=len(segments),
        )
        gripper_moves = 0
        executed = 0
        for first, last, grip in segments:
            try:
                self._env.execute_waypoints(joints[first:last], speed_factor=factor)
            except ValueError as exc:
                return ExecutionResult(
                    ok=False,
                    steps_executed=executed,
                    error=ApiError(
                        code=ErrorCode.SAFETY_INTERLOCK,
                        message=f"workspace guard refused the policy's chunk: {exc}",
                        recoverable=True,
                    ),
                )
            except Exception as exc:  # libfranka reflexes, network, limits
                return ExecutionResult(
                    ok=False,
                    steps_executed=executed,
                    error=ApiError(
                        code=ErrorCode.CONTROLLER_FAILED, message=f"{type(exc).__name__}: {exc}"
                    ),
                )
            executed = last
            if grip is not None:
                gripper_moves += 1
                if not self._env.set_gripper_width(grip)["ok"]:
                    return ExecutionResult(
                        ok=False,
                        steps_executed=executed,
                        error=ApiError(
                            code=ErrorCode.CONTROLLER_FAILED,
                            message="the policy's chunk moved the arm but the gripper command failed",
                        ),
                    )
        gripper_moved = gripper_moves > 0
        after = self._env.get_observations()
        ee_after = np.asarray(after["ee_transform"], dtype=np.float64)[:3, 3]
        return ExecutionResult(
            ok=True,
            steps_executed=int(joints.shape[0]),
            diagnostics={
                "policy": policy,
                "chunk_size": int(actions.shape[0]),
                "steps": int(joints.shape[0]),
                "speed_factor": factor,
                "gripper_width_m": round(width, 4),
                "gripper_moved": bool(gripper_moved),
                "gripper_moves": gripper_moves,
                "segments": len(segments),
                "floored_at_step": floored_at,
                "gripper_width_after_m": round(float(after["gripper_width_m"]), 4),
                "ee_after": [round(float(v), 4) for v in ee_after],
                "inference_s": round(inference_s, 3),
                "elapsed_s": round(time.monotonic() - started, 3),
            },
        )

    # -- verdicts and metadata ---------------------------------------------

    def check_success(self) -> bool:
        """Measure the task's postcondition; record the evidence; be conservative.

        A fresh observation is taken (no motion), the task's predicate from
        :mod:`~cap_harness.franka_real.success` runs on it, and the full report
        -- which camera testified, what it measured, which criteria passed --
        is written beside the run's logs. The protocol wants a bool, so an
        unknown verdict (no predicate, or the scene could not be measured)
        returns False; the report distinguishes "failed" from "could not tell".
        """
        from cap_harness.franka_real.success import evaluate_success

        verdict: Optional[bool]
        if self._task_context is None:
            verdict, report = None, {"verdict": None, "note": "no task bound"}
        else:
            try:
                observation = self.get_observation()
            except Exception as exc:
                verdict = None
                report = {
                    "task": self._task_context.task_name,
                    "note": f"final observation failed: {type(exc).__name__}: {exc}",
                }
            else:
                verdict, report = evaluate_success(
                    self._task_context.task_name, observation, self._env.config
                )
        report["verdict"] = verdict
        self.last_success_report = report
        root = getattr(self._run_observer, "root", None)
        if root is not None:
            try:
                import json
                from pathlib import Path

                logs = Path(root) / "logs"
                logs.mkdir(parents=True, exist_ok=True)
                (logs / "success-report.json").write_text(
                    json.dumps(report, indent=1, default=str), encoding="utf-8"
                )
            except Exception:
                pass  # the verdict stands even when the evidence cannot be filed
        return bool(verdict)

    def get_task_metadata(self) -> dict[str, Any]:
        context = self._task_context
        return {
            "embodiment": EMBODIMENT,
            "station": self._env.config.station,
            "arms": [ARM],
            "dof": ARM_DOF,
            "cameras": [self._env.config.camera.role]
            + ([self._env.config.scene_camera.role] if self._env.config.scene_camera else []),
            "language": "" if context is None else context.language,
            "scene_reset": ""
            if context is None
            else str(context.metadata.get("scene_reset", "")),
            "goal": {} if context is None else dict(context.metadata.get("goal", {})),
        }

    def get_controller_metadata(self) -> dict[str, Any]:
        config = self._env.config
        return {
            "control_frequency_hz": self.control_frequency,
            "control_period_s": self.control_period_s,
            "arm_names": [ARM],
            "controller": "panda_py_motion_generator",
            "action_lower_bounds": list(config.joint_limits_lower),
            "action_upper_bounds": list(config.joint_limits_upper),
            "workspace_box": {
                "x": list(config.workspace.x),
                "y": list(config.workspace.y),
                "z": list(config.workspace.z),
            },
            "physical_motion_authorized": self._allow_physical_motion,
            "motions_executed": self._env.motions_executed,
            "wrist_correction_hand_xyz": list(self._wrist_correction["offset_hand_xyz"]),
            "wrist_correction_samples": int(self._wrist_correction.get("samples", 0)),
            "wrist_level": dict(self.last_level_report),
        }

    def close(self) -> None:
        self.finish_episode_recording()
        self._env.close()


def _fit_table_plane(
    depth: np.ndarray, intrinsics: np.ndarray, camera_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """The dominant plane of a depth image in the base frame.

    Returns (unit normal with positive z, a point on the plane, inlier
    fraction), or None when there is too little depth. Successive least-squares
    refits with tightening inlier bands: the table is most of a survey look,
    so the first fit lands near it and the bands shed the blocks and the
    basket.
    """
    if depth.ndim != 2 or intrinsics.shape != (3, 3):
        return None
    # ~60 rows of samples whatever the resolution: stride 8 at 480 lines.
    stride = max(1, int(round(min(depth.shape) / (480.0 / PLANE_STRIDE))))
    sub = depth[::stride, ::stride]
    rows, cols = np.mgrid[0 : sub.shape[0], 0 : sub.shape[1]]
    u = cols.astype(np.float64) * stride
    v = rows.astype(np.float64) * stride
    valid = np.isfinite(sub) & (sub > 0.1) & (sub < 1.5)
    if int(valid.sum()) < 200:
        return None
    d = sub[valid]
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    camera_points = np.stack([(u[valid] - cx) / fx * d, (v[valid] - cy) / fy * d, d], axis=1)
    # No BLAS below this line, on purpose: a matrix product here deadlocked
    # against the episode recorder's threads (OpenBLAS and Python threads,
    # the old story), and this runs inside every look on the live arm.
    rotation = camera_matrix[:3, :3]
    points = np.einsum("ij,nj->ni", rotation, camera_points) + camera_matrix[:3, 3]
    # Seed by consensus, not least squares: from a survey height the floor
    # beyond the table edge and the robot base are in view, and one
    # least-squares plane over everything landed 39 degrees off with 8 %
    # inliers (live, 2026-11-17). Three-point samples, the plane most points
    # agree with wins, then the refits tighten it.
    generator = np.random.default_rng(0)
    best_inside = None
    best_count = 0
    for _ in range(PLANE_RANSAC_TRIALS):
        trio = points[generator.choice(points.shape[0], 3, replace=False)]
        candidate = np.cross(trio[1] - trio[0], trio[2] - trio[0])
        length = float(np.linalg.norm(candidate))
        if length < 1e-9:
            continue
        candidate = candidate / length
        distance = np.abs(np.einsum("nj,j->n", points - trio[0], candidate))
        inside = distance < PLANE_TOLERANCES_M[0]
        count = int(inside.sum())
        if count > best_count:
            best_count, best_inside = count, inside
    if best_inside is None or best_count < 50:
        return None
    selected = points[best_inside]
    normal = np.array([0.0, 0.0, 1.0])
    centroid = points.mean(axis=0)
    for tolerance in PLANE_TOLERANCES_M:
        centroid = selected.mean(axis=0)
        normal = _smallest_axis(selected - centroid)
        if normal[2] < 0:
            normal = -normal
        distance = np.abs(np.einsum("nj,j->n", points - centroid, normal))
        inside = distance < tolerance
        if int(inside.sum()) < 50:
            return None
        selected = points[inside]
    return normal, centroid, float(selected.shape[0]) / float(points.shape[0])


def _smallest_axis(centred: np.ndarray) -> np.ndarray:
    """Unit direction of least variance of centred (N, 3) points, BLAS-free.

    The 3x3 covariance by elementwise sums, then power iteration on
    (trace * I - C), whose dominant eigenvector is C's least.
    """
    covariance = np.einsum("ni,nj->ij", centred, centred) / max(centred.shape[0], 1)
    shifted = np.trace(covariance) * np.eye(3) - covariance
    vector = np.array([0.0, 0.0, 1.0])
    for _ in range(60):
        candidate = np.einsum("ij,j->i", shifted, vector)
        norm = float(np.linalg.norm(candidate))
        if norm < 1e-12:
            break
        candidate = candidate / norm
        if float(np.abs(candidate @ vector)) > 1.0 - 1e-12:
            vector = candidate
            break
        vector = candidate
    return vector / float(np.linalg.norm(vector))


def _rotation_taking(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The smallest rotation matrix taking unit vector ``source`` onto ``target``."""
    s = np.asarray(source, dtype=np.float64) / np.linalg.norm(source)
    t = np.asarray(target, dtype=np.float64) / np.linalg.norm(target)
    axis = np.cross(s, t)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.clip(s @ t, -1.0, 1.0))
    if sine < 1e-9:
        return np.eye(3) if cosine > 0 else -np.eye(3)
    axis = axis / sine
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + sine * k + (1.0 - cosine) * (k @ k)


__all__ = ["ARM", "EMBODIMENT", "FrankaRealAdapter"]
