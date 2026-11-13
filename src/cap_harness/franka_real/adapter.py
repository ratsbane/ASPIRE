"""EnvironmentAdapter for a single-arm Franka bench.

The shape follows :class:`cap_harness.yam_real.adapter.YamRealAdapter` with the
same two commitments and one simplification. The commitments: anything that can
command the arm is refused unless ``allow_physical_motion`` was set at
construction (one gate, every command path calls it first), and construction
performs no motion. The simplification: this plant does not stream per-tick
commands -- every motion is a guarded waypoint batch through the Franka motion
generator -- so ``step`` is a one-waypoint batch and a trajectory's ``dt_s``
is advisory, recorded as such.

``check_success`` returns False always: like the YAM, no task success
predicate runs on this hardware *yet*. The bench this port grew on verifies
stacking tasks by touch (a probe finds the pile one block taller), and wiring
that in as a per-task predicate is the planned next step; until it lands, a
run's success is judged from its artifacts by whoever ran it.
"""

from __future__ import annotations

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
JOINT_NAMES = tuple(f"panda_joint{index + 1}" for index in range(ARM_DOF))
CAMERA_WAIT_S = 2.0


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
    ) -> None:
        self._env = env
        self._registry = registry or FrankaRealTaskRegistry()
        self._run_observer = run_observer
        self._allow_physical_motion = bool(allow_physical_motion)
        self._task_context = task_context
        self._hand_eye = pose_to_matrix(
            Pose(
                position=np.asarray(env.config.camera.hand_eye_position, dtype=np.float64),
                quaternion_wxyz=np.asarray(
                    env.config.camera.hand_eye_quaternion_wxyz, dtype=np.float64
                ),
                frame="ee",
            )
        )

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
                metadata={"scene_reset": metadata.scene_reset, "seed": int(seed)},
            )
        elif self._task_context is None:
            raise ValueError("reset needs a franka_real task name or metadata")
        return self.get_observation()

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

    def get_observation(self) -> Observation:
        state = self.get_robot_state()
        deadline = time.monotonic() + CAMERA_WAIT_S
        frame = self._env.capture()
        while frame is None and time.monotonic() < deadline:
            time.sleep(0.05)
            frame = self._env.capture()
        if frame is None:
            raise ValueError(
                "the wrist camera produced no frame; the Observation contract "
                "requires one, and a silent placeholder would be mistaken for a scene"
            )
        ee = pose_to_matrix(state.end_effector_poses[ARM])
        camera_pose = matrix_to_pose(ee @ self._hand_eye, frame=BASE_FRAME)
        camera = CameraObservation(
            rgb=frame.rgb,
            depth_m=frame.depth_m,
            intrinsics=frame.intrinsics,
            frame=BASE_FRAME,
            camera_pose=camera_pose,
            timestamp_s=frame.timestamp_s,
        )
        return Observation(
            cameras={self._env.config.camera.role: camera},
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

    # -- verdicts and metadata ---------------------------------------------

    def check_success(self) -> bool:
        """No task success predicate runs on this hardware yet; see module docstring."""
        return False

    def get_task_metadata(self) -> dict[str, Any]:
        context = self._task_context
        return {
            "embodiment": EMBODIMENT,
            "station": self._env.config.station,
            "arms": [ARM],
            "dof": ARM_DOF,
            "cameras": [self._env.config.camera.role],
            "language": "" if context is None else context.language,
            "scene_reset": ""
            if context is None
            else str(context.metadata.get("scene_reset", "")),
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
        }

    def close(self) -> None:
        self._env.close()


__all__ = ["ARM", "EMBODIMENT", "FrankaRealAdapter"]
