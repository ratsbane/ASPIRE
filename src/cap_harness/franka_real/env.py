"""The Franka plant: one arm, one gripper, one wrist camera, one path that moves.

``RealFrankaEnv`` mirrors :class:`cap_harness.yam_real.env.RealYamEnv`'s rule:
:meth:`execute_waypoints` is the **only** way anything in this package commands
the arm. ``move_home`` and ``step``-sized moves are built on it rather than
beside it, so every recorded motion went through the same guard and the same
motion generator.

The clients are injected rather than constructed here -- ``panda`` and
``gripper`` are panda_py objects on hardware and fakes in tests -- and ``fk``
is a callable returning the 4x4 EE transform for a joint vector, so the
workspace guard and the published EE pose use one kinematic story. On hardware
that callable is ``panda_py.fk``; tests inject a model.

**The guard is not optional.** Every waypoint's forward kinematics must land
inside the station's workspace box before anything is sent. The bench this
port grew on enforces the same box in its own skills; a generated program that
asks for a pose outside it gets a refusal, not a clipped motion.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import numpy as np

from cap_harness.franka_real.config import FrankaStationConfig
from cap_harness.yam_real.cameras import Camera, CameraFrame

ARM_DOF = 7


class RealFrankaEnv:
    """Single-arm Franka plant behind one actuation chokepoint."""

    def __init__(
        self,
        config: FrankaStationConfig,
        *,
        panda: Any,
        gripper: Any,
        camera: Optional[Camera],
        fk: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self.config = config
        self._panda = panda
        self._gripper = gripper
        self._camera = camera
        self._fk = fk
        self._closed = False
        self.motions_executed = 0
        """Count of waypoint batches actually sent to the arm, for provenance."""

    # -- observation -------------------------------------------------------

    def fk_matrix(self, joint_positions: np.ndarray) -> np.ndarray:
        """4x4 EE transform in the base frame for one joint vector."""
        return np.asarray(self._fk(np.asarray(joint_positions, dtype=np.float64)))

    def get_observations(self) -> dict[str, np.ndarray]:
        """Measured joints, gripper, and the EE transform from those joints.

        The EE pose comes from forward kinematics of the measured joints -- the
        same function the guard uses -- rather than from the controller's own
        pose readout, so what the record says about the hand and what the guard
        checked about the hand can never be two different kinematic stories.
        """
        state = self._panda.get_state()
        q = np.asarray(state.q, dtype=np.float64).reshape(ARM_DOF)
        dq = np.asarray(state.dq, dtype=np.float64).reshape(ARM_DOF)
        width = float(self._gripper.read_once().width)
        return {
            "joint_pos": q,
            "joint_vel": dq,
            "gripper_width_m": width,
            "ee_transform": self.fk_matrix(q),
        }

    def capture(self) -> Optional[CameraFrame]:
        """The wrist camera's most recent frame, or None without a camera."""
        if self._camera is None:
            return None
        return self._camera.read()

    # -- the chokepoint ----------------------------------------------------

    def guard_waypoints(self, waypoints: np.ndarray) -> Optional[str]:
        """Why this batch must be refused, or None when every waypoint is safe."""
        points = np.asarray(waypoints, dtype=np.float64).reshape(-1, ARM_DOF)
        lower = np.asarray(self.config.joint_limits_lower)
        upper = np.asarray(self.config.joint_limits_upper)
        for index, q in enumerate(points):
            if not np.all(np.isfinite(q)):
                return f"waypoint {index} is not finite"
            if np.any(q < lower - 1e-9) or np.any(q > upper + 1e-9):
                return f"waypoint {index} exceeds a joint limit"
            position = self.fk_matrix(q)[:3, 3]
            if not self.config.workspace.contains(position):
                return (
                    f"waypoint {index} puts the end effector at "
                    f"{np.round(position, 3).tolist()}, outside the workspace box"
                )
        return None

    def execute_waypoints(
        self, waypoints: np.ndarray, *, speed_factor: Optional[float] = None
    ) -> dict[str, Any]:
        """Send one guarded waypoint batch through the panda_py motion generator.

        The generator retimes the batch under the robot's own limits, so the
        trajectory's requested ``dt_s`` is advisory here and is recorded as
        such. That trade is deliberate: a hobbyist bench gets Franka's reflexes
        and jerk limits on every motion instead of a hand-rolled 100 Hz loop.
        """
        points = np.asarray(waypoints, dtype=np.float64).reshape(-1, ARM_DOF)
        refusal = self.guard_waypoints(points)
        if refusal is not None:
            raise ValueError(refusal)
        factor = float(speed_factor if speed_factor is not None else self.config.speed_factor)
        started = time.monotonic()
        self._panda.move_to_joints([row for row in points], speed_factor=factor)
        self.motions_executed += 1
        return {
            "waypoints": int(points.shape[0]),
            "speed_factor": factor,
            "elapsed_s": round(time.monotonic() - started, 3),
            "retimed_by": "panda_py_motion_generator",
        }

    def move_home(self, *, speed_factor: Optional[float] = None) -> dict[str, Any]:
        return self.execute_waypoints(
            np.asarray(self.config.home_joints, dtype=np.float64).reshape(1, ARM_DOF),
            speed_factor=speed_factor,
        )

    # -- gripper -----------------------------------------------------------

    def set_gripper_width(self, width_m: float) -> dict[str, Any]:
        """Width move when opening, force-closure when closing onto something.

        The Franka gripper's plain ``move`` reports failure when an object
        stops the fingers, which is exactly the case a manipulation program
        means by "close". A command below the current width therefore grasps
        with the station's configured force and generous epsilons: holding an
        object of any width within the commanded travel counts as success,
        matching how every other embodiment's normalized gripper behaves.
        """
        width = float(np.clip(width_m, 0.0, self.config.gripper_max_width_m))
        current = float(self._gripper.read_once().width)
        if width < current - 1e-4:
            grasped = bool(
                self._gripper.grasp(
                    width,
                    speed=0.05,
                    force=float(self.config.gripper_force_n),
                    epsilon_inner=self.config.gripper_max_width_m,
                    epsilon_outer=self.config.gripper_max_width_m,
                )
            )
        else:
            grasped = bool(self._gripper.move(width=width, speed=0.05))
        return {
            "commanded_width_m": round(width, 4),
            "measured_width_m": round(float(self._gripper.read_once().width), 4),
            "ok": grasped,
        }

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._camera is not None:
            try:
                self._camera.close()
            except Exception:
                pass


__all__ = ["ARM_DOF", "RealFrankaEnv"]
