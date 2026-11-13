"""In-process IK for the Franka Panda through panda_py's analytical solver.

panda_py ships Franka's closed-form 7-DOF solution parameterized by the wrist
joint, so a station needs no GPU planning service to resolve a pose: the
solver runs in-process and the answer is checked against the same forward
kinematics the plant's workspace guard uses. A pose the solver cannot reach,
or reaches only outside the joint limits, is a typed ``IK_FAILED`` --
never a substituted nearby pose.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from cap_harness.contracts import IKResult, Pose, RobotState
from cap_harness.errors import ApiError, ErrorCode

ARM_DOF = 7
POSITION_TOLERANCE_M = 0.005

PANDA_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
PANDA_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


class FrankaIKProvider:
    """IKProvider over ``panda_py.ik``, seeded from the measured state."""

    def __init__(
        self,
        ik: Optional[Callable[..., Any]] = None,
        fk: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        if ik is None or fk is None:
            import panda_py

            ik = ik or panda_py.ik
            fk = fk or panda_py.fk
        self._ik = ik
        self._fk = fk

    def solve_ik(
        self, target_pose: Pose, robot_state: RobotState, *, arm: str = "primary"
    ) -> IKResult:
        def failure(code: ErrorCode, message: str) -> IKResult:
            return IKResult(
                ok=False,
                arm=arm,
                error=ApiError(code=code, message=message),
                embodiment=robot_state.embodiment,
            )

        if arm not in robot_state.joint_positions:
            return failure(ErrorCode.INVALID_REQUEST, f"robot state has no arm {arm!r}")
        seed = np.asarray(robot_state.joint_positions[arm], dtype=np.float64)
        if seed.shape != (ARM_DOF,):
            return failure(
                ErrorCode.INVALID_REQUEST, "the Franka solver needs a 7-DOF seed state"
            )
        if target_pose.frame != robot_state.base_frame:
            return failure(
                ErrorCode.INVALID_REQUEST,
                f"target frame {target_pose.frame!r} is not the robot base frame "
                f"{robot_state.base_frame!r}",
            )
        matrix = np.asarray(target_pose.as_matrix(), dtype=np.float64)

        # q_7 parameterizes Franka's redundancy, and the right value depends on
        # the requested orientation: a down-grasp turned 48 degrees needs the
        # wrist turned with it, so pinning q_7 at the seed's value makes every
        # rotated pose "unreachable" (found live, first stack attempt). Sweep
        # the joint's range and keep the valid solution nearest the seed.
        best: np.ndarray | None = None
        best_cost = float("inf")
        for q7 in np.arange(PANDA_LOWER[6] + 0.05, PANDA_UPPER[6] - 0.05 + 1e-9, 0.1):
            try:
                candidate = np.asarray(
                    self._ik(matrix, q_init=seed, q_7=float(q7)), dtype=np.float64
                ).reshape(ARM_DOF)
            except TypeError:
                try:
                    candidate = np.asarray(self._ik(matrix), dtype=np.float64).reshape(ARM_DOF)
                except Exception as exc:
                    return failure(ErrorCode.IK_FAILED, f"{type(exc).__name__}: {exc}")
            except Exception as exc:
                return failure(ErrorCode.IK_FAILED, f"{type(exc).__name__}: {exc}")
            if not np.all(np.isfinite(candidate)):
                continue
            if np.any(candidate < PANDA_LOWER - 1e-9) or np.any(candidate > PANDA_UPPER + 1e-9):
                continue
            transform = np.asarray(self._fk(candidate))
            if float(np.linalg.norm(transform[:3, 3] - matrix[:3, 3])) > POSITION_TOLERANCE_M:
                continue
            # Orientation must survive too: a pose is position AND rotation.
            alignment = float(np.trace(transform[:3, :3].T @ matrix[:3, :3]))
            if alignment < 1.0 + 2.0 * np.cos(np.deg2rad(3.0)):
                continue
            cost = float(np.linalg.norm(candidate - seed))
            if cost < best_cost:
                best, best_cost = candidate, cost
        if best is None:
            return failure(
                ErrorCode.IK_UNREACHABLE,
                "no in-limit solution at any wrist angle reaches the requested pose",
            )
        reached = np.asarray(self._fk(best))[:3, 3]
        residual = float(np.linalg.norm(reached - matrix[:3, 3]))
        return IKResult(
            ok=True,
            joint_positions=best,
            arm=arm,
            diagnostics={
                "position_residual_mm": round(residual * 1000, 3),
                "q7": round(float(best[6]), 4),
                "seed_distance_rad": round(best_cost, 4),
            },
            embodiment=robot_state.embodiment,
        )


__all__ = ["FrankaIKProvider"]
