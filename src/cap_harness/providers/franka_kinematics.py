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
PANDA_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
IK_LIMIT_MARGIN_RAD = 0.08  # a solution closer than this to any joint stop is a last resort
IK_CLEAR_SWING_RAD = 2.0    # a clear solution farther than this from the seed is not worth the swing
"""The canonical elbow-up branch. The analytical solver returns the branch its
seed is in, and an arm that has wound into another branch over a long run
(live, run 0229: shoulder at -105 degrees, forearm turned back) can find no
in-limit solution there for poses it reached minutes earlier. A pose the
measured seed cannot reach is tried again from here before it is refused."""


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

        seeded_from = "measured"
        swept = self._sweep(matrix, seed)
        if isinstance(swept, IKResult):
            return swept
        best, best_cost = swept
        if best is None:
            seeded_from = "home"
            swept = self._sweep(matrix, PANDA_HOME)
            if isinstance(swept, IKResult):
                return swept
            best, best_cost = swept
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
                "seed_distance_rad": round(float(np.linalg.norm(best - seed)), 4),
                "seeded_from": seeded_from,
            },
            embodiment=robot_state.embodiment,
        )

    def _sweep(self, matrix: np.ndarray, seed: np.ndarray):
        """The valid solution nearest ``seed`` over the wrist joint's range.

        q_7 parameterizes Franka's redundancy, and the right value depends on
        the requested orientation: a down-grasp turned 48 degrees needs the
        wrist turned with it, so pinning q_7 at the seed's value makes every
        rotated pose "unreachable" (found live, first stack attempt).
        Returns ``(best, cost)`` with ``best`` None when nothing in the seed's
        branch is valid, or an ``IKResult`` failure when the solver itself
        raised.
        """
        best: np.ndarray | None = None
        best_cost = float("inf")
        cramped: np.ndarray | None = None  # the nearest solution hard against a stop
        cramped_cost = float("inf")
        for q7 in np.arange(PANDA_LOWER[6] + 0.05, PANDA_UPPER[6] - 0.05 + 1e-9, 0.1):
            try:
                candidate = np.asarray(
                    self._ik(matrix, q_init=seed, q_7=float(q7)), dtype=np.float64
                ).reshape(ARM_DOF)
            except TypeError:
                try:
                    candidate = np.asarray(self._ik(matrix), dtype=np.float64).reshape(ARM_DOF)
                except Exception as exc:
                    return IKResult(
                        ok=False,
                        error=ApiError(
                            code=ErrorCode.IK_FAILED, message=f"{type(exc).__name__}: {exc}"
                        ),
                    )
            except Exception as exc:
                return IKResult(
                    ok=False,
                    error=ApiError(
                        code=ErrorCode.IK_FAILED, message=f"{type(exc).__name__}: {exc}"
                    ),
                )
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
            # A solution hard against a joint stop is one the motion generator
            # will not converge at (run 0445: base and wrist 0.04 rad from
            # their limits, three refusals, no reflex). Prefer a solution that
            # keeps IK_LIMIT_MARGIN_RAD clear, as long as it lies within
            # IK_CLEAR_SWING_RAD of the seed: the clear alternative for 0445's
            # pose was the other base branch, 6.4 rad away, a swing the
            # workspace guard would refuse anyway.
            margin = float(np.min(np.minimum(candidate - PANDA_LOWER, PANDA_UPPER - candidate)))
            if margin < IK_LIMIT_MARGIN_RAD:
                if cost < cramped_cost:
                    cramped, cramped_cost = candidate, cost
                continue
            if cost < best_cost:
                best, best_cost = candidate, cost
        if best is None or (cramped is not None and best_cost > IK_CLEAR_SWING_RAD and cramped_cost < best_cost):
            return cramped, cramped_cost
        return best, best_cost


__all__ = ["FrankaIKProvider"]
