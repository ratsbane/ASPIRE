"""Off-robot tests for the single-arm Franka embodiment.

The fakes model what matters to the contracts: a 7-DOF arm whose forward
kinematics is a simple injective map into the workspace, a gripper whose close
is a grasp and whose open is a move, and the synthetic camera the YAM station
already uses for off-robot runs. What is under test is the adapter boundary --
the motion interlock, the workspace guard, the typed results -- not Franka
dynamics.
"""

from __future__ import annotations

import numpy as np
import pytest

from cap_harness.contracts import ArmCommand, Pose, RobotAction, RobotState, Trajectory
from cap_harness.errors import ErrorCode
from cap_harness.franka_real.adapter import FrankaRealAdapter
from cap_harness.franka_real.config import FrankaStationConfig
from cap_harness.franka_real.env import ARM_DOF, RealFrankaEnv
from cap_harness.franka_real.registry import FrankaRealTaskRegistry
from cap_harness.providers.franka_kinematics import FrankaIKProvider
from cap_harness.registry import FRANKA_REAL_PUBLIC_TOOL_NAMES, ToolRegistry
from cap_harness.yam_real.cameras import SyntheticCamera

HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


def fake_fk(q: np.ndarray) -> np.ndarray:
    """An injective toy FK: home lands mid-workspace, joints move it linearly."""
    q = np.asarray(q, dtype=np.float64)
    offset = q - HOME
    transform = np.eye(4)
    transform[:3, 3] = [
        0.45 + 0.1 * float(offset[0]),
        0.0 + 0.1 * float(offset[1]),
        0.30 + 0.1 * float(offset[3]),
    ]
    return transform


class FakeState:
    def __init__(self, q):
        self.q = list(q)
        self.dq = [0.0] * ARM_DOF


class FakePanda:
    def __init__(self):
        self.q = np.array(HOME)
        self.batches = []

    def get_state(self):
        return FakeState(self.q)

    def move_to_joint_position(self, waypoints, speed_factor=0.1):
        self.batches.append((np.asarray(waypoints, dtype=np.float64), speed_factor))
        self.q = np.asarray(waypoints, dtype=np.float64).reshape(-1, ARM_DOF)[-1]
        return True


class FakeGripperState:
    def __init__(self, width):
        self.width = width


class FakeGripper:
    def __init__(self):
        self.width = 0.08
        self.grasps = []

    def read_once(self):
        return FakeGripperState(self.width)

    def move(self, width, speed):
        self.width = float(width)
        return True

    def grasp(self, width, speed, force, epsilon_inner, epsilon_outer):
        self.grasps.append({"width": width, "force": force})
        self.width = max(float(width), 0.028)  # a block stops the fingers
        return True


def build(allow_motion: bool = True):
    env = RealFrankaEnv(
        FrankaStationConfig(),
        panda=FakePanda(),
        gripper=FakeGripper(),
        camera=SyntheticCamera(),
        fk=fake_fk,
    )
    adapter = FrankaRealAdapter(env, allow_physical_motion=allow_motion)
    adapter.reset("stack_blocks", 0)
    return env, adapter


def home_trajectory(adapter, target):
    state = adapter.get_robot_state()
    return Trajectory(
        joint_positions=np.asarray(target, dtype=np.float64).reshape(-1, ARM_DOF),
        dt_s=0.05,
        joint_names=state.joint_names["primary"],
        planner="joint_interpolation",
        collision_aware=False,
        expected_start=state,
        embodiment="franka_real",
    )


def test_observation_carries_a_posed_wrist_camera():
    _, adapter = build()
    observation = adapter.get_observation()
    assert set(observation.cameras) == {"wrist"}
    camera = observation.cameras["wrist"]
    # The camera's own optical frame is named; its pose maps it to base. Both
    # set to "base" would silently skip point-cloud transforms (found live).
    assert camera.frame == "wrist_camera"
    assert camera.camera_pose.frame == "base"
    # The camera rides the hand: its pose must sit near the FK answer for the hand.
    hand = adapter.get_robot_state().end_effector_poses["primary"].position
    assert np.linalg.norm(camera.camera_pose.position - hand) < 0.15
    assert observation.robot_state.embodiment == "franka_real"


def test_unauthorized_motion_is_a_typed_interlock_not_an_exception():
    env, adapter = build(allow_motion=False)
    result = adapter.execute_trajectory(home_trajectory(adapter, HOME))
    assert result.ok is False
    assert result.error.code is ErrorCode.SAFETY_INTERLOCK
    assert env._panda.batches == []
    gripper = adapter.set_gripper(0.0)
    assert gripper.ok is False and gripper.error.code is ErrorCode.SAFETY_INTERLOCK
    assert adapter.get_observation() is not None  # observing stays allowed


def test_workspace_guard_refuses_an_out_of_box_waypoint():
    env, adapter = build()
    runaway = HOME.copy()
    runaway[1] += 4.0  # beyond the joint limit AND far outside the box
    result = adapter.execute_trajectory(home_trajectory(adapter, runaway))
    assert result.ok is False
    assert result.error.code is ErrorCode.SAFETY_INTERLOCK
    assert env._panda.batches == []


def test_a_trajectory_executes_through_the_single_chokepoint():
    env, adapter = build()
    target = HOME.copy()
    target[0] += 0.2
    result = adapter.execute_trajectory(home_trajectory(adapter, target))
    assert result.ok is True and result.steps_executed == 1
    assert env.motions_executed == 1
    assert np.allclose(env._panda.q, target)
    assert result.diagnostics["retimed_by"] == "panda_py_motion_generator"


def test_step_moves_one_waypoint_and_the_gripper():
    env, adapter = build()
    target = HOME.copy()
    target[3] += 0.1
    action = RobotAction(
        arms={
            "primary": ArmCommand(
                mode="joint_position",
                target=target,
                gripper_position=0.3,
                embodiment="franka_real",
            )
        }
    )
    result = adapter.step(action)
    assert result.ok is True
    assert env._gripper.grasps, "closing from open must be a force grasp, not a move"


def test_closing_on_a_block_still_reports_success():
    """The fingers stop at the block, not the commanded width; that is a grasp."""
    env, adapter = build()
    result = adapter.set_gripper(0.1)
    assert result.ok is True
    assert env._gripper.width == pytest.approx(0.028)


def test_go_home_opens_and_returns_to_the_profile_home():
    env, adapter = build()
    env._panda.q = HOME + 0.1
    result = adapter.go_home()
    assert result.ok is True
    assert np.allclose(env._panda.q, HOME)
    assert env._gripper.width == pytest.approx(0.08)


def test_the_registry_resolves_only_supported_tasks():
    registry = FrankaRealTaskRegistry()
    metadata = registry.resolve("stack_blocks", 0)
    assert metadata.scene_reset == "skill"
    with pytest.raises(ValueError):
        registry.resolve("fold_laundry", 0)


def test_extension_tools_register_under_the_approved_prefix():
    from cap_harness.api import CapApi

    _, adapter = build()
    api = CapApi(adapter, default_camera="wrist")
    registry = api.register_tools(
        ToolRegistry(public_extension_allowlist=FRANKA_REAL_PUBLIC_TOOL_NAMES)
    )
    names = {spec.name for spec in registry.all_specs(public_only=True)}
    assert "franka_real.get_task_metadata" in names
    assert "franka_real.get_controller_metadata" in names
    assert "move_to_pose" in names


def test_ik_provider_round_trips_through_the_same_fk():
    def toy_ik(matrix, q_init=None, q_7=None):
        """Invert fake_fk for the three coupled joints."""
        q = HOME.copy()
        q[0] = (matrix[0, 3] - 0.45) / 0.1 + HOME[0]
        q[1] = (matrix[1, 3] - 0.0) / 0.1 + HOME[1]
        q[3] = (matrix[2, 3] - 0.30) / 0.1 + HOME[3]
        return q

    provider = FrankaIKProvider(ik=toy_ik, fk=fake_fk)
    _, adapter = build()
    state = adapter.get_robot_state()
    # The toy FK has identity rotation, so the target must too.
    target = Pose(
        position=np.array([0.5, 0.05, 0.25]),
        quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        frame="base",
    )
    result = provider.solve_ik(target, state)
    assert result.ok is True
    assert np.allclose(fake_fk(result.joint_positions)[:3, 3], target.position)

    unreachable = Pose(
        position=np.array([0.5, 20.0, 0.25]),
        quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        frame="base",
    )
    refused = provider.solve_ik(unreachable, state)
    assert refused.ok is False
    assert refused.error.code in (ErrorCode.IK_UNREACHABLE, ErrorCode.IK_FAILED)


def test_real_solver_reaches_a_yawed_down_grasp():
    """Regression for the first live stack attempt: a down-grasp at -48 deg
    yaw was 'unreachable' while q_7 stayed pinned at the seed's value."""
    panda_py = pytest.importorskip("panda_py")
    import math

    provider = FrankaIKProvider(ik=panda_py.ik, fk=panda_py.fk)
    state = RobotState(
        joint_positions={"primary": HOME},
        joint_velocities={"primary": np.zeros(ARM_DOF)},
        end_effector_poses={
            "primary": Pose(
                position=np.asarray(panda_py.fk(HOME))[:3, 3],
                quaternion_wxyz=np.array([0.0, 1.0, 0.0, 0.0]),
                frame="base",
            )
        },
        gripper_positions={"primary": 1.0},
        base_frame="base",
        embodiment="franka_real",
    )
    yaw = math.radians(-48.0)
    target = Pose(
        position=np.array([0.437, 0.002, 0.107]),
        quaternion_wxyz=np.array(
            [0.0, math.cos(yaw / 2.0), math.sin(yaw / 2.0), 0.0]
        ),
        frame="base",
    )
    result = provider.solve_ik(target, state)
    assert result.ok is True, result.error and result.error.message
    reached = np.asarray(panda_py.fk(result.joint_positions))
    assert np.linalg.norm(reached[:3, 3] - target.position) < 0.005
    # The achieved orientation matches the yawed grasp, not just the position.
    alignment = float(np.trace(reached[:3, :3].T @ target.as_matrix()[:3, :3]))
    assert alignment > 1.0 + 2.0 * np.cos(np.deg2rad(3.0))


def test_colorblob_provider_segments_a_blue_region():
    from cap_harness.providers.colorblob import ColorBlobSegmentationProvider
    from cap_harness.contracts import CameraObservation, Pose

    rgb = np.full((60, 80, 3), 120, dtype=np.uint8)
    rgb[20:40, 30:60] = (40, 50, 200)   # a blue patch, 600 px
    camera = CameraObservation(
        rgb=rgb,
        depth_m=np.full((60, 80), 0.5),
        intrinsics=np.array([[100.0, 0, 40], [0, 100.0, 30], [0, 0, 1]]),
        frame="base",
        camera_pose=Pose(
            position=np.zeros(3),
            quaternion_wxyz=np.array([1.0, 0, 0, 0]),
            frame="base",
        ),
    )
    provider = ColorBlobSegmentationProvider()
    found = provider.segment_text(camera, "blue block")
    assert found.ok is True
    mask = found.segmentations[0].mask
    assert mask[30, 45] and not mask[5, 5]
    assert found.segmentations[0].score <= 0.8  # a heuristic never claims certainty

    missing = provider.segment_text(camera, "red block")
    assert missing.ok is False

    unknown = provider.segment_text(camera, "the mug")
    assert unknown.ok is False
