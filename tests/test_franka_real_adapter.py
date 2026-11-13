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


def test_a_straight_dense_path_decimates_to_its_endpoints():
    """The generator rests at every waypoint it is given (a 19-waypoint
    descent took ~2 minutes live), so colinear interior waypoints add stops
    and no information."""
    env, adapter = build()
    target = HOME.copy()
    target[3] += 0.5
    dense = np.linspace(HOME, target, 19)
    result = adapter.execute_trajectory(home_trajectory(adapter, dense))
    assert result.ok is True
    sent, _ = env._panda.batches[-1]
    assert sent.shape[0] == 2
    assert np.allclose(sent[0], HOME) and np.allclose(sent[-1], target)


def test_decimation_keeps_corners_and_stays_true_to_the_plan():
    env, adapter = build()
    mid = HOME.copy()
    mid[1] += 0.4
    end = HOME.copy()
    end[1] += 0.4
    end[3] += 0.4
    dense = np.vstack([np.linspace(HOME, mid, 10), np.linspace(mid, end, 10)[1:]])
    result = adapter.execute_trajectory(home_trajectory(adapter, dense))
    assert result.ok is True
    sent, _ = env._panda.batches[-1]
    assert 3 <= sent.shape[0] <= 4  # the corner survives, the straights do not
    # Fidelity: every planned waypoint lies within tolerance of the sent path.
    tolerance = env.config.waypoint_chord_tolerance_rad
    for q in dense:
        best = min(
            _point_to_segment(q, sent[i], sent[i + 1])
            for i in range(sent.shape[0] - 1)
        )
        assert best <= tolerance + 1e-9


def _point_to_segment(point, start, end):
    chord = end - start
    length_sq = float(chord @ chord)
    if length_sq < 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.clip((point - start) @ chord / length_sq, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * chord)))


def test_an_unprovable_chord_falls_back_to_the_dense_path():
    """Decimation re-guards the chords it executes in place of the planned
    waypoints; when that cannot be shown safe, slow and dense wins."""
    env, adapter = build()
    mid = HOME.copy()
    mid[1] += 0.4
    end = HOME.copy()
    end[1] += 0.4
    end[3] += 0.4
    dense = np.vstack([np.linspace(HOME, mid, 10), np.linspace(mid, end, 10)[1:]])
    planned = {tuple(np.round(q, 9)) for q in dense}
    real_guard = env.guard_waypoints

    def refuse_synthetic_samples(points):
        for q in np.asarray(points, dtype=np.float64).reshape(-1, ARM_DOF):
            if tuple(np.round(q, 9)) not in planned:
                return "synthetic chord sample refused"
        return real_guard(points)

    env.guard_waypoints = refuse_synthetic_samples
    result = adapter.execute_trajectory(home_trajectory(adapter, dense))
    assert result.ok is True
    sent, _ = env._panda.batches[-1]
    assert sent.shape[0] == dense.shape[0]


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


def test_recommanding_the_held_width_is_a_noop():
    """Planned trajectories carry per-waypoint gripper values, and after a
    carry the final value re-commands the width the fingers already hold.
    Live, libfranka failed that command while gripping ('libfranka gripper:
    Command failed!') and poisoned an otherwise successful motion."""
    env, _ = build()
    env._gripper.width = 0.028  # holding a block
    result = env.set_gripper_width(0.0285)
    assert result["ok"] is True
    assert result["noop"] == "already at width"
    assert env._gripper.grasps == []
    assert env._gripper.width == pytest.approx(0.028)


def test_reclosing_while_holding_never_reaches_libfranka():
    """The live re-stack's failed moves: the trajectory re-asserts the
    program's close command (0.02 m) while the fingers hold a block at
    0.028 m, and libfranka rejects a fresh grasp mid-grip. Re-closing while
    holding must be a successful no-op; opening clears the hold."""
    env, adapter = build()
    adapter.set_gripper(0.25)  # the fake block stops the fingers at 0.028
    assert len(env._gripper.grasps) == 1
    result = env.set_gripper_width(0.02)
    assert result["ok"] is True
    assert result["noop"] == "already grasping"
    assert len(env._gripper.grasps) == 1  # no second grasp command was sent
    env.set_gripper_width(0.08)  # open releases the hold
    result = env.set_gripper_width(0.02)  # a fresh close grasps again
    assert "noop" not in result
    assert len(env._gripper.grasps) == 2


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


def test_scene_pose_solver_recovers_a_known_camera():
    """Synthetic truth: place a camera, generate its view of the plane and two
    landmarks, and demand the solver hand the pose back."""
    from cap_harness.franka_real.scene_calibration import solve_pose

    angle = np.deg2rad(32.0)
    yaw_true = np.deg2rad(14.0)
    pitch = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)],
    ])
    spin = np.array([
        [np.cos(yaw_true), -np.sin(yaw_true), 0],
        [np.sin(yaw_true), np.cos(yaw_true), 0],
        [0, 0, 1],
    ])
    rotation_true = spin @ pitch
    position_true = np.array([0.52, 0.31, 0.47])

    world_landmarks = {
        "blue": np.array([0.461, -0.156, 0.075]),
        "red": np.array([0.394, -0.062, 0.072]),
    }
    inverse = rotation_true.T
    landmarks = {
        name: (inverse @ (w - position_true), w) for name, w in world_landmarks.items()
    }
    table_z = 0.0177
    # The table plane, seen from the camera: world normal +z, height table_z.
    normal_cam = inverse @ np.array([0.0, 0.0, 1.0])
    point_on_plane_cam = inverse @ (np.array([0.45, 0.0, table_z]) - position_true)
    offset = -float(normal_cam @ point_on_plane_cam)

    transform, report = solve_pose((normal_cam, offset), landmarks, table_z)
    assert np.allclose(transform[:3, 3], position_true, atol=1e-6)
    assert np.allclose(transform[:3, :3], rotation_true, atol=1e-6)
    assert all(v < 0.01 for v in report["per_landmark_residual_mm"].values())
    assert abs(report["table_z_residual_mm"]) < 0.01


def test_calibration_landmark_rejects_the_robot_base_leds():
    """Live failure 2026-11-13: the base column's blue LED strips out-pixelled
    the real blue block and dragged the solve 146 mm off. A landmark blob must
    stand above the table plane, however many pixels it has."""
    from cap_harness.franka_real.scene_calibration import colour_top_centroid

    # Camera looking straight down from 0.5 m; plane normal toward the camera.
    plane = (np.array([0.0, 0.0, -1.0]), 0.5)
    depth = np.full((60, 80), 0.5)
    rgb = np.full((60, 80, 3), 160, dtype=np.uint8)
    blue = (40, 50, 200)
    # LED glow: the biggest blue blob, depth at the plane (zero raise).
    rgb[0:10, 0:60] = blue
    # The real block: smaller, its whole blob 0.06 m above the plane.
    rgb[25:40, 30:45] = blue
    depth[25:40, 30:45] = 0.44
    columns, rows_idx = np.meshgrid(np.arange(80), np.arange(60))
    points = np.dstack([
        (columns - 40) * depth / 100.0,
        (rows_idx - 30) * depth / 100.0,
        depth,
    ])

    centroid = colour_top_centroid(rgb, points, "blue", plane)
    raise_m = float(centroid @ plane[0] + plane[1])
    assert 0.05 < raise_m < 0.07  # the block's top, not the glow

    # With only the glow in view the solve must refuse, not guess.
    rgb[25:40, 30:45] = 160
    depth[25:40, 30:45] = 0.5
    points = np.dstack([
        (columns - 40) * depth / 100.0,
        (rows_idx - 30) * depth / 100.0,
        depth,
    ])
    with pytest.raises(ValueError, match="no plausible blue block"):
        colour_top_centroid(rgb, points, "blue", plane)


def test_station_profile_round_trips_a_scene_camera(tmp_path):
    import json

    from cap_harness.franka_real.config import load_station_config

    profile = {
        "camera": {"serial": "218622274622"},
        "scene_camera": {
            "serial": "938422073914",
            "fps": 15,
            "pose_position": [0.52, 0.31, 0.47],
            "pose_quaternion_wxyz": [0.9, 0.1, 0.1, 0.4],
            "calibrated": "test",
        },
    }
    (tmp_path / "bench-two-cams.json").write_text(json.dumps(profile))
    config = load_station_config("bench-two-cams", tmp_path)
    assert config.camera.serial == "218622274622"
    assert config.scene_camera.serial == "938422073914"
    assert config.scene_camera.pose_position == (0.52, 0.31, 0.47)


# -- earned success verdicts -------------------------------------------------


def bench_state(q=HOME) -> RobotState:
    return RobotState(
        joint_positions={"primary": np.asarray(q, dtype=np.float64)},
        joint_velocities={"primary": np.zeros(ARM_DOF)},
        end_effector_poses={
            "primary": Pose(
                position=fake_fk(q)[:3, 3],
                quaternion_wxyz=np.array([0.0, 1.0, 0.0, 0.0]),
                frame="base",
            )
        },
        gripper_positions={"primary": 1.0},
        base_frame="base",
        embodiment="franka_real",
    )


def topdown_observation(blocks, q=HOME, bleed_ring=None):
    """A synthetic nadir view at (0.45, 0, 0.60): blocks as (x, y, top_z, half).

    ``bleed_ring`` names a colour whose mask's outer pixels get the *table's*
    depth: silhouette depth-bleed, the failure that smeared a live scene-camera
    localization of a 55 mm block into a 225 mm point cloud.
    """
    from cap_harness.contracts import CameraObservation, Observation

    cam_z, f, cx, cy = 0.60, 400.0, 320.0, 240.0
    rgb = np.full((480, 640, 3), 120, dtype=np.uint8)
    depth = np.full((480, 640), cam_z - 0.017)
    colours = {"red": (200, 40, 40), "green": (40, 200, 40), "blue": (40, 50, 200),
               "wood": (150, 130, 100)}  # matches no colour rule; depth-only tests
    for key, (x, y, top_z, half) in blocks.items():
        colour = key.rstrip("0123456789")  # "blue2": a second blob of the same colour
        z = cam_z - top_z  # camera x = world x - 0.45, camera y = -world y
        u0 = int(cx + (x - half - 0.45) / z * f)
        u1 = int(cx + (x + half - 0.45) / z * f)
        v0 = int(cy + (-y - half) / z * f)
        v1 = int(cy + (-y + half) / z * f)
        rgb[v0:v1, u0:u1] = colours[colour]
        depth[v0:v1, u0:u1] = z
        if colour == bleed_ring:
            ring = np.full((480, 640), False)
            ring[v0:v1, u0:u1] = True
            ring[v0 + 4 : v1 - 4, u0 + 4 : u1 - 4] = False
            depth[ring] = cam_z - 0.017
    camera = CameraObservation(
        rgb=rgb,
        depth_m=depth,
        intrinsics=np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]]),
        frame="wrist_camera",
        camera_pose=Pose(  # 180 deg about x: optical +z looks straight down
            position=np.array([0.45, 0.0, cam_z]),
            quaternion_wxyz=np.array([0.0, 1.0, 0.0, 0.0]),
            frame="base",
        ),
    )
    return Observation(cameras={"wrist": camera}, robot_state=bench_state(q))


def test_stack_verdict_measures_height_and_survives_depth_bleed():
    from cap_harness.franka_real.success import evaluate_success

    observation = topdown_observation({
        "green": (0.46, -0.156, 0.044, 0.022),  # exposed ring under blue
        "blue": (0.46, -0.156, 0.071, 0.014),   # pile height, painted on top
    }, bleed_ring="blue")
    config = FrankaStationConfig()
    verdict, report = evaluate_success("stack_blocks", observation, config)
    assert verdict is True, report
    assert report["blue"]["top_z"] == pytest.approx(0.071, abs=0.004)
    assert report["blue_to_green_m"] < 0.02
    # The same scene is not an unstack.
    verdict, report = evaluate_success("unstack_blocks", observation, config)
    assert verdict is False and report["blue_down"] is False


def test_unstack_verdict_wants_blue_down_and_clear_of_green():
    from cap_harness.franka_real.success import evaluate_success

    observation = topdown_observation({
        "green": (0.46, -0.156, 0.044, 0.014),
        "blue": (0.37, 0.10, 0.044, 0.014),
        # A bigger blue blob outside the workspace box: the first live verdict
        # elected 8572 px of something blue at the robot's base over the block.
        "blue2": (0.10, 0.0, 0.12, 0.030),
    })
    config = FrankaStationConfig()
    verdict, report = evaluate_success("unstack_blocks", observation, config)
    assert verdict is True, report
    assert report["blue"]["position"][0] == pytest.approx(0.37, abs=0.01)
    assert report["blue_to_green_m"] == pytest.approx(0.272, abs=0.02)
    verdict, _ = evaluate_success("stack_blocks", observation, config)
    assert verdict is False


def test_move_red_verdict_wants_red_resting_on_clear_table():
    from cap_harness.franka_real.success import evaluate_success

    config = FrankaStationConfig()
    clear = topdown_observation({
        "red": (0.40, -0.05, 0.044, 0.014),
        "green": (0.55, 0.20, 0.044, 0.014),
        "blue": (0.30, 0.25, 0.044, 0.014),
    })
    verdict, report = evaluate_success("move_red", clear, config)
    assert verdict is True, report

    crowded = topdown_observation({
        "red": (0.52, 0.17, 0.044, 0.014),
        "green": (0.55, 0.20, 0.044, 0.014),
    })
    verdict, report = evaluate_success("move_red", crowded, config)
    assert verdict is False and report["clear_of_others"] is False


def test_reach_home_verdict_reads_the_measured_joints():
    from cap_harness.franka_real.success import evaluate_success

    config = FrankaStationConfig()
    verdict, _ = evaluate_success("reach_home", topdown_observation({}, q=HOME), config)
    assert verdict is True
    away = topdown_observation({}, q=HOME + 0.2)
    verdict, report = evaluate_success("reach_home", away, config)
    assert verdict is False and report["max_joint_deviation_rad"] > 0.1


def test_clutter_tasks_carry_goal_geometry_to_programs():
    """One source of truth: the registry's goal reaches programs through the
    extension tool and reaches check_success through the registry."""
    registry = FrankaRealTaskRegistry()
    wall = registry.resolve("build_wall", 0)
    assert wall.goal["slots"] >= 3
    assert wall.goal["surveyed"] is True  # measured 2026-11-13, no longer placeholder
    assert wall.to_manifest_record()["goal"]["wall_axis"] == "x"
    # The measured wall lines, not just the square footprint: programs need
    # them for hand clearance (the rim bump of 2026-11-13).
    for name in ("pick_from_clutter", "build_wall"):
        goal = registry.resolve(name, 0).goal
        assert goal["box_x"][0] < goal["box_x"][1]
        assert goal["box_y"][0] < goal["box_y"][1]
    _, adapter = build()
    adapter.reset("pick_from_clutter", 0)
    metadata = adapter.get_task_metadata()
    assert metadata["goal"] == dict(registry.resolve("pick_from_clutter", 0).goal)


def test_clutter_example_programs_pass_the_sandbox_validator():
    import ast
    from pathlib import Path

    from cap_harness.runtime import _ProgramValidator

    examples = Path(__file__).resolve().parent.parent / "examples" / "franka_real"
    for name in ("pick_from_clutter", "build_wall"):
        source = (examples / name / f"{name}.py").read_text()
        _ProgramValidator().visit(ast.parse(source))  # raises on a violation


def test_build_wall_verdict_counts_depth_slab_occupancy():
    from cap_harness.franka_real.success import evaluate_success

    goal = FrankaRealTaskRegistry().resolve("build_wall", 0).goal
    config = FrankaStationConfig()
    blocks = {
        f"wood{index}": (goal["wall_start"] + index * goal["slot_pitch"],
                         goal["wall_fixed"], 0.044, 0.022)
        for index in range(goal["slots"])
    }
    verdict, report = evaluate_success("build_wall", topdown_observation(blocks), config)
    assert verdict is True, report
    assert report["slots_filled"] == goal["slots"]

    gap = dict(blocks)
    gap.pop("wood2")
    verdict, report = evaluate_success("build_wall", topdown_observation(gap), config)
    assert verdict is False
    assert report["slot_points"]["slot_2"] < report["threshold"]
    assert report["slots_filled"] == goal["slots"] - 1


def test_pick_from_clutter_verdict_wants_a_block_at_the_target():
    from cap_harness.franka_real.success import evaluate_success

    goal = FrankaRealTaskRegistry().resolve("pick_from_clutter", 0).goal
    config = FrankaStationConfig()
    tx, ty = goal["pick_target"]
    placed = topdown_observation({"wood": (tx, ty, 0.044, 0.022)})
    verdict, report = evaluate_success("pick_from_clutter", placed, config)
    assert verdict is True, report

    elsewhere = topdown_observation({"wood": (0.55, 0.25, 0.044, 0.022)})
    verdict, report = evaluate_success("pick_from_clutter", elsewhere, config)
    assert verdict is False
    assert report["points_at_target"] < report["threshold"]


def test_slab_verdict_prefers_the_scene_camera_over_a_warped_wrist_view():
    """Live phantom: the wrist's off-nadir reconstruction lifted the table
    itself into the block band (58k band points, 92 'at' an empty target).
    With a scene camera present, the slab predicate must believe it alone."""
    from cap_harness.contracts import Observation
    from cap_harness.franka_real.success import evaluate_success

    goal = FrankaRealTaskRegistry().resolve("pick_from_clutter", 0).goal
    tx, ty = goal["pick_target"]
    # The wrist sees a warped-table patch in the band exactly at the target...
    warped_wrist = topdown_observation({"wood": (tx, ty, 0.035, 0.04)})
    # ...while the calibrated scene camera sees a genuinely empty target.
    clean_scene = topdown_observation({})
    both = Observation(
        cameras={
            "wrist": warped_wrist.cameras["wrist"],
            "scene": clean_scene.cameras["wrist"],
        },
        robot_state=bench_state(),
    )
    config = FrankaStationConfig()
    verdict, report = evaluate_success("pick_from_clutter", both, config)
    assert verdict is False, report
    assert report["camera"] == "scene"
    # Without a scene camera the wrist still testifies (and is fooled): that
    # fallback is the documented cost of a one-camera station.
    verdict, report = evaluate_success("pick_from_clutter", warped_wrist, config)
    assert verdict is True and report["camera"] == "wrist"


def test_check_success_files_its_evidence_with_the_run(tmp_path):
    import json
    from types import SimpleNamespace

    env = RealFrankaEnv(
        FrankaStationConfig(),
        panda=FakePanda(),
        gripper=FakeGripper(),
        camera=SyntheticCamera(),
        fk=fake_fk,
    )
    adapter = FrankaRealAdapter(
        env, allow_physical_motion=False, run_observer=SimpleNamespace(root=tmp_path)
    )
    adapter.reset("stack_blocks", 0)
    verdict = adapter.check_success()
    assert isinstance(verdict, bool)
    report = json.loads((tmp_path / "logs" / "success-report.json").read_text())
    assert report["task"] == "stack_blocks"
    assert bool(report["verdict"]) == verdict
    assert adapter.last_success_report["task"] == "stack_blocks"


def test_observation_carries_both_cameras_when_the_station_has_two():
    from cap_harness.franka_real.config import FrankaSceneCameraConfig

    config = FrankaStationConfig(
        scene_camera=FrankaSceneCameraConfig(
            serial="x",
            pose_position=(0.52, 0.31, 0.47),
            pose_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
    )
    env = RealFrankaEnv(
        config,
        panda=FakePanda(),
        gripper=FakeGripper(),
        camera=SyntheticCamera(),
        scene_camera=SyntheticCamera(width=96, height=64),
        fk=fake_fk,
    )
    adapter = FrankaRealAdapter(env, allow_physical_motion=False)
    adapter.reset("observe_station", 0)
    observation = adapter.get_observation()
    assert set(observation.cameras) == {"wrist", "scene"}
    scene = observation.cameras["scene"]
    assert scene.frame == "scene_camera"
    assert scene.camera_pose.frame == "base"
    assert np.allclose(scene.camera_pose.position, [0.52, 0.31, 0.47])
    # The wrist pose rides the hand; the scene pose must not.
    assert not np.allclose(
        scene.camera_pose.position, observation.cameras["wrist"].camera_pose.position
    )
