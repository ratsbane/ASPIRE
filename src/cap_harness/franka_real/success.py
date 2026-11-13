"""Earned task verdicts for the real Franka bench.

The real-robot embodiments in this harness return ``check_success() -> False``
and leave success to be judged from the video. This station can do better: its
tasks are about coloured blocks whose world positions a camera can measure, so
the verdict is a measurement, not a review. Each predicate re-observes the
scene after the program ends and checks the task's physical postcondition --
the blue block's top face risen to pile height over green, or back down on
clear table -- against positions the bench has verified by touch.

Two honesty rules shape the geometry here. First, every localization is a
**top-face band centroid**: within the colour's blob, only points within
``TOP_BAND_M`` of the blob's top height vote. Mask-edge pixels on an oblique
view carry the *background's* depth (a 55 mm block once became a 225 mm point
cloud this way); those points land at table height and the band excludes them,
which is the same defence the scene calibration uses. Second, a predicate that
cannot measure what it needs says so: the verdict is ``None`` (unknown), never
a cheerful ``True``, and the report records which camera testified and what it
saw. ``joint_motion`` has no recorded goal, so it stays unknown by design.

Heights are in the base frame and come from the bench's touch-verified block
geometry: blocks are about 55 x 28 x 27 mm, the table surface sits near
z = 0.017, a lying block's top near 0.044, a two-block pile's near 0.071, and
a block stood on end reaches about 0.072.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from cap_harness.contracts import Observation
from cap_harness.franka_real.config import FrankaStationConfig
from cap_harness.franka_real.scene_calibration import back_project
from cap_harness.geometry import pose_to_matrix
from cap_harness.providers.colorblob import MIN_COMPONENT_PX, _RULES

TOP_BAND_M = 0.012
MIN_BAND_POINTS = 20
TABLE_Z = 0.017
#: The wrist views the table obliquely from home and the 98th-percentile top
#: reads high: a touch-verified pile top of 0.075 measured 0.0875 live. These
#: bands budget for that bias: a lying block (true top 0.044) measures near
#: 0.055 and must count as down; a pile must still clear the stacked floor.
SINGLE_TOP_MAX_M = 0.062
STACKED_TOP_MIN_M = 0.065
#: A 55 mm block stood on end tops out near 0.072; above this it is held or wrong.
BLOCK_TOP_MAX_M = 0.085
#: Blue's top-face centroid near green's: squarely on the pile. Green under a
#: stacked blue shows only its edges, and on a touch-verified live stack that
#: ring measured 0.034 from blue's centre; the tolerance leaves room for it.
STACK_XY_TOL_M = 0.045
#: Off the pile means clear of it by more than a block's long side.
UNSTACK_CLEAR_M = 0.070
MOVE_CLEAR_M = 0.060
HOME_TOL_RAD = 0.05

#: Cameras asked to testify, in order of trust: the wrist looks from above and
#: localizes to ~10 mm; the oblique scene camera is centimetre-grade but its
#: top-band statistics survive silhouette depth-bleed (proven by calibration).
CAMERA_PREFERENCE = ("wrist", "scene")

#: Depth-slab occupancy (wooden blocks have no colour rule): a spot counts as
#: holding a block when at least this many depth points sit in the block band
#: above the table inside the spot's window.
SLAB_Z_MIN = TABLE_Z + 0.012
SLAB_Z_MAX = TABLE_Z + 0.060
MIN_SLAB_POINTS = 60
SLAB_CROSS_HALF_M = 0.025


def locate_colour_top(
    observation: Observation, colour: str, workspace: Any = None
) -> Optional[dict[str, Any]]:
    """Base-frame top-face centroid of a colour, from the first camera that sees it.

    With a ``workspace`` box, a blob whose centroid lands outside it is passed
    over for the next-largest: the first live verdict elected 8572 px of
    something blue at the robot's base over the 55 mm block on the table.
    """
    for role in CAMERA_PREFERENCE:
        camera = observation.cameras.get(role)
        if camera is None or camera.depth_m is None or camera.camera_pose is None:
            continue
        found = _top_centroid_in_base(camera, colour, workspace)
        if found is not None:
            found["camera"] = role
            return found
    return None


def _top_centroid_in_base(
    camera: Any, colour: str, workspace: Any = None
) -> Optional[dict[str, Any]]:
    from scipy import ndimage

    rgb = np.asarray(camera.rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    depth = np.asarray(camera.depth_m, dtype=np.float64)
    r, g, b = (rgb[:, :, i].astype(np.int16) for i in range(3))
    saturation = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    mask = _RULES[colour](r, g, b) & (saturation > 25) & np.isfinite(depth) & (depth > 0)
    labels, count = ndimage.label(mask)
    if count == 0:
        return None
    sizes = ndimage.sum_labels(np.ones_like(labels), labels, index=range(1, count + 1))
    ranked = sorted(
        ((int(size), index + 1) for index, size in enumerate(sizes) if size >= MIN_COMPONENT_PX),
        reverse=True,
    )
    points_all = back_project(depth, np.asarray(camera.intrinsics))
    transform = pose_to_matrix(camera.camera_pose)
    for size, label in ranked[:5]:
        points = points_all[labels == label] @ transform[:3, :3].T + transform[:3, 3]
        points = points[np.isfinite(points).all(axis=1)]
        # Anything far below the table or a forearm's height above it is not a
        # resting block's surface; dropping it keeps the wall and gross depth
        # failures from electing the "top".
        points = points[(points[:, 2] > TABLE_Z - 0.03) & (points[:, 2] < 0.40)]
        if points.shape[0] < MIN_BAND_POINTS:
            continue
        top_z = float(np.percentile(points[:, 2], 98.0))
        band = points[points[:, 2] > top_z - TOP_BAND_M]
        if band.shape[0] < MIN_BAND_POINTS:
            continue
        centroid = band.mean(axis=0)
        if workspace is not None and not workspace.contains(centroid):
            continue
        return {
            "colour": colour,
            "position": [round(float(v), 4) for v in centroid],
            "top_z": round(top_z, 4),
            "pixels": size,
            "band_points": int(band.shape[0]),
        }
    return None


#: For slab occupancy the preference inverts: the fixed scene camera's table
#: plane is calibrated to a millimetre, while the wrist's off-nadir
#: reconstruction warps the table itself up into the block band (58k phantom
#: band points from one home view, found live). Masked localization tolerates
#: that warp because a mask selects an object; a maskless slab cannot.
SLAB_CAMERA_PREFERENCE = ("scene", "wrist")


def _block_band_points(observation: Observation) -> tuple[np.ndarray, Optional[str]]:
    """Base-frame depth points in the block band, from the best single camera.

    No segmentation model sits in this path: occupancy is pure geometry, which
    is the right dependency direction for a verdict. One camera testifies --
    never a union, so one bad reconstruction cannot pollute the other's view.
    """
    for role in SLAB_CAMERA_PREFERENCE:
        camera = observation.cameras.get(role)
        if camera is None or camera.depth_m is None or camera.camera_pose is None:
            continue
        depth = np.asarray(camera.depth_m, dtype=np.float64)
        points = back_project(depth, np.asarray(camera.intrinsics)).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        transform = pose_to_matrix(camera.camera_pose)
        points = points @ transform[:3, :3].T + transform[:3, 3]
        return points[(points[:, 2] > SLAB_Z_MIN) & (points[:, 2] < SLAB_Z_MAX)], role
    return np.zeros((0, 3)), None


def _spot_occupied(points: np.ndarray, x: float, y: float, half_x: float, half_y: float) -> int:
    if points.shape[0] == 0:
        return 0
    inside = (np.abs(points[:, 0] - x) <= half_x) & (np.abs(points[:, 1] - y) <= half_y)
    return int(np.count_nonzero(inside))


def _task_goal(task_name: str) -> dict[str, Any]:
    from cap_harness.franka_real.registry import FrankaRealTaskRegistry

    return dict(FrankaRealTaskRegistry().resolve(task_name).goal)


def _xy_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    return math.hypot(
        a["position"][0] - b["position"][0], a["position"][1] - b["position"][1]
    )


def evaluate_success(
    task_name: str, observation: Observation, config: FrankaStationConfig
) -> tuple[Optional[bool], dict[str, Any]]:
    """(verdict, report) for one task's physical postcondition.

    ``None`` means the predicate could not measure what it needed (or no
    predicate exists for the task); the report always says why.
    """
    report: dict[str, Any] = {"task": task_name, "method": "camera_top_band"}

    if task_name == "observe_station":
        wrist = observation.cameras.get("wrist")
        seen = wrist is not None and wrist.rgb is not None
        report["method"] = "observation_presence"
        report["wrist_frame_present"] = bool(seen)
        return bool(seen), report

    if task_name == "reach_home":
        report["method"] = "joint_deviation"
        joints = observation.robot_state.joint_positions.get("primary")
        if joints is None:
            report["note"] = "no measured joints in the observation"
            return None, report
        deviation = float(
            np.max(np.abs(np.asarray(joints) - np.asarray(config.home_joints)))
        )
        report["max_joint_deviation_rad"] = round(deviation, 4)
        report["tolerance_rad"] = HOME_TOL_RAD
        return deviation < HOME_TOL_RAD, report

    if task_name == "stack_blocks":
        blue = locate_colour_top(observation, "blue", config.workspace)
        green = locate_colour_top(observation, "green", config.workspace)
        report["blue"] = blue
        report["green"] = green
        if blue is None:
            report["note"] = "blue block not seen by any camera"
            return None, report
        stacked_height = blue["top_z"] >= STACKED_TOP_MIN_M
        report["stacked_height"] = stacked_height
        if green is not None:
            distance = _xy_distance(blue, green)
            report["blue_to_green_m"] = round(distance, 4)
            report["on_green"] = distance < STACK_XY_TOL_M
            return bool(stacked_height and report["on_green"]), report
        # Green fully hidden beneath blue is what a square stack looks like
        # from above; the height criterion carries the verdict and the report
        # says the lateral check could not run.
        report["note"] = "green not seen; lateral check skipped"
        return bool(stacked_height), report

    if task_name == "unstack_blocks":
        blue = locate_colour_top(observation, "blue", config.workspace)
        green = locate_colour_top(observation, "green", config.workspace)
        report["blue"] = blue
        report["green"] = green
        if blue is None or green is None:
            report["note"] = "unstacking exposes both blocks; one was not seen"
            return None, report
        down = blue["top_z"] <= SINGLE_TOP_MAX_M
        distance = _xy_distance(blue, green)
        report["blue_down"] = down
        report["blue_to_green_m"] = round(distance, 4)
        report["clear_of_green"] = distance > UNSTACK_CLEAR_M
        return bool(down and report["clear_of_green"]), report

    if task_name == "move_red":
        red = locate_colour_top(observation, "red", config.workspace)
        report["red"] = red
        if red is None:
            report["note"] = "red block not seen by any camera"
            return None, report
        # Lying (0.044) or stood on end (0.072) both count as set down.
        resting = TABLE_Z + 0.010 < red["top_z"] <= BLOCK_TOP_MAX_M
        report["resting"] = resting
        clear = True
        for colour in ("blue", "green"):
            other = locate_colour_top(observation, colour, config.workspace)
            if other is not None:
                distance = _xy_distance(red, other)
                report[f"red_to_{colour}_m"] = round(distance, 4)
                clear = clear and distance > MOVE_CLEAR_M
        report["clear_of_others"] = clear
        return bool(resting and clear), report

    if task_name == "pick_from_clutter":
        report["method"] = "depth_slab_occupancy"
        goal = _task_goal(task_name)
        target = goal["pick_target"]
        points, witness = _block_band_points(observation)
        count = _spot_occupied(
            points, float(target[0]), float(target[1]), 0.040, 0.040
        )
        report["camera"] = witness
        report["pick_target"] = list(target)
        report["points_at_target"] = count
        report["threshold"] = MIN_SLAB_POINTS
        if witness is None:
            report["note"] = "no camera could testify"
            return None, report
        # An empty band from a real camera is a measurement: nothing there.
        return count >= MIN_SLAB_POINTS, report

    if task_name == "build_wall":
        report["method"] = "depth_slab_occupancy"
        goal = _task_goal(task_name)
        along_x = goal.get("wall_axis", "x") == "x"
        start = float(goal["wall_start"])
        fixed = float(goal["wall_fixed"])
        pitch = float(goal["slot_pitch"])
        along_half = pitch * 0.35
        points, witness = _block_band_points(observation)
        report["camera"] = witness
        if witness is None:
            report["note"] = "no camera could testify"
            return None, report
        slots: dict[str, int] = {}
        filled = 0
        for index in range(int(goal["slots"])):
            centre = start + index * pitch
            x, y = (centre, fixed) if along_x else (fixed, centre)
            half_x = along_half if along_x else SLAB_CROSS_HALF_M
            half_y = SLAB_CROSS_HALF_M if along_x else along_half
            count = _spot_occupied(points, x, y, half_x, half_y)
            slots[f"slot_{index}"] = count
            if count >= MIN_SLAB_POINTS:
                filled += 1
        report["slot_points"] = slots
        report["threshold"] = MIN_SLAB_POINTS
        report["slots_filled"] = filled
        report["slots_wanted"] = int(goal["slots"])
        return filled == int(goal["slots"]), report

    report["method"] = "none"
    report["note"] = f"no success predicate for {task_name!r}"
    return None, report


__all__ = ["evaluate_success", "locate_colour_top"]
