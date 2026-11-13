"""Fixed scene-camera extrinsics from what the bench already knows.

No checkerboard: the calibration targets are the table plane and blocks whose
positions the bench has verified by touch. The depth cloud's dominant plane
fixes two tilts and the height; two or more coloured-block top faces, matched
to their known world coordinates, fix the remaining translation and the yaw by
2-D Procrustes. Every input is stated with its provenance, and the solve
reports per-landmark residuals instead of a single cheerful number.

Honesty about accuracy: landmark positions carry 6-12 mm error bars and the
camera sits oblique, so the solved pose is good to roughly a centimetre in
xy and a few degrees of yaw. That serves scene attribution, episode video,
and re-localization checks; it does not serve grasping, which stays with the
wrist camera looking straight down.

Usage (on the station host):

    python -m cap_harness.franka_real.scene_calibration \\
        --serial 938422073914 --table-z 0.0177 \\
        --landmark blue 0.461 -0.156 0.075 \\
        --landmark red 0.394 -0.062 0.072 \\
        --write configs/franka/franka-bench.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cap_harness.geometry import matrix_to_quaternion_wxyz
from cap_harness.providers.colorblob import MIN_COMPONENT_PX, _RULES

TOP_BAND_M = 0.012
MIN_PLANE_POINTS = 5000


def back_project(depth_m: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """(H, W, 3) camera-frame points; invalid depth becomes NaN."""
    height, width = depth_m.shape
    columns, rows = np.meshgrid(np.arange(width), np.arange(height))
    z = np.where(depth_m > 0, depth_m, np.nan)
    x = (columns - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (rows - intrinsics[1, 2]) * z / intrinsics[1, 1]
    return np.dstack([x, y, z])


def fit_table_plane(points: np.ndarray, *, passes: int = 3) -> tuple[np.ndarray, float]:
    """Dominant plane as (unit normal toward the camera, offset): n.p + d = 0.

    Starts from every valid point in a working depth band and tightens by
    rejecting 3-sigma outliers, which drops the blocks, the robot base and the
    wall as long as the table dominates the view.
    """
    cloud = points.reshape(-1, 3)
    cloud = cloud[np.isfinite(cloud[:, 2])]
    cloud = cloud[(cloud[:, 2] > 0.35) & (cloud[:, 2] < 1.1)]
    if cloud.shape[0] < MIN_PLANE_POINTS:
        raise ValueError(f"only {cloud.shape[0]} depth points in the working band")
    selected = cloud
    normal, offset = None, None
    for _ in range(passes):
        centroid = selected.mean(axis=0)
        _, _, vt = np.linalg.svd(selected - centroid, full_matrices=False)
        normal = vt[2]
        if normal[2] > 0:  # orient toward the camera (points have +z depth)
            normal = -normal
        offset = -float(normal @ centroid)
        distance = np.abs(cloud @ normal + offset)
        sigma = max(float(distance[distance < 0.02].std()), 0.002)
        selected = cloud[distance < 3.0 * sigma]
    return normal, offset


def colour_top_centroid(
    rgb: np.ndarray,
    points: np.ndarray,
    colour: str,
    plane: tuple[np.ndarray, float],
) -> np.ndarray:
    """Camera-frame centroid of the colour's highest face."""
    from scipy import ndimage

    normal, offset = plane
    r, g, b = (rgb[:, :, i].astype(np.int16) for i in range(3))
    saturation = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    mask = _RULES[colour](r, g, b) & (saturation > 25) & np.isfinite(points[:, :, 2])
    labels, count = ndimage.label(mask)
    if count == 0:
        raise ValueError(f"no {colour} region in the scene view")
    sizes = ndimage.sum_labels(np.ones_like(labels), labels, index=range(1, count + 1))
    biggest = int(np.argmax(sizes)) + 1
    if sizes[biggest - 1] < MIN_COMPONENT_PX:
        raise ValueError(f"largest {colour} region is only {int(sizes[biggest - 1])} px")
    blob = points[labels == biggest]
    # Height above the plane, positive toward the camera side (up).
    height = blob @ normal + offset
    top = blob[height > float(height.max()) - TOP_BAND_M]
    return top.mean(axis=0)


def solve_pose(
    plane: tuple[np.ndarray, float],
    landmarks: dict[str, tuple[np.ndarray, np.ndarray]],
    table_z: float,
) -> tuple[np.ndarray, dict]:
    """T_world_camera from the plane and >= 2 (camera point, world point) pairs."""
    if len(landmarks) < 2:
        raise ValueError("solving yaw and translation needs at least two landmarks")
    normal, offset = plane

    # Tilt: rotate the plane's up-normal onto world +z (Rodrigues).
    up = np.asarray(normal, dtype=np.float64)
    axis = np.cross(up, [0.0, 0.0, 1.0])
    sin = float(np.linalg.norm(axis))
    cos = float(up @ [0.0, 0.0, 1.0])
    if sin < 1e-12:
        tilt = np.eye(3) if cos > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        k = axis / sin
        cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        tilt = np.eye(3) + sin * cross + (1 - cos) * (cross @ cross)

    cam = np.array([tilt @ p for p, _ in landmarks.values()])
    world = np.array([w for _, w in landmarks.values()])

    # Yaw and xy translation: 2-D Procrustes without scale.
    cam_centre, world_centre = cam[:, :2].mean(axis=0), world[:, :2].mean(axis=0)
    correlation = (cam[:, :2] - cam_centre).T @ (world[:, :2] - world_centre)
    u, _, vt = np.linalg.svd(correlation)
    spin = vt.T @ u.T
    if np.linalg.det(spin) < 0:
        u[:, -1] *= -1
        spin = vt.T @ u.T
    yaw = np.eye(3)
    yaw[:2, :2] = spin
    rotation = yaw @ tilt
    translation = np.zeros(3)
    translation[:2] = world_centre - spin @ cam_centre
    # Height from the landmarks, cross-checked against the plane below.
    translation[2] = float(np.mean(world[:, 2] - cam[:, 2]))

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    report: dict = {"per_landmark_residual_mm": {}}
    for name, (p, w) in landmarks.items():
        predicted = rotation @ p + translation
        report["per_landmark_residual_mm"][name] = round(
            float(np.linalg.norm(predicted - w)) * 1000, 1
        )
    # Any point on the fitted plane should land at the table height.
    plane_point = -offset * up
    report["table_z_residual_mm"] = round(
        float((rotation @ plane_point + translation)[2] - table_z) * 1000, 1
    )
    report["camera_position"] = [round(float(v), 4) for v in translation]
    return transform, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--table-z", type=float, default=0.0177)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--landmark",
        nargs=4,
        action="append",
        metavar=("COLOUR", "X", "Y", "TOP_Z"),
        required=True,
        help="a block colour and the world position of its top-face centre",
    )
    parser.add_argument("--write", type=Path, help="station profile JSON to merge into")
    parser.add_argument("--wrist-serial", help="recorded alongside when writing a profile")
    args = parser.parse_args()

    from cap_harness.yam_real.cameras import RealSenseCamera

    camera = RealSenseCamera(args.serial, resolution=(640, 480), fps=args.fps)
    try:
        frame = None
        for _ in range(20):  # auto-exposure settle
            frame = camera.read() or frame
        if frame is None:
            raise ValueError("the scene camera produced no frame")
    finally:
        camera.close()

    points = back_project(frame.depth_m, frame.intrinsics)
    plane = fit_table_plane(points)
    landmarks = {}
    for colour, x, y, top in args.landmark:
        centroid = colour_top_centroid(frame.rgb, points, colour, plane)
        landmarks[colour] = (centroid, np.array([float(x), float(y), float(top)]))
    transform, report = solve_pose(plane, landmarks, args.table_z)
    quaternion = matrix_to_quaternion_wxyz(transform[:3, :3])
    print(json.dumps(report, indent=1))

    if args.write:
        profile = json.loads(args.write.read_text()) if args.write.is_file() else {}
        if args.wrist_serial:
            profile.setdefault("camera", {})["serial"] = args.wrist_serial
        profile["scene_camera"] = {
            "serial": args.serial,
            "fps": args.fps,
            "pose_position": [round(float(v), 4) for v in transform[:3, 3]],
            "pose_quaternion_wxyz": [round(float(v), 6) for v in quaternion],
            "calibrated": "plane+landmark solve; see scene_calibration.py",
        }
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps(profile, indent=1) + "\n")
        print(f"wrote {args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
