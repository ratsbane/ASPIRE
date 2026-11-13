"""Station configuration for a single-arm Franka Panda bench.

A station profile is a JSON file named by the ``--station`` argument, resolved
under ``configs/franka`` unless a config root overrides it. Every field has a
working default for the reference bench (one Panda on FCI, one wrist-mounted
RealSense D405), so a profile only states what differs.

The hand-eye transform deserves honesty: it is *derived*, not solved. The bench
this port grew on calibrates the wrist camera with a planar map (a 2D affine
fit at one viewing height), which fixes the camera's lateral offset precisely
and says nothing about its full 6-DOF pose. The profile carries the planar
offsets plus a nominal mounting rotation and height; consumers that need
millimetre localization should confirm by touch rather than trust the frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class FrankaCameraConfig:
    """One wrist-mounted RGB-D camera."""

    serial: str | None = None
    """RealSense serial; None opens the only connected device, refusing two."""

    role: str = "wrist"
    resolution: tuple[int, int] = (640, 480)
    fps: int = 30
    hand_eye_position: tuple[float, float, float] = (-0.0595, -0.0087, -0.0537)
    """Camera origin in the EE frame, metres: the x/y come from the bench's
    planar camera map (its offset vector is the camera's nadir displacement
    from the TCP), the z from a measured table depth at a known TCP height.
    Derived, see module docstring; refine by touch, not trust."""

    hand_eye_quaternion_wxyz: tuple[float, float, float, float] = (
        0.7071068,
        0.0,
        0.0,
        -0.7071068,
    )
    """Camera orientation in the EE frame: optical z along EE z (toward the
    table when the hand points down), image axes matching the bench's measured
    u->world-y, v->world-x convention at yaw zero."""


@dataclass(frozen=True, slots=True)
class FrankaSceneCameraConfig:
    """A fixed third-person RGB-D camera, posed in the base frame.

    Unlike the wrist camera its pose is static, solved by
    :mod:`cap_harness.franka_real.scene_calibration` from the table plane and
    touch-verified block landmarks. Centimetre-grade: attribution and episode
    video, not grasping.
    """

    serial: str
    role: str = "scene"
    resolution: tuple[int, int] = (640, 480)
    fps: int = 15
    pose_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pose_quaternion_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    calibrated: str | None = None


@dataclass(frozen=True, slots=True)
class FrankaWorkspace:
    """Axis-aligned box, base frame, that every commanded EE position must stay in.

    The z floor is above the table plane by less than a block height: low enough
    to probe and grasp lying blocks, high enough that no planned waypoint can
    press the fingertips through the tabletop.
    """

    x: tuple[float, float] = (0.25, 0.66)
    y: tuple[float, float] = (-0.33, 0.33)
    z: tuple[float, float] = (0.003, 0.70)

    def contains(self, position) -> bool:
        px, py, pz = (float(v) for v in position[:3])
        return (
            self.x[0] <= px <= self.x[1]
            and self.y[0] <= py <= self.y[1]
            and self.z[0] <= pz <= self.z[1]
        )


@dataclass(frozen=True, slots=True)
class FrankaStationConfig:
    """Everything the env needs to own one Panda, one gripper, one camera."""

    station: str = "franka-bench"
    robot_host: str = "192.168.0.253"
    camera: FrankaCameraConfig = field(default_factory=FrankaCameraConfig)
    scene_camera: FrankaSceneCameraConfig | None = None
    workspace: FrankaWorkspace = field(default_factory=FrankaWorkspace)
    home_joints: tuple[float, ...] = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
    control_frequency_hz: float = 20.0
    speed_factor: float = 0.12
    """panda_py motion-generator speed scale for executed waypoints."""

    waypoint_chord_tolerance_rad: float = 0.05
    """Decimation fidelity for executed batches. The motion generator comes to
    rest at every waypoint it is given, so a dense planned path executes as a
    crawl; a waypoint within this joint-space distance of the straight chord
    between its kept neighbours is dropped before sending. 0 disables
    decimation and sends every planned waypoint."""

    gripper_max_width_m: float = 0.08
    gripper_force_n: float = 25.0
    joint_limits_lower: tuple[float, ...] = (
        -2.8973,
        -1.7628,
        -2.8973,
        -3.0718,
        -2.8973,
        -0.0175,
        -2.8973,
    )
    joint_limits_upper: tuple[float, ...] = (
        2.8973,
        1.7628,
        2.8973,
        -0.0698,
        2.8973,
        3.7525,
        2.8973,
    )

    @property
    def control_period_s(self) -> float:
        return 1.0 / float(self.control_frequency_hz)


def load_station_config(
    name: str = "franka-bench", config_root: Path | None = None
) -> FrankaStationConfig:
    """Load a station profile, tolerating a missing file for the default station.

    A named station other than the default must exist; the default falls back
    to the dataclass defaults so a fresh checkout can observe a reference bench
    without writing a profile first.
    """
    root = Path(config_root) if config_root is not None else _repo_root() / "configs/franka"
    path = root / f"{name}.json"
    if not path.is_file():
        if name == "franka-bench":
            return FrankaStationConfig()
        raise ValueError(f"unknown Franka station {name!r}: no profile at {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    camera = FrankaCameraConfig(
        **{k: tuple(v) if isinstance(v, list) else v for k, v in raw.pop("camera", {}).items()}
    )
    scene_raw = raw.pop("scene_camera", None)
    scene = None
    if scene_raw is not None:
        scene = FrankaSceneCameraConfig(
            **{k: tuple(v) if isinstance(v, list) else v for k, v in scene_raw.items()}
        )
    workspace = FrankaWorkspace(
        **{k: tuple(v) for k, v in raw.pop("workspace", {}).items()}
    )
    fields = {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}
    return FrankaStationConfig(
        station=name, camera=camera, scene_camera=scene, workspace=workspace, **fields
    )


__all__ = [
    "FrankaCameraConfig",
    "FrankaSceneCameraConfig",
    "FrankaStationConfig",
    "FrankaWorkspace",
    "load_station_config",
]
