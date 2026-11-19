"""Build a live Franka station from a profile. Hardware imports stay here.

Construction connects to FCI and opens the camera but commands no motion,
matching the YAM rule that building a station is observation-safe. panda_py
and pyrealsense2 are imported lazily so the harness stays importable -- and
the off-robot tests runnable -- on a machine with neither.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from cap_harness.franka_real.config import FrankaStationConfig, load_station_config
from cap_harness.franka_real.env import RealFrankaEnv
from cap_harness.yam_real.cameras import Camera, RealSenseCamera, ThreadedCamera


def _open_camera(config: FrankaStationConfig) -> Optional[Camera]:
    import pyrealsense2 as rs

    serial = config.camera.serial
    if serial is None:
        devices = [
            device.get_info(rs.camera_info.serial_number)
            for device in rs.context().query_devices()
        ]
        if not devices:
            raise ValueError("no RealSense connected and the profile names no serial")
        if len(devices) > 1:
            raise ValueError(
                f"several RealSense devices connected ({sorted(devices)}); "
                "the station profile must name one by serial"
            )
        serial = devices[0]
    return ThreadedCamera(
        RealSenseCamera(serial, resolution=config.camera.resolution, fps=config.camera.fps)
    )


def build_real_station(
    station: str = "franka-bench", config_root: Path | None = None
) -> RealFrankaEnv:
    """Connect to the configured Panda, its gripper, and the wrist camera."""
    import panda_py
    from panda_py import libfranka

    config = load_station_config(station, config_root)
    # panda_py's default is RealtimeConfig.kIgnore: the 1 kHz control thread
    # runs at ordinary priority and loses packets to desktop load (2026-11-19:
    # communication_constraints_violation reflexes with Chrome busy on omen,
    # command success rate 60-77%). On a realtime kernel, enforce: libfranka
    # then takes SCHED_FIFO for the control thread, or refuses to start
    # loudly if it cannot (the user's rtprio limit must allow it; omen's is 99).
    realtime = libfranka.RealtimeConfig.kIgnore
    if libfranka.has_realtime_kernel():
        realtime = libfranka.RealtimeConfig.kEnforce
    panda = panda_py.Panda(config.robot_host, realtime_config=realtime)
    gripper = libfranka.Gripper(config.robot_host)
    camera = _open_camera(config)
    scene = None
    if config.scene_camera is not None:
        scene = ThreadedCamera(
            RealSenseCamera(
                config.scene_camera.serial,
                resolution=config.scene_camera.resolution,
                fps=config.scene_camera.fps,
            )
        )
    return RealFrankaEnv(
        config, panda=panda, gripper=gripper, camera=camera, scene_camera=scene,
        fk=panda_py.fk,
    )


__all__ = ["build_real_station"]
