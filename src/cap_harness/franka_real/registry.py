"""Typed discovery for the supported real-Franka tasks.

Follows :mod:`cap_harness.yam_real.registry`: a hardware task is a language
goal plus a physical scene. ``scene_reset`` says how the scene is restored
between trials -- and this station has a value the YAM set does not need:
``"skill"``, meaning the bench resets itself (the station's own unstack
routine returns the scene to a start state without a human), which is what
makes unattended multi-trial campaigns possible on this hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


class FrankaRealRegistryError(ValueError):
    """Raised when a real-Franka task reference is invalid."""


@dataclass(frozen=True, slots=True)
class FrankaRealTaskMetadata:
    """Static metadata identifying one real-robot task."""

    task_name: str
    language: str
    task_id: int = 0
    family: str = "franka_real"
    scene_reset: str = "manual"
    #: Task-specific goal geometry (box footprint, wall slots, targets).
    #: Programs read it through ``franka_real.get_task_metadata()["goal"]`` and
    #: ``check_success`` reads it from the registry, so both act on one spec.
    goal: Mapping[str, Any] = field(default_factory=dict)

    @property
    def suite_name(self) -> str:
        return self.task_name

    @property
    def task_ref(self) -> str:
        return f"{self.task_name}:{self.task_id}"

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "goal": dict(self.goal),
            "language": self.language,
            "scene_reset": self.scene_reset,
            "suite_name": self.suite_name,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "task_ref": self.task_ref,
        }


FRANKA_REAL_TASKS: tuple[FrankaRealTaskMetadata, ...] = (
    FrankaRealTaskMetadata(
        "observe_station", "Observe the Franka bench.", scene_reset="none"
    ),
    FrankaRealTaskMetadata(
        "reach_home", "Return the arm to the station home pose.", scene_reset="none"
    ),
    FrankaRealTaskMetadata(
        "joint_motion", "Execute a bounded joint motion.", scene_reset="none"
    ),
    FrankaRealTaskMetadata(
        "stack_blocks",
        "Stack the blue block on the green block.",
        scene_reset="skill",
    ),
    FrankaRealTaskMetadata(
        "unstack_blocks",
        "Take the blue block off the green block and set it on clear table.",
        scene_reset="skill",
    ),
    FrankaRealTaskMetadata(
        "move_red",
        "Pick up the red block and set it down on a clear patch of table.",
        scene_reset="none",
    ),
    FrankaRealTaskMetadata(
        "pick_from_clutter",
        "Pick one wooden block out of the box of loose blocks and set it on the table.",
        scene_reset="manual",
        # Surveyed 2026-11-13 (no motion): mesh basket measured from the wrist
        # depth cloud (SAM3 found five block instances inside at tops
        # 0.055-0.066). The basket's far side nears the workspace y limit;
        # blocks were all seen at y <= 0.29. box_x/box_y are the measured wall
        # lines: picks need hand clearance from them (a grasp 9 mm from the
        # east wall drove the hand into the rim and fired the cartesian
        # reflex, live), and the square half-extent is too coarse for that.
        goal={
            "box_centre": [0.44, 0.23],
            "box_half_extent": 0.11,
            "box_x": [0.36, 0.51],
            "box_y": [0.10, 0.36],
            "box_rim_z": 0.10,
            "pick_target": [0.34, -0.20],
            "surveyed": True,
        },
    ),
    FrankaRealTaskMetadata(
        "build_wall",
        "Build a wall of wooden blocks across the table, one block per slot.",
        scene_reset="manual",
        # Wall runs along +x (blocks end to end, long axis along the wall) so
        # every place yaw stays near zero, well inside the proven wrist range.
        # Surveyed 2026-11-13: box as in pick_from_clutter; the wall line sits
        # at y=-0.12, clear of the standing red block (-0.062) and the pile
        # (-0.156) by more than the slab's cross window.
        goal={
            "wall_axis": "x",
            "wall_start": 0.28,
            "wall_fixed": -0.12,
            "slot_pitch": 0.060,
            "slots": 5,
            "box_centre": [0.44, 0.23],
            "box_half_extent": 0.11,
            "box_x": [0.36, 0.51],
            "box_y": [0.10, 0.36],
            "box_rim_z": 0.10,
            "surveyed": True,
        },
    ),
)


class FrankaRealTaskRegistry:
    """Resolve only the explicitly supported real-Franka task set."""

    def __init__(self) -> None:
        self._by_name = {task.task_name: task for task in FRANKA_REAL_TASKS}

    def suites(self) -> tuple[str, ...]:
        return tuple(self._by_name)

    def resolve(self, suite: str, task_id: int = 0) -> FrankaRealTaskMetadata:
        name = str(suite)
        if name not in self._by_name:
            raise FrankaRealRegistryError(
                f"unknown franka_real task {name!r}; supported: {', '.join(self._by_name)}"
            )
        if int(task_id) != 0:
            raise FrankaRealRegistryError(
                f"franka_real tasks are single-variant; task_id must be 0, got {task_id}"
            )
        return self._by_name[name]


__all__ = [
    "FRANKA_REAL_TASKS",
    "FrankaRealRegistryError",
    "FrankaRealTaskMetadata",
    "FrankaRealTaskRegistry",
]
