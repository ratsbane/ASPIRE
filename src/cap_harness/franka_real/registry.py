"""Typed discovery for the supported real-Franka tasks.

Follows :mod:`cap_harness.yam_real.registry`: a hardware task is a language
goal plus a physical scene. ``scene_reset`` says how the scene is restored
between trials -- and this station has a value the YAM set does not need:
``"skill"``, meaning the bench resets itself (the station's own unstack
routine returns the scene to a start state without a human), which is what
makes unattended multi-trial campaigns possible on this hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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

    @property
    def suite_name(self) -> str:
        return self.task_name

    @property
    def task_ref(self) -> str:
        return f"{self.task_name}:{self.task_id}"

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "family": self.family,
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
