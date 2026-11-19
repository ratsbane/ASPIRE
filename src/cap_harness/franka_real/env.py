"""The Franka plant: one arm, one gripper, one wrist camera, one path that moves.

``RealFrankaEnv`` mirrors :class:`cap_harness.yam_real.env.RealYamEnv`'s rule:
:meth:`execute_waypoints` is the **only** way anything in this package commands
the arm. ``move_home`` and ``step``-sized moves are built on it rather than
beside it, so every recorded motion went through the same guard and the same
motion generator.

The clients are injected rather than constructed here -- ``panda`` and
``gripper`` are panda_py objects on hardware and fakes in tests -- and ``fk``
is a callable returning the 4x4 EE transform for a joint vector, so the
workspace guard and the published EE pose use one kinematic story. On hardware
that callable is ``panda_py.fk``; tests inject a model.

**The guard is not optional.** Every waypoint's forward kinematics must land
inside the station's workspace box before anything is sent. The bench this
port grew on enforces the same box in its own skills; a generated program that
asks for a pose outside it gets a refusal, not a clipped motion.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

import numpy as np

from cap_harness.franka_real.config import FrankaStationConfig
from cap_harness.yam_real.cameras import Camera, CameraFrame

ARM_DOF = 7

# Contact descent. The trigger is the one the bench this port grew on settled
# on after three predecessors broke: geometric contact is a *rate*, not a
# level. Controller lag drifts with arm configuration and a gravity-model
# force estimate under-reads in the same configurations, but neither ever
# *stops* the hand; a surface does. So contact is the hand's own descent rate
# collapsing below a fraction of the commanded speed (or a force rise, when
# the estimate is honest), sustained, and only after an initial travel that
# clears the controller's engagement transient.
CONTACT_DT_S = 0.01
CONTACT_BASELINE_START_TICKS = 30  # skip the engagement transient, then
CONTACT_BASELINE_END_TICKS = 50  # zero the force estimate over the next 0.2 s
CONTACT_SUSTAIN_TICKS = 5  # a release is irreversible; one noisy sample must not cause it
CONTACT_VELOCITY_FRACTION = 0.25
CONTACT_VELOCITY_WINDOW_S = 0.10
CONTACT_GUARD_TRAVEL_M = 0.018  # no verdict before the command has sunk this far
CONTACT_UNLOAD_TICKS = 30  # ease the command back up to the contact height
CONTACT_MIN_DESCENT_M = 0.025  # the guard travel plus room to measure a rate
CONTACT_MAX_SPEED_M_S = 0.05
REFLEX_RETRY_PAUSE_S = 0.5  # after a reflex: recover, settle, send the batch once more
REFLEX_RETRIES = 2          # how many times a batch is resent after reflexes before it is a failure
REFLEX_JOIN_PAUSE_S = 1.5   # after a reflex ends a controller: let its recovery log before the join
CONTACT_STIFFNESS = (1500.0, 1500.0, 600.0, 60.0, 60.0, 60.0)
"""Cartesian impedance during a contact descent: [x, y, z] N/m, [rx, ry, rz] Nm/rad.

z stays the soft axis: the press on contact is roughly K_z times the lag the
hand carries when it arrives. Measured 2026-11-17: at 300 N/m the hand
trails a 20 mm/s command by ~20 mm (joint friction), and with the plane held
stiff it also sticks in free air for long enough to look stalled -- three
false contacts 26 mm above the table at ~1 N. At 600 N/m a stick lasts about
half as long, and the stall path below waits it out.
"""
CONTACT_STALL_SUSTAIN_TICKS = 30
"""A stall must persist this long (0.3 s) to count; a force rise needs only
CONTACT_SUSTAIN_TICKS. Free-air stick-slip at 600 N/m and 20 mm/s lasts
~F_friction / (K_z * v) = 5 N / 12 N/s = 0.4 s at the most, usually less; a
real stop lasts for ever."""
CONTACT_SERIES_EVERY_TICKS = 5
"""Every fifth control tick is kept in the descent report (20 Hz)."""
CONTACT_MAX_LAG_M = 0.035
"""How far the command may lead the hand: the press budget on a miss.

Measured live (2026-11-17, bare hand, 20 mm/s, 300 N/m in z): the hand follows
the sinking command at roughly half its speed and trails it by 21 mm when a
floor 42 mm down is reached -- joint friction, not contact. A floor that bounds
the command therefore stops the hand short of anything; the floor bounds the
hand instead, and the command may run this far below it, which at 300 N/m is
about 10 N on a miss, under the arm's collision reflex thresholds.
"""


class RealFrankaEnv:
    """Single-arm Franka plant behind one actuation chokepoint."""

    def __init__(
        self,
        config: FrankaStationConfig,
        *,
        panda: Any,
        gripper: Any,
        camera: Optional[Camera],
        fk: Callable[[np.ndarray], np.ndarray],
        scene_camera: Optional[Camera] = None,
        impedance_controller: Optional[Callable[[], Any]] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.config = config
        self._clock = clock if clock is not None else time.monotonic
        """Monotonic seconds; the contact descent integrates by it. Tests inject one."""
        self._panda = panda
        self._gripper = gripper
        self._camera = camera
        self._scene_camera = scene_camera
        self._fk = fk
        self._impedance_controller = impedance_controller
        """Factory for a panda_py CartesianImpedance controller; tests inject a fake."""
        self._closed = False
        self._gripper_holding = False
        self.motions_executed = 0
        """Count of waypoint batches actually sent to the arm, for provenance."""
        self._gripper_lock = threading.Lock()
        """Serializes libfranka gripper traffic between commands and a sampler."""
        self.events: list[dict[str, Any]] = []
        """What the plant was told, host-timestamped, for the episode recorder."""

    # -- observation -------------------------------------------------------

    def fk_matrix(self, joint_positions: np.ndarray) -> np.ndarray:
        """4x4 EE transform in the base frame for one joint vector."""
        return np.asarray(self._fk(np.asarray(joint_positions, dtype=np.float64)))

    def get_observations(self) -> dict[str, np.ndarray]:
        """Measured joints, gripper, and the EE transform from those joints.

        The EE pose comes from forward kinematics of the measured joints -- the
        same function the guard uses -- rather than from the controller's own
        pose readout, so what the record says about the hand and what the guard
        checked about the hand can never be two different kinematic stories.
        """
        state = self._panda.get_state()
        q = np.asarray(state.q, dtype=np.float64).reshape(ARM_DOF)
        dq = np.asarray(state.dq, dtype=np.float64).reshape(ARM_DOF)
        width = self.read_gripper_width()
        return {
            "joint_pos": q,
            "joint_vel": dq,
            "gripper_width_m": float(width if width is not None else float("nan")),
            "ee_transform": self.fk_matrix(q),
        }

    def external_wrench(self) -> np.ndarray:
        """The estimated external wrench at the hand in the base frame (6), zeros if unexposed."""
        state = self._panda.get_state()
        wrench = getattr(state, "O_F_ext_hat_K", None)
        if wrench is None:
            return np.zeros(6, dtype=np.float64)
        return np.asarray(wrench, dtype=np.float64).reshape(6)

    def read_gripper_width(self, *, wait: bool = True) -> Optional[float]:
        """Measured finger opening, or None when a command holds the gripper and ``wait`` is off."""
        if not self._gripper_lock.acquire(blocking=wait):
            return None
        try:
            return float(self._gripper.read_once().width)
        finally:
            self._gripper_lock.release()

    # -- episode bookkeeping ----------------------------------------------

    def note_event(self, kind: str, **fields: Any) -> None:
        """Append one host-timestamped command event for the episode recorder."""
        self.events.append({"t_s": time.monotonic(), "kind": kind, **fields})

    def robot_time_ms(self) -> Optional[float]:
        """The robot clock, in milliseconds, read on the calling thread; None if unexposed."""
        stamp = getattr(self._panda.get_state(), "time", None)
        if stamp is None:
            return None
        for name in ("to_msec", "toMSec"):
            method = getattr(stamp, name, None)
            if callable(method):
                return float(method())
        method = getattr(stamp, "to_sec", None)
        if callable(method):
            return float(method()) * 1000.0
        try:
            return float(stamp)
        except (TypeError, ValueError):
            return None

    def start_arm_log(self, samples: int) -> bool:
        """Turn on panda_py's state log (fed by its control loop); False if the plant has none."""
        enable = getattr(self._panda, "enable_logging", None)
        if not callable(enable):
            return False
        enable(int(samples))
        return True

    def take_arm_log(self) -> dict[str, np.ndarray]:
        """Stop the log and return it as one array per channel, rows in time order."""
        get_log = getattr(self._panda, "get_log", None)
        if not callable(get_log):
            return {}
        raw = get_log()
        disable = getattr(self._panda, "disable_logging", None)
        if callable(disable):
            disable()
        log: dict[str, np.ndarray] = {}
        for key, rows in dict(raw).items():
            if len(rows) == 0:
                continue
            log[str(key)] = np.asarray(
                [np.asarray(row, dtype=np.float64).reshape(-1) for row in rows]
            )
        return log

    def capture(self) -> Optional[CameraFrame]:
        """The wrist camera's most recent frame, or None without a camera."""
        if self._camera is None:
            return None
        return self._camera.read()

    def capture_scene(self) -> Optional[CameraFrame]:
        """The fixed scene camera's most recent frame, when the station has one."""
        if self._scene_camera is None:
            return None
        return self._scene_camera.read()

    # -- the chokepoint ----------------------------------------------------

    def guard_waypoints(self, waypoints: np.ndarray) -> Optional[str]:
        """Why this batch must be refused, or None when every waypoint is safe."""
        points = np.asarray(waypoints, dtype=np.float64).reshape(-1, ARM_DOF)
        lower = np.asarray(self.config.joint_limits_lower)
        upper = np.asarray(self.config.joint_limits_upper)
        for index, q in enumerate(points):
            if not np.all(np.isfinite(q)):
                return f"waypoint {index} is not finite"
            if np.any(q < lower - 1e-9) or np.any(q > upper + 1e-9):
                return f"waypoint {index} exceeds a joint limit"
            position = self.fk_matrix(q)[:3, 3]
            if not self.config.workspace.contains(position):
                return (
                    f"waypoint {index} puts the end effector at "
                    f"{np.round(position, 3).tolist()}, outside the workspace box"
                )
        return None

    def _decimate_waypoints(self, points: np.ndarray) -> np.ndarray:
        """Drop the waypoints the motion generator would only stop at.

        The generator comes to rest at every waypoint it is given, so a dense
        planned path executes as a crawl (measured live: a 19-waypoint descent
        took ~2 minutes). Between two kept waypoints the generator's
        time-synchronized profiles move each joint monotonically toward the
        next rest, staying near the joint-space chord, so a planned waypoint
        that lies on that chord adds a stop and no information. Keep the first
        and last, extend each chord greedily while every dropped waypoint
        stays within the configured tolerance of it, and re-run the workspace
        guard over the chords themselves (sampled at the dropped waypoints'
        parameters); if the sparser path cannot be shown safe, send the dense
        one and be slow rather than clever.
        """
        tolerance = float(self.config.waypoint_chord_tolerance_rad)
        if tolerance <= 0.0 or points.shape[0] <= 2:
            return points

        def deviation(start: np.ndarray, end: np.ndarray, middle: np.ndarray) -> float:
            chord = end - start
            length_sq = float(chord @ chord)
            if length_sq < 1e-12:
                return float(np.linalg.norm(middle - start))
            fraction = float(np.clip((middle - start) @ chord / length_sq, 0.0, 1.0))
            return float(np.linalg.norm(middle - (start + fraction * chord)))

        kept = [0]
        anchor = 0
        end = 1
        while end < points.shape[0] - 1:
            within = all(
                deviation(points[anchor], points[end + 1], points[middle]) <= tolerance
                for middle in range(anchor + 1, end + 1)
            )
            if within:
                end += 1
            else:
                kept.append(end)
                anchor = end
                end += 1
        kept.append(points.shape[0] - 1)
        if len(kept) == points.shape[0]:
            return points
        # The executed path follows the chords, not the planned waypoints the
        # chords replace: guard the chords at the dropped parameters too.
        samples = []
        for left, right in zip(kept, kept[1:]):
            for middle in range(left + 1, right):
                fraction = (middle - left) / (right - left)
                samples.append(points[left] + fraction * (points[right] - points[left]))
        if samples and self.guard_waypoints(np.asarray(samples)) is not None:
            return points
        return points[kept]

    def execute_waypoints(
        self, waypoints: np.ndarray, *, speed_factor: Optional[float] = None
    ) -> dict[str, Any]:
        """Send one guarded waypoint batch through the panda_py motion generator.

        The generator retimes the batch under the robot's own limits, so the
        trajectory's requested ``dt_s`` is advisory here and is recorded as
        such. That trade is deliberate: a hobbyist bench gets Franka's reflexes
        and jerk limits on every motion instead of a hand-rolled 100 Hz loop.
        """
        points = np.asarray(waypoints, dtype=np.float64).reshape(-1, ARM_DOF)
        refusal = self.guard_waypoints(points)
        if refusal is not None:
            raise ValueError(refusal)
        sent = self._decimate_waypoints(points)
        factor = float(speed_factor if speed_factor is not None else self.config.speed_factor)
        started = time.monotonic()
        self.note_event(
            "waypoints",
            target_q=[round(float(v), 5) for v in sent[-1]],
            waypoints_sent=int(sent.shape[0]),
            speed_factor=factor,
        )
        converged = bool(
            self._panda.move_to_joint_position([row for row in sent], speed_factor=factor)
        )
        self.motions_executed += 1
        self.note_event("waypoints_done", converged=converged)
        retries = 0
        while not converged and retries < REFLEX_RETRIES:
            # A reflex aborts the motion and the generator reports
            # non-convergence. The communication reflex is transient (2026-11-19:
            # a dozen in a day on an idle-clean link, each recovered by the
            # robot within a second, each one costing a run), so recover and
            # send the same batch again, twice if need be (run 0431: two in a
            # row on one hover); a failure after that is the caller's.
            retries += 1
            recover = getattr(self._panda, "recover", None)
            if callable(recover):
                recover()
            time.sleep(REFLEX_RETRY_PAUSE_S)
            self.note_event(
                "waypoints_retry", target_q=[round(float(v), 5) for v in sent[-1]], attempt=retries
            )
            converged = bool(
                self._panda.move_to_joint_position([row for row in sent], speed_factor=factor)
            )
            self.motions_executed += 1
            self.note_event("waypoints_done", converged=converged, retried=True)
        if not converged:
            raise RuntimeError(
                "the motion generator reported non-convergence at the final waypoint"
            )
        return {
            "waypoints": int(points.shape[0]),
            "waypoints_sent": int(sent.shape[0]),
            "speed_factor": factor,
            "elapsed_s": round(time.monotonic() - started, 3),
            "retimed_by": "panda_py_motion_generator",
        }

    def _make_impedance_controller(self) -> Any:
        if self._impedance_controller is not None:
            return self._impedance_controller()
        from panda_py import controllers

        return controllers.CartesianImpedance()

    def descend_until_contact(
        self,
        *,
        floor_z: float,
        speed_m_s: Optional[float] = None,
        contact_force_n: Optional[float] = None,
        min_descent_m: Optional[float] = None,
    ) -> dict[str, Any]:
        """Lower the hand straight down until it meets something, or the floor.

        ``min_descent_m`` is how far above the floor the hand must start
        (``CONTACT_MIN_DESCENT_M`` by default); a retry after a reflex passes
        0 so that it may finish a descent the reflex cut short near the floor
        (run 0389: aborted 1.9 cm above the block, the retry refused).

        The second, and only other, way this package moves the arm. The motion
        generator behind :meth:`execute_waypoints` is open-loop and contact-
        blind: a place commanded to a camera-measured height meets the table a
        few millimetres early and the collision reflex aborts the move (three
        of four live placements, 2026-11-14/15). Here the hand rides a
        cartesian impedance controller whose command sinks at a steady rate
        from the current pose, in the current x, y and orientation, and stops
        when the hand stops following it (see the module constants for the
        trigger), then eases the command back to the contact height so a
        release does not let the stored spring drive the hand on. The floor is
        the stop for a miss, not the target: a caller puts it well below the
        contact it expects, and the workspace box bounds it like any waypoint.
        It bounds the *hand*: the command may lead the hand by up to
        ``CONTACT_MAX_LAG_M`` (the hand trails a sinking command by ~20 mm on
        this arm, friction, not contact), and a stop is ``"floor"`` when the
        hand reaches the floor, ``"press"`` when the command runs out of lead
        first, ``"contact"`` when the hand stops following.
        """
        speed = float(
            speed_m_s if speed_m_s is not None else self.config.contact_descent_speed_m_s
        )
        force_limit = float(
            contact_force_n if contact_force_n is not None else self.config.contact_force_n
        )
        if not 0.0 < speed <= CONTACT_MAX_SPEED_M_S:
            raise ValueError(f"descent speed must be in (0, {CONTACT_MAX_SPEED_M_S}] m/s")
        if force_limit <= 0.0:
            raise ValueError("contact force must be positive")
        floor_z = float(floor_z)
        q = np.asarray(self._panda.get_state().q, dtype=np.float64).reshape(ARM_DOF)
        here = self.fk_matrix(q)[:3, 3]
        if not self.config.workspace.contains([here[0], here[1], floor_z]):
            raise ValueError(
                f"floor {floor_z:.3f} m is outside the workspace box at "
                f"x={here[0]:.3f}, y={here[1]:.3f}"
            )
        start_z = float(here[2])
        min_descent = float(min_descent_m if min_descent_m is not None else CONTACT_MIN_DESCENT_M)
        if start_z - floor_z < min_descent:
            raise ValueError(
                f"the hand at z={start_z:.3f} m is within {min_descent} m of the "
                f"floor {floor_z:.3f} m; start the descent higher"
            )
        controller = self._make_impedance_controller()
        # z is soft on purpose (the contact spring); x, y and the orientation
        # are not. At 400 N/m the hand drifted ~13 mm toward the base during
        # every 3 cm placement descent (runs 0007-0010, 2026-11-17: pre-place
        # at x 0.500, hand at x 0.486-0.489 on contact), joint friction the
        # lateral spring could not overcome, and the block went where the hand
        # went. Stiffer in the plane, the same friction costs a few millimetres.
        controller.set_impedance(np.diag(CONTACT_STIFFNESS))
        controller.set_nullspace_stiffness(1.0)
        # The controller commands in its own pose readout; the guard above
        # judged the floor by the same forward kinematics every waypoint gets.
        position = np.asarray(self._panda.get_position(), dtype=np.float64)
        x, y = float(position[0]), float(position[1])
        orientation = self._panda.get_orientation()
        tz = float(position[2])
        actual = tz
        guard_z = tz - CONTACT_GUARD_TRAVEL_M
        window = max(2, int(CONTACT_VELOCITY_WINDOW_S / CONTACT_DT_S))
        fz0 = float(self._panda.get_state().O_F_ext_hat_K[2])
        baseline: list[float] = []
        recent: list[float] = []
        settle = 0
        above = 0
        stalled_for = 0
        unload_ticks = 0
        peak_rise = 0.0
        contact_z: Optional[float] = None
        stop = "timeout"
        command_floor = floor_z - CONTACT_MAX_LAG_M
        runtime = (tz - command_floor) / speed + 2.0
        started = self._clock()
        # A downsampled record of the command, the hand and the force, so a
        # floor stop can be told apart from a trigger that failed to fire.
        series: list[list[float]] = []
        tick = 0
        fz = fz0
        # The loop is paced by panda_py's context at 100 Hz, but a busy
        # interpreter (the episode recorder's threads, live) slowed it to
        # ~45 Hz. The command therefore sinks by wall-clock time, not by tick,
        # and the stall window is a wall-clock window, so the descent speed
        # and the trigger mean what they say at any achieved rate.
        last_tick_t = started
        recent_t: list[float] = []
        self.note_event("descend", floor_z=round(floor_z, 4), speed_m_s=speed, start_z=round(start_z, 4))
        self._panda.start_controller(controller)
        aborted = False
        try:
            with self._panda.create_context(
                frequency=int(round(1.0 / CONTACT_DT_S)), max_runtime=runtime
            ) as context:
                while context.ok():
                    now = self._clock()
                    dt = min(max(now - last_tick_t, 0.0), 5 * CONTACT_DT_S)
                    last_tick_t = now
                    actual = float(self._panda.get_position()[2])
                    if tick % CONTACT_SERIES_EVERY_TICKS == 0:
                        series.append(
                            [round(now - started, 2), round(tz, 4), round(actual, 4), round(fz, 2)]
                        )
                    tick += 1
                    if contact_z is not None:
                        tz = min(
                            contact_z,
                            tz + (contact_z - tz) / max(CONTACT_UNLOAD_TICKS - unload_ticks, 1),
                        )
                        controller.set_control(np.array([x, y, tz]), orientation, q)
                        unload_ticks += 1
                        if unload_ticks >= CONTACT_UNLOAD_TICKS:
                            break
                        continue
                    tz = max(command_floor, tz - speed * dt)
                    controller.set_control(np.array([x, y, tz]), orientation, q)
                    fz = float(self._panda.get_state().O_F_ext_hat_K[2])
                    if settle < CONTACT_BASELINE_END_TICKS:
                        settle += 1
                        if settle > CONTACT_BASELINE_START_TICKS:
                            baseline.append(fz)
                        continue
                    if baseline:
                        fz0, baseline = float(np.mean(baseline)), []
                    recent.append(actual)
                    recent_t.append(now)
                    while recent_t and now - recent_t[0] > CONTACT_VELOCITY_WINDOW_S + 1e-6:
                        recent.pop(0)
                        recent_t.pop(0)
                    span = now - recent_t[0]
                    descending = (
                        (recent[0] - actual) / span
                        if len(recent) >= window and span > 0.0
                        else speed
                    )
                    rise = abs(fz - fz0)
                    peak_rise = max(peak_rise, rise)
                    stalled = descending < CONTACT_VELOCITY_FRACTION * speed
                    above = above + 1 if rise > force_limit else 0
                    stalled_for = stalled_for + 1 if stalled else 0
                    if tz < guard_z and (
                        above >= CONTACT_SUSTAIN_TICKS
                        or stalled_for >= CONTACT_STALL_SUSTAIN_TICKS
                    ):
                        contact_z = actual
                        stop = "contact"
                        continue
                    if actual <= floor_z + 1e-4:
                        stop = "floor"
                        break
                    if tz <= command_floor + 1e-4:
                        stop = "press"
                        break
        except BaseException:
            aborted = True
            raise
        finally:
            if aborted:
                # A reflex ended the controller's thread early and panda_py
                # runs its automatic error recovery on that thread, logging
                # through Python as it goes. stop_controller joins the thread
                # with the interpreter lock held, so the recovery's log call
                # waits for the lock and the join waits for the recovery:
                # the hang of runs 0395 and 0422 (gdb: std::thread::join
                # under panda_py). A sleep releases the lock long enough for
                # the recovery to finish before the join.
                time.sleep(REFLEX_JOIN_PAUSE_S)
            self._panda.stop_controller()
        self.motions_executed += 1
        end_z = contact_z if contact_z is not None else actual
        # Where the hand ended in the plane: the command holds x, y at 400 N/m
        # while the hand sinks, and the same joint friction that lets it trail
        # the command in z can let it drift sideways. Six of six round-trip
        # placements landed 11-14 mm short in x (2026-11-17); this says whether
        # the descent is where that goes.
        try:
            end_xy = np.asarray(self._panda.get_position(), dtype=np.float64)[:2]
            drift_xy = [round(float(end_xy[0] - x), 4), round(float(end_xy[1] - y), 4)]
        except Exception:
            drift_xy = None
        self.note_event(
            "descend_done",
            stop=stop,
            contact_z=None if contact_z is None else round(contact_z, 4),
            end_z=round(end_z, 4),
        )
        return {
            "contact": contact_z is not None,
            "contact_z": None if contact_z is None else round(contact_z, 4),
            "start_z": round(start_z, 4),
            "floor_z": round(floor_z, 4),
            "travel_m": round(start_z - end_z, 4),
            "stop": stop,
            "force_rise_n": round(peak_rise, 2),
            "lag_m": round(actual - tz, 4),
            "drift_xy_m": drift_xy,
            "speed_m_s": speed,
            "elapsed_s": round(self._clock() - started, 3),
            "tick_hz": round(tick / max(self._clock() - started, 1e-6), 1),
            "controller": "panda_py_cartesian_impedance",
            "series_columns": ["t_s", "command_z", "hand_z", "fz_n"],
            "series": series,
        }

    def recover_from_reflex(self) -> None:
        """Clear the robot's reflex error and let it settle before another command."""
        recover = getattr(self._panda, "recover", None)
        if callable(recover):
            recover()
        time.sleep(REFLEX_RETRY_PAUSE_S)
        self.note_event("reflex_recovered")

    def move_home(self, *, speed_factor: Optional[float] = None) -> dict[str, Any]:
        return self.execute_waypoints(
            np.asarray(self.config.home_joints, dtype=np.float64).reshape(1, ARM_DOF),
            speed_factor=speed_factor,
        )

    # -- gripper -----------------------------------------------------------

    def set_gripper_width(self, width_m: float) -> dict[str, Any]:
        """Command the fingers under the gripper lock and note what happened."""
        with self._gripper_lock:
            report = self._set_gripper_width_locked(width_m)
        self.note_event(
            "gripper",
            **{key: report.get(key) for key in ("commanded_width_m", "measured_width_m", "ok", "noop")},
        )
        return report

    def _set_gripper_width_locked(self, width_m: float) -> dict[str, Any]:
        """Width move when opening, force-closure when closing onto something.

        The Franka gripper's plain ``move`` reports failure when an object
        stops the fingers, which is exactly the case a manipulation program
        means by "close". A command below the current width therefore grasps
        with the station's configured force and generous epsilons: holding an
        object of any width within the commanded travel counts as success,
        matching how every other embodiment's normalized gripper behaves.
        """
        width = float(np.clip(width_m, 0.0, self.config.gripper_max_width_m))
        current = float(self._gripper.read_once().width)
        if abs(width - current) < 0.002:
            # Commanding the width the fingers already hold is a no-op, and it
            # must be one: planned trajectories carry per-waypoint gripper
            # values, and re-commanding the held width after a carry made
            # libfranka fail the whole, otherwise successful, motion
            # ("libfranka gripper: Command failed!" while gripping a block).
            return {
                "commanded_width_m": round(width, 4),
                "measured_width_m": round(current, 4),
                "ok": True,
                "noop": "already at width",
            }
        if width < current - 1e-4:
            if self._gripper_holding:
                # The fingers are already force-closed on something (they
                # stopped above the commanded width). Planned trajectories
                # re-assert the program's last close command after every
                # motion, and libfranka rejects a fresh grasp mid-grip
                # ("libfranka gripper: Command failed!"); physically there is
                # nothing to do, so this holds the verdict a grasp would want.
                return {
                    "commanded_width_m": round(width, 4),
                    "measured_width_m": round(current, 4),
                    "ok": True,
                    "noop": "already grasping",
                }
            grasped = bool(
                self._gripper.grasp(
                    width,
                    speed=0.05,
                    force=float(self.config.gripper_force_n),
                    epsilon_inner=self.config.gripper_max_width_m,
                    epsilon_outer=self.config.gripper_max_width_m,
                )
            )
            self._gripper_holding = grasped
        else:
            grasped = bool(self._gripper.move(width=width, speed=0.05))
            self._gripper_holding = False
        return {
            "commanded_width_m": round(width, 4),
            "measured_width_m": round(float(self._gripper.read_once().width), 4),
            "ok": grasped,
        }

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for camera in (self._camera, self._scene_camera):
            if camera is not None:
                try:
                    camera.close()
                except Exception:
                    pass


__all__ = ["ARM_DOF", "RealFrankaEnv"]
