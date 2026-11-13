---
task: "Build a wall of wooden blocks across the table, one block per slot."
---

# Wall building on the Franka bench

## Goal and setup

A box of loose wooden blocks (each about 55 x 28 x 27 mm) sits on the table.
Build a single-course wall: one block laid flat in each slot, end to end along
the wall line. The wall line, slot pitch and count, box footprint and rim
height all come from `franka_real.get_task_metadata()["goal"]` -- the same
spec the success predicate measures against, so build what it says, not what
looks nice. Until `goal["surveyed"]` is true the geometry is a placeholder:
survey with the box on the bench first.

The wall runs along +x so every block is placed at a yaw near zero, well
inside the proven wrist range; a block's long axis lies along the wall. Picks
follow the clutter policy (see the pick_from_clutter context): segment
"wooden block" instances yourself, back-project with an explicit `"base"`
target frame, reject candidates outside the box or with implausible extents,
exclude blocks the hand envelope cannot clear from the measured wall lines
(`box_x`/`box_y` -- bumping the rim fires the cartesian reflex), take the
highest top face with finger clearance, and always lift straight back above
the rim before moving laterally. When only wall-huggers remain, stop and
report `wall_blocked`: a partial wall beats a bumped basket.

## Build policy

1. For each slot, in order: pick a block from the box (at most two grasp
   attempts per slot), carry high, place at the slot centre with the wall's
   yaw, release narrow, retreat. Prefer candidates whose scanned footprint
   shows the long side (lying flat); when only standing blocks remain, take
   one, place it at its own height (its bottom hangs 55 mm below the top,
   not 28 -- the flat-height assumption fired the cartesian reflex live) and
   report the orientation honestly.
2. Verify each placement by re-segmenting above the slot: a block instance
   within 0.03 m of the slot centre. Record the measured position; a slot
   that cannot be verified is reported, not papered over.
3. If the box runs out of candidates, stop and report the slots completed.
   A partial wall honestly reported beats a guessed-at full one.

## Recovery

A refused approach or descent aborts the whole build through `go_home` -- a
wall program that shrugs past refusals will stack failures into later slots.
A close on air re-surveys the box and retries once before moving on. Tall
moves need a generous `max_steps` (the plan's waypoint count is checked
against it before decimation). Expect this task to be long even with brisk
motion: tens of moves plus a scan per slot; budget the run timeout
accordingly.
