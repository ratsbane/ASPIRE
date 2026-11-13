---
task: "Stack the blue block on the green block."
---

# Block stacking on the Franka bench

## Goal and setup

Two wooden blocks, about 55 x 28 x 27 mm, lie loose on the table. Put the blue
one squarely on top of the green one and end with the gripper clear of the pile.
The table surface sits near z = 0.017 in the base frame; a lying block's top is
near z = 0.044, a two-block pile's near z = 0.071.

Use `localize_object` for metric grounding and reject a localization whose
centroid falls outside the workspace box or whose extents are implausible for
these blocks. The wrist is the only camera: it sees best looking straight down
from 0.25-0.35 m, and a localization taken from directly above a block is more
trustworthy than one taken at a slant from far away. Act through
`cap_move_to_pose` and `cap_set_gripper`; grasp a block across its short axis
with the fingers' closing line turned to the block's long axis, close before
lifting, and keep every approach vertical near the table.

## Grasp and place

1. Observe from a survey pose above the table centre, localize both blocks,
   and re-localize the pick block from directly above it before committing.
2. Approach the blue block from a hover 0.08 m above it, descend to an
   end-effector height of table + 0.010 m, close, and lift straight up.
3. Carry at 0.20 m or higher, descend over the green block to its measured top
   plus the held block's hang, release with a narrow opening, and retreat up.
4. Verify by re-localizing blue: success is its centroid risen by about one
   block height and laterally within 0.015 m of green's centre. Report what
   was measured, not what was intended.

## Recovery

A failed grasp (gripper closed near its minimum with nothing held) means the
block was shoved: re-localize it before trying again, at most twice. If a
localization cannot be trusted or the scene stops matching the task, stop and
report rather than guessing.
