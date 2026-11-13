---
task: "Take the blue block off the green block and set it on clear table."
---

# Block unstacking on the Franka bench

## Goal and setup

The blue block (about 55 x 28 x 27 mm) sits on top of the green one. Lift it
off and set it down flat on a clear patch of table, ending with the gripper
clear and both blocks one-high. This is the station's own scene reset: stack
and unstack together make unattended multi-trial campaigns possible.

The table surface sits near z = 0.017 in the base frame; a lying block's top is
near z = 0.044, a two-block pile's near z = 0.071. Use `localize_object` for
metric grounding and reject a localization whose centroid falls outside the
workspace box. The wrist camera sees best looking straight down from
0.25-0.35 m; re-localize the pile from directly above it before committing.
Green may be fully hidden beneath blue -- the pile's position is blue's
position, and green's absence from a top-down view is not an error.

## Grasp and place

1. Observe from a survey pose above the table centre and localize blue; the
   pile is where blue is. Localize green and red too if visible, so the
   destination stays clear of everything.
2. Choose a destination at least 0.10 m from the pile and from every other
   localized block.
3. Approach blue from a hover above its measured top, descend so the fingers
   engage the top block only -- end-effector height near the pile's top minus
   0.017 m, well above green -- close, and lift straight up.
4. Carry at 0.20 m or higher, lower to a table-level place (end-effector at
   table + 0.014 m), release with a narrow opening, retreat up, and open.
5. Verify by re-localizing blue over the destination: success is its centroid
   back down at one-block height and laterally within 0.02 m of the chosen
   spot. Report what was measured, not what was intended.

## Recovery

A failed grasp (gripper closed near its minimum with nothing held) means the
block was shoved or the pile toppled: re-localize before trying again, at most
twice. A refused approach or descent must abort the plan through `go_home`,
never be shrugged past. Tall moves (lift, carry, lower, retreat) need a
generous `max_steps`; the default budget refuses them.
