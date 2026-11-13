---
task: "Pick one wooden block out of the box of loose blocks and set it on the table."
---

# Picking from clutter on the Franka bench

## Goal and setup

A box of loose, unpainted wooden blocks (each about 55 x 28 x 27 mm) sits on
the table. Take one block out and set it down flat at the task's pick target,
ending with the gripper clear. The box footprint, rim height and target come
from `franka_real.get_task_metadata()["goal"]` -- the registry is the one
source of truth the success predicate also reads. Until `goal["surveyed"]` is
true those numbers are placeholders: run a no-motion survey with the box on
the bench before trusting them.

Colour rules see nothing here: segmentation must come from the model service
(`segment_text` with "wooden block"), which returns every instance it finds.
`localize_object` would silently hand back only the best-scored one -- in
clutter, do the instance selection yourself: back-project each mask with
`mask_to_point_cloud(seg, "wrist", "base")` (the target frame must be named;
the default is not this station's base), estimate geometry, and reject blobs
whose centroid is outside the box footprint or whose extents are implausible
for one block.

## Pick policy

1. Survey from above the box centre, high enough to clear the rim.
2. Choose the candidate with the **highest top face**; among near-equals,
   prefer one whose neighbours are at least 0.035 m away in xy (finger
   clearance). Topmost-first also un-clutters the box with every pick.
3. Hover above the rim, descend vertically to the block's measured top minus
   0.017 m, close, and lift straight back above the rim before any lateral
   motion -- the box walls are obstacles the workspace guard does not know.
4. A close on air means the block was shoved: open, re-survey, re-choose, at
   most three attempts.
5. Carry high, place at a height that preserves the held block's bottom
   clearance: the scan's footprint says how the block hangs (a footprint
   showing the long ~55 mm side lies flat, 28 mm tall in hand; a narrower
   footprint stands on end, 55 mm tall), and lowering a standing block to
   flat-block height drives it into the table and fires the cartesian reflex
   (measured live on the first pick). Release narrow, retreat, verify by
   re-segmenting above the target: success is a block instance within 0.03 m
   of it. Report what was measured, including the block's orientation.

## Recovery

Refused approach or descent aborts through `go_home`. Tall moves need a
generous `max_steps`. If the box yields no candidates the scene does not match
the task: stop and report rather than hunting the table for strays.
