---
task: "Pick up the red block and set it down on a clear patch of table."
---

# Moving the red block

Red is the block this bench's own colour pipeline has never seen; every grasp
of it so far began from a human's words. The task is simple on purpose: find
it with `localize_object`, re-localize from directly above before committing,
grasp it across its short axis at its measured yaw, and set it down on a spot
of bare table at least 0.10 m from every block you can see. Verify by
re-localizing red at the destination. Blocks are about 55 x 28 x 27 mm; the
table sits near z = 0.017; a lying block's top near 0.044. Reject implausible
localizations rather than acting on them, and a failed grasp (closed near
minimum width) ends the attempt honestly after going home.
