# Put the closed fingertips on a clear patch of table and leave the arm
# there, so a human can measure the hand (2026-11-17: how far the wrist
# camera's housing sits above the pads). No go_home: the next run homes it.

SPOT = [0.60, -0.07]     # front of the table, clear of the row spots and their camera footprints
SAFE_Z = 0.10
LONG_MOVE_STEPS = 600

close_gripper()
hover = move_to_pose({"position": [SPOT[0], SPOT[1], SAFE_Z], "frame": "base",
                      "rpy_deg": [180.0, 0.0, 0.0]}, max_steps=LONG_MOVE_STEPS)
touch = None
if hover.ok:
    touch = franka_real.descend_to_contact(floor_z=0.004)
result = {
    "ok": bool(hover.ok and touch is not None and touch.ok),
    "hover": bool(hover.ok),
    "stop": None if touch is None or not touch.ok else touch.diagnostics["stop"],
    "contact_z": None if touch is None or not touch.ok else touch.diagnostics["contact_z"],
    "note": "arm left on the table for measurement; the next run homes it",
}
