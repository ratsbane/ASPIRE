# Does the eye level itself at survey height? Survey above the row, read the
# adapter's levelling report, and localize the row blocks with it in effect.
# The wrist D405's depth is only good to ~0.5 m, so the parked pose is no
# test; the survey height is.

SURVEY_Z = 0.33
LONG_MOVE_STEPS = 600

go_home()  # start from home: a run can begin wherever the last one left the arm (run 0121)
readings = []
for sx, sy in ([0.50, 0.0], [0.45, -0.12]):
    moved = move_to_pose({"position": [sx, sy, SURVEY_Z], "frame": "base",
                          "rpy_deg": [180.0, 0.0, 0.0]}, max_steps=LONG_MOVE_STEPS)
    entry = {"survey": [sx, sy], "moved": bool(moved.ok)}
    if moved.ok:
        get_observation()
        entry["level"] = franka_real.get_controller_metadata()["wrist_level"]
        for colour in ("blue", "green", "red"):
            found = localize_object(colour + " block")
            entry[colour] = None if not found.ok else {
                "position": [round(float(v), 4) for v in found.geometry.pose.position],
                "extents": [round(float(v), 4) for v in found.geometry.extents]}
    readings.append(entry)
go_home()
result = {"ok": all(r["moved"] for r in readings), "readings": readings}
