# Does the wrist camera still agree with the scene camera? Survey from above
# the row and localize the same blocks through both cameras; a constant
# offset between them is a shifted wrist mount (hand-eye), not a moved block.
# Motivation, 2026-11-17: after run 0010's reflex the wrist readings of an
# untouched block moved ~3.5 cm in y, and every grasp since missed by that.

SURVEY_Z = 0.33
LONG_MOVE_STEPS = 600
SURVEYS = [[0.50, 0.0], [0.45, -0.12]]  # the first matches the round trip's survey
COLOURS = ("blue", "green", "red")


def read(colour, camera):
    found = localize_object(colour + " block", camera_name=camera)
    if not found.ok:
        return None
    position = [round(float(v), 4) for v in found.geometry.pose.position]
    extents = [round(float(v), 4) for v in found.geometry.extents]
    return {"position": position, "extents": extents}


go_home()  # start from home: a run can begin wherever the last one left the arm (run 0121)
readings = []
for sx, sy in SURVEYS:
    moved = move_to_pose({"position": [sx, sy, SURVEY_Z], "frame": "base",
                          "rpy_deg": [180.0, 0.0, 0.0]}, max_steps=LONG_MOVE_STEPS)
    entry = {"survey": [sx, sy], "moved": bool(moved.ok), "blocks": {}}
    if moved.ok:
        for colour in COLOURS:
            wrist = read(colour, "wrist")
            scene = read(colour, "scene")
            delta = None
            if wrist is not None and scene is not None:
                delta = [round(wrist["position"][i] - scene["position"][i], 4) for i in range(3)]
            entry["blocks"][colour] = {"wrist": wrist, "scene": scene, "wrist_minus_scene": delta}
    readings.append(entry)
go_home()
result = {"ok": all(r["moved"] for r in readings), "readings": readings}
