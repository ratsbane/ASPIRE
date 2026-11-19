import math

# Where can the hand get down to the table, pointing straight down? Every
# band in the stack_blocks programs (the pick's reach floor, the touch
# calibration's work box, the sweep's working band) was set from refusals
# seen in passing; this measures them. Each target is approached from a
# hover above it, so the path is short and the refusal, if any, is the
# target's own: IK, the workspace box, or the step budget. Nothing is
# touched: the low height stays above a lying block.

HOVER_Z = 0.20
HEIGHTS = [0.10, 0.05]
XS = [0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42]
YS = [0.0, -0.20, -0.26, -0.30]
LONG_MOVE_STEPS = 900
DOWN = [0.0, 1.0, 0.0, 0.0]


def pose(x, y, z):
    return {"position": [x, y, z], "frame": "base", "quaternion_wxyz": DOWN}


def attempt(x, y, z):
    moved = move_to_pose(pose(x, y, z), max_steps=LONG_MOVE_STEPS)
    if moved.ok:
        hand = get_robot_state().end_effector_poses["primary"].position
        return {"ok": True, "hand": [round(float(v), 4) for v in hand]}
    return {"ok": False, "error": str(moved.error.message)[:140], "code": str(moved.error.code)}


go_home()
readings = []
for y in YS:
    for x in XS:
        hover = move_to_pose(pose(x, y, HOVER_Z), max_steps=LONG_MOVE_STEPS)
        entry = {"x": x, "y": y, "hover": bool(hover.ok)}
        if not hover.ok:
            entry["hover_error"] = str(hover.error.message)[:140]
            readings.append(entry)
            continue
        for z in HEIGHTS:
            entry[str(z)] = attempt(x, y, z)
            if not entry[str(z)]["ok"]:
                break
        move_to_pose(pose(x, y, HOVER_Z), max_steps=LONG_MOVE_STEPS)
        readings.append(entry)
go_home()

reach = {}
for entry in readings:
    low = entry.get("0.05") or {}
    high = entry.get("0.1") or {}
    key = str(entry["y"])
    if low.get("ok"):
        reach.setdefault(key, {})["x_min_at_0.05"] = min(entry["x"], reach.get(key, {}).get("x_min_at_0.05", 9.0))
    if high.get("ok"):
        reach.setdefault(key, {})["x_min_at_0.10"] = min(entry["x"], reach.get(key, {}).get("x_min_at_0.10", 9.0))
result = {"ok": all(e["hover"] for e in readings), "reach": reach, "readings": readings}
