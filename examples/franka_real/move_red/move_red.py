import math

TABLE_Z = 0.017
GRASP_EE_Z = TABLE_Z + 0.010
HOVER = 0.08
CARRY_Z = 0.22
SURVEY = {"position": [0.42, 0.0, 0.31], "frame": "base", "rpy_deg": [180.0, 0.0, 0.0]}
CANDIDATES = ((0.37, 0.10), (0.40, -0.05), (0.52, 0.05), (0.36, -0.08), (0.50, 0.18))

FAILED_MOVE = {"stage": None}


LONG_MOVE_STEPS = 600


def must(move_result, stage):
    if not move_result.ok and FAILED_MOVE["stage"] is None:
        FAILED_MOVE["stage"] = stage
        FAILED_MOVE["error"] = str(move_result.error.message)[:160]
    return move_result.ok


def down_quat(yaw_rad):
    return [0.0, math.cos(yaw_rad / 2.0), math.sin(yaw_rad / 2.0), 0.0]


def box_yaw(geometry):
    matrix = geometry.pose.as_matrix()
    axis = 0 if geometry.extents[0] >= geometry.extents[1] else 1
    return math.atan2(float(matrix[1, axis]), float(matrix[0, axis]))


def locate(name, above=None):
    if above is not None:
        move_to_pose({"position": [above[0], above[1], 0.31], "frame": "base",
                      "rpy_deg": [180.0, 0.0, 0.0]})
    found = localize_object(name)
    if not found.ok:
        return None
    centre = [float(v) for v in found.geometry.pose.position]
    if not (0.25 < centre[0] < 0.66 and -0.33 < centre[1] < 0.33):
        return None
    return found.geometry


move_to_pose(SURVEY)
red = locate("red block")
others = [g for g in (locate("blue block"), locate("green block")) if g is not None]
if red is None:
    result = {"ok": False, "stage": "find", "note": "red not localized"}
else:
    red = locate("red block", above=red.pose.position) or red
    rx, ry = float(red.pose.position[0]), float(red.pose.position[1])
    keep_clear = [(float(g.pose.position[0]), float(g.pose.position[1])) for g in others]
    # The program sandbox has no `next`; a plain loop reads better anyway.
    dest = None
    for x, y in CANDIDATES:
        clear = all(math.hypot(x - ox, y - oy) >= 0.10 for ox, oy in keep_clear)
        if clear and math.hypot(x - rx, y - ry) >= 0.08:
            dest = (x, y)
            break
    if dest is None:
        result = {"ok": False, "stage": "destination", "note": "no clear candidate"}
    else:
        yaw = box_yaw(red)
        quat = down_quat(yaw)
        open_gripper()
        reached = must(move_to_pose({"position": [rx, ry, GRASP_EE_Z + HOVER],
                                     "frame": "base", "quaternion_wxyz": quat}), "hover") \
            and must(move_to_pose({"position": [rx, ry, GRASP_EE_Z], "frame": "base",
                                   "quaternion_wxyz": quat}), "descend")
        if not reached:
            go_home()
            result = {"ok": False, "stage": FAILED_MOVE["stage"],
                      "error": FAILED_MOVE.get("error")}
        else:
            set_gripper(0.25)
            held = get_robot_state().gripper_positions["primary"]
            must(move_to_pose({"position": [rx, ry, CARRY_Z], "frame": "base",
                               "quaternion_wxyz": quat}, max_steps=LONG_MOVE_STEPS), "lift")
            if held < 0.08:
                open_gripper()
                go_home()
                result = {"ok": False, "stage": "grasp", "note": "closed on air",
                          "held": held}
            else:
                dx, dy = dest
                must(move_to_pose({"position": [dx, dy, CARRY_Z], "frame": "base",
                                   "quaternion_wxyz": quat}, max_steps=LONG_MOVE_STEPS), "carry")
                must(move_to_pose({"position": [dx, dy, GRASP_EE_Z + 0.004],
                                   "frame": "base", "quaternion_wxyz": quat}), "lower")
                set_gripper(min(1.0, held + 0.2))
                must(move_to_pose({"position": [dx, dy, CARRY_Z], "frame": "base",
                                   "quaternion_wxyz": quat}, max_steps=LONG_MOVE_STEPS), "retreat")
                open_gripper()
                check = locate("red block", above=[dx, dy])
                near = check is not None and math.hypot(
                    float(check.pose.position[0]) - dx,
                    float(check.pose.position[1]) - dy) < 0.02
                go_home()
                result = {"ok": bool(near), "stage": "moved" if near else "not_verified",
                          "dest": [dx, dy],
                          "found_at": None if check is None else
                          [round(float(v), 4) for v in check.pose.position],
                          "move_failures": FAILED_MOVE}
