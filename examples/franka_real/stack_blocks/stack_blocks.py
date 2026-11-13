import math

TABLE_Z = 0.017
GRASP_EE_Z = TABLE_Z + 0.010
HOVER = 0.08
CARRY_Z = 0.22
SURVEY = {"position": [0.44, -0.05, 0.31], "frame": "base", "rpy_deg": [180.0, 0.0, 0.0]}


FAILED_MOVE = {"stage": None}


def must(move_result, stage):
    """A failed motion must stop the plan, not be shrugged past: the first live
    run closed the gripper on air at survey height because three refused moves
    went unchecked."""
    if not move_result.ok and FAILED_MOVE["stage"] is None:
        FAILED_MOVE["stage"] = stage
        FAILED_MOVE["error"] = str(move_result.error.message)[:160]
    return move_result.ok


def down_quat(yaw_rad):
    """wxyz, gripper pointing down, fingers' closing line turned by yaw."""
    return [0.0, math.cos(yaw_rad / 2.0), math.sin(yaw_rad / 2.0), 0.0]


def box_yaw(geometry):
    """World yaw of the box's long axis, from its oriented pose."""
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
blue = locate("blue block")
green = locate("green block")
if blue is None or green is None:
    result = {"ok": False, "stage": "find", "blue": blue is not None,
              "green": green is not None}
else:
    blue = locate("blue block", above=blue.pose.position) or blue
    bx, by = float(blue.pose.position[0]), float(blue.pose.position[1])
    yaw = box_yaw(blue)
    grasp_quat = down_quat(yaw)
    open_gripper()
    reached = must(move_to_pose({"position": [bx, by, GRASP_EE_Z + HOVER], "frame": "base",
                                 "quaternion_wxyz": grasp_quat}), "hover") \
        and must(move_to_pose({"position": [bx, by, GRASP_EE_Z], "frame": "base",
                               "quaternion_wxyz": grasp_quat}), "descend")
    if not reached:
        go_home()
        result = {"ok": False, "stage": FAILED_MOVE["stage"],
                  "error": FAILED_MOVE.get("error")}
    else:
        set_gripper(0.25)
        held = get_robot_state().gripper_positions["primary"]
        must(move_to_pose({"position": [bx, by, CARRY_Z], "frame": "base",
                           "quaternion_wxyz": grasp_quat}), "lift")
        if held < 0.08:
            open_gripper()
            go_home()
            result = {"ok": False, "stage": "grasp", "note": "closed on air", "held": held}
        else:
            gx, gy = float(green.pose.position[0]), float(green.pose.position[1])
            # Depth noise inflates the PCA box vertically; a lying block's top is
            # documented at about 0.044, so an estimate outside [0.035, 0.055] is
            # noise, not news.
            green_top = float(green.pose.position[2]) + float(green.extents[2]) / 2.0
            green_top = min(max(green_top, 0.035), 0.055)
            place_quat = down_quat(box_yaw(green))
            place_z = green_top + (GRASP_EE_Z - TABLE_Z) + 0.004
            move_to_pose({"position": [gx, gy, CARRY_Z], "frame": "base",
                          "quaternion_wxyz": place_quat})
            move_to_pose({"position": [gx, gy, place_z], "frame": "base",
                          "quaternion_wxyz": place_quat})
            set_gripper(min(1.0, held + 0.2))
            move_to_pose({"position": [gx, gy, CARRY_Z], "frame": "base",
                          "quaternion_wxyz": place_quat})
            open_gripper()
            check = locate("blue block", above=[gx, gy])
            risen = check is not None and float(check.pose.position[2]) > green_top - 0.005
            near = check is not None and math.hypot(
                float(check.pose.position[0]) - gx, float(check.pose.position[1]) - gy) < 0.015
            go_home()
            result = {"ok": bool(risen and near), "stage": "stacked" if risen and near
                      else "not_stacked", "risen": risen, "near": near}
