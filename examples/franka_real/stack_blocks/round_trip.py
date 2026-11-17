import math

# A self-resetting practice stage: carry one known block from A to B by
# contact, verify, then carry it back. Each leg is one pick (survey, localize,
# hover, descend, close, lift) and one place (pre-place above the expected
# contact, descend_to_contact, release, retreat, re-localize). The block is
# blue, lying flat on the sorted row; B is the bare patch between the row and
# the standing lime block that touch_table.py validated. Legs alternate so
# the scene ends where it began and the next run needs no human reset.

TABLE_Z = 0.017
GRASP_OFFSET = 0.017
BLOCK_LONG_M = 0.055
BLOCK_FLAT_M = 0.028
PRE_PLACE_ABOVE_M = 0.03
FLOOR_BELOW_M = 0.015
CARRY_Z = 0.22
SAFE_Z = 0.14
SURVEY_Z = 0.33
LONG_MOVE_STEPS = 600
NEAR_M = 0.07
VERIFY_M = 0.03
HELD_MIN = 0.08        # gripper fraction below which the close met air

COLOUR = "blue"
A = [0.50, 0.00]
B = [0.50, -0.15]
ROUND_TRIPS = 3

FAILED_MOVE = {"stage": None, "error": None}


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


def survey(x, y):
    return move_to_pose({"position": [x, y, SURVEY_Z], "frame": "base",
                         "rpy_deg": [180.0, 0.0, 0.0]}, max_steps=LONG_MOVE_STEPS)


def find(near):
    if not must(survey(near[0], near[1]), "survey"):
        return None
    found = localize_object(COLOUR + " block")
    if not found.ok:
        return None
    px, py, pz = (float(v) for v in found.geometry.pose.position)
    if not (0.25 < px < 0.66 and -0.33 < py < 0.33):
        return None
    if math.hypot(px - near[0], py - near[1]) > NEAR_M:
        return None
    extents = [float(v) for v in found.geometry.extents]
    return {"x": px, "y": py, "top": pz + extents[2] / 2.0,
            "footprint": max(extents[0], extents[1]),
            "geometry": found.geometry}


def contact_ee_z(block):
    vertical = BLOCK_FLAT_M if block["footprint"] >= 0.045 else BLOCK_LONG_M
    return TABLE_Z + (vertical - GRASP_OFFSET)


def pose(x, y, z, quat):
    return {"position": [x, y, z], "frame": "base", "quaternion_wxyz": quat}


def pick(block):
    quat = down_quat(box_yaw(block["geometry"]) + math.pi / 2.0)
    grasp_z = max(block["top"] - GRASP_OFFSET, TABLE_Z + 0.008)
    open_gripper()
    if not must(move_to_pose(pose(block["x"], block["y"], SAFE_Z, quat),
                             max_steps=LONG_MOVE_STEPS), "hover"):
        return None
    if not must(move_to_pose(pose(block["x"], block["y"], grasp_z, quat)), "descend"):
        return None
    set_gripper(0.25)
    held = float(get_robot_state().gripper_positions["primary"])
    if held < HELD_MIN:
        open_gripper()
        must(move_to_pose(pose(block["x"], block["y"], SAFE_Z, quat)), "lift")
        return {"held": held, "quat": quat, "ok": False}
    if not must(move_to_pose(pose(block["x"], block["y"], CARRY_Z, quat),
                             max_steps=LONG_MOVE_STEPS), "lift"):
        return None
    return {"held": held, "quat": quat, "ok": True}


def place(block, grasp, slot):
    expect_z = contact_ee_z(block)
    quat = grasp["quat"]
    if not must(move_to_pose(pose(slot[0], slot[1], CARRY_Z, quat),
                             max_steps=LONG_MOVE_STEPS), "carry"):
        return None
    if not must(move_to_pose(pose(slot[0], slot[1], expect_z + PRE_PLACE_ABOVE_M, quat),
                             max_steps=LONG_MOVE_STEPS), "pre_place"):
        return None
    touch = franka_real.descend_to_contact(floor_z=max(expect_z - FLOOR_BELOW_M, 0.004))
    must(touch, "lower")
    report = touch.diagnostics if touch.ok else {}
    set_gripper(min(1.0, grasp["held"] + 0.2))
    must(move_to_pose(pose(slot[0], slot[1], SAFE_Z, quat), max_steps=LONG_MOVE_STEPS),
         "retreat")
    open_gripper()
    return {"expected_z": round(expect_z, 4),
            "contact": bool(touch.ok and report.get("contact")),
            "contact_z": report.get("contact_z"),
            "stop": report.get("stop"),
            "force_rise_n": report.get("force_rise_n"),
            "lag_m": report.get("lag_m")}


legs = []
origin = A
for trip in range(ROUND_TRIPS):
    for target in (B, A):
        leg = {"trip": trip, "from": origin, "to": target, "status": "started"}
        legs.append(leg)
        block = find(origin)
        if block is None:
            leg["status"] = "not_found"
            break
        leg["picked_from"] = [round(block["x"], 4), round(block["y"], 4)]
        leg["footprint"] = round(block["footprint"], 4)
        grasp = pick(block)
        if grasp is None:
            leg["status"] = "move_failed"
            break
        leg["held"] = round(grasp["held"], 3)
        if not grasp["ok"]:
            leg["status"] = "air_close"
            break
        placed = place(block, grasp, target)
        if placed is None:
            leg["status"] = "move_failed"
            break
        leg.update(placed)
        after = find(target)
        if after is None:
            leg["status"] = "lost"
            break
        error = math.hypot(after["x"] - target[0], after["y"] - target[1])
        leg["found_at"] = [round(after["x"], 4), round(after["y"], 4)]
        leg["error_m"] = round(error, 4)
        leg["status"] = "verified" if error < VERIFY_M else "off_target"
        origin = [after["x"], after["y"]]
    if legs[-1]["status"] not in ("verified", "off_target"):
        break
go_home()

verified = sum(1 for leg in legs if leg["status"] == "verified")
result = {
    "ok": FAILED_MOVE["stage"] is None and verified == 2 * ROUND_TRIPS,
    "verified": verified,
    "of": 2 * ROUND_TRIPS,
    "legs": legs,
    "stage": FAILED_MOVE["stage"],
    "error": FAILED_MOVE["error"],
}
