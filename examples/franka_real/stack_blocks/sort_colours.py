import math

# Sort the coloured blocks into a rainbow row: red, green, blue, left to
# right along x at y = 0 on the open table. A demo program for a young
# visitor, so the row is tidy and the report is simple. Bench state is
# tonight's: blue rides green near (0.449, -0.218), red stands near
# (0.394, -0.062), and a second lime-green block stands near (0.358, -0.203)
# and is left alone -- each pick surveys directly above its block's known
# address so localize_object's single answer is the right witness. Blue is
# picked before green (it is on top), and red before green so the green
# place never descends beside the still-standing red.

TABLE_Z = 0.017
GRASP_OFFSET = 0.017  # fingers engage this far below a block's measured top
BLOCK_LONG_M = 0.055
BLOCK_FLAT_M = 0.028
# Both live placements (flat blue, standing red) met the table before the
# nominal 3 mm gap closed and fired the cartesian reflex; the standing red
# then toppled and was knocked 16 cm toward the base during recovery. A
# thicker gap costs a short drop and buys a clean release.
RELEASE_GAP_M = 0.008
CARRY_Z = 0.22
SAFE_Z = 0.14
SURVEY_Z = 0.33
LONG_MOVE_STEPS = 600
NEAR_M = 0.07    # a localized block must sit this close to its known address
VERIFY_M = 0.03
MAX_ATTEMPTS = 2

ROW_Y = 0.0
PLAN = [
    {"colour": "blue", "near": [0.5168, 0.0328], "slot": [0.50, ROW_Y]},
    {"colour": "red", "near": [0.394, -0.062], "slot": [0.32, ROW_Y]},
    {"colour": "green", "near": [0.454, -0.2054], "slot": [0.41, ROW_Y]},
]

FAILED_MOVE = {"stage": None}


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
    # Long surveys (slot to far block) exceed the default step budget; a
    # refused survey left one live localization looking from the wrong pose.
    return move_to_pose({"position": [x, y, SURVEY_Z], "frame": "base",
                         "rpy_deg": [180.0, 0.0, 0.0]}, max_steps=LONG_MOVE_STEPS)


def find(colour, near):
    if not must(survey(near[0], near[1]), "survey"):
        return None
    found = localize_object(colour + " block")
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


def place_ee_z(block):
    # Standing blocks hang 55 mm below the grip, not 28: lowering one to the
    # flat-block height fired the cartesian reflex live.
    vertical = BLOCK_FLAT_M if block["footprint"] >= 0.045 else BLOCK_LONG_M
    return TABLE_Z + (vertical - GRASP_OFFSET) + RELEASE_GAP_M


row = []
aborted = False
for step in PLAN:
    if aborted:
        break
    placed = None
    attempt = 0
    while attempt < MAX_ATTEMPTS and placed is None and not aborted:
        attempt += 1
        block = find(step["colour"], step["near"])
        if block is None:
            row.append({"colour": step["colour"], "status": "not_found"})
            break
        quat = down_quat(box_yaw(block["geometry"]))
        grasp_z = max(block["top"] - GRASP_OFFSET, TABLE_Z + 0.008)
        open_gripper()
        reached = must(move_to_pose({"position": [block["x"], block["y"], SAFE_Z],
                                     "frame": "base", "quaternion_wxyz": quat},
                                    max_steps=LONG_MOVE_STEPS), "hover") \
            and must(move_to_pose({"position": [block["x"], block["y"], grasp_z],
                                   "frame": "base", "quaternion_wxyz": quat},
                                  max_steps=LONG_MOVE_STEPS), "descend")
        if not reached:
            aborted = True
            break
        set_gripper(0.25)
        held = get_robot_state().gripper_positions["primary"]
        lifted = must(move_to_pose({"position": [block["x"], block["y"], CARRY_Z],
                                    "frame": "base", "quaternion_wxyz": quat},
                                   max_steps=LONG_MOVE_STEPS), "lift")
        if held < 0.08:
            open_gripper()
            continue  # closed on air; re-survey and retry once
        slot = step["slot"]
        ok = lifted \
            and must(move_to_pose({"position": [slot[0], slot[1], CARRY_Z],
                                   "frame": "base", "quaternion_wxyz": quat},
                                  max_steps=LONG_MOVE_STEPS), "carry")
        if not ok:
            aborted = True
            break
        # Lower is contact-tolerant: a cartesian reflex here means the block
        # met the table a few millimetres early (depth bias), which is a
        # completed place, not a failure -- the first live clutter pick proved
        # the release lands the block on target. Record it, release anyway.
        must(move_to_pose({"position": [slot[0], slot[1], place_ee_z(block)],
                           "frame": "base", "quaternion_wxyz": quat},
                          max_steps=LONG_MOVE_STEPS), "lower")
        set_gripper(min(1.0, held + 0.2))
        ok = must(move_to_pose({"position": [slot[0], slot[1], CARRY_Z],
                                "frame": "base", "quaternion_wxyz": quat},
                               max_steps=LONG_MOVE_STEPS), "retreat")
        open_gripper()
        check = find(step["colour"], slot)
        distance = None
        if check is not None:
            distance = math.hypot(check["x"] - slot[0], check["y"] - slot[1])
        placed = {"colour": step["colour"], "slot": slot,
                  "status": "placed" if distance is not None and distance < VERIFY_M
                  else "not_verified",
                  "picked_from": [round(block["x"], 4), round(block["y"], 4)],
                  "found_at": None if check is None else
                  [round(check["x"], 4), round(check["y"], 4)],
                  "orientation": "flat" if block["footprint"] >= 0.045
                  else "standing",
                  "attempts": attempt}
        row.append(placed)
    if placed is None and not aborted and (not row or row[-1]["colour"] != step["colour"]):
        row.append({"colour": step["colour"], "status": "grasp_air",
                    "attempts": attempt})

go_home()
if aborted:
    result = {"ok": False, "stage": FAILED_MOVE["stage"],
              "error": FAILED_MOVE.get("error"), "row": row}
else:
    sorted_count = sum(1 for r in row if r.get("status") == "placed")
    result = {"ok": sorted_count == len(PLAN), "sorted": sorted_count,
              "of": len(PLAN), "row": row, "move_failures": FAILED_MOVE}
