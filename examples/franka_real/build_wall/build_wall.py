import math

TABLE_Z = 0.017
GRASP_OFFSET = 0.017
BLOCK_LONG_M = 0.055
BLOCK_FLAT_M = 0.028
RELEASE_GAP_M = 0.003
CARRY_Z = 0.22
LONG_MOVE_STEPS = 600
GRASP_TRIES_PER_SLOT = 2

goal = franka_real.get_task_metadata()["goal"]
BOX = [float(v) for v in goal["box_centre"]]
BOX_HALF = float(goal["box_half_extent"])
SAFE_Z = max(float(goal["box_rim_z"]) + 0.06, 0.14)
ALONG_X = goal.get("wall_axis", "x") == "x"
WALL_START = float(goal["wall_start"])
WALL_FIXED = float(goal["wall_fixed"])
PITCH = float(goal["slot_pitch"])
SLOTS = int(goal["slots"])
# Long axis along the wall: yaw 0 for an x wall, near 90 deg for a y wall.
WALL_YAW = 0.0 if ALONG_X else 1.55

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


def is_flat(block):
    """A footprint showing the block's long side means it lies flat."""
    return block["footprint"] >= 0.045


def place_ee_z(block):
    """Preserve the held block's bottom clearance instead of assuming flat.

    Live lesson from the first clutter pick: a block grasped standing on end
    hangs 55 mm below the top, not 28, and lowering it to flat-block height
    drove its bottom into the table and fired the cartesian reflex."""
    vertical = BLOCK_FLAT_M if is_flat(block) else BLOCK_LONG_M
    return TABLE_Z + (vertical - GRASP_OFFSET) + RELEASE_GAP_M


def survey(x, y):
    return move_to_pose({"position": [x, y, 0.33], "frame": "base",
                         "rpy_deg": [180.0, 0.0, 0.0]})


def scan():
    found = segment_text("wrist", "wooden block")
    if not found.ok:
        return []
    out = []
    for seg in found.segmentations[:12]:
        cloud = mask_to_point_cloud(seg, "wrist", "base")
        geometry = estimate_geometry(cloud)
        px, py, pz = (float(v) for v in geometry.pose.position)
        extents = [float(v) for v in geometry.extents]
        if max(extents) > 0.10:
            continue
        out.append({"x": px, "y": py, "top": pz + extents[2] / 2.0,
                    "footprint": max(extents[0], extents[1]),
                    "geometry": geometry})
    return out


def in_box(block):
    return (abs(block["x"] - BOX[0]) <= BOX_HALF + 0.02
            and abs(block["y"] - BOX[1]) <= BOX_HALF + 0.02)


def choose(blocks):
    """The wall wants blocks lying flat; among those, highest clear top wins.

    A standing block still gets picked when nothing lies flat -- it is placed
    at its own height and reported, since a standing block in a slot is a
    verifiable block, just not a pretty one."""
    lying = [b for b in blocks if is_flat(b)]
    pool = lying if lying else blocks
    ordered = sorted(pool, key=lambda b: -b["top"])
    for block in ordered:
        clear = all(math.hypot(block["x"] - o["x"], block["y"] - o["y"]) >= 0.035
                    for o in blocks if o is not block)
        if clear:
            return block
    return ordered[0]


def slot_centre(index):
    along = WALL_START + index * PITCH
    return (along, WALL_FIXED) if ALONG_X else (WALL_FIXED, along)


def pick_one():
    """One block out of the box, held above the rim: (status, grip, block)."""
    for attempt in range(GRASP_TRIES_PER_SLOT):
        survey(BOX[0], BOX[1])
        blocks = [b for b in scan() if in_box(b)]
        if not blocks:
            return "empty", 0.0, None
        pick = choose(blocks)
        quat = down_quat(box_yaw(pick["geometry"]))
        grasp_z = max(pick["top"] - GRASP_OFFSET, TABLE_Z + 0.008)
        open_gripper()
        reached = must(move_to_pose({"position": [pick["x"], pick["y"], SAFE_Z],
                                     "frame": "base", "quaternion_wxyz": quat},
                                    max_steps=LONG_MOVE_STEPS), "hover") \
            and must(move_to_pose({"position": [pick["x"], pick["y"], grasp_z],
                                   "frame": "base", "quaternion_wxyz": quat},
                                  max_steps=LONG_MOVE_STEPS), "descend")
        if not reached:
            return "aborted", 0.0, None
        set_gripper(0.25)
        grip = get_robot_state().gripper_positions["primary"]
        must(move_to_pose({"position": [pick["x"], pick["y"], SAFE_Z],
                           "frame": "base", "quaternion_wxyz": quat},
                          max_steps=LONG_MOVE_STEPS), "lift")
        if grip < 0.08:
            open_gripper()
            continue  # shoved it; re-survey and retry
        return "held", grip, pick
    return "failed", 0.0, None


place_quat = down_quat(WALL_YAW)
slots_report = []
status = "built"
for index in range(SLOTS):
    sx, sy = slot_centre(index)
    picked, grip, block = pick_one()
    if picked == "empty":
        status = "box_empty"
        slots_report.append({"slot": index, "ok": False, "note": "box empty"})
        break
    if picked in ("aborted", "failed"):
        status = picked
        slots_report.append({"slot": index, "ok": False, "note": picked})
        break
    must(move_to_pose({"position": [sx, sy, CARRY_Z], "frame": "base",
                       "quaternion_wxyz": place_quat}, max_steps=LONG_MOVE_STEPS),
         "carry")
    must(move_to_pose({"position": [sx, sy, place_ee_z(block)], "frame": "base",
                       "quaternion_wxyz": place_quat}, max_steps=LONG_MOVE_STEPS),
         "lower")
    set_gripper(min(1.0, grip + 0.2))
    must(move_to_pose({"position": [sx, sy, CARRY_Z], "frame": "base",
                       "quaternion_wxyz": place_quat}, max_steps=LONG_MOVE_STEPS),
         "retreat")
    open_gripper()
    survey(sx, sy)
    nearest = None
    for block in scan():
        distance = math.hypot(block["x"] - sx, block["y"] - sy)
        if nearest is None or distance < nearest["distance"]:
            nearest = {"distance": distance, "x": block["x"], "y": block["y"]}
    verified = nearest is not None and nearest["distance"] < 0.03
    slots_report.append({
        "slot": index, "ok": bool(verified),
        "target": [round(sx, 4), round(sy, 4)],
        "orientation": "flat" if is_flat(block) else "standing",
        "found_at": None if nearest is None else
        [round(nearest["x"], 4), round(nearest["y"], 4)],
    })

go_home()
built = sum(1 for s in slots_report if s.get("ok"))
result = {"ok": built == SLOTS, "status": status, "slots_built": built,
          "slots_wanted": SLOTS, "slots": slots_report,
          "move_failures": FAILED_MOVE}
