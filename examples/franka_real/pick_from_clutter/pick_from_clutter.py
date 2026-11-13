import math

TABLE_Z = 0.017
GRASP_OFFSET = 0.017  # fingers engage this far below a block's measured top
BLOCK_LONG_M = 0.055
BLOCK_FLAT_M = 0.028
RELEASE_GAP_M = 0.003
CARRY_Z = 0.22
LONG_MOVE_STEPS = 600
MAX_ATTEMPTS = 3

goal = franka_real.get_task_metadata()["goal"]
BOX = [float(v) for v in goal["box_centre"]]
BOX_HALF = float(goal["box_half_extent"])
BOX_X = [float(v) for v in goal["box_x"]]  # measured wall lines, not a square
BOX_Y = [float(v) for v in goal["box_y"]]
TARGET = [float(v) for v in goal["pick_target"]]
SAFE_Z = max(float(goal["box_rim_z"]) + 0.06, 0.14)  # clears the box rim
# Hand envelope half-spans at grasp depth: along the finger-closing axis the
# open fingers span wide; along the block's long axis the hand body is slim.
HAND_CLOSE_HALF = 0.060
HAND_ALONG_HALF = 0.040

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


def place_ee_z(block):
    """Place height that preserves the held block's bottom clearance.

    The first live pick grabbed a block standing on end and lowered it to the
    flat-block height; its bottom hit the table 25 mm early and the cartesian
    reflex aborted the move. The scan's footprint says which way the block
    hangs: a footprint showing the long side lies flat (28 mm tall in hand),
    anything narrower stands on end (55 mm)."""
    vertical = BLOCK_FLAT_M if block["footprint"] >= 0.045 else BLOCK_LONG_M
    return TABLE_Z + (vertical - GRASP_OFFSET) + RELEASE_GAP_M


def survey(x, y):
    return move_to_pose({"position": [x, y, 0.33], "frame": "base",
                         "rpy_deg": [180.0, 0.0, 0.0]})


def scan():
    """Every wooden-block instance in view, with base-frame geometry."""
    found = segment_text("wrist", "wooden block")
    if not found.ok:
        return []
    out = []
    for seg in found.segmentations[:12]:
        cloud = mask_to_point_cloud(seg, "wrist", "base")
        geometry = estimate_geometry(cloud)
        px, py, pz = (float(v) for v in geometry.pose.position)
        extents = [float(v) for v in geometry.extents]
        if max(extents) > 0.10:  # merged neighbours or residual bleed, not one block
            continue
        out.append({"x": px, "y": py, "top": pz + extents[2] / 2.0,
                    "footprint": max(extents[0], extents[1]),
                    "geometry": geometry})
    return out


def in_box(block):
    return (abs(block["x"] - BOX[0]) <= BOX_HALF + 0.02
            and abs(block["y"] - BOX[1]) <= BOX_HALF + 0.02)


def wall_clear(block):
    """Whether the descending hand clears the basket walls at this grasp.

    Live lesson: a grasp 9 mm from the east wall drove the hand into the rim
    and fired the cartesian reflex. The fingers close across the block's long
    axis, so the wide span points along closing and the slim hand body along
    the block; a block the envelope cannot clear is unpickable by vertical
    descend and is excluded, not attempted."""
    yaw = box_yaw(block["geometry"])
    need_x = (abs(math.sin(yaw)) * HAND_CLOSE_HALF
              + abs(math.cos(yaw)) * HAND_ALONG_HALF)
    need_y = (abs(math.cos(yaw)) * HAND_CLOSE_HALF
              + abs(math.sin(yaw)) * HAND_ALONG_HALF)
    return (block["x"] - BOX_X[0] >= need_x and BOX_X[1] - block["x"] >= need_x
            and block["y"] - BOX_Y[0] >= need_y and BOX_Y[1] - block["y"] >= need_y)


def choose(blocks):
    """Highest top face wins; among all, prefer one with finger clearance."""
    ordered = sorted(blocks, key=lambda b: -b["top"])
    for block in ordered:
        clear = all(math.hypot(block["x"] - o["x"], block["y"] - o["y"]) >= 0.035
                    for o in blocks if o is not block)
        if clear:
            return block
    return ordered[0]


result = {"ok": False, "stage": "find"}
held_block = None
held = 0.0
aborted = False
survey(BOX[0], BOX[1])
attempt = 0
while attempt < MAX_ATTEMPTS and held_block is None and not aborted:
    attempt += 1
    blocks = [b for b in scan() if in_box(b)]
    if not blocks:
        result = {"ok": False, "stage": "find", "attempt": attempt,
                  "note": "no block candidates in the box"}
        break
    pickable = [b for b in blocks if wall_clear(b)]
    if not pickable:
        result = {"ok": False, "stage": "find", "attempt": attempt,
                  "note": "all candidates hug the basket wall"}
        break
    pick = choose(pickable)
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
        aborted = True
        break
    set_gripper(0.25)
    held = get_robot_state().gripper_positions["primary"]
    # Straight back above the rim before anything lateral: box walls are
    # obstacles the workspace guard does not know about.
    must(move_to_pose({"position": [pick["x"], pick["y"], SAFE_Z], "frame": "base",
                       "quaternion_wxyz": quat}, max_steps=LONG_MOVE_STEPS), "lift")
    if held < 0.08:
        open_gripper()
        result = {"ok": False, "stage": "grasp", "attempt": attempt,
                  "note": "closed on air; block shoved"}
        survey(BOX[0], BOX[1])
        continue
    held_block = pick

if aborted:
    go_home()
    result = {"ok": False, "stage": FAILED_MOVE["stage"],
              "error": FAILED_MOVE.get("error")}
elif held_block is not None:
    quat = down_quat(box_yaw(held_block["geometry"]))
    must(move_to_pose({"position": [held_block["x"], held_block["y"], CARRY_Z],
                       "frame": "base", "quaternion_wxyz": quat},
                      max_steps=LONG_MOVE_STEPS), "rise")
    must(move_to_pose({"position": [TARGET[0], TARGET[1], CARRY_Z], "frame": "base",
                       "quaternion_wxyz": quat}, max_steps=LONG_MOVE_STEPS), "carry")
    must(move_to_pose({"position": [TARGET[0], TARGET[1], place_ee_z(held_block)],
                       "frame": "base", "quaternion_wxyz": quat},
                      max_steps=LONG_MOVE_STEPS), "lower")
    set_gripper(min(1.0, held + 0.2))
    must(move_to_pose({"position": [TARGET[0], TARGET[1], CARRY_Z], "frame": "base",
                       "quaternion_wxyz": quat}, max_steps=LONG_MOVE_STEPS), "retreat")
    open_gripper()
    survey(TARGET[0], TARGET[1])
    nearest = None
    for block in scan():
        distance = math.hypot(block["x"] - TARGET[0], block["y"] - TARGET[1])
        if nearest is None or distance < nearest["distance"]:
            nearest = {"distance": distance, "x": block["x"], "y": block["y"],
                       "top": block["top"]}
    go_home()
    placed = nearest is not None and nearest["distance"] < 0.03
    result = {"ok": bool(placed), "stage": "placed" if placed else "not_verified",
              "target": TARGET, "attempts": attempt,
              "orientation": "flat" if held_block["footprint"] >= 0.045 else "standing",
              "picked_from": [round(held_block["x"], 4), round(held_block["y"], 4),
                              round(held_block["top"], 4)],
              "found_at": None if nearest is None else
              [round(nearest["x"], 4), round(nearest["y"], 4)],
              "move_failures": FAILED_MOVE}
else:
    go_home()
