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
FINGER_HALF_OPEN_M = 0.040   # the fingers descend open: 80 mm between pads
FINGER_PAD_HALF_M = 0.008
NEIGHBOUR_CLEAR_M = 0.010

COLOUR = "blue"
FIRST_LOOKS = [[0.50, 0.00], [0.50, -0.12]]  # blue rests near the first; probing shoves it toward the second
FIRST_NEAR_M = 0.15
B_CHOICES = [[0.50, -0.15], [0.50, 0.00]]  # the far one from A is B, so a leg is a real carry
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


def find(near, radius=NEAR_M):
    if not must(survey(near[0], near[1]), "survey"):
        return None
    found = localize_object(COLOUR + " block")
    if not found.ok:
        return None
    px, py, pz = (float(v) for v in found.geometry.pose.position)
    if not (0.25 < px < 0.66 and -0.33 < py < 0.33):
        return None
    if math.hypot(px - near[0], py - near[1]) > radius:
        return None
    extents = [float(v) for v in found.geometry.extents]
    return {"x": px, "y": py, "top": pz + extents[2] / 2.0,
            "footprint": max(extents[0], extents[1]),
            "geometry": found.geometry}


def neighbours(block):
    """Every other block instance in the survey view, as oriented rectangles:
    (x, y, yaw of the long axis, half length, half width)."""
    found = segment_text("wrist", "block")
    if not found.ok:
        return []
    out = []
    for seg in found.segmentations[:12]:
        geometry = estimate_geometry(mask_to_point_cloud(seg, "wrist", "base"))
        px, py = float(geometry.pose.position[0]), float(geometry.pose.position[1])
        extents = [float(v) for v in geometry.extents]
        if max(extents) > 0.10 or math.hypot(px - block["x"], py - block["y"]) < 0.02:
            continue
        # Silhouette bleed inflates a nadir footprint (a 55 x 28 block read
        # 62-67 mm long and wider than it is); every block on this bench is
        # the same size, so the footprint is capped at what the block can be.
        long_half = min(max(extents[0], extents[1]) / 2.0, BLOCK_LONG_M / 2.0)
        short_half = min(min(extents[0], extents[1]) / 2.0, BLOCK_FLAT_M / 2.0)
        out.append((px, py, box_yaw(geometry), long_half, short_half))
    return out


def finger_clear(block, yaw, others):
    # With down_quat(yaw) the fingers travel along `yaw`; open, they land
    # FINGER_HALF_OPEN_M either side of the block along that line. A pad
    # landing inside a neighbour's oriented footprint, grown by the pad and a
    # clearance, is a collision (run 0010: the descent to grasp fired the
    # reflex). A circle of the longest extent was too cautious (run 0013: a
    # neighbour 8 cm away blocked the good grasp and the fallback closed on air).
    margin = FINGER_PAD_HALF_M + NEIGHBOUR_CLEAR_M
    for sign in (1.0, -1.0):
        fx = block["x"] + sign * FINGER_HALF_OPEN_M * math.cos(yaw)
        fy = block["y"] + sign * FINGER_HALF_OPEN_M * math.sin(yaw)
        for ox, oy, oyaw, long_half, short_half in others:
            dx, dy = fx - ox, fy - oy
            along = dx * math.cos(oyaw) + dy * math.sin(oyaw)
            across = -dx * math.sin(oyaw) + dy * math.cos(oyaw)
            if abs(along) < long_half + margin and abs(across) < short_half + margin:
                return [round(ox, 3), round(oy, 3)]
    return None


def choose_yaw(block, others):
    """The grasp across the short axis, or the neighbour that blocks it.

    Closing along the long axis is not a fallback: with 80 mm between the pads
    and a 55 mm block the margin is 12 mm a side, inside the localization
    error, and both live attempts (runs 0013, 0014) closed on air.
    """
    yaw = box_yaw(block["geometry"]) + math.pi / 2.0
    return yaw, finger_clear(block, yaw, others)


def contact_ee_z(block):
    vertical = BLOCK_FLAT_M if block["footprint"] >= 0.045 else BLOCK_LONG_M
    return TABLE_Z + (vertical - GRASP_OFFSET)


def pose(x, y, z, quat):
    return {"position": [x, y, z], "frame": "base", "quaternion_wxyz": quat}


def pick(block, yaw):
    quat = down_quat(yaw)
    grasp_z = max(block["top"] - GRASP_OFFSET, TABLE_Z + 0.008)
    open_gripper()
    if not must(move_to_pose(pose(block["x"], block["y"], SAFE_Z, quat),
                             max_steps=LONG_MOVE_STEPS), "hover"):
        return None
    # The last centimetres by contact descent with the floor at the grasp
    # height: the open-loop generator once stopped 8 mm high inside its joint
    # tolerance and still reported converged, and the pads then closed above
    # the block (run 0015). The hand reaches a floor to a millimetre; a
    # contact stop above it means a pad is standing on something.
    if not must(move_to_pose(pose(block["x"], block["y"], grasp_z + PRE_PLACE_ABOVE_M, quat),
                             max_steps=LONG_MOVE_STEPS), "approach"):
        return None
    reach = franka_real.descend_to_contact(floor_z=grasp_z)
    if not must(reach, "descend"):
        return None
    if reach.diagnostics["stop"] != "floor":
        open_gripper()
        must(move_to_pose(pose(block["x"], block["y"], SAFE_Z, quat),
                          max_steps=LONG_MOVE_STEPS), "lift")
        return {"held": 0.0, "quat": quat, "ok": False,
                "approach": reach.diagnostics["stop"],
                "approach_z": reach.diagnostics["contact_z"]}
    set_gripper(0.25)
    held = float(get_robot_state().gripper_positions["primary"])
    if held < HELD_MIN:
        open_gripper()
        must(move_to_pose(pose(block["x"], block["y"], SAFE_Z, quat),
                          max_steps=LONG_MOVE_STEPS), "lift")
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
    # Where the hand is at release is where the block is, to a few mm: the
    # truth the eye is checked against after the re-localize.
    hand = get_robot_state().end_effector_poses["primary"].position
    truth = [float(hand[0]), float(hand[1])]
    set_gripper(min(1.0, grasp["held"] + 0.2))
    must(move_to_pose(pose(slot[0], slot[1], SAFE_Z, quat), max_steps=LONG_MOVE_STEPS),
         "retreat")
    open_gripper()
    return {"expected_z": round(expect_z, 4),
            "contact": bool(touch.ok and report.get("contact")),
            "contact_z": report.get("contact_z"),
            "stop": report.get("stop"),
            "force_rise_n": report.get("force_rise_n"),
            "lag_m": report.get("lag_m"),
            "drift_xy_m": report.get("drift_xy_m"),
            "released_at": [round(truth[0], 4), round(truth[1], 4)]}


legs = []
origin = FIRST_LOOKS[0]
A = None  # fixed by the first look: the block is carried back to where it was found
for trip in range(ROUND_TRIPS):
    for outbound in (True, False):
        leg = {"trip": trip, "from": origin, "status": "started"}
        legs.append(leg)
        if A is None:
            block = None
            for look in FIRST_LOOKS:
                block = find(look, FIRST_NEAR_M)
                if block is not None:
                    break
        else:
            block = find(origin, NEAR_M)
        if block is None:
            leg["status"] = "not_found"
            break
        if A is None:
            A = [round(block["x"], 4), round(block["y"], 4)]
            B = max(B_CHOICES, key=lambda spot: math.hypot(spot[0] - A[0], spot[1] - A[1]))
            leg["a"] = A
            leg["b"] = B
        target = B if outbound else A
        leg["to"] = target
        leg["picked_from"] = [round(block["x"], 4), round(block["y"], 4)]
        leg["footprint"] = round(block["footprint"], 4)
        others = neighbours(block)
        yaw, blocked_by = choose_yaw(block, others)
        leg["neighbours"] = [[round(o[0], 3), round(o[1], 3)] for o in others]
        if blocked_by is not None:
            leg["status"] = "blocked"
            leg["blocked_by"] = blocked_by
            break
        grasp = pick(block, yaw)
        if grasp is None:
            leg["status"] = "move_failed"
            break
        leg["held"] = round(grasp["held"], 3)
        if not grasp["ok"]:
            leg["status"] = "approach_blocked" if "approach" in grasp else "air_close"
            leg["approach"] = grasp.get("approach")
            leg["approach_z"] = grasp.get("approach_z")
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
        # Teach the eye from the hand: seen against released-at. The camera
        # bracket moves; this keeps its correction current, every leg.
        lesson = franka_real.report_hand_eye_residual(
            [after["x"], after["y"]], placed["released_at"])
        leg["eye_residual_xy"] = lesson["residual_xy"]
        leg["eye_drift"] = bool(lesson["drift"])
        leg["eye_offset_xyz"] = lesson["offset_xyz"]
        leg["status"] = "verified" if error < VERIFY_M else "off_target"
        origin = [after["x"], after["y"]]
        if leg["status"] != "verified":
            # An off-target block may be against a neighbour; the next pick
            # would be a gamble (run 0010: a 33 mm miss, then the descent to
            # grasp fired the reflex). Stop and say so.
            break
    if legs[-1]["status"] != "verified":
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
