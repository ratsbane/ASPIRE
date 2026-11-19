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
NEAR_FLOOR_M = 0.008   # a contact this close above the grasp floor is the table under the pads, not the block
CARRY_Z = 0.22
SAFE_Z = 0.14
SURVEY_Z = 0.33
LONG_MOVE_STEPS = 900   # a lift from the spare spot once planned past 600 (run 0134)
NEAR_M = 0.07
NADIR_M = 0.04   # farther than this from the survey spot, look again from above the block
VERIFY_M = 0.03
VERIFY_LOOK_M = 0.20  # how far from the target the verify will accept the block as read
DROPPED_M = 0.05      # seen farther than this from the release point: it left the jaws, not an eye error
LYING_TOP_MAX = TABLE_Z + BLOCK_FLAT_M + 0.004  # a lying block tops out here; a higher reading is bleed
RELOOK_TRUST_M = 0.015  # a hover re-look that moves the block farther than this from the survey's reading,
                        # and then a pad still lands on it, is distrusted (run 0315: 3 cm, twelve times)
HELD_MIN = 0.08        # gripper fraction below which the close met air
HELD_SQUARE = 0.30     # narrower than this on a 28 mm block (~0.36): the pads are on a corner
LENGTHWISE_HELD = 0.55  # a 28 mm block holds at ~0.36; across its 56 mm length at ~0.74
FINGER_HALF_OPEN_M = 0.040   # the fingers descend open: 80 mm between pads
MAX_APPROACHES = 12          # grasp attempts per pick, re-looks included
CAMERA_OFFSET = (-0.0595, -0.0087)  # the wrist camera's origin from the TCP, in the hand's x, y
CAMERA_HALF_M = 0.021        # the D405 housing is 42 mm square
CAMERA_CLEAR_M = 0.0         # the camera's footprint must not overlap another block's
FINGER_PAD_HALF_M = 0.008
NEIGHBOUR_CLEAR_M = 0.004  # a neighbour 9 cm off-nadir reads ~2.5 cm toward the camera; the
                            # contact approach catches a pad on it and retries shifted (run 0101)

# The colour and the spots vary by run seed, for a dataset that is not one
# block on one pair of spots (Doug, 2026-11-17 evening).
SEED = int(get_task_context().metadata.get("seed", 0))
COLOURS = ["blue", "green", "red"]
COLOUR = COLOURS[SEED % 3]   # the first choice; a colour the looks cannot find yields to the next
JITTER_M = 0.02
FIRST_LOOKS = [[0.50, 0.00], [0.50, -0.12], [0.50, 0.12], [0.55, -0.22],
               [0.45, -0.26]]  # the row, the far spot, the near side, the spares, red's corner
BASKET_Y = 0.12  # the basket stands beyond this on the positive side (its blue block read at y 0.17 after a bump); nothing seen there is a task block
FIRST_NEAR_M = 0.15
A = [0.53 + JITTER_M * ((SEED // 3) % 3 - 1), 0.00 + JITTER_M * ((SEED // 9) % 3 - 1)]
B = [0.53 + JITTER_M * ((SEED // 27) % 3 - 1), -0.15 + JITTER_M * ((SEED // 3) % 3 - 1)]
C = [0.57, -0.07]     # the spare spots, when A or B has a block in it: the front of the table
D = [0.57, 0.06]      # (the old (0.42, -0.12) had no IK for the retreat or the survey, run 0155;
E = [0.48, -0.22]     # with three colours in play two spares were not enough, run 0159;
F = [0.47, -0.10]     # with four blocks (two greens) C and D full ruled out A and B by
G = [0.47, 0.08]      # clearance and E held the block itself, run 0321: the table's middle
                      # at x 0.60 the grasp search's shifts reached x 0.627 with the fingers open
                      # and twice pushed the red block off the front edge, runs 0221 and 0224)
PICK_X_MAX = 0.60     # no grasp attempt is centred beyond this: the front edge is near
TARGET_CLEAR_M = 0.07  # another block this close to a spot makes it unusable
REHOME_M = 0.08   # found farther than this from A: the first carry takes it home
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


def has_depth(seg, depth):
    """Whether a mask covers enough valid depth to back-project.

    mask_to_point_cloud raises on a mask with no valid depth, and a sandboxed
    program cannot catch that (runs 0040, 0169); a confident mask at the
    image's edge or over a depth hole is such a mask.
    """
    covered = depth[seg.mask]
    return int((covered > 0.1).sum()) >= 50


STANDING_FOOTPRINT_MAX = 0.052  # a block on end is 28 mm square; bleed widens it, never to a block's length


def is_standing(top, footprint):
    """On end (top ~0.072, ~28 mm square) or lying (top ~0.045, its length showing).

    A high top alone is not enough: with the leveling finding no table plane
    a lying green read 0.062 high with a 62 mm footprint (run 0349) and the
    pads closed on air at a standing block's mid-height, twelve times.
    """
    if top > 0.06:
        return footprint < STANDING_FOOTPRINT_MAX
    if top < 0.05:
        return False
    return footprint < 0.045


def see_lying(near, radius):
    """The best-scoring lying block of the colour within reach of `near`.

    localize_object hands back the one best mask, which for "green block" is
    the standing lime block as often as the row's green (run 0166 chased
    it to the table's edge and the workspace guard said no). Standing blocks
    are never carried: too tall for the camera, and not the task.
    """
    found = segment_text("wrist", COLOUR + " block")
    if not found.ok:
        return None
    depth = get_observation().cameras["wrist"].depth_m
    best = None
    for seg in found.segmentations[:8]:
        if float(seg.score) < 0.5 or not has_depth(seg, depth):
            continue
        geometry = estimate_geometry(mask_to_point_cloud(seg, "wrist", "base"))
        px, py = float(geometry.pose.position[0]), float(geometry.pose.position[1])
        extents = [float(v) for v in geometry.extents]
        top = float(geometry.pose.position[2]) + extents[2] / 2.0
        if max(extents) > 0.12:
            continue  # merged blobs
        # A block on end tops at ~0.072 and shows a ~28 mm square footprint;
        # lying it tops at ~0.045 (bleed can read 0.05) and shows its length.
        # A shrunken mask of a lying block reads a small footprint too (run
        # 0291: the fingers then closed 1 cm above it, eleven times), so the
        # footprint only decides when the top is between the two heights.
        standing = is_standing(top, max(extents[0], extents[1]))
        if not (0.30 < px < 0.62 and -0.31 < py < BASKET_Y):
            continue  # beyond where the hand can work above it (the box is 0.33; a lift at 0.334 was refused),
            # or in the basket (run 0219 reached for a blue block seen standing in it)
        if standing and abs(py) > 0.27:
            continue  # a standing block at the table's edge is the lime's corner, not the task (run 0166)
        if math.hypot(px - near[0], py - near[1]) > radius:
            continue
        # Lying first; a standing one of the colour is still carried (blue
        # once slipped from the jaws and landed on end mid-table, run 0189,
        # and the lying-only rule then called it lost). The fingers close
        # across its 28 mm at mid-height; the camera clears its top by 3 cm.
        # Then the one nearest `near`, not the best-scoring: with two green
        # blocks on the table the verify look at the placed one chose the
        # other, 15 cm away, and called the placement a drop (run 0313).
        rank = (0 if standing else 1, -math.hypot(px - near[0], py - near[1]))
        if best is None or rank > best[1]:
            best = (geometry, rank)
    return None if best is None else best[0]


def find(near, radius=NEAR_M):
    if not must(survey(near[0], near[1]), "survey"):
        return None
    geometry = see_lying(near, radius)
    if geometry is None:
        return None
    px, py, pz = (float(v) for v in geometry.pose.position)
    if math.hypot(px - near[0], py - near[1]) > NADIR_M:
        # A block well off the survey's nadir reads worse (silhouette bleed,
        # and a loose camera's tilt costs more at the edge of the image):
        # look again from straight above it before trusting the position
        # (run 0061: an air close on a block read 9 cm off-nadir at the
        # table's edge).
        if not must(survey(px, py), "survey"):
            return None
        again = see_lying([px, py], 0.06)
        if again is not None:
            geometry = again
            px, py, pz = (float(v) for v in geometry.pose.position)
    extents = [float(v) for v in geometry.extents]
    return {"x": px, "y": py, "top": pz + extents[2] / 2.0,
            "footprint": max(extents[0], extents[1]),
            "standing": is_standing(pz + extents[2] / 2.0, max(extents[0], extents[1])),
            "geometry": geometry}


def neighbours(block):
    """Every other block instance in the survey view, as oriented rectangles:
    (x, y, yaw of the long axis, half length, half width)."""
    found = segment_text("wrist", "block")
    if not found.ok:
        return []
    depth = get_observation().cameras["wrist"].depth_m
    out = []
    for seg in found.segmentations[:12]:
        if float(seg.score) < 0.3 or not has_depth(seg, depth):
            continue  # a mask with no valid depth cannot be back-projected
        geometry = estimate_geometry(mask_to_point_cloud(seg, "wrist", "base"))
        px, py = float(geometry.pose.position[0]), float(geometry.pose.position[1])
        extents = [float(v) for v in geometry.extents]
        if max(extents) > 0.10:
            continue
        # Silhouette bleed inflates a nadir footprint (a 55 x 28 block read
        # 62-67 mm long and wider than it is); every block on this bench is
        # the same size, so the footprint is capped at what the block can be.
        long_half = min(max(extents[0], extents[1]) / 2.0, BLOCK_LONG_M / 2.0)
        short_half = min(min(extents[0], extents[1]) / 2.0, BLOCK_FLAT_M / 2.0)
        oyaw = box_yaw(geometry)
        # The block itself, seen again: an instance whose footprint contains
        # the block's centre. (A plain 2 cm radius missed a re-detection 2.2 cm
        # off and "blocked" the pick with the block, run 0109. A touching
        # neighbour's centre is at least a width away, outside its own box.)
        dx, dy = block["x"] - px, block["y"] - py
        along = dx * math.cos(oyaw) + dy * math.sin(oyaw)
        across = -dx * math.sin(oyaw) + dy * math.cos(oyaw)
        if abs(along) < long_half + 0.005 and abs(across) < short_half + 0.005:
            continue
        if math.hypot(dx, dy) < 0.025:
            continue  # a re-detection of the block 1.9 cm off (run 0193); a touching neighbour's centre is 2.8 cm away
        # Standing blocks matter to the camera (55 mm tall against a camera
        # underside ~50 mm above the pads, measured 2026-11-17); lying ones
        # (28 mm) pass 29 mm under it.
        top = float(geometry.pose.position[2]) + extents[2] / 2.0
        out.append((px, py, oyaw, long_half, short_half,
                    is_standing(top, max(extents[0], extents[1]))))
    return out


def finger_axis(yaw):
    # down_quat(yaw) points the hand's x axis along `yaw`; the Franka Hand's
    # fingers travel along the hand's y axis, i.e. along yaw - 90 degrees.
    # (Checked 2026-11-17 against the hold width: 59 mm, the block's length,
    # after a day of "closing across the short axis" with 12 mm of margin.)
    return yaw - math.pi / 2.0


def fold(yaw):
    while yaw > math.pi:
        yaw -= 2.0 * math.pi
    while yaw <= -math.pi:
        yaw += 2.0 * math.pi
    return yaw


def camera_clearance(block, yaw, others):
    """Distance from the wrist camera's footprint to the nearest other block.

    The camera hangs CAMERA_OFFSET from the fingertips along the hand's x
    axis (which down_quat(yaw) points along `yaw`); it is part of the hand's
    footprint. Run 0114's seven failed attempts at one pick were the camera
    body landing on blue's end and on green, never the fingers: contact
    whenever its footprint overlapped a block, floor whenever it cleared.
    """
    cx = block["x"] + CAMERA_OFFSET[0] * math.cos(yaw) - CAMERA_OFFSET[1] * math.sin(yaw)
    cy = block["y"] + CAMERA_OFFSET[0] * math.sin(yaw) + CAMERA_OFFSET[1] * math.cos(yaw)
    nearest = 1.0
    for ox, oy, oyaw, long_half, short_half, tall in others:
        if not tall:
            continue
        dx, dy = cx - ox, cy - oy
        along = abs(dx * math.cos(oyaw) + dy * math.sin(oyaw)) - long_half
        across = abs(-dx * math.sin(oyaw) + dy * math.cos(oyaw)) - short_half
        nearest = min(nearest, max(along, across) - CAMERA_HALF_M)
    return nearest


JOINT_LOWER = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
JOINT_UPPER = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
LIMIT_MARGIN_RAD = 0.3


def joint_margin(x, y, z, yaw):
    """How far the IK solution for this hand pose stays from every joint limit."""
    ik = solve_ik(Pose([x, y, z], down_quat(yaw), "base"))
    if not ik.ok:
        return -1.0
    q = [float(v) for v in ik.joint_positions]
    return min(min(q[i] - JOINT_LOWER[i], JOINT_UPPER[i] - q[i]) for i in range(7))


def grasp_yaw(block, others):
    """Fingers across the block's 28 mm width, by a yaw the arm can hold.

    The gripper is symmetric, so yaw and yaw + 180 are the same grasp for
    the fingers but not for the arm or the camera. Of the two: a yaw whose
    IK at the pick hover or at either place spot winds a joint to within
    0.3 rad of its limit is out (run 0129: the base joint at -2.88 rad,
    one degree from its stop, and the motion generator stalled on the
    pre-place); then the camera footprint farther from every other block;
    then the shorter wrist turn.
    """
    base = box_yaw(block["geometry"])  # the long axis; fingers travel at yaw - 90, across the width
    options = []
    for candidate in (fold(base), fold(base + math.pi)):
        margin = min(joint_margin(block["x"], block["y"], SAFE_Z, candidate),
                     joint_margin(A[0], A[1], 0.06, candidate),
                     joint_margin(B[0], B[1], 0.06, candidate),
                     joint_margin(C[0], C[1], 0.06, candidate))
        options.append((margin >= LIMIT_MARGIN_RAD,
                        round(camera_clearance(block, candidate, others), 3),
                        -abs(candidate), candidate, round(margin, 3)))
    options.sort(reverse=True)
    return options[0][3], options[0][1], options[0][4]


def finger_clear(block, yaw, others):
    # Open, the pads land FINGER_HALF_OPEN_M either side of the block along
    # the finger axis. A pad landing inside a neighbour's oriented footprint,
    # grown by the pad and a clearance, is a collision (run 0010: the descent
    # to grasp fired the reflex). A circle of the longest extent was too
    # cautious (run 0013).
    margin = FINGER_PAD_HALF_M + NEIGHBOUR_CLEAR_M
    axis = finger_axis(yaw)
    for sign in (1.0, -1.0):
        fx = block["x"] + sign * FINGER_HALF_OPEN_M * math.cos(axis)
        fy = block["y"] + sign * FINGER_HALF_OPEN_M * math.sin(axis)
        for ox, oy, oyaw, long_half, short_half, tall in others:
            dx, dy = fx - ox, fy - oy
            along = dx * math.cos(oyaw) + dy * math.sin(oyaw)
            across = -dx * math.sin(oyaw) + dy * math.cos(oyaw)
            if abs(along) < long_half + margin and abs(across) < short_half + margin:
                return [round(ox, 3), round(oy, 3)]
    return None


def choose_yaw(block, others):
    """The grasp across the short axis, or the neighbour that blocks it.

    Closing across the length is not a fallback: with 80 mm between the pads
    and a 55 mm block the margin is 12 mm a side, inside the localization
    error -- which, it turned out, is the grasp every run before 0106 made.
    """
    yaw, camera_room, margin = grasp_yaw(block, others)
    blocked_by = finger_clear(block, yaw, others)
    if blocked_by is None and margin < LIMIT_MARGIN_RAD:
        blocked_by = ["joints", margin]
    if blocked_by is None and camera_room < CAMERA_CLEAR_M:
        blocked_by = ["camera", round(camera_room, 3)]
    if blocked_by is not None:
        # The fingers' line is blocked across the width; a bleed-inflated
        # footprint can also have the axes the wrong way round (run 0145:
        # "blocked" by green 6 cm off along what was really the width). Try
        # the fingers turned 90 degrees: across the length, 12 mm of margin
        # a side, with the shifted retries to cover it.
        for candidate in (fold(yaw + math.pi / 2.0), fold(yaw - math.pi / 2.0)):
            alt_margin = min(joint_margin(block["x"], block["y"], SAFE_Z, candidate),
                             joint_margin(A[0], A[1], 0.06, candidate),
                             joint_margin(B[0], B[1], 0.06, candidate),
                             joint_margin(C[0], C[1], 0.06, candidate))
            if alt_margin < LIMIT_MARGIN_RAD:
                continue
            if finger_clear(block, candidate, others) is not None:
                continue
            if camera_clearance(block, candidate, others) < CAMERA_CLEAR_M:
                continue
            return candidate, None
    return yaw, blocked_by


def contact_ee_z(block):
    vertical = BLOCK_LONG_M if block.get("standing") or block["footprint"] < 0.045 else BLOCK_FLAT_M
    return TABLE_Z + (vertical - GRASP_OFFSET)


def pose(x, y, z, quat):
    return {"position": [x, y, z], "frame": "base", "quaternion_wxyz": quat}


def pick(block, yaw):
    quat = down_quat(yaw)
    top = block["top"]
    if not block.get("standing"):
        # A lying block's top is where its 28 mm puts it; a mask's bleed can
        # read it a centimetre high (0.0545, run 0339), and a grasp floor set
        # from that put a pad standing on the block within the near-floor
        # band, so the "floor" close met air, eleven times.
        top = min(top, LYING_TOP_MAX)
    grasp_z = max(top - GRASP_OFFSET, TABLE_Z + 0.008)
    turned = False
    recentred = False
    relooked = False
    trust_relook = True
    open_gripper()
    if not must(move_to_pose(pose(block["x"], block["y"], SAFE_Z, quat),
                             max_steps=LONG_MOVE_STEPS), "hover"):
        return None
    # The last centimetres by contact descent with the floor at the grasp
    # height: the open-loop generator once stopped 8 mm high inside its joint
    # tolerance and still reported converged, and the pads then closed above
    # the block (run 0015). The hand reaches a floor to a millimetre; a
    # contact stop above it means a pad is standing on the block, i.e. the
    # block lies off along the finger axis by more than the open fingers
    # forgive -- so the approach is retried shifted along that axis (runs
    # 0033, 0049: the eye's look-to-look wobble on a loose mount is ~2 cm).
    # Shifts along the finger axis (a pad on the block, or air: the block is
    # off along the fingers) and across it (air with the block off the
    # pads' line). The eye after a camera change can be 4 cm out; across the
    # width the straddle forgives 2.6 cm along the fingers and ~2 across.
    shifts = [(0.0, 0.0), (0.025, 0.0), (-0.025, 0.0), (0.05, 0.0), (-0.05, 0.0),
              (0.0, 0.025), (0.0, -0.025), (0.025, 0.025), (-0.025, -0.025),
              (0.025, -0.025), (-0.025, 0.025)]
    axis = finger_axis(yaw)
    x0, y0 = block["x"], block["y"]
    attempts = []
    held = 0.0
    queue = list(shifts)
    while queue and len(attempts) < MAX_APPROACHES:
        along, across = queue.pop(0)
        x = x0 + along * math.cos(axis) - across * math.sin(axis)
        if x > PICK_X_MAX:
            continue  # an attempt that far forward would sweep the block off the table
        y = y0 + along * math.sin(axis) + across * math.cos(axis)
        if not must(move_to_pose(pose(x, y, grasp_z + PRE_PLACE_ABOVE_M, quat),
                                 max_steps=LONG_MOVE_STEPS), "approach"):
            return None
        reach = franka_real.descend_to_contact(floor_z=grasp_z)
        if not must(reach, "descend"):
            return None
        stop = reach.diagnostics["stop"]
        contact_z = reach.diagnostics["contact_z"]
        if stop == "contact" and contact_z is not None and contact_z - grasp_z < NEAR_FLOOR_M:
            # The pads met the table a few millimetres above the floor: the
            # front of the table stands ~5 mm higher than the row (measured
            # 2026-11-17), and at the spare spot D the lying-block floor sat
            # 2 mm below where the pads touch, so every descent stopped on
            # "contact" and each retry shoved the block (run 0300). The pads
            # straddle the block at this height all the same: close.
            stop = "floor"
            attempt_stop = "floor_by_contact"
        else:
            attempt_stop = stop
        attempt = {"along": along, "across": across, "stop": attempt_stop, "z": contact_z}
        attempts.append(attempt)
        if stop == "contact":
            # A pad on the block pushes it (Doug watched it, run 0125: the
            # block moved and the shifted retries closed on empty air). Look
            # again from the hover height before searching any further, and
            # search around where the block is now.
            if not must(move_to_pose(pose(x, y, SAFE_Z, quat), max_steps=LONG_MOVE_STEPS),
                        "re_look"):
                return None
            seen = see_lying([x0, y0], 0.08) if trust_relook else None
            if seen is not None:
                sx, sy = float(seen.pose.position[0]), float(seen.pose.position[1])
                attempt["moved_to"] = [round(sx, 4), round(sy, 4)]
                x0, y0 = sx, sy
            if along == 0.0 and across == 0.0 and not turned:
                # A pad on the block with the hand centred over it means the
                # fingers travel along its length and a pad meets its end
                # (run 0209: the same contact at zero shift through five
                # re-looks). Turn the fingers 90 degrees before shifting.
                yaw = fold(yaw + math.pi / 2.0)
                quat = down_quat(yaw)
                axis = finger_axis(yaw)
                turned = True
                attempt["turned"] = True
                queue = list(shifts)
            elif (along == 0.0 and across == 0.0 and trust_relook
                  and math.hypot(x0 - block["x"], y0 - block["y"]) > RELOOK_TRUST_M):
                # The fingers are turned, the hand is centred where the
                # re-look from the hover put the block, and a pad is on it
                # again: that look is not telling the truth (run 0315: the
                # hover read the blue 3 cm from the survey's reading, and
                # twelve descents on its spot all met a block). The survey
                # saw it well enough for a pad to find it; search from there.
                trust_relook = False
                x0, y0 = block["x"], block["y"]
                attempt["reverted_to_survey"] = True
                queue = list(shifts)
            else:
                # The search starts over around where the block is now (run
                # 0206: after a re-look only one shift remained, and it missed).
                queue = [shift for shift in shifts if shift != (along, across)]
            continue
        if stop == "floor":
            set_gripper(0.25)
            held = float(get_robot_state().gripper_positions["primary"])
            attempt["held"] = round(held, 3)
            if held >= LENGTHWISE_HELD and not turned and not block.get("standing"):
                # The pads closed on the block's 56 mm length, not its 28 mm
                # width: a bleed-inflated mask had the axes the wrong way
                # round. That grip has 12 mm of margin and let go on the
                # carry (run 0286). The gripper reading is the tell: open,
                # turn the fingers 90 degrees, and grasp again.
                attempt["lengthwise"] = True
                open_gripper()
                if not must(move_to_pose(pose(x, y, grasp_z + PRE_PLACE_ABOVE_M, quat),
                                         max_steps=LONG_MOVE_STEPS), "re_approach"):
                    return None
                yaw = fold(yaw + math.pi / 2.0)
                quat = down_quat(yaw)
                axis = finger_axis(yaw)
                turned = True
                x0, y0 = x, y
                queue = list(shifts)
                continue
            if HELD_MIN <= held < HELD_SQUARE and not recentred:
                # The pads closed on a corner or an edge, not across the
                # block (run 0297: a 17 mm hold on a 28 mm block slipped out
                # before the carry). Open, look again from above to recentre,
                # and search again from there, once.
                attempt["corner"] = True
                recentred = True
                open_gripper()
                if not must(move_to_pose(pose(x, y, SAFE_Z, quat), max_steps=LONG_MOVE_STEPS),
                            "re_look"):
                    return None
                seen = see_lying([x0, y0], 0.08)
                if seen is not None:
                    x0, y0 = float(seen.pose.position[0]), float(seen.pose.position[1])
                    attempt["moved_to"] = [round(x0, 4), round(y0, 4)]
                queue = list(shifts)
                continue
            if held >= HELD_MIN:
                block["x"], block["y"] = x, y  # where the fingers actually hold it
                break
            open_gripper()
            if along == 0.0 and across == 0.0 and not relooked:
                # Air at the hand's first, centred close: the survey from
                # 33 cm up had the block somewhere else (run 0303: a block
                # on end read as lying, the fingers closing 3 cm beside it,
                # eleven times). Look again from the hover, straight above,
                # where the top face is unmistakable, and search from there.
                relooked = True
                attempt["relooked"] = True
                if not must(move_to_pose(pose(x, y, SAFE_Z, quat), max_steps=LONG_MOVE_STEPS),
                            "re_look"):
                    return None
                seen = see_lying([x0, y0], 0.10)
                if seen is not None:
                    x0, y0 = float(seen.pose.position[0]), float(seen.pose.position[1])
                    ext = [float(v) for v in seen.extents]
                    top = float(seen.pose.position[2]) + ext[2] / 2.0
                    attempt["moved_to"] = [round(x0, 4), round(y0, 4), round(top, 4)]
                    if is_standing(top, max(ext[0], ext[1])) and not block.get("standing"):
                        block["standing"] = True
                        block["top"] = top
                        grasp_z = max(top - GRASP_OFFSET, TABLE_Z + 0.008)
                queue = list(shifts)
                continue
        if not must(move_to_pose(pose(x, y, grasp_z + PRE_PLACE_ABOVE_M, quat),
                                 max_steps=LONG_MOVE_STEPS), "re_approach"):
            return None
    else:
        open_gripper()
        must(move_to_pose(pose(x0, y0, SAFE_Z, quat), max_steps=LONG_MOVE_STEPS), "lift")
        last = attempts[-1]
        return {"held": held, "quat": quat, "ok": False,
                "approach": "air" if last["stop"] == "floor" else last["stop"],
                "approach_z": last["z"], "attempts": attempts}
    if not must(move_to_pose(pose(block["x"], block["y"], CARRY_Z, quat),
                             max_steps=LONG_MOVE_STEPS), "lift"):
        return None
    return {"held": held, "quat": quat, "ok": True, "attempts": attempts}


def blocks_in_view():
    """Every block in one look from above the row's centre: (x, y) pairs.

    One look serves every spot: a survey straight above a far spot (D at
    x 0.60) left the motion generator unconverged at the edge of reach
    (run 0178), and the row-centre look sees all the spots anyway.
    """
    if not must(survey(0.50, -0.05), "survey"):
        return None
    found = segment_text("wrist", "block")
    if not found.ok:
        return []
    depth = get_observation().cameras["wrist"].depth_m
    seen = []
    for seg in found.segmentations[:12]:
        if float(seg.score) < 0.3 or not has_depth(seg, depth):
            continue
        geometry = estimate_geometry(mask_to_point_cloud(seg, "wrist", "base"))
        if max(float(v) for v in geometry.extents) > 0.12:
            continue
        seen.append((float(geometry.pose.position[0]), float(geometry.pose.position[1])))
    return seen


def spot_clear(spot, seen):
    """Whether no block in `seen` sits within TARGET_CLEAR_M of a place spot."""
    for px, py in seen:
        if math.hypot(px - spot[0], py - spot[1]) < TARGET_CLEAR_M:
            return False
    return True


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


go_home()  # start from home: a run can begin wherever the last one left the arm (run 0121)
legs = []
origin = FIRST_LOOKS[0]
rehome = False
for trip in range(ROUND_TRIPS):
    for outbound in (True, False):
        leg = {"trip": trip, "from": origin, "status": "started"}
        legs.append(leg)
        if len(legs) == 1:
            block = None
            # The seed's colour first; if no lying block of it is within the
            # hand's working area (red at the table's edge, runs 0173, 0182),
            # the next colour, so the arm keeps practising.
            for colour in [COLOUR] + [c for c in COLOURS if c != COLOUR]:
                COLOUR = colour
                for look in FIRST_LOOKS:
                    block = find(look, FIRST_NEAR_M)
                    if block is not None:
                        break
                if block is not None:
                    break
        else:
            block = find(origin, NEAR_M)
        if block is None:
            leg["status"] = "not_found"
            break
        if len(legs) == 1:
            # A block found far from A (probing shoves it) goes home first;
            # the legs then alternate from there.
            rehome = math.hypot(block["x"] - A[0], block["y"] - A[1]) > REHOME_M
            leg["rehome"] = rehome
        target = (A if rehome else B) if outbound else (B if rehome else A)
        # A spot with another block in it (run 0085 left the lime standing at
        # A) is not a place; take the first clear spot that is not where the
        # block already is, or stop.
        chosen = None
        seen = blocks_in_view()
        if seen is None:
            leg["status"] = "move_failed"
            break
        for spot in (target, B if target == A else A, C, D, E, F, G):
            if math.hypot(spot[0] - block["x"], spot[1] - block["y"]) < TARGET_CLEAR_M:
                continue
            if spot_clear(spot, seen):
                chosen = spot
                break
        if chosen is None:
            leg["status"] = "target_blocked"
            break
        leg["target_moved"] = chosen != target
        target = chosen
        leg["to"] = target
        leg["picked_from"] = [round(block["x"], 4), round(block["y"], 4)]
        leg["footprint"] = round(block["footprint"], 4)
        leg["standing"] = bool(block["standing"])
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
        leg["approach_attempts"] = grasp.get("attempts")
        if not grasp["ok"]:
            leg["status"] = "air_close" if grasp.get("approach") == "air" else "approach_blocked"
            leg["approach"] = grasp.get("approach")
            leg["approach_z"] = grasp.get("approach_z")
            break
        placed = place(block, grasp, target)
        if placed is None:
            leg["status"] = "move_failed"
            break
        leg.update(placed)
        # The verify looks wide: the hand just put the block
        # here, and an untaught eye (new bracket, run 0085) can read it 10 cm
        # off -- which is exactly the residual the verify exists to report.
        after = find(target, VERIFY_LOOK_M)
        if after is None:
            # One wild look (run 0185: the block read 24 cm from where the
            # hand had just released it) is not a lost block: look once more.
            after = find(target, VERIFY_LOOK_M)
        leg["eye_level"] = franka_real.get_controller_metadata().get("wrist_level")
        if after is None:
            leg["status"] = "lost"
            break
        error = math.hypot(after["x"] - target[0], after["y"] - target[1])
        leg["found_at"] = [round(after["x"], 4), round(after["y"], 4)]
        leg["error_m"] = round(error, 4)
        slipped = math.hypot(after["x"] - placed["released_at"][0],
                             after["y"] - placed["released_at"][1])
        if slipped > DROPPED_M:
            # The block is nowhere near where the hand released it: it left
            # the jaws on the carry (runs 0189, 0202), and that is not an eye
            # error to learn from (a 16 cm "residual" once went in at half
            # weight and bent the correction by 8 cm).
            leg["status"] = "dropped"
            leg["slipped_m"] = round(slipped, 4)
            break
        # Report the placement residual: seen against released-at. Recorded,
        # not taught (weight 0): the lessons walked the correction 3 cm in
        # one run (0351, every residual +2-3 cm in base y, none shrinking)
        # and at 6 cm the survey and the hover disagreed by 6 cm and the pads
        # closed on air (0353). Touch, first in every batch, is the authority
        # until the controlled test says what a placement residual measures.
        lesson = franka_real.report_hand_eye_residual(
            [after["x"], after["y"]], placed["released_at"], weight=0.0)
        leg["eye_residual_xy"] = lesson["residual_xy"]
        leg["eye_drift"] = bool(lesson["drift"])
        leg["eye_offset_xyz"] = lesson["offset_xyz"]
        looks = 1
        while lesson["drift"] and looks < 3:
            # A large residual: look again before calling the placement
            # off-target (run 0065: a 29 mm jump on one look).
            again = find(target)
            if again is None:
                break
            looks += 1
            after = again
            error = math.hypot(after["x"] - target[0], after["y"] - target[1])
            lesson = franka_real.report_hand_eye_residual(
                [after["x"], after["y"]], placed["released_at"], weight=0.0)
            leg["found_at"] = [round(after["x"], 4), round(after["y"], 4)]
            leg["error_m"] = round(error, 4)
            leg["eye_residual_xy"] = lesson["residual_xy"]
            leg["eye_offset_xyz"] = lesson["offset_xyz"]
        leg["looks"] = looks
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
    "colour": COLOUR,
    "spots": {"A": A, "B": B, "C": C, "D": D, "E": E, "F": F, "G": G},
    "verified": verified,
    "of": 2 * ROUND_TRIPS,
    "legs": legs,
    "stage": FAILED_MOVE["stage"],
    "error": FAILED_MOVE["error"],
}
