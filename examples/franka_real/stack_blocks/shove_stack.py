import math

# Relocate the stacked pair by pushing, not picking: fingers closed into a
# bundle, contact low on the bottom block, quasistatic slide along -y. Runs
# under stack_blocks because that is the postcondition the shove must keep:
# blue still on green, wherever the pair ends up.

PUSH_EE_Z = 0.030        # fingertips low on the bottom block's side
BLOCK_HALF = 0.014
BUNDLE_HALF = 0.014      # closed fingertip bundle radius
CONTACT = BLOCK_HALF + BUNDLE_HALF
TARGET_Y = -0.205        # where the stack centre should end up
DONE_Y = -0.185          # close enough: clear of the wall line's slab windows
APPROACH_GAP = 0.055     # start the descend this far from the stack centre
SAFE_Z = 0.16
SURVEY_Z = 0.31
LONG_MOVE_STEPS = 600
MAX_PUSHES = 2
DOWN = [0.0, 1.0, 0.0, 0.0]

FAILED_MOVE = {"stage": None}


def must(move_result, stage):
    if not move_result.ok and FAILED_MOVE["stage"] is None:
        FAILED_MOVE["stage"] = stage
        FAILED_MOVE["error"] = str(move_result.error.message)[:160]
    return move_result.ok


def nadir(x, y):
    return {"position": [x, y, SURVEY_Z], "frame": "base",
            "rpy_deg": [180.0, 0.0, 0.0]}


def locate(name, above=None):
    if above is not None:
        move_to_pose(nadir(above[0], above[1]))
    found = localize_object(name)
    if not found.ok:
        return None
    centre = [float(v) for v in found.geometry.pose.position]
    if not (0.25 < centre[0] < 0.66 and -0.33 < centre[1] < 0.33):
        return None
    return found.geometry


move_to_pose(nadir(0.46, -0.16))
blue = locate("blue block")
if blue is not None:
    blue = locate("blue block", above=blue.pose.position) or blue

if blue is None:
    go_home()
    result = {"ok": False, "stage": "find", "note": "blue not localized"}
else:
    bx = float(blue.pose.position[0])
    by = float(blue.pose.position[1])
    start = [round(bx, 4), round(by, 4)]
    set_gripper(0.02)  # the pusher: a closed bundle, nothing to drop
    pushes = 0
    aborted = False
    while pushes < MAX_PUSHES and by > DONE_Y and not aborted:
        pushes += 1
        approach_y = by + APPROACH_GAP
        end_y = TARGET_Y + CONTACT
        ok = must(move_to_pose({"position": [bx, approach_y, SAFE_Z],
                                "frame": "base", "quaternion_wxyz": DOWN},
                               max_steps=LONG_MOVE_STEPS), "hover") \
            and must(move_to_pose({"position": [bx, approach_y, PUSH_EE_Z],
                                   "frame": "base", "quaternion_wxyz": DOWN},
                                  max_steps=LONG_MOVE_STEPS), "descend") \
            and must(move_to_pose({"position": [bx, end_y, PUSH_EE_Z],
                                   "frame": "base", "quaternion_wxyz": DOWN},
                                  max_steps=LONG_MOVE_STEPS), "push") \
            and must(move_to_pose({"position": [bx, end_y + 0.015, PUSH_EE_Z],
                                   "frame": "base", "quaternion_wxyz": DOWN},
                                  max_steps=LONG_MOVE_STEPS), "disengage") \
            and must(move_to_pose({"position": [bx, end_y + 0.015, SAFE_Z],
                                   "frame": "base", "quaternion_wxyz": DOWN},
                                  max_steps=LONG_MOVE_STEPS), "rise")
        if not ok:
            aborted = True
            break
        blue = locate("blue block", above=[bx, TARGET_Y])
        if blue is None:
            break
        bx = float(blue.pose.position[0])
        by = float(blue.pose.position[1])

    top = None
    if blue is not None:
        top = float(blue.pose.position[2]) + float(blue.extents[2]) / 2.0
    green = locate("green block")
    gap = None
    if blue is not None and green is not None:
        gap = math.hypot(bx - float(green.pose.position[0]),
                         by - float(green.pose.position[1]))
    moved = blue is not None and by <= DONE_Y
    # Height carries the intact-stack check: the bench has a second green
    # block (standing at the pick target), so localize_object's single green
    # may be the wrong one and the gap below is data, not a gate. Green fully
    # hidden under blue is likewise consistent with an intact stack.
    stacked = blue is not None and top is not None and top >= 0.055
    go_home()
    result = {"ok": bool(moved and stacked and not aborted),
              "stage": "shoved" if moved else
              (FAILED_MOVE["stage"] if aborted else "short"),
              "from": start,
              "to": None if blue is None else [round(bx, 4), round(by, 4)],
              "blue_top": None if top is None else round(top, 4),
              "blue_green_gap": None if gap is None else round(gap, 4),
              "pushes": pushes,
              "move_failures": FAILED_MOVE}
