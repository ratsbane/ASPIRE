import math

# Spread a clump of blocks so each one can be picked. The pick has no move
# for blocks that touch (run 0344: blue wedged between two greens, twelve
# pads on a neighbour). The move here is a sweep, as Doug described it:
# start in the clear beside the clump, lower the closed fingers to just
# above the table, drive straight through the clump at that height to
# roughly ten centimetres beyond it, lift, look again. The sweep crosses
# the clump's long axis, so the bundle splits a row instead of pushing it
# along as a train that stays touching. Pushes always run toward the
# table's middle; the pusher is the closed bundle of shove_stack.py.

TABLE_Z = 0.017
PUSH_EE_Z = 0.030        # fingertips low on a lying block's side (shove_stack.py)
SAFE_Z = 0.14
SURVEY_Z = 0.33
LONG_MOVE_STEPS = 900
BLOCK_LONG_M = 0.055
BLOCK_FLAT_M = 0.028
CLUMP_M = 0.065          # centres closer than this: the pads cannot get between them
LEAN_TOP = 0.055         # a lying block tops at ~0.045 (bleed read 0.0523, run 0345); higher, not standing, it leans
APPROACH_GAP = 0.035     # the descent lands this far clear of the clump's edge (eye error 1-2 cm, bundle 1.4)
PUSH_BEYOND = 0.10       # the sweep ends this far past the clump's centre
MIN_SWEEP = 0.07         # a sweep the table's edge cuts shorter than this is not worth making
MAX_SWEEPS = 3
LOOK = [0.50, 0.00]
COLOURS = ["blue", "green", "red"]
TABLE_X = (0.30, 0.62)
TABLE_Y = (-0.31, 0.20)
HAND_X = (0.34, 0.56)    # where a push may end: the front edge took two blocks (runs 0221, 0224)
HAND_Y = (-0.26, 0.18)   # the table runs to about 0.20 on the positive side (the basket is gone)
START_X = (0.32, 0.645)  # where a push may start: in the clear, out past the table's front edge (nothing
START_Y = (-0.28, 0.19)  # to drop). The box ends at 0.66, and the hover's joint-space path bulges past its
                         # target (run 0397: a start at 0.656 planned through 0.661 and was refused), so
                         # 0.645 is the limit; a stray beyond x 0.58 is marooned, and the pick reaches to
                         # 0.60 anyway. Run 0347: a clump at x 0.55 had no room either way inside the end box.
BAND_X = (0.40, 0.56)    # where a block can be probed and picked: nearer the base the hand cannot get
BAND_Y = (-0.26, 0.15)   # down (calibration 0358: 20 of 25 probes at x 0.30-0.36 refused); beyond 0.56
                         # the front edge is near. A stray outside the band is swept in; one with no
                         # room behind it (the base side) is marooned and reported.
DOWN = [0.0, 1.0, 0.0, 0.0]

FAILED_MOVE = {"stage": None, "error": None}


def must(move_result, stage):
    if not move_result.ok and FAILED_MOVE["stage"] is None:
        FAILED_MOVE["stage"] = stage
        FAILED_MOVE["error"] = str(move_result.error.message)[:160]
    return move_result.ok


def pose(x, y, z):
    return {"position": [x, y, z], "frame": "base", "quaternion_wxyz": DOWN}


def survey(x, y):
    return move_to_pose({"position": [x, y, SURVEY_Z], "frame": "base",
                         "rpy_deg": [180.0, 0.0, 0.0]}, max_steps=LONG_MOVE_STEPS)


def has_depth(seg, depth):
    covered = depth[seg.mask]
    return int((covered > 0.1).sum()) >= 50


def box_yaw(geometry):
    matrix = geometry.pose.as_matrix()
    axis = 0 if geometry.extents[0] >= geometry.extents[1] else 1
    return math.atan2(float(matrix[1, axis]), float(matrix[0, axis]))


def blocks_in_view():
    """Every block the wrist camera sees from here: colour, x, y, top, long-axis yaw, footprint."""
    out = []
    depth = get_observation().cameras["wrist"].depth_m
    for colour in COLOURS:
        found = segment_text("wrist", colour + " block")
        if not found.ok:
            continue
        for seg in found.segmentations[:8]:
            if float(seg.score) < 0.5 or not has_depth(seg, depth):
                continue
            geometry = estimate_geometry(mask_to_point_cloud(seg, "wrist", "base"))
            px, py, pz = (float(v) for v in geometry.pose.position)
            extents = [float(v) for v in geometry.extents]
            if max(extents) > 0.12:
                continue
            if not (TABLE_X[0] < px < TABLE_X[1] and TABLE_Y[0] < py < TABLE_Y[1]):
                continue
            if any(o["colour"] == colour and math.hypot(o["x"] - px, o["y"] - py) < 0.025 for o in out):
                continue
            out.append({"colour": colour, "x": round(px, 4), "y": round(py, 4),
                        "top": round(pz + extents[2] / 2.0, 4), "yaw": round(box_yaw(geometry), 3),
                        "footprint": round(max(extents[0], extents[1]), 4)})
    return out


def scan(at):
    if not must(survey(at[0], at[1]), "survey"):
        return None
    return blocks_in_view()


def leaning(block):
    # Tilted against a neighbour: higher than a lying block, lower than one on end.
    # Advisory only: a lone lying red read 0.0523 and 0.0601 on consecutive
    # looks (run 0345), so the top alone cannot tell a leaner from bleed.
    return LEAN_TOP < block["top"] < 0.06


def clumps(blocks):
    """Groups of two or more blocks whose centres chain within CLUMP_M, largest first.
    Each group is the list of its members."""
    groups = []
    taken = [False] * len(blocks)
    for i in range(len(blocks)):
        if taken[i]:
            continue
        group = [i]
        taken[i] = True
        k = 0
        while k < len(group):
            a = blocks[group[k]]
            for j in range(len(blocks)):
                if not taken[j] and math.hypot(a["x"] - blocks[j]["x"], a["y"] - blocks[j]["y"]) < CLUMP_M:
                    taken[j] = True
                    group.append(j)
            k += 1
        if len(group) > 1:
            groups.append([blocks[m] for m in group])
    groups.sort(key=lambda g: -len(g))
    return groups


def long_axis(group):
    """Yaw of the direction the clump is longest in: the row's line, or a lone block's length."""
    if len(group) == 1:
        return group[0]["yaw"]
    cx = sum(b["x"] for b in group) / len(group)
    cy = sum(b["y"] for b in group) / len(group)
    sxx = sum((b["x"] - cx) ** 2 for b in group)
    syy = sum((b["y"] - cy) ** 2 for b in group)
    sxy = sum((b["x"] - cx) * (b["y"] - cy) for b in group)
    return 0.5 * math.atan2(2.0 * sxy, sxx - syy)


def clamp(v, lo, hi):
    return min(hi, max(lo, v))


def margin(x, y, box_x=HAND_X, box_y=HAND_Y):
    """How far inside a box a point is (negative outside)."""
    return min(x - box_x[0], box_x[1] - x, y - box_y[0], box_y[1] - y)


def plan_sweep(group, along_axis=False):
    """Start and end of a sweep through the clump, pushing toward the table's middle:
    across its long axis (the bundle splits a row), or along it when asked (a train
    push that at least brings a clump in from the edge, where the across sweep has
    no room). None when the edge leaves no room either way."""
    cx = sum(b["x"] for b in group) / len(group)
    cy = sum(b["y"] for b in group) / len(group)
    across = long_axis(group) + (0.0 if along_axis else math.pi / 2.0)
    # The clump's half-width along the sweep line, not its full radius: a row
    # of three is 12 cm long but 4 cm across, and the start sits off its side.
    reach = max(abs((b["x"] - cx) * math.cos(across) + (b["y"] - cy) * math.sin(across))
                for b in group) + BLOCK_LONG_M / 2.0
    best = None
    for sign in (1.0, -1.0):
        dx, dy = sign * math.cos(across), sign * math.sin(across)
        start = [cx - dx * (reach + APPROACH_GAP), cy - dy * (reach + APPROACH_GAP)]
        if margin(start[0], start[1], START_X, START_Y) < 0.0:
            continue  # the clear side is off the table's working area
        # Shorten the push, along its own line, until it stays inside the box.
        beyond = PUSH_BEYOND
        for step in range(20):
            if margin(cx + dx * beyond, cy + dy * beyond) >= 0.0:
                break
            beyond -= 0.005
        if beyond < MIN_SWEEP:
            continue
        end = [cx + dx * beyond, cy + dy * beyond]
        rank = (round(margin(end[0], end[1]), 3), beyond)
        if best is None or rank > best[0]:
            best = (rank, {"start": [round(v, 4) for v in start], "end": [round(v, 4) for v in end],
                           "centre": [round(cx, 4), round(cy, 4)], "reach": round(reach, 4),
                           "direction": [round(dx, 3), round(dy, 3)], "members": len(group),
                           "along_axis": along_axis})
    if best is None and not along_axis:
        return plan_sweep(group, along_axis=True)
    return None if best is None else best[1]


def plan_bring_in(block):
    """A sweep that brings a stray block into the working band, toward the
    table's middle. None for a block already in the band; a plan marked
    marooned when there is no room behind the block for the start."""
    dx, dy = 0.0, 0.0
    if block["x"] > BAND_X[1]:
        dx = -1.0
    elif block["x"] < BAND_X[0]:
        dx = 1.0
    if block["y"] > BAND_Y[1]:
        dy = -1.0
    elif block["y"] < BAND_Y[0]:
        dy = 1.0
    if dx == 0.0 and dy == 0.0:
        return None
    norm = math.hypot(dx, dy)
    dx, dy = dx / norm, dy / norm
    reach = BLOCK_LONG_M / 2.0
    plan = {"stray": [block["colour"], block["x"], block["y"]],
            "direction": [round(dx, 3), round(dy, 3)], "members": 1, "along_axis": False}
    start = [block["x"] - dx * (reach + APPROACH_GAP), block["y"] - dy * (reach + APPROACH_GAP)]
    if margin(start[0], start[1], START_X, START_Y) < 0.0:
        plan["marooned"] = True
        return plan
    beyond = PUSH_BEYOND
    for step in range(20):
        if margin(block["x"] + dx * beyond, block["y"] + dy * beyond) >= 0.0:
            break
        beyond -= 0.005
    if beyond < MIN_SWEEP:
        plan["marooned"] = True
        return plan
    plan["start"] = [round(v, 4) for v in start]
    plan["end"] = [round(block["x"] + dx * beyond, 4), round(block["y"] + dy * beyond, 4)]
    plan["centre"] = [block["x"], block["y"]]
    plan["reach"] = round(reach, 4)
    return plan


def sweep(plan):
    sx, sy = plan["start"]
    ex, ey = plan["end"]
    dx, dy = plan["direction"]
    set_gripper(0.02)  # the pusher: a closed bundle, nothing to drop
    return (must(move_to_pose(pose(sx, sy, SAFE_Z), max_steps=LONG_MOVE_STEPS), "hover")
            and must(move_to_pose(pose(sx, sy, PUSH_EE_Z), max_steps=LONG_MOVE_STEPS), "descend")
            and must(move_to_pose(pose(ex, ey, PUSH_EE_Z), max_steps=LONG_MOVE_STEPS), "push")
            and must(move_to_pose(pose(ex - 0.015 * dx, ey - 0.015 * dy, PUSH_EE_Z),
                                  max_steps=LONG_MOVE_STEPS), "disengage")
            and must(move_to_pose(pose(ex - 0.015 * dx, ey - 0.015 * dy, SAFE_Z),
                                  max_steps=LONG_MOVE_STEPS), "rise"))


go_home()
before = scan(LOOK)
blocks = before
sweeps = []
marooned = []
stage = None
while blocks is not None and len(sweeps) < MAX_SWEEPS and FAILED_MOVE["stage"] is None:
    groups = clumps(blocks)
    if groups:
        # The clump as seen from straight above it: off the survey's nadir a
        # block reads worse, and the sweep's start must be in the clear.
        group = groups[0]
        cx = sum(b["x"] for b in group) / len(group)
        cy = sum(b["y"] for b in group) / len(group)
        near = scan([clamp(cx, 0.36, 0.56), clamp(cy, -0.24, 0.12)])
        if near is None:
            break
        groups = clumps([b for b in near if math.hypot(b["x"] - cx, b["y"] - cy) < 0.14])
        if not groups:
            blocks = scan(LOOK)
            continue
        plan = plan_sweep(groups[0])
        if plan is None:
            stage = "no room to sweep"
            sweeps.append({"ok": False, "no_room_for": [[b["colour"], b["x"], b["y"]] for b in groups[0]]})
            break
        plan["ok"] = sweep(plan)
        sweeps.append(plan)
        blocks = scan(LOOK)
        continue
    # No clump: a stray outside the working band is swept in, one a pass.
    stray = None
    for b in blocks:
        if any(m[0] == b["colour"] and math.hypot(m[1] - b["x"], m[2] - b["y"]) < 0.03
               for m in marooned):
            continue
        plan = plan_bring_in(b)
        if plan is None:
            continue
        if plan.get("marooned"):
            marooned.append(plan["stray"])
            continue
        stray = plan
        break
    if stray is None:
        break
    stray["ok"] = sweep(stray)
    sweeps.append(stray)
    blocks = scan(LOOK)

open_gripper()
go_home()
after = blocks
remaining = clumps(after) if after is not None else None
if FAILED_MOVE["stage"] is not None:
    stage = FAILED_MOVE["stage"]
elif remaining:
    stage = "still clumped"
elif stage is None:
    stage = "clear" if sweeps else "already clear"
result = {"ok": after is not None and FAILED_MOVE["stage"] is None and not remaining,
          "stage": stage,
          "sweeps": sweeps,
          "before": before,
          "after": after,
          "remaining": None if remaining is None else [[b["colour"] for b in g] for g in remaining],
          "marooned": marooned,
          "strays": None if after is None else [b["colour"] for b in after if plan_bring_in(b) is not None],
          "leaning": None if after is None else [b["colour"] for b in after if leaning(b)],
          "move_failures": FAILED_MOVE}
