import math

# Truth by touch, for the eye. The wrist camera's bracket is loose
# (2026-11-17: it rotated a few degrees and its readings moved 4-10 cm), so
# the hand-eye numbers cannot be trusted between campaigns. This program
# measures where two blocks really are by feel -- a grid of contact descents
# with the fingers closed, a pad landing on a block top stops high, the table
# stops low -- and reports each block's touched centroid beside the wrist
# camera's reading of it, plus the fitted correction (a translation in the
# base frame at the survey height, and a yaw from the two blocks' bearing).
# The wrapper that launches a campaign applies the correction to the station
# config; the program only measures.

TABLE_Z = 0.017
BARE_HAND_TABLE_Z = 0.0128   # closed pads at the table plane, measured
HIT_ABOVE_TABLE_M = 0.012    # a stop this far above the table is on a block
HIT_FORCE_N = 2.5            # ...and pushed back: a stall alone near the base is a phantom
PROBE_FORCE_N = 3.0          # press no harder than this; a pad on an edge shoves a block
PROBE_X_MIN = 0.37           # at ~0.35 the descent stalled on itself (run 0019); a hit now needs force too
SEEN_RADIUS_M = 0.25         # probing shoves the block a few cm each calibration; look wide
BODY_BELOW_PADS_M = 0.050    # the hand body's underside sits this far above the pads
SURVEY_Z = 0.33
LONG_MOVE_STEPS = 600
COARSE_PITCH_M = 0.02
COARSE_HALF = 2              # 5 x 5 cells, +-4 cm around the camera's estimate
FINE_PITCH_M = 0.01
FINE_HALF = 1                # 3 x 3 cells, +-1 cm around the coarse centroid
PROBE_ABOVE_M = 0.03

# Two flat blocks on the row: blue near (0.49, 0.0) and the green near
# (0.41, 0.0) -- "green block" also matches the standing lime at (0.36, -0.20),
# so the instance nearest the expected spot is taken. Run 0019 tried the lime:
# the segmentation picked the wrong green, and probing at x ~0.35 reported
# contacts 6-10 cm up where nothing stands (a near-base stall), so a second
# target has to sit where the arm probes honestly.
TARGETS = [
    {"name": "blue", "prompt": "blue block", "near": [0.49, 0.0]},
    {"name": "green", "prompt": "green block", "near": [0.41, 0.0]},
]
# Those two spots are where the blocks start a session; after a day of
# campaigns they are wherever the last run left them (run 0233: blue in the
# basket, green at the front edge, red on end), so the targets are now
# whichever lying blocks the looks find, named by colour.
COLOURS = ["blue", "green", "red"]
LOOKS = [[0.49, 0.0], [0.45, -0.15], [0.50, 0.10]]
WORK_X = (0.42, 0.56)   # a target here keeps the whole probe grid (2 rings of 2 cm) short of x 0.60
                        # and clear of the base side, where the hand cannot get down (0358: 20 of 25
                        # probes on a red at x 0.34 refused, every cell at x 0.36 and nearer)
PROBE_X_MAX = 0.60      # no probe beyond this: the front edge is near (calibration 0306 pushed the
                        # green block off the table from x 0.605, and the arm found an empty table)
WORK_Y = (-0.28, 0.12)  # the basket stands beyond y 0.12
STANDING_TOP_M = 0.06   # a block on end (top ~0.072): probed with the floor just below its top
STANDING_FLOOR_BELOW_TOP_M = 0.025  # a miss beside a standing block stops here, the body still clear of it

FAILED_MOVE = {"stage": None, "error": None}


def must(move_result, stage):
    if not move_result.ok and FAILED_MOVE["stage"] is None:
        FAILED_MOVE["stage"] = stage
        FAILED_MOVE["error"] = str(move_result.error.message)[:160]
    return move_result.ok


def survey(x, y):
    return move_to_pose({"position": [x, y, SURVEY_Z], "frame": "base",
                         "rpy_deg": [180.0, 0.0, 0.0]}, max_steps=LONG_MOVE_STEPS)


def down(x, y, z):
    return move_to_pose({"position": [x, y, z], "frame": "base",
                         "rpy_deg": [180.0, 0.0, 0.0]}, max_steps=LONG_MOVE_STEPS)


def see(target):
    """The wrist camera's estimate of the target: the best-scoring instance
    within reach of its expected spot. (Nearest-to-the-spot measured green
    twice as "blue" in run 0032: a colour prompt returns every block, scored.)"""
    found = segment_text("wrist", target["prompt"])
    if not found.ok:
        return None
    best = None
    for seg in found.segmentations[:8]:
        if float(seg.score) < 0.5:
            continue  # a weak mask may hold no valid depth, and back-projecting it raises (run 0040)
        geometry = estimate_geometry(mask_to_point_cloud(seg, "wrist", "base"))
        px, py = float(geometry.pose.position[0]), float(geometry.pose.position[1])
        extents = [float(v) for v in geometry.extents]
        if max(extents) > 0.12:
            continue
        if math.hypot(px - target["near"][0], py - target["near"][1]) > SEEN_RADIUS_M:
            continue
        score = float(seg.score)
        if best is None or score > best[1]:
            best = (geometry, score)
    if best is None or best[1] < 0.5:
        return None
    geometry = best[0]
    px, py, pz = (float(v) for v in geometry.pose.position)
    extents = [float(v) for v in geometry.extents]
    return {"x": px, "y": py, "top": pz + extents[2] / 2.0, "extents": extents}


def probe(x, y, top_est, floor_z):
    """One contact descent at (x, y): (stop height, pushed back), "unreachable", or None."""
    approach = down(x, y, max(top_est, BARE_HAND_TABLE_Z) + PROBE_ABOVE_M)
    if not approach.ok:
        # A cell the arm cannot reach straight down is a cell without data,
        # not a failed calibration (run 0024 died on one at the grid's edge).
        return "unreachable"
    touch = franka_real.descend_to_contact(floor_z=floor_z, contact_force_n=PROBE_FORCE_N)
    if not must(touch, "probe"):
        return None
    report = touch.diagnostics
    if report["stop"] == "contact":
        return float(report["contact_z"]), float(report["force_rise_n"]) >= HIT_FORCE_N
    return floor_z, False


def grid_hits(cx, cy, pitch, half, top_est, floor_z):
    """Probe the grid from the centre outward, ring by ring.

    A sweep from one side shoved the block ahead of the pads (calibration
    0292: hits 6 cm wide across a 28 mm block, the centroid a cell toward
    +x, and the eye then overshot every pick by that much). Centre first,
    the pads land on the block's top where they cannot push it, and the
    edge probes that can come last; a ring with no hit after a ring with
    some ends the grid.
    """
    hits = []
    probes = []
    cells = [(i, j) for i in range(-half, half + 1) for j in range(-half, half + 1)]
    cells.sort(key=lambda c: (max(abs(c[0]), abs(c[1])), abs(c[0]) + abs(c[1])))
    ring_hits = 0
    ring = 0
    for i, j in cells:
        if max(abs(i), abs(j)) != ring:
            if hits and ring_hits == 0:
                break  # the block does not reach this ring; farther probes only shove
            ring = max(abs(i), abs(j))
            ring_hits = 0
        x, y = cx + i * pitch, cy + j * pitch
        if not (PROBE_X_MIN < x < PROBE_X_MAX and -0.31 < y < 0.31):
            # Outside the probe box counts as unreachable: a grid cut
            # by the box leans the centroid like one cut by reach.
            probes.append([round(x, 4), round(y, 4), None, False])
            continue
        felt = probe(x, y, top_est, floor_z)
        if felt is None:
            return hits, probes, False
        if felt == "unreachable":
            probes.append([round(x, 4), round(y, 4), None, False])
            continue
        z, pushed = felt
        hit = pushed and z > BARE_HAND_TABLE_Z + HIT_ABOVE_TABLE_M and z > floor_z + 0.004
        probes.append([round(x, 4), round(y, 4), round(z, 4), hit])
        if hit:
            hits.append((x, y))
            ring_hits += 1
    return hits, probes, True


def centroid(points):
    return [sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)]


def lying_blocks_in_view():
    """Every block of any colour the camera sees from here: lying ones first, then standing, best first."""
    out = []
    for colour in COLOURS:
        found = segment_text("wrist", colour + " block")
        if not found.ok:
            continue
        for seg in found.segmentations[:8]:
            if float(seg.score) < 0.5:
                continue
            geometry = estimate_geometry(mask_to_point_cloud(seg, "wrist", "base"))
            px, py, pz = (float(v) for v in geometry.pose.position)
            extents = [float(v) for v in geometry.extents]
            top = pz + extents[2] / 2.0
            if max(extents) > 0.12:
                continue
            if not (WORK_X[0] < px < WORK_X[1] and WORK_Y[0] < py < WORK_Y[1]):
                continue
            standing = top > STANDING_TOP_M
            out.append((float(seg.score), colour,
                        {"x": px, "y": py, "top": top, "extents": extents, "standing": standing}))
    # Lying blocks first: a standing one is probed with its floor just below
    # its top, so its grid says less about the eye than a lying one's.
    out.sort(key=lambda item: (item[2]["standing"], -item[0]))
    return out


go_home()  # start from home: a run can begin wherever the last one left the arm (run 0121)
measurements = []
targets = []
for look in LOOKS:
    if len(targets) >= 2:
        break
    if not survey(look[0], look[1]).ok:
        continue
    for score, colour, seen in lying_blocks_in_view():
        if any(math.hypot(seen["x"] - t["seen"]["x"], seen["y"] - t["seen"]["y"]) < 0.06
               for t in targets):
            continue  # the same block from another look
        if any(t["name"] == colour for t in targets):
            continue  # one of each colour: the probe's centroid is per prompt
        targets.append({"name": colour, "prompt": colour + " block", "near": [seen["x"], seen["y"]],
                        "seen": seen, "look": look})
        if len(targets) >= 2:
            break
if not targets:
    measurements.append({"name": None, "status": "not_seen", "looks": LOOKS})
for target in targets:
    entry = {"name": target["name"], "look": target["look"]}
    measurements.append(entry)
    near = target["near"]
    if not survey(near[0], near[1]).ok:
        # A refused survey (step budget, reach) loses this target, not the
        # calibration: one block measured is enough (run 0052).
        entry["status"] = "not_surveyed"
        continue
    seen = see(target)  # from straight above it, as the picks look
    entry["seen"] = seen
    if seen is None:
        entry["status"] = "not_seen"
        continue
    standing = bool(seen["top"] > STANDING_TOP_M)
    entry["standing"] = standing
    if standing:
        # A block on end: a miss beside it must stop well above the table,
        # or the hand body (BODY_BELOW_PADS_M above the pads) sweeps it over
        # while the pads go down. The floor sits just below its top; a hit
        # is a contact above that floor with the force rise, as for a lying one.
        top_est = seen["top"]
        floor_z = seen["top"] - STANDING_FLOOR_BELOW_TOP_M
    else:
        top_est = 0.028 + TABLE_Z
        floor_z = BARE_HAND_TABLE_Z - 0.008  # the floor for a miss is the table
    close_gripper()
    hits, probes, ok = grid_hits(seen["x"], seen["y"], COARSE_PITCH_M, COARSE_HALF, top_est, floor_z)
    entry["coarse_probes"] = probes
    if not ok:
        entry["status"] = "move_failed"
        break
    unreachable = sum(1 for p in probes if p[2] is None)
    if unreachable > len(probes) // 5:
        # Half a grid finds half a block: the centroid of the hits leans
        # toward the reachable side (run 0028: green at x 0.42 lost 10 of 15
        # cells and came out 12 mm off). Not evidence.
        entry["status"] = "partial_grid"
        entry["unreachable"] = unreachable
        continue
    if not hits:
        entry["status"] = "no_hits"
        continue
    # The coarse centroid is the measurement, to about a centimetre. A fine
    # pass at 1 cm pitch was tried (runs 0019-0052): nine more pads on the
    # block's edges at 3 N walked it 2-3 cm per calibration, and the
    # "residual" was then partly the block moving. The placements refine.
    truth = centroid(hits)
    entry["touched"] = [round(v, 4) for v in truth]
    entry["seen_minus_touched"] = [round(seen["x"] - truth[0], 4), round(seen["y"] - truth[1], 4)]
    entry["status"] = "measured"
    # A touched position is the strongest evidence the eye gets: taken whole.
    # It is also the only evidence along the block's length: the fingers
    # centre a held block across their travel, so a placement residual can
    # see the eye's error on that axis, but along the length a biased grasp
    # is carried from pick to place and back and reads as zero forever. Six
    # grasps of run 0231 all sat 15-20 mm from one end with residuals under
    # a centimetre; the correction file then held no touch sample at all.
    lesson = franka_real.report_hand_eye_residual(
        [seen["x"], seen["y"]], truth, source="touch", weight=1.0)
    entry["eye_offset_xyz"] = lesson["offset_xyz"]
    entry["eye_drift"] = bool(lesson["drift"])
    # A refused retreat (reach limit above the block) costs nothing: the
    # measurement is already reported, and go_home follows (run 0072).
    down(truth[0], truth[1], 0.14)
go_home()

measured = [m for m in measurements if m.get("status") == "measured"]
fit = None
if measured:
    ex = sum(m["seen_minus_touched"][0] for m in measured) / len(measured)
    ey = sum(m["seen_minus_touched"][1] for m in measured) / len(measured)
    fit = {"translation_error_xy": [round(ex, 4), round(ey, 4)], "blocks": len(measured)}
    if len(measured) == 2:
        a, b = measured
        seen_bearing = math.atan2(b["seen"]["y"] - a["seen"]["y"], b["seen"]["x"] - a["seen"]["x"])
        true_bearing = math.atan2(b["touched"][1] - a["touched"][1], b["touched"][0] - a["touched"][0])
        fit["yaw_error_deg"] = round(math.degrees(seen_bearing - true_bearing), 2)
result = {
    # One block by feel fixes the translation, which is what the eye needs;
    # the second only adds the yaw estimate.
    "ok": FAILED_MOVE["stage"] is None and len(measured) >= 1,
    "survey_z": SURVEY_Z,
    "measurements": measurements,
    "fit": fit,
    "stage": FAILED_MOVE["stage"],
    "error": FAILED_MOVE["error"],
}
