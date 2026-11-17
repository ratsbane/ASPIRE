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
BODY_BELOW_PADS_M = 0.050    # the hand body's underside sits this far above the pads
SURVEY_Z = 0.33
LONG_MOVE_STEPS = 600
COARSE_PITCH_M = 0.02
COARSE_HALF = 2              # 5 x 5 cells, +-4 cm around the camera's estimate
FINE_PITCH_M = 0.01
FINE_HALF = 1                # 3 x 3 cells, +-1 cm around the coarse centroid
PROBE_ABOVE_M = 0.03

# Two blocks far apart: blue lying flat on the row, the lime block standing
# near (0.36, -0.20). Their bearing fixes the yaw of the correction.
TARGETS = [
    {"name": "blue", "prompt": "blue block", "standing": False},
    {"name": "lime", "prompt": "green block", "standing": True},
]

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
    """The wrist camera's estimate of the target from the survey pose."""
    if target["standing"]:
        found = segment_text("wrist", target["prompt"])
        if not found.ok:
            return None
        best = None
        for seg in found.segmentations[:8]:
            geometry = estimate_geometry(mask_to_point_cloud(seg, "wrist", "base"))
            extents = [float(v) for v in geometry.extents]
            if max(extents) > 0.12:
                continue
            if best is None or extents[2] > best[1]:
                best = (geometry, extents[2])
        if best is None:
            return None
        geometry = best[0]
    else:
        found = localize_object(target["prompt"])
        if not found.ok:
            return None
        geometry = found.geometry
    px, py, pz = (float(v) for v in geometry.pose.position)
    extents = [float(v) for v in geometry.extents]
    return {"x": px, "y": py, "top": pz + extents[2] / 2.0, "extents": extents}


def probe(x, y, top_est, floor_z):
    """One contact descent at (x, y); the pads' stopping height, or None."""
    if not must(down(x, y, max(top_est, BARE_HAND_TABLE_Z) + PROBE_ABOVE_M), "pre_probe"):
        return None
    touch = franka_real.descend_to_contact(floor_z=floor_z)
    if not must(touch, "probe"):
        return None
    report = touch.diagnostics
    if report["stop"] == "contact":
        return float(report["contact_z"])
    return floor_z


def grid_hits(cx, cy, pitch, half, top_est, floor_z):
    hits = []
    probes = []
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            x, y = cx + i * pitch, cy + j * pitch
            if not (0.27 < x < 0.64 and -0.31 < y < 0.31):
                continue
            z = probe(x, y, top_est, floor_z)
            if z is None:
                return hits, probes, False
            hit = z > BARE_HAND_TABLE_Z + HIT_ABOVE_TABLE_M and z > floor_z + 0.004
            probes.append([round(x, 4), round(y, 4), round(z, 4), hit])
            if hit:
                hits.append((x, y))
    return hits, probes, True


def centroid(points):
    return [sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)]


measurements = []
for target in TARGETS:
    entry = {"name": target["name"]}
    measurements.append(entry)
    near = [0.50, 0.0] if target["name"] == "blue" else [0.40, -0.17]
    if not must(survey(near[0], near[1]), "survey"):
        break
    seen = see(target)
    entry["seen"] = seen
    if seen is None:
        entry["status"] = "not_seen"
        continue
    top_est = 0.055 + TABLE_Z if target["standing"] else 0.028 + TABLE_Z
    # The floor for a miss: the table for a flat block; for a standing one
    # the hand body would sweep the block's top if the pads went that low,
    # so the floor stays a body's height below the block top.
    floor_z = BARE_HAND_TABLE_Z - 0.008
    if target["standing"]:
        floor_z = max(floor_z, top_est - BODY_BELOW_PADS_M + 0.01)
    close_gripper()
    hits, probes, ok = grid_hits(seen["x"], seen["y"], COARSE_PITCH_M, COARSE_HALF, top_est, floor_z)
    entry["coarse_probes"] = probes
    if not ok:
        entry["status"] = "move_failed"
        break
    if not hits:
        entry["status"] = "no_hits"
        continue
    coarse = centroid(hits)
    entry["coarse_centroid"] = [round(v, 4) for v in coarse]
    hits2, probes2, ok = grid_hits(coarse[0], coarse[1], FINE_PITCH_M, FINE_HALF, top_est, floor_z)
    entry["fine_probes"] = probes2
    if not ok:
        entry["status"] = "move_failed"
        break
    truth = centroid(hits + hits2) if hits2 else coarse
    entry["touched"] = [round(v, 4) for v in truth]
    entry["seen_minus_touched"] = [round(seen["x"] - truth[0], 4), round(seen["y"] - truth[1], 4)]
    entry["status"] = "measured"
    # A touched position is the strongest evidence the eye gets: taken whole.
    lesson = franka_real.report_hand_eye_residual(
        [seen["x"], seen["y"]], truth, source="touch")
    entry["eye_offset_xyz"] = lesson["offset_xyz"]
    entry["eye_drift"] = bool(lesson["drift"])
    must(down(truth[0], truth[1], 0.14), "retreat")
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
    "ok": FAILED_MOVE["stage"] is None and len(measured) == len(TARGETS),
    "survey_z": SURVEY_Z,
    "measurements": measurements,
    "fit": fit,
    "stage": FAILED_MOVE["stage"],
    "error": FAILED_MOVE["error"],
}
