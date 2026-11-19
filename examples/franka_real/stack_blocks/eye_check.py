import math

# The eye against the hand's truth. Two blocks the hand released at known
# positions (run 0361: a few millimetres, by the hand's own pose at release)
# are looked at from straight above, at three heights and two wrist yaws,
# and every reading is compared with the truth, with the levelling report
# of that look beside it. Run once with the levelling on and once with
# CAP_HARNESS_WRIST_LEVEL=0; the residuals say whether the eye's 1-4 cm
# scatter (2026-11-18/19) is the plane fit, the height, the yaw, or the
# camera pose itself.

TRUTHS = {"green_a": [0.5048, 0.0012], "green_b": [0.5249, -0.1678]}
PROMPT = "green block"
HEIGHTS = [0.33, 0.25, 0.20]
YAWS_DEG = [0.0, 90.0]
LONG_MOVE_STEPS = 900
NEAR_M = 0.08

FAILED_MOVE = {"stage": None, "error": None}


def must(move_result, stage):
    if not move_result.ok and FAILED_MOVE["stage"] is None:
        FAILED_MOVE["stage"] = stage
        FAILED_MOVE["error"] = str(move_result.error.message)[:160]
    return move_result.ok


def down_quat(yaw_rad):
    return [0.0, math.cos(yaw_rad / 2.0), math.sin(yaw_rad / 2.0), 0.0]


def has_depth(seg, depth):
    covered = depth[seg.mask]
    return int((covered > 0.1).sum()) >= 50


def read(truth):
    """The nearest instance of the prompt to the truth, as the eye sees it from here."""
    observation = get_observation()
    depth = observation.cameras["wrist"].depth_m
    found = segment_text("wrist", PROMPT)
    if not found.ok:
        return None
    best = None
    for seg in found.segmentations[:8]:
        if float(seg.score) < 0.5 or not has_depth(seg, depth):
            continue
        geometry = estimate_geometry(mask_to_point_cloud(seg, "wrist", "base"))
        px, py, pz = (float(v) for v in geometry.pose.position)
        extents = [float(v) for v in geometry.extents]
        if max(extents) > 0.12:
            continue
        distance = math.hypot(px - truth[0], py - truth[1])
        if distance > NEAR_M:
            continue
        if best is None or distance < best["distance"]:
            best = {"x": round(px, 4), "y": round(py, 4), "top": round(pz + extents[2] / 2.0, 4),
                    "footprint": round(max(extents[0], extents[1]), 4), "distance": round(distance, 4),
                    "residual_xy": [round(px - truth[0], 4), round(py - truth[1], 4)]}
    return best


go_home()
readings = []
for name in sorted(TRUTHS):
    truth = TRUTHS[name]
    for height in HEIGHTS:
        for yaw_deg in YAWS_DEG:
            moved = move_to_pose({"position": [truth[0], truth[1], height], "frame": "base",
                                  "quaternion_wxyz": down_quat(math.radians(yaw_deg))},
                                 max_steps=LONG_MOVE_STEPS)
            entry = {"block": name, "truth": truth, "height": height, "yaw_deg": yaw_deg,
                     "moved": bool(moved.ok)}
            if moved.ok:
                seen = read(truth)
                entry["seen"] = seen
                level = franka_real.get_controller_metadata().get("wrist_level")
                entry["level"] = level
                hand = get_robot_state().end_effector_poses["primary"].position
                entry["hand"] = [round(float(hand[0]), 4), round(float(hand[1]), 4), round(float(hand[2]), 4)]
            else:
                must(moved, name + " " + str(height) + " " + str(yaw_deg))
            readings.append(entry)
go_home()

summary = {}
for entry in readings:
    seen = entry.get("seen")
    if seen is None:
        continue
    key = str(entry["height"]) + "/" + str(int(entry["yaw_deg"]))
    summary.setdefault(key, []).append(seen["residual_xy"])
means = {}
for key in summary:
    rows = summary[key]
    means[key] = {"n": len(rows),
                  "mean_xy": [round(sum(r[0] for r in rows) / len(rows), 4),
                              round(sum(r[1] for r in rows) / len(rows), 4)],
                  "max_m": round(max(math.hypot(r[0], r[1]) for r in rows), 4)}
result = {"ok": FAILED_MOVE["stage"] is None and all(e.get("seen") is not None for e in readings),
          "levelling": None if not readings or readings[0].get("level") is None
          else ("disabled" if readings[0]["level"].get("reason") == "disabled" else "on"),
          "means": means,
          "readings": readings,
          "move_failures": FAILED_MOVE}
