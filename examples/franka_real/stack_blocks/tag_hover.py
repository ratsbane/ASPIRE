# Hover still over each AprilTag on the table, long enough that the frames in
# the middle of each hold show the hand's pose beyond doubt, whatever the clock
# pairing between camera and arm (Doug, 2026-11-20: move the camera to fixed
# positions over the tags). Two heights and two wrist yaws per tag. Nothing is
# touched; the lowest hover is 20 cm above the table.

LONG_MOVE_STEPS = 900
HOLD_LOOKS = 60       # observations at each hold: ~8 s still on the direct cable (25 gave only ~3.5 s, 2026-11-22)
TAGS = {"0": [0.502, 0.241], "1": [0.509, -0.266]}   # where the survey looks put them
HEIGHTS = [0.30, 0.22]
DOWN = [0.0, 1.0, 0.0, 0.0]
TURNED = [0.0, 0.7071068, 0.7071068, 0.0]

holds = []
go_home()
for name in sorted(TAGS):
    x, y = TAGS[name]
    for z in HEIGHTS:
        for label, quat in (("yaw0", DOWN), ("yaw90", TURNED)):
            moved = move_to_pose({"position": [x, y, z], "frame": "base", "quaternion_wxyz": quat},
                                 max_steps=LONG_MOVE_STEPS)
            entry = {"tag": name, "z": z, "yaw": label, "moved": bool(moved.ok)}
            if moved.ok:
                for look in range(HOLD_LOOKS):
                    get_observation()
                hand = get_robot_state().end_effector_poses["primary"].position
                entry["hand"] = [round(float(v), 4) for v in hand]
            else:
                entry["error"] = str(moved.error.message)[:140]
            holds.append(entry)
go_home()
result = {"ok": all(h["moved"] for h in holds), "holds": holds}
