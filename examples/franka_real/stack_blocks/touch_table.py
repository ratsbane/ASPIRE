# Live validation of franka_real.descend_to_contact with nothing in hand: close
# the fingers, hover over a clear patch of table, descend by contact three
# times, and report where the hand stopped each time. A stop at the floor
# means the trigger missed, a cartesian reflex means it fired late.
# The spot sits between the sorted row (y = 0) and the standing lime block
# (y = -0.20) with a hand's width to spare on both sides.
#
# Measured 2026-11-17 (run 0004): contact at z = 0.0128 on all three descents,
# within 0.1 mm of each other -- the closed pads sit 4 mm below the grip line
# that meets the table at 0.017 with a block in hand. Each touch pressed ~8 N
# before the trigger fired, because at 300 N/m the command already trails the
# hand by ~23 mm (joint friction) when the pads arrive. An earlier run with the
# floor bounding the command instead of the hand never reached the table.

TABLE_Z = 0.017          # grip line at the table plane, with a block in hand
BARE_HAND_TABLE_Z = 0.0128  # closed pads at the table plane, measured
PRE_ABOVE_M = 0.03
FLOOR_BELOW_M = 0.015
SAFE_Z = 0.14
LONG_MOVE_STEPS = 600
SPOT = [0.50, -0.15]
REPEATS = 3

FAILED_MOVE = {"stage": None, "error": None}


def must(move_result, stage):
    if not move_result.ok and FAILED_MOVE["stage"] is None:
        FAILED_MOVE["stage"] = stage
        FAILED_MOVE["error"] = str(move_result.error.message)[:160]
    return move_result.ok


def above(z):
    return move_to_pose({"position": [SPOT[0], SPOT[1], z], "frame": "base",
                         "rpy_deg": [180.0, 0.0, 0.0]}, max_steps=LONG_MOVE_STEPS)


touches = []
close_gripper()
if must(above(SAFE_Z), "hover"):
    for attempt in range(REPEATS):
        if not must(above(TABLE_Z + PRE_ABOVE_M), "pre_place"):
            break
        touch = franka_real.descend_to_contact(
            floor_z=max(TABLE_Z - FLOOR_BELOW_M, 0.004))
        entry = {"attempt": attempt, "ok": bool(touch.ok)}
        if touch.ok:
            report = touch.diagnostics
            for key in ("contact", "contact_z", "start_z", "travel_m", "stop",
                        "force_rise_n", "elapsed_s", "speed_m_s"):
                entry[key] = report[key]
            if report["contact_z"] is not None:
                entry["error_mm"] = round(
                    (float(report["contact_z"]) - BARE_HAND_TABLE_Z) * 1000.0, 1)
        else:
            must(touch, "descend")
            entry["error"] = str(touch.error.message)[:160]
        touches.append(entry)
        if not touch.ok or not must(above(SAFE_Z), "lift"):
            break
go_home()

contacts = [t for t in touches if t.get("contact")]
result = {
    "ok": FAILED_MOVE["stage"] is None and len(contacts) == REPEATS,
    "expected_contact_z": BARE_HAND_TABLE_Z,
    "contacts": len(contacts),
    "of": REPEATS,
    "touches": touches,
    "stage": FAILED_MOVE["stage"],
    "error": FAILED_MOVE["error"],
}
