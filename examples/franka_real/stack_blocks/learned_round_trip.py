import math
# The paired gate's other arm: the learned policy drives, the program watches.
#
# round_trip.py is the written skill; its clean runs are the dream's training
# data. This program runs the same scene with the policy at the wheel: from
# home it asks the policy host for a chunk of joint targets at a time
# (franka_real.policy_chunk executes each through the workspace guard and the
# robot's reflexes), and between chunks it watches the gripper and the hand for
# stage boundaries -- a close on something at table height is a grasp, an open
# after a carry is a release -- and for the stop conditions: a refused chunk, a
# controller fault, a chunk budget, or a hand that has stopped moving. It then
# surveys the table and reports which blocks moved. The record has the same
# shape as the program's where it can (legs, verified-style counts), so the
# gate script pairs them seed by seed. The policy never sees the program's
# eye; it sees what it was trained on, the two cameras and the state.
#
# Promotion rule (2026-11-17, written before any result): the policy matches
# or beats the program over >= 20 paired scenes with no safety events.

SURVEY_Z = 0.33
LONG_MOVE_STEPS = 900
POLICY = "current"
CHUNK_STEPS = 10          # of the policy's 100-step chunk, at the dataset's 20 Hz: half a second
CHUNK_BUDGET = 720        # six minutes of policy time, about a program run and a half
HELD_MIN = 0.08           # gripper fraction: below this the close met air
HELD_MAX = 0.60           # above this the fingers are open, or on nothing
GRASP_Z_MAX = 0.10        # a close with the hand this low is a grasp attempt, not a gesture
CARRY_MIN_M = 0.08        # a release this far from the grasp is a carried leg
STILL_M = 0.005           # the hand moving less than this per chunk is not moving
STILL_CHUNKS = 40         # ~twenty seconds of a still hand ends the run
MOVED_M = 0.05            # a block seen this far from where it started has moved
LOOK = [0.50, 0.00]
COLOURS = ["blue", "green", "red"]
TABLE_X = (0.30, 0.62)
TABLE_Y = (-0.31, 0.20)
SEED = int(get_task_context().metadata.get("seed", 0))
FAILED_MOVE = {"stage": None, "error": None}


def must(move_result, stage):
    if not move_result.ok and FAILED_MOVE["stage"] is None:
        FAILED_MOVE["stage"] = stage
        FAILED_MOVE["error"] = str(move_result.error.message)[:160]
    return move_result.ok


def survey(x, y):
    return move_to_pose({"position": [x, y, SURVEY_Z], "frame": "base",
                         "rpy_deg": [180.0, 0.0, 0.0]}, max_steps=LONG_MOVE_STEPS)


def has_depth(seg, depth):
    covered = depth[seg.mask]
    return int((covered > 0.1).sum()) >= 50


def blocks_in_view():
    """Every block of every colour the wrist camera sees from here, as [colour, x, y, top]."""
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
            if any(o[0] == colour and math.hypot(o[1] - px, o[2] - py) < 0.025 for o in out):
                continue
            out.append([colour, round(px, 4), round(py, 4), round(pz + extents[2] / 2.0, 4)])
    return out


def table_scan():
    if not must(survey(LOOK[0], LOOK[1]), "survey"):
        return None
    return blocks_in_view()


def hand():
    state = get_robot_state()
    position = state.end_effector_poses["primary"].position
    return ([float(position[0]), float(position[1]), float(position[2])],
            float(state.gripper_positions["primary"]))


go_home()
before = table_scan()
go_home()   # the policy's episodes all start at home

legs = []
events = []
grasp = None
chunks = 0
still = 0
last_xyz = None
stop = None
while chunks < CHUNK_BUDGET:
    chunk = franka_real.policy_chunk(policy=POLICY, steps=CHUNK_STEPS)
    chunks += 1
    if not chunk.ok:
        stop = str(chunk.error.code)
        events.append({"chunk": chunks, "stop": stop, "error": str(chunk.error.message)[:200]})
        break
    xyz, held = hand()
    if grasp is None:
        if HELD_MIN <= held <= HELD_MAX and xyz[2] < GRASP_Z_MAX:
            grasp = {"at": [round(xyz[0], 4), round(xyz[1], 4)], "chunk": chunks, "held": round(held, 3)}
    elif held > HELD_MAX or held < HELD_MIN:
        carried = math.hypot(xyz[0] - grasp["at"][0], xyz[1] - grasp["at"][1])
        legs.append({"picked_from": grasp["at"], "released_at": [round(xyz[0], 4), round(xyz[1], 4)],
                     "held": grasp["held"], "carry_m": round(carried, 4),
                     "chunks": chunks - grasp["chunk"],
                     "status": "carried" if carried >= CARRY_MIN_M else "released_near"})
        grasp = None
    if last_xyz is not None and math.dist(xyz, last_xyz) < STILL_M:
        still += 1
    else:
        still = 0
    last_xyz = xyz
    if still >= STILL_CHUNKS:
        stop = "no_progress"
        events.append({"chunk": chunks, "stop": stop})
        break
if stop is None:
    stop = "budget"

open_gripper()
go_home()
after = table_scan()
go_home()

moved = []
if before is not None and after is not None:
    for colour, x, y, top in before:
        nearest = None
        for c2, x2, y2, top2 in after:
            if c2 != colour:
                continue
            d = math.hypot(x2 - x, y2 - y)
            if nearest is None or d < nearest:
                nearest = d
        moved.append({"colour": colour, "from": [x, y],
                      "moved_m": None if nearest is None else round(nearest, 4),
                      "moved": nearest is None or nearest > MOVED_M})

carried = sum(1 for leg in legs if leg["status"] == "carried")
safety = [e for e in events if e.get("stop") in ("SAFETY_INTERLOCK", "CONTROLLER_FAILED",
                                                  "ErrorCode.SAFETY_INTERLOCK", "ErrorCode.CONTROLLER_FAILED")]
result = {
    "ok": FAILED_MOVE["stage"] is None and carried >= 6 and not safety,
    "producer": "policy",
    "policy": POLICY,
    "seed": SEED,
    "chunks": chunks,
    "chunk_steps": CHUNK_STEPS,
    "stop": stop,
    "legs": legs,
    "carried": carried,
    "events": events,
    "safety_events": len(safety),
    "before": before,
    "after": after,
    "moved": moved,
    "failed_move": FAILED_MOVE,
}
