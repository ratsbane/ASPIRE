import math
# The paired gate's other arm for a language-conditioned policy (SmolVLA,
# 2026-11-22): the program decides each leg's goal, the policy drives it.
#
# The written skill (round_trip.py) carries one colour between spots A and B
# six times. Which block and where it goes depend on state the cameras do not
# show (the colour from the seed, the leg count, the spots' jitter), which is
# why the earlier, goal-less policies hovered between options. Here each leg
# starts with a survey: the program finds the block it would carry and the
# spot it would carry it to, and says so in the words the policy was trained
# on, "carry the blue block from (52, -15) to (53, 0)" (centimetres, as
# franka_raw_to_lerobot.py --per-leg wrote them). The policy then moves the arm
# chunk by chunk under the same guards as learned_round_trip.py until it
# releases or runs out of budget, and the survey that starts the next leg
# checks where the block landed: within VERIFY_M of the goal is a verified leg.

SURVEY_Z = 0.33
LONG_MOVE_STEPS = 900
POLICY = "current"
FULL_CHUNK = 50           # SmolVLA's chunk: 50 steps, 2.5 s at 20 Hz
# Steps executed per chunk before the policy looks again (the execution
# horizon). The whole chunk by default; CAP_HARNESS_POLICY_STEPS sets a shorter
# one (e.g. 20: re-observe every second). The budget and the dither window are
# kept in seconds, so a shorter horizon gets proportionally more chunks.
CHUNK_STEPS = int(get_task_context().metadata.get("policy_steps") or FULL_CHUNK)
LEG_CHUNK_BUDGET = (40 * FULL_CHUNK) // CHUNK_STEPS   # 100 s of policy time per leg; the program takes ~40 s
LEGS = 6
HELD_MIN = 0.08
HELD_MAX = 0.60
GRASP_Z_MAX = 0.10
DITHER_CHUNKS = (12 * FULL_CHUNK) // CHUNK_STEPS      # 30 s without travel
DITHER_M = 0.03
VERIFY_M = 0.03           # the program's own verify tolerance
LOOK = [0.50, 0.00]
COLOURS = ["blue", "green", "red"]
TABLE_X = (0.38, 0.60)
TABLE_Y = (-0.28, 0.16)
SEED = int(get_task_context().metadata.get("seed", 0))
COLOUR = COLOURS[SEED % 3]
JITTER_M = 0.02
A = [0.53 + JITTER_M * ((SEED // 3) % 3 - 1), 0.00 + JITTER_M * ((SEED // 9) % 3 - 1)]
B = [0.53 + JITTER_M * ((SEED // 27) % 3 - 1), -0.15 + JITTER_M * ((SEED // 3) % 3 - 1)]
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
    out = []
    depth = get_observation().cameras["wrist"].depth_m
    for colour in COLOURS:
        found = segment_text("wrist", colour + " block")
        if not found.ok:
            continue
        for seg in found.segmentations[:8]:
            if float(seg.score) < 0.5 or not has_depth(seg, depth):
                continue
            g = estimate_geometry(mask_to_point_cloud(seg, "wrist", "base"))
            px, py, pz = (float(v) for v in g.pose.position)
            ext = [float(v) for v in g.extents]
            if max(ext) > 0.12:
                continue
            if not (TABLE_X[0] < px < TABLE_X[1] and TABLE_Y[0] < py < TABLE_Y[1]):
                continue
            if any(o[0] == colour and math.hypot(o[1] - px, o[2] - py) < 0.025 for o in out):
                continue
            out.append([colour, px, py])
    return out


def scan():
    if not must(survey(LOOK[0], LOOK[1]), "survey"):
        return None
    return blocks_in_view()


def hand():
    state = get_robot_state()
    p = state.end_effector_poses["primary"].position
    return [float(p[0]), float(p[1]), float(p[2])], float(state.gripper_positions["primary"])


def cm(p):
    return "(" + str(round(100 * p[0])) + ", " + str(round(100 * p[1])) + ")"


def pick_goal(blocks):
    """The block of the run's colour nearest A or B, and the other spot as its goal."""
    best = None
    for colour, x, y in blocks:
        if colour != COLOUR:
            continue
        da, db = math.hypot(x - A[0], y - A[1]), math.hypot(x - B[0], y - B[1])
        d = min(da, db)
        if best is None or d < best[0]:
            best = (d, [x, y], B if da <= db else A)
    return None if best is None else (best[1], best[2])


go_home()
legs = []
events = []
chunks_total = 0
stop = None
blocks = scan()
# The seed's colour first, then the next, as round_trip.py chooses: the program
# run before this one carried whatever colour it found, and the gate's table
# may hold only that one (gate 610: two policy runs in three stopped at "block
# not found" before a single chunk, looking for red or green with only blue out).
if blocks is not None:
    for colour in [COLOUR] + [c for c in COLOURS if c != COLOUR]:
        if any(b[0] == colour for b in blocks):
            COLOUR = colour
            break
for leg_index in range(LEGS):
    if blocks is None:
        stop = "survey"
        break
    goal = pick_goal(blocks)
    if goal is None:
        stop = "block not found"
        break
    start, target = goal
    instruction = "carry the " + COLOUR + " block from " + cm(start) + " to " + cm(target)
    leg = {"instruction": instruction, "from": [round(start[0], 4), round(start[1], 4)],
           "to": [round(target[0], 4), round(target[1], 4)], "chunks": 0, "grasped": False}
    go_home()  # every training leg began where the last one ended; home is the safe common start
    released = False
    recent = []
    while leg["chunks"] < LEG_CHUNK_BUDGET:
        chunk = franka_real.policy_chunk(policy=POLICY, steps=CHUNK_STEPS, instruction=instruction)
        leg["chunks"] += 1
        chunks_total += 1
        if not chunk.ok:
            stop = str(chunk.error.code)
            events.append({"leg": leg_index + 1, "chunk": leg["chunks"], "stop": stop,
                           "error": str(chunk.error.message)[:200]})
            break
        xyz, held = hand()
        if not leg["grasped"] and HELD_MIN <= held <= HELD_MAX and xyz[2] < GRASP_Z_MAX:
            leg["grasped"] = True
            leg["grasp_at"] = [round(xyz[0], 4), round(xyz[1], 4)]
        elif leg["grasped"] and held > HELD_MAX:
            released = True
            leg["released_at"] = [round(xyz[0], 4), round(xyz[1], 4)]
            break
        recent.append([xyz[0], xyz[1]])
        if leg["grasped"]:
            recent = []
        if len(recent) >= DITHER_CHUNKS:
            w = recent[-DITHER_CHUNKS:]
            if max(math.hypot(a[0] - b[0], a[1] - b[1]) for a in w for b in w) < DITHER_M:
                leg["stop"] = "dithering"
                break
    if stop is not None:
        legs.append(leg)
        break
    open_gripper()
    go_home()
    blocks = scan()
    # Did the block reach the goal? The nearest block of the colour to the goal.
    landed = None
    if blocks is not None:
        for colour, x, y in blocks:
            if colour == COLOUR:
                d = math.hypot(x - target[0], y - target[1])
                if landed is None or d < landed:
                    landed = d
    leg["released"] = released
    leg["error_m"] = None if landed is None else round(landed, 4)
    leg["status"] = "verified" if (released and landed is not None and landed < VERIFY_M) else "not_verified"
    legs.append(leg)
    if leg["status"] != "verified":
        stop = stop or "leg not verified"
        break

open_gripper()
go_home()
verified = sum(1 for leg in legs if leg.get("status") == "verified")
safety = [e for e in events if e.get("stop") in ("ErrorCode.SAFETY_INTERLOCK", "ErrorCode.CONTROLLER_FAILED")]
result = {
    "ok": verified == LEGS and not safety,
    "producer": "policy",
    "conditioning": "instruction",
    "policy_steps": CHUNK_STEPS,
    "colour": COLOUR,
    "verified": verified,
    "carried": verified,
    "legs": legs,
    "chunks": chunks_total,
    "stop": stop,
    "events": events,
    "safety_events": len(safety),
    "failed_move": FAILED_MOVE,
}
