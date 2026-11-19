# Motion only: home, then a loop of survey poses around the table, then home.
# No looks, no grasps. A run of this with the recorder on is the control loop
# under the harness's own load and nothing else; a communication reflex here
# is the host, not the skill (2026-11-19: communication_constraints_violation
# reflexes in calibrations and program runs, the link itself clean under ping).

SURVEY_Z = 0.33
LONG_MOVE_STEPS = 900
LOOKS = [[0.50, 0.0], [0.50, -0.15], [0.45, -0.22], [0.50, 0.10], [0.55, -0.05],
         [0.45, 0.05], [0.50, -0.10], [0.50, 0.0]]
ROUNDS = 2

FAILED = []


def survey(x, y):
    return move_to_pose({"position": [x, y, SURVEY_Z], "frame": "base",
                         "rpy_deg": [180.0, 0.0, 0.0]}, max_steps=LONG_MOVE_STEPS)


go_home()
moves = 0
for round_index in range(ROUNDS):
    for look in LOOKS:
        moved = survey(look[0], look[1])
        moves += 1
        if not moved.ok:
            FAILED.append({"move": moves, "look": look, "error": str(moved.error.message)[:160]})
go_home()
result = {"ok": not FAILED, "moves": moves, "failed": FAILED}
