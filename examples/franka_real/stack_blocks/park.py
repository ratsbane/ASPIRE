# Park: recover from whatever the last run left and go home.
homed = go_home()
state = get_robot_state()
position = state.end_effector_poses["primary"].position
result = {"ok": bool(homed.ok), "error": None if homed.ok else str(homed.error.message)[:200],
          "hand": [round(float(v), 4) for v in position],
          "q": [round(float(v), 3) for v in state.joint_positions["primary"]]}
