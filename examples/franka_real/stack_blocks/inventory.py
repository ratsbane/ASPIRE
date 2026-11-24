import math

# Observation only: survey the table from look height and list what the eye
# finds for each prompt (position, extents, score), merged across looks. No
# grasps; the arm stays at SURVEY_Z. Written 2026-11-24, when Doug added
# blocks of new shapes and objects that are not blocks, to plan the varied-data
# campaign around them.

SURVEY_Z = 0.33
LONG_MOVE_STEPS = 900
LOOKS = [[0.50, 0.0], [0.50, -0.15], [0.45, -0.24], [0.50, 0.12], [0.57, -0.07], [0.45, 0.05]]
PROMPTS = ["blue block", "red block", "green block", "wooden block", "blue cylinder",
           "half round block", "can", "stuffed animal", "toy", "object"]
MERGE_M = 0.03


def survey(x, y):
    return move_to_pose({"position": [x, y, SURVEY_Z], "frame": "base",
                         "rpy_deg": [180.0, 0.0, 0.0]}, max_steps=LONG_MOVE_STEPS)


def has_depth(seg, depth):
    return int((depth[seg.mask] > 0.1).sum()) >= 50


go_home()
found = []
failed = []
for look in LOOKS:
    if not survey(look[0], look[1]).ok:
        failed.append(look)
        continue
    depth = get_observation().cameras["wrist"].depth_m
    for prompt in PROMPTS:
        seen = segment_text("wrist", prompt)
        if not seen.ok:
            continue
        for seg in seen.segmentations[:8]:
            if float(seg.score) < 0.4 or not has_depth(seg, depth):
                continue
            g = estimate_geometry(mask_to_point_cloud(seg, "wrist", "base"))
            p = [round(float(v), 3) for v in g.pose.position]
            e = [round(float(v), 3) for v in g.extents]
            same = None
            for f in found:
                if f["prompt"] == prompt and math.hypot(f["position"][0] - p[0], f["position"][1] - p[1]) < MERGE_M:
                    same = f
            if same is None:
                found.append({"prompt": prompt, "position": p, "extents": e,
                              "score": round(float(seg.score), 2), "looks": 1})
            else:
                same["looks"] += 1
                if float(seg.score) > same["score"]:
                    same.update({"position": p, "extents": e, "score": round(float(seg.score), 2)})
go_home()
result = {"ok": not failed, "note": "observation only", "found": found, "failed_looks": failed}
