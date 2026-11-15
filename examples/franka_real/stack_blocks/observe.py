# Observation only: where does each coloured block read from the current pose?
seen = {}
for colour in ("red", "green", "blue"):
    found = localize_object(colour + " block")
    if found.ok:
        p = [round(float(v), 4) for v in found.geometry.pose.position]
        e = [round(float(v), 4) for v in found.geometry.extents]
        seen[colour] = {"position": p, "extents": e}
    else:
        seen[colour] = None
result = {"ok": True, "note": "observation only", "seen": seen}
