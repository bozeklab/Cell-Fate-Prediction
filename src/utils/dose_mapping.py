DOSES = {
    "High": list(range(1, 47)),
    "Unknown": list(range(47, 130)) + list(range(202, 274)),
    "Medium": list(range(130, 166)),
    "Low": list(range(166, 202)),
    "Control": list(range(274, 310)),
}
DOSE_LOOKUP = {video_id: dose for dose, ids in DOSES.items() for video_id in ids}
