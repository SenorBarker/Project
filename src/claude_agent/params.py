# Tunable numeric knobs for the Editor (multicam scoring/cutting logic).
# Optuna-tunable — do not hand-edit based on a single run's feel; adjust
# through the grade-and-suggest loop. Separate from CONSTITUTION.md, which
# governs the Producer's qualitative/rule-based decisions.

params = {
    "time_window": 15,  # single value, used two ways by design: (1) in
                         # sequence_scoring, the span over which shakiness/
                         # blur/framing are smoothed; (2) in editor(), the
                         # minimum hold time before reconsidering a cut. Same
                         # span both times deliberately -- a shot is scored
                         # over the exact length it would be held if chosen.

    # score_formula() weights, keyed to match the score name they weight --
    # e.g. weights["sharpness_ROI"] * sharpness_ROI, not a_sharpness_ROI * sharpness_ROI
    "weights": {
        # frame_total_score terms (per-frame quality)
        "sharpness_ROI":   1,
        "sharpness_image": 1,
        "framing":         1,
        "subject_size":    1,
        # sequence_total_score terms (camera-motion penalty)
        "whip":        1,
        "double_whip": 1,
        "wobble":      1,
        "steps":       1,
        "sway":        1,
    },
}
