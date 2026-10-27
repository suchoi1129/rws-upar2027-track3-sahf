"""Shared constants. Slot order is fixed by the challenge (PoseTrack18 layout)."""
import os
import numpy as np

NAMES = ["nose", "head_bottom", "head_top", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
         "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
         "left_knee", "right_knee", "left_ankle", "right_ankle"]
NK = 17
# Slots 1,2 are head_bottom/head_top (NOT eyes) -> never swapped. Ears 3,4 are unlabeled.
FLIP_PAIRS = [(5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
FLIP_INDEX = list(range(NK))
for a, b in FLIP_PAIRS:
    FLIP_INDEX[a], FLIP_INDEX[b] = b, a
EVAL_SLOTS = [0, 1, 2] + list(range(5, 17))
# OKS sigmas: COCO values applied by slot order (official scoring program not available).
SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
DATA = os.environ.get("UPAR_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))  # used by local tools only
DOMAINS = ["market", "pa100k", "peta"]
