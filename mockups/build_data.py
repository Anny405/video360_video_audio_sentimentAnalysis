"""Preprocess video + behavior CSVs into a single data.js for the HTML demo.

Reads:
  test_run/test_input.mp4                                 (1920x360 panorama)
  test_run/behavior_outputs/person_slot{0,2,3}_*.csv      (per-frame behavior flags)

Writes:
  mockups/data.js   (window.VIDEO360_DATA = {...})

Run:
  .venv_test/bin/python mockups/build_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "test_run" / "test_input.mp4"
BEHAV_DIR = ROOT / "test_run" / "behavior_outputs"
WEIGHTS = ROOT / "test_run" / "yolov8n.pt"
OUT_JS = ROOT / "mockups" / "data.js"

ACTIVE_SLOTS = [0, 1, 2, 3]
NAMES = {0: "Alex", 1: "Jordan", 2: "Maya", 3: "Sam"}
SAMPLE_EVERY_N = 4  # detect on every Nth frame


def assign_to_slot(xc: float, slot_bounds: list[tuple[int, int]]) -> int:
    for i, (a, b) in enumerate(slot_bounds):
        if a <= xc < b:
            return i
    mids = [(a + b) / 2 for a, b in slot_bounds]
    return int(np.argmin([abs(xc - m) for m in mids]))


def detect_bboxes_per_slot(video_path: Path, slot_bounds, sample_every_n=4):
    model = YOLO(str(WEIGHTS))
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video {W}x{H} {fps:.2f}fps {nframes} frames")

    tracks: dict[int, list[dict]] = {i: [] for i in ACTIVE_SLOTS}
    fidx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx % sample_every_n == 0:
            res = model.predict(frame, conf=0.25, iou=0.45, verbose=False)[0]
            t = fidx / fps
            slot_best: dict[int, tuple[float, list[float]]] = {}
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                conf = res.boxes.conf.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy()
                for box, c, k in zip(xyxy, conf, cls):
                    if int(k) != 0:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    if (x2 - x1) < 30 or (y2 - y1) < 60:
                        continue
                    xc = (x1 + x2) / 2.0
                    slot = assign_to_slot(xc, slot_bounds)
                    if slot not in ACTIVE_SLOTS:
                        continue
                    if slot not in slot_best or c > slot_best[slot][0]:
                        slot_best[slot] = (float(c), [x1, y1, x2, y2])
            for slot in ACTIVE_SLOTS:
                if slot in slot_best:
                    c, b = slot_best[slot]
                    tracks[slot].append({
                        "t": round(t, 3),
                        "x1": round(b[0] / W, 4),
                        "y1": round(b[1] / H, 4),
                        "x2": round(b[2] / W, 4),
                        "y2": round(b[3] / H, 4),
                        "conf": round(c, 3),
                    })
        fidx += 1
    cap.release()
    for s in ACTIVE_SLOTS:
        print(f"  slot {s}: {len(tracks[s])} bboxes")
    return tracks, fps, W, H, nframes


def load_behaviors():
    out: dict[int, list[dict]] = {}
    for s in ACTIVE_SLOTS:
        path = BEHAV_DIR / f"person_slot{s}_640x900_behaviors.csv"
        df = pd.read_csv(path)
        df = df.iloc[::SAMPLE_EVERY_N].reset_index(drop=True)
        rows = []
        for _, r in df.iterrows():
            rows.append({
                "t": round(float(r["t"]), 3),
                "yaw": None if pd.isna(r["yaw_proxy"]) else round(float(r["yaw_proxy"]), 3),
                "pitch": None if pd.isna(r["pitch_proxy"]) else round(float(r["pitch_proxy"]), 3),
                "smile": round(float(r["smile_score"]), 3) if not pd.isna(r["smile_score"]) else 0.0,
                "look_down": bool(r["look_down_flag"]),
                "look_up": bool(r["look_up_flag"]),
                "turn_left": bool(r["turn_left_flag"]),
                "turn_right": bool(r["turn_right_flag"]),
                "nodding": bool(r["head_nodding_flag"]),
                "smiling": bool(r["smiling_flag"]),
                "hand_moving": bool(r["hand_moving"]),
                "notes": bool(r["take_notes_flag"]),
            })
        out[s] = rows
    return out


def aggregate(behaviors, fps, duration):
    summary = {}
    for s, rows in behaviors.items():
        if not rows:
            continue
        n = len(rows)
        # each row represents SAMPLE_EVERY_N frames worth of time
        sec_per_row = SAMPLE_EVERY_N / fps
        notes_sec = sum(1 for r in rows if r["notes"]) * sec_per_row
        smile_sec = sum(1 for r in rows if r["smiling"]) * sec_per_row
        nodding_sec = sum(1 for r in rows if r["nodding"]) * sec_per_row
        turn_events = sum(1 for r in rows if r["turn_left"] or r["turn_right"])
        look_down_sec = sum(1 for r in rows if r["look_down"]) * sec_per_row
        avg_smile = round(float(np.mean([r["smile"] for r in rows])), 3)
        # rough engagement = forward-facing share: not look_down and not turn_left/right
        forward = sum(1 for r in rows if not (r["look_down"] or r["turn_left"] or r["turn_right"]))
        forward_pct = round(forward / n * 100, 1) if n else 0.0
        summary[s] = {
            "notes_sec": round(notes_sec, 1),
            "smile_sec": round(smile_sec, 1),
            "nodding_sec": round(nodding_sec, 1),
            "look_down_sec": round(look_down_sec, 1),
            "turn_events": turn_events,
            "avg_smile": avg_smile,
            "forward_pct": forward_pct,
        }
    return summary


def main():
    cap = cv2.VideoCapture(str(VIDEO))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    step = W / 4.0
    slot_bounds = [(int(i * step), int((i + 1) * step)) for i in range(4)]

    print("== detecting bboxes ==")
    tracks, fps, W, H, nframes = detect_bboxes_per_slot(VIDEO, slot_bounds, SAMPLE_EVERY_N)

    print("== loading behaviors ==")
    behaviors = load_behaviors()
    for s in ACTIVE_SLOTS:
        print(f"  slot {s}: {len(behaviors[s])} rows")

    duration = nframes / fps
    print("== aggregating ==")
    summary = aggregate(behaviors, fps, duration)
    for s, st in summary.items():
        print(f"  slot {s}: {st}")

    payload = {
        "video": {
            "src": "../test_run/test_input.mp4",
            "width": W,
            "height": H,
            "fps": round(fps, 3),
            "duration": round(duration, 3),
            "frames": nframes,
        },
        "participants": [
            {
                "slot": s,
                "name": NAMES[s],
                "color": {0: "#EC4899", 1: "#A78BFA", 2: "#60A5FA", 3: "#34D399"}[s],
                "summary": summary.get(s, {}),
            }
            for s in ACTIVE_SLOTS
        ],
        "tracks": {str(s): tracks[s] for s in ACTIVE_SLOTS},
        "behaviors": {str(s): behaviors[s] for s in ACTIVE_SLOTS},
        "sample_every_n": SAMPLE_EVERY_N,
    }

    OUT_JS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JS.open("w") as f:
        f.write("window.VIDEO360_DATA = ")
        json.dump(payload, f, separators=(",", ":"))
        f.write(";\n")
    size_kb = OUT_JS.stat().st_size / 1024
    print(f"\nwrote {OUT_JS} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
