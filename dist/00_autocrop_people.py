"""Auto-crop each participant from a stitched 360 classroom video.

Pipeline:
  1. Sample frames, run YOLO person detection per slot, collect bboxes.
  2. Compute a median bbox per slot -> one fixed crop window.
  3. Re-read the video and write one cropped MP4 per active slot.
  4. Save a debug image showing the crop windows on the first frame.

CLI:
    python 00_autocrop_people.py \\
        --input /path/to/output_top.mp4 \\
        --out   /path/to/autocrop_out/

Programmatic:
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("autocrop", "dist/00_autocrop_people.py")
    mod  = module_from_spec(spec); spec.loader.exec_module(mod)
    mod.autocrop("video.mp4", "out/")
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from ultralytics import YOLO


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def compute_crop_window_from_bbox(
    bbox_xyxy: Sequence[float],
    frame_w: int,
    frame_h: int,
    out_w: int = 640,
    out_h: int = 900,
    margin_x: float = 0.55,
    margin_top: float = 0.25,
    margin_bottom: float = 1.10,
    down_shift: float = 0.20,
) -> list[int]:
    x1, y1, x2, y2 = bbox_xyxy
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0 + down_shift * bh

    ex1 = x1 - margin_x * bw
    ex2 = x2 + margin_x * bw
    ey1 = y1 - margin_top * bh
    ey2 = y2 + margin_bottom * bh

    target_ar = out_w / out_h
    w = ex2 - ex1
    h = ey2 - ey1

    cur_ar = w / h if h > 1e-6 else target_ar
    if cur_ar > target_ar:
        h = w / target_ar
    else:
        w = h * target_ar

    crop_x1 = clamp(cx - w / 2.0, 0, frame_w - 2)
    crop_y1 = clamp(cy - h / 2.0, 0, frame_h - 2)
    crop_x2 = clamp(cx + w / 2.0, crop_x1 + 1, frame_w - 1)
    crop_y2 = clamp(cy + h / 2.0, crop_y1 + 1, frame_h - 1)

    return [int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)]


def assign_to_slot(x_center: float, bounds: Sequence[tuple[int, int]]) -> int:
    for i, (a, b) in enumerate(bounds):
        if a <= x_center < b:
            return i
    mids = [(a + b) / 2 for a, b in bounds]
    return int(np.argmin([abs(x_center - m) for m in mids]))


def median_bbox(bboxes: list[list[float]]) -> list[float]:
    arr = np.array(bboxes, dtype=np.float32)
    return np.median(arr, axis=0).tolist()


def autocrop(
    input_video_path: str | Path,
    out_dir: str | Path,
    *,
    out_w: int = 640,
    out_h: int = 900,
    top_h: int = 0,
    low_y0: int = 0,
    low_h: int | None = None,
    region_x_bounds: list[tuple[int, int]] | None = None,
    sample_every_n_frames: int = 10,
    max_sample_frames: int = 300,
    yolo_model: str = "yolov8n.pt",
    yolo_conf: float = 0.25,
    yolo_iou: float = 0.45,
    min_box_w: int = 40,
    min_box_h: int = 60,
    margin_x: float = 0.55,
    margin_top: float = 0.25,
    margin_bottom: float = 1.10,
    down_shift: float = 0.20,
) -> dict[int, str]:
    """Run the full autocrop pipeline. Returns {slot_idx: output_video_path}."""
    input_video_path = str(input_video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Pass 1: sample frames -> per-slot bboxes ---
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = in_fps if in_fps > 0 else 30.0

    if low_h is None:
        low_h = H - top_h
    if low_h <= 0:
        raise RuntimeError(f"Invalid low_h={low_h}. Check top_h/low_y0/low_h.")

    if region_x_bounds is None:
        step = W / 4.0
        region_x_bounds = [(int(i * step), int((i + 1) * step)) for i in range(4)]

    num_slots = len(region_x_bounds)
    print(f"Frame size: {W}x{H} | top_h={top_h} low_y0={low_y0} low_h={low_h}")
    print(f"Slot bounds: {region_x_bounds}")

    model = YOLO(yolo_model)
    slot_bboxes: list[list[list[float]]] = [[] for _ in range(num_slots)]

    sampled = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % sample_every_n_frames != 0:
            frame_idx += 1
            continue

        roi = frame[low_y0 : low_y0 + low_h, :, :]
        if roi.size == 0:
            raise RuntimeError(f"Empty ROI. H={H} low_y0={low_y0} low_h={low_h}")

        # ultralytics expects BGR for numpy input
        results = model.predict(roi, conf=yolo_conf, iou=yolo_iou, verbose=False)
        det = results[0]

        if det.boxes is not None and len(det.boxes) > 0:
            for b, c in zip(det.boxes.xyxy.cpu().numpy(), det.boxes.cls.cpu().numpy()):
                if int(c) != 0:  # 0 = person
                    continue
                x1, y1, x2, y2 = b.tolist()
                x1f, y1f, x2f, y2f = x1, y1 + low_y0, x2, y2 + low_y0
                if (x2f - x1f) < min_box_w or (y2f - y1f) < min_box_h:
                    continue
                xc = (x1f + x2f) / 2.0
                slot_bboxes[assign_to_slot(xc, region_x_bounds)].append(
                    [x1f, y1f, x2f, y2f]
                )

        sampled += 1
        frame_idx += 1
        if sampled >= max_sample_frames:
            break
    cap.release()

    print("\nCollected bbox counts:")
    for i in range(num_slots):
        print(f"  slot {i}: {len(slot_bboxes[i])}")

    # --- Compute median bbox per slot ---
    median_bboxes: list[list[float] | None] = [None] * num_slots
    active_slots: list[int] = []
    for i in range(num_slots):
        if not slot_bboxes[i]:
            print(f"Warning: slot {i} empty, skipping.")
        else:
            median_bboxes[i] = median_bbox(slot_bboxes[i])
            active_slots.append(i)

    if not active_slots:
        raise RuntimeError(
            "No participant detections found. "
            "Try lowering yolo_conf, increasing max_sample_frames, or check layout."
        )

    # --- Build crop windows ---
    crop_windows: dict[int, list[int]] = {}
    for i in active_slots:
        crop_windows[i] = compute_crop_window_from_bbox(
            median_bboxes[i], W, H,
            out_w=out_w, out_h=out_h,
            margin_x=margin_x, margin_top=margin_top,
            margin_bottom=margin_bottom, down_shift=down_shift,
        )

    print("\nCrop windows:")
    for i in active_slots:
        cw = crop_windows[i]
        print(f"  slot {i}: {cw}  size={(cw[2]-cw[0], cw[3]-cw[1])}")

    # --- Pass 2: crop & export ---
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers: dict[int, cv2.VideoWriter] = {}
    out_paths: dict[int, str] = {}
    for i in active_slots:
        op = str(out_dir / f"person_slot{i}_{out_w}x{out_h}.mp4")
        w = cv2.VideoWriter(op, fourcc, fps, (out_w, out_h))
        if not w.isOpened():
            raise RuntimeError(f"Cannot open writer: {op}")
        writers[i] = w
        out_paths[i] = op

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for i in active_slots:
            x1, y1, x2, y2 = crop_windows[i]
            crop = frame[y1:y2, x1:x2, :]
            if crop.size == 0:
                print(f"Warning: empty crop on frame {frame_idx}, slot {i}")
                continue
            crop_resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            writers[i].write(crop_resized)
        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"  exported {frame_idx} frames")

    cap.release()
    for w in writers.values():
        w.release()

    print("\nDone. Outputs:")
    for i in active_slots:
        print(f"  slot {i}: {out_paths[i]}")

    # --- Sanity check debug frame ---
    cap = cv2.VideoCapture(input_video_path)
    ok, frame = cap.read()
    cap.release()
    if ok:
        dbg = frame.copy()
        for i, (sx0, sx1) in enumerate(region_x_bounds):
            cv2.rectangle(dbg, (sx0, 5), (sx1, 40), (255, 255, 0), 2)
            cv2.putText(dbg, f"slot {i}", (sx0 + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        for i in active_slots:
            x1, y1, x2, y2 = crop_windows[i]
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(dbg, f"P_slot{i}", (x1 + 10, y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        dbg_path = str(out_dir / "debug_crop_windows.jpg")
        cv2.imwrite(dbg_path, dbg)
        print(f"Saved debug image: {dbg_path}")

    return out_paths


def main() -> None:
    p = argparse.ArgumentParser(description="Auto-crop participants from a 360 video")
    p.add_argument("--input", required=True, help="Path to the stitched 360 video")
    p.add_argument("--out",   required=True, help="Output directory for cropped clips")
    p.add_argument("--out-w", type=int, default=640)
    p.add_argument("--out-h", type=int, default=900)
    p.add_argument("--top-h", type=int, default=0)
    p.add_argument("--low-y0", type=int, default=0)
    p.add_argument("--low-h", type=int, default=None)
    p.add_argument("--sample-every", type=int, default=10)
    p.add_argument("--max-samples", type=int, default=300)
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou",  type=float, default=0.45)
    args = p.parse_args()

    autocrop(
        args.input, args.out,
        out_w=args.out_w, out_h=args.out_h,
        top_h=args.top_h, low_y0=args.low_y0, low_h=args.low_h,
        sample_every_n_frames=args.sample_every,
        max_sample_frames=args.max_samples,
        yolo_model=args.model,
        yolo_conf=args.conf, yolo_iou=args.iou,
    )


if __name__ == "__main__":
    main()
