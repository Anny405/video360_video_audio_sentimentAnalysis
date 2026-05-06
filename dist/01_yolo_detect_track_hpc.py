"""Extract per-frame pose + face biometrics from cropped participant videos.

Runs YOLOv8 Pose for body keypoints and MediaPipe FaceMesh for the smile
score. Outputs only raw biometric signals — behavior flags / engagement
are derived in step 02. Auto-adapts to 0/1/N GPUs.

Inputs : a directory of `person_slot*.mp4` videos (output of step 00).
Outputs: one `<participant>_features.csv` per input video.

CLI:
    python 01_yolo_detect_track_hpc.py \\
        --input-dir  /path/to/autocrop_out/ \\
        --output-dir /path/to/features_yolo01/ \\
        --model      yolov8n-pose.pt
"""

from __future__ import annotations

import argparse
import multiprocessing as mp_proc
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


# YOLO pose keypoint indices (COCO 17-point skeleton)
KP_NOSE = 0
KP_L_EYE, KP_R_EYE = 1, 2
KP_L_EAR, KP_R_EAR = 3, 4
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_ELBOW, KP_R_ELBOW = 7, 8
KP_L_WRIST, KP_R_WRIST = 9, 10

# MediaPipe FaceMesh landmark indices used for expression features
MP_LEFT_MOUTH = 61
MP_RIGHT_MOUTH = 291
MP_LEFT_CHEEK = 234
MP_RIGHT_CHEEK = 454

FACE_PAD = 0.25  # padding around body bbox when cropping face region


def detect_devices() -> tuple[list[str], int]:
    """Return (device_list, n_workers) based on available hardware."""
    n = torch.cuda.device_count()
    if n == 0:
        return ["cpu"], 1
    if n == 1:
        return ["cuda:0"], 1
    return [f"cuda:{i}" for i in range(n)], n


def extract_face_features(face_mesh, frame: np.ndarray, bbox_xyxy: np.ndarray) -> dict:
    """Crop the face region using a body bbox, run FaceMesh, return smile_score.

    Returns NaN when the face cannot be located.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy.astype(int)

    pad_x = int((x2 - x1) * FACE_PAD)
    pad_y = int((y2 - y1) * FACE_PAD)
    crop = frame[max(0, y1 - pad_y):min(h, y2 + pad_y),
                 max(0, x1 - pad_x):min(w, x2 + pad_x)]

    if crop.size == 0:
        return {"smile_score": np.nan}

    result = face_mesh.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    if not result.multi_face_landmarks:
        return {"smile_score": np.nan}

    lm = result.multi_face_landmarks[0].landmark
    ch, cw = crop.shape[:2]

    def pt(idx: int) -> np.ndarray:
        return np.array([lm[idx].x * cw, lm[idx].y * ch])

    mouth_width = np.linalg.norm(pt(MP_RIGHT_MOUTH) - pt(MP_LEFT_MOUTH))
    face_width = np.linalg.norm(pt(MP_RIGHT_CHEEK) - pt(MP_LEFT_CHEEK))

    if face_width == 0:
        return {"smile_score": np.nan}

    return {"smile_score": float(mouth_width / face_width)}


def process_single_video(
    video_path: Path,
    output_dir: Path,
    device: str,
    model: YOLO,
    face_mesh,
    batch_size: int = 8,
) -> str:
    """Process one cropped video and write raw biometric features CSV.

    No flag derivation here — that's step 02's job.
    """
    participant_id = video_path.stem
    output_csv = output_dir / f"{participant_id}_features.csv"

    if output_csv.exists():
        return f"skip (done): {participant_id}"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return f"error: cannot open {video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Cap the dt used for wrist-speed so gaps in detection don't produce
    # spuriously low/wrong velocities.
    MAX_SPEED_DT = 0.2

    records: list[dict] = []
    batch_frames: list[np.ndarray] = []
    batch_raw: list[np.ndarray] = []
    batch_idx: list[int] = []
    frame_idx = 0
    prev_lw = prev_rw = prev_t = None

    def flush_batch(frames, raw_frames, indices):
        nonlocal prev_lw, prev_rw, prev_t
        results_list = model(frames, device=device, verbose=False, conf=0.5, half=False)
        batch_records: list[dict] = []

        for res, raw_frame, fidx in zip(results_list, raw_frames, indices):
            t = fidx / fps
            row = {
                "frame_idx": fidx, "t": float(t),
                "face_detected": 0, "pose_detected": 0,
                "yaw_proxy": np.nan, "pitch_proxy": np.nan,
                "smile_score": np.nan,
                "left_wrist_x": np.nan, "left_wrist_y": np.nan,
                "right_wrist_x": np.nan, "right_wrist_y": np.nan,
                "left_wrist_vis": np.nan, "right_wrist_vis": np.nan,
                "left_elbow_y": np.nan, "right_elbow_y": np.nan,
                "left_shoulder_y": np.nan, "right_shoulder_y": np.nan,
                "left_wrist_speed": np.nan, "right_wrist_speed": np.nan,
            }

            if res.keypoints is None or len(res.keypoints) == 0:
                batch_records.append(row)
                continue

            row["pose_detected"] = 1

            kpts = res.keypoints.xyn[0].cpu().numpy()
            conf = res.keypoints.conf[0].cpu().numpy()

            row["left_wrist_x"], row["left_wrist_y"] = kpts[KP_L_WRIST]
            row["right_wrist_x"], row["right_wrist_y"] = kpts[KP_R_WRIST]
            row["left_wrist_vis"] = float(conf[KP_L_WRIST])
            row["right_wrist_vis"] = float(conf[KP_R_WRIST])
            row["left_elbow_y"] = float(kpts[KP_L_ELBOW][1])
            row["right_elbow_y"] = float(kpts[KP_R_ELBOW][1])
            row["left_shoulder_y"] = float(kpts[KP_L_SHOULDER][1])
            row["right_shoulder_y"] = float(kpts[KP_R_SHOULDER][1])

            if conf[KP_L_EAR] > 0.3 and conf[KP_R_EAR] > 0.3:
                ear_cx = (kpts[KP_L_EAR][0] + kpts[KP_R_EAR][0]) / 2.0
                row["yaw_proxy"] = float(kpts[KP_NOSE][0] - ear_cx)
            else:
                eye_cx = (kpts[KP_L_EYE][0] + kpts[KP_R_EYE][0]) / 2.0
                row["yaw_proxy"] = float(kpts[KP_NOSE][0] - eye_cx) * 2.0

            eye_y = (kpts[KP_L_EYE][1] + kpts[KP_R_EYE][1]) / 2.0
            row["pitch_proxy"] = float(kpts[KP_NOSE][1] - eye_y)

            if prev_t is not None and 0 < (dt := t - prev_t) <= MAX_SPEED_DT:
                if prev_lw is not None:
                    row["left_wrist_speed"] = float(np.linalg.norm(kpts[KP_L_WRIST] - prev_lw) / dt)
                if prev_rw is not None:
                    row["right_wrist_speed"] = float(np.linalg.norm(kpts[KP_R_WRIST] - prev_rw) / dt)

            prev_lw = kpts[KP_L_WRIST].copy()
            prev_rw = kpts[KP_R_WRIST].copy()
            prev_t = t

            if res.boxes is not None and len(res.boxes) > 0:
                bbox = res.boxes.xyxy[0].cpu().numpy()
                bx1, by1, bx2, by2 = bbox
                face_h = (by2 - by1) * 0.30
                face_feats = extract_face_features(
                    face_mesh, raw_frame,
                    np.array([bx1, by1, bx2, by1 + face_h]),
                )
                row.update(face_feats)
                if not np.isnan(face_feats.get("smile_score", np.nan)):
                    row["face_detected"] = 1

            batch_records.append(row)
        return batch_records

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        batch_raw.append(frame.copy())
        batch_frames.append(frame)
        batch_idx.append(frame_idx)
        frame_idx += 1

        if len(batch_frames) >= batch_size:
            records.extend(flush_batch(batch_frames, batch_raw, batch_idx))
            batch_frames, batch_raw, batch_idx = [], [], []

    if batch_frames:
        records.extend(flush_batch(batch_frames, batch_raw, batch_idx))

    cap.release()
    df = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return f"done [{device}]: {participant_id}  ({len(df)} frames)"


def worker(
    video_paths: list[str],
    output_dir: str,
    model_path: str,
    device: str,
    batch_size: int,
) -> None:
    """One process owns one device. Loads its own model and FaceMesh once."""
    output_dir = Path(output_dir)
    model = YOLO(model_path)

    # static_image_mode=False enables frame-to-frame tracking — better for video
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    for vp in video_paths:
        print(process_single_video(Path(vp), output_dir, device, model, face_mesh,
                                   batch_size=batch_size), flush=True)

    face_mesh.close()


def split_workload(video_paths: list[str], n: int) -> list[list[str]]:
    return [video_paths[i::n] for i in range(n)]


def run(
    input_dir: str | Path,
    output_dir: str | Path,
    model_path: str = "yolov8n-pose.pt",
    batch_size: int = 8,
) -> None:
    """Process all `person_slot*.mp4` files under input_dir."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(str(p) for p in input_dir.glob("person_slot*.mp4"))
    devices, n_workers = detect_devices()

    print(f"{len(video_files)} videos | {n_workers} worker(s) | devices: {devices}")

    if not video_files:
        print("no input videos found")
        return

    if n_workers == 1:
        worker(video_files, str(output_dir), model_path, devices[0], batch_size)
    else:
        # CUDA needs the spawn start method when forking workers
        try:
            mp_proc.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        shards = split_workload(video_files, n_workers)
        procs = []
        for device, shard in zip(devices, shards):
            p = mp_proc.Process(
                target=worker,
                args=(shard, str(output_dir), model_path, device, batch_size),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

    print("\nall done")


def main() -> None:
    p = argparse.ArgumentParser(description="Extract pose + face features from cropped videos")
    p.add_argument("--input-dir",  required=True, help="Directory containing person_slot*.mp4")
    p.add_argument("--output-dir", required=True, help="Where to write *_features.csv")
    p.add_argument("--model", default="yolov8n-pose.pt", help="YOLO pose model file")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    run(args.input_dir, args.output_dir, model_path=args.model, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
