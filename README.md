# video360 — Participant Behavior Analysis from 360° Classroom Video

A three-stage pipeline that takes a wide-angle (360°) classroom recording, isolates each participant, and produces per-frame behavioral features and an interactive timeline.

## Pipeline Overview

```
360° stitched video (1920×360)
        │
        ▼
00_autocrop_people       ← YOLO person detection → per-participant video crops
        │
        ▼
01_yolo_detect_track_hpc ← YOLO Pose + MediaPipe FaceMesh → feature CSVs
        │
        ▼
02_face_headpose_speaker_notetaking ← behavior inference + interactive visualization
```

## Notebooks

### `00_autocrop_people.ipynb`
Detects up to 4 participants in the stitched panoramic video and exports a cropped video clip for each.

- Samples frames with YOLOv8 to estimate a stable bounding box per participant slot
- Computes a fixed crop window per slot (aspect-ratio-corrected, configurable margins)
- Exports one `person_slotN_640x900.mp4` per active slot

Key config knobs: `MARGIN_X`, `MARGIN_TOP`, `MARGIN_BOTTOM`, `DOWN_SHIFT`, `REGION_X_BOUNDS`

### `01_yolo_detect_track_hpc.ipynb`
Runs YOLOv8 Pose and MediaPipe FaceMesh on each cropped video to extract per-frame signals.

**Features extracted per frame:**

| Feature | Description |
|---|---|
| `yaw_proxy` | Horizontal head turn (nose vs. ear/eye midpoint) |
| `pitch_proxy` | Vertical head tilt (nose vs. eye midpoint) |
| `smile_score` | Mouth width / face width ratio |
| `mouth_open_score` | Lip gap / face width ratio |
| `left/right_wrist_x/y` | Normalized wrist positions |
| `left/right_wrist_speed` | Wrist velocity between frames |
| `engagement_score` | Composite score [0, 1] |

Automatically adapts to 0, 1, or N GPUs using multiprocessing.

### `02_face_headpose_speaker_notetaking.ipynb`
Applies threshold-based behavior rules to feature CSVs and renders an interactive Plotly timeline.

**Behaviors detected:**

| Flag | Logic |
|---|---|
| `look_down_flag` | pitch > threshold |
| `head_nodding_flag` | rolling pitch range > threshold |
| `smiling_flag` | smile_score > 0.42 |
| `speaking_like_flag` | mouth_open_score > 0.08 |
| `take_notes_flag` | look_down AND hand near bottom AND hand moving |

All thresholds are grouped in a single `params` dict for easy tuning.

## Requirements

```
ultralytics>=8.4
mediapipe==0.10.13
opencv-python
pandas
numpy
torch
torchvision
plotly
```

The notebooks are designed for **Google Colab** with input/output stored in Google Drive under `MyDrive/CREATE Lab/video_agent360/`.

## Data Flow

```
Google Drive
└── CREATE Lab/video_agent360/
    ├── output_top.mp4              ← raw 360° input
    ├── autocrop_out/
    │   ├── person_slot0_640x900.mp4
    │   ├── person_slot2_640x900.mp4
    │   ├── person_slot3_640x900.mp4
    │   └── debug_crop_windows.jpg
    └── features_yolo01/
        ├── person_slot0_640x900_features.csv
        ├── person_slot2_640x900_features.csv
        └── person_slot3_640x900_features.csv
```
