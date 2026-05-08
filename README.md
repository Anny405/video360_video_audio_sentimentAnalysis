# video360 — Multimodal Meeting Analysis

A two-track pipeline + web UI that turns a 360° meeting recording plus 4 lapel-mic recordings into one synchronized view: per-participant cropped video, head-pose / gaze / note-taking behavior flags, dominant-channel speaker diarization, Whisper transcription with bleed-suppressed attribution, and per-utterance sentiment plotted on Russell's valence/arousal circumplex.

> **Privacy note:** This repo is **code only**. Real meeting recordings are excluded via `.gitignore`. To see the full UI, run the pipelines on your own data — see [Run locally](#run-locally) below.

🔗 **Live (code-only) page:** the GitHub Pages deployment of this repo serves [`index.html`](./index.html) at the root and [`mockups/app.html`](./mockups/app.html) for the unified UI shell. The shell needs locally-built `data.js` + `audio_data.js` to populate.

---

## Pipeline overview

```
360° panorama mp4 ───┐                                ┌─── 4 lapel-mic wavs
                     ▼                                ▼
   dist/00_autocrop_people.py            test_run/audio/decrosstalk.py
   dist/01_yolo_detect_track_hpc.py      test_run/audio/merge_dominant.py
   dist/02_face_headpose_…py             test_run/audio/transcribe.py  (mlx-whisper)
                     │                                │
                     │                  test_run/audio/sentiment.py    (DistilRoBERTa→V/A)
                     │                                │
        mockups/build_data.py             mockups/build_audio_data.py
                     │                                │
                     └────────► mockups/app.html ◄────┘
                                (panorama + 4 cards + timeline + circumplex)
```

## Repo layout

```
.
├── dist/                       # video pipeline (CLI scripts)
│   ├── 00_autocrop_people.py
│   ├── 01_yolo_detect_track_hpc.py
│   ├── 02_face_headpose_speaker_notetaking.py
│   └── requirements.txt
├── test_run/audio/             # audio pipeline (CLI scripts)
│   ├── decrosstalk.py          # FIR mic-bleed cancellation
│   ├── merge_dominant.py       # loudest-channel merge → mono + speaker labels
│   ├── transcribe.py           # mlx-whisper per channel + merged dialogue
│   └── sentiment.py            # DistilRoBERTa-emotion → V/A → 4 quadrants
├── mockups/                    # web UI
│   ├── app.html                # unified panorama + 4 cards + circumplex
│   ├── demo.html               # video-only drill-down
│   ├── demo_audio.html         # audio-only drill-down
│   ├── index.html              # original static design mockup
│   ├── build_data.py           # video pipeline outputs → data.js
│   ├── build_audio_data.py     # audio pipeline outputs → audio_data.js
│   └── serve.py                # tiny Range-aware static server (browser-friendly mp4 streaming)
├── *.ipynb                     # original notebooks (cleared outputs)
├── index.html                  # root landing page (code-only deployment)
└── README.md
```

## Run locally

```bash
# 1. clone + venv
git clone https://github.com/Anny405/video360_video_audio_sentimentAnalysis.git
cd video360_video_audio_sentimentAnalysis
python3 -m venv .venv && source .venv/bin/activate
pip install -r dist/requirements.txt
pip install mlx-whisper transformers soundfile

# 2. drop in your own recordings
#    test_run/test_input.mp4                     ← 360° panorama
#    test_run/audio/<your>-ch1.wav … ch4.wav     ← 4 lapel mics

# 3. video chain
python dist/00_autocrop_people.py --input test_run/test_input.mp4 --out test_run/autocrop_out
python dist/01_yolo_detect_track_hpc.py --input-dir test_run/autocrop_out --output-dir test_run/features_yolo01
python dist/02_face_headpose_speaker_notetaking.py --input-dir test_run/features_yolo01 --output-dir test_run/behavior_outputs --no-html

# 4. audio chain (run from inside test_run/audio)
cd test_run/audio
python decrosstalk.py *.wav
python merge_dominant.py *_clean.wav -o merged_demo --names S1 S2 S3
python transcribe.py *_clean.wav --merge -o transcripts
python sentiment.py
cd ../..

# 5. build demo data + serve
python mockups/build_data.py
python mockups/build_audio_data.py
python mockups/serve.py 8765
# open http://127.0.0.1:8765/mockups/app.html
```

## What's in `mockups/app.html`

A single-page interactive view that:

- Plays the 360° panorama at the top with **YOLO bbox overlays** (live position + confidence)
- Renders **4 participant cards**, each with: cropped per-person video (synced to master), live yaw/pitch/smile readouts, mini speaker timeline, deduped Whisper transcript with auto-scroll
- Below: **multi-lane speaker timeline** (combined + per-speaker) clickable to seek
- **Sentiment circumplex** SVG plot: 46 utterance dots + 3 speaker centroids on Russell's V/A plane, with quadrant tints, fine labels, hover detail, click-to-seek, and a sigmoid-stretch toggle for low-emotion meetings
- Master clock = panorama; cropped videos and merged audio follow with 0.18s tolerance

## Sentiment method (sentiment.py)

7-class DistilRoBERTa output (`anger / disgust / fear / joy / neutral / sadness / surprise`) → weighted Russell anchors → `(V, A)` centroid → nearest of 13 fine labels (`delighted, glad, pleased, satisfied, calm, content, miserable, bored, tired, depressed, frustrated, annoyed, alarmed`) → 4 quadrants (`Q1 joyful · Q2 angry · Q3 depressed · Q4 content`).

## License

MIT. Recordings excluded for privacy.
