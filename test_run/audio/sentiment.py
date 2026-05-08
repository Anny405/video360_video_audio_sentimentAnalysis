"""sentiment.py — score each transcript utterance on Russell's valence/arousal plane.

Pipeline placement:
    transcribe.py --merge  →  transcripts/merged.json   (start, end, speaker, text)
    sentiment.py           →  transcripts/merged_sentiment.json
                              (… + sentiment: {v, a, label_4, label_12, top_emotion, conf})

Method:
    1. Run j-hartmann/emotion-english-distilroberta-base on each utterance.
       Returns probabilities over 7 classes: anger, disgust, fear, joy, neutral,
       sadness, surprise.
    2. Each emotion has a fixed (V, A) anchor from Russell 1980 / NRC-VAD.
       Compute weighted centroid:  v = Σ p_i · v_i,  a = Σ p_i · a_i.
    3. Project (v, a) onto:
       - 4 quadrants: Q1 joyful / Q2 angry / Q3 depressed / Q4 content
       - 12 fine labels: alarmed/annoyed/frustrated/depressed/miserable/bored/
                          tired/calm/content/satisfied/pleased/delighted/glad

Usage:
    .venv_test/bin/python test_run/audio/sentiment.py
    # optional: --input <merged.json> --output <merged_sentiment.json>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# ---- emotion → (valence, arousal) anchors --------------------------------
# Values from Russell (1980) + Mehrabian/Russell affect grids; valence and arousal in [-1, 1].
EMOTION_ANCHORS = {
    "joy":      {"v": +0.81, "a": +0.51},
    "surprise": {"v": +0.40, "a": +0.67},
    "anger":    {"v": -0.51, "a": +0.59},
    "fear":     {"v": -0.64, "a": +0.60},
    "disgust":  {"v": -0.60, "a": +0.35},
    "sadness":  {"v": -0.63, "a": -0.27},
    "neutral":  {"v":  0.00, "a":  0.00},
}

# 13 fine-grained circumplex points placed at canonical (v, a) positions on the unit circle,
# each tagged with its 4-quadrant home (matches the user's reference figure: angry/joyful/
# depressed/content corners). label_4 is derived from label_12 to stay consistent: a sub-zero-
# threshold (v,a) won't flip label_12=content but stay on Q1/Q2 because of strict sign rules.
FINE_LABELS = [
    # (label,        v,     a,    quadrant)
    ("delighted",  +0.71, +0.71, "Q1"),   # NE
    ("glad",       +0.92, +0.38, "Q1"),   # ENE
    ("pleased",    +0.92, -0.38, "Q4"),   # ESE
    ("satisfied",  +0.71, -0.71, "Q4"),   # SE
    ("calm",       +0.38, -0.92, "Q4"),   # SSE
    ("content",    +0.00, -0.00, "Q4"),   # origin → reads as "neutral / content"
    ("miserable",  -0.38, -0.92, "Q3"),   # SSW
    ("bored",      -0.71, -0.71, "Q3"),   # SW
    ("tired",      -0.92, -0.38, "Q3"),   # WSW
    ("depressed",  -0.92, +0.00, "Q3"),   # W   (corner label for Q3 in the figure)
    ("frustrated", -0.92, +0.38, "Q2"),   # WNW
    ("annoyed",    -0.71, +0.71, "Q2"),   # NW
    ("alarmed",    -0.38, +0.92, "Q2"),   # NNW
]
FINE_TO_QUAD = {lab: q for lab, _, _, q in FINE_LABELS}


def nearest_label_12(v: float, a: float) -> tuple[str, str]:
    """Return (label_12, label_4) for the nearest fine-grained anchor.
    Snapping label_4 from label_12 guarantees consistency near the axes."""
    best, best_d = None, math.inf
    for name, lv, la, _ in FINE_LABELS:
        d = (v - lv) ** 2 + (a - la) ** 2
        if d < best_d:
            best, best_d = name, d
    return best, FINE_TO_QUAD[best]


def aggregate_va(probs: dict[str, float]) -> tuple[float, float]:
    v = sum(probs[e] * EMOTION_ANCHORS[e]["v"] for e in probs)
    a = sum(probs[e] * EMOTION_ANCHORS[e]["a"] for e in probs)
    return float(v), float(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--input",  default=str(here / "transcripts" / "merged.json"))
    ap.add_argument("--output", default=str(here / "transcripts" / "merged_sentiment.json"))
    ap.add_argument("--model",  default="j-hartmann/emotion-english-distilroberta-base")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise SystemExit(f"input not found: {in_path}")

    with in_path.open(encoding="utf-8") as f:
        utterances = json.load(f)
    if not args.quiet:
        print(f"loaded {len(utterances)} utterances from {in_path.name}")

    # Lazy import so help text is fast.
    from transformers import pipeline

    if not args.quiet:
        print(f"loading {args.model} (first run downloads ~330MB)...")
    clf = pipeline(
        "text-classification",
        model=args.model,
        top_k=None,        # return all 7 class scores
        truncation=True,
        device="cpu",
    )

    quad_counts = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    label_counts: dict[str, int] = {}
    out = []
    for i, u in enumerate(utterances):
        text = (u.get("text") or "").strip()
        if not text:
            # preserve segment but no sentiment
            out.append({**u, "sentiment": None})
            continue

        scores = clf(text)[0]                                      # list of {label, score}
        probs = {s["label"]: float(s["score"]) for s in scores}    # 7 keys
        # sanity: ensure all 7 keys present (model uses lowercase labels already)
        for k in EMOTION_ANCHORS:
            probs.setdefault(k, 0.0)

        v, a = aggregate_va(probs)
        lab12, q = nearest_label_12(v, a)
        top = max(probs, key=probs.get)

        out.append({
            **u,
            "sentiment": {
                "v": round(v, 4),
                "a": round(a, 4),
                "label_4": q,
                "label_12": lab12,
                "top_emotion": top,
                "conf": round(probs[top], 4),
            },
        })
        quad_counts[q] += 1
        label_counts[lab12] = label_counts.get(lab12, 0) + 1

        if not args.quiet and (i + 1) % 10 == 0:
            print(f"  scored {i+1}/{len(utterances)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    if not args.quiet:
        print(f"\nwrote {out_path}  ({out_path.stat().st_size/1024:.1f} KB)")
        print("\nQuadrant distribution:")
        total = sum(quad_counts.values()) or 1
        for q in ("Q1", "Q2", "Q3", "Q4"):
            n = quad_counts[q]
            label = {"Q1": "joyful", "Q2": "angry", "Q3": "depressed", "Q4": "content"}[q]
            print(f"  {q} {label:<10} {n:>3}  ({100*n/total:5.1f}%)")
        print("\nFine-label distribution:")
        for lab, n in sorted(label_counts.items(), key=lambda x: -x[1]):
            print(f"  {lab:<11} {n}")


if __name__ == "__main__":
    main()
