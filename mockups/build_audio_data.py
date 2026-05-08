"""Preprocess audio diarization (+ optional whisper transcript) into audio_data.js.

Reads:
  test_run/audio/merged_demo.json          diarization: [{start, end, speaker, channel}]
  test_run/audio/merged_demo.wav           audio for <audio> element + metadata
  test_run/audio/transcripts/merged.json   whisper output (optional): [{speaker, start, end, text}]
                                           NOTE: whisper transcribes each channel independently,
                                           so a single utterance often appears under multiple
                                           speakers due to mic bleed. We dedupe by checking
                                           who was the dominant speaker at the segment's mid-time
                                           (per merged_demo.json) and only keeping segments
                                           whose claimed speaker matches.

Writes:
  mockups/audio_data.js   (window.VIDEO360_AUDIO = {...})

Run:
  .venv_test/bin/python mockups/build_audio_data.py
"""

from __future__ import annotations

import json
import wave
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "test_run" / "audio"
SEG_JSON = AUDIO_DIR / "merged_demo.json"
WAV_PATH = AUDIO_DIR / "merged_demo.wav"
TRANSCRIPT_JSON = AUDIO_DIR / "transcripts" / "merged.json"
SENTIMENT_JSON = AUDIO_DIR / "transcripts" / "merged_sentiment.json"
OUT_JS = ROOT / "mockups" / "audio_data.js"

# stable color per speaker (matches video demo palette)
SPEAKER_COLORS = {
    "S1": "#EC4899",  # pink
    "S2": "#60A5FA",  # blue
    "S3": "#34D399",  # emerald
    "S4": "#A78BFA",  # violet
}

MERGE_GAP = 0.3  # merge adjacent same-speaker segments closer than this


def wav_meta(path: Path) -> dict:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        nframes = w.getnframes()
        ch = w.getnchannels()
        sw = w.getsampwidth()
    return {
        "sample_rate": sr,
        "channels": ch,
        "sample_width_bytes": sw,
        "duration": round(nframes / sr, 3),
        "frames": nframes,
    }


def merge_adjacent(segments, gap=MERGE_GAP):
    """Collapse consecutive same-speaker segments separated by < gap seconds."""
    if not segments:
        return []
    out = [dict(segments[0])]
    for s in segments[1:]:
        prev = out[-1]
        if s["speaker"] == prev["speaker"] and (s["start"] - prev["end"]) < gap:
            prev["end"] = s["end"]
        else:
            out.append(dict(s))
    return out


def aggregate_speakers(segments, total_dur):
    stats = defaultdict(lambda: {"total_sec": 0.0, "segment_count": 0, "channels": set()})
    for s in segments:
        spk = s["speaker"]
        dur = s["end"] - s["start"]
        stats[spk]["total_sec"] += dur
        stats[spk]["segment_count"] += 1
        if s.get("channel") is not None:
            stats[spk]["channels"].add(int(s["channel"]))

    speakers = []
    for spk, st in stats.items():
        if spk == "silence":
            continue
        speakers.append({
            "id": spk,
            "name": f"Speaker {spk[1:]}",
            "color": SPEAKER_COLORS.get(spk, "#9CA3AF"),
            "channels": sorted(st["channels"]),
            "total_sec": round(st["total_sec"], 2),
            "segment_count": st["segment_count"],
            "avg_segment_sec": round(st["total_sec"] / st["segment_count"], 2) if st["segment_count"] else 0,
            "share_pct": round(st["total_sec"] / total_dur * 100, 1),
        })
    speakers.sort(key=lambda x: -x["total_sec"])

    silence_sec = stats.get("silence", {}).get("total_sec", 0.0)
    speech_sec = sum(s["total_sec"] for s in speakers)
    return speakers, {
        "silence_sec": round(silence_sec, 2),
        "silence_pct": round(silence_sec / total_dur * 100, 1),
        "speech_sec": round(speech_sec, 2),
        "speech_pct": round(speech_sec / total_dur * 100, 1),
    }


def find_overlaps(segments):
    """Detect segments where two non-silence speakers overlap.
    With merge_dominant pipeline overlaps shouldn't appear in the merged json,
    but we expose the count for UI honesty."""
    n_overlap = 0
    sp = [s for s in segments if s["speaker"] != "silence"]
    for i in range(len(sp) - 1):
        if sp[i]["end"] > sp[i + 1]["start"]:
            n_overlap += 1
    return n_overlap


def detect_interruptions(segments, gap=0.3):
    """Count speaker switches with < gap silence between them — proxy for interruption."""
    n = 0
    sp = [s for s in segments if s["speaker"] != "silence"]
    for i in range(1, len(sp)):
        if sp[i]["speaker"] != sp[i - 1]["speaker"] and (sp[i]["start"] - sp[i - 1]["end"]) < gap:
            n += 1
    return n


def build_dominant_lookup(diar_segments):
    """Return a fn t -> speaker dominant at time t (or None if silence)."""
    starts = [s["start"] for s in diar_segments]

    def dominant_at(t):
        # binary search for the segment containing t
        lo, hi = 0, len(diar_segments) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            seg = diar_segments[mid]
            if seg["end"] <= t:
                lo = mid + 1
            elif seg["start"] > t:
                hi = mid - 1
            else:
                return seg["speaker"] if seg["speaker"] != "silence" else None
        return None

    return dominant_at


def load_and_dedupe_transcript(diar_segments, valid_speakers):
    """Load whisper merged.json (or merged_sentiment.json if present) and dedupe by
    dominant-channel attribution.

    Whisper transcribes each channel independently, so the same utterance often appears
    under multiple speakers (mic bleed). We trust merged_demo.json: at the segment's
    midpoint, only one speaker is dominant. Keep the transcript segment whose claimed
    speaker matches the dominant speaker; drop the rest. If sentiment.py has been run,
    its output (with `sentiment` field) is preferred.
    """
    src = SENTIMENT_JSON if SENTIMENT_JSON.exists() else TRANSCRIPT_JSON
    if not src.exists():
        return [], {"loaded": 0, "kept": 0, "dropped": 0,
                    "available": False, "has_sentiment": False, "source": None}

    with src.open() as f:
        raw = json.load(f)
    has_sentiment = src is SENTIMENT_JSON

    dominant_at = build_dominant_lookup(diar_segments)
    kept, dropped = [], 0
    for seg in raw:
        spk = seg.get("speaker")
        if spk not in valid_speakers:
            dropped += 1
            continue
        mid = (seg["start"] + seg["end"]) / 2
        dom = dominant_at(mid)
        if dom is None or dom != spk:
            dropped += 1
            continue
        text = (seg.get("text") or "").strip()
        if not text:
            dropped += 1
            continue
        item = {
            "speaker": spk,
            "start": round(float(seg["start"]), 3),
            "end": round(float(seg["end"]), 3),
            "text": text,
        }
        if has_sentiment and seg.get("sentiment"):
            item["sentiment"] = seg["sentiment"]
        kept.append(item)
    kept.sort(key=lambda x: x["start"])
    return kept, {
        "loaded": len(raw), "kept": len(kept), "dropped": dropped,
        "available": True, "has_sentiment": has_sentiment, "source": src.name,
    }


def aggregate_sentiment(transcript, valid_speakers):
    """Per-speaker (V, A) centroid + quadrant histogram, plus session-level totals."""
    by_spk = {sid: {"v_sum": 0.0, "a_sum": 0.0, "n": 0,
                    "Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0,
                    "labels_12": defaultdict(int)}
              for sid in valid_speakers}
    total = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0, "n": 0}
    for t in transcript:
        s = t.get("sentiment")
        if not s:
            continue
        spk = t["speaker"]
        if spk not in by_spk:
            continue
        b = by_spk[spk]
        b["v_sum"] += s["v"]
        b["a_sum"] += s["a"]
        b["n"] += 1
        b[s["label_4"]] += 1
        b["labels_12"][s["label_12"]] += 1
        total[s["label_4"]] += 1
        total["n"] += 1

    result = {}
    for sid, b in by_spk.items():
        if b["n"] == 0:
            result[sid] = {"n": 0}
            continue
        # dominant fine label (mode)
        dom_label = max(b["labels_12"].items(), key=lambda kv: kv[1])[0]
        result[sid] = {
            "n": b["n"],
            "v_mean": round(b["v_sum"] / b["n"], 4),
            "a_mean": round(b["a_sum"] / b["n"], 4),
            "Q1": b["Q1"], "Q2": b["Q2"], "Q3": b["Q3"], "Q4": b["Q4"],
            "dominant_label": dom_label,
        }
    return result, total


def main():
    with SEG_JSON.open() as f:
        segments_raw = json.load(f)
    print(f"loaded {len(segments_raw)} raw segments")

    wm = wav_meta(WAV_PATH)
    print(f"wav: {wm['duration']}s, {wm['sample_rate']}Hz, {wm['channels']}ch")

    segments_merged = merge_adjacent(segments_raw)
    print(f"after merging adjacent same-speaker (<{MERGE_GAP}s gap): {len(segments_merged)} segments")

    speakers, silence_stats = aggregate_speakers(segments_merged, wm["duration"])
    print("speakers:")
    for s in speakers:
        print(f"  {s['id']} ({s['name']}): {s['total_sec']}s ({s['share_pct']}%) · "
              f"{s['segment_count']} segs · avg {s['avg_segment_sec']}s · ch {s['channels']}")
    print(f"silence: {silence_stats['silence_sec']}s ({silence_stats['silence_pct']}%)")

    interruptions = detect_interruptions(segments_merged)
    overlaps = find_overlaps(segments_merged)

    valid_speakers = {s["id"] for s in speakers}
    transcript, t_stats = load_and_dedupe_transcript(segments_raw, valid_speakers)
    if t_stats["available"]:
        print(f"transcript: source={t_stats['source']}, loaded {t_stats['loaded']}, "
              f"kept {t_stats['kept']}, dropped {t_stats['dropped']} (bleed/mismatch)")
        per_spk_text = defaultdict(int)
        for t in transcript:
            per_spk_text[t["speaker"]] += 1
        for sid, n in sorted(per_spk_text.items()):
            print(f"  {sid}: {n} utterances")
    else:
        print(f"transcript: not found at {TRANSCRIPT_JSON} — text fields will be empty")

    sentiment_per_speaker, sentiment_totals = ({}, {})
    if t_stats.get("has_sentiment"):
        sentiment_per_speaker, sentiment_totals = aggregate_sentiment(transcript, valid_speakers)
        print("sentiment per speaker (centroid V, A · dominant label · n):")
        for sid, s in sentiment_per_speaker.items():
            if s["n"] == 0:
                print(f"  {sid}: no utterances")
            else:
                print(f"  {sid}: V{s['v_mean']:+.2f} A{s['a_mean']:+.2f} · "
                      f"{s['dominant_label']:<10} · n={s['n']} "
                      f"[Q1={s['Q1']} Q2={s['Q2']} Q3={s['Q3']} Q4={s['Q4']}]")
        # attach centroid + dominant label to speaker entries
        for sp in speakers:
            sp["sentiment"] = sentiment_per_speaker.get(sp["id"], {"n": 0})

    payload = {
        "audio": {
            "src": "../test_run/audio/merged_demo.wav",
            **wm,
        },
        "segments": [
            {
                "start": round(s["start"], 3),
                "end": round(s["end"], 3),
                "speaker": s["speaker"],
                "channel": s.get("channel"),
            }
            for s in segments_merged
        ],
        "speakers": speakers,
        "transcript": transcript,
        "silence": silence_stats,
        "stats": {
            "total_segments": len(segments_merged),
            "raw_segments": len(segments_raw),
            "interruptions": interruptions,
            "overlaps": overlaps,
            "has_text": t_stats["available"],
            "transcript_loaded": t_stats["loaded"],
            "transcript_kept": t_stats["kept"],
            "transcript_dropped": t_stats["dropped"],
            "has_sentiment": t_stats.get("has_sentiment", False),
            "sentiment_totals": sentiment_totals,
        },
    }

    OUT_JS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JS.open("w") as f:
        f.write("window.VIDEO360_AUDIO = ")
        json.dump(payload, f, separators=(",", ":"))
        f.write(";\n")
    print(f"\nwrote {OUT_JS} ({OUT_JS.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
