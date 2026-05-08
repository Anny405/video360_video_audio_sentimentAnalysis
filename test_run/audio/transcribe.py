"""
Per-channel Whisper transcription for multi-mic roleplay recordings.

Each input wav is one speaker's mic (ideally already de-bled). We run
mlx-whisper on each channel independently, producing:

    {stem}.txt    plain text
    {stem}.srt    SRT subtitles (sentence-level timestamps)
    {stem}.json   full whisper result incl. word-level timestamps

If --merge is given, also writes:

    merged.txt    speaker-labeled dialogue, sorted by start time
    merged.srt    speaker-labeled SRT

Speaker labels default to "S1, S2, ..." by file order; pass --names to
override (e.g. --names Alice Bob Charlie Dan).

Usage (Apple Silicon, M-series; uses MLX for max speed):
    python transcribe.py *_clean.wav --merge --language zh
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import mlx_whisper

# Reasonable defaults for mlx-community models. Turbo is the sweet spot
# (≈ large-v3 quality, ~5x faster on Apple Silicon).
DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


# --------------------------- formatting helpers ---------------------------

def _ts(seconds: float, comma: bool = True) -> str:
    """seconds -> HH:MM:SS,mmm  (or  HH:MM:SS.mmm if comma=False)"""
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    sep = "," if comma else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def write_txt(segments: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(seg["text"].strip() + "\n")


def write_srt(segments: list[dict], path: Path, speaker: str | None = None) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            text = seg["text"].strip()
            if speaker:
                text = f"{speaker}: {text}"
            f.write(f"{i}\n{_ts(seg['start'])} --> {_ts(seg['end'])}\n{text}\n\n")


def write_json(result: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# --------------------------- main pipeline --------------------------------

def transcribe_one(
    wav_path: Path,
    out_dir: Path,
    model: str,
    language: str | None,
    initial_prompt: str | None,
    word_timestamps: bool,
    verbose: bool,
) -> dict:
    if verbose:
        print(f"[{wav_path.name}] transcribing...", flush=True)
    result = mlx_whisper.transcribe(
        str(wav_path),
        path_or_hf_repo=model,
        language=language,
        initial_prompt=initial_prompt,
        word_timestamps=word_timestamps,
        condition_on_previous_text=False,  # less drift on long files
        verbose=False,
    )
    stem = wav_path.stem
    write_txt(result["segments"], out_dir / f"{stem}.txt")
    write_srt(result["segments"], out_dir / f"{stem}.srt")
    write_json(result, out_dir / f"{stem}.json")
    if verbose:
        dur = result["segments"][-1]["end"] if result["segments"] else 0.0
        print(f"[{wav_path.name}] done — {len(result['segments'])} segments, "
              f"~{dur:.1f}s, lang={result.get('language')}", flush=True)
    return result


def merge_results(
    per_channel: list[tuple[str, dict]],
    out_dir: Path,
) -> None:
    """Interleave segments from all channels into one timeline."""
    merged: list[dict] = []
    for speaker, result in per_channel:
        for seg in result["segments"]:
            text = seg["text"].strip()
            if not text:
                continue
            merged.append({
                "speaker": speaker,
                "start": seg["start"],
                "end": seg["end"],
                "text": text,
            })
    merged.sort(key=lambda s: s["start"])

    # merged.txt: human-readable dialogue
    with (out_dir / "merged.txt").open("w", encoding="utf-8") as f:
        for s in merged:
            f.write(f"[{_ts(s['start'], comma=False)} - {_ts(s['end'], comma=False)}] "
                    f"{s['speaker']}: {s['text']}\n")

    # merged.srt: subtitle with speaker prefix
    with (out_dir / "merged.srt").open("w", encoding="utf-8") as f:
        for i, s in enumerate(merged, 1):
            f.write(f"{i}\n{_ts(s['start'])} --> {_ts(s['end'])}\n"
                    f"{s['speaker']}: {s['text']}\n\n")

    # merged.json: full structured timeline
    with (out_dir / "merged.json").open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("inputs", nargs="+", help="input wav files (one per speaker)")
    ap.add_argument("-o", "--out-dir", default="transcripts",
                    help="output directory (default: transcripts/)")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL,
                    help=f"mlx-whisper model repo (default: {DEFAULT_MODEL})")
    ap.add_argument("-l", "--language", default=None,
                    help="ISO code, e.g. 'zh', 'en'. Default: auto-detect")
    ap.add_argument("-p", "--initial-prompt", default=None,
                    help="prompt to bias decoding (names, jargon, etc.)")
    ap.add_argument("--no-words", action="store_true",
                    help="disable word-level timestamps (faster)")
    ap.add_argument("--merge", action="store_true",
                    help="also write merged speaker-labeled dialogue")
    ap.add_argument("--names", nargs="+", default=None,
                    help="speaker labels, one per input (default: S1, S2, ...)")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    inputs = [Path(p) for p in args.inputs]
    for p in inputs:
        if not p.exists():
            sys.exit(f"input not found: {p}")

    if args.names:
        if len(args.names) != len(inputs):
            sys.exit(f"--names count ({len(args.names)}) "
                     f"must match input count ({len(inputs)})")
        names = args.names
    else:
        names = [f"S{i+1}" for i in range(len(inputs))]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_channel: list[tuple[str, dict]] = []
    for name, wav in zip(names, inputs):
        result = transcribe_one(
            wav, out_dir,
            model=args.model,
            language=args.language,
            initial_prompt=args.initial_prompt,
            word_timestamps=not args.no_words,
            verbose=not args.quiet,
        )
        per_channel.append((name, result))

    if args.merge:
        merge_results(per_channel, out_dir)
        if not args.quiet:
            print(f"merged dialogue -> {out_dir/'merged.txt'}, "
                  f"{out_dir/'merged.srt'}, {out_dir/'merged.json'}")


if __name__ == "__main__":
    main()
