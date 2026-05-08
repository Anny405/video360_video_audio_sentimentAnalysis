"""
merge_dominant.py — merge N mic channels into a single mono stream that
always plays the loudest ("dominant") channel, with a side file labeling
who is talking when.

Why this kills residual bleed: the direct path from speaker j to mic j is
always shorter than to any other mic, so mic j's signal is louder when j
talks. Picking the loudest channel at every moment selects the mic that
"owns" the current speaker — bleed in the other mics is gated out by
construction.

Sample-rate, total length and timestamps are preserved exactly.

Outputs (with default --out-prefix merged):
    merged.wav     mono mix
    merged.srt     speaker-labeled subtitle (one cue per speaker turn)
    merged.json    list of {start, end, speaker, channel}
    merged.tsv     tab-separated for scripts
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import oaconvolve


def merge_dominant(
    in_paths: list[str],
    out_prefix: str,
    names: list[str] | None = None,
    frame_ms: float = 20.0,
    smooth_ms: float = 300.0,
    crossfade_ms: float = 30.0,
    silence_db: float = -50.0,
    margin_db: float = 2.0,
    verbose: bool = True,
) -> int:
    # ---- 1. Load & align ----------------------------------------------------
    raw, sr, subs = [], None, []
    for p in in_paths:
        info = sf.info(p)
        subs.append(info.subtype)
        x, s = sf.read(p, always_2d=False, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1).astype(np.float32)
        if sr is None:
            sr = s
        elif s != sr:
            raise ValueError(f"sample-rate mismatch: {p} ({s}) vs {sr}")
        raw.append(x)
    N = min(len(x) for x in raw)
    signals = np.stack([np.ascontiguousarray(x[:N]) for x in raw], axis=0)
    n_ch = signals.shape[0]
    if names is None:
        names = [f"S{i+1}" for i in range(n_ch)]
    elif len(names) != n_ch:
        raise ValueError(f"--names count ({len(names)}) "
                         f"must match input count ({n_ch})")
    if verbose:
        print(f"loaded {n_ch} ch, {N} samples @ {sr} Hz ({N/sr:.1f}s)")

    # ---- 2. Frame energy in dB ---------------------------------------------
    fn = max(1, int(round(sr * frame_ms / 1000)))
    nf = N // fn
    framed = signals[:, : nf * fn].reshape(n_ch, nf, fn)
    energy_db = 10.0 * np.log10(
        (framed.astype(np.float64) ** 2).mean(axis=2) + 1e-12
    )

    # ---- 3. Smooth energies (avoid winner flicker) -------------------------
    sf_n = max(1, int(round(smooth_ms / frame_ms)))
    energy_s = uniform_filter1d(energy_db, size=sf_n, axis=1, mode="nearest")
    max_db = energy_s.max(axis=0)
    voiced = max_db > silence_db

    # ---- 4. Hysteretic winner selection -------------------------------------
    winner = np.empty(nf, dtype=np.int32)
    cur = int(energy_s[:, 0].argmax())
    winner[0] = cur
    for f in range(1, nf):
        cand = int(energy_s[:, f].argmax())
        if cand != cur and energy_s[cand, f] > energy_s[cur, f] + margin_db:
            cur = cand
        winner[f] = cur

    # ---- 5. Per-sample one-hot gains -> Hann smoothed crossfade ------------
    winner_ps = np.repeat(winner, fn).astype(np.int32)
    if len(winner_ps) < N:
        winner_ps = np.concatenate([
            winner_ps,
            np.full(N - len(winner_ps), winner_ps[-1], dtype=np.int32),
        ])

    cf_n = max(3, int(round(sr * crossfade_ms / 1000)))
    if cf_n % 2 == 0:
        cf_n += 1
    kernel = np.hanning(cf_n).astype(np.float32)
    kernel /= kernel.sum()  # keeps gain-sum = 1, so no level pumping

    output = np.zeros(N, dtype=np.float32)
    for i in range(n_ch):
        mask = (winner_ps == i).astype(np.float32)
        smoothed = oaconvolve(mask, kernel, mode="same").astype(np.float32)
        output += signals[i] * smoothed
        if verbose:
            pct = 100.0 * (winner == i).mean()
            print(f"  ch{i+1} ({names[i]:<10}) dominant in "
                  f"{(winner==i).sum():>6} / {nf} frames ({pct:5.1f}%)")

    # ---- 6. Speaker intervals (collapse consecutive same-winner frames) ----
    intervals: list[dict] = []
    cur_w, cur_v, start = int(winner[0]), bool(voiced[0]), 0
    for f in range(1, nf):
        w, v = int(winner[f]), bool(voiced[f])
        if w != cur_w or v != cur_v:
            intervals.append({
                "start": round(start * frame_ms / 1000, 3),
                "end":   round(f     * frame_ms / 1000, 3),
                "speaker": names[cur_w] if cur_v else "silence",
                "channel": cur_w if cur_v else None,
            })
            cur_w, cur_v, start = w, v, f
    intervals.append({
        "start": round(start * frame_ms / 1000, 3),
        "end":   round(nf    * frame_ms / 1000, 3),
        "speaker": names[cur_w] if cur_v else "silence",
        "channel": cur_w if cur_v else None,
    })

    # ---- 7. Write outputs ---------------------------------------------------
    out_audio = f"{out_prefix}.wav"
    sub = subs[0] if subs[0] in ("PCM_16", "PCM_24", "PCM_32", "FLOAT") \
          else "PCM_24"
    peak = float(np.max(np.abs(output)))
    if peak > 0.999:
        output *= 0.999 / peak
    sf.write(out_audio, output, sr, subtype=sub)

    with open(f"{out_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(intervals, f, ensure_ascii=False, indent=2)

    def fmt(t: float) -> str:
        ms = int(round(t * 1000))
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    with open(f"{out_prefix}.srt", "w", encoding="utf-8") as f:
        idx = 1
        for iv in intervals:
            if iv["speaker"] == "silence":
                continue
            f.write(f"{idx}\n{fmt(iv['start'])} --> {fmt(iv['end'])}\n"
                    f"[{iv['speaker']}]\n\n")
            idx += 1

    with open(f"{out_prefix}.tsv", "w", encoding="utf-8") as f:
        f.write("start\tend\tspeaker\tchannel\n")
        for iv in intervals:
            ch = "" if iv["channel"] is None else iv["channel"]
            f.write(f"{iv['start']:.3f}\t{iv['end']:.3f}\t"
                    f"{iv['speaker']}\t{ch}\n")

    if verbose:
        speech = sum(1 for iv in intervals if iv["speaker"] != "silence")
        print(f"wrote {out_audio}, .srt/.json/.tsv  "
              f"({len(intervals)} intervals: {speech} speech, "
              f"{len(intervals)-speech} silence)")
    return sr


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("inputs", nargs="+", help="input wav files (one per mic)")
    ap.add_argument("-o", "--out-prefix", default="merged",
                    help="prefix for output files (default: merged)")
    ap.add_argument("--names", nargs="+", default=None,
                    help="speaker labels (one per input)")
    ap.add_argument("--frame-ms", type=float, default=20.0)
    ap.add_argument("--smooth-ms", type=float, default=300.0,
                    help="energy smoothing window (larger = more stable)")
    ap.add_argument("--crossfade-ms", type=float, default=30.0,
                    help="audio crossfade at speaker switches")
    ap.add_argument("--silence-db", type=float, default=-50.0)
    ap.add_argument("--margin-db", type=float, default=2.0,
                    help="hysteresis: new winner must beat current by this")
    args = ap.parse_args()

    merge_dominant(
        args.inputs, args.out_prefix,
        names=args.names,
        frame_ms=args.frame_ms,
        smooth_ms=args.smooth_ms,
        crossfade_ms=args.crossfade_ms,
        silence_db=args.silence_db,
        margin_db=args.margin_db,
    )


if __name__ == "__main__":
    main()
