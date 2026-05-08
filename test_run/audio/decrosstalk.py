"""
Multi-channel crosstalk (mic-bleed) cancellation for N-speaker recordings.

Each input wav = one mic worn by/near one speaker. Each mic also picks up
the other speakers ("bleed"). We model the bleed as a short FIR from each
of the OTHER channels, fit the FIR on frames where the target speaker is
NOT dominant (so we never learn to reconstruct the target itself), then
subtract the predicted bleed from the full channel.

- Length / sample-rate / timestamps: unchanged.
- Speakers: never removed; only bleed of others is subtracted.
- Speed: closed-form least-squares + FFT convolution; a few seconds per
  channel for typical session lengths.

Usage:
    python decrosstalk.py ch1.wav ch2.wav ch3.wav ch4.wav
        -> writes ch1_clean.wav, ch2_clean.wav, ...
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from numpy.lib.stride_tricks import sliding_window_view


def cancel_crosstalk(
    in_paths: list[str],
    out_paths: list[str],
    filter_len: int = 512,
    frame_ms: float = 20.0,
    dominance_db: float = 2.0,
    silence_db: float = -50.0,
    max_train_samples: int = 200_000,
    chunk_size: int = 20_000,
    ridge: float = 1e-4,
    seed: int = 0,
    verbose: bool = True,
) -> int:
    # ---- 1. Load & align ----------------------------------------------------
    raw, sr, subtypes = [], None, []
    for p in in_paths:
        info = sf.info(p)
        subtypes.append(info.subtype)
        x, s = sf.read(p, always_2d=False, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1).astype(np.float32)
        if sr is None:
            sr = s
        elif s != sr:
            raise ValueError(f"sample-rate mismatch in {p}: {s} vs {sr}")
        raw.append(x)
    N = min(len(x) for x in raw)
    signals = np.stack([np.ascontiguousarray(x[:N]) for x in raw], axis=0)
    n_ch = signals.shape[0]
    if verbose:
        print(f"loaded {n_ch} ch, {N} samples @ {sr} Hz "
              f"({N/sr:.1f}s)")

    # ---- 2. Frame energy + dominance ---------------------------------------
    frame_n = max(1, int(round(sr * frame_ms / 1000.0)))
    n_frames = N // frame_n
    framed = signals[:, : n_frames * frame_n].reshape(n_ch, n_frames, frame_n)
    energy = (framed.astype(np.float64) ** 2).mean(axis=2) + 1e-12
    energy_db = 10.0 * np.log10(energy)
    max_db = energy_db.max(axis=0)
    is_dominant = energy_db >= (max_db - dominance_db)   # (C, F)
    has_voice = max_db > silence_db                       # (F,)

    out = signals.copy()
    L = filter_len
    rng = np.random.default_rng(seed)

    # ---- 3. Per-channel: fit FIR on "i is silent" frames, subtract ----------
    for i in range(n_ch):
        ref_chs = [j for j in range(n_ch) if j != i]

        # Frames where target i is NOT dominant AND somebody else is talking
        train_frames = (~is_dominant[i]) & has_voice
        if train_frames.sum() < 50:
            if verbose:
                print(f"[ch{i}] not enough training frames "
                      f"({int(train_frames.sum())}); leaving as-is")
            continue

        # Sample indices inside those frames
        frame_starts = np.flatnonzero(train_frames) * frame_n
        offsets = np.arange(frame_n)
        idx = (frame_starts[:, None] + offsets[None, :]).ravel()
        idx = idx[(idx >= L - 1) & (idx < N)]
        if len(idx) > max_train_samples:
            idx = np.sort(rng.choice(idx, size=max_train_samples, replace=False))
        n_train = len(idx)

        # Sliding-window views: row s of sw[k] is signals[ref_chs[k], s : s+L]
        sws = [sliding_window_view(signals[j], L) for j in ref_chs]

        D = (n_ch - 1) * L
        XtX = np.zeros((D, D), dtype=np.float64)
        Xty = np.zeros(D, dtype=np.float64)

        # Build the design matrix in chunks to keep memory low.
        # At sample t, regression vector for ref j is
        #     [ref_j[t], ref_j[t-1], ..., ref_j[t-L+1]]
        # which is sw[t-L+1] reversed.
        for cs in range(0, n_train, chunk_size):
            sub = idx[cs : cs + chunk_size]
            cols = [sw[sub - L + 1][:, ::-1] for sw in sws]
            Xc = np.concatenate(cols, axis=1).astype(np.float64)
            yc = signals[i, sub].astype(np.float64)
            XtX += Xc.T @ Xc
            Xty += Xc.T @ yc

        # Tikhonov ridge for numerical stability
        XtX += (ridge * np.trace(XtX) / D) * np.eye(D)
        h = np.linalg.solve(XtX, Xty).astype(np.float32).reshape(n_ch - 1, L)

        # Apply: predicted bleed = sum_j (h_j * ref_j); subtract
        pred = np.zeros(N, dtype=np.float32)
        for k, j in enumerate(ref_chs):
            pred += fftconvolve(signals[j], h[k], mode="full")[:N]
        out[i] = signals[i] - pred

        if verbose:
            res_db = 10 * np.log10(((signals[i] - out[i]) ** 2).mean() + 1e-12)
            in_db = 10 * np.log10((signals[i] ** 2).mean() + 1e-12)
            print(f"[ch{i}] trained on {n_train} samples; "
                  f"removed {res_db - in_db:+.1f} dB worth of bleed energy")

    # ---- 4. Save ------------------------------------------------------------
    # Avoid hard clipping; scale only if necessary
    peak = float(np.max(np.abs(out)))
    if peak > 0.999:
        out *= 0.999 / peak
        if verbose:
            print(f"normalized peak {peak:.3f} -> 0.999 to avoid clipping")

    for i, p in enumerate(out_paths):
        sub = subtypes[i] if subtypes[i] in ("PCM_16", "PCM_24", "PCM_32", "FLOAT") \
              else "PCM_24"
        sf.write(p, out[i], sr, subtype=sub)
        if verbose:
            print(f"  wrote {p}  ({sub})")

    return sr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="input wav paths (one per mic)")
    ap.add_argument("-s", "--out-suffix", default="_clean",
                    help="suffix appended to each output filename (default: _clean)")
    ap.add_argument("-L", "--filter-len", type=int, default=512,
                    help="FIR taps per reference channel (default: 512 ≈ 10ms @48k)")
    ap.add_argument("--frame-ms", type=float, default=20.0)
    ap.add_argument("--dominance-db", type=float, default=2.0,
                    help="lower = stricter (more frames excluded from training)")
    ap.add_argument("--silence-db", type=float, default=-50.0)
    ap.add_argument("--max-train", type=int, default=200_000)
    args = ap.parse_args()

    out_paths = [
        str(Path(p).with_name(Path(p).stem + args.out_suffix + ".wav"))
        for p in args.inputs
    ]
    cancel_crosstalk(
        args.inputs, out_paths,
        filter_len=args.filter_len,
        frame_ms=args.frame_ms,
        dominance_db=args.dominance_db,
        silence_db=args.silence_db,
        max_train_samples=args.max_train,
    )


if __name__ == "__main__":
    main()
