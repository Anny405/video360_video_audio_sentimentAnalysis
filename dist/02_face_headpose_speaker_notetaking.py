"""Apply behavior threshold rules to feature CSVs and emit timeline outputs.

Inputs : per-participant `*_features.csv` (output of step 01).
Outputs: per-participant `*_behaviors.csv` plus an optional Plotly HTML
         timeline showing when each behavior is active.

CLI:
    python 02_face_headpose_speaker_notetaking.py \\
        --input-dir  /path/to/features_yolo01/ \\
        --output-dir /path/to/behavior_outputs/

    # skip the HTML render
    python 02_face_headpose_speaker_notetaking.py ... --no-html
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Action-layer thresholds. Speaking is handled separately via the per-person audio track.
DEFAULT_PARAMS: dict[str, float] = {
    "look_down_th": 0.06,
    "look_up_th": -0.02,
    "yaw_left_th": -0.05,
    "yaw_right_th": 0.05,
    "nod_range_th": 0.04,
    "nod_window": 15,
    "smile_th": 0.42,
    "wrist_y_th": 0.65,
    "wrist_speed_th": 0.02,
}


def infer_behaviors(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Add behavior flag columns and a `take_notes_flag` based on `params`."""
    if params is None:
        params = DEFAULT_PARAMS
    df = df.copy()

    for col in [
        "yaw_proxy", "pitch_proxy", "smile_score",
        "left_wrist_x", "left_wrist_y", "right_wrist_x", "right_wrist_y",
        "left_wrist_speed", "right_wrist_speed",
    ]:
        if col in df.columns:
            df[col] = df[col].interpolate(limit_direction="both")

    df["look_down_flag"] = df["pitch_proxy"] > params["look_down_th"]
    df["look_up_flag"] = df["pitch_proxy"] < params["look_up_th"]
    df["turn_left_flag"] = df["yaw_proxy"] < params["yaw_left_th"]
    df["turn_right_flag"] = df["yaw_proxy"] > params["yaw_right_th"]

    win = int(params["nod_window"])
    df["pitch_range"] = (
        df["pitch_proxy"].rolling(win, min_periods=5).max()
        - df["pitch_proxy"].rolling(win, min_periods=5).min()
    )
    df["head_nodding_flag"] = df["pitch_range"] > params["nod_range_th"]

    df["smiling_flag"] = df["smile_score"] > params["smile_th"]

    df["hand_near_bottom"] = (
        (df["left_wrist_y"] > params["wrist_y_th"])
        | (df["right_wrist_y"] > params["wrist_y_th"])
    )
    df["hand_moving"] = (
        (df["left_wrist_speed"] > params["wrist_speed_th"])
        | (df["right_wrist_speed"] > params["wrist_speed_th"])
    )
    df["take_notes_flag"] = (
        df["look_down_flag"] & df["hand_near_bottom"] & df["hand_moving"]
    )

    return df


def _sec_to_mmss(x: float) -> str:
    if pd.isna(x):
        return "NA"
    m = int(x // 60)
    s = int(x % 60)
    return f"{m:02d}:{s:02d}"


def render_timeline_html(df: pd.DataFrame, out_path: str | Path,
                         time_col: str = "t") -> None:
    """Save an interactive Plotly behavior timeline as a standalone HTML file."""
    import plotly.graph_objects as go

    behavior_cols = [
        "look_down_flag", "head_nodding_flag", "smiling_flag",
        "take_notes_flag",
    ]
    behavior_names = {
        "look_down_flag": "look_down",
        "head_nodding_flag": "nodding",
        "smiling_flag": "smiling",
        "take_notes_flag": "take_notes",
    }
    behavior_cols = [c for c in behavior_cols if c in df.columns]

    plot_df = df.copy()
    plot_df["time_str"] = plot_df[time_col].apply(_sec_to_mmss)

    def get_active(row):
        active = [behavior_names[c] for c in behavior_cols
                  if pd.notna(row[c]) and bool(row[c])]
        return ", ".join(active) if active else "none"

    plot_df["active_behaviors"] = plot_df.apply(get_active, axis=1)

    fig = go.Figure()
    for i, col in enumerate(behavior_cols):
        y_vals = np.where(plot_df[col].fillna(False).astype(bool), i, np.nan)
        fig.add_trace(go.Scatter(
            x=plot_df[time_col], y=y_vals, mode="markers",
            name=behavior_names[col],
            customdata=plot_df[["time_str", "active_behaviors"]].values,
            hovertemplate=(
                "time: %{customdata[0]}<br>"
                "seconds: %{x:.2f}<br>"
                "active: %{customdata[1]}<br>"
                "<extra></extra>"
            ),
            marker=dict(size=7),
        ))

    fig.update_layout(
        title="Interactive Behavior Timeline",
        xaxis_title="time (seconds)",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(behavior_cols))),
            ticktext=[behavior_names[c] for c in behavior_cols],
        ),
        hovermode="x unified",
        height=500, width=1400,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))


def run(
    input_dir: str | Path,
    output_dir: str | Path,
    params: dict | None = None,
    render_html: bool = True,
) -> dict[str, pd.DataFrame]:
    """Process every `*_features.csv` under input_dir."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_files = sorted(input_dir.glob("*_features.csv"))
    if not feature_files:
        print(f"no *_features.csv found under {input_dir}")
        return {}

    print("found feature files:")
    for f in feature_files:
        print(" -", f.name)

    all_dfs: dict[str, pd.DataFrame] = {}
    for csv_path in feature_files:
        participant_id = csv_path.stem.replace("_features", "")
        out_csv = output_dir / f"{participant_id}_behaviors.csv"

        print(f"\nProcessing {participant_id} ...")
        df = pd.read_csv(csv_path)
        df = infer_behaviors(df, params)
        df.to_csv(out_csv, index=False)
        all_dfs[participant_id] = df
        print(f"  saved: {out_csv}  ({len(df)} rows)")

        if render_html:
            html_path = output_dir / f"{participant_id}_timeline.html"
            render_timeline_html(df, html_path)
            print(f"  timeline: {html_path}")

    return all_dfs


def main() -> None:
    p = argparse.ArgumentParser(description="Infer behaviors from feature CSVs")
    p.add_argument("--input-dir",  required=True, help="Directory of *_features.csv")
    p.add_argument("--output-dir", required=True, help="Where to write *_behaviors.csv")
    p.add_argument("--no-html", action="store_true", help="Skip Plotly timeline HTML")
    args = p.parse_args()

    run(args.input_dir, args.output_dir, render_html=not args.no_html)


if __name__ == "__main__":
    main()
