"""Render a synced side-by-side comparison video from two source videos and a delta.csv."""

import argparse
import csv
import sys

import cv2
import numpy as np


def load_delta(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load delta.csv and return (times, deltas) as numpy arrays."""
    times = []
    deltas = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time_vid2_seconds"]))
            deltas.append(float(row["delta_seconds"]))
    return np.array(times), np.array(deltas)


def interpolate_delta(t: float, times: np.ndarray, deltas: np.ndarray) -> float:
    """Linearly interpolate delta at time t, clamping outside range."""
    return float(np.interp(t, times, deltas))


def render_sync(
    video1_path: str,
    video2_path: str,
    delta_csv_path: str,
    output_path: str,
    fps: float | None = None,
    label1: str = "Rider 1",
    label2: str = "Rider 2",
) -> None:
    """Render synced comparison video."""
    times, deltas = load_delta(delta_csv_path)

    cap2 = cv2.VideoCapture(video2_path)
    cap1 = cv2.VideoCapture(video1_path)

    if not cap2.isOpened() or not cap1.isOpened():
        raise RuntimeError("Could not open one or both video files")

    vid2_fps = cap2.get(cv2.CAP_PROP_FPS)
    vid2_width = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid2_height = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid2_frames = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    vid1_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_fps = fps if fps is not None else vid2_fps
    out_width = vid2_width
    out_height = vid2_height + vid1_height

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (out_width, out_height))

    duration = vid2_frames / vid2_fps
    frame_interval = 1.0 / out_fps
    t = 0.0

    while t < duration:
        # Read vid2 frame at time t
        cap2.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ret2, frame2 = cap2.read()
        if not ret2:
            break

        # Read vid1 frame at time t + delta(t)
        delta = interpolate_delta(t, times, deltas)
        t1 = t + delta
        cap1.set(cv2.CAP_PROP_POS_MSEC, t1 * 1000.0)
        ret1, frame1 = cap1.read()
        if not ret1:
            # Past end of vid1 — use black frame
            frame1 = np.zeros((vid1_height, vid2_width, 3), dtype=np.uint8)

        # Resize vid1 frame to match vid2 width if needed
        h1, w1 = frame1.shape[:2]
        if w1 != out_width:
            new_h = int(h1 * out_width / w1)
            frame1 = cv2.resize(frame1, (out_width, new_h))
            # Update height for stacking
            if frame1.shape[0] != vid1_height:
                frame1 = cv2.resize(frame1, (out_width, vid1_height))

        # Draw overlay text
        delta_str = f"{delta:+.1f}s"
        color = (0, 200, 0) if delta >= 0 else (0, 0, 220)

        cv2.putText(
            frame1, label1, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
        cv2.putText(
            frame2, f"{label2}  {delta_str}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
        )

        stacked = np.vstack([frame1, frame2])
        writer.write(stacked)
        t += frame_interval

    writer.release()
    cap1.release()
    cap2.release()
    print(f"Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Render synced comparison video")
    parser.add_argument("video1", help="Path to video 1")
    parser.add_argument("video2", help="Path to video 2")
    parser.add_argument("delta_csv", help="Path to delta.csv")
    parser.add_argument("output", nargs="?", default="output/synced.mp4", help="Output path")
    parser.add_argument("--fps", type=float, default=None, help="Output FPS override")
    parser.add_argument("--label1", default="Rider 1", help="Label for video 1")
    parser.add_argument("--label2", default="Rider 2", help="Label for video 2")
    args = parser.parse_args()

    render_sync(
        args.video1, args.video2, args.delta_csv, args.output,
        fps=args.fps, label1=args.label1, label2=args.label2,
    )


if __name__ == "__main__":
    main()
