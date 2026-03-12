"""Tests for render_sync.py"""

import csv
import os
import tempfile

import cv2
import numpy as np
import pytest

from render_sync import load_delta, interpolate_delta, render_sync


@pytest.fixture
def delta_csv(tmp_path):
    """Create a simple delta.csv for testing."""
    path = tmp_path / "delta.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_vid2_seconds", "delta_seconds"])
        writer.writerow([0.0, 2.0])
        writer.writerow([5.0, 3.0])
        writer.writerow([10.0, 2.5])
    return str(path)


@pytest.fixture
def make_test_video(tmp_path):
    """Factory to create a short test video with solid color frames."""

    def _make(filename, color, num_frames=30, fps=10, width=160, height=120):
        path = str(tmp_path / filename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        for _ in range(num_frames):
            frame = np.full((height, width, 3), color, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return path

    return _make


class TestLoadDelta:
    def test_loads_csv(self, delta_csv):
        times, deltas = load_delta(delta_csv)
        assert len(times) == 3
        assert len(deltas) == 3
        assert times[0] == 0.0
        assert deltas[1] == 3.0

    def test_returns_numpy_arrays(self, delta_csv):
        times, deltas = load_delta(delta_csv)
        assert isinstance(times, np.ndarray)
        assert isinstance(deltas, np.ndarray)


class TestInterpolateDelta:
    def test_exact_sample_points(self, delta_csv):
        times, deltas = load_delta(delta_csv)
        assert interpolate_delta(0.0, times, deltas) == pytest.approx(2.0)
        assert interpolate_delta(5.0, times, deltas) == pytest.approx(3.0)

    def test_midpoint_interpolation(self, delta_csv):
        times, deltas = load_delta(delta_csv)
        # Midpoint between (0, 2.0) and (5, 3.0) => 2.5
        assert interpolate_delta(2.5, times, deltas) == pytest.approx(2.5)

    def test_clamps_before_start(self, delta_csv):
        times, deltas = load_delta(delta_csv)
        assert interpolate_delta(-1.0, times, deltas) == pytest.approx(2.0)

    def test_clamps_after_end(self, delta_csv):
        times, deltas = load_delta(delta_csv)
        assert interpolate_delta(20.0, times, deltas) == pytest.approx(2.5)


class TestRenderSync:
    def test_produces_output_file(self, make_test_video, delta_csv, tmp_path):
        vid1 = make_test_video("vid1.mp4", (255, 0, 0))  # blue
        vid2 = make_test_video("vid2.mp4", (0, 255, 0))  # green
        output = str(tmp_path / "out.mp4")

        render_sync(vid1, vid2, delta_csv, output)

        assert os.path.exists(output)
        cap = cv2.VideoCapture(output)
        assert cap.isOpened()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Stacked vertically: width same, height doubled
        assert w == 160
        assert h == 240
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert frame_count > 0
        cap.release()

    def test_different_widths_resized(self, make_test_video, delta_csv, tmp_path):
        vid1 = make_test_video("vid1.mp4", (255, 0, 0), width=200, height=120)
        vid2 = make_test_video("vid2.mp4", (0, 255, 0), width=160, height=120)
        output = str(tmp_path / "out.mp4")

        render_sync(vid1, vid2, delta_csv, output)

        cap = cv2.VideoCapture(output)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # Should use vid2's width
        assert w == 160
        cap.release()

    def test_custom_fps(self, make_test_video, delta_csv, tmp_path):
        vid1 = make_test_video("vid1.mp4", (255, 0, 0), fps=10)
        vid2 = make_test_video("vid2.mp4", (0, 255, 0), fps=10)
        output = str(tmp_path / "out.mp4")

        render_sync(vid1, vid2, delta_csv, output, fps=5)

        cap = cv2.VideoCapture(output)
        fps = cap.get(cv2.CAP_PROP_FPS)
        assert fps == pytest.approx(5.0, abs=0.5)
        cap.release()
