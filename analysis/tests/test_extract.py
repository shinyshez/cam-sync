import pytest
import numpy as np
from extract import extract_frames


def test_extract_frames_returns_list_of_arrays(synthetic_video):
    """Test that extract_frames returns a list of numpy arrays with shape (H, W, 3)."""
    frames, timestamps = extract_frames(synthetic_video)

    assert isinstance(frames, list), "frames should be a list"
    assert len(frames) > 0, "frames should not be empty"

    for frame in frames:
        assert isinstance(frame, np.ndarray), "each frame should be a numpy array"
        assert frame.ndim == 3, "frame should be 3-dimensional"
        assert frame.shape[2] == 3, "frame should have 3 color channels"


def test_extract_frames_respects_sample_fps(synthetic_video):
    """Test that a 10s@30fps video sampled at 2fps returns exactly 20 frames.

    Note: Our synthetic video is 3s@10fps, so at 2fps we expect 6 frames.
    """
    sample_fps = 2.0
    frames, timestamps = extract_frames(synthetic_video, sample_fps=sample_fps)

    # 3 seconds at 2 fps = 6 frames
    expected_frames = 6
    assert len(frames) == expected_frames, f"Expected {expected_frames} frames at {sample_fps} fps"


def test_extract_frames_returns_timestamps(synthetic_video):
    """Test that extract_frames returns parallel list of float timestamps in seconds."""
    frames, timestamps = extract_frames(synthetic_video)

    assert isinstance(timestamps, list), "timestamps should be a list"
    assert len(timestamps) == len(frames), "timestamps should match number of frames"

    for ts in timestamps:
        assert isinstance(ts, float), "each timestamp should be a float"
        assert ts >= 0, "timestamps should be non-negative"

    # Timestamps should be increasing
    for i in range(1, len(timestamps)):
        assert timestamps[i] > timestamps[i-1], "timestamps should be monotonically increasing"


def test_extract_frames_crop_top_half(synthetic_video):
    """With crop_top_fraction=0.5, frames should be top half of original height."""
    frames_full, _ = extract_frames(synthetic_video)
    frames_cropped, _ = extract_frames(synthetic_video, crop_top_fraction=0.5)

    full_h = frames_full[0].shape[0]
    cropped_h = frames_cropped[0].shape[0]

    assert cropped_h == full_h // 2, f"Expected height {full_h // 2}, got {cropped_h}"
    assert frames_cropped[0].shape[1] == frames_full[0].shape[1], "Width should be unchanged"
    assert frames_cropped[0].shape[2] == 3, "Should still have 3 channels"


def test_extract_frames_crop_preserves_count(synthetic_video):
    """Cropping should not change the number of frames extracted."""
    frames_full, ts_full = extract_frames(synthetic_video, sample_fps=2.0)
    frames_cropped, ts_cropped = extract_frames(synthetic_video, sample_fps=2.0, crop_top_fraction=0.5)

    assert len(frames_cropped) == len(frames_full)
    assert ts_cropped == ts_full


def test_extract_frames_crop_none_is_full_frame(synthetic_video):
    """crop_top_fraction=None should return full frames (default behavior)."""
    frames, _ = extract_frames(synthetic_video)
    # Synthetic video is 240 height
    assert frames[0].shape[0] == 240
