import numpy as np
from scipy.signal import savgol_filter


def compute_delta(path: list[tuple[int, int]], sample_fps: float, smooth: bool = True) -> dict:
    """Compute time delta between two video sequences from DTW alignment path.

    The delta represents how far ahead or behind video 2 is relative to video 1:
    - Positive delta: video 2 is ahead (faster)
    - Negative delta: video 2 is behind (slower)
    - Delta = t1 - t2

    Args:
        path: List of (i, j) tuples representing alignment path (i=video1, j=video2)
        sample_fps: Frames per second used for sampling
        smooth: Whether to apply Savitzky-Golay smoothing (default: True)

    Returns:
        Dictionary with:
        - 'time_vid2': Time points in video 2 (seconds)
        - 'delta_seconds': Time difference at each point (seconds)
    """
    # Convert frame indices to timestamps
    path_array = np.array(path)
    frame_idx_1 = path_array[:, 0]
    frame_idx_2 = path_array[:, 1]

    # Convert to seconds
    time_1 = frame_idx_1 / sample_fps
    time_2 = frame_idx_2 / sample_fps

    # Compute delta: how far ahead/behind is video 2
    # delta = t1 - t2
    # If delta > 0: at this point in vid2, vid1 is further along (vid2 is ahead/faster)
    # If delta < 0: at this point in vid2, vid1 is behind (vid2 is behind/slower)
    delta_seconds = time_1 - time_2

    # Rate-of-change cap: delta can change by at most 1 frame per frame of vid2
    # elapsed time. At 10fps, if vid2 advances 0.1s, delta can shift by 0.1s.
    # When vid2 doesn't advance (horizontal DTW run), delta is held constant.
    frame_duration = 1.0 / sample_fps
    for i in range(1, len(delta_seconds)):
        dt2 = time_2[i] - time_2[i - 1]
        # Cap proportional to vid2 time elapsed (0 elapsed = 0 change allowed)
        max_change = max(dt2, 0.0)
        diff = delta_seconds[i] - delta_seconds[i - 1]
        if abs(diff) > max_change:
            delta_seconds[i] = delta_seconds[i - 1] + np.sign(diff) * max_change

    # Apply smoothing if requested
    if smooth and len(delta_seconds) > 5:
        # Savitzky-Golay filter: 5-second window auto-scaled to fps
        window_length = int(round(10.0 * sample_fps))
        if window_length % 2 == 0:
            window_length += 1  # must be odd
        window_length = max(5, min(window_length, len(delta_seconds) if len(delta_seconds) % 2 == 1 else len(delta_seconds) - 1))
        delta_seconds = savgol_filter(delta_seconds, window_length=window_length, polyorder=2)

    return {
        "time_vid2": time_2,
        "delta_seconds": delta_seconds
    }
