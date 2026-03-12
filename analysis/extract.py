import cv2
import numpy as np


def extract_frames(video_path: str, sample_fps: float = 2.0, crop_top_fraction: float | None = None) -> tuple[list[np.ndarray], list[float]]:
    """Extract frames from a video file at a specified sampling rate.

    Args:
        video_path: Path to the video file
        sample_fps: Frames per second to sample (default: 2.0)
        crop_top_fraction: If set, keep only the top fraction of each frame (e.g. 0.5 = top half).
                          Useful for ignoring bike/handlebars in helmet cam footage.

    Returns:
        A tuple of (frames, timestamps) where:
        - frames: List of numpy arrays with shape (H, W, 3)
        - timestamps: List of float timestamps in seconds
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if source_fps == 0:
        raise ValueError(f"Invalid FPS in video: {video_path}")

    # Calculate frame interval for sampling
    frame_interval = source_fps / sample_fps

    frames = []
    timestamps = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if this frame should be sampled
        if frame_idx % int(round(frame_interval)) == 0:
            if crop_top_fraction is not None:
                h = frame.shape[0]
                frame = frame[:int(h * crop_top_fraction), :, :]
            frames.append(frame)
            # Calculate timestamp in seconds
            timestamp = frame_idx / source_fps
            timestamps.append(timestamp)

        frame_idx += 1

    cap.release()

    return frames, timestamps
