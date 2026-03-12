import pytest
import numpy as np
import cv2
import tempfile
import os


@pytest.fixture
def synthetic_video():
    """Create a synthetic test video: 3 seconds, 10 fps, with frame numbers burned in."""
    width, height = 320, 240
    fps = 10
    duration = 3  # seconds
    total_frames = fps * duration

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.avi', delete=False)
    video_path = temp_file.name
    temp_file.close()

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Write frames with frame numbers
    for frame_num in range(total_frames):
        # Create a solid color frame (blue background)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (100, 50, 50)  # BGR

        # Burn in the frame number
        cv2.putText(frame, f"Frame {frame_num}", (50, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)

    out.release()

    yield video_path

    # Cleanup
    if os.path.exists(video_path):
        os.unlink(video_path)


@pytest.fixture
def synthetic_frames():
    """Generate synthetic numpy array frames for testing."""
    frames = []
    # Create 10 frames of solid colors
    for i in range(10):
        frame = np.ones((240, 320, 3), dtype=np.uint8) * (i * 25)
        frames.append(frame)
    return frames


@pytest.fixture
def synthetic_embeddings():
    """Generate synthetic normalized embeddings for testing."""
    # Create 20 random embeddings of dimension 384 (DINOv2 ViT-S/14 output size)
    embeddings = np.random.randn(20, 384).astype(np.float32)
    # L2 normalize each embedding
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings
