import pytest
import numpy as np
import tempfile
import cv2
from streaming import extract_and_embed_streaming


def create_test_video(path, duration_sec=5, fps=10):
    """Create a test video for streaming tests."""
    width, height = 320, 240
    total_frames = int(duration_sec * fps)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for frame_num in range(total_frames):
        # Create frame with gradient
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = [(frame_num * 5) % 255, 100, 150]
        cv2.putText(frame, f"F{frame_num}", (width//2, height//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)

    out.release()


def test_streaming_returns_embeddings_and_timestamps():
    """Test that streaming extraction returns embeddings and timestamps."""
    with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as tmp:
        video_path = tmp.name

    try:
        create_test_video(video_path, duration_sec=3, fps=10)

        # Stream with 2 fps sampling
        embeddings, timestamps = extract_and_embed_streaming(
            video_path, sample_fps=2.0, device='cpu', dtype='float32'
        )

        # Should get 6 frames (3 seconds at 2 fps)
        assert embeddings.shape[0] == 6, "Should extract 6 frames at 2 fps over 3 seconds"
        assert embeddings.shape[1] == 384, "Should have 384-dim embeddings"
        assert len(timestamps) == 6, "Should have 6 timestamps"
        assert isinstance(embeddings, np.ndarray), "Should return numpy array"
        assert embeddings.dtype == np.float32, "Should respect dtype parameter"

    finally:
        import os
        if os.path.exists(video_path):
            os.unlink(video_path)


def test_streaming_memory_efficient():
    """Test that streaming uses minimal memory (no frame accumulation)."""
    import tracemalloc

    with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as tmp:
        video_path = tmp.name

    try:
        # Create 10 second video at 10 fps = 100 frames
        create_test_video(video_path, duration_sec=10, fps=10)

        tracemalloc.start()
        baseline, _ = tracemalloc.get_traced_memory()

        # Extract at 5 fps = 50 frames
        embeddings, timestamps = extract_and_embed_streaming(
            video_path, sample_fps=5.0, device='cpu', dtype='float16'
        )

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Peak memory should be much less than 50 frames × 320×240×3 = 11.5 MB
        # Should be dominated by model weights (~100 MB) not frames
        peak_mb = (peak - baseline) / 1024 / 1024

        # With streaming, we should never hold all frames in memory
        # Memory should be mostly model + small batch, not all frames
        assert peak_mb < 500, f"Peak memory should be < 500 MB for streaming, got {peak_mb:.1f} MB"

        assert embeddings.shape[0] == 50, "Should have 50 embeddings"
        assert len(timestamps) == 50, "Should have 50 timestamps"

    finally:
        import os
        if os.path.exists(video_path):
            os.unlink(video_path)


def test_streaming_float16_support():
    """Test that streaming respects dtype parameter."""
    with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as tmp:
        video_path = tmp.name

    try:
        create_test_video(video_path, duration_sec=2, fps=10)

        # Test float16
        embeddings_f16, _ = extract_and_embed_streaming(
            video_path, sample_fps=2.0, device='cpu', dtype='float16'
        )

        assert embeddings_f16.dtype == np.float16, "Should return float16 when requested"

        # Test float32
        embeddings_f32, _ = extract_and_embed_streaming(
            video_path, sample_fps=2.0, device='cpu', dtype='float32'
        )

        assert embeddings_f32.dtype == np.float32, "Should return float32 when requested"

    finally:
        import os
        if os.path.exists(video_path):
            os.unlink(video_path)


def test_streaming_progress_callback_called():
    """Test that progress_callback is called with frames_processed and total_estimate."""
    with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as tmp:
        video_path = tmp.name

    try:
        create_test_video(video_path, duration_sec=3, fps=10)

        calls = []
        def callback(frames_processed, total_estimate):
            calls.append({'frames_processed': frames_processed, 'total_estimate': total_estimate})

        embeddings, timestamps = extract_and_embed_streaming(
            video_path, sample_fps=2.0, device='cpu', dtype='float32',
            batch_size=4, progress_callback=callback
        )

        # Should have been called at least once
        assert len(calls) > 0, "Callback should be called at least once"
        # Each call should have frames_processed and total_estimate
        for c in calls:
            assert 'frames_processed' in c
            assert 'total_estimate' in c
            assert c['total_estimate'] > 0
        # Last call's frames_processed should equal total embeddings
        assert calls[-1]['frames_processed'] == len(timestamps)

    finally:
        import os
        if os.path.exists(video_path):
            os.unlink(video_path)


def test_streaming_progress_callback_none_works():
    """Test that progress_callback=None (default) still works (regression)."""
    with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as tmp:
        video_path = tmp.name

    try:
        create_test_video(video_path, duration_sec=2, fps=10)

        # Should not raise with no callback
        embeddings, timestamps = extract_and_embed_streaming(
            video_path, sample_fps=2.0, device='cpu', dtype='float32'
        )
        assert embeddings.shape[0] > 0

    finally:
        import os
        if os.path.exists(video_path):
            os.unlink(video_path)


def test_streaming_normalized_embeddings():
    """Test that streaming embeddings are L2 normalized."""
    with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as tmp:
        video_path = tmp.name

    try:
        create_test_video(video_path, duration_sec=2, fps=10)

        embeddings, _ = extract_and_embed_streaming(
            video_path, sample_fps=2.0, device='cpu', dtype='float32'
        )

        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=0.01), "Embeddings should be L2 normalized"

    finally:
        import os
        if os.path.exists(video_path):
            os.unlink(video_path)
