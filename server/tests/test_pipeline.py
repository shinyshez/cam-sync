import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
import numpy as np

from server.pipeline import run_pipeline


@pytest.fixture
def fake_videos(tmp_path):
    """Create two fake video file paths (content doesn't matter — we mock extraction)."""
    vid1 = tmp_path / "vid1.mp4"
    vid2 = tmp_path / "vid2.mp4"
    vid1.write_bytes(b"fake")
    vid2.write_bytes(b"fake")
    return str(vid1), str(vid2), str(tmp_path)


def _mock_embed(video_path, sample_fps=2.0, device="cpu", dtype="float32",
                batch_size=32, crop_top_fraction=None, progress_callback=None):
    """Return fake embeddings and timestamps, calling progress_callback."""
    n = 10
    embeddings = np.random.randn(n, 384).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    timestamps = [i / sample_fps for i in range(n)]
    if progress_callback:
        for i in range(1, n + 1):
            progress_callback(frames_processed=i, total_estimate=n)
    return embeddings, timestamps


@patch("server.pipeline.extract_and_embed_streaming", side_effect=_mock_embed)
@patch("server.pipeline.compute_cost_matrix")
@patch("server.pipeline.align_sequences")
@patch("server.pipeline.compute_delta")
def test_pipeline_progress_callback(mock_delta, mock_align, mock_cost, mock_embed, fake_videos):
    """Progress callback receives messages with required keys."""
    vid1, vid2, output_dir = fake_videos

    # Setup mocks
    mock_cost.return_value = np.ones((10, 10))
    mock_align.return_value = [(i, i) for i in range(10)]
    mock_delta.return_value = {
        "time_vid2": np.arange(10) / 2.0,
        "delta_seconds": np.zeros(10),
    }

    messages = []
    def callback(msg):
        messages.append(msg)

    run_pipeline(vid1, vid2, output_dir, progress_callback=callback)

    # Should have messages from multiple stages
    stages = [m["stage"] for m in messages]
    assert "embedding_vid1" in stages
    assert "embedding_vid2" in stages
    assert "aligning" in stages
    assert "complete" in stages

    # Each message should have required keys
    for m in messages:
        assert "stage" in m
        assert "percent" in m


@patch("server.pipeline.extract_and_embed_streaming", side_effect=_mock_embed)
@patch("server.pipeline.compute_cost_matrix")
@patch("server.pipeline.align_sequences")
@patch("server.pipeline.compute_delta")
def test_pipeline_produces_delta_csv(mock_delta, mock_align, mock_cost, mock_embed, fake_videos):
    """Pipeline writes delta.csv to output directory."""
    vid1, vid2, output_dir = fake_videos

    mock_cost.return_value = np.ones((10, 10))
    mock_align.return_value = [(i, i) for i in range(10)]
    mock_delta.return_value = {
        "time_vid2": np.arange(10) / 2.0,
        "delta_seconds": np.linspace(0, 1, 10),
    }

    run_pipeline(vid1, vid2, output_dir)

    csv_path = os.path.join(output_dir, "delta.csv")
    assert os.path.exists(csv_path), "delta.csv should be created"

    with open(csv_path) as f:
        lines = f.readlines()
    assert lines[0].strip() == "time_vid2_seconds,delta_seconds"
    assert len(lines) == 11  # header + 10 rows
