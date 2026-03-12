import pytest
import numpy as np
from embeddings import compute_embeddings


def test_embed_returns_correct_shape(synthetic_frames):
    """Test that compute_embeddings returns ndarray of shape (N, embed_dim)."""
    embeddings = compute_embeddings(synthetic_frames)

    assert isinstance(embeddings, np.ndarray), "embeddings should be a numpy array"
    assert embeddings.ndim == 2, "embeddings should be 2-dimensional"
    assert embeddings.shape[0] == len(synthetic_frames), "first dimension should match number of frames"
    assert embeddings.shape[1] == 384, "DINOv2 ViT-S/14 should produce 384-dim embeddings"


def test_embeddings_are_normalized(synthetic_frames):
    """Test that all output vectors have L2 norm ≈ 1.0."""
    embeddings = compute_embeddings(synthetic_frames)

    norms = np.linalg.norm(embeddings, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-5), "all embeddings should be L2-normalized"


def test_similar_frames_have_high_similarity(synthetic_frames):
    """Test that two near-identical frames have cosine similarity > 0.9."""
    # Create two nearly identical frames (same solid color)
    frame1 = np.ones((240, 320, 3), dtype=np.uint8) * 128
    frame2 = np.ones((240, 320, 3), dtype=np.uint8) * 128

    embeddings = compute_embeddings([frame1, frame2])

    # Compute cosine similarity
    cosine_sim = np.dot(embeddings[0], embeddings[1])

    assert cosine_sim > 0.9, f"Similar frames should have high cosine similarity, got {cosine_sim}"


def test_different_frames_have_lower_similarity():
    """Test that a red frame vs a blue frame has lower similarity than two red frames."""
    # Create colored frames
    red_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    red_frame[:, :, 2] = 255  # Red channel in BGR

    blue_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    blue_frame[:, :, 0] = 255  # Blue channel in BGR

    red_frame2 = red_frame.copy()

    # Compute embeddings
    embeddings = compute_embeddings([red_frame, blue_frame, red_frame2])

    # Similarity between two red frames
    red_red_sim = np.dot(embeddings[0], embeddings[2])

    # Similarity between red and blue frames
    red_blue_sim = np.dot(embeddings[0], embeddings[1])

    assert red_red_sim > red_blue_sim, f"Similar colors should have higher similarity than different colors"


# TDD Cycle 2: Float16 embeddings tests

def test_float16_output(synthetic_frames):
    """Test that embeddings can be returned as float16."""
    embeddings = compute_embeddings(synthetic_frames, dtype="float16")

    assert embeddings.dtype == np.float16, "Embeddings should be float16 when requested"
    assert embeddings.shape[0] == len(synthetic_frames), "Should return embedding for each frame"
    assert embeddings.shape[1] == 384, "Embeddings should be 384-dimensional"


def test_float16_preserves_normalization(synthetic_frames):
    """Test that L2 norm ≈ 1.0 is preserved with float16."""
    embeddings = compute_embeddings(synthetic_frames, dtype="float16")

    norms = np.linalg.norm(embeddings.astype(np.float32), axis=1)

    # Float16 has less precision, but norms should still be close to 1.0
    assert np.allclose(norms, 1.0, atol=0.01), "Float16 embeddings should still be approximately normalized"


def test_float16_cosine_similarity_accuracy(synthetic_frames):
    """Test that cosine similarity is preserved within acceptable tolerance for float16."""
    # Compute both float32 and float16 versions
    embeddings_f32 = compute_embeddings(synthetic_frames, dtype="float32")
    embeddings_f16 = compute_embeddings(synthetic_frames, dtype="float16")

    # Compare cosine similarities
    for i in range(min(5, len(synthetic_frames))):
        for j in range(i+1, min(5, len(synthetic_frames))):
            sim_f32 = np.dot(embeddings_f32[i], embeddings_f32[j])
            sim_f16 = np.dot(embeddings_f16[i].astype(np.float32),
                           embeddings_f16[j].astype(np.float32))

            diff = abs(sim_f32 - sim_f16)
            assert diff < 0.01, f"Cosine similarity difference should be < 0.01, got {diff}"
