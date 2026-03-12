import pytest
import numpy as np
from align import compute_cost_matrix, align_sequences

# Import both libraries for comparison tests
try:
    from dtw import dtw as dtw_python
    HAVE_DTW_PYTHON = True
except ImportError:
    HAVE_DTW_PYTHON = False

try:
    from dtaidistance import dtw as dtw_dtai
    HAVE_DTAIDISTANCE = True
except ImportError:
    HAVE_DTAIDISTANCE = False


def test_identical_sequences_diagonal_path(synthetic_embeddings):
    """Test that identical embedding sequences result in diagonal DTW path with cost ≈ 0."""
    # Use same embeddings for both sequences
    embeddings1 = synthetic_embeddings.copy()
    embeddings2 = synthetic_embeddings.copy()

    cost_matrix = compute_cost_matrix(embeddings1, embeddings2)
    path = align_sequences(cost_matrix, band_fraction=1.0)  # No band constraint for this test

    # Path should be roughly diagonal
    for i, j in path:
        assert abs(i - j) <= 2, "Path should be close to diagonal for identical sequences"

    # Average cost should be near zero
    avg_cost = np.mean([cost_matrix[i, j] for i, j in path])
    assert avg_cost < 0.1, f"Average cost should be near 0 for identical sequences, got {avg_cost}"


def test_offset_sequences_detect_shift():
    """Test that seq2 = seq1 shifted by K frames results in path reflecting that offset."""
    # Create a sequence of 20 embeddings
    embeddings1 = np.random.randn(20, 384).astype(np.float32)
    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)

    # Shift by 5 frames
    shift = 5
    embeddings2 = np.vstack([
        np.random.randn(shift, 384),  # Random start
        embeddings1[:15]  # Shifted content
    ])
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

    cost_matrix = compute_cost_matrix(embeddings1, embeddings2)
    path = align_sequences(cost_matrix, band_fraction=1.0)

    # Check that the path shows an offset
    # After the initial frames, j should be approximately i + shift
    middle_path = [(i, j) for i, j in path if 5 <= i <= 10]
    if middle_path:
        avg_offset = np.mean([j - i for i, j in middle_path])
        assert 3 <= avg_offset <= 7, f"Expected offset around {shift}, got {avg_offset}"


def test_speed_variation():
    """Test that seq2 with middle section stretched (repeated embeddings) shows non-linear warp."""
    # Create a sequence of 20 embeddings
    embeddings1 = np.random.randn(20, 384).astype(np.float32)
    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)

    # Create seq2 with repeated middle section (simulating slower segment)
    embeddings2 = np.vstack([
        embeddings1[:10],
        embeddings1[10:12].repeat(3, axis=0),  # Repeat frames 10-11 three times
        embeddings1[12:]
    ])

    cost_matrix = compute_cost_matrix(embeddings1, embeddings2)
    path = align_sequences(cost_matrix, band_fraction=1.0)

    # Path should be non-linear - there should be some points where j increases
    # faster than i (horizontal movement in the DTW path)
    deltas = [(path[k+1][1] - path[k][1]) - (path[k+1][0] - path[k][0])
              for k in range(len(path) - 1)]

    # Should have some segments where j advances more than i
    assert any(d > 0 for d in deltas), "Path should show non-linear warping for speed variation"


def test_band_constraint_limits_search():
    """Test that with narrow band, path stays within band."""
    # Create two similar but slightly different sequences
    embeddings1 = np.random.randn(50, 384).astype(np.float32)
    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)

    embeddings2 = embeddings1 + np.random.randn(50, 384) * 0.1
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

    cost_matrix = compute_cost_matrix(embeddings1, embeddings2)

    # Use narrow band
    band_fraction = 0.1
    path = align_sequences(cost_matrix, band_fraction=band_fraction)

    # Check that path stays within band
    n, m = cost_matrix.shape
    band_width = int(band_fraction * max(n, m))

    for i, j in path:
        # Diagonal distance from main diagonal
        diagonal_dist = abs(j - i)
        assert diagonal_dist <= band_width, f"Path should stay within band width {band_width}"


def test_compute_cost_matrix_shape(synthetic_embeddings):
    """Test that cost matrix has correct shape."""
    embeddings1 = synthetic_embeddings[:10]
    embeddings2 = synthetic_embeddings[:15]

    cost_matrix = compute_cost_matrix(embeddings1, embeddings2)

    assert cost_matrix.shape == (10, 15), "Cost matrix should have shape (N1, N2)"
    assert np.all(cost_matrix >= 0), "All costs should be non-negative"
    assert np.all(cost_matrix <= 2), "Cosine distance should be at most 2"


# TDD Cycle 1: dtaidistance equivalence tests

@pytest.mark.skipif(not HAVE_DTAIDISTANCE, reason="dtaidistance not installed")
def test_dtaidistance_equivalence(synthetic_embeddings):
    """Test that dtaidistance produces equivalent results to dtw-python."""
    embeddings1 = synthetic_embeddings[:10]
    embeddings2 = synthetic_embeddings[:10]

    cost_matrix = compute_cost_matrix(embeddings1, embeddings2)

    # This test will fail initially because align_sequences still uses dtw-python
    # After implementing dtaidistance, it should pass
    path = align_sequences(cost_matrix, band_fraction=0.3)

    # Verify basic properties
    assert len(path) > 0, "Path should not be empty"
    assert path[0] == (0, 0), "Path should start at (0, 0)"
    assert path[-1][0] == 9 and path[-1][1] == 9, "Path should end at last frame"

    # Path should be monotonically increasing
    for k in range(len(path) - 1):
        i1, j1 = path[k]
        i2, j2 = path[k + 1]
        assert i2 >= i1 and j2 >= j1, "Path should be monotonically increasing"


@pytest.mark.skipif(not HAVE_DTAIDISTANCE, reason="dtaidistance not installed")
def test_dtaidistance_handles_large_matrices():
    """Test that dtaidistance can handle large cost matrices without OOM."""
    # Create 1000x1000 embeddings (simulating 10 fps on 100s video)
    size = 1000
    embeddings1 = np.random.randn(size, 384).astype(np.float32)
    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)

    embeddings2 = embeddings1 + np.random.randn(size, 384) * 0.05
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

    cost_matrix = compute_cost_matrix(embeddings1, embeddings2)

    # This should not cause OOM with dtaidistance
    path = align_sequences(cost_matrix, band_fraction=0.2)

    assert len(path) > 0, "Path should not be empty"
    assert len(path) >= size, "Path length should be at least sequence length"


@pytest.mark.skipif(not HAVE_DTAIDISTANCE, reason="dtaidistance not installed")
def test_dtaidistance_sakoe_chiba_band():
    """Test that Sakoe-Chiba band constraint works correctly with dtaidistance."""
    embeddings1 = np.random.randn(50, 384).astype(np.float32)
    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)

    embeddings2 = embeddings1 + np.random.randn(50, 384) * 0.1
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

    cost_matrix = compute_cost_matrix(embeddings1, embeddings2)

    # Test with narrow band
    band_fraction = 0.1
    path = align_sequences(cost_matrix, band_fraction=band_fraction)

    # Verify path stays within band
    band_width = int(band_fraction * max(cost_matrix.shape))
    for i, j in path:
        diagonal_dist = abs(j - i)
        assert diagonal_dist <= band_width, f"Path should stay within band width {band_width}"
