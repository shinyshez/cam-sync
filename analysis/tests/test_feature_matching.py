"""Tests for feature matching module."""

import pytest
import numpy as np
import torch
from feature_matching import (
    compute_cost_from_matches,
    extract_keypoints_streaming,
    compute_cost_matrix_from_features
)


def test_cost_from_matches_zero():
    """Zero matches should give maximum cost."""
    cost = compute_cost_from_matches(0)
    assert cost == 2.0


def test_cost_from_matches_scale():
    """Matches at scale should give cost near 0."""
    cost = compute_cost_from_matches(250, scale=250.0)
    assert cost == pytest.approx(1.0, abs=0.01)


def test_cost_from_matches_high():
    """Many matches should give cost near 0."""
    cost = compute_cost_from_matches(500, scale=250.0)
    assert cost == 0.0


def test_cost_from_matches_monotonic():
    """Cost should decrease as matches increase."""
    cost_low = compute_cost_from_matches(50)
    cost_mid = compute_cost_from_matches(150)
    cost_high = compute_cost_from_matches(300)

    assert cost_low > cost_mid > cost_high


def test_extract_keypoints_returns_features(synthetic_video):
    """Should return features and timestamps."""
    features, timestamps = extract_keypoints_streaming(
        synthetic_video, sample_fps=1.0, device="cpu"
    )

    assert len(features) > 0
    assert len(timestamps) == len(features)
    assert all(isinstance(t, float) for t in timestamps)


def test_extract_keypoints_feature_structure(synthetic_video):
    """Features should have correct structure."""
    features, _ = extract_keypoints_streaming(
        synthetic_video, sample_fps=1.0, device="cpu"
    )

    feat = features[0]
    assert 'keypoints' in feat
    assert 'descriptors' in feat
    assert 'image_size' in feat
    assert isinstance(feat['keypoints'], torch.Tensor)
    assert isinstance(feat['descriptors'], torch.Tensor)


def test_extract_keypoints_max_keypoints(synthetic_video):
    """Should respect max_keypoints limit."""
    features, _ = extract_keypoints_streaming(
        synthetic_video, sample_fps=1.0, max_keypoints=512, device="cpu"
    )

    for feat in features:
        assert feat['keypoints'].shape[0] <= 512


def test_compute_cost_matrix_shape():
    """Cost matrix should have correct shape."""
    # Create mock features
    features1 = [
        {
            'keypoints': torch.rand(100, 2),
            'descriptors': torch.rand(100, 256),
            'image_size': torch.tensor([480, 640])
        }
        for _ in range(10)
    ]
    features2 = [
        {
            'keypoints': torch.rand(100, 2),
            'descriptors': torch.rand(100, 256),
            'image_size': torch.tensor([480, 640])
        }
        for _ in range(12)
    ]

    cost_matrix = compute_cost_matrix_from_features(
        features1, features2, band_fraction=0.5, device="cpu"
    )

    assert cost_matrix.shape == (10, 12)


def test_compute_cost_matrix_band_constraint():
    """Cost matrix should only compute within band."""
    features1 = [
        {
            'keypoints': torch.rand(50, 2),
            'descriptors': torch.rand(50, 256),
            'image_size': torch.tensor([480, 640])
        }
        for _ in range(20)
    ]
    features2 = [
        {
            'keypoints': torch.rand(50, 2),
            'descriptors': torch.rand(50, 256),
            'image_size': torch.tensor([480, 640])
        }
        for _ in range(20)
    ]

    cost_matrix = compute_cost_matrix_from_features(
        features1, features2, band_fraction=0.2, device="cpu"
    )

    # Outside band should have default high cost
    # Band width = 0.2 * 20 = 4
    # Position (0, 10) is outside band (distance = 10 > 4)
    assert cost_matrix[0, 10] == 2.0
    assert cost_matrix[10, 0] == 2.0


def test_compute_cost_matrix_diagonal():
    """Cost matrix diagonal should have lower costs for similar sequences."""
    # Create features with similar keypoints
    base_kpts = torch.rand(100, 2)
    features1 = [
        {
            'keypoints': base_kpts + torch.randn(100, 2) * 0.01,  # Small noise
            'descriptors': torch.rand(100, 256),
            'image_size': torch.tensor([480, 640])
        }
        for _ in range(5)
    ]
    features2 = features1  # Same features = same frames

    cost_matrix = compute_cost_matrix_from_features(
        features1, features2, band_fraction=1.0, device="cpu"
    )

    # Diagonal should have low costs (same frame = many matches)
    diagonal_costs = np.diag(cost_matrix)
    off_diagonal_costs = cost_matrix[~np.eye(5, dtype=bool)]

    assert np.mean(diagonal_costs) < np.mean(off_diagonal_costs)
