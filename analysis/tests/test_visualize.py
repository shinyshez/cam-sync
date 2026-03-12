import pytest
import numpy as np
import os
import tempfile
import matplotlib
matplotlib.use('Agg')  # Headless rendering
from visualize import plot_delta, plot_cost_matrix


def test_plot_delta_creates_file():
    """Test that plot_delta produces a PNG file at the given path."""
    # Create sample delta result
    delta_result = {
        "time_vid2": np.linspace(0, 10, 50),
        "delta_seconds": np.sin(np.linspace(0, 2*np.pi, 50)) * 5
    }

    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        output_path = tmp.name

    try:
        # Create plot
        plot_delta(delta_result, output_path)

        # Verify file exists and has content
        assert os.path.exists(output_path), "Plot file should be created"
        assert os.path.getsize(output_path) > 0, "Plot file should have content"

    finally:
        # Cleanup
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_plot_cost_matrix_creates_file():
    """Test that plot_cost_matrix produces a PNG heatmap."""
    # Create sample cost matrix (20x25)
    cost_matrix = np.random.rand(20, 25).astype(np.float32)

    # Create sample path
    path = [(i, i) for i in range(20)]

    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        output_path = tmp.name

    try:
        # Create plot
        plot_cost_matrix(cost_matrix, path, output_path)

        # Verify file exists and has content
        assert os.path.exists(output_path), "Heatmap file should be created"
        assert os.path.getsize(output_path) > 0, "Heatmap file should have content"

    finally:
        # Cleanup
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_plot_delta_with_empty_data():
    """Test that plot_delta handles edge cases gracefully."""
    # Single data point
    delta_result = {
        "time_vid2": np.array([0.0]),
        "delta_seconds": np.array([1.0])
    }

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        output_path = tmp.name

    try:
        plot_delta(delta_result, output_path)
        assert os.path.exists(output_path), "Should handle single data point"
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)
