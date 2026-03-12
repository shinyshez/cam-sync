import pytest
import numpy as np
from delta import compute_delta


def test_identical_alignment_zero_delta():
    """Test that diagonal path at 2fps results in delta ≈ 0 everywhere."""
    sample_fps = 2.0

    # Create a perfect diagonal path (20 frames)
    path = [(i, i) for i in range(20)]

    result = compute_delta(path, sample_fps, smooth=False)

    assert "time_vid2" in result, "Result should contain 'time_vid2'"
    assert "delta_seconds" in result, "Result should contain 'delta_seconds'"

    # All deltas should be approximately 0
    assert np.allclose(result["delta_seconds"], 0.0, atol=0.01), \
        "Delta should be near 0 for identical alignment"


def test_constant_offset_delta():
    """Test that path shifted by 10 frames at 2fps results in delta ≈ -5.0s constant.

    Path (i, i+10) means frame i of vid1 aligns with frame i+10 of vid2.
    This means vid2 took more time to reach the same location, so vid2 is behind.
    """
    sample_fps = 2.0
    offset_frames = 10

    # Create path with constant offset: (0,10), (1,11), (2,12), ...
    # This means vid2 is BEHIND by 10 frames
    path = [(i, i + offset_frames) for i in range(20)]

    result = compute_delta(path, sample_fps, smooth=False)

    expected_delta = -offset_frames / sample_fps  # -10 frames / 2 fps = -5 seconds

    # All deltas should be approximately -5.0 (vid2 is behind)
    assert np.allclose(result["delta_seconds"], expected_delta, atol=0.01), \
        f"Delta should be approximately {expected_delta}s for constant offset"


def test_smoothing_reduces_noise():
    """Test that noisy path produces smoothed delta with lower variance than raw."""
    sample_fps = 2.0

    # Create a noisy path around the diagonal
    np.random.seed(42)
    noise = np.random.randint(-2, 3, size=50)
    path = [(i, max(0, i + noise[i])) for i in range(50)]

    # Compute both raw and smoothed
    result_raw = compute_delta(path, sample_fps, smooth=False)
    result_smoothed = compute_delta(path, sample_fps, smooth=True)

    # Smoothed should have lower variance
    var_raw = np.var(result_raw["delta_seconds"])
    var_smoothed = np.var(result_smoothed["delta_seconds"])

    assert var_smoothed < var_raw, \
        f"Smoothed variance ({var_smoothed}) should be less than raw variance ({var_raw})"


def test_output_format():
    """Test that result returns dict with time_vid2 and delta_seconds of equal length."""
    sample_fps = 2.0
    path = [(i, i) for i in range(15)]

    result = compute_delta(path, sample_fps)

    # Check structure
    assert isinstance(result, dict), "Result should be a dict"
    assert "time_vid2" in result, "Result should contain 'time_vid2'"
    assert "delta_seconds" in result, "Result should contain 'delta_seconds'"

    # Check types
    assert isinstance(result["time_vid2"], np.ndarray), "time_vid2 should be numpy array"
    assert isinstance(result["delta_seconds"], np.ndarray), "delta_seconds should be numpy array"

    # Check lengths
    assert len(result["time_vid2"]) == len(result["delta_seconds"]), \
        "time_vid2 and delta_seconds should have equal length"

    # Check lengths match path
    assert len(result["time_vid2"]) == len(path), \
        "Output length should match path length"


def test_delta_computation_formula():
    """Test that delta is computed as t1 - t2 (how far ahead/behind video 2 is)."""
    sample_fps = 1.0  # 1 fps for easy calculation

    # Path where video 2 is ahead: (5, 3) means frame 5 of vid1 matches frame 3 of vid2
    # At time 3s in vid2, we're at time 5s in vid1, so vid2 is 2s ahead (delta = +2)
    path = [(5, 3)]

    result = compute_delta(path, sample_fps, smooth=False)

    # delta = t1 - t2 = 5.0 - 3.0 = 2.0
    # Positive delta means video 2 is ahead (faster)
    assert result["delta_seconds"][0] == 2.0, \
        "Delta should be t1 - t2"


def test_rate_cap_clamps_spikes():
    """A sudden 5-frame jump in delta at 10fps should be clamped to max 0.1s/step."""
    sample_fps = 10.0

    # Steady delta of 0 for 20 frames, then sudden jump to +5 frames offset, then back
    # Raw delta: 0,0,0,...,0, +0.5, +0.5, +0.5, 0,0,...
    # But we build this via path: diagonal, then vid1 jumps ahead 5 frames, then resumes
    path = []
    for i in range(20):
        path.append((i, i))  # delta = 0
    for i in range(20, 30):
        path.append((i + 5, i))  # delta = +0.5s (5 frames at 10fps)
    for i in range(30, 50):
        path.append((i, i))  # delta = 0 again

    result = compute_delta(path, sample_fps, smooth=False)
    delta = result["delta_seconds"]

    # Max allowed change per step is 1/sample_fps = 0.1s
    step_changes = np.abs(np.diff(delta))
    max_change = 1.0 / sample_fps
    assert np.all(step_changes <= max_change + 1e-9), \
        f"Max step change {step_changes.max():.3f} exceeds cap {max_change}"


def test_rate_cap_preserves_gradual_change():
    """A gradually changing delta should pass through the rate cap untouched."""
    sample_fps = 10.0

    # Delta increases by 0.05s per step (well under 0.1s cap)
    # Build path where vid1 gains half a frame per step
    path = []
    for i in range(100):
        j = max(0, i - i // 2)  # vid1 index grows faster
        path.append((i, j))

    result_raw = compute_delta(path, sample_fps, smooth=False)

    # The rate cap shouldn't alter a signal that's already within bounds
    # Check that the overall trend is preserved (start-to-end delta difference)
    delta = result_raw["delta_seconds"]
    # Delta should be monotonically changing (increasing) since vid1 steadily pulls ahead
    # Just check it wasn't flattened
    assert delta[-1] != delta[0], "Rate cap should not flatten gradual trends"


def test_smoothing_window_scales_with_fps():
    """Smoothing window should be ~5 seconds of data, auto-scaled to fps."""
    # At 10fps, 5 seconds = 50 samples, rounded to odd = 51
    # At 1fps, 5 seconds = 5 samples, rounded to odd = 5
    # At 2fps, 5 seconds = 10 samples, rounded to odd = 11
    path_100 = [(i, i) for i in range(100)]

    # Create a noisy path to see smoothing effect
    np.random.seed(42)
    noisy_path = [(i + np.random.randint(-3, 4), i) for i in range(100)]

    result_10fps = compute_delta(noisy_path, sample_fps=10.0, smooth=True)
    result_1fps = compute_delta(noisy_path, sample_fps=1.0, smooth=True)

    # 10fps should smooth more aggressively (wider window in samples)
    var_10fps = np.var(result_10fps["delta_seconds"])
    var_1fps = np.var(result_1fps["delta_seconds"])
    assert var_10fps < var_1fps, \
        f"10fps variance ({var_10fps:.4f}) should be less than 1fps ({var_1fps:.4f})"


def test_rate_cap_scales_with_fps():
    """At 1fps the cap is 1.0s/step; at 10fps it's 0.1s/step."""
    # Same spike path at different fps
    path = [(i, i) for i in range(10)] + [(i + 10, i) for i in range(10, 20)]

    result_1fps = compute_delta(path, sample_fps=1.0, smooth=False)
    result_10fps = compute_delta(path, sample_fps=10.0, smooth=False)

    changes_1fps = np.abs(np.diff(result_1fps["delta_seconds"]))
    changes_10fps = np.abs(np.diff(result_10fps["delta_seconds"]))

    # 10fps should be more aggressively clamped
    assert changes_10fps.max() <= 0.1 + 1e-9
    assert changes_1fps.max() <= 1.0 + 1e-9
