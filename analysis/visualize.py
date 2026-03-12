import matplotlib.pyplot as plt
import numpy as np


def plot_delta(delta_result: dict, output_path: str):
    """Plot the time delta curve and save to file.

    Args:
        delta_result: Dictionary with 'time_vid2' and 'delta_seconds' arrays
        output_path: Path to save the PNG file
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    time_vid2 = delta_result["time_vid2"]
    delta_seconds = delta_result["delta_seconds"]

    # Plot delta curve
    ax.plot(time_vid2, delta_seconds, 'b-', linewidth=2, label='Time Delta')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Zero (tied)')
    ax.fill_between(time_vid2, 0, delta_seconds,
                     where=(delta_seconds >= 0), alpha=0.3, color='green',
                     label='Video 2 ahead')
    ax.fill_between(time_vid2, 0, delta_seconds,
                     where=(delta_seconds < 0), alpha=0.3, color='red',
                     label='Video 2 behind')

    ax.set_xlabel('Time in Video 2 (seconds)', fontsize=12)
    ax.set_ylabel('Time Delta (seconds)', fontsize=12)
    ax.set_title('Video Time Synchronization Delta', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_cost_matrix(cost_matrix: np.ndarray, path: list[tuple[int, int]], output_path: str):
    """Plot the DTW cost matrix as a heatmap with the alignment path overlaid.

    Args:
        cost_matrix: Cost matrix of shape (N1, N2)
        path: List of (i, j) tuples representing the DTW alignment path
        output_path: Path to save the PNG file
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    im = ax.imshow(cost_matrix, cmap='viridis', aspect='auto', origin='lower')

    # Overlay the DTW path
    if path:
        path_array = np.array(path)
        ax.plot(path_array[:, 1], path_array[:, 0], 'r-', linewidth=2, label='DTW Path')
        ax.plot(path_array[:, 1], path_array[:, 0], 'wo', markersize=3, alpha=0.5)

    ax.set_xlabel('Video 2 Frame Index', fontsize=12)
    ax.set_ylabel('Video 1 Frame Index', fontsize=12)
    ax.set_title('DTW Cost Matrix with Alignment Path', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Distance', fontsize=11)

    if path:
        ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
