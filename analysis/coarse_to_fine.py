#!/usr/bin/env python3
"""Coarse-to-fine alignment using DINOv2 embeddings."""

import numpy as np
from pathlib import Path
from typing import Tuple, List
from streaming import extract_and_embed_streaming
from align import compute_cost_matrix, align_sequences


def compute_narrow_band_from_path(
    coarse_path: List[Tuple[int, int]],
    coarse_n: int,
    coarse_m: int,
    fine_n: int,
    fine_m: int,
    band_width: int = 10
) -> np.ndarray:
    """Create a band mask for fine alignment based on coarse path.

    Args:
        coarse_path: DTW path from coarse alignment [(i, j), ...]
        coarse_n: Number of frames in coarse video 1
        coarse_m: Number of frames in coarse video 2
        fine_n: Number of frames in fine video 1
        fine_m: Number of frames in fine video 2
        band_width: Width of band around path in frames

    Returns:
        Boolean mask of shape (fine_n, fine_m) where True = compute cost
    """
    # Scale factor from coarse to fine
    scale_i = fine_n / coarse_n
    scale_j = fine_m / coarse_m

    # Create mask initialized to False
    mask = np.zeros((fine_n, fine_m), dtype=bool)

    # For each point in coarse path, mark a band in fine resolution
    for coarse_i, coarse_j in coarse_path:
        # Map to fine resolution
        fine_i = int(coarse_i * scale_i)
        fine_j = int(coarse_j * scale_j)

        # Mark band around this point
        i_start = max(0, fine_i - band_width)
        i_end = min(fine_n, fine_i + band_width + 1)
        j_start = max(0, fine_j - band_width)
        j_end = min(fine_m, fine_j + band_width + 1)

        mask[i_start:i_end, j_start:j_end] = True

    return mask


def compute_cost_matrix_masked(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Compute cosine distance cost matrix only for masked frame pairs.

    Args:
        embeddings1: L2-normalized embeddings (N, D) for video 1
        embeddings2: L2-normalized embeddings (M, D) for video 2
        mask: Boolean mask (N, M) indicating which pairs to compute

    Returns:
        Cost matrix with high cost (2.0) for unmasked pairs, cosine distance for masked
    """
    n, m = mask.shape
    # Start with high cost everywhere
    cost_matrix = np.full((n, m), 2.0, dtype=np.float64)

    # Compute full cosine distance, then apply mask
    full_cost = compute_cost_matrix(embeddings1, embeddings2)
    cost_matrix[mask] = full_cost[mask]

    masked_pairs = mask.sum()
    total_pairs = n * m
    print(f"\nComputing cost matrix with masked DINOv2 cosine distance...")
    print(f"  Matrix size: {n}x{m}")
    print(f"  Masked pairs: {masked_pairs} ({100*masked_pairs/total_pairs:.1f}% of total)")

    return cost_matrix


def align_coarse_to_fine(
    video1_path: str,
    video2_path: str,
    coarse_fps: float = 0.5,
    fine_fps: float = 1.0,
    coarse_band: float = 0.5,
    fine_band_width: int = 10,
    device: str = "cpu",
    output_dir: Path = None,
    crop_top_fraction: float | None = 0.5,
    embedding_dtype: str = "float32",
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[float], List[float]]:
    """Two-stage coarse-to-fine DINOv2 embedding alignment.

    Args:
        video1_path: Path to video 1
        video2_path: Path to video 2
        coarse_fps: FPS for coarse pass (default: 0.5)
        fine_fps: FPS for fine pass (default: 1.0)
        coarse_band: Band fraction for coarse DTW (default: 0.5, wide)
        fine_band_width: Width of refined band in frames (default: 10)
        device: Compute device
        output_dir: Optional output directory to save coarse results
        crop_top_fraction: Keep only top fraction of frame (default: 0.5, ignores bike/bars)
        embedding_dtype: Embedding dtype, "float32" or "float16" (default: "float32")

    Returns:
        (cost_matrix, path, timestamps1, timestamps2) at fine resolution
    """
    print("=" * 70)
    print("COARSE-TO-FINE ALIGNMENT (DINOv2)")
    print("=" * 70)

    # ========== COARSE PASS ==========
    print(f"\n{'='*70}")
    print("STAGE 1: COARSE ALIGNMENT")
    print(f"{'='*70}")
    print(f"  Sampling at {coarse_fps} fps for fast rough alignment\n")

    print(f"[1/4] Extracting coarse embeddings from video 1...")
    coarse_emb1, coarse_ts1 = extract_and_embed_streaming(
        video1_path, sample_fps=coarse_fps, device=device,
        dtype=embedding_dtype, crop_top_fraction=crop_top_fraction,
    )
    print(f"  Extracted {len(coarse_ts1)} frames, shape: {coarse_emb1.shape}")

    print(f"\n[2/4] Extracting coarse embeddings from video 2...")
    coarse_emb2, coarse_ts2 = extract_and_embed_streaming(
        video2_path, sample_fps=coarse_fps, device=device,
        dtype=embedding_dtype, crop_top_fraction=crop_top_fraction,
    )
    print(f"  Extracted {len(coarse_ts2)} frames, shape: {coarse_emb2.shape}")

    print(f"\n[3/4] Computing coarse cost matrix...")
    coarse_cost = compute_cost_matrix(coarse_emb1, coarse_emb2)
    print(f"  Cost range: [{coarse_cost.min():.3f}, {coarse_cost.max():.3f}]")

    print(f"\n[4/4] Computing coarse DTW alignment...")
    coarse_path = align_sequences(coarse_cost, band_fraction=coarse_band)
    print(f"  Coarse path length: {len(coarse_path)}")

    # Save coarse results if output directory provided
    if output_dir is not None:
        print(f"\n[Coarse] Saving coarse alignment results...")
        coarse_dir = output_dir / "coarse"
        coarse_dir.mkdir(parents=True, exist_ok=True)

        # Compute and save coarse delta
        coarse_delta_raw = compute_delta_from_timestamps(coarse_path, coarse_ts1, coarse_ts2, smooth=True)
        coarse_delta = {
            'time_vid2': coarse_delta_raw['timestamps_vid2'],
            'delta_seconds': coarse_delta_raw['delta']
        }

        # Save coarse delta CSV
        coarse_csv = coarse_dir / "delta.csv"
        import csv
        with open(coarse_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time_vid2_seconds', 'delta_seconds'])
            for t, d in zip(coarse_delta['time_vid2'], coarse_delta['delta_seconds']):
                writer.writerow([t, d])
        print(f"         coarse/delta.csv")

        # Save coarse visualizations
        from visualize import plot_delta, plot_cost_matrix

        coarse_delta_png = coarse_dir / "delta.png"
        plot_delta(coarse_delta, str(coarse_delta_png))
        print(f"         coarse/delta.png")

        coarse_heatmap_png = coarse_dir / "heatmap.png"
        plot_cost_matrix(coarse_cost, coarse_path, str(coarse_heatmap_png))
        print(f"         coarse/heatmap.png")

        print(f"  ✓ Coarse results saved to {coarse_dir}")

    # ========== FINE PASS ==========
    print(f"\n{'='*70}")
    print("STAGE 2: FINE ALIGNMENT")
    print(f"{'='*70}")
    print(f"  Sampling at {fine_fps} fps with narrow band around coarse path\n")

    print(f"[5/8] Extracting fine embeddings from video 1...")
    fine_emb1, fine_ts1 = extract_and_embed_streaming(
        video1_path, sample_fps=fine_fps, device=device,
        dtype=embedding_dtype, crop_top_fraction=crop_top_fraction,
    )
    print(f"  Extracted {len(fine_ts1)} frames, shape: {fine_emb1.shape}")

    print(f"\n[6/8] Extracting fine embeddings from video 2...")
    fine_emb2, fine_ts2 = extract_and_embed_streaming(
        video2_path, sample_fps=fine_fps, device=device,
        dtype=embedding_dtype, crop_top_fraction=crop_top_fraction,
    )
    print(f"  Extracted {len(fine_ts2)} frames, shape: {fine_emb2.shape}")

    print(f"\n[7/8] Computing narrow band mask from coarse path...")
    band_mask = compute_narrow_band_from_path(
        coarse_path,
        len(coarse_ts1),
        len(coarse_ts2),
        len(fine_ts1),
        len(fine_ts2),
        band_width=fine_band_width
    )
    masked_pairs = band_mask.sum()
    total_pairs = len(fine_ts1) * len(fine_ts2)
    reduction = 100 * (1 - masked_pairs / total_pairs)
    print(f"  Band mask: {masked_pairs} pairs ({100*masked_pairs/total_pairs:.1f}% of matrix)")
    print(f"  Reduction: {reduction:.1f}% fewer pairs to compute")

    print(f"\n[8/8] Computing fine cost matrix within band...")
    fine_cost = compute_cost_matrix_masked(
        fine_emb1, fine_emb2,
        mask=band_mask,
    )
    print(f"  Cost range: [{fine_cost.min():.3f}, {fine_cost.max():.3f}]")

    print(f"\n[Final] Computing fine DTW alignment...")
    fine_path = align_sequences(fine_cost, band_fraction=1.0)  # No additional band needed
    print(f"  Fine path length: {len(fine_path)}")

    print(f"\n{'='*70}")
    print("COARSE-TO-FINE ALIGNMENT COMPLETE")
    print(f"{'='*70}")

    return fine_cost, fine_path, fine_ts1, fine_ts2


def compute_delta_from_timestamps(
    path: List[Tuple[int, int]],
    timestamps1: List[float],
    timestamps2: List[float],
    smooth: bool = True,
    smooth_window_sec: float = 10.0,
) -> dict:
    """Compute time delta from DTW path and timestamps.

    Args:
        path: DTW alignment path [(i, j), ...]
        timestamps1: Timestamps for video 1 frames
        timestamps2: Timestamps for video 2 frames
        smooth: Apply smoothing
        smooth_window_sec: Smoothing window in seconds

    Returns:
        Dictionary with 'timestamps_vid2' and 'delta'
    """
    from scipy.signal import savgol_filter

    path_array = np.array(path)
    time_1 = np.array([timestamps1[i] for i in path_array[:, 0]])
    time_2 = np.array([timestamps2[j] for j in path_array[:, 1]])

    delta_seconds = time_1 - time_2

    # Rate-of-change cap: delta can change by at most 1 frame per frame of vid2
    # elapsed time. When vid2 doesn't advance (horizontal DTW run), delta is held.
    for i in range(1, len(delta_seconds)):
        dt2 = time_2[i] - time_2[i - 1]
        max_change = max(dt2, 0.0)
        diff = delta_seconds[i] - delta_seconds[i - 1]
        if abs(diff) > max_change:
            delta_seconds[i] = delta_seconds[i - 1] + np.sign(diff) * max_change

    # Apply smoothing: 5-second window auto-scaled to effective fps
    if smooth and len(delta_seconds) > 5:
        median_dt = np.median(np.diff(time_2)) if len(time_2) > 1 else 1.0
        effective_fps = 1.0 / median_dt if median_dt > 0 else 1.0
        window_length = int(round(smooth_window_sec * effective_fps))
        if window_length % 2 == 0:
            window_length += 1
        window_length = max(5, min(window_length, len(delta_seconds) if len(delta_seconds) % 2 == 1 else len(delta_seconds) - 1))
        delta_seconds = savgol_filter(delta_seconds, window_length=window_length, polyorder=2)

    return {
        "timestamps_vid2": time_2,
        "delta": delta_seconds
    }


if __name__ == '__main__':
    import sys
    from visualize import plot_delta, plot_cost_matrix
    import csv

    # Parse optional arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('video1')
    parser.add_argument('video2')
    parser.add_argument('output_dir', nargs='?', default='output/coarse_to_fine')
    parser.add_argument('--coarse-fps', type=float, default=0.5)
    parser.add_argument('--fine-fps', type=float, default=1.0)
    parser.add_argument('--fine-band-width', type=int, default=10)
    parser.add_argument('--embedding-dtype', default='float32', choices=['float32', 'float16'],
                        help='Embedding dtype for memory efficiency (default: float32)')
    parser.add_argument('--crop-top', type=float, default=0.5,
                        help='Keep top fraction of frame, 0.5=top half (default: 0.5)')
    parser.add_argument('--no-crop', action='store_true',
                        help='Disable frame cropping (use full frame)')
    parser.add_argument('--smooth-window', type=float, default=10.0,
                        help='Smoothing window in seconds (default: 10)')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    video1_path = args.video1
    video2_path = args.video2
    output_dir = Path(args.output_dir)

    crop_top_fraction = None if args.no_crop else args.crop_top

    print(f"Configuration:")
    print(f"  Coarse FPS: {args.coarse_fps}")
    print(f"  Fine FPS: {args.fine_fps}")
    print(f"  Fine band width: ±{args.fine_band_width} frames")
    print(f"  Embedding dtype: {args.embedding_dtype}")
    print(f"  Crop top: {f'{crop_top_fraction:.0%}' if crop_top_fraction else 'disabled'}")
    print(f"  Device: {args.device}")
    print()

    # Run coarse-to-fine alignment
    cost_matrix, path, ts1, ts2 = align_coarse_to_fine(
        video1_path, video2_path,
        coarse_fps=args.coarse_fps,
        fine_fps=args.fine_fps,
        fine_band_width=args.fine_band_width,
        device=args.device,
        output_dir=output_dir,
        crop_top_fraction=crop_top_fraction,
        embedding_dtype=args.embedding_dtype,
    )

    # Compute delta
    print(f"\nComputing time delta...")
    delta_result_raw = compute_delta_from_timestamps(path, ts1, ts2, smooth=True, smooth_window_sec=args.smooth_window)
    # Rename keys to match visualize.py expectations
    delta_result = {
        'time_vid2': delta_result_raw['timestamps_vid2'],
        'delta_seconds': delta_result_raw['delta']
    }

    # Save and visualize results
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Output] Saving results to {output_dir}/...")

    # Save delta CSV
    csv_path = output_dir / "delta.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_vid2_seconds', 'delta_seconds'])
        for t, d in zip(delta_result['time_vid2'], delta_result['delta_seconds']):
            writer.writerow([t, d])
    print(f"         delta.csv")

    # Plot delta
    delta_png = output_dir / "delta.png"
    plot_delta(delta_result, str(delta_png))
    print(f"         delta.png")

    # Plot heatmap
    heatmap_png = output_dir / "heatmap.png"
    plot_cost_matrix(cost_matrix, path, str(heatmap_png))
    print(f"         heatmap.png")

    print(f"\n{'='*70}")
    print("✓ Analysis complete!")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
