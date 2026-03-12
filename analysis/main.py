#!/usr/bin/env python3
"""
Vid-Sync: MTB Helmet Cam Comparison Tool

Compares two helmet-cam videos of the same mountain bike trail to compute
continuous time-delta showing how far ahead or behind rider 2 is relative to rider 1.
"""

import argparse
import os
import sys
import numpy as np

from extract import extract_frames
from embeddings import compute_embeddings
from streaming import extract_and_embed_streaming
from align import compute_cost_matrix, align_sequences
from delta import compute_delta
from visualize import plot_delta, plot_cost_matrix


def main():
    parser = argparse.ArgumentParser(
        description='Compare two helmet-cam videos to compute rider time delta',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  %(prog)s --vid1 rider1.mp4 --vid2 rider2.mp4
  %(prog)s --vid1 rider1.mp4 --vid2 rider2.mp4 --sample-fps 1 --output-dir results/
        """
    )

    parser.add_argument('--vid1', required=True, help='Path to video 1 (reference)')
    parser.add_argument('--vid2', required=True, help='Path to video 2 (comparison)')
    parser.add_argument('--sample-fps', type=float, default=2.0,
                       help='Sampling rate in frames per second (default: 2.0)')
    parser.add_argument('--output-dir', default='.',
                       help='Output directory for results (default: current directory)')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                       help='Device for DINOv2 inference (default: cpu)')
    parser.add_argument('--band-fraction', type=float, default=0.2,
                       help='DTW band constraint fraction (default: 0.2)')
    parser.add_argument('--embedding-dtype', default='float32', choices=['float32', 'float16'],
                       help='Embedding dtype for memory efficiency (default: float32)')
    parser.add_argument('--method', default='dinov2', choices=['dinov2', 'feature-matching'],
                       help='Alignment method: dinov2 embeddings or feature-matching (default: dinov2)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.vid1):
        print(f"Error: Video 1 not found: {args.vid1}", file=sys.stderr)
        return 1

    if not os.path.exists(args.vid2):
        print(f"Error: Video 2 not found: {args.vid2}", file=sys.stderr)
        return 1

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    method_name = "Feature Matching" if args.method == 'feature-matching' else "DINOv2 Embeddings"
    print(f"Vid-Sync: MTB Helmet Cam Comparison ({method_name})")
    print("=" * 50)

    if args.method == 'feature-matching':
        # Use SuperPoint + LightGlue feature matching
        from feature_matching import align_with_feature_matching

        cost_matrix, path, timestamps1, timestamps2 = align_with_feature_matching(
            args.vid1,
            args.vid2,
            sample_fps=args.sample_fps,
            band_fraction=args.band_fraction,
            device=args.device
        )

    else:
        # Use DINOv2 embeddings (original method)
        # Step 1 & 2: Stream frames and compute embeddings (memory-efficient)
        print(f"\n[1/4] Streaming video 1: extracting frames at {args.sample_fps} fps and computing embeddings...")
        embeddings1, timestamps1 = extract_and_embed_streaming(
            args.vid1,
            sample_fps=args.sample_fps,
            device=args.device,
            dtype=args.embedding_dtype
        )
        print(f"      Shape: {embeddings1.shape}, dtype: {embeddings1.dtype}")

        print(f"\n[2/4] Streaming video 2: extracting frames at {args.sample_fps} fps and computing embeddings...")
        embeddings2, timestamps2 = extract_and_embed_streaming(
            args.vid2,
            sample_fps=args.sample_fps,
            device=args.device,
            dtype=args.embedding_dtype
        )
        print(f"      Shape: {embeddings2.shape}, dtype: {embeddings2.dtype}")

        # Step 3: Compute alignment
        print(f"\n[3/4] Computing DTW alignment...")
        cost_matrix = compute_cost_matrix(embeddings1, embeddings2)
        print(f"      Cost matrix shape: {cost_matrix.shape}")

        path = align_sequences(cost_matrix, band_fraction=args.band_fraction)
        print(f"      Path length: {len(path)}")

    # Step 4: Compute delta
    step_num = "[5/5]" if args.method == 'feature-matching' else "[4/4]"
    print(f"\n{step_num} Computing time delta...")
    delta_result = compute_delta(path, args.sample_fps, smooth=True)

    mean_delta = np.mean(delta_result["delta_seconds"])
    std_delta = np.std(delta_result["delta_seconds"])
    min_delta = np.min(delta_result["delta_seconds"])
    max_delta = np.max(delta_result["delta_seconds"])

    print(f"      Mean delta: {mean_delta:.2f}s (± {std_delta:.2f}s)")
    print(f"      Range: [{min_delta:.2f}s, {max_delta:.2f}s]")

    # Interpret result
    if mean_delta > 0:
        print(f"      → Video 2 is ahead by an average of {abs(mean_delta):.2f}s")
    elif mean_delta < 0:
        print(f"      → Video 2 is behind by an average of {abs(mean_delta):.2f}s")
    else:
        print(f"      → Videos are synchronized (tied)")

    # Save results
    print(f"\n[Output] Saving results to {args.output_dir}/...")

    # Save delta CSV
    csv_path = os.path.join(args.output_dir, "delta.csv")
    with open(csv_path, 'w') as f:
        f.write("time_vid2_seconds,delta_seconds\n")
        for t, d in zip(delta_result["time_vid2"], delta_result["delta_seconds"]):
            f.write(f"{t},{d}\n")
    print(f"         delta.csv")

    # Save delta plot
    delta_plot_path = os.path.join(args.output_dir, "delta.png")
    plot_delta(delta_result, delta_plot_path)
    print(f"         delta.png")

    # Save cost matrix heatmap
    heatmap_path = os.path.join(args.output_dir, "heatmap.png")
    plot_cost_matrix(cost_matrix, path, heatmap_path)
    print(f"         heatmap.png")

    print("\n" + "=" * 50)
    print("✓ Analysis complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
