"""Pipeline wrapper with progress reporting for the server."""

import logging
import os
import sys

logger = logging.getLogger("vid-sync.pipeline")

# Add analysis directory to path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))

from streaming import extract_and_embed_streaming
from align import compute_cost_matrix, align_sequences
from delta import compute_delta


def run_pipeline(vid1_path: str, vid2_path: str, output_dir: str,
                 sample_fps: float = 2.0, band_fraction: float = 0.2,
                 device: str = "cpu", progress_callback=None):
    """Run the full analysis pipeline with progress reporting.

    Args:
        vid1_path: Path to video 1
        vid2_path: Path to video 2
        output_dir: Directory for output files
        sample_fps: Sampling rate in fps
        band_fraction: DTW band constraint fraction
        device: Device for inference
        progress_callback: Called with dict {"stage", "frames", "total", "percent"}
    """
    def _report(stage, frames=0, total=0, percent=0.0):
        if progress_callback:
            progress_callback({"stage": stage, "frames": frames, "total": total, "percent": percent})

    # Stage 1: Embed video 1
    logger.info("Embedding video 1: %s (fps=%.1f)", vid1_path, sample_fps)
    def _cb_vid1(frames_processed, total_estimate):
        pct = (frames_processed / max(total_estimate, 1)) * 25  # 0-25%
        _report("embedding_vid1", frames=frames_processed, total=total_estimate, percent=pct)

    embeddings1, timestamps1 = extract_and_embed_streaming(
        vid1_path, sample_fps=sample_fps, device=device, progress_callback=_cb_vid1
    )
    logger.info("Video 1 done: %d embeddings", len(timestamps1))

    # Stage 2: Embed video 2
    logger.info("Embedding video 2: %s", vid2_path)
    def _cb_vid2(frames_processed, total_estimate):
        pct = 25 + (frames_processed / max(total_estimate, 1)) * 25  # 25-50%
        _report("embedding_vid2", frames=frames_processed, total=total_estimate, percent=pct)

    embeddings2, timestamps2 = extract_and_embed_streaming(
        vid2_path, sample_fps=sample_fps, device=device, progress_callback=_cb_vid2
    )
    logger.info("Video 2 done: %d embeddings", len(timestamps2))

    # Stage 3: Align
    logger.info("Computing cost matrix (%d x %d) and DTW alignment (band=%.2f)",
                len(timestamps1), len(timestamps2), band_fraction)
    _report("aligning", percent=50)
    cost_matrix = compute_cost_matrix(embeddings1, embeddings2)
    path = align_sequences(cost_matrix, band_fraction=band_fraction)
    logger.info("Alignment done: path length %d", len(path))
    _report("aligning", percent=75)

    # Stage 4: Compute delta
    logger.info("Computing delta curve")
    _report("computing_delta", percent=80)
    delta_result = compute_delta(path, sample_fps, smooth=True)

    # Write delta.csv
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "delta.csv")
    with open(csv_path, "w") as f:
        f.write("time_vid2_seconds,delta_seconds\n")
        for t, d in zip(delta_result["time_vid2"], delta_result["delta_seconds"]):
            f.write(f"{t},{d}\n")
    logger.info("Wrote %s (%d rows)", csv_path, len(delta_result["time_vid2"]))

    _report("complete", percent=100)

    return delta_result
