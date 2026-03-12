"""Feature matching based frame extraction and alignment.

This module provides memory-efficient processing using SuperPoint + LightGlue
for lighting-invariant place recognition, replacing DINOv2 embeddings.
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Dict


def extract_keypoints_streaming(video_path: str,
                                sample_fps: float = 2.0,
                                max_keypoints: int = 2048,
                                device: str = "cpu",
                                crop_top_fraction: float | None = None) -> Tuple[List[Dict], List[float]]:
    """Extract frames and compute keypoints/descriptors in streaming fashion.

    Instead of storing frames, this function:
    1. Extracts a frame
    2. Computes keypoints and descriptors
    3. Discards the frame
    4. Repeats for next frame

    Args:
        video_path: Path to video file
        sample_fps: Frames per second to sample (default: 2.0)
        max_keypoints: Maximum keypoints per frame (default: 2048)
        device: Device for computation (default: "cpu")
        crop_top_fraction: If set, keep only the top fraction of each frame (e.g. 0.5 = top half)

    Returns:
        Tuple of (features_list, timestamps):
        - features_list: List of dicts with 'keypoints', 'descriptors', 'image_size'
        - timestamps: List of float timestamps in seconds
    """
    from lightglue import SuperPoint
    from lightglue.utils import numpy_image_to_torch

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps == 0:
        raise ValueError(f"Invalid FPS in video: {video_path}")

    # Calculate frame interval for sampling
    frame_interval = source_fps / sample_fps

    # Load extractor
    extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)

    features_list = []
    timestamps = []
    frame_idx = 0
    frames_processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if this frame should be sampled
        if frame_idx % int(round(frame_interval)) == 0:
            timestamp = frame_idx / source_fps

            # Crop to top fraction if requested
            if crop_top_fraction is not None:
                h = frame.shape[0]
                frame = frame[:int(h * crop_top_fraction), :, :]

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to torch tensor
            image_tensor = numpy_image_to_torch(rgb_frame).to(device)

            # Extract keypoints and descriptors
            with torch.no_grad():
                feats = extractor.extract(image_tensor)

            # Store features (move to CPU to free GPU memory)
            num_kpts = feats['keypoints'][0].shape[0]
            features = {
                'keypoints': feats['keypoints'][0].cpu(),
                'descriptors': feats['descriptors'][0].cpu(),
                'image_size': torch.tensor([frame.shape[0], frame.shape[1]])
            }

            features_list.append(features)
            timestamps.append(timestamp)
            frames_processed += 1

            # Progress output every 10 frames
            if frames_processed % 10 == 0:
                print(f"      Processed {frames_processed} frames, {num_kpts} keypoints at {timestamp:.1f}s", flush=True)

            # Frame is discarded here (garbage collected)

        frame_idx += 1

    cap.release()

    return features_list, timestamps


def compute_cost_from_matches(num_matches: int, scale: float = 250.0) -> float:
    """Convert number of matches to a cost value for DTW.

    Args:
        num_matches: Number of geometric matches found
        scale: Scaling factor (matches at this value give cost ~0)

    Returns:
        Cost value in range [0, 2]:
        - 0 matches → cost = 2.0 (maximum, dissimilar)
        - scale matches → cost = 0.0 (minimum, very similar)
        - Linear interpolation between
    """
    # Linear cost: 2.0 - min(2.0, num_matches / scale)
    # This gives cost = 0 when num_matches >= 2*scale
    return max(0.0, 2.0 - (num_matches / scale))


def compute_cost_matrix_from_features(features1: List[Dict],
                                      features2: List[Dict],
                                      band_fraction: float = 0.2,
                                      device: str = "cpu") -> np.ndarray:
    """Compute cost matrix using feature matching within DTW band.

    Only computes matches for frame pairs within the Sakoe-Chiba band
    to avoid O(N*M) computation.

    Args:
        features1: List of feature dicts for video 1
        features2: List of feature dicts for video 2
        band_fraction: Sakoe-Chiba band width as fraction of sequence length
        device: Device for matching

    Returns:
        Cost matrix of shape (len(features1), len(features2))
    """
    from lightglue import LightGlue

    n = len(features1)
    m = len(features2)

    # Initialize cost matrix with high cost (no match)
    cost_matrix = np.full((n, m), 2.0, dtype=np.float32)

    # Load matcher
    matcher = LightGlue(features='superpoint').eval().to(device)

    # Compute band width
    band_width = int(max(n, m) * band_fraction)

    print(f"Computing cost matrix with feature matching...")
    print(f"  Matrix size: {n}×{m}")
    print(f"  Band width: ±{band_width} frames")

    # Only compute within band
    pairs_computed = 0
    for i in range(n):
        j_start = max(0, i - band_width)
        j_end = min(m, i + band_width + 1)

        for j in range(j_start, j_end):
            # Match features between frames i and j
            feats0 = {k: v.to(device).unsqueeze(0) for k, v in features1[i].items() if k != 'image_size'}
            feats1 = {k: v.to(device).unsqueeze(0) for k, v in features2[j].items() if k != 'image_size'}

            with torch.no_grad():
                matches = matcher({'image0': feats0, 'image1': feats1})

            # Extract number of matches
            # LightGlue returns matches as a list containing a tensor
            if 'matches' in matches:
                match_obj = matches['matches']
                if isinstance(match_obj, list) and len(match_obj) > 0:
                    # Get the tensor from the list
                    match_tensor = match_obj[0]
                    if isinstance(match_tensor, torch.Tensor) and match_tensor.ndim >= 1:
                        num_matches = match_tensor.shape[0]
                    else:
                        num_matches = 0
                elif isinstance(match_obj, torch.Tensor):
                    num_matches = match_obj.shape[0] if match_obj.ndim == 2 else match_obj.shape[1]
                else:
                    num_matches = 0
            else:
                num_matches = 0

            # Convert to cost
            cost_matrix[i, j] = compute_cost_from_matches(num_matches)

            pairs_computed += 1

        # Progress reporting every 10 frames
        if (i + 1) % 10 == 0:
            percent = 100 * (i + 1) / n
            print(f"  Processed {i+1}/{n} frames ({percent:.1f}%, {pairs_computed} pairs)", flush=True)

    print(f"  Total pairs computed: {pairs_computed}")

    return cost_matrix


def align_with_feature_matching(video1_path: str,
                                video2_path: str,
                                sample_fps: float = 2.0,
                                band_fraction: float = 0.2,
                                device: str = "cpu") -> Tuple[np.ndarray, List[Tuple[int, int]], List[float], List[float]]:
    """Complete feature-matching based alignment pipeline.

    Args:
        video1_path: Path to first video
        video2_path: Path to second video
        sample_fps: Sampling rate in fps
        band_fraction: DTW band fraction
        device: Device for computation

    Returns:
        Tuple of (cost_matrix, path, timestamps1, timestamps2)
    """
    from align import align_sequences

    print(f"\n[1/4] Extracting keypoints from video 1 at {sample_fps} fps...")
    features1, timestamps1 = extract_keypoints_streaming(
        video1_path, sample_fps=sample_fps, device=device
    )
    print(f"      Extracted {len(features1)} frames with keypoints")
    if len(features1) > 0:
        print(f"      Average keypoints per frame: {np.mean([f['keypoints'].shape[0] for f in features1]):.0f}")

    print(f"\n[2/4] Extracting keypoints from video 2 at {sample_fps} fps...")
    features2, timestamps2 = extract_keypoints_streaming(
        video2_path, sample_fps=sample_fps, device=device
    )
    print(f"      Extracted {len(features2)} frames with keypoints")
    if len(features2) > 0:
        print(f"      Average keypoints per frame: {np.mean([f['keypoints'].shape[0] for f in features2]):.0f}")

    print(f"\n[3/4] Computing cost matrix with feature matching...")
    cost_matrix = compute_cost_matrix_from_features(
        features1, features2, band_fraction=band_fraction, device=device
    )
    print(f"      Cost matrix shape: {cost_matrix.shape}")
    print(f"      Cost range: [{cost_matrix.min():.3f}, {cost_matrix.max():.3f}]")

    print(f"\n[4/4] Computing DTW alignment...")
    path = align_sequences(cost_matrix, band_fraction=band_fraction)
    print(f"      Path length: {len(path)}")

    return cost_matrix, path, timestamps1, timestamps2
