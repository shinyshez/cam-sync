"""Streaming frame extraction and embedding computation.

This module provides memory-efficient processing by extracting and embedding
frames one at a time, never holding all decoded frames in memory simultaneously.
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from embeddings import _get_dinov2_model


def extract_and_embed_streaming(video_path: str,
                                sample_fps: float = 2.0,
                                device: str = "cpu",
                                dtype: str = "float32",
                                batch_size: int = 32,
                                crop_top_fraction: float | None = None,
                                progress_callback=None) -> tuple[np.ndarray, list[float]]:
    """Extract frames and compute embeddings in streaming fashion (memory-efficient).

    Instead of loading all frames into memory, this function:
    1. Extracts a frame
    2. Computes its embedding
    3. Discards the frame
    4. Repeats for next frame

    This reduces peak memory from ~6 GB to ~100 MB for typical videos.

    Args:
        video_path: Path to video file
        sample_fps: Frames per second to sample (default: 2.0)
        device: Device for DINOv2 inference (default: "cpu")
        dtype: Output dtype, "float32" or "float16" (default: "float32")
        batch_size: Number of frames to process together (default: 32)
        crop_top_fraction: If set, keep only the top fraction of each frame (e.g. 0.5 = top half)

    Returns:
        Tuple of (embeddings, timestamps):
        - embeddings: Array of shape (N, 384) with L2-normalized embeddings
        - timestamps: List of N float timestamps in seconds
    """
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

    # Estimate total sampled frames for progress reporting
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    estimated_total = int(total_frames / max(1, int(round(frame_interval))))

    # Load model
    model = _get_dinov2_model(device)

    # Define preprocessing transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    embeddings_list = []
    timestamps = []
    frame_idx = 0
    batch_frames = []
    batch_timestamps = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if this frame should be sampled
        if frame_idx % int(round(frame_interval)) == 0:
            timestamp = frame_idx / source_fps
            if crop_top_fraction is not None:
                h = frame.shape[0]
                frame = frame[:int(h * crop_top_fraction), :, :]
            batch_frames.append(frame)
            batch_timestamps.append(timestamp)

            # Process batch when full
            if len(batch_frames) >= batch_size:
                batch_embeddings = _process_batch(batch_frames, model, transform, device, dtype)
                embeddings_list.append(batch_embeddings)
                timestamps.extend(batch_timestamps)

                if progress_callback:
                    progress_callback(frames_processed=len(timestamps), total_estimate=estimated_total)

                # Clear batch (frames are discarded here)
                batch_frames = []
                batch_timestamps = []

        frame_idx += 1

    # Process remaining frames
    if batch_frames:
        batch_embeddings = _process_batch(batch_frames, model, transform, device, dtype)
        embeddings_list.append(batch_embeddings)
        timestamps.extend(batch_timestamps)

        if progress_callback:
            progress_callback(frames_processed=len(timestamps), total_estimate=estimated_total)

    cap.release()

    # Concatenate all embeddings
    if not embeddings_list:
        return np.array([]).reshape(0, 384), []

    all_embeddings = np.concatenate(embeddings_list, axis=0)

    return all_embeddings, timestamps


def _process_batch(frames: list[np.ndarray],
                   model,
                   transform,
                   device: str,
                   dtype: str) -> np.ndarray:
    """Process a batch of frames and return embeddings.

    Args:
        frames: List of BGR frames (H, W, 3)
        model: DINOv2 model
        transform: Preprocessing transform
        device: Device for inference
        dtype: Output dtype

    Returns:
        Embeddings array of shape (batch_size, 384)
    """
    # Convert BGR to RGB and preprocess
    batch_tensors = []
    for frame in frames:
        rgb_frame = frame[:, :, ::-1].copy()
        tensor = transform(rgb_frame)
        batch_tensors.append(tensor)

    # Stack into batch
    batch = torch.stack(batch_tensors).to(device)

    # Extract embeddings
    with torch.no_grad():
        output = model(batch)

    # Convert to numpy
    batch_embeddings = output.cpu().numpy()

    # L2 normalize
    norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
    batch_embeddings = batch_embeddings / norms

    # Cast to float32 first, then to requested dtype
    batch_embeddings = batch_embeddings.astype(np.float32)

    if dtype == "float16":
        batch_embeddings = batch_embeddings.astype(np.float16)
    elif dtype != "float32":
        raise ValueError(f"Unsupported dtype: {dtype}")

    return batch_embeddings
