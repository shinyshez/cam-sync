import torch
import numpy as np
from torchvision import transforms


# Global cache for the model
_model_cache = None


def _get_dinov2_model(device: str = "cpu"):
    """Load and cache DINOv2 model."""
    global _model_cache
    if _model_cache is None:
        _model_cache = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        _model_cache.eval()
    return _model_cache.to(device)


def compute_embeddings(frames: list[np.ndarray], batch_size: int = 32, device: str = "cpu", dtype: str = "float32") -> np.ndarray:
    """Compute DINOv2 embeddings for a list of frames.

    Args:
        frames: List of numpy arrays with shape (H, W, 3) in BGR format
        batch_size: Batch size for processing (default: 32)
        device: Device to run inference on (default: "cpu")
        dtype: Output dtype, either "float32" or "float16" (default: "float32")

    Returns:
        Numpy array of shape (N, 384) with L2-normalized embeddings
    """
    model = _get_dinov2_model(device)

    # Define preprocessing transform
    # DINOv2 expects RGB images normalized with ImageNet stats
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    embeddings = []

    # Process in batches
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]

        # Convert BGR to RGB and preprocess
        batch_tensors = []
        for frame in batch_frames:
            # OpenCV uses BGR, convert to RGB
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

        embeddings.append(batch_embeddings)

    # Concatenate all batches
    all_embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)

    # Convert to requested dtype
    if dtype == "float16":
        all_embeddings = all_embeddings.astype(np.float16)
    elif dtype != "float32":
        raise ValueError(f"Unsupported dtype: {dtype}. Use 'float32' or 'float16'")

    return all_embeddings
