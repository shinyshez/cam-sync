import numpy as np

# Try dtaidistance first (faster), fall back to dtw-python
try:
    from dtaidistance import dtw as dtw_dtai
    HAVE_DTAIDISTANCE = True
except ImportError:
    HAVE_DTAIDISTANCE = False

try:
    from dtw import dtw as dtw_python
    HAVE_DTW_PYTHON = True
except ImportError:
    HAVE_DTW_PYTHON = False


def compute_cost_matrix(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance matrix between two embedding sequences.

    Args:
        embeddings1: Array of shape (N1, D) with L2-normalized embeddings
        embeddings2: Array of shape (N2, D) with L2-normalized embeddings

    Returns:
        Cost matrix of shape (N1, N2) where cost[i,j] = 1 - cosine_similarity(i, j)
    """
    # Since embeddings are L2-normalized, cosine similarity is just the dot product
    # Cosine distance = 1 - cosine similarity
    cosine_similarity = embeddings1 @ embeddings2.T
    cost_matrix = 1.0 - cosine_similarity

    # Clip to handle floating point precision issues (can get small negative values)
    cost_matrix = np.clip(cost_matrix, 0, 2)

    return cost_matrix


def align_sequences(cost_matrix: np.ndarray, band_fraction: float = 0.2) -> list[tuple[int, int]]:
    """Align two sequences using Dynamic Time Warping with Sakoe-Chiba band constraint.

    Args:
        cost_matrix: Cost matrix of shape (N1, N2)
        band_fraction: Fraction of sequence length to use as band width (default: 0.2)

    Returns:
        List of (i, j) tuples representing the alignment path from (0, 0) to (N1-1, N2-1)
    """
    n, m = cost_matrix.shape

    # Calculate Sakoe-Chiba band width
    band_width = int(band_fraction * max(n, m))

    # Convert to float64
    cost_matrix_f64 = cost_matrix.astype(np.float64)

    # Try dtaidistance first (faster C implementation)
    if HAVE_DTAIDISTANCE:
        try:
            # dtaidistance expects cost matrix in specific format
            # Use warping_path with custom distance matrix
            path_dtai = dtw_dtai.warping_path(
                cost_matrix_f64,
                window=band_width if band_width > 0 else None,
                use_c=True  # Use fast C implementation
            )
            # Convert to list of tuples
            path = [(i, j) for i, j in path_dtai]
            return path
        except Exception as e:
            # Fall back to dtw-python if dtaidistance fails
            if not HAVE_DTW_PYTHON:
                raise RuntimeError(f"dtaidistance failed and dtw-python not available: {e}")

    # Fall back to dtw-python library
    if HAVE_DTW_PYTHON:
        if band_width > 0:
            alignment = dtw_python(cost_matrix_f64, window_type='sakoechiba',
                                  window_args={'window_size': band_width})
        else:
            alignment = dtw_python(cost_matrix_f64)

        path = list(zip(alignment.index1, alignment.index2))
        return path

    raise RuntimeError("No DTW library available. Install dtaidistance or dtw-python")
