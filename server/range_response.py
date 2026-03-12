"""HTTP Range header parsing and file serving utilities."""

from typing import Generator


def parse_range(header: str | None, file_size: int) -> tuple[int, int]:
    """Parse an HTTP Range header into (start, end) byte offsets.

    Args:
        header: Range header value (e.g. "bytes=0-499") or None
        file_size: Total file size in bytes

    Returns:
        Tuple of (start, end) inclusive byte offsets
    """
    if header is None:
        return (0, file_size - 1)

    # Strip "bytes=" prefix
    range_spec = header.replace("bytes=", "")

    if range_spec.startswith("-"):
        # Suffix range: bytes=-200 means last 200 bytes
        suffix_len = int(range_spec[1:])
        start = max(0, file_size - suffix_len)
        return (start, file_size - 1)

    parts = range_spec.split("-", 1)
    start = int(parts[0])

    if parts[1] == "":
        # Open-ended: bytes=500-
        return (start, file_size - 1)

    end = min(int(parts[1]), file_size - 1)
    return (start, end)


def file_iterator(path: str, start: int, end: int, chunk_size: int = 65536) -> Generator[bytes, None, None]:
    """Yield chunks of a file from start to end (inclusive).

    Args:
        path: File path
        start: Start byte offset (inclusive)
        end: End byte offset (inclusive)
        chunk_size: Bytes per chunk
    """
    with open(path, "rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            read_size = min(chunk_size, remaining)
            data = f.read(read_size)
            if not data:
                break
            remaining -= len(data)
            yield data
