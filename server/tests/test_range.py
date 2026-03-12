import pytest
import tempfile
import os
from server.range_response import parse_range, file_iterator


class TestParseRange:
    def test_full_range(self):
        """bytes=0-499 with file_size=1000 → (0, 499)"""
        assert parse_range("bytes=0-499", 1000) == (0, 499)

    def test_open_end(self):
        """bytes=500- with file_size=1000 → (500, 999)"""
        assert parse_range("bytes=500-", 1000) == (500, 999)

    def test_suffix_range(self):
        """bytes=-200 with file_size=1000 → (800, 999)"""
        assert parse_range("bytes=-200", 1000) == (800, 999)

    def test_none_returns_full(self):
        """No range header → (0, file_size-1)"""
        assert parse_range(None, 1000) == (0, 999)

    def test_clamps_to_file_size(self):
        """End beyond file size gets clamped."""
        assert parse_range("bytes=0-5000", 1000) == (0, 999)


class TestFileIterator:
    def test_reads_correct_bytes(self):
        """file_iterator yields exactly the requested byte range."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"0123456789" * 100)  # 1000 bytes
            path = f.name

        try:
            data = b"".join(file_iterator(path, 10, 19, chunk_size=8))
            assert data == b"0123456789"
            assert len(data) == 10
        finally:
            os.unlink(path)

    def test_reads_to_end(self):
        """file_iterator handles reading to end of file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"abcdef")
            path = f.name

        try:
            data = b"".join(file_iterator(path, 3, 5, chunk_size=64))
            assert data == b"def"
        finally:
            os.unlink(path)
