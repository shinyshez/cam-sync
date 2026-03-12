import pytest
import os
import tempfile
from unittest.mock import patch, AsyncMock

from fastapi.testclient import TestClient
from server.app import create_app


@pytest.fixture
def tmp_output(tmp_path):
    """Create a temp output dir with a delta.csv."""
    csv_path = tmp_path / "delta.csv"
    csv_path.write_text("time_vid2_seconds,delta_seconds\n0.0,0.5\n1.0,1.0\n")
    return str(tmp_path)


@pytest.fixture
def mode_c_app(tmp_path, tmp_output):
    """App in Mode C (pipeline done, results ready)."""
    vid1 = tmp_path / "vid1.mp4"
    vid2 = tmp_path / "vid2.mp4"
    vid1.write_bytes(b"\x00" * 100)
    vid2.write_bytes(b"\x00" * 100)
    app = create_app(
        mode="cli",
        ready=True,
        vid1_path=str(vid1),
        vid2_path=str(vid2),
        output_dir=tmp_output,
    )
    return TestClient(app)


@pytest.fixture
def mode_d_app():
    """App in Mode D (web app, not ready)."""
    app = create_app(mode="webapp", ready=False)
    return TestClient(app)


class TestStatus:
    def test_mode_c_status(self, mode_c_app):
        r = mode_c_app.get("/api/status")
        assert r.status_code == 200
        data = r.json()
        assert data["mode"] == "cli"
        assert data["ready"] is True

    def test_mode_d_status(self, mode_d_app):
        r = mode_d_app.get("/api/status")
        assert r.status_code == 200
        data = r.json()
        assert data["mode"] == "webapp"
        assert data["ready"] is False


class TestVideoServing:
    def test_video_range_206(self, mode_c_app):
        r = mode_c_app.get("/api/video/1", headers={"Range": "bytes=0-49"})
        assert r.status_code == 206
        assert len(r.content) == 50
        assert "Content-Range" in r.headers

    def test_video_full(self, mode_c_app):
        r = mode_c_app.get("/api/video/1")
        assert r.status_code == 200
        assert len(r.content) == 100

    def test_video_invalid_n(self, mode_c_app):
        r = mode_c_app.get("/api/video/3")
        assert r.status_code == 404


class TestDeltaCsv:
    def test_serves_csv(self, mode_c_app):
        r = mode_c_app.get("/api/delta.csv")
        assert r.status_code == 200
        assert "time_vid2_seconds" in r.text

    def test_not_ready(self, mode_d_app):
        r = mode_d_app.get("/api/delta.csv")
        assert r.status_code == 404


class TestUpload:
    def test_upload_creates_job(self, mode_d_app):
        r = mode_d_app.post(
            "/api/upload",
            files=[
                ("files", ("vid1.mp4", b"\x00" * 50, "video/mp4")),
                ("files", ("vid2.mp4", b"\x00" * 50, "video/mp4")),
            ],
        )
        assert r.status_code == 200
        data = r.json()
        assert "job_id" in data

    def test_upload_needs_two_files(self, mode_d_app):
        r = mode_d_app.post(
            "/api/upload",
            files=[("files", ("vid1.mp4", b"\x00" * 50, "video/mp4"))],
        )
        assert r.status_code == 400


class TestRedirect:
    def test_root_redirects(self, mode_c_app):
        r = mode_c_app.get("/", follow_redirects=False)
        assert r.status_code == 307
        assert "/viewer.html" in r.headers["location"]
