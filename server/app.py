"""FastAPI application for vid-sync server."""

import asyncio
import logging
import os
import uuid

from fastapi import FastAPI, Request, UploadFile, File, WebSocket, HTTPException
from fastapi.responses import (
    RedirectResponse, StreamingResponse, FileResponse, Response, JSONResponse,
)
from fastapi.staticfiles import StaticFiles

from server.range_response import parse_range, file_iterator

logger = logging.getLogger("vid-sync")


def create_app(mode: str = "webapp", ready: bool = False,
               vid1_path: str = "", vid2_path: str = "",
               output_dir: str = "") -> FastAPI:
    """Create and configure the FastAPI app.

    Args:
        mode: "cli" (Mode C) or "webapp" (Mode D)
        ready: Whether results are available
        vid1_path: Path to video 1 (Mode C)
        vid2_path: Path to video 2 (Mode C)
        output_dir: Directory containing results
    """
    app = FastAPI()

    # App state
    app.state.mode = mode
    app.state.ready = ready
    app.state.vid1_path = vid1_path
    app.state.vid2_path = vid2_path
    app.state.output_dir = output_dir
    app.state.jobs = {}  # job_id -> {vid1_path, vid2_path, output_dir, queue}

    # Serve viewer.html from the viewer/ directory
    viewer_dir = os.path.join(os.path.dirname(__file__), "..", "viewer")

    @app.get("/")
    async def root():
        return RedirectResponse(url="/viewer.html")

    @app.get("/viewer.html")
    async def serve_viewer():
        viewer_path = os.path.join(viewer_dir, "viewer.html")
        if os.path.exists(viewer_path):
            return FileResponse(viewer_path, media_type="text/html")
        raise HTTPException(status_code=404)

    @app.get("/api/status")
    async def status():
        return {
            "mode": app.state.mode,
            "ready": app.state.ready,
        }

    @app.get("/api/video/{n}")
    async def serve_video(n: int, request: Request):
        if n == 1:
            path = app.state.vid1_path
        elif n == 2:
            path = app.state.vid2_path
        else:
            raise HTTPException(status_code=404, detail="Video must be 1 or 2")

        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Video not found")

        file_size = os.path.getsize(path)
        range_header = request.headers.get("range")

        if range_header:
            start, end = parse_range(range_header, file_size)
            content_length = end - start + 1
            headers = {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
            }
            return StreamingResponse(
                file_iterator(path, start, end),
                status_code=206,
                media_type="video/mp4",
                headers=headers,
            )

        return Response(
            content=open(path, "rb").read(),
            media_type="video/mp4",
            headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
        )

    @app.get("/api/delta.csv")
    async def serve_delta():
        if not app.state.ready or not app.state.output_dir:
            raise HTTPException(status_code=404, detail="Results not ready")
        csv_path = os.path.join(app.state.output_dir, "delta.csv")
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=404, detail="delta.csv not found")
        return FileResponse(csv_path, media_type="text/csv")

    @app.post("/api/upload")
    async def upload(files: list[UploadFile] = File(...)):
        if len(files) < 2:
            return JSONResponse(status_code=400, content={"error": "Need exactly 2 video files"})

        job_id = str(uuid.uuid4())[:8]
        job_dir = os.path.join("/tmp", "vid-sync-jobs", job_id)
        os.makedirs(job_dir, exist_ok=True)

        v1_path = os.path.join(job_dir, "vid1.mp4")
        v2_path = os.path.join(job_dir, "vid2.mp4")

        for i, f in enumerate(files[:2]):
            dest = v1_path if i == 0 else v2_path
            with open(dest, "wb") as out:
                content = await f.read()
                out.write(content)
                logger.info("Uploaded %s (%d bytes) → %s", f.filename, len(content), dest)

        queue = asyncio.Queue()
        app.state.jobs[job_id] = {
            "vid1_path": v1_path,
            "vid2_path": v2_path,
            "output_dir": job_dir,
            "queue": queue,
        }

        # Set video paths eagerly so /api/video/{n} works as soon as upload completes
        app.state.vid1_path = v1_path
        app.state.vid2_path = v2_path
        app.state.output_dir = job_dir

        logger.info("Created job %s in %s", job_id, job_dir)
        return {"job_id": job_id}

    @app.post("/api/run/{job_id}")
    async def run_job(job_id: str, request: Request):
        job = app.state.jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # Read optional params from request body
        params = {}
        try:
            params = await request.json()
        except Exception:
            pass
        sample_fps = float(params.get("sample_fps", 2.0))
        band_fraction = float(params.get("band_fraction", 0.2))

        from server.pipeline import run_pipeline

        logger.info("Starting pipeline for job %s (fps=%.1f, band=%.2f)", job_id, sample_fps, band_fraction)

        async def _run():
            def progress_cb(msg):
                stage = msg.get("stage", "")
                pct = msg.get("percent", 0)
                frames = msg.get("frames", 0)
                total = msg.get("total", 0)
                logger.info("[%s] %5.1f%% (%d/%d frames)", stage, pct, frames, total)

                # Set ready BEFORE queueing "complete" so the client
                # can immediately fetch /api/video and /api/delta.csv
                if stage == "complete":
                    app.state.ready = True
                    logger.info("Pipeline complete — results ready to serve")

                try:
                    job["queue"].put_nowait(msg)
                except asyncio.QueueFull:
                    pass

            try:
                await asyncio.to_thread(
                    run_pipeline,
                    job["vid1_path"],
                    job["vid2_path"],
                    job["output_dir"],
                    sample_fps=sample_fps,
                    band_fraction=band_fraction,
                    progress_callback=progress_cb,
                )
            except Exception as e:
                logger.error("Pipeline failed for job %s: %s", job_id, e)
                try:
                    job["queue"].put_nowait({"stage": "error", "percent": 0, "error": str(e)})
                except asyncio.QueueFull:
                    pass

        asyncio.create_task(_run())
        return {"status": "started"}

    @app.websocket("/api/ws/{job_id}")
    async def ws_progress(ws: WebSocket, job_id: str):
        await ws.accept()
        logger.info("WebSocket connected for job %s", job_id)
        job = app.state.jobs.get(job_id)
        if not job:
            logger.warning("WebSocket: job %s not found", job_id)
            await ws.close(code=4004, reason="Job not found")
            return

        queue = job["queue"]
        try:
            while True:
                msg = await queue.get()
                await ws.send_json(msg)
                if msg.get("stage") == "complete":
                    logger.info("WebSocket: sent 'complete' to client for job %s", job_id)
                    break
                if msg.get("stage") == "error":
                    logger.error("WebSocket: sent error to client for job %s", job_id)
                    break
        except Exception as e:
            logger.error("WebSocket error for job %s: %s", job_id, e)
        finally:
            await ws.close()

    return app
