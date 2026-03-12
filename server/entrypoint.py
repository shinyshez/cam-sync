"""CLI entrypoint for vid-sync server.

Mode C: python -m server.entrypoint /data/vid1.mp4 /data/vid2.mp4
    → runs pipeline, then serves viewer with results
Mode D: python -m server.entrypoint
    → starts server with upload UI
"""

import argparse
import logging
import os
import sys


def parse_mode(argv: list[str]) -> tuple[str, dict]:
    """Parse command-line args and determine mode.

    Returns:
        (mode, args_dict) where mode is "cli", "webapp", or "test"
    """
    parser = argparse.ArgumentParser(description="Vid-Sync Server")
    parser.add_argument("videos", nargs="*", help="Two video file paths (Mode C)")
    parser.add_argument("--test", action="store_true", help="Run test suites")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--band-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)

    parsed = parser.parse_args(argv[1:])

    if parsed.test:
        return "test", {}

    if len(parsed.videos) >= 2:
        return "cli", {
            "vid1": parsed.videos[0],
            "vid2": parsed.videos[1],
            "sample_fps": parsed.sample_fps,
            "band_fraction": parsed.band_fraction,
            "device": parsed.device,
            "host": parsed.host,
            "port": parsed.port,
        }

    return "webapp", {
        "host": parsed.host,
        "port": parsed.port,
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    mode, args = parse_mode(sys.argv)

    if mode == "test":
        import subprocess
        r1 = subprocess.run(
            [sys.executable, "-m", "pytest", "analysis/tests/", "-v"],
            cwd="/app",
        )
        r2 = subprocess.run(
            [sys.executable, "-m", "pytest", "server/tests/", "-v"],
            cwd="/app",
        )
        r3 = subprocess.run(["npx", "playwright", "test"], cwd="/app/viewer")
        sys.exit(max(r1.returncode, r2.returncode, r3.returncode))

    if mode == "cli":
        # Mode C: run pipeline first, then serve
        from server.pipeline import run_pipeline

        output_dir = "/tmp/vid-sync-output"
        print(f"Running pipeline: {args['vid1']} vs {args['vid2']}")

        def progress(msg):
            stage = msg.get("stage", "")
            pct = msg.get("percent", 0)
            frames = msg.get("frames", 0)
            total = msg.get("total", 0)
            print(f"  [{pct:5.1f}%] {stage} ({frames}/{total} frames)")

        run_pipeline(
            args["vid1"], args["vid2"], output_dir,
            sample_fps=args["sample_fps"],
            band_fraction=args["band_fraction"],
            device=args["device"],
            progress_callback=progress,
        )

        print(f"\nPipeline complete. Starting server on {args['host']}:{args['port']}")

        from server.app import create_app
        import uvicorn

        app = create_app(
            mode="cli",
            ready=True,
            vid1_path=args["vid1"],
            vid2_path=args["vid2"],
            output_dir=output_dir,
        )
        uvicorn.run(app, host=args["host"], port=args["port"])

    else:
        # Mode D: serve webapp directly
        from server.app import create_app
        import uvicorn

        print(f"Starting vid-sync webapp on {args['host']}:{args['port']}")
        app = create_app(mode="webapp", ready=False)
        uvicorn.run(app, host=args["host"], port=args["port"])


if __name__ == "__main__":
    main()
