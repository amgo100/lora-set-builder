#!/usr/bin/env python3
"""Start the LoRA Set Builder.

    python run.py projects/anna                 # open (or create) a project
    python run.py projects/anna --port 7861
"""

import argparse
import webbrowser

import uvicorn

from lsb.server import create_app


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA training-set builder for Qwen-Image-2.1 + ComfyUI")
    ap.add_argument("project", help="project folder (created if missing)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    app = create_app(args.project)
    url = f"http://{args.host}:{args.port}/"
    print(f"LoRA Set Builder: {url}  (project: {app.state.project.root})")
    if not args.no_browser:
        webbrowser.open(url)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
