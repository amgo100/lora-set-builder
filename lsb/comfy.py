"""Minimal ComfyUI HTTP client (works through the RunPod HTTP proxy)."""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Any

import httpx


class ComfyError(RuntimeError):
    pass


def _auth_guard(r: httpx.Response) -> None:
    """Turn the password proxy's refusals into readable errors."""
    if r.status_code == 401:
        raise ComfyError(
            "ComfyUI refused the request (401): the ComfyUI password in Setup is missing "
            "or wrong. It is the pod's COMFYUI_PASSWORD."
        )
    if r.status_code == 429:
        raise ComfyError(
            "ComfyUI's password proxy is refusing requests after repeated wrong passwords (429). "
            "Correct the password and wait a moment."
        )


class ComfyClient:
    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 60,
        password: str = "",
    ):
        """password: COMFYUI_PASSWORD of a pod made from RunPod's official
        ComfyUI template. Its proxy accepts it as a Bearer token on every
        request (HTTP Basic with any username also works)."""
        self.url = url.rstrip("/")
        self.client_id = uuid.uuid4().hex
        hdrs = dict(headers or {})
        if password:
            hdrs["Authorization"] = f"Bearer {password}"
        self.http = httpx.Client(
            base_url=self.url,
            headers=hdrs,
            timeout=timeout,
            follow_redirects=True,
            event_hooks={"response": [_auth_guard]},
        )
        # local file hash -> name on the ComfyUI server
        self._uploaded: dict[str, str] = {}

    def close(self) -> None:
        self.http.close()

    # ------------------------------------------------------------------ info
    def check(self) -> dict[str, Any]:
        """Connection test: server stats plus whether the Qwen-Image-2.1 nodes
        and the configured model files exist."""
        r = self.http.get("/system_stats")
        r.raise_for_status()
        stats = r.json()
        report: dict[str, Any] = {"ok": True, "system": stats.get("system", {}), "missing": []}
        devices = stats.get("devices") or []
        if devices:
            report["gpu"] = devices[0].get("name")
        for node in ("TextEncodeQwenImage21", "QwenImage21Cache", "UNETLoader", "CLIPLoader"):
            r = self.http.get(f"/object_info/{node}")
            if r.status_code != 200 or node not in (r.json() or {}):
                report["missing"].append(f"node {node}")
                report["ok"] = False
        report["models"] = self._model_lists()
        return report

    def _model_lists(self) -> dict[str, list[str]]:
        out = {}
        for folder in ("diffusion_models", "text_encoders", "vae", "loras"):
            try:
                r = self.http.get(f"/models/{folder}")
                out[folder] = r.json() if r.status_code == 200 else []
            except httpx.HTTPError:
                out[folder] = []
        return out

    # ------------------------------------------------------------------ io
    def upload(self, path: Path) -> str:
        data = path.read_bytes()
        digest = hashlib.sha1(data).hexdigest()[:16]
        if digest in self._uploaded:
            return self._uploaded[digest]
        name = f"lsb_{digest}{path.suffix.lower() or '.png'}"
        r = self.http.post(
            "/upload/image",
            files={"image": (name, data, "application/octet-stream")},
            data={"overwrite": "true", "type": "input"},
        )
        if r.status_code != 200:
            raise ComfyError(f"upload failed ({r.status_code}): {r.text[:300]}")
        info = r.json()
        server_name = info["name"] if not info.get("subfolder") else f"{info['subfolder']}/{info['name']}"
        self._uploaded[digest] = server_name
        return server_name

    def queue(self, graph: dict[str, Any]) -> str:
        r = self.http.post("/prompt", json={"prompt": graph, "client_id": self.client_id})
        if r.status_code != 200:
            raise ComfyError(_explain(r))
        return r.json()["prompt_id"]

    def history(self, prompt_id: str) -> dict[str, Any] | None:
        r = self.http.get(f"/history/{prompt_id}")
        r.raise_for_status()
        return (r.json() or {}).get(prompt_id)

    def fetch_output(self, entry: dict[str, Any]) -> bytes:
        """Download the first image a finished prompt produced."""
        status = entry.get("status") or {}
        if status.get("status_str") == "error":
            msgs = [m for m in status.get("messages", []) if m and m[0] == "execution_error"]
            detail = msgs[0][1].get("exception_message", "") if msgs else ""
            raise ComfyError(f"ComfyUI execution error: {detail}".strip())
        for node_out in (entry.get("outputs") or {}).values():
            for img in node_out.get("images", []):
                r = self.http.get(
                    "/view",
                    params={
                        "filename": img["filename"],
                        "subfolder": img.get("subfolder", ""),
                        "type": img.get("type", "output"),
                    },
                )
                r.raise_for_status()
                return r.content
        raise ComfyError("finished without an image output")

    def cancel(self, prompt_ids: list[str]) -> None:
        if prompt_ids:
            self.http.post("/queue", json={"delete": prompt_ids})
        self.http.post("/interrupt")


class RunpodServerlessClient:
    """Same interface as ComfyClient, but talks to a RunPod serverless endpoint
    running the official worker (runpod-workers/worker-comfyui, v5+).

    The worker has no upload step: input images travel inside each request as
    base64, so `upload` only prepares a small JPEG copy and returns the name
    the LoadImage node will use. RunPod limits /run requests to 10 MB, so
    references are shrunk to ~1.2 MP (the model resizes them to its own pixel
    budget anyway).
    """

    # RUNPOD_API_BASE overrides the API address (used by tools/mock_comfy.py)
    API = os.environ.get("RUNPOD_API_BASE", "https://api.runpod.ai/v2")
    MAX_PIXELS = 1_200_000
    REQUEST_LIMIT = 9_500_000

    def __init__(self, endpoint_id: str, api_key: str, timeout: float = 60):
        if not endpoint_id:
            raise ComfyError("RunPod serverless: set the endpoint ID in Setup")
        if not api_key:
            raise ComfyError("RunPod serverless: set the API key in Setup (or RUNPOD_API_KEY)")
        self.endpoint = endpoint_id.strip()
        self.http = httpx.Client(
            base_url=f"{self.API}/{self.endpoint}",
            headers={"Authorization": f"Bearer {api_key.strip()}"},
            timeout=timeout,
        )
        self._images: dict[str, str] = {}  # name -> base64 JPEG

    def close(self) -> None:
        self.http.close()

    def check(self) -> dict[str, Any]:
        r = self.http.get("/health")
        if r.status_code == 401:
            raise ComfyError("RunPod rejected the API key (401)")
        if r.status_code == 404:
            raise ComfyError("RunPod endpoint not found (404): check the endpoint ID")
        r.raise_for_status()
        h = r.json()
        w = h.get("workers", {})
        return {
            "ok": True,
            "missing": [],
            "serverless": True,
            "workers": w,
            "jobs": h.get("jobs", {}),
            "note": (
                f"Endpoint reachable. Workers: {w.get('ready', 0)} ready, {w.get('idle', 0)} idle, "
                f"{w.get('running', 0)} running. Nodes and model files are checked by the worker "
                "when the first job runs."
            ),
        }

    def upload(self, path: Path) -> str:
        import base64
        import io

        from PIL import Image, ImageOps

        data = path.read_bytes()
        digest = hashlib.sha1(data).hexdigest()[:16]
        name = f"lsb_{digest}.jpg"
        if name not in self._images:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
            px = img.width * img.height
            if px > self.MAX_PIXELS:
                s = (self.MAX_PIXELS / px) ** 0.5
                img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=93)
            self._images[name] = base64.b64encode(buf.getvalue()).decode()
        return name

    def queue(self, graph: dict[str, Any]) -> str:
        names = sorted(
            {n["inputs"]["image"] for n in graph.values() if n.get("class_type") == "LoadImage"}
        )
        images = [{"name": n, "image": self._images[n]} for n in names]
        payload = {"input": {"workflow": graph, "images": images}}
        size = sum(len(i["image"]) for i in images)
        if size > self.REQUEST_LIMIT:
            raise ComfyError(
                f"request is {size / 1e6:.1f} MB, over RunPod's 10 MB limit: send fewer references"
            )
        r = self.http.post("/run", json=payload)
        if r.status_code != 200:
            raise ComfyError(f"RunPod /run failed ({r.status_code}): {r.text[:300]}")
        return r.json()["id"]

    def history(self, job_id: str) -> dict[str, Any] | None:
        r = self.http.get(f"/status/{job_id}")
        if r.status_code == 404:
            raise ComfyError("RunPod no longer has this job (results expire about 30 min after finishing)")
        r.raise_for_status()
        st = r.json()
        status = st.get("status")
        if status in ("IN_QUEUE", "IN_PROGRESS"):
            return None
        if status == "COMPLETED":
            return st
        detail = st.get("error") or (st.get("output") or {}).get("error") or ""
        raise ComfyError(f"RunPod job {status.lower() if status else 'failed'}: {str(detail)[:400]}")

    def fetch_output(self, entry: dict[str, Any]) -> bytes:
        import base64

        out = entry.get("output") or {}
        if isinstance(out, dict) and out.get("error"):
            details = out.get("details")
            raise ComfyError(f"worker error: {out['error']}" + (f" {details}" if details else ""))
        for img in (out.get("images") or []) if isinstance(out, dict) else []:
            if img.get("type") == "s3_url":
                r = httpx.get(img["data"], timeout=120)
                r.raise_for_status()
                return r.content
            data = img.get("data", "")
            if "," in data[:100] and data.startswith("data:"):
                data = data.split(",", 1)[1]
            return base64.b64decode(data)
        errs = out.get("errors") if isinstance(out, dict) else None
        raise ComfyError("job finished without an image" + (f": {errs}" if errs else ""))

    def cancel(self, job_ids: list[str]) -> None:
        for jid in job_ids:
            try:
                self.http.post(f"/cancel/{jid}")
            except httpx.HTTPError:
                pass


def make_client(
    comfy: dict[str, Any], api_key: str = "", timeout: float = 60, password: str = ""
):
    """Pick the backend from the settings: a ComfyUI server (local or a RunPod
    pod) or a RunPod serverless endpoint."""
    if comfy.get("backend") == "serverless":
        return RunpodServerlessClient(comfy.get("runpod_endpoint_id", ""), api_key, timeout=timeout)
    return ComfyClient(comfy["url"], comfy.get("headers") or {}, timeout=timeout, password=password)


def _explain(r: httpx.Response) -> str:
    try:
        body = r.json()
    except ValueError:
        return f"ComfyUI rejected the prompt ({r.status_code}): {r.text[:300]}"
    err = body.get("error", {})
    parts = [err.get("message", "ComfyUI rejected the prompt")]
    for node_id, info in (body.get("node_errors") or {}).items():
        for e in info.get("errors", []):
            parts.append(f"{node_id} ({info.get('class_type')}): {e.get('message')} {e.get('details', '')}")
    return " | ".join(parts)
