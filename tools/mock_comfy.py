#!/usr/bin/env python3
"""A fake ComfyUI server for trying the app without a GPU.

    python tools/mock_comfy.py --port 8188

It accepts the same endpoints the app uses, checks that the submitted graph
is wired correctly (every link points at an existing node and output), and
returns a placeholder image showing the seed and the start of the prompt.
"""

import argparse
import io
import random
import threading
import time
import uuid

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageDraw

OUTPUTS = {
    "UNETLoader": 1, "LoraLoaderModelOnly": 1, "QwenImage21Cache": 1, "CLIPLoader": 1,
    "VAELoader": 1, "LoadImage": 2, "TextEncodeQwenImage21": 3, "EmptyLatentImage": 1,
    "KSampler": 1, "VAEDecode": 1, "SaveImage": 0,
}
app = FastAPI()
uploads: dict[str, bytes] = {}
jobs: dict[str, dict] = {}
images: dict[str, bytes] = {}
DELAY = 1.5
PASSWORD = ""  # --password: behave like RunPod's ComfyUI template proxy
_fails = {"n": 0}


@app.middleware("http")
async def password_proxy(request, call_next):
    import base64
    if not PASSWORD or request.url.path.startswith("/v2/"):
        return await call_next(request)
    auth = request.headers.get("authorization", "")
    given = ""
    if auth.startswith("Bearer "):
        given = auth[7:]
    elif auth.startswith("Basic "):
        given = base64.b64decode(auth[6:]).decode().split(":", 1)[-1]
    if given == PASSWORD:
        _fails["n"] = 0
        return await call_next(request)
    _fails["n"] += 1
    if _fails["n"] > 10:
        return Response("too many attempts", status_code=429)
    return Response("password required", status_code=401, headers={"WWW-Authenticate": 'Basic realm="ComfyUI"'})


@app.get("/system_stats")
def stats():
    return {"system": {"comfyui_version": "mock"}, "devices": [{"name": "Mock GPU 48GB"}]}


@app.get("/object_info/{node}")
def object_info(node: str):
    return {node: {"input": {}}} if node in OUTPUTS else {}


@app.get("/models/{folder}")
def models(folder: str):
    return {
        "diffusion_models": ["qwen_image_2.1_int8_convrot.safetensors"],
        "text_encoders": ["qwen3vl_8b_int8_convrot.safetensors"],
        "vae": ["qwen_image_2.1_vae_bf16.safetensors"],
        "loras": [],
    }.get(folder, [])


@app.post("/upload/image")
async def upload(image: UploadFile = File(...), overwrite: str = Form("false"), type: str = Form("input")):
    uploads[image.filename] = await image.read()
    return {"name": image.filename, "subfolder": "", "type": "input"}


def validate(graph: dict) -> list[str]:
    errs = []
    for nid, node in graph.items():
        ct = node.get("class_type")
        if ct not in OUTPUTS:
            errs.append(f"{nid}: unknown class {ct}")
        for k, v in node.get("inputs", {}).items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[1], int):
                src = graph.get(v[0])
                if src is None:
                    errs.append(f"{nid}.{k}: links to missing node {v[0]}")
                elif v[1] >= OUTPUTS.get(src["class_type"], 0):
                    errs.append(f"{nid}.{k}: {src['class_type']} has no output {v[1]}")
        if ct == "LoadImage" and node["inputs"]["image"] not in uploads:
            errs.append(f"{nid}: image {node['inputs']['image']} was not uploaded")
    return errs


@app.post("/prompt")
def prompt(body: dict):
    graph = body["prompt"]
    errs = validate(graph)
    if errs:
        return JSONResponse(status_code=400, content={"error": {"message": "invalid prompt"},
                            "node_errors": {"x": {"class_type": "?", "errors": [{"message": e} for e in errs]}}})
    pid = uuid.uuid4().hex
    jobs[pid] = {"graph": graph, "t": time.time()}
    return {"prompt_id": pid, "number": len(jobs)}


def render(graph: dict) -> bytes:
    lat = next((n["inputs"] for n in graph.values() if n["class_type"] == "EmptyLatentImage"), None)
    w, h = (lat["width"], lat["height"]) if lat else (896, 1344)
    seed = next(n["inputs"]["seed"] for n in graph.values() if n["class_type"] == "KSampler")
    text = next(n["inputs"]["prompt"] for n in graph.values() if n["class_type"] == "TextEncodeQwenImage21")
    rnd = random.Random(seed)
    img = Image.new("RGB", (w // 4, h // 4), tuple(rnd.randint(60, 200) for _ in range(3)))
    d = ImageDraw.Draw(img)
    d.ellipse([img.width * .3, img.height * .15, img.width * .7, img.height * .45], fill=(230, 200, 170))
    d.text((8, 8), f"seed {seed % 100000}", fill="white")
    words, line, y = text.split(), "", img.height * .55
    for wd in words[:40]:
        if len(line) + len(wd) > 30:
            d.text((8, y), line, fill="white"); y += 12; line = ""
        line += wd + " "
    d.text((8, y), line, fill="white")
    buf = io.BytesIO(); img.save(buf, "PNG"); return buf.getvalue()


@app.get("/history/{pid}")
def history(pid: str):
    job = jobs.get(pid)
    if not job or time.time() - job["t"] < DELAY:
        return {}
    name = f"{pid}.png"
    if name not in images:
        images[name] = render(job["graph"])
    return {pid: {"status": {"status_str": "success"}, "outputs": {"save": {"images": [{"filename": name, "subfolder": "", "type": "output"}]}}}}


@app.get("/view")
def view(filename: str, subfolder: str = "", type: str = "output"):
    if filename not in images:
        raise HTTPException(404)
    return Response(images[filename], media_type="image/png")


@app.post("/queue")
def queue(body: dict):
    for pid in body.get("delete", []):
        jobs.pop(pid, None)
    return {}


@app.post("/interrupt")
def interrupt():
    return {}


# ---------------------------------------------------------------- RunPod serverless
# Mimics https://api.runpod.ai/v2/<endpoint>/... running worker-comfyui.
# Point the app at it with RUNPOD_API_BASE=http://127.0.0.1:8188/v2
rp_jobs: dict[str, dict] = {}


@app.get("/v2/{endpoint}/health")
def rp_health(endpoint: str):
    return {"jobs": {"inQueue": 0, "inProgress": 0}, "workers": {"idle": 1, "ready": 1, "running": 0}}


@app.post("/v2/{endpoint}/run")
def rp_run(endpoint: str, body: dict):
    import base64
    inp = body["input"]
    for img in inp.get("images", []):
        uploads[img["name"]] = base64.b64decode(img["image"])
    jid = "rp-" + uuid.uuid4().hex[:12]
    errs = validate(inp["workflow"])
    rp_jobs[jid] = {"graph": inp["workflow"], "t": time.time(), "errs": errs}
    return {"id": jid, "status": "IN_QUEUE"}


@app.get("/v2/{endpoint}/status/{jid}")
def rp_status(endpoint: str, jid: str):
    import base64
    job = rp_jobs.get(jid)
    if not job:
        raise HTTPException(404)
    if job.get("cancelled"):
        return {"id": jid, "status": "CANCELLED"}
    if time.time() - job["t"] < DELAY:
        return {"id": jid, "status": "IN_PROGRESS"}
    if job["errs"]:
        return {"id": jid, "status": "FAILED", "error": "; ".join(job["errs"])}
    data = base64.b64encode(render(job["graph"])).decode()
    return {"id": jid, "status": "COMPLETED", "output": {"images": [{"filename": "x.png", "type": "base64", "data": data}]}}


@app.post("/v2/{endpoint}/cancel/{jid}")
def rp_cancel(endpoint: str, jid: str):
    if jid in rp_jobs:
        rp_jobs[jid]["cancelled"] = True
    return {"id": jid, "status": "CANCELLED"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--password", default="", help="require this password, like RunPod's ComfyUI template")
    a = ap.parse_args()
    PASSWORD = a.password
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
