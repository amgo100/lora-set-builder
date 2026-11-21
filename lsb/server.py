"""Web API and UI server."""

from __future__ import annotations

import io
import shutil
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from PIL import Image, ImageOps

from . import export as exporter
from . import identity
from .comfy import ComfyError, make_client
from .jobs import Generator
from .project import TAKE_FIELDS, Project, _deep_merge, new_id, normalise_take
from .shots import (
    anchor_take,
    compose_caption,
    compose_prompt,
    default_shot_list,
    source_caption,
    turnaround_take,
)

STATIC = Path(__file__).parent / "static"
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def create_app(project_dir: str) -> FastAPI:
    p = Project(project_dir)
    gen = Generator(p)
    app = FastAPI(title="LoRA Set Builder")
    app.state.project = p
    app.state.gen = gen

    # ------------------------------------------------------------ helpers
    def n_refs_for(take: dict[str, Any]) -> int:
        max_refs = int(p.settings["generation"].get("max_refs", 4))
        refs = len(p.reference_sources())
        if take["mode"] == "t2i":
            return 0
        if take["mode"] == "edit":
            return 1 + min(max(0, refs - 1), max_refs - 1)
        return min(refs, max_refs)

    def take_view(take: dict[str, Any]) -> dict[str, Any]:
        t = dict(take)
        t["prompt_preview"] = compose_prompt(take, p.settings, max(1, n_refs_for(take)))
        if not take.get("caption_edited"):
            t["caption"] = compose_caption(take, p.settings)
        t["picked_count"] = sum(1 for v in take["variants"] if v.get("picked"))
        return t

    def get_take(take_id: str) -> dict[str, Any]:
        try:
            return p.take(take_id)
        except KeyError:
            raise HTTPException(404, "take not found")

    # ------------------------------------------------------------ ui
    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    # ------------------------------------------------------------ state
    @app.get("/api/state")
    def state():
        with p.lock:
            sources = []
            for s in p.data["sources"]:
                s2 = dict(s)
                s2["caption_effective"] = s.get("caption") or source_caption(p.settings)
                sources.append(s2)
            return {
                "settings": p.settings,
                "sources": sources,
                "takes": [take_view(t) for t in p.data["takes"]],
                "busy": gen.busy(),
                "last_error": gen.last_error,
                "identity_scoring": identity.available(),
                "runpod_key_set": bool(p.runpod_api_key()),
                "runpod_key_from_env": p.secret_from_env("runpod_api_key"),
                "comfy_password_set": bool(p.comfyui_password()),
                "comfy_password_from_env": p.secret_from_env("comfyui_password"),
                "project_dir": str(p.root),
            }

    @app.post("/api/settings")
    def update_settings(patch: dict = Body(...)):
        with p.lock:
            p.data["settings"] = _deep_merge(p.settings, patch)
            p.save()
        return {"ok": True}

    @app.post("/api/secrets")
    def set_secrets(body: dict = Body(...)):
        for name in ("runpod_api_key", "comfyui_password"):
            if body.get(name):
                p.set_secret(name, body[name])
        return {"ok": True}

    @app.post("/api/comfy/check")
    def comfy_check():
        c = p.settings["comfy"]
        try:
            client = make_client(c, p.runpod_api_key(), timeout=20, password=p.comfyui_password())
        except Exception as e:
            return {"ok": False, "error": str(e)}
        try:
            rep = client.check()
        except Exception as e:
            msg = str(e) if isinstance(e, ComfyError) else f"{type(e).__name__}: {e}"
            return {"ok": False, "error": msg}
        finally:
            client.close()
        if rep.get("serverless"):
            return rep
        models = rep.get("models", {})
        for key, folder in (("unet_name", "diffusion_models"), ("clip_name", "text_encoders"), ("vae_name", "vae")):
            if models.get(folder) and c[key] not in models[folder]:
                rep["missing"].append(f"{folder}/{c[key]}")
                rep["ok"] = False
        if c.get("lora_name") and models.get("loras") and c["lora_name"] not in models["loras"]:
            rep["missing"].append(f"loras/{c['lora_name']}")
            rep["ok"] = False
        return rep

    # ------------------------------------------------------------ files
    def _thumb(path: Path, w: int) -> Response:
        if not path.exists():
            raise HTTPException(404)
        if not w:
            return FileResponse(path)
        cache = path.parent / ".thumbs" / f"{path.stem}_{w}.jpg"
        if not cache.exists() or cache.stat().st_mtime < path.stat().st_mtime:
            cache.parent.mkdir(exist_ok=True)
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            img.thumbnail((w, w * 2))
            img.save(cache, "JPEG", quality=85)
        return FileResponse(cache)

    @app.get("/files/sources/{name}")
    def source_file(name: str, w: int = 0):
        return _thumb(p.sources_dir / Path(name).name, w)

    @app.get("/files/takes/{take_id}/{name}")
    def take_file(take_id: str, name: str, w: int = 0):
        return _thumb(p.take_dir(Path(take_id).name) / Path(name).name, w)

    # ------------------------------------------------------------ sources
    @app.post("/api/sources")
    async def upload_sources(files: list[UploadFile] = File(...)):
        added = []
        for f in files:
            ext = Path(f.filename or "").suffix.lower()
            if ext not in IMAGE_EXT:
                continue
            data = await f.read()
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
            sid = new_id("s")
            name = f"{sid}.png"
            img.save(p.sources_dir / name)
            with p.lock:
                is_first_refs = len(p.reference_sources()) < 4
                src = {
                    "id": sid,
                    "file": name,
                    "original_name": f.filename,
                    "width": img.width,
                    "height": img.height,
                    "use_as_ref": is_first_refs,
                    "ref_order": len(p.data["sources"]),
                    "in_dataset": True,
                    "caption": "",
                    "framing": "",
                }
                p.data["sources"].append(src)
                added.append(sid)
        with p.lock:
            p.save()
        return {"added": added}

    @app.patch("/api/sources/{sid}")
    def patch_source(sid: str, patch: dict = Body(...)):
        with p.lock:
            try:
                src = p.source(sid)
            except KeyError:
                raise HTTPException(404)
            for k in ("use_as_ref", "in_dataset", "caption", "ref_order", "framing"):
                if k in patch:
                    src[k] = patch[k]
            p.save()
        return {"ok": True}

    @app.delete("/api/sources/{sid}")
    def delete_source(sid: str):
        with p.lock:
            try:
                src = p.source(sid)
            except KeyError:
                raise HTTPException(404)
            p.data["sources"].remove(src)
            (p.sources_dir / src["file"]).unlink(missing_ok=True)
            p.save()
        return {"ok": True}

    # ------------------------------------------------------------ takes
    @app.post("/api/takes/default")
    def make_default(body: dict = Body(...)):
        n = int(body.get("n", 20))
        takes = default_shot_list(n, keep_outfit=bool(body.get("keep_outfit")))
        with p.lock:
            if body.get("replace"):
                keep = [t for t in p.data["takes"] if t["variants"] or t["role"] != "dataset"]
                p.data["takes"] = keep
            p.data["takes"].extend(takes)
            p.save()
        return {"added": len(takes)}

    @app.post("/api/takes")
    def add_take(body: dict = Body(...)):
        kind = body.get("template", "blank")
        if kind == "anchor":
            take = anchor_take()
        elif kind == "sheet":
            take = turnaround_take()
        else:
            take = normalise_take({"title": "New take", "framing": "medium", "angle": "facing the camera"})
        with p.lock:
            # anchors and reference sheets go first
            if take["role"] != "dataset":
                p.data["takes"].insert(0, take)
            else:
                p.data["takes"].append(take)
            p.save()
        return {"id": take["id"]}

    @app.patch("/api/takes/{take_id}")
    def patch_take(take_id: str, patch: dict = Body(...)):
        with p.lock:
            take = get_take(take_id)
            allowed = set(TAKE_FIELDS) | {
                "title", "role", "mode", "edit_source", "aspect", "prompt_override", "caption",
            }
            for k, v in patch.items():
                if k in allowed:
                    take[k] = v
            if "caption" in patch:
                take["caption_edited"] = bool(patch["caption"].strip())
            p.save()
        return {"ok": True}

    @app.post("/api/takes/{take_id}/duplicate")
    def duplicate_take(take_id: str):
        with p.lock:
            take = get_take(take_id)
            copy = {k: v for k, v in take.items() if k != "variants"}
            copy["id"] = new_id("t")
            copy["title"] = (take.get("title") or "") + " (copy)"
            copy = normalise_take(copy)
            idx = p.data["takes"].index(take)
            p.data["takes"].insert(idx + 1, copy)
            p.save()
        return {"id": copy["id"]}

    @app.delete("/api/takes/{take_id}")
    def delete_take(take_id: str):
        with p.lock:
            take = get_take(take_id)
            p.data["takes"].remove(take)
            shutil.rmtree(p.take_dir(take_id), ignore_errors=True)
            p.save()
        return {"ok": True}

    @app.post("/api/takes/{take_id}/generate")
    def generate(take_id: str, body: dict = Body(default={})):
        get_take(take_id)
        count = int(body.get("count") or p.settings["generation"]["variants_per_take"])
        return {"queued": gen.request(take_id, max(1, min(count, 16)))}

    @app.post("/api/generate_missing")
    def generate_missing(body: dict = Body(default={})):
        """Queue variants for every dataset take that has nothing picked and
        nothing in progress."""
        count = int(body.get("count") or p.settings["generation"]["variants_per_take"])
        queued = 0
        with p.lock:
            todo = [
                t["id"]
                for t in p.data["takes"]
                if t["role"] == "dataset"
                and not any(v.get("picked") for v in t["variants"])
                and not any(v["status"] in ("queued", "running") for v in t["variants"])
                and (body.get("include_tried") or not any(v["status"] == "done" for v in t["variants"]))
            ]
        for tid in todo:
            queued += len(gen.request(tid, count))
        return {"takes": len(todo), "queued": queued}

    @app.delete("/api/takes/{take_id}/variants")
    def clear_variants(take_id: str, which: str = "unpicked"):
        with p.lock:
            take = get_take(take_id)
            keep, drop = [], []
            # images a queued refinement still has to send
            needed = {
                v["parent"]
                for v in take["variants"]
                if v.get("parent") and v["status"] in ("queued", "running")
            }
            for v in take["variants"]:
                busy = v["status"] in ("queued", "running")
                if busy or v["id"] in needed or (which == "unpicked" and v.get("picked")):
                    keep.append(v)
                else:
                    drop.append(v)
            for v in drop:
                if v.get("file"):
                    (p.take_dir(take_id) / v["file"]).unlink(missing_ok=True)
            take["variants"] = keep
            p.save()
        return {"removed": len(drop)}

    @app.post("/api/variants/{take_id}/{vid}/pick")
    def pick(take_id: str, vid: str, body: dict = Body(default={})):
        with p.lock:
            take = get_take(take_id)
            try:
                v = p.variant(take_id, vid)
            except KeyError:
                raise HTTPException(404)
            new_val = bool(body.get("picked", not v.get("picked")))
            if new_val and body.get("exclusive", True):
                for other in take["variants"]:
                    other["picked"] = False
            v["picked"] = new_val
            p.save()
        return {"picked": new_val}

    @app.post("/api/variants/{take_id}/{vid}/refine")
    def refine(take_id: str, vid: str, body: dict = Body(default={})):
        """Queue edits of one variant with a correction applied; the first
        reuses the variant's seed."""
        correction = str(body.get("correction") or "").strip()
        if not correction:
            raise HTTPException(400, "write what to change")
        with p.lock:
            get_take(take_id)
            try:
                v = p.variant(take_id, vid)
            except KeyError:
                raise HTTPException(404)
            if v["status"] != "done" or not v.get("file"):
                raise HTTPException(400, "variant has no image yet")
        count = int(body.get("count") or p.settings["generation"]["variants_per_take"])
        ids = gen.request(take_id, max(1, min(count, 16)), parent=vid, correction=correction)
        return {"queued": ids}

    @app.post("/api/variants/{take_id}/{vid}/promote")
    def promote(take_id: str, vid: str, body: dict = Body(default={})):
        """Copy a generated image into Sources (e.g. an anchor or a reference
        sheet) so later takes use it as an identity reference."""
        with p.lock:
            take = get_take(take_id)
            try:
                v = p.variant(take_id, vid)
            except KeyError:
                raise HTTPException(404)
            if not v.get("file"):
                raise HTTPException(400, "variant has no image yet")
            sid = new_id("s")
            name = f"{sid}.png"
            shutil.copy(p.take_dir(take_id) / v["file"], p.sources_dir / name)
            with Image.open(p.sources_dir / name) as im:
                w, h = im.size
            p.data["sources"].append(
                {
                    "id": sid,
                    "file": name,
                    "original_name": f"generated: {take.get('title') or take_id}",
                    "width": w,
                    "height": h,
                    "use_as_ref": True,
                    "ref_order": -1 if take["role"] == "anchor" else len(p.data["sources"]),
                    "in_dataset": bool(body.get("in_dataset", take["role"] == "anchor")),
                    "caption": "",
                    "framing": take.get("framing", ""),
                    "generated": True,
                }
            )
            v["picked"] = True
            p.save()
        return {"source_id": sid}

    @app.post("/api/cancel")
    def cancel():
        return {"cancelled": gen.cancel_all()}

    # ------------------------------------------------------------ export
    @app.get("/api/export/report")
    def export_report():
        with p.lock:
            rep = exporter.report(p)
            rep["items"] = [
                {"label": i["label"], "caption": i["caption"], "kind": i["kind"]}
                for i in exporter.dataset_items(p)
            ]
            return rep

    @app.post("/api/export")
    def do_export():
        with p.lock:
            try:
                return exporter.export(p)
            except ValueError as e:
                raise HTTPException(400, str(e))

    @app.get("/api/export/download")
    def download():
        z = p.export_dir / f"{p.settings['name']}_dataset.zip"
        if not z.exists():
            raise HTTPException(404, "export first")
        return FileResponse(z, filename=z.name)

    return app
