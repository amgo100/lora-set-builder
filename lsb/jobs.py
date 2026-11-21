"""Background generation: queue variants on ComfyUI and collect results.

ComfyUI keeps its own queue, so every requested variant is submitted at once
and one thread polls /history for the ones still in flight.
"""

from __future__ import annotations

import random
import threading
import time
import traceback
from typing import Any

from . import identity
from .comfy import ComfyError, make_client
from .graph import build_graph
from .project import Project, new_id
from .shots import compose_prompt, output_size


class Generator:
    def __init__(self, project: Project):
        self.project = project
        self._client: ComfyClient | None = None
        self._client_key: tuple | None = None
        self._wake = threading.Event()
        self._stop = False
        self.last_error = ""
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # --------------------------------------------------------------- client
    def client(self):
        c = self.project.settings["comfy"]
        api_key = self.project.runpod_api_key()
        password = self.project.comfyui_password()
        key = (
            password,
            c.get("backend", "pod"),
            c["url"],
            tuple(sorted((c.get("headers") or {}).items())),
            c.get("runpod_endpoint_id", ""),
            api_key,
        )
        if self._client is None or key != self._client_key:
            if self._client:
                self._client.close()
            self._client = make_client(c, api_key, password=password)
            self._client_key = key
        return self._client

    # --------------------------------------------------------------- api
    def request(self, take_id: str, count: int) -> list[str]:
        """Add `count` new variants to a take and mark them queued."""
        p = self.project
        with p.lock:
            take = p.take(take_id)
            ids = []
            for _ in range(count):
                v = {
                    "id": new_id("v"),
                    "file": "",
                    "seed": random.randint(1, 2**48),
                    "prompt": "",
                    "refs": [],
                    "status": "queued",
                    "prompt_id": "",
                    "picked": False,
                    "score": None,
                    "error": "",
                    "created": time.time(),
                }
                take["variants"].append(v)
                ids.append(v["id"])
            p.save()
        self._wake.set()
        return ids

    def cancel_all(self) -> int:
        p = self.project
        cancelled = []
        with p.lock:
            for take in p.data["takes"]:
                for v in take["variants"]:
                    if v["status"] in ("queued", "running"):
                        if v.get("prompt_id"):
                            cancelled.append(v["prompt_id"])
                        v["status"] = "cancelled"
            p.save()
        try:
            self.client().cancel(cancelled)
        except Exception:  # server may be gone; local state is already updated
            pass
        return len(cancelled)

    def busy(self) -> dict[str, int]:
        counts = {"queued": 0, "running": 0}
        for take in self.project.data["takes"]:
            for v in take["variants"]:
                if v["status"] in counts:
                    counts[v["status"]] += 1
        return counts

    # --------------------------------------------------------------- loop
    def _loop(self) -> None:
        while not self._stop:
            try:
                did_work = self._submit_queued() | self._poll_running()
            except Exception as e:  # keep the worker alive
                self.last_error = f"{type(e).__name__}: {e}"
                traceback.print_exc()
                did_work = False
                time.sleep(3)
            if not did_work:
                self._wake.wait(timeout=2.0)
                self._wake.clear()

    def _refs_for(self, take: dict[str, Any]) -> list:
        """Local paths of the images sent with this take, in <imageN> order."""
        p = self.project
        max_refs = int(p.settings["generation"].get("max_refs", 4))
        refs = [p.path_of_source(s) for s in p.reference_sources()]
        if take["mode"] == "t2i":
            return []
        if take["mode"] == "edit":
            src = p.source(take["edit_source"]) if take.get("edit_source") else None
            if src is None:
                raise ComfyError("edit mode: choose which source photo to edit")
            first = p.path_of_source(src)
            others = [r for r in refs if r != first]
            return [first] + others[: max(0, max_refs - 1)]
        if not refs:
            raise ComfyError("no reference images: mark at least one source as a reference")
        return refs[:max_refs]

    def _submit_queued(self) -> bool:
        p = self.project
        with p.lock:
            pending = [
                (t, v)
                for t in p.data["takes"]
                for v in t["variants"]
                if v["status"] == "queued"
            ]
        if not pending:
            return False
        try:
            client = self.client()
        except ComfyError as e:  # e.g. serverless selected but no API key
            with p.lock:
                for _, v in pending:
                    v.update(status="error", error=str(e))
                p.save()
            return True
        settings = p.settings
        for take, v in pending:
            try:
                ref_paths = self._refs_for(take)
                names = [client.upload(path) for path in ref_paths]
                prompt = compose_prompt(take, settings, len(names))
                w, h = output_size(take.get("aspect", "portrait"), settings["comfy"]["long_side"])
                graph = build_graph(
                    comfy=settings["comfy"],
                    prompt=prompt,
                    seed=v["seed"],
                    width=w,
                    height=h,
                    images=names,
                    use_image_latent=take["mode"] == "edit",
                    filename_prefix=f"lsb/{settings['name']}/{take['id']}",
                )
                prompt_id = client.queue(graph)
                with p.lock:
                    v.update(
                        status="running",
                        prompt_id=prompt_id,
                        prompt=prompt,
                        refs=[str(r.name) for r in ref_paths],
                        error="",
                    )
            except Exception as e:
                with p.lock:
                    v.update(status="error", error=str(e)[:500])
            finally:
                with p.lock:
                    p.save()
        return True

    def _poll_running(self) -> bool:
        p = self.project
        with p.lock:
            running = [
                (t, v)
                for t in p.data["takes"]
                for v in t["variants"]
                if v["status"] == "running"
            ]
        if not running:
            return False
        client = self.client()
        changed = False
        for take, v in running:
            try:
                entry = client.history(v["prompt_id"])
                if not entry:
                    continue
                data = client.fetch_output(entry)
                out_dir = p.take_dir(take["id"])
                out_dir.mkdir(parents=True, exist_ok=True)
                fname = f"{v['id']}.png"
                (out_dir / fname).write_bytes(data)
                score = identity.score_image(p, out_dir / fname)
                with p.lock:
                    v.update(status="done", file=fname, score=score)
                changed = True
            except Exception as e:
                with p.lock:
                    v.update(status="error", error=str(e)[:500])
                changed = True
        if changed:
            with p.lock:
                p.save()
        else:
            time.sleep(1.5)
        return changed
