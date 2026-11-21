"""Optional identity score: how close a variant's face is to the references.

Uses InsightFace (ArcFace embeddings, model pack "buffalo_l"). If insightface
or onnxruntime is not installed the score is simply None and the app works
without it. Install with:  pip install insightface onnxruntime

The score is the cosine similarity between the largest face in the variant and
the mean embedding of the reference faces. For photos of the same person it is
typically above ~0.5; below ~0.3 is usually a different-looking face. Treat it
as a sorting aid, not a verdict - stylised characters often score low.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

_lock = threading.Lock()
_app: Any = None
_unavailable = False
_ref_cache: dict[tuple, Any] = {}


def available() -> bool:
    return _get_app() is not None


def _get_app():
    global _app, _unavailable
    if _app is not None or _unavailable:
        return _app
    with _lock:
        if _app is not None or _unavailable:
            return _app
        try:
            from insightface.app import FaceAnalysis  # type: ignore

            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            _app = app
        except Exception:
            _unavailable = True
    return _app


def _embedding(path: Path):
    import numpy as np
    from PIL import Image, ImageOps

    app = _get_app()
    if app is None:
        return None
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    arr = np.asarray(img)[:, :, ::-1]  # RGB -> BGR for insightface
    faces = app.get(arr)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    e = face.normed_embedding
    return e / (np.linalg.norm(e) + 1e-9)


def _reference_embedding(project) -> Any:
    import numpy as np

    refs = [project.path_of_source(s) for s in project.reference_sources()]
    key = tuple((str(r), r.stat().st_mtime) for r in refs if r.exists())
    if not key:
        return None
    if key in _ref_cache:
        return _ref_cache[key]
    embs = [e for e in (_embedding(r) for r in refs) if e is not None]
    if not embs:
        _ref_cache[key] = None
        return None
    mean = np.mean(embs, axis=0)
    mean = mean / (np.linalg.norm(mean) + 1e-9)
    _ref_cache[key] = mean
    return mean


def score_image(project, path: Path) -> float | None:
    if project.settings.get("subject_kind") == "character" and not project.reference_sources():
        return None
    if _get_app() is None:
        return None
    try:
        ref = _reference_embedding(project)
        if ref is None:
            return None
        e = _embedding(path)
        if e is None:
            return 0.0  # no face found: sort it to the bottom
        return round(float((ref * e).sum()), 3)
    except Exception:
        return None
