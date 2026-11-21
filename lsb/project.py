"""Project state: one folder per LoRA, with everything in project.yaml.

Layout of a project folder:

    project.yaml          settings, sources, takes, variants, picks
    sources/              imported source photos (EXIF-rotated copies)
    takes/<take_id>/      generated variants for each take
    export/               dataset + ai-toolkit config written by Export
"""

from __future__ import annotations

import copy
import os
import re
import secrets
import threading
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SETTINGS: dict[str, Any] = {
    "name": "my_character",
    "trigger": "ohwx",
    # "person" (real photos) or "character" (invented from a description)
    "subject_kind": "person",
    # the class word used in prompts and captions: man, woman, boy, girl, person, character...
    "class_word": "person",
    # free-text description of the subject. For an invented character this is
    # what defines it; for a real person keep it short (age range, build, hair)
    "description": "",
    "comfy": {
        # "pod": a ComfyUI server (local or a RunPod pod) at `url`
        # "serverless": a RunPod serverless endpoint running worker-comfyui
        "backend": "pod",
        "runpod_endpoint_id": "",
        "url": "http://127.0.0.1:8188",
        # extra HTTP headers, e.g. {"Authorization": "Bearer ..."} behind a proxy
        "headers": {},
        "unet_name": "qwen_image_2.1_int8_convrot.safetensors",
        "clip_name": "qwen3vl_8b_int8_convrot.safetensors",
        "vae_name": "qwen_image_2.1_vae_bf16.safetensors",
        # set to use the QwenImage21Cache node from the official template
        "use_cache_node": True,
        "steps": 40,
        "cfg": 1.0,
        "sampler_name": "euler",
        "scheduler": "simple",
        # pixel budget for reference images inside TextEncodeQwenImage21
        "ref_resolution": 1024,
        # output size: long side of the generated image (multiple of 32)
        "long_side": 1344,
        # optional LoRA applied while generating (round 2 / bootstrapping)
        "lora_name": "",
        "lora_strength": 0.8,
        "negative_prompt": "",
    },
    "generation": {
        "variants_per_take": 4,
        # how many reference images to send with each take (1-10)
        "max_refs": 4,
    },
    "export": {
        "max_long_side": 1536,
        "include_sources": True,
    },
}

ASPECTS = {"portrait": (2, 3), "square": (1, 1), "landscape": (3, 2)}


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def new_id(prefix: str = "") -> str:
    return prefix + secrets.token_hex(3)


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:40] or "take"


class Project:
    """Thread-safe wrapper around project.yaml."""

    def __init__(self, root: os.PathLike | str):
        self.root = Path(root).resolve()
        self.lock = threading.RLock()
        self.data: dict[str, Any] = {}
        self.load()

    # ---------------------------------------------------------------- paths
    @property
    def file(self) -> Path:
        return self.root / "project.yaml"

    @property
    def sources_dir(self) -> Path:
        return self.root / "sources"

    def take_dir(self, take_id: str) -> Path:
        return self.root / "takes" / take_id

    @property
    def export_dir(self) -> Path:
        return self.root / "export"

    # ---------------------------------------------------------------- io
    def load(self) -> None:
        with self.lock:
            self.root.mkdir(parents=True, exist_ok=True)
            self.sources_dir.mkdir(exist_ok=True)
            (self.root / "takes").mkdir(exist_ok=True)
            raw: dict[str, Any] = {}
            if self.file.exists():
                raw = yaml.safe_load(self.file.read_text(encoding="utf-8")) or {}
            self.data = {
                "settings": _deep_merge(DEFAULT_SETTINGS, raw.get("settings", {})),
                "sources": raw.get("sources", []) or [],
                "takes": raw.get("takes", []) or [],
            }
            for take in self.data["takes"]:
                normalise_take(take)
            if not self.file.exists():
                self.save()

    def save(self) -> None:
        with self.lock:
            tmp = self.file.with_suffix(".yaml.tmp")
            tmp.write_text(
                yaml.safe_dump(self.data, sort_keys=False, allow_unicode=True, width=100),
                encoding="utf-8",
            )
            os.replace(tmp, self.file)

    # ---------------------------------------------------------------- secrets
    # Secrets are kept out of project.yaml, in .secrets.yaml next to it:
    #   runpod_api_key    RunPod API key (serverless)      env RUNPOD_API_KEY
    #   comfyui_password  COMFYUI_PASSWORD of a pod made   env COMFYUI_PASSWORD
    #                     from RunPod's ComfyUI template
    # An environment variable takes precedence over the file.
    SECRET_ENV = {"runpod_api_key": "RUNPOD_API_KEY", "comfyui_password": "COMFYUI_PASSWORD"}

    @property
    def secrets_file(self) -> Path:
        return self.root / ".secrets.yaml"

    def _secrets(self) -> dict[str, str]:
        if self.secrets_file.exists():
            return yaml.safe_load(self.secrets_file.read_text(encoding="utf-8")) or {}
        return {}

    def secret(self, name: str) -> str:
        env = os.environ.get(self.SECRET_ENV[name], "")
        return env or str(self._secrets().get(name, "") or "")

    def secret_from_env(self, name: str) -> bool:
        return bool(os.environ.get(self.SECRET_ENV[name]))

    def set_secret(self, name: str, value: str) -> None:
        if name not in self.SECRET_ENV:
            raise KeyError(name)
        data = self._secrets()
        data[name] = value.strip()
        self.secrets_file.write_text(yaml.safe_dump(data), encoding="utf-8")
        try:
            os.chmod(self.secrets_file, 0o600)
        except OSError:
            pass

    def runpod_api_key(self) -> str:
        return self.secret("runpod_api_key")

    def comfyui_password(self) -> str:
        return self.secret("comfyui_password")

    # ---------------------------------------------------------------- lookup
    @property
    def settings(self) -> dict[str, Any]:
        return self.data["settings"]

    def take(self, take_id: str) -> dict[str, Any]:
        for t in self.data["takes"]:
            if t["id"] == take_id:
                return t
        raise KeyError(take_id)

    def source(self, source_id: str) -> dict[str, Any]:
        for s in self.data["sources"]:
            if s["id"] == source_id:
                return s
        raise KeyError(source_id)

    def variant(self, take_id: str, variant_id: str) -> dict[str, Any]:
        for v in self.take(take_id)["variants"]:
            if v["id"] == variant_id:
                return v
        raise KeyError(variant_id)

    def reference_sources(self) -> list[dict[str, Any]]:
        """Sources marked as identity references, best first."""
        refs = [s for s in self.data["sources"] if s.get("use_as_ref")]
        refs.sort(key=lambda s: s.get("ref_order", 99))
        return refs

    def path_of_source(self, src: dict[str, Any]) -> Path:
        return self.sources_dir / src["file"]

    def path_of_variant(self, take_id: str, variant: dict[str, Any]) -> Path:
        return self.take_dir(take_id) / variant["file"]


TAKE_FIELDS = (
    "framing",
    "angle",
    "expression",
    "outfit",
    "setting",
    "lighting",
    "extra",
)


def normalise_take(take: dict[str, Any]) -> dict[str, Any]:
    take.setdefault("id", new_id("t"))
    take.setdefault("title", "")
    # role: dataset (goes into the training set), anchor (text-to-image to
    # create a character), ref (makes extra reference images, e.g. a turnaround)
    take.setdefault("role", "dataset")
    # mode: reference (new image from identity refs), edit (change a source
    # photo), t2i (from the description only)
    take.setdefault("mode", "reference")
    take.setdefault("edit_source", "")
    take.setdefault("aspect", "portrait")
    for f in TAKE_FIELDS:
        take.setdefault(f, "")
    take.setdefault("prompt_override", "")
    take.setdefault("caption", "")
    take.setdefault("caption_edited", False)
    take.setdefault("variants", [])
    return take
