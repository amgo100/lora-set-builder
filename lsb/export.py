"""Write the training set: images + .txt captions + an ai-toolkit config."""

from __future__ import annotations

import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageOps

from .project import Project
from .shots import compose_caption, source_caption


def dataset_items(p: Project) -> list[dict[str, Any]]:
    """Everything that will go into the dataset, in order."""
    s = p.settings
    items: list[dict[str, Any]] = []
    if s["export"].get("include_sources", True):
        for src in p.data["sources"]:
            if src.get("in_dataset"):
                items.append(
                    {
                        "kind": "source",
                        "path": p.path_of_source(src),
                        "caption": src.get("caption") or source_caption(s),
                        "label": src["file"],
                        "framing": src.get("framing", ""),
                    }
                )
    for take in p.data["takes"]:
        if take.get("role", "dataset") != "dataset":
            continue
        caption = take["caption"] if take.get("caption_edited") else compose_caption(take, s)
        for v in take["variants"]:
            if v.get("picked") and v.get("file"):
                items.append(
                    {
                        "kind": "take",
                        "path": p.path_of_variant(take["id"], v),
                        "caption": caption,
                        "label": take.get("title") or take["id"],
                        "framing": take.get("framing", ""),
                        "angle": take.get("angle", ""),
                        "setting": take.get("setting", ""),
                    }
                )
    return items


def report(p: Project) -> dict[str, Any]:
    """Balance check shown before export."""
    items = dataset_items(p)
    takes = [t for t in p.data["takes"] if t.get("role", "dataset") == "dataset"]
    unpicked = [t.get("title") or t["id"] for t in takes if not any(v.get("picked") for v in t["variants"])]
    framing = Counter(i.get("framing") or "unspecified (source)" for i in items)
    angles = Counter(i.get("angle") for i in items if i.get("angle"))
    captions = Counter(i["caption"] for i in items)
    warnings = []
    n = len(items)
    if n < 10:
        warnings.append(f"Only {n} images. 15-30 varied images is a comfortable size for a character LoRA.")
    if n and framing.get("close-up", 0) / n < 0.2:
        warnings.append("Few close-ups: the face may come out less accurate.")
    if n and framing.get("full-body", 0) / n < 0.15:
        warnings.append("Few full-body shots: body proportions may drift.")
    shared = sum(k for k in captions.values() if k > 1)
    if shared:
        warnings.append(
            f"{shared} images share a caption with another image (often source photos with the "
            "default caption). A caption that names each image's clothing, setting and angle trains better."
        )
    if not p.settings.get("trigger"):
        warnings.append("No trigger word set.")
    return {
        "count": n,
        "framing": dict(framing),
        "angles": dict(angles),
        "unpicked_takes": unpicked,
        "warnings": warnings,
    }


def _suggested_steps(n: int) -> int:
    return int(max(1000, min(3000, round(n * 100 / 250) * 250)))


def ai_toolkit_config(p: Project, n_images: int, dataset_path: str) -> dict[str, Any]:
    s = p.settings
    trig = s["trigger"]
    cls = s["class_word"]
    return {
        "job": "extension",
        "config": {
            "name": s["name"],
            "process": [
                {
                    "type": "sd_trainer",
                    "training_folder": "output",
                    "device": "cuda:0",
                    "trigger_word": trig,
                    "network": {"type": "lora", "linear": 32, "linear_alpha": 32},
                    "save": {"dtype": "float16", "save_every": 250, "max_step_saves_to_keep": 6},
                    "datasets": [
                        {
                            "folder_path": dataset_path,
                            "caption_ext": "txt",
                            "caption_dropout_rate": 0.05,
                            "shuffle_tokens": False,
                            "cache_latents_to_disk": True,
                            "resolution": [768, 1024, 1280],
                        }
                    ],
                    "train": {
                        "batch_size": 1,
                        "steps": _suggested_steps(n_images),
                        "gradient_accumulation": 1,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "flowmatch",
                        "optimizer": "adamw8bit",
                        "lr": 1e-4,
                        "dtype": "bf16",
                        "cache_text_embeddings": True,
                    },
                    "model": {
                        "name_or_path": "Qwen/Qwen-Image-2.1",
                        "arch": "qwen_image_2",
                        "quantize": True,
                        "quantize_te": True,
                        "low_vram": False,
                    },
                    "sample": {
                        "sampler": "flowmatch",
                        "sample_every": 250,
                        "width": 1024,
                        "height": 1024,
                        "prompts": [
                            f"{trig} {cls}, close-up portrait, facing the camera, neutral expression, plain grey backdrop",
                            f"{trig} {cls}, full-body shot, walking along a beach at sunset, wearing a linen shirt",
                            f"{trig} {cls}, medium shot, sitting at a wooden table reading a book, warm lamp light",
                            f"{trig} {cls}, three-quarter view, laughing, rainy city street at night",
                            f"a {cls}, close-up portrait, facing the camera, plain grey backdrop",
                        ],
                        "neg": "",
                        "seed": 42,
                        "walk_seed": True,
                        "guidance_scale": 4,
                        "sample_steps": 30,
                    },
                }
            ],
        },
        "meta": {"name": "[name]", "version": "1.0"},
    }


def export(p: Project) -> dict[str, Any]:
    s = p.settings
    items = dataset_items(p)
    if not items:
        raise ValueError("Nothing to export: pick at least one variant or include a source photo.")
    out = p.export_dir
    ds = out / "dataset"
    if ds.exists():
        shutil.rmtree(ds)
    ds.mkdir(parents=True)
    max_side = int(s["export"].get("max_long_side", 1536))
    manifest = []
    for i, item in enumerate(items, start=1):
        stem = f"{i:03d}_{item['kind']}"
        img = ImageOps.exif_transpose(Image.open(item["path"])).convert("RGB")
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.LANCZOS)
        img.save(ds / f"{stem}.png")
        (ds / f"{stem}.txt").write_text(item["caption"] + "\n", encoding="utf-8")
        manifest.append({"file": f"{stem}.png", "from": item["label"], "caption": item["caption"]})

    remote_path = f"/workspace/datasets/{s['name']}"
    cfg = ai_toolkit_config(p, len(items), remote_path)
    (out / f"{s['name']}_ai_toolkit.yaml").write_text(
        "# ai-toolkit config for Qwen-Image-2.1 (arch qwen_image_2).\n"
        f"# Upload the dataset folder to {remote_path} on the pod, or edit folder_path.\n"
        "# Steps are a starting point (~100 per image); check the samples every 250 steps.\n"
        + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (out / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")

    zpath = out / f"{s['name']}_dataset.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(ds.iterdir()):
            z.write(f, f"{s['name']}/{f.name}")
        z.write(out / f"{s['name']}_ai_toolkit.yaml", f"{s['name']}_ai_toolkit.yaml")
    return {"count": len(items), "folder": str(ds), "zip": str(zpath), "zip_name": zpath.name}
