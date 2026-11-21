"""ComfyUI API-format graph for Qwen-Image-2.1.

Built from the official ComfyUI template "Image Edit (Qwen Image 2.1)"
(Comfy-Org/workflow_templates, image_qwen_image_2_1_image_edit.json), flattened
out of its subgraph and without the optional prompt-enhancer branch:

    UNETLoader -> [LoraLoaderModelOnly] -> [QwenImage21Cache] -> KSampler
    CLIPLoader (qwen_image) + VAELoader + LoadImage x N -> TextEncodeQwenImage21
    TextEncodeQwenImage21.positive/negative -> KSampler
    latent: EmptyLatentImage (new composition) or TextEncodeQwenImage21.latent
            (edit mode: canvas follows image_1)
    KSampler -> VAEDecode -> SaveImage

If a future ComfyUI renames an input, this is the only file to change.
"""

from __future__ import annotations

from typing import Any


def build_graph(
    *,
    comfy: dict[str, Any],
    prompt: str,
    seed: int,
    width: int,
    height: int,
    images: list[str],
    use_image_latent: bool,
    filename_prefix: str,
) -> dict[str, Any]:
    """Return a graph ready to POST to /prompt.

    images: names of files already uploaded to ComfyUI's input folder, in the
    order they should appear as <image1>, <image2>, ...
    """
    if len(images) > 10:
        raise ValueError("Qwen-Image-2.1 accepts at most 10 reference images")
    if use_image_latent and not images:
        raise ValueError("edit mode needs an image to edit")

    g: dict[str, Any] = {}
    g["unet"] = {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": comfy["unet_name"], "weight_dtype": "default"},
    }
    model_ref: list[Any] = ["unet", 0]

    if comfy.get("lora_name"):
        g["lora"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_ref,
                "lora_name": comfy["lora_name"],
                "strength_model": float(comfy.get("lora_strength", 1.0)),
            },
        }
        model_ref = ["lora", 0]

    if comfy.get("use_cache_node", True):
        g["cache"] = {
            "class_type": "QwenImage21Cache",
            "inputs": {"model": model_ref, "device": "auto", "dtype": "default"},
        }
        model_ref = ["cache", 0]

    g["clip"] = {
        "class_type": "CLIPLoader",
        "inputs": {
            "clip_name": comfy["clip_name"],
            "type": "qwen_image",
            "device": "default",
        },
    }
    g["vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": comfy["vae_name"]}}

    enc_inputs: dict[str, Any] = {
        "clip": ["clip", 0],
        "prompt": prompt,
        "negative_prompt": comfy.get("negative_prompt", "") or "",
        "resolution": int(comfy.get("ref_resolution", 1024)),
    }
    if images:
        enc_inputs["vae"] = ["vae", 0]
    for i, name in enumerate(images, start=1):
        node_id = f"img{i}"
        g[node_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
        enc_inputs[f"images.image_{i}"] = [node_id, 0]
    g["encode"] = {"class_type": "TextEncodeQwenImage21", "inputs": enc_inputs}

    if use_image_latent:
        latent_ref: list[Any] = ["encode", 2]
    else:
        g["empty"] = {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": int(width), "height": int(height), "batch_size": 1},
        }
        latent_ref = ["empty", 0]

    g["sampler"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_ref,
            "positive": ["encode", 0],
            "negative": ["encode", 1],
            "latent_image": latent_ref,
            "seed": int(seed),
            "steps": int(comfy.get("steps", 40)),
            "cfg": float(comfy.get("cfg", 1.0)),
            "sampler_name": comfy.get("sampler_name", "euler"),
            "scheduler": comfy.get("scheduler", "simple"),
            "denoise": 1.0,
        },
    }
    g["decode"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]},
    }
    g["save"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["decode", 0], "filename_prefix": filename_prefix},
    }
    return g
