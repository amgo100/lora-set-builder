# RunPod serverless worker for LoRA Set Builder: ComfyUI + Qwen-Image-2.1.
#
# Build and push (from this folder):
#   docker build --platform linux/amd64 -t <dockerhub-user>/lsb-qwen21-worker:1 .
#   docker push <dockerhub-user>/lsb-qwen21-worker:1
# Then create a serverless endpoint from that image (GPU with 24 GB+; 48 GB is comfortable).
#
# WORKER_VERSION: a current tag from https://github.com/runpod-workers/worker-comfyui/releases
# COMFYUI_REF: a ComfyUI release that has the Qwen-Image-2.1 nodes
#   (TextEncodeQwenImage21, QwenImage21Cache). If the worker's bundled ComfyUI
#   is already new enough, set it to "keep".
ARG WORKER_VERSION=5.10.0
FROM runpod/worker-comfyui:${WORKER_VERSION}-base

ARG COMFYUI_REF=master
RUN if [ "$COMFYUI_REF" != "keep" ]; then \
      cd /comfyui && git fetch --tags origin && git checkout "$COMFYUI_REF" && \
      ( [ "$COMFYUI_REF" != "master" ] || git pull --ff-only origin master ) && \
      pip install --no-cache-dir -r requirements.txt ; \
    fi

# Bake the models into the image (~16 GB, int8 versions). Set BAKE_MODELS=0 to
# use a network volume instead; see the README for the folder layout.
ARG BAKE_MODELS=1
RUN if [ "$BAKE_MODELS" = "1" ]; then \
      comfy model download --url https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/diffusion_models/qwen_image_2.1_int8_convrot.safetensors \
        --relative-path models/diffusion_models --filename qwen_image_2.1_int8_convrot.safetensors && \
      comfy model download --url https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors \
        --relative-path models/text_encoders --filename qwen3vl_8b_int8_convrot.safetensors && \
      comfy model download --url https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors \
        --relative-path models/vae --filename qwen_image_2.1_vae_bf16.safetensors ; \
    fi
