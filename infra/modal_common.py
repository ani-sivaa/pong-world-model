"""Canonical Modal app/image/volume spec — seeded by the main thread.

modal-infra: EXTEND this (import app/image/volume), do not redefine — the image
cache keys on this exact spec, and preflight already built it.
"""
from pathlib import Path

import modal

APP_NAME = "worldmodel-pong"
VOL_NAME = "worldmodel-vol"
REPO_ROOT = Path(__file__).resolve().parent.parent

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "numpy",
        "imageio",
        "pillow",
        "matplotlib",
        "onnx",
        "onnxruntime",
        "tqdm",
    )
    .env({"WM_ROOT": "/vol", "MPLBACKEND": "Agg"})
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/root/proj",
        ignore=[
            ".venv/**", ".git/**", "data/**", "checkpoints/**",
            "results/**", "__pycache__/**", "**/__pycache__/**",
        ],
    )
)
