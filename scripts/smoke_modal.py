"""Preflight: confirm Modal launches a GPU job, sees the repo, and persists to the volume.

Run: .venv/bin/python -m modal run scripts/smoke_modal.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from infra.modal_common import app, image, volume  # noqa: E402


@app.function(gpu="T4", image=image, volumes={"/vol": volume}, timeout=600)
def gpu_smoke() -> dict:
    import os
    import torch

    os.chdir("/root")
    sys.path.insert(0, "/root")
    import config  # proves the repo mounted

    assert torch.cuda.is_available(), "CUDA not available on Modal worker"
    dev = torch.device("cuda")
    net = torch.nn.Sequential(
        torch.nn.Conv2d(4, 32, 3, padding=1), torch.nn.ReLU(),
        torch.nn.Conv2d(32, 1, 3, padding=1),
    ).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x = torch.rand(16, 4, 64, 64, device=dev)
    y = torch.rand(16, 1, 64, 64, device=dev)
    t0 = time.time()
    for _ in range(20):
        loss = torch.nn.functional.mse_loss(net(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()

    Path("/vol").mkdir(exist_ok=True)
    Path("/vol/smoke.txt").write_text(f"gpu={torch.cuda.get_device_name(0)}\n")
    volume.commit()
    return {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "seed": config.SEED,
        "train_20steps_s": round(time.time() - t0, 3),
    }


@app.local_entrypoint()
def main():
    result = gpu_smoke.remote()
    print("MODAL_SMOKE_OK", result)
