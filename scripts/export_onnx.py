"""Export a policy checkpoint to ONNX for the web demo, with a parity gate.

Usage:
    python -m scripts.export_onnx --ckpt checkpoints/ppo_baseline.pt \
        --out web/model_baseline.onnx

Contract (INTERFACES.md "ONNX contract"): input "frames" float32 [1,4,64,64]
in [0,1] (oldest->newest), output "logits" float32 [1,3], opset 17. Parity
gate: max|torch-ort| < 1e-4 on 64 random inputs + 8 real frame stacks from
data/local/frames.npy, and 100% argmax agreement on the real stacks.
"""
import argparse
import os
import sys

import numpy as np
import torch

import config
from agents.policy import LogitsOnly, load_policy

REAL_DATA_DIR = os.path.join("data", "local")
N_RANDOM = 64
N_REAL = 8
TOL = 1e-4


def real_frame_stacks(data_dir: str, n: int, seed: int = 0) -> np.ndarray:
    """n float32 [4,64,64] stacks in [0,1] from consecutive dataset frames.

    Index t is valid iff no done in [t-3, t] (contract sampling rule), so a
    stack never straddles an episode/stream boundary.
    """
    frames = np.load(os.path.join(data_dir, "frames.npy"), mmap_mode="r")
    dones = np.load(os.path.join(data_dir, "dones.npy"))
    T = len(dones)
    done_in_window = np.array(
        [dones[t - 3 : t + 1].any() for t in range(3, T)]
    )
    valid = np.arange(3, T)[~done_in_window]
    if len(valid) < n:
        raise RuntimeError(f"only {len(valid)} valid stack indices, need {n}")
    rng = np.random.default_rng(seed)
    picks = rng.choice(valid, size=n, replace=False)
    stacks = np.stack(
        [frames[t - 3 : t + 1].astype(np.float32) / 255.0 for t in picks]
    )
    return stacks  # [n, 4, 64, 64]


def export(ckpt_path: str, out_path: str) -> None:
    model = LogitsOnly(load_policy(ckpt_path, "cpu")).eval()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    dummy = torch.zeros(1, config.FRAME_STACK, 64, 64, dtype=torch.float32)
    torch.onnx.export(
        model,
        (dummy,),
        out_path,
        dynamo=False,  # legacy TorchScript exporter: small, stable graph
        opset_version=config.ONNX["opset"],
        input_names=[config.ONNX["input_name"]],
        output_names=[config.ONNX["output_name"]],
        # no dynamic_axes: shape fixed at [1,4,64,64] per the contract
    )
    print(f"exported {out_path} ({os.path.getsize(out_path)} bytes)")

    # ---- parity gate -------------------------------------------------------
    import onnxruntime as ort

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    in_name = config.ONNX["input_name"]

    rng = np.random.default_rng(213)
    random_inputs = rng.random(
        (N_RANDOM, config.FRAME_STACK, 64, 64), dtype=np.float32
    )
    real_inputs = real_frame_stacks(REAL_DATA_DIR, N_REAL)

    def run_both(stack: np.ndarray):
        x = stack[None]  # [1,4,64,64]
        with torch.no_grad():
            t_logits = model(torch.from_numpy(x)).numpy()
        (o_logits,) = sess.run(None, {in_name: x})
        return t_logits, o_logits

    max_diff = 0.0
    for stack in random_inputs:
        t_logits, o_logits = run_both(stack)
        max_diff = max(max_diff, float(np.abs(t_logits - o_logits).max()))

    argmax_agree = 0
    for stack in real_inputs:
        t_logits, o_logits = run_both(stack)
        max_diff = max(max_diff, float(np.abs(t_logits - o_logits).max()))
        argmax_agree += int(t_logits.argmax() == o_logits.argmax())

    if max_diff >= TOL:
        print(f"ONNX_PARITY_FAIL max_diff={max_diff:.3e} >= {TOL:.0e}")
        sys.exit(1)
    if argmax_agree != N_REAL:
        print(f"ONNX_PARITY_FAIL argmax agreement {argmax_agree}/{N_REAL}")
        sys.exit(1)
    print(f"ONNX_PARITY_OK max_diff={max_diff:.3e}")
    print(f"argmax agreement on real stacks: {argmax_agree}/{N_REAL} (100%)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="policy checkpoint (.pt)")
    ap.add_argument("--out", required=True, help="output .onnx path")
    args = ap.parse_args()
    export(args.ckpt, args.out)


if __name__ == "__main__":
    main()
