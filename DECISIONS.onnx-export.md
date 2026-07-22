# DECISIONS — onnx-export

## Exporter: legacy TorchScript (`torch.onnx.export(dynamo=False)`)

Torch 2.13 defaults to the dynamo/torch.export path. Chose the legacy exporter
because the model is a plain feed-forward Nature-CNN (no control flow, no
dynamic shapes), where the TorchScript tracer produces a small, stable,
well-understood graph that onnxruntime-web's CDN build consumes without issue.
The dynamo exporter adds no value here and emits a larger, decomposition-heavy
graph whose op choices shift across torch versions — a portability risk for a
browser runtime pinned by CDN. The deprecation warning at export time is
accepted and expected.

## Contract conformance

- Opset **17**, from `config.ONNX["opset"]` (matches onnxruntime-web support).
- Input `frames` float32 **fixed [1,4,64,64]** in [0,1]; output `logits`
  float32 [1,3] (names from `config.ONNX`). No `dynamic_axes` on purpose: the
  demo always infers batch-1, and a fixed shape lets ORT fold shape logic away.
- Model loaded via `agents.policy.load_policy` on CPU, `.eval()`, wrapped in
  `agents.policy.LogitsOnly` (drops the value head — demo only needs argmax).
- Verified post-export with `onnx.checker` + graph inspection: opset 17,
  `frames [1,4,64,64]` -> `logits [1,3]`.

## Parity gate (runs inside the export script; nonzero exit on failure)

- 64 seeded-random float32 [1,4,64,64] inputs in [0,1] **plus 8 real frame
  stacks** from `data/local/frames.npy` (indices with 4 consecutive frames —
  no done in [t-3, t] per the INTERFACES.md sampling rule; frames /255).
- Threshold: max|torch − ort| < 1e-4 over all logits, plus 100% argmax
  agreement on the real stacks. Real stacks matter because dataset frames are
  sparse binary images, a very different activation regime than uniform noise.

## Results — `web/model_baseline.onnx` (from `checkpoints/ppo_baseline.pt`, 9M-step PPO)

```
.venv/bin/python -m scripts.export_onnx --ckpt checkpoints/ppo_baseline.pt --out web/model_baseline.onnx
ONNX_PARITY_OK max_diff=1.255e-05
argmax agreement on real stacks: 8/8 (100%)
```

- File size: **1,366,711 bytes (~1.3 MB)** — fine for a localhost fetch.
- `web/model_dream.onnx`: not yet exported; same command with
  `--ckpt checkpoints/dream_agent.pt --out web/model_dream.onnx` when asked.
