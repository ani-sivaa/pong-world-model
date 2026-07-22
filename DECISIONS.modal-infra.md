# DECISIONS — modal-infra

Owner: modal-infra agent. Files: `infra/remote.py`, `infra/hello.py`, this file.
`infra/modal_common.py` (app/image/volume) is frozen main-thread spec — imported,
never redefined, so the prebuilt image cache stays valid.

## Timeout strategy
Modal 1.x fixes a function's `timeout` at decoration time; it cannot be changed
per call. Both runner functions therefore use one generous ceiling:
`FUNC_TIMEOUT = max(config.CAPS.values()) + 900 = 8100 s` (longest phase cap,
wm_train=7200, plus margin for container start, data loading, and the volume
commit). Per-phase enforcement stays where it belongs — inside each training
module via `config.CAPS` — and the functions accept an advisory `timeout_hint`
arg that is merely echoed into the result dict for logging. If Modal itself
kills the call at 8100 s, `.remote()` raises locally and `run_anywhere` treats
it like any other failure (log + local fallback).

## Errors return, never raise
`_run_module_body` catches `BaseException` (and `SystemExit` — argparse exits
and `sys.exit(0)` must not count as failures; only nonzero codes do) and returns
`{module, argv, timeout_hint, seconds, ok, error}` with the full traceback
string. Rationale: a raised remote exception surfaces locally as a
`modal.exception.RemoteError`-ish wrapper with a mangled traceback, and — worse —
would skip our `finally: volume.commit()`, losing partial artifacts (e.g. a
checkpoint saved before the crash). Returning data keeps the overnight
orchestrator in control: it inspects `ok` and decides retry/fallback/skip.
`volume.commit()` failures are appended to `error` and flip `ok` to False.

## run_anywhere fallback semantics
1. `launch()` (ephemeral `app.run()` + `.remote()`).
2. On ANY local exception (auth, network, timeout) OR a returned `ok=False`:
   append a loud `## MODAL FAILURE` section to FAILURES.md (UTC timestamp,
   module, argv, full traceback, exact local retry command), rewrite argv
   replacing `--scale <x>` / `--scale=<x>` with `scale_fallback` (default
   "local"; no-op if `--scale` absent), then rerun via subprocess
   `[.venv python, -m, module, *argv]` from the repo root with **WM_ROOT
   removed** so artifacts land in local `data/ checkpoints/ results/`.
3. Local output streams live and the last ~60 lines are kept; a
   `- local fallback result: ok=… returncode=…` line is appended to the same
   FAILURES.md entry. Returned dict carries `local_fallback: True` and the
   original `remote_error` so nothing is silently swallowed.

## Modal 1.x quirks hit (v1.5.2, profile ani-sivaa)
- **`modal volume get` is unusable on this network.** The CLI downloads via the
  blob CDN `clobber.modal-storage.com`, and this network resets TLS to that
  host (`curl: (35) Connection reset by peer`, reproducible, even for a 48-byte
  file — volume reads always route through blob storage). `api.modal.com` gRPC
  works fine. `fetch()` therefore tries the CLI first (`--force` for
  overwrite), and on nonzero exit falls back to `vol_read`, a small Modal
  function that returns base64 chunks of ≤1 MB (inline function results under
  ~2 MiB ride the gRPC channel, never the blob CDN). Directory fetches are
  handled by a `type: dir` listing probe + per-file pulls. Verified: sentinel
  retrieved byte-exact via the fallback.
- `modal.enable_output()` must wrap `app.run()` or build/run logs are invisible.
- Remote containers always have the `modal` client installed even though the
  frozen image doesn't pip-install it — importing `infra.remote` remotely works.
- `infra/` has no `__init__.py` (namespace package). Fine on py3.11 for both
  `python -m infra.remote` and remote `runpy.run_module("infra.hello")`; not
  adding one since ownership is limited to remote.py/hello.py.
- The image env already sets `WM_ROOT=/vol`, so `config` resolves volume paths
  even at container import time; the body re-sets it defensively and pre-creates
  `/vol/{data,checkpoints,results}`.
- GPU function uses `gpu="T4", cpu=8` — 8 cores are cheap next to the GPU and
  help the dataloader.

## Verification (2026-07-22, all green)
- `.venv/bin/python -m infra.remote --module infra.hello --cpu -- --tag cputest`
  → ok=True, sentinel `/vol/results/hello_cputest.txt`.
- same `--gpu` → ok=True, `cuda_available: True`, `Tesla T4`.
- `--fetch results/hello_gputest.txt /tmp/wm_fetch_test/hello_gputest.txt`
  → byte-exact roundtrip (via function-channel fallback; CLI path blocked).
- `--module infra.does_not_exist --cpu --anywhere -- --scale full` → remote
  ImportError returned (not raised), FAILURES.md entry written, argv rewritten
  to `--scale local`, local subprocess attempted, its failure captured in the
  dict. Entry marked "[infra self-test — expected failure, ignore]".

## CLI incantations for the main thread
```bash
# remote GPU training / CPU collection (run from repo root, background shells):
.venv/bin/python -m infra.remote --module pong.collect     --cpu --anywhere -- --scale full
.venv/bin/python -m infra.remote --module wm.train         --gpu --anywhere -- --scale full
.venv/bin/python -m infra.remote --module agents.train_ppo --gpu --anywhere -- --scale full
# (drop --anywhere for Modal-only, exit code 1 on ok=False; result dict printed
#  as JSON after "=== REMOTE RESULT ===")

# artifact retrieval (remote path is volume-relative):
.venv/bin/python -m infra.remote --fetch checkpoints/wm_v1.pt      checkpoints/wm_v1.pt
.venv/bin/python -m infra.remote --fetch results/                  results/
# or from python: from infra.remote import fetch, launch, run_anywhere
```
