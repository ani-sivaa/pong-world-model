"""Remote execution over the frozen Modal app/image/volume (infra.modal_common).

Remote side: run_module_gpu / run_module_cpu execute any repo module's CLI via
runpy inside the container (repo at /root, artifacts on /vol via WM_ROOT=/vol),
commit the volume, and RETURN a result dict — errors come back as data, never
as raised exceptions, so the caller decides what to do.

Local side (main-thread API):
    launch(module, argv, gpu=True)                -> result dict (Modal only)
    run_anywhere(module, argv, scale_fallback)    -> Modal, else loud FAILURES.md
                                                     entry + local subprocess retry
    fetch(remote_path, local_path)                -> `modal volume get --force`

CLI (how the main thread starts remote jobs from background shells):
    .venv/bin/python -m infra.remote --module wm.train --gpu -- --scale full
    .venv/bin/python -m infra.remote --module pong.collect --cpu -- --scale full
    .venv/bin/python -m infra.remote --fetch checkpoints/wm_v1.pt checkpoints/wm_v1.pt
Everything after the bare `--` is passed verbatim to the target module's argv.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import runpy
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import modal

import config
from infra.modal_common import VOL_NAME, app, image, volume

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PY = str(REPO_ROOT / ".venv" / "bin" / "python")
FAILURES_MD = REPO_ROOT / "FAILURES.md"

# Generous, fixed at decoration time (Modal 1.x can't change timeout per call):
# the longest phase cap plus margin for container start + data load + commit.
FUNC_TIMEOUT = max(config.CAPS.values()) + 900  # 7200 + 900 = 8100 s


# --------------------------------------------------------------- remote ----
def _run_module_body(module: str, argv: list[str], timeout_hint: int) -> dict:
    """Shared container body. Returns {module, argv, seconds, ok, error}."""
    t0 = time.time()
    result = {
        "module": module,
        "argv": list(argv),
        "timeout_hint": timeout_hint,
        "ok": False,
        "error": None,
        "seconds": 0.0,
    }
    try:
        os.chdir("/root")
        if "/root" not in sys.path:
            sys.path.insert(0, "/root")
        os.environ["WM_ROOT"] = "/vol"  # image sets it too; belt and braces
        for sub in ("data", "checkpoints", "results"):
            os.makedirs(f"/vol/{sub}", exist_ok=True)
        sys.argv = [module] + list(argv)
        runpy.run_module(module, run_name="__main__")
        result["ok"] = True
    except SystemExit as e:  # argparse errors / explicit sys.exit inside module
        if e.code in (0, None):
            result["ok"] = True
        else:
            result["error"] = f"SystemExit({e.code!r})\n{traceback.format_exc()}"
    except BaseException:
        result["error"] = traceback.format_exc()
    finally:
        result["seconds"] = round(time.time() - t0, 2)
        try:
            volume.commit()
        except Exception as e:  # surface commit failure without masking the run error
            msg = f"volume.commit() failed: {e!r}"
            result["error"] = f"{result['error']}\n{msg}" if result["error"] else msg
            result["ok"] = False
    return result


@app.function(image=image, gpu="T4", cpu=8, volumes={"/vol": volume}, timeout=FUNC_TIMEOUT)
def run_module_gpu(module: str, argv: list[str], timeout_hint: int = 0) -> dict:
    """Run `python -m <module> <argv...>` on a T4. timeout_hint is advisory only."""
    return _run_module_body(module, argv, timeout_hint)


@app.function(image=image, cpu=8, volumes={"/vol": volume}, timeout=FUNC_TIMEOUT)
def run_module_cpu(module: str, argv: list[str], timeout_hint: int = 0) -> dict:
    """CPU-only variant (8 cores) for data collection etc."""
    return _run_module_body(module, argv, timeout_hint)


_CHUNK = 1_000_000  # b64 of 1e6 bytes ≈ 1.34 MB, under Modal's ~2 MiB inline-result cap


@app.function(image=image, volumes={"/vol": volume}, timeout=900)
def vol_read(remote_path: str, offset: int = 0, length: int = _CHUNK) -> dict:
    """Read a chunk of a volume file (or list a dir) via the function channel.

    Exists because `modal volume get` downloads through modal-storage.com,
    which some networks reset; inline function results ride the gRPC API
    channel instead. Chunks must stay under ~2 MiB to remain inline.
    """
    try:
        volume.reload()  # warm containers: pick up the latest commit
    except Exception:
        pass
    p = os.path.join("/vol", remote_path.lstrip("/"))
    if os.path.isdir(p):
        files = []
        for root, _, names in os.walk(p):
            for n in names:
                files.append(os.path.relpath(os.path.join(root, n), p))
        return {"type": "dir", "files": sorted(files)}
    size = os.path.getsize(p)
    with open(p, "rb") as f:
        f.seek(offset)
        data = f.read(length)
    return {"type": "file", "size": size, "offset": offset,
            "data_b64": base64.b64encode(data).decode()}


# ---------------------------------------------------------------- local ----
def launch(module: str, argv: list[str] | None = None, gpu: bool = True,
           timeout_hint: int = 0) -> dict:
    """Run the module remotely via an ephemeral app run; return the result dict."""
    argv = list(argv or [])
    fn = run_module_gpu if gpu else run_module_cpu
    with modal.enable_output():
        with app.run():
            return fn.remote(module, argv, timeout_hint)


def _replace_scale(argv: list[str], scale: str) -> list[str]:
    """Rewrite any `--scale <x>` / `--scale=<x>` to the fallback scale."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--scale" and i + 1 < len(argv):
            out += ["--scale", scale]
            i += 2
        elif a.startswith("--scale="):
            out.append(f"--scale={scale}")
            i += 1
        else:
            out.append(a)
            i += 1
    return out


def _log_failure(module: str, argv: list[str], error: str, local_cmd: list[str]) -> None:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = (
        f"\n## MODAL FAILURE — falling back to local — {stamp}\n"
        f"- module: `{module}`\n"
        f"- argv: `{argv}`\n"
        f"- remote error:\n\n```\n{error.rstrip()}\n```\n\n"
        f"- action: local retry: `{' '.join(local_cmd)}`\n"
    )
    with open(FAILURES_MD, "a") as f:
        f.write(entry)


def _append_failures(line: str) -> None:
    with open(FAILURES_MD, "a") as f:
        f.write(line)


def run_anywhere(module: str, argv: list[str] | None = None,
                 scale_fallback: str = "local", gpu: bool = True) -> dict:
    """Try Modal; on ANY exception or ok=False, log loudly to FAILURES.md and
    rerun locally (subprocess, WM_ROOT unset, --scale rewritten to fallback)."""
    argv = list(argv or [])
    try:
        result = launch(module, argv, gpu=gpu)
        if result.get("ok"):
            return result
        error = result.get("error") or "remote returned ok=False with no error string"
    except BaseException:
        error = traceback.format_exc()

    local_argv = _replace_scale(argv, scale_fallback)
    cmd = [VENV_PY, "-m", module, *local_argv]
    _log_failure(module, argv, error, cmd)

    env = {k: v for k, v in os.environ.items() if k != "WM_ROOT"}  # local paths
    t0 = time.time()
    tail: list[str] = []
    try:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)
        assert proc.stdout is not None
        for line in proc.stdout:  # stream while keeping a tail for the dict
            print(line, end="", flush=True)
            tail.append(line)
            if len(tail) > 60:
                tail.pop(0)
        returncode = proc.wait()
    except BaseException:
        returncode = -1
        tail.append(traceback.format_exc())
    seconds = round(time.time() - t0, 2)

    ok = returncode == 0
    _append_failures(
        f"- local fallback result: ok={ok} returncode={returncode} "
        f"seconds={seconds}\n"
    )
    return {
        "module": module,
        "argv": local_argv,
        "ok": ok,
        "local_fallback": True,
        "returncode": returncode,
        "seconds": seconds,
        "remote_error": error,
        "error": None if ok else "".join(tail[-30:]),
    }


def fetch(remote_path: str, local_path: str) -> str:
    """Copy a file/dir off worldmodel-vol. remote_path is volume-relative
    (e.g. "checkpoints/wm_v1.pt"). Tries `modal volume get --force`; if that
    fails (this network resets connections to modal-storage.com, the blob CDN
    the CLI downloads through), falls back to chunked reads over the Modal
    function-call channel, which rides the working api.modal.com gRPC path."""
    local = Path(local_path)
    if not local.is_absolute():
        local = REPO_ROOT / local
    local.parent.mkdir(parents=True, exist_ok=True)
    cmd = [VENV_PY, "-m", "modal", "volume", "get", "--force",
           VOL_NAME, remote_path, str(local)]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode == 0:
        return str(local)
    print(f"[infra.remote] `modal volume get` failed (rc={proc.returncode}); "
          "falling back to function-channel fetch", file=sys.stderr)
    _fetch_via_function(remote_path, local)
    return str(local)


def _fetch_via_function(remote_path: str, local: Path) -> None:
    """Pull a file or directory off the volume via vol_read chunks."""

    def pull_file(rpath: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        offset = 0
        with open(dest, "wb") as out:
            while True:
                part = vol_read.remote(rpath, offset, _CHUNK)
                data = base64.b64decode(part["data_b64"])
                out.write(data)
                offset += len(data)
                if offset >= part["size"] or not data:
                    break
        print(f"[infra.remote] fetched {rpath} -> {dest} ({offset} bytes)")

    with modal.enable_output(), app.run():
        probe = vol_read.remote(remote_path, 0, 0)
        if probe["type"] == "dir":
            for rel in probe["files"]:
                pull_file(f"{remote_path.rstrip('/')}/{rel}", local / rel)
        else:
            pull_file(remote_path, local)


# ------------------------------------------------------------------ CLI ----
def _main() -> int:
    # Everything after a bare `--` goes verbatim to the target module.
    cli = sys.argv[1:]
    if "--" in cli:
        cut = cli.index("--")
        own, passthrough = cli[:cut], cli[cut + 1:]
    else:
        own, passthrough = cli, []

    p = argparse.ArgumentParser(
        prog="infra.remote",
        description="Run a repo module on Modal (or fetch volume artifacts).")
    p.add_argument("--module", help="dotted module to run, e.g. wm.train")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--gpu", action="store_true", help="T4 (default)")
    mode.add_argument("--cpu", action="store_true", help="CPU-only, 8 cores")
    p.add_argument("--anywhere", action="store_true",
                   help="use run_anywhere (local fallback on any failure)")
    p.add_argument("--scale-fallback", default="local",
                   help="scale substituted into argv for the local fallback")
    p.add_argument("--fetch", nargs=2, metavar=("REMOTE", "LOCAL"),
                   help="fetch from the volume instead of running a module")
    args = p.parse_args(own)

    if args.fetch:
        out = fetch(args.fetch[0], args.fetch[1])
        print("FETCHED", out)
        return 0

    if not args.module:
        p.error("--module is required unless --fetch is used")
    gpu = not args.cpu
    if args.anywhere:
        result = run_anywhere(args.module, passthrough,
                              scale_fallback=args.scale_fallback, gpu=gpu)
    else:
        result = launch(args.module, passthrough, gpu=gpu)
    print("=== REMOTE RESULT ===")
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(_main())
