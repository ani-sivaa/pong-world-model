"""GIF helpers: single-clip animations and labeled side-by-side montages.

Frame conventions: uint8 numpy arrays, either [T, H, W] (grayscale, replicated
to RGB before writing) or [T, H, W, 3] (RGB). Upscaling is nearest-neighbor via
np.repeat on the H and W axes (dtype-preserving, equivalent to np.kron with a
ones block).

Writer: imageio.v3.imwrite(..., plugin="pillow", extension=".gif",
duration=<ms per frame>, loop=0). The v3 Pillow plugin passes `duration`
straight to PIL, which takes MILLISECONDS; loop=0 means loop forever. The
legacy v2 mimsave path changed duration semantics (seconds vs ms) across
imageio releases, so it is avoided entirely.
"""
from pathlib import Path

import numpy as np
import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont

_LABEL_STRIP_H = 16  # pixel height of the black text strip above each clip
_SEPARATOR_W = 2     # white separator width (pixels) between montage columns


def _as_path_with_parents(out_path):
    """Normalize str/Path to Path and create parent directories."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def _to_rgb(frames, name):
    """Validate uint8 [T,H,W] or [T,H,W,3] and return uint8 [T,H,W,3]."""
    frames = np.asarray(frames)
    assert frames.dtype == np.uint8, f"{name} must be uint8, got {frames.dtype}"
    assert frames.ndim in (3, 4), (
        f"{name} must be [T,H,W] or [T,H,W,3], got shape {frames.shape}"
    )
    assert frames.shape[0] >= 1, f"{name} needs at least one frame, got shape {frames.shape}"
    if frames.ndim == 3:
        return np.stack([frames, frames, frames], axis=-1)
    assert frames.shape[-1] == 3, (
        f"{name} last axis must be 3 (RGB), got shape {frames.shape}"
    )
    return frames


def _upscale(frames_rgb, scale):
    """Nearest-neighbor upscale uint8 [T,H,W,3] by an integer factor on H and W."""
    assert float(scale) == int(scale) and int(scale) >= 1, (
        f"scale must be a positive integer, got {scale!r}"
    )
    scale = int(scale)
    if scale == 1:
        return frames_rgb
    return np.repeat(np.repeat(frames_rgb, scale, axis=1), scale, axis=2)


def _write_gif(frames_rgb, out_path, fps):
    """Write uint8 [T,H,W,3] frames to out_path as an infinitely looping GIF."""
    assert fps > 0, f"fps must be positive, got {fps!r}"
    duration_ms = max(1, int(round(1000.0 / fps)))
    frames_rgb = np.ascontiguousarray(frames_rgb)
    iio.imwrite(
        str(out_path),
        frames_rgb,
        plugin="pillow",
        extension=".gif",
        duration=duration_ms,
        loop=0,
    )


def _label_strip(text, width):
    """Render `text` white-on-black onto a uint8 [_LABEL_STRIP_H, width, 3] strip."""
    img = Image.new("RGB", (int(width), _LABEL_STRIP_H), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((2, 2), str(text), fill=(255, 255, 255), font=ImageFont.load_default())
    return np.asarray(img, dtype=np.uint8)


def save_gif(frames, out_path, fps=30, scale=4):
    """Save a clip as a looping GIF with nearest-neighbor upscaling.

    Args:
        frames: uint8 [T, H, W] (grayscale, replicated to RGB) or [T, H, W, 3].
        out_path: str or Path for the .gif; parent dirs are created.
        fps: playback rate; per-frame duration is round(1000/fps) milliseconds.
        scale: positive integer nearest-neighbor upscale factor for H and W.

    Returns:
        Path(out_path).
    """
    out_path = _as_path_with_parents(out_path)
    rgb = _upscale(_to_rgb(frames, "frames"), scale)
    _write_gif(rgb, out_path, fps)
    return out_path


def side_by_side(clips, labels, out_path, fps=30, scale=4):
    """Save a horizontal montage of clips, each with a text label strip, as a GIF.

    Clips may differ in length T; shorter clips are padded by repeating their
    last frame up to max(T). Each clip is upscaled by `scale`, topped with a
    16px black strip carrying its label (PIL default font, white text), and the
    columns are joined with 2px white vertical separators into one
    [T, 16 + H*scale, sum(W_i*scale) + separators, 3] RGB animation.

    Args:
        clips: non-empty list of uint8 [T_i, H, W] (or [T_i, H, W, 3]) arrays;
            all clips must share the same frame height H (widths may differ).
        labels: list of str, one per clip.
        out_path: str or Path for the .gif; parent dirs are created.
        fps: playback rate; per-frame duration is round(1000/fps) milliseconds.
        scale: positive integer nearest-neighbor upscale factor for H and W.

    Returns:
        Path(out_path).
    """
    assert isinstance(clips, (list, tuple)) and len(clips) > 0, (
        "clips must be a non-empty list of uint8 arrays"
    )
    assert len(labels) == len(clips), (
        f"got {len(labels)} labels for {len(clips)} clips"
    )
    out_path = _as_path_with_parents(out_path)

    rgb_clips = [_to_rgb(clip, f"clips[{i}]") for i, clip in enumerate(clips)]
    heights = sorted({c.shape[1] for c in rgb_clips})
    assert len(heights) == 1, (
        f"all clips must share the same frame height H, got heights {heights}"
    )

    t_max = max(c.shape[0] for c in rgb_clips)
    columns = []
    for clip, label in zip(rgb_clips, labels):
        t = clip.shape[0]
        if t < t_max:  # pad by repeating the last frame
            pad = np.repeat(clip[-1:], t_max - t, axis=0)
            clip = np.concatenate([clip, pad], axis=0)
        clip = _upscale(clip, scale)
        strip = _label_strip(label, clip.shape[2])          # [16, W', 3]
        strip = np.repeat(strip[None, :, :, :], t_max, axis=0)  # [T, 16, W', 3]
        columns.append(np.concatenate([strip, clip], axis=1))

    total_h = columns[0].shape[1]
    separator = np.full((t_max, total_h, _SEPARATOR_W, 3), 255, dtype=np.uint8)
    parts = []
    for j, col in enumerate(columns):
        if j > 0:
            parts.append(separator)
        parts.append(col)
    montage = np.concatenate(parts, axis=2)

    _write_gif(montage, out_path, fps)
    return out_path
