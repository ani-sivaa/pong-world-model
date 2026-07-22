# DECISIONS — eval-utils

## GIF writer: imageio.v3 + Pillow plugin, duration in milliseconds

`gifs._write_gif` (shared by `save_gif` and `side_by_side`) uses

```python
imageio.v3.imwrite(str(out_path), frames, plugin="pillow", extension=".gif",
                   duration=max(1, int(round(1000.0 / fps))), loop=0)
```

Why this and not `imageio.mimsave` (v2): the legacy v2 pillow path changed
`duration` semantics across imageio releases (seconds in older versions,
milliseconds after the ~2.28 alignment with v3), so v2 code is silently
wrong on one side of that boundary. The v3 PillowPlugin has had one stable
contract the whole time: kwargs are forwarded to `PIL.Image.save`, where
`duration` is milliseconds per frame and `loop=0` means loop forever.
`extension=".gif"` pins the codec regardless of the output filename.
Duration is clamped to >= 1 ms; note GIF stores centiseconds, so fps=30
(33 ms) plays back at effectively 30 ms/frame — inherent to the format.

## Label rendering (side_by_side)

- PIL `ImageFont.load_default()` (bitmap font, zero external files — the
  sized `load_default(size=...)` variant needs newer Pillow, so avoided).
- Each clip gets a 16 px tall black strip above it, text drawn white at
  (2, 2), rendered at the clip's *upscaled* width so text isn't blocky.
  Overlong labels are clipped at the strip edge (no wrapping).
- 2 px white vertical separator columns between clips, spanning strip+clip.

## Contract judgment calls (no deviations, only tightenings)

- All three output functions return `Path(out_path)` (normalized), accept
  str or Path, and mkdir parents. `plt.close(fig)` after every savefig.
- Upscaling: `np.repeat` on H then W axes — nearest-neighbor, uint8-
  preserving, equivalent to the suggested `np.kron` but with no dtype/
  overflow subtlety. `scale` must be a positive integer (asserted).
- `save_gif` / `side_by_side` accept grayscale [T,H,W] (replicated to RGB
  via `np.stack`) or already-RGB [T,H,W,3]; dtype must be uint8 (asserted
  with clear messages).
- `side_by_side`: clips must share frame height H (asserted); widths may
  differ. Shorter clips padded by repeating their last frame to max T.
- `line_plot`: xs is the single shared x-sequence; every series length is
  asserted equal to len(xs). figsize (8,5), grid alpha 0.4, dpi=150,
  tight_layout, log_y via `ax.set_yscale("log")`.
- `mse_curve`: strict uint8 [T,H,W] inputs, float32 math on /255-normalized
  pixels, returns float32 [T].
- `summarize_episodes`: means computed in float64, returned as plain python
  float/int (json-serializable). Rates are boolean-mask means over the same
  N, so win_rate + loss_rate + trunc_rate == 1.0.
- `evalutils/__init__.py` re-exports the three submodules via relative
  imports, so `from evalutils import plots, gifs, metrics` works; importing
  the package forces the Agg backend (plots.py calls `matplotlib.use("Agg")`
  before importing pyplot).
