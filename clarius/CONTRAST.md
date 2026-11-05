# Clarius image formation and contrast exploration

This experiment changes image rendering only. It does not generate outlines,
classify dark regions, read reference masks, or change segmentation code.
Existing annotation images and masks remain untouched.

**Preferred rendering: `env_gamma08`**, selected after reviewing the comparison
images. Use the recorded Clarius B-mode stream with gamma 0.8, then resize the
entire frame to 1418 × 853. **The current image-processing/export stage runs in
MATLAB.** The original contrast exploration was implemented in Python; those
historical experiments remain reproducible below.

## Current MATLAB export pipeline

Python prepares unchanged RAW frame bytes from the existing capture manifest;
it performs archive extraction, LZO decompression, and timestamp-based frame
selection only. It does not import NumPy/Pillow, apply contrast, resize samples,
or write image files. Capture identifiers and the non-`exam##` selection are
preserved. The one-frame RAW headers record one frame; timestamps and sample
bytes remain intact.

MATLAB `export_clarius_bmode` reads those bytes, applies
`uint8(floor(255 * (double(env)/255).^0.8 + 0.5))`, resizes the complete frame
with `imresize(...,'bilinear','Antialiasing',true)`, and writes PNGs and JSON
provenance. No Hilbert transform, second logarithm, or per-image min/max
normalization is applied to the already processed uint8 B-mode stream.
MATLAB documents its resizing kernel and antialiasing behavior in
[imresize](https://www.mathworks.com/help/matlab/ref/imresize.html).

Run from the repository root:

```powershell
python tools/prepare_clarius_frames.py --manifest .\clarius_annotations\manual_segmentation\manifest.json --output .\clarius_annotations\matlab_env_gamma08\prepared --lzop "C:\Users\abhit\Documents\Hoyt Research\lzop.exe"
matlab -batch "addpath('clarius'); export_clarius_bmode('clarius_annotations/matlab_env_gamma08/prepared/frames.json','clarius_annotations/matlab_env_gamma08');"
```

The 157 model inputs are in `clarius_annotations/matlab_env_gamma08/images`.
Use `--cases 001 105` and fresh output folders for a small preparation trial.
For a decompressed single exam, `runme2_export.m` uses the same MATLAB rendering
function and writes `exported_images_env_gamma08`. Older RF exports are optional
and kept in a separate legacy folder.

Checks:

```powershell
python -m unittest discover -s tests -p "test_clarius_frame_preparation.py" -v
matlab -batch "addpath('clarius'); test_clarius_bmode"
```

The Python checks verify exact sample-byte/timestamp preservation, frame
selection, and truncated-input rejection. MATLAB checks known gamma values,
black preservation, full-frame dimensions, constant-tone resizing, RAW array
orientation, and integer timestamp precision. MATLAB can also compare its
images with the historical Python exports by accepting that image folder as
an optional third argument. Rendering SSIM in that audit measures exporter
agreement, not segmentation accuracy. Small grayscale rounding differences
between MATLAB and Pillow resizing are possible; outputs are not claimed to
be byte-identical.

## Findings

The existing MATLAB exporter constructs an image from RF:

```text
E = abs(hilbert(RF, along depth))
L = 20 log10(1 + E)
gray = (L - min(L)) / (max(L) - min(L))
```

Its `-100` additive offset cancels during min/max normalization; changing that
offset cannot fix the gray haze. The normalization maps the entire frame's
observed range to black–white. In the examined recordings this lifts many
weak echoes into visible gray and makes low-echo regions less distinct.

The archives also contain **`*_env.raw.lzo`**, an already 8-bit Clarius B-mode
stream. Our earlier annotation exporter used only RF. Clarius documents these
as separate data streams, with the B-mode data in pre-scan-converted ultrasound
coordinates: [official format documentation](https://github.com/clariusdev/research/blob/master/docs/raw-data-format.md).
The B-mode stream should not receive another Hilbert transform or logarithm.

Eight renderings were visually compared on 12 recordings, then three candidates
were exported for all 157 non-`exam##` recordings:

| Output | Mapping | Finding |
| --- | --- | --- |
| `env_native` | Preserve original 0–255 grayscale before full-frame resize | Unmodified grayscale baseline for the Clarius-style appearance. |
| `env_gamma08` | `(gray / 255)^0.8` | Selected rendering; mildly raises midtones while keeping exact black black. |
| `rf_dr50` | RF amplitude dB, 50 dB window below its 99.9th-percentile amplitude | Reduces the washed-out RF appearance without switching data streams. |

The pilot additionally includes 40/60 dB RF windows, a fixed B-mode intensity
window, and mild CLAHE. The 40 dB RF window suppressed more weak signal; 60 dB
retained more weak gray. CLAHE often emphasized speckle and was not selected
as the default. These are visual observations, not measured lesion-detection
improvements or a quantitative match to a website screenshot.

Across all 157 images, the median of per-frame **native** intensity medians
was 27 for original B-mode, 42 for gamma 0.8 B-mode, and 80 for 50 dB RF.
The median native black-pixel fractions were 19.42%, 19.42%, and 4.70%,
respectively. These statistics describe different renderings/acquisitions;
they are not image-quality scores.

## RF window definition

```text
A = percentile(E, 99.9)
relative_db = 20 log10(E / A)
gray = clip((relative_db + D) / D, 0, 1)
```

`D` is the display dynamic range in dB. Zero signal maps to black. The upper
0.1% of amplitudes may saturate by design. `A` is recalculated per frame;
this is a robust display normalization, not calibrated gain or backscatter.
Clarius exposes dynamic range and noise-floor controls, but our RF display
window is not a recovered internal Clarius lookup table:
[official parameters](https://github.com/clariusdev/research/blob/master/docs/low-level-parameters.md).

## Acquisition and geometry checks

All 157 selected RF/B-mode first frames have exactly matching recorded
timestamps. Their metadata reports equal imaging depths, while RF uses
MLA 1 and B-mode MLA 8. RF has 192 lateral lines (184 in four captures),
whereas B-mode has 384. The recorded lateral element extents differ in
cases **105–107 and 117–120**, so existing RF masks cannot simply be assumed
to match B-mode images. Even matching metadata does not establish pixel-level
correspondence between distinct acquisition streams.

Every source frame is resized in full to **1418 × 853**, the currently requested
viewing grid. No source rows or columns are cropped, and no registration or
warping is performed. The fixed display aspect is not physically calibrated
scan conversion. The native B-mode stream itself also lacks the final display
scan conversion and potentially other processing present in the website image;
an exact reproduction of that screenshot is not established.

## Reproduce the historical Python exploration

Use the scientific Python environment containing NumPy, SciPy, Pillow and
scikit-image, plus the existing `lzop.exe`:

```powershell
python tools/explore_clarius_contrast.py --manifest .\clarius_annotations\manual_segmentation\manifest.json --output .\clarius_annotations\contrast_exploration\pilot --lzop "C:\path\to\lzop.exe" --cases 001 004 010 025 031 046 055 070 090 096 132 154 --panels
python tools/explore_clarius_contrast.py --manifest .\clarius_annotations\manual_segmentation\manifest.json --output .\clarius_annotations\contrast_exploration\all_images --lzop "C:\path\to\lzop.exe" --variants env_native env_gamma08 rf_dr50
```

The reader prepares only the required first RF frame and closest-timestamp
B-mode frame from the same archive. It does not extract archive member paths
to disk or decompress whole cines unnecessarily. Each output manifest records
sample dimensions, selected frame, timestamp difference, acquisition metadata,
tone settings, native intensity statistics, and output hashes. Failed cases
are reported instead of silently omitted. Generated data stays in the ignored
`clarius_annotations/` folder.

To export only the preferred rendering to a fresh output folder, use:

```powershell
python tools/explore_clarius_contrast.py --manifest .\clarius_annotations\manual_segmentation\manifest.json --output .\clarius_annotations\contrast_gamma08 --lzop "C:\path\to\lzop.exe" --variants env_gamma08
```

Numerical tests verify the Hilbert depth axis, known amplitude-to-dB mappings,
monotonic tone curves, black preservation, unchanged grids and zero-signal
behavior. No SSIM against unrelated website anatomy is calculated.
