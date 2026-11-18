# Physically calibrated envelope and B-mode GLCM

This is the current focused GLCM validation path. It adapts the
`batch_glcm_envelope_bmode.m` supplied by the PhD student to this repository's
Clarius captures. All signal/image processing and GLCM calculations run in
MATLAB. Python extracts unchanged RF frame bytes, copies YAML, and packages
text reports. No CNN, GLSZM, clustering or automatic tissue selection enters
this experiment.

The older Python texture viewer remains available as a historical prototype.
Its pixel offsets, anisotropic window caps and square shape-processing grid
do not establish physical isotropy. Its mathematical GLCM tests did not
validate the millimeter scale of those measurements.

## Processing and calibration

1. Read an original 16-bit RF frame and its matching YAML.
2. Obtain axial spacing from `1540 / (2 * sample_rate_Hz) * 1000` mm.
3. Obtain lateral positions from receive-element positions times probe pitch.
   YAML pitch is in micrometers. Uniform, zero-steering Cartesian geometry
   is required; curved, steered and nonuniform grids fail explicitly.
4. Form the linear envelope with `abs(hilbert(RF))` along sample rows.
5. Anti-alias before downsampling, then interpolate onto a square physical
   grid. The pilot uses 0.130 by 0.130 mm, without lateral upsampling.
6. Quantize the entire isotropic envelope once per level setting. Separately
   form `20*log10(envelope/Eref)`, clip to -60..0 dB, and quantize B-mode.
7. Calculate symmetric directional GLCMs inside square physical windows.

The recorded uint8 `*_env.raw` stream is already B-mode grayscale, despite
its filename. It is not the uncompressed linear-envelope amplitude. This
experiment derives both analysis domains from RF, matching the supplied
script. It does not apply Hilbert detection or another logarithm to recorded
ENV pixels or to the preferred gamma-0.8 PNGs. Existing exports and masks
remain unchanged.

Anti-aliasing follows the supplied FIR design: order up to 48, normalized
cutoff `0.90 / downsampling_ratio`, zero-phase `filtfilt`, then physical-coordinate
linear interpolation. It serves sampling conversion, rather than optional
speckle denoising. The grid's final center may end less than one spacing unit
before the native final center; this remainder is recorded.

The sound speed is an assumption. Isotropic sampling does not establish
isotropic acoustic resolution. Depth coordinates are relative to the first
stored RF sample; delay samples are recorded without assuming absolute
registration to another stream. No inverse TGC is applied.

## Two features and four directions

With normalized symmetric pair probabilities `P(i,j)`:

- Contrast: `sum(P(i,j) * (i-j)^2)`.
- Entropy: `-sum(P(i,j) * log2(P(i,j)))`, ignoring zero-probability terms.

Both endpoints must be inside the patch and valid ROI. NaNs exclude pixels.
The image-wide quantization is fixed across windows; an optional native-grid
ROI mask excludes pairs without changing that quantization. Zero envelope
amplitude is excluded to preserve the supplied script's convention.

Offsets are `[0 1; -1 0; -1 1; -1 -1]`: horizontal, axial and two diagonals.
Directions stay separate. At 0.130 mm spacing, axial/horizontal pair distance
is 0.130 mm and diagonal distance is about 0.184 mm. They are not pooled as
equal-distance measurements.

Contrast is saved raw, with `contrast/(levels-1)^2` in the summary as a
cross-level rescaling of the same feature. Entropy is in bits. Neither raw
feature values nor bin-normalized contrast are automatically invariant to
quantization, gain, window size or acquisition settings.

## Parameters and pilot results

The sweep uses 8, 16, 32 and 64 quantization levels and 5, 10 and 20 pulse
lengths. Pulse length is `2 * sound_speed / transmit_frequency`. Two cycles
per pulse is the supplied script's assumption, not a measured pulse duration.
The code uses each capture's transmit frequency and records requested and
achieved sizes. Window sides are rounded to odd pixel counts, at least three.
Reported side length is pixel count times isotropic spacing.

At 16 MHz, wavelength is 0.09625 mm and the assumed pulse length is 0.1925 mm:

| Requested window | Requested mm | Side pixels | Achieved side mm |
| --- | ---: | ---: | ---: |
| 5 PL | 0.9625 | 7 | 0.91 |
| 10 PL | 1.9250 | 15 | 1.95 |
| 20 PL | 3.8500 | 29 | 3.77 |

Windows target 80% overlap and require at least 50% valid pixels. Actual mean
overlap and valid pair counts are saved. Only full windows are calculated;
gray plot borders indicate missing full-window estimates, not cropped inputs.

Four actual captures (001, 004, 010, 070) completed all settings, producing
384 per-capture/domain/configuration/direction rows and 96 parameter summaries
that retain direction separation. Source/YAML hashes remain unchanged.

At 64 levels, the smallest pilot windows are only 7–9 pixels per side. They
contain approximately 42–72 horizontal pairs against 4096 possible matrix
entries. Increasing the window improves count support but reduces spatial
locality. For capture 001, B-mode occupied-matrix fractions rise from about
1.41% at 5 PL to 4.79% at 10 PL and 10.32% at 20 PL.

Entropy also depends on sample count: with N directed pairs, a symmetric
matrix has at most 2N occupied entries. A larger-window entropy increase need
not indicate a biological change. These results motivate intermediate
16/32-level, 10-PL comparisons; they do not establish an optimal setting.
The pilot is a technical and parameter-sensitivity check, not segmentation
accuracy, tissue identification or a lesion-detection experiment.

## Run

From the repository root:

```powershell
python -B tools/prepare_clarius_glcm.py --output .\clarius_annotations\physical_glcm\prepared --lzop "C:\Users\abhit\Documents\Hoyt Research\lzop.exe" --cases 001 004 010 070
matlab -batch "addpath('clarius'); test_clarius_glcm_physical('clarius_annotations/physical_glcm/validation.json');"
matlab -batch "addpath('clarius'); batch_glcm_envelope_bmode('clarius_annotations/physical_glcm/prepared/frames.json','clarius_annotations/physical_glcm/pilot');"
matlab -batch "addpath('clarius'); review_physical_glcm('clarius_annotations/physical_glcm/pilot');"
python -B tools/report_physical_glcm.py
```

Use a fresh output directory for a new analysis run; the batch refuses to
overwrite feature MAT files. Supply `'Levels',[16 32]` or
`'WindowPulseLengths',[10 20]` to restrict a batch. The full sweep is the
default. `IsoSpacingMm=0` selects the coarser native spacing per capture;
the default fixed 0.130 mm instead rejects incompatible coarser acquisitions.

For an explicit native-RF ROI:

```matlab
[rf,z,x,m] = read_clarius_rf_physical(rawPath,yamlPath);
result = clarius_glcm_physical(rf,z,x, ...
    'TransmitFrequencyHz',m.transmit_frequency_Hz, ...
    'Levels',32,'WindowPulseLengths',10,'RoiMask',nativeRfMask);
```

Do not resize an existing ENV/PNG mask onto RF by dimensions alone: the
streams have different sample grids. Source geometry and registration must
be established before incorporating MedSAM or reference masks into this path.

## Files and verification

- `clarius_isotropic_envelope.m`: physical resampling and anti-aliasing.
- `clarius_glcm_direction.m`: exact masked symmetric counts and two features.
- `clarius_glcm_physical.m`: shared envelope/B-mode quantization and windows.
- `read_clarius_rf_physical.m`: header/YAML validation and Cartesian calibration.
- `batch_glcm_envelope_bmode.m`: application-specific batch adapter.
- `review_physical_glcm.m`: physical-coordinate figures and parameter tables.
- `test_clarius_glcm_physical.m`: 31 technical checks against MATLAB
  `graycomatrix`, independent pair enumeration, known textures, quantization,
  physical grids, anti-aliasing and metadata guards. Maximum feature reference
  error was zero in the local run.
- `tests/test_prepare_clarius_glcm.py`: three byte-preservation/error tests.

Outputs are under `clarius_annotations/physical_glcm/pilot/`: `review.html`,
`REPORT.md`, `summary.csv`, `parameter_comparison.csv`, physical-axis PNGs and
`*_physical_glcm.mat` files. Maps have layout row-center × column-center ×
direction × feature; NaN means unsupported. The MAT also stores the isotropic
amplitudes, bin images, coordinate axes, options and source provenance.

The supplied script is preserved unchanged in the ignored local audit folder.
SHA256: `E5A1AB8653390369C77B424B5E6ACD2D8DDA20DCCD0A66A9D873339394C18C9C`.
Its 13-feature extractor and hardcoded workstation paths were replaced by
the focused two-feature adapter. The older Python workflows remain labeled
as uncalibrated historical prototypes.

Reference documentation: [Clarius raw format and metadata](https://github.com/clariusdev/research/blob/master/docs/raw-data-format.md),
[MATLAB graycomatrix](https://www.mathworks.com/help/images/ref/graycomatrix.html).
