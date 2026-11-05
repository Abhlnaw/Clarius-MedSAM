# Local GLSZM implementation and comparison

GLSZM (gray-level size-zone matrix) counts connected zones of equal discretized
intensity. `P(i,j)` is the number of zones with gray-level index `i` and size
`j`. The implementation follows the [PyRadiomics GLSZM definitions](https://pyradiomics.readthedocs.io/en/stable/features.html#gray-level-size-zone-matrix-glszm-features),
using 8-connectivity in 2D. It is implemented with the existing NumPy, SciPy, and
scikit-image dependencies; it does not require installing PyRadiomics.

The existing **Combined + GLCM q0.985** workflow remains the default. GLSZM is
available as an experimental alternative for comparison.

## What is implemented

`clarius_glszm.py` exposes:

- `glszm_matrix(gray, mask, levels=8)`: exact matrix for a nonempty 2D ROI.
- `glszm_features(gray, mask, levels=8)`: all 16 standard GLSZM features.
- `local_glszm_features(gray, mask, window, levels=8, step=(8,2))`: two-scale
  local feature maps and a supported-pixel mask.

Discretization is `min(floor(gray * levels), levels - 1)`, then indices are
shifted to 1..levels for the feature formulas. Background is excluded, not
treated as an extra tissue gray level. NaN or arbitrary intensities outside the
mask cannot affect zones. A mask hole can split a zone; diagonal equal-level
pixels remain connected under 8-connectivity.

Each local window constructs its own connected zones. A zone is clipped at
window, image, and parent-mask boundaries. The code does not compute global
component sizes and call them local GLSZM features. A single valid pixel is
sufficient for GLSZM support, unlike GLCM's neighboring-pair requirement.

All 16 ROI features are available. The local maps use eight descriptors at each
of the existing half/full window scales:

| Descriptor | Information summarized |
| --- | --- |
| Small-area emphasis | Weight on small zones |
| Large-area emphasis | Weight on large zones |
| Gray-level nonuniformity, normalized | Concentration of zones among gray levels |
| Size-zone nonuniformity, normalized | Concentration among zone sizes |
| Zone percentage | Number of zones / number of ROI pixels |
| Gray-level variance | Spread of zone gray levels |
| Zone variance | Spread of zone sizes |
| Zone entropy | Entropy of the joint gray-level/zone-size distribution |

The nonnormalized nonuniformities and four joint area/gray-level emphasis
features are included in the ROI API but omitted from local clustering to limit
redundancy. Saved maps contain raw formula values. `log1p` compresses the GLSZM
channels before z-scoring for clustering; that transformation is recorded in
metadata. Brightness-plus-GLSZM uses the existing four intensity channels with
the same block weights as Combined, distributing the texture weight over 16
GLSZM channels instead of eight GLCM channels.

Automatic GLSZM selection uses the same seed scoring, brightness bounds,
spatial opening, envelope, closing, parent clipping, and largest-component rule
as the current selector. Its appearance gate uses the average bounded squared
seed-normalized distance across all 16 GLSZM channels at q0.985. This is a new
descriptor model; the GLCM correlation cutoff does not have an equivalent
raw-value meaning for GLSZM. All choices were fixed before the comparison.

## Exact versus sampled local maps

`--glszm-step 1 1` computes an independent exact window matrix at every masked
pixel. This is expensive for full RF sample grids. The default `8 2` evaluates
exact matrices every eight depth samples and two lateral samples, including the
last row/column, then uses mask-normalized bilinear feature interpolation.
This approximates intermediate feature values; it is not an exact dense GLSZM
at every pixel. Only features are interpolated; input intensities and masks
retain their original native grid. The step and interpolation method are saved.

Zones count native samples, not square millimeters. Unequal RF depth/lateral
sampling and display normalization limit interpretation across acquisitions.
Global exact GLSZM counts are rotation-independent on the 8-connected grid;
anisotropic windows and the sampling lattice do not inherit a general rotation
invariance claim.

## Trying it in the viewer

Use your normal input, checkpoint, and device arguments, with a fresh output
directory and one of:

```powershell
--auto-source glszm --output .\clarius_annotations\exam05_glszm
--auto-source combined_glszm --output .\clarius_annotations\exam05_combined_glszm
```

Leave **Low / High variation** selected and click **Split latest region** for
automatic selection. **GLSZM** and **Combined GLSZM** are source-group inspection
views; switching back to automatic uses the last inspected source. Their maps
are shared in the cache. GLCM and GLSZM source caches remain separate.

`--glcm-levels 8` sets quantization for either descriptor family. The new alias
`--auto-texture-quantile 0.985` also works for the existing correlation-quantile
option. Zero disables the feature-distance gate. The default source remains
Combined/GLCM. Changing GLSZM settings does not reinterpret an already saved
mask; recompute before saving or use a separate output folder.

Saved metadata adds zone connectivity, sampling step, interpolation, clustering
transformation, and generic `distance_gate_*` diagnostics. Historical
`correlation_*` fields remain readable; `correlation_gate_applied` is false for
GLSZM. Use `distance_gate_family` to identify the active model.

## Reproducing the comparison

```powershell
python -B -m unittest discover -s tests -p "test_clarius_*.py"
python -B tools/compare_glszm.py
```

The comparison reads the existing saved parent masks and original images for
exam05 frames 001, 002, 003, 005, 010 and exam04 frames 001, 002. This keeps the
MedSAM prompt/mask identical for all methods and avoids repeating CUDA inference.
Quantization, window extents, group count, and q0.985 are held fixed. Fresh
feature extraction, clustering, and refinement are timed; image I/O, plotting,
and GPU encoding are excluded. GLSZM feature extraction is shared between its
two variants, but its full extraction time is charged to each independent total.

The optional local reference file is the prior experiment's approximate red
outline for exam05 frame 001 only. It is used only for evaluation, never by the
algorithms. It already incorporates parent clipping and the original dark
exclusion, so it is not an expert anatomical reference. Other frames report
mask overlap and size, which cannot establish which result is more accurate.

Generated overlays, masks, timing/metric JSON, and `REPORT.md` go into the ignored
`clarius_annotations/glszm_comparison/` directory. The script also compares dense
and sampled maps on a native-grid patch and checks a finer `4 1` sampling step
on the reference frame. Use `--case EXAM ANNOTATION_FOLDER FRAME` repeatedly
to select other saved cases, `--reference PATH` for a matching local `reference`
array in an NPZ file (`--reference-case EXAM_FRAME` identifies its case), and
`--skip-sensitivity` to omit the extra sampling checks. `--quantiles` controls a
small sensitivity sweep on outlined cases; those measurements do not change
application defaults.

Tests include the published 5x5 GLSZM matrix, independent graph-traversal counts,
known feature values, zone-size conservation, rotations, holes, clipped windows,
mask isolation, lattice-node agreement, single pixels, viewer caching, and
save/reload. These check computation and integration, not anatomical accuracy.

## Results on the existing local data

All **53 Clarius tests passed**. Seven saved frames were compared with identical
parent masks. On exam05 frame 001, the only frame with the approximate red
reference, the results were:

| Method at q0.985 | Dice vs rough reference | Reference covered (recall) | Pixels outside reference |
| --- | ---: | ---: | ---: |
| Current brightness + GLCM correlation | 0.9236 | 91.9% | 6,039 |
| GLSZM | 0.6690 | 50.3% | 97 |
| Brightness + GLSZM | 0.8815 | 89.4% | 11,468 |

Pure GLSZM removes almost all spillover on that frame by also removing about
half of the desired region. Adding brightness retains more tissue but increases
the lower-left extension. The pure GLSZM seed also switches to a deeper
population on exam05 frames 003 and 010; these are visibly different selections,
not evidence of better anatomical segmentation. Brightness-plus-GLSZM overlaps
the current masks more consistently (IoU 0.852-0.947 across the seven frames).

Median measured CPU times over the seven frames, including fresh feature maps,
clustering, and automatic selection, were 4.41 s for the current method, 8.03 s
for GLSZM, and 8.20 s for brightness-plus-GLSZM. These are single-run timings on
the local setup, not a GPU benchmark; the first reference-frame runs were 6.84,
17.31, and 17.72 s respectively.

Sampling approximation was checked separately. On a 128x32 native patch, dense
maps took 3.38 s versus 0.24 s for step (8,2). Feature RMSE divided by each
dense feature's 5th-to-95th percentile range was 0.069-0.196, so intermediate
sampled values must not be presented as exact. The finer full-frame step (4,1)
took 68.2 s for feature extraction. It changed pure-GLSZM mask IoU to 0.636
relative to step (8,2), with rough Dice falling to 0.4872. The brightness-plus-
GLSZM mask was more stable (IoU 0.974), but Dice was still lower at 0.8741.

A seven-point descriptor-quantile check (0.90, 0.95, 0.98, 0.985, 0.99, 0.995,
1.00) did not beat the current reference-frame result. The best scores in that
check were 0.7701 for GLSZM and 0.9046 for brightness-plus-GLSZM, both at q1.00.
This is a sensitivity check on the same rough reference, not held-out tuning or
a claim that every possible GLSZM configuration is inferior.

**Recommendation:** keep the current Combined/GLCM q0.985 default. Use GLSZM
as an experimental texture descriptor and inspect the generated comparison
overlays. Its matrix calculations are validated, but this fixed GLSZM-based
region selector did not improve the outlined boundary and was less stable when
brightness was omitted. Expert masks across more acquisitions are needed to
judge general segmentation accuracy.
