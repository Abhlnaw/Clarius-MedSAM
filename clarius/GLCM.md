# Local GLCM texture refinement

The default **Low / High variation** mode now automatically refines a Combined
or GLCM proposal using `clarius_auto.py`. The GLCM math below describes the source
features. See [the current automatic workflow](README.md) for controls and output.

The viewer now defaults to **Combined**: local brightness statistics plus
lower-weight GLCM descriptors. **Brightness** restores the previous approach,
and **GLCM** keeps pure co-occurrence analysis. See [the workflow guide](README.md)
for current controls. The GLCM mathematics below applies to the GLCM block in
both Combined and GLCM modes.

## What a GLCM measures

A gray-level co-occurrence matrix (GLCM) is a histogram of pairs of pixel
intensities. Choose a spatial offset, such as one column to the right. For every
valid pair, count a pixel with quantized gray level `i` followed by its neighbor
with level `j` in entry `C[i,j]`. Divide by the total count to obtain probabilities
`P[i,j]` whose sum is one.

For example, the row `[0, 0, 1, 1]` has three right-neighbor pairs: `(0,0)`,
`(0,1)`, and `(1,1)`. Before normalization its directed matrix is:

```text
        neighbor
         0  1
pixel 0 [1, 1]
      1 [0, 1]
```

This implementation adds the transpose to make the matrix symmetric. The
normalized matrix becomes `[[1/3, 1/6], [1/6, 1/3]]`. Both `(i,j)` and `(j,i)`
contribute equally, so reversing the pair direction does not change its features.

An intensity histogram ignores where pixels occur. Two patches can contain the
same number of bright and dark pixels, hence the same mean and standard deviation,
but arrange those pixels as a checkerboard or broad stripes. Their pair counts,
and therefore their GLCM features, differ.

GLCM is a feature extractor. It does not itself choose segmentation boundaries
or decide that a region is abnormal. The separate grouping and selection steps
convert the local feature maps into a refined mask.

## Features calculated here

For each symmetric normalized matrix, `i` and `j` are gray-level indices:

| Feature | Formula | Interpretation |
| --- | --- | --- |
| Contrast | `sum(P[i,j] * (i-j)^2)` | Large neighboring brightness differences increase it. |
| Homogeneity | `sum(P[i,j] / (1+(i-j)^2))` | Similar neighboring levels increase it; range is above zero through one. |
| Energy | `sqrt(sum(P[i,j]^2))` | Concentrated pair probabilities increase it; it is not acoustic energy. |
| Correlation | `sum(P[i,j]*(i-mu_i)*(j-mu_j)) / (sigma_i*sigma_j)` | Measures the relationship between paired levels, from -1 to 1. |

The implementation defines correlation as one when marginal variance is zero,
matching scikit-image's convention. A constant patch therefore has contrast zero,
homogeneity one, energy one, and correlation one. High correlation alone does not
mean interesting tissue. Dark and bright constant patches have the same features.

These definitions follow the [scikit-image GLCM documentation](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycoprops).
The tests compare the implementation directly with `graycomatrix` and
`graycoprops`, and independently enumerate pairs for masks containing holes.

## Processing the selected MedSAM region

1. **Quantize brightness.** The viewer supplies display intensities normalized
   to `[0,1]`. Map them into 8 gray levels with
   `min(floor(gray * 8), 7)`. This creates a 8 by 8 histogram instead of a
   sparse 256 by 256 one. It reduces computation and small intensity differences,
   but also loses brightness detail. The quantizer is fixed in normalized display
   units; it is not fitted separately to each local neighborhood.
2. **Choose two neighborhood sizes.** The large window is approximately one
   eighth of the mask's occupied height and width, capped at 129 rows by 33 columns. Brightness mode retains quarter-mask windows without these caps. The small window is half
   that size. Sizes are rounded to odd numbers and are at least three pixels.
   The window extent comes from the selected mask, not the whole image.
3. **Count local pairs.** At every masked pixel, construct a GLCM for each local
   window and each offset `(row_delta, column_delta)`:
   `(0,1)`, `(1,1)`, `(1,0)`, and `(1,-1)`. These are horizontal, diagonal,
   vertical, and opposite diagonal neighbors on the sample grid. Diagonal pairs
   are sqrt(2) grid units apart, whereas axial pairs are one grid unit apart.
4. **Respect the mask and window.** Both pair endpoints must lie inside both
   the MedSAM mask and the local window. Background, holes, and out-of-image
   pixels do not contribute; they are not replaced with black pixels. Near a
   boundary there are fewer valid pairs, which can make estimates less stable.
5. **Calculate four features per direction.** Normalize each directional
   histogram separately, compute the four properties above, then average each
   property equally across directions with valid pairs. This is averaging
   features, not pooling histograms before calculating them. It reduces
   orientation dependence but does not guarantee rotation invariance.
6. **Combine scales.** Each pixel receives eight values: four half-window
   features followed by four full-window features. Features are calculated at
   every pixel without resizing or subsampling the feature maps. Pixels without
   any valid direction at either scale receive zero features and label zero.
   They are excluded from clustering. An entirely unsupported region produces
   an explanatory message rather than a fabricated split.
7. **Group similar feature vectors.** Standardize each feature over supported
   pixels using its mean and standard deviation. Run fixed-seed, multiple-start
   k-means++/Lloyd clustering on an evenly distributed sample of at most 20,000
   feature vectors, then assign every supported pixel to a center. The default
   requests up to three groups. Uniform data may yield fewer; the algorithm does
   not automatically infer the correct number of tissue types. Group IDs are
   ordered by cluster-center half-window contrast, not clinical meaning.
8. **Select and save.** Click a colored group to keep the connected patch under the cursor. Disable
   **Keep clicked connected patch only** to keep all disconnected matches too. This is always a subset of the original MedSAM mask.
   Save retains the parent mask, all GLCM feature maps and groups, and the selected
   refined mask. Original segmentation proposals are not overwritten.

The implementation streams gray-level pair histograms through summed-area tables.
This gives exact rectangular window counts without allocating a separate full
8 by 8 matrix at every pixel. Computation runs on the CPU; the MedSAM model
can still use CUDA. Larger image regions and more gray levels cost more time.

## Saved data and compatibility

New JSON texture records have `method: local_glcm_v2` for pure GLCM, or `local_hybrid_v1` for Combined and record gray levels,
quantization rule, offsets, symmetry, actual half/full windows, directional
aggregation, feature names, requested/actual groups, and supported pixel counts.
The NPZ arrays remain:

- `texture_N_features`: unstandardized features with shape `(height, width, channels)`: 12 channels for Combined, four for Brightness, eight for GLCM.
- `texture_N_labels`: integer labels `1..K`; zero outside the parent mask or
  where pair support is absent.
- `texture_N_selected_mask`: boolean refined mask, empty until a group is clicked.

`N` is the zero-based parent region index. Feature names give the exact channel
order. Existing annotations with four mean/standard-deviation features remain
readable and are not silently relabeled as GLCM. Clicking **Split latest region**
explicitly replaces the latest region's stored refinement with the chosen method.

## What this does and does not establish

Large windows average over more speckle and expose broader texture differences,
but mix textures across internal boundaries and can obscure small structures.
The automatic window rule is a starting heuristic, not a validated optimal scale.
GLCM also depends on quantization, offsets, gain, dynamic range, and image
processing. Similar patterns can occur in different tissues or artifacts.

Clarius RF-derived exports use a sample/line grid without scan conversion. Equal
pixel offsets in depth and laterally do not necessarily represent equal physical
distances. These settings make no millimeter-scale claims. Per-frame display
normalization also limits comparisons across frames or acquisitions.

The resulting groups describe texture, not lesions or diagnoses. Review them
against the original image. Refinement cannot add tissue outside the MedSAM
mask, and a single group need not match a complete anatomical structure.

## Validation

```powershell
python -m unittest discover -s tests -p "test_clarius_*.py" -v
```

The 24 tests cover reference GLCMs, known constant/checkerboard patterns,
directions and offsets, mask holes, edges, invalid input, unsupported pixels,
automatic window sizing, deterministic grouping of spatially different patterns,
and headless viewer selection, save/reload, undo, and legacy annotations.

A separate local smoke test ran the actual MedSAM ViT-B checkpoint on CUDA and
refined a 49,097-pixel proposal in a 1520 by 192 Clarius frame. GLCM refinement
took approximately 2.7 seconds on that machine. This checks end-to-end execution,
not segmentation accuracy; no annotated ground truth was used. The test artifacts
remain in the ignored local annotations directory and are not part of Git.

## Hybrid weighting and appearance descriptions

Combined appends eight GLCM channels to four intensity channels (half-window
mean/std, full-window mean/std). After per-channel z-score normalization, mean
channels receive weight `1/sqrt(2)`, standard-deviation channels `0.25`, and
GLCM channels `0.35/sqrt(8)`. This prevents the eight GLCM features from dominating
by their number and restores sensitivity to brightness differences. Saved feature
arrays are unweighted; JSON records the weights used for clustering. Brightness
and pure GLCM modes use unit feature weights.

Group descriptions compare the group's original display-intensity mean and
standard deviation with the parent region. A mean difference exceeding 0.03 on
the normalized 0..1 scale is called darker or brighter. Variation is called higher
above `1.15 * parent_std + 0.01`, or lower below `0.85 * parent_std - 0.01`. These
thresholds provide reproducible descriptions, not clinical significance. Groups
remain a coarse partition; small or rare features can still be merged.

## Automatic spatial refinement

This stage uses a compact source population as an automatic seed rather than
merging whole groups by one global standard deviation. Candidate shapes are
opened with radius 9 on a normalized 256 by 256 region grid. The seed score is
`compact_area / (1 + (MAD_of_local_mean / 0.05)^2)`, with MAD scaled by 1.4826.
This discourages a large but mixed-intensity group from providing an overly broad
appearance model. A largest-component fallback is recorded when opening removes
all candidates.

From the seed, brightness limits are `median - max(2.5*MAD, 0.05)` and
`median + max(3*MAD, 0.08)`. GLCM/intensity feature deviations are scaled by
their seed 85th-percentile absolute deviations, bounded per channel, and compared
with a seed-derived distance threshold. Local speckle standard deviation provides
an independent compatibility check so one tissue population can cross GLCM color
boundaries. These are heuristic thresholds in normalized display units.

Compatible evidence forms a spatial scaffold. Dark cavities may be temporarily
closed only for shape estimation, preventing erosion of a thin tissue rim above
a hole. Bright interfaces remain barriers. Spatial opening selects the component
overlapping the automatic seed; a radius-5 envelope bounds recovery, and a
radius-3 closing repairs small classification gaps. The dark exclusion is then
reapplied on the native grid, followed by largest-component selection and parent
mask clipping. Original source labels and learned diagnostics are saved.

The algorithm has no fixed top/bottom cutoff. It still assumes the intended tissue
is represented by a compact, coherent population in the MedSAM proposal. This
assumption can fail; it is not an anatomical classifier or clinical validation.
