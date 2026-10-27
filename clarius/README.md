# Clarius ultrasound additions: workflow guide

This guide covers the MATLAB export, interactive annotation, and texture refinement added to the MedSAM codebase. See [the project README](../README.md) for a summary of the changes.

1. In MATLAB, run `clarius/runme2_export.m` from this repository. It is a copy
   of your `runme2.m`; set its exam directory near the top as needed. The original
   MATLAB script is unchanged. This copy exports clean grayscale RF-derived
   B-mode pixels to the exam's `exported_images` directory and JSON provenance
   beside each PNG. Rerun exports before using old screenshot-based frames.
2. From the MedSAM directory, in the Python environment you use for inference:

   ```powershell
   python MedSAM_Clarius.py --checkpoint .\medsam_vit_b.pth --device cuda:0 -i "C:\Users\abhit\Documents\Hoyt Research\data\exam04\exported_images"
   ```

   For a single image, pass its filename instead of a directory. For CPU use
   `--device cpu`. This viewer uses the existing Matplotlib dependency, not PyQt5.
3. Drag with the left mouse button around one target. Release to generate its
   mask. Repeat for additional regions; Undo removes the last region. Use the
   Matplotlib toolbar to zoom, then turn off zoom/pan before drawing a box.
4. Save writes to `clarius_annotations` (override with `--output`). Previous/Next
   and window close also save. Existing annotations reload on return to a frame.
   Use a separate output directory for each exam to avoid filename collisions.

Each image produces an NPZ containing `masks` with shape `(regions, height, width)`,
a JSON file with boxes, region IDs, pixel areas and display brightness statistics,
and a PNG overlay. Separate masks preserve overlapping selections. Region IDs are
generic annotation labels, not anatomical or diagnostic predictions.

These exports are unscanconverted sample/line images. The viewer stretches them
to a convenient display shape; saved masks always retain the original pixel grid.
Physical measurements require verified acquisition geometry and scan conversion.
The hardcoded depth/pitch values in the MATLAB viewer are not used for measurements.
Per-frame display normalization means brightness statistics are not calibrated
or directly comparable across frames. Standard deviation is intensity variation,
not an abnormality score. The optional texture step combines local intensity statistics for appearance grouping.

MedSAM provides box-prompted proposals, not guaranteed ultrasound boundaries or
lesion detection. Inspect every result; redraw a tighter box or undo an incorrect
proposal. This version does not include manual boundary painting or tracking
between frames. Encoding and inference run synchronously and may briefly pause
the window, especially on CPU.

## Refine an oversized mask using texture

After drawing a box, click **Split latest by GLCM**. The latest MedSAM region
is partitioned into up to three appearance groups using local GLCM contrast,
homogeneity, energy, and correlation at half and full window sizes. Pair counts
exclude pixels outside that region or the local window. The original MedSAM mask
is preserved. See [how GLCM works](GLCM.md) for the complete algorithm and formulas.

Neighborhood sizes are chosen automatically from the selected mask: the larger
window is approximately one quarter of the occupied region's height and width,
and the smaller window is half that size, with a minimum of three pixels per
direction. No slider adjustments or physical measurements are needed. This is a
relative-scale heuristic; it does not estimate acquisition geometry or guarantee
the best scale for every tissue pattern. Larger regions receive larger filters.

Click a colored group to select it as the refined mask. All pixels of that group
are selected, including disconnected patches. Other groups become faint. Click
**Show MedSAM masks** to return to box drawing. Texture analysis applies to the
latest region; Undo that region if you need to work on an earlier one.

Save adds `texture_N_labels`, `texture_N_features`, and `texture_N_selected_mask`
arrays to the existing NPZ, where N is the zero-based parent region index.
Labels are 1..K where pair support exists; zero means outside the parent or
unsupported at either scale. Feature arrays now have eight GLCM channels. JSON
records method `local_glcm_v1`, quantization, offsets, window sizes, feature names,
support counts, and the selected group (null until clicked). Older four-feature
annotations remain readable until explicitly recomputed. The overlay PNG saves
the current display. These arrays reload with annotations; Split recomputes them
using the automatic scales. Group IDs are per-region appearance clusters, not
consistent tissue classes across frames. This can narrow an oversized proposal,
but will not recover tissue excluded by MedSAM or guarantee an anatomical boundary.

Algorithm checks (in your inference Python environment):

```powershell
python -m unittest discover -s tests -p "test_clarius_*.py" -v
```
