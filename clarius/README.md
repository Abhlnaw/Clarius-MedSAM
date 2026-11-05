# Clarius ultrasound workflow

## Run

Set the exam directory in `clarius/runme2_export.m`, then export the underlying B-mode pixels (not plot screenshots):

```powershell
matlab -batch "run('clarius/runme2_export.m')"
```

Start the interactive viewer using your inference Python environment:

```powershell
python MedSAM_Clarius.py --checkpoint .\medsam_vit_b.pth --device cuda:0 -i "C:\Users\abhit\Documents\Hoyt Research\data\exam05\exported_images" --output ".\clarius_annotations\exam05_glcm_test" --glcm-levels 8
```

The input can also be a single image. Use a separate output folder per exam. Use `--device cpu` without CUDA. The original external `runme2.m` is unchanged; its copy in this repository exports the RF-derived B-mode pixels and JSON provenance.

## Automatic region selection

1. Draw the MedSAM bounding box around the target.
2. Leave **Low / High variation** selected and click **Split latest region**.
3. The region is found and selected automatically. Review the overlay and Save.

No tissue click, painting, or secondary crop is required. The manual Add/Exclude/Keep-selected workflow was removed.

The automatic mode uses **Combined** as its default source proposal. To use pure GLCM, select **GLCM**, then return to **Low / High variation**, or start with `--auto-source glcm`. Combined, Brightness, and GLCM remain available as inspection views; their original click-to-select behavior is unchanged.

### How automatic refinement differs from the earlier variation merge

The earlier implementation merged whole colors by their overall standard deviation and chose the largest region. That could cut tissue into color bands, retain unrelated lower areas, or choose a mixed population as its intensity model.

The new algorithm in `clarius_auto.py` instead:

- Finds compact candidate bodies within the source proposal, suppressing narrow bridges and tails before comparing them.
- Scores candidate area with a penalty for dispersion of local mean intensity, so a large mixed population does not automatically win.
- Learns robust brightness and texture ranges from that automatic seed.
- Recovers tissue across source colors using the seed's two-scale GLCM correlation-distance range (q0.985 by default).
- Uses spatial opening and a bounded boundary envelope to limit lower ribbons and distant spillover.
- Keeps the component attached to the automatic seed, then the largest final component.
- Reapplies a dark-cavity exclusion after cleanup, rather than indiscriminately filling holes. The final mask always remains inside the parent MedSAM mask.

It has no hardcoded depth cutoff and does not simply choose the top color. Shape processing occurs on a 256 by 256 normalized region grid to account for the unequal RF row/column sampling. This is a geometric heuristic, not physical scan conversion.

Source groups still describe appearance, not anatomy. The method is an automatic tissue-region heuristic, not a trained liver or portal-vein detector. Dark shadows can resemble cavities, and very different acquisitions can require further validation. Review the resulting mask.

## Settings and performance

Experimental **GLSZM** and **Combined GLSZM** views and automatic source options
are also available. They count connected equal-gray-level zones in local windows;
the current Combined/GLCM q0.985 default is unchanged. See the
[GLSZM implementation and comparison](GLSZM.md) for controls, exact versus
sampled maps, and the reproducible exam04/exam05 comparison.

`--glcm-levels` accepts 8 (default), 16, 32, or 64. Increasing levels substantially increases CPU GLCM cost. MedSAM encoding runs on the requested device; GLCM and spatial refinement run on CPU.

`--auto-correlation-quantile` defaults to **0.985** in automatic Low / High
variation mode, for either Combined or GLCM as the source. It sets a percentile
of the seed's normalized correlation-feature distance, not a raw correlation
coefficient of 0.985. Higher quantiles are generally more permissive. Use
`--auto-correlation-quantile 0.980` to compare a tighter setting, or
`--auto-correlation-quantile 0` for the earlier feature/source/speckle rule.
Inspection modes retain their usual clustering behavior.

The status and saved `automatic_refinement` metadata identify the setting used.
Existing annotations are loaded as saved; click **Split latest region** again
to apply the new default, then Save, or use a fresh output folder such as
`--output .\clarius_annotations\exam05_q0985` to compare results.

On the investigated exam05 frame, q0.985 removed about 42% of the current
spillover relative to the approximate red outline while retaining about 97% of
the current selection inside it. The lower-left extension remains partly
present, and some holes enlarge on other frames. This setting is an appearance
heuristic and still needs review on different acquisitions.

The viewer caches up to three image embeddings, shares GLCM maps between Combined and GLCM, and caches automatic results. Revisiting a proposal avoids recalculating its features. Creating a new parent region invalidates texture caches.

On the local RTX 5070 laptop setup, warm GPU image encoding measured about 0.2 seconds. First exam05 Combined/automatic calculations took roughly 4?7 seconds at eight levels; the previous 32-level texture pass took about 27 seconds. Console timings report each operation. First calculations are synchronous and may briefly pause the window.

## Saving

Save writes all original regions on the current frame and each region's latest refinement. Navigation and closing save when edits are pending. Outputs are updated in place:

- **`.regions.npz`:** original `masks`, plus `texture_N_features`, `texture_N_labels`, `texture_N_selected_mask`, and (for automatic results) `texture_N_source_labels`. `N` is the zero-based parent region index. Labels are 1 for the automatically retained region and 0 outside; the original source partition is stored separately.
- **`.regions.json`:** boxes, pixel measurements, GLCM settings, `method: automatic_tissue_v1`, source method, and `automatic_refinement` diagnostics: seed scores, scale/radii, learned thresholds, and retained area.
- **`.overlay.png`:** the full-frame overlay at viewer proportions, at twice the canvas resolution with a 2048-pixel longest-edge cap. Zoom does not crop the saved image.

Mask and feature arrays retain their native sample grid. Resizing the PNG is for presentation, not physical calibration. `shape_hw` and `overlay_shape_hw` record the separate dimensions. Original model masks and source partitions are preserved. Old annotation records remain readable.

## Validation

For SSIM, Dice, and IoU against shared per-image references, see
[reference-mask evaluation](EVALUATION.md). The current seven-image SSIM results
use explicitly provisional AI-drawn references; they require PI/expert review
before being treated as anatomical ground truth.

```powershell
python -m unittest discover -s tests -p "test_clarius_*.py" -v
```

Tests compare GLCM features with scikit-image and exercise automatic recovery across source labels, dark-cavity preservation, rejection of narrow lower lobes, seed scoring, reversed target position, mask containment, source-view switching, caching, save/reload, and overlay proportions.

Local evaluation includes saved exam04/exam05 frames with both source methods and additional exam05 frames processed using the existing box prompt and the actual CUDA checkpoint. No tissue selections or correction inputs are supplied. Comparisons are qualitative; there are no expert ground-truth masks, so these checks do not establish anatomical accuracy.

Per-frame normalization also means brightness statistics are not calibrated across acquisitions. Measurements are in pixels until acquisition geometry and scan conversion are verified.
