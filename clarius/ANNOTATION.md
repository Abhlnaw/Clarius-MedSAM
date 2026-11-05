# Manual annotation workflow

The current annotation set uses **1418 × 853** full-frame ultrasound images
for both manual masks and model evaluation. The wide appearance comes from
bilinear resizing of the entire RF-derived frame; it does not crop the image
or establish calibrated physical spacing.

## Export first frames

Run from the repository root with MATLAB, its Signal Processing and Image
Processing functions, Python, and a local `lzop` executable installed:

```powershell
python tools/export_annotation_frames.py --data "C:\path\to\data" --output ".\clarius_annotations\manual_segmentation" --lzop "C:\path\to\lzop.exe" --matlab "C:\Program Files\MATLAB\R2026b\bin\matlab.exe"
```

The exporter skips top-level folders matching `exam` followed by digits,
reads each RF recording in the other folders' `.tar` archives, and prepares
only its first frame. It never extracts archive member paths onto disk.
MATLAB applies the same Hilbert-envelope/log scaling and per-frame intensity
normalization as `runme2_export.m`, then resizes the full frame to 1418 × 853.
PNG sidecars and the manifest retain source identifiers, timestamps and native
dimensions. Original archives remain unchanged.

## Review the current 157-image set

These helpers are specific to the current dataset and its annotation recipes:

```powershell
python tools/review_drawn_outlines.py .\clarius_annotations\manual_segmentation
python tools/draft_annotation_masks.py .\clarius_annotations\manual_segmentation
python tools/package_annotation_masks.py .\clarius_annotations\manual_segmentation
```

The first command reads the first 25 red drawings and uses the explicit
interior seeds in `review_drawn_outlines.py`. It detects red strokes, bridges
eligible endpoint gaps of at most 20 pixels, preserves enclosed holes, and
records every bridge. This is conversion assistance, not anatomical approval.

Draft generation requires local `draft_work/groups.json`, `anchors.json` and,
optionally, `visual_corrections.json`. Anchors use the 444 × 267 review grid;
the tool renders on the full image grid and transfers anchors only within
their supplied recording groups. These recipes and the underlying image data
remain local under the ignored annotation folder. They must be authored and
reviewed for another dataset; these helpers do not automatically identify
anatomy in arbitrary images. Run helper generation in a fresh working folder
if existing conversion/draft outputs have been manually edited.

The packaged folders are:

| Folder | Purpose |
| --- | --- |
| `model_inputs/` | Clean ultrasound reconstructed in `clean_source_export/images/`; use these for the model. |
| `masks/` | Binary candidates, foreground 255/background 0. |
| `outlines/` | Red mask boundaries on clean ultrasound for review. |
| `human_drawings/` | Unmodified copies of the user's original 25 drawings. |
| `review/` | Contact sheets, findings and conversion/draft audit records. |

The packager checks dimensions, binary values, foreground connectivity and
original drawing hashes. It refuses to overwrite a differing existing target,
protecting later manual edits. `mask_manifest.json` records each mask's source,
hash and pending independent review status. Keep an approved-mask copy
separate from generation outputs, and record reviewer approval explicitly.

Red drawings modify input pixels. Do **not** test the model on marked-up
images; use `model_inputs/` with matching clean image filenames instead.
Have the reviewer inspect and correct full-resolution outer boundaries and
internal exclusions without viewing model predictions. Preserve the complete
1418 × 853 canvas and filenames when editing masks. Drafts and conversions
are provisional until independently reviewed; no final SSIM/Dice/IoU accuracy
claim follows from their technical validation alone. See
[evaluation](EVALUATION.md) for reference-based metric computation.
