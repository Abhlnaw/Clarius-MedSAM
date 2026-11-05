# SSIM evaluation against reference segmentations

**Revision 2 supersedes the rejected first references and their scores.** Use
`clarius_annotations/ssim_reference_evaluation_v2/`. The v1 folder is retained
only as an audit record.

Every method is scored against the same reference mask for each image. This
answers agreement with that reference, not similarity between the two methods
and not similarity between an ultrasound image and a binary mask.

## Current exploratory references

The revised exam05 frame 001 anchor approximately traces the user's red outline
from the conversation image. Its inner vessel contour is separately AI-authored.
Consecutive original frames are registered with smoothed-image TV-L1 optical
flow, and the anchor vertices are advected to frames 002, 003, 005, and 010.
Exam04 uses a separately revised original-image anchor and propagation from
frame 001 to 002; anatomy is not copied across examinations. Image-alignment
residuals and flow parameters are saved in `registration_audit.json`.

These are provisional annotation aids requiring visual review. Uncertain deep
interfaces, especially in exam04, are documented in the recipe. They are not
verified anatomical ground truth. The annotator had previously seen related
model outputs, so this is not a blinded or independent validation study.

The contours were frozen before running this evaluation. Reference construction
uses the user-guided/source-image anchors and original-image registration.
It does not use either predicted mask, segmentation brightness thresholds, or
MedSAM clipping. Reference foreground beyond the parent mask remains in
the evaluation so an imperfect MedSAM parent cannot silently erase errors.
Hashes identify the exact reference PNGs used.

Local review and audit files are under
`clarius_annotations/ssim_reference_evaluation_v2/`:

- `START_HERE.md`: links to the ultrasound review images.
- `anchor_recipe.json`: authored anchors, source paths, protocol, and uncertainty notes.
- `contours.json`: dense anchor/propagated vertices for all seven images.
- `overlays/*.reference.png`: **2048x1232 RGB ultrasound images with red outlines**.
- `labels_native/*.labels.png`: internal binary labels for scoring only, at the
  original RF grid (1536x192 for exam05; 1520x192 for exam04).
- `reference_review.png`: contact sheet of the seven ultrasound overlays.
- `reference_audit.json`: dimensions, areas, hashes, and prior-output-exposure note.
- `evaluation_manifest.json`: references and the two predicted arrays per frame.
- `scores/per_image.csv`, `scores/metrics.json`, `scores/REPORT.md`: results.

These images, masks, and local data paths stay in the ignored annotation folder.
The individual review images match the 1418:853 presentation proportions of the
user's red example. This is a display export, not physical scan conversion or
newly acquired image detail. Never resize the internal labels to match the
presentation PNG before scoring. The old binary tall-strip files were not
suitable as human review images.

## Metric definition

Standard SSIM uses native binary 0/1 arrays, `data_range=1`, 11x11 Gaussian
windows, sigma=1.5, K1=0.01, K2=0.03, and population covariance. These match the
[scikit-image settings for the Wang SSIM formulation](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity).
No colored overlays, display stretching, scan conversion, or resizing are used.
An identical prediction/reference gives 1. SSIM is not a probability of a
correct segmentation.

Large shared background can inflate full-frame SSIM. `roi_mean_ssim` therefore
also reports a separately named, custom mean of the local SSIM map over a common
evaluation domain: the saved parent mask plus the reference and **both**
predictions. Standard image-border window centers are excluded. This ROI score
is not substituted for standard full-frame SSIM. Both methods use exactly the
same evaluation domain for a frame.

Dice, IoU, precision, and recall use full native arrays. These foreground metrics
help interpret SSIM. Both-empty masks use Dice=IoU=1; undefined precision or
recall is recorded as null. Scores are not pooled across differently sized
images: the summary is a per-image macro mean and sample SD. Frames from the
same examination are correlated, so SD is descriptive, not a patient-level
confidence interval. Unequal RF pixel spacing also limits physical interpretation
of the native 11x11 window.

## Preliminary results

Both pipelines retain the previously evaluated q0.985 settings, eight gray
levels, and the same saved MedSAM parents. Brightness-plus-GLSZM uses step (8,2).
No algorithms or thresholds were retuned to these new reference masks.

| Method | SSIM mean +/- frame SD | ROI mean SSIM | Mean Dice | Mean IoU |
| --- | ---: | ---: | ---: | ---: |
| Combined / GLCM | 0.8499 +/- 0.0350 | 0.7719 | 0.8643 | 0.7636 |
| Brightness + GLSZM | 0.8365 +/- 0.0196 | 0.7567 | 0.8501 | 0.7403 |

Combined/GLCM had higher SSIM on four frames; brightness-plus-GLSZM had higher
SSIM on three. The macro mean difference is 0.01335. This is agreement with
unreviewed AI outlines, not proof of anatomical superiority. It also reuses
images involved in prior development and is not a held-out test.

The old v1 boundary-sensitivity results do not apply to these revised references.

The next step for an anatomical evaluation is PI/expert review or replacement
of these references. A corrected reference can change the ranking. The local
MATLAB mask found for exam03 was not used because its target class has not been
confirmed to match this task.

## Reproduction and reviewed references

```powershell
python -B tools/propagate_reference_contours.py --recipe .\clarius_annotations\ssim_reference_evaluation_v2\anchor_recipe.json --output .\clarius_annotations\ssim_reference_evaluation_v2\contours.json
python -B tools/create_reference_masks.py --contours .\clarius_annotations\ssim_reference_evaluation_v2\contours.json --output .\clarius_annotations\ssim_reference_evaluation_v2 --predictions .\clarius_annotations\glszm_comparison
python -B tools/evaluate_segmentations.py --manifest .\clarius_annotations\ssim_reference_evaluation_v2\evaluation_manifest.json --output .\clarius_annotations\ssim_reference_evaluation_v2\scores
python -B -m unittest discover -s tests -p "test_clarius_*.py"
```

For expert-reviewed masks, create a new manifest pointing to the corrected
native-grid masks, record their actual provenance/review status, and specify the
target class ID. Relative paths resolve against the manifest directory. Example:

```json
{
  "reference_provenance": "Describe the actual annotator and target definition",
  "reference_review_status": "Record the actual review status",
  "cases": [{
    "id": "exam05_frame_001",
    "exam": "exam05",
    "reference": {"path": "reviewed/frame_001.png", "label": 1},
    "predictions": {
      "combined_glcm": {"path": "predictions/frame_001.npz", "key": "combined"},
      "brightness_glszm": {"path": "predictions/frame_001.npz", "key": "combined_glszm"}
    },
    "evaluation_mask": {"path": "predictions/frame_001.npz", "key": "parent"}
  }]
}
```

The evaluator preserves indexed PNG label IDs, requires an explicit NPZ array
key, checks binary arrays and native shapes, and requires paired methods across
all images. It rejects changed references when their recorded SHA256 is present.
When intentionally correcting a reference, create/update its manifest and hash
and use a new output folder. Do not call an AI mask expert-reviewed simply by
changing its metadata.
