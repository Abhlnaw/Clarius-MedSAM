# Texture threshold experiments

These scripts reproduce the exam05 lower-left boundary investigation without
changing application defaults or existing annotations. They use the project's
normal Python dependencies; no CUDA inference is needed when parent masks and
source features have already been saved.

Run from the repository root, in order:

```powershell
python -B tools/texture_thresholds/threshold_experiment.py
python -B tools/texture_thresholds/threshold_features_experiment.py
python -B tools/texture_thresholds/threshold_crosscheck.py
```

Use the Python interpreter with NumPy, SciPy, scikit-image, and Matplotlib
installed. The scripts expect your local saved annotations:

- `clarius_annotations/exam05_glcm_test/frame_001.png.regions.json` and `.npz`.
- The same pair for `frame_002.png`.
- Frames 003, 005, and 010 under
  `clarius_annotations/glcm_validation/automatic_evaluation/holdout/`.
- Original images at the `source` paths in those JSON files.

The first two scripts use frame 001; the final script also needs the other four
frames. Those images and saved annotations are local inputs and are excluded
from Git. Generated plots, metrics, and masks go into the ignored directory
`clarius_annotations/glcm_validation/threshold_experiment/`.

## What is tested

`threshold_experiment.py` captures intermediate arrays from the existing
automatic selector, then reconstructs its final stages. It checks exact baseline
agreement before changing acceptance rules, brightness bounds, feature distance,
or speckle limits. Saved source labels/features are reused; older annotations
without source labels recompute the default eight-level Combined proposal.

`threshold_features_experiment.py` compares thresholds on individual channels
and feature families. `threshold_crosscheck.py` refines the correlation-distance
quantile and compares fixed settings across five frames. It checks parent-mask
containment and preservation of the original dark-pixel exclusion.

These are research experiments tied to the current selector's internal array
names and the 12-channel Combined feature layout. They deliberately do not add
threshold controls or change the viewer's behavior. Baseline equality checks
should flag incompatible future selector changes.

## Reference and findings

The red boundary was approximately traced from the user's screenshot and mapped
to native image coordinates. These coordinates are used for evaluation only;
they never enter the segmentation algorithm. Existing parent-mask clipping and
dark-pixel exclusion are applied to this rough reference. It is not an expert
annotation, and the resulting numbers are not clinical accuracy measurements.

The search evaluated 213 general/brightness variants, 114 feature-specific
variants, and 50 nearby correlation quantiles on frame 001:

| Setting | Current wanted pixels retained | Current spill removed |
| --- | ---: | ---: |
| General feature threshold reduced to 35%, existing OR acceptance | 100% | 0% |
| General feature threshold mandatory, original threshold | 93.8% | 26.2% |
| Correlation-distance threshold at seed quantile 98% | 96.4% | 44.5% |
| Correlation-distance threshold at seed quantile 98.5% | 97.0% | 42.0% |

"Wanted retained" measures the current mask's pixels inside the rough reference
that remain selected. "Spill removed" measures the current mask's pixels outside
that reference that are removed. These percentages do not measure recovery of
all desired tissue.

The 98.5% correlation setting increased rough-reference Dice from 0.91566 to
0.92365; its frame-specific normalized distance threshold was 3.01921. This
removes part of the extension, but leaves a visible remainder. On frames
002-010 it retains 87.6%-93.6% of each full original mask. Without reference
outlines for those frames, that shrinkage cannot all be counted as improvement;
some visible holes enlarge. Application defaults remain unchanged.

Binary search does not optimize boundary agreement here: overlap rises and
falls as a threshold changes, while connectivity operations can change the
selected component. The scripts use a broad sweep followed by a finer local
sweep. A robust default needs reference masks on several acquisitions.

The packaged scripts reproduced these measurements, and all 37 existing
Clarius regression tests passed before publication.

## Generated outputs

- `comparison.png`: general threshold comparisons.
- `feature_comparison.png`: feature-specific comparisons.
- `cross_frame_comparison.png`: current and experimental masks on five frames.
- `threshold_tradeoff.png`: overlap, retention, and spill tradeoff.
- `metrics.json`, `feature_metrics.json`, `crosscheck.json`: full measurements.
- `candidates.npz`, `feature_candidates.npz`: selected experimental masks.
