# Evaluation with supplied verified contours

On 17 November 2026, we evaluated 98 images from 33 recorded examinations
against the contours in `final_ground_truths/`. The user identified these as
verified by their PhD student. Files 006 and 045 have no contour and were
excluded at the user's request. The PI's continuing segmentation revisions are
a separate source of future references.

## Protocol

Every method received the matching clean MATLAB `env_gamma08` image at
1418 by 853 pixels. All MedSAM prompts were the fixed full-image box
`[0, 0, 1418, 853]`. The verified contour supplied neither a box nor a seed.
This tests the requested automatic workflow, although MedSAM technically
still requires a spatial prompt. Refinements shared the same saved parent.

We retained the existing algorithms and defaults: 8 gray levels, up to three
source groups, automatic selection at q0.985, and GLSZM sampling step (8,2).
The brightness baseline uses brightness proposals followed by the same
automatic selector. GLCM, Combined GLCM, GLSZM and Combined GLSZM also use
that selector. No parameter selection used these reference scores.

Red contour conversion retains the largest enclosed compartment and separate
internal exclusions. Short endpoint seams up to 10 pixels are closed and
logged. Image 021 has the explicitly authorized straight connector between
its two outer endpoints. The conversion assigns stroke pixels to the nearest
compartment. All 100 original files remain unchanged. Reference/prediction
arrays retain their native grid, with no resizing for scoring.

## Results

These are per-image macro means. Zero overlap means Dice equals zero.
All 588 image/method pairs were scored. There were no algorithm exceptions.

| Method | SSIM | ROI SSIM | Dice | IoU | Zero overlap / 98 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MedSAM | 0.3475 | 0.3056 | 0.4771 | 0.3179 | 0 |
| Brightness + automatic selection | 0.6404 | 0.6087 | 0.4765 | 0.3813 | 27 |
| GLCM + automatic selection | 0.6768 | 0.6463 | 0.3555 | 0.2808 | 43 |
| Brightness + GLCM | 0.7063 | 0.6804 | 0.4565 | 0.3840 | 38 |
| GLSZM + automatic selection | 0.6895 | 0.6622 | 0.5391 | 0.4082 | 12 |
| Brightness + GLSZM | 0.6987 | 0.6714 | 0.4334 | 0.3511 | 36 |

Combined GLCM has slightly higher SSIM, Dice and IoU than brightness + GLSZM
in this comparison. Pure GLSZM has the highest mean Dice and IoU among the
six methods. These rankings do not establish reliable automatic liver
segmentation. The parent often covers almost the entire field, and texture
selection can retain a deeper anatomical region with zero liver overlap.

Full-frame SSIM includes shared background. ROI SSIM averages the same local
SSIM map over a common union of the parent, reference and all predictions,
excluding invalid border centers. Dice and IoU measure foreground overlap
on complete arrays. See [metric definitions](EVALUATION.md).

Repeated captures within an examination are correlated. Thirty-three exam
IDs do not establish 33 independent patients. Earlier seven-frame results used
provisional references, different images and targeted prompts, so they cannot
be directly compared with this test. Further tuning should use separate
development groups and a fresh evaluation set.

## Reproduce locally

Keep the supplied reference folder and matching clean MATLAB exports local.
The source lookup defaults to
`clarius_annotations/liver_env_gamma08_v2/source_manifest.json` and uses its
image paths only, not its previous annotations.

```powershell
python -B tools/evaluate_verified_outlines.py --checkpoint .\medsam_vit_b.pth --device cuda:0 --workers 8
python -B tools/report_verified_evaluation.py .\clarius_annotations\verified_evaluation_20261117
```

The first command prepares references, runs MedSAM on CUDA, computes CPU
texture features and scores the masks. Eight workers reduce batch wall time
but need sufficient RAM. Per-case times include worker contention and cannot
be compared directly with earlier serial benchmarks. For an interrupted run,
reuse audited references with `--reuse-prepared`; `--refine-only` additionally
requires all parents to exist and validates their configuration.

Outputs under `clarius_annotations/verified_evaluation_20261117/` include:

- `REPORT.md` and `scores/summary.csv`: aggregate results.
- `review.html`: native-proportion prediction/reference comparisons.
- `scores/per_image.csv` and `scores/per_exam.csv`: detailed paired scores.
- `reference_manifest.json`: provenance, hashes and contour seam records.
- `parents/`, `predictions/` and `details/`: masks, diagnostics and runtimes.
- `validation.json`: paired-row, dimension and unchanged-input checks.

Raw images, masks and temporary presentation assets remain ignored. The
repository stores evaluation code, aggregate results and the presentation
content/build source.
