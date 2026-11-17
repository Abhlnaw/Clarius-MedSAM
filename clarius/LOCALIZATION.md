# Learned liver-box localization

The optional localizer predicts a liver bounding box from a clean exported
ultrasound image, then passes that box to the existing frozen MedSAM model.
It replaces manual box drawing on new frames when explicitly enabled.

## Experiment

The experiment uses the 98 supplied PhD-student-verified references from
`verified_evaluation_20261117`. A bounding rectangle of each training mask
provides its box target; internal lumen exclusions do not change the box.
The inputs are the matching clean MATLAB env_gamma08 images, never the
images containing red annotations. MATLAB image formation remains unchanged.

Five outer folds keep all captures with the same examination ID together.
An inner validation group selects training duration using box IoU. Each
scored CNN box comes from a model that saw neither that examination's
training labels nor validation labels. All 98 image hashes are unique.
Examination IDs do not establish independent patients, and this dataset was
previously inspected during method development. These are exploratory
cross-validation results, rather than an external clinical validation.

Controls include:

- The previous full-image prompt and saved parent/refinement masks.
- The mean box of the same training cases, with no image-specific prediction.
- The reference-derived box, explicitly labeled an assisted oracle comparison.

Predicted, mean and oracle boxes all receive a fixed margin of 0.03 image
width/height per edge, clipped to the image. No held-out scores select that
margin. The existing MedSAM weights, input preparation and threshold remain
fixed. Combined GLCM refinement uses the existing eight levels, q0.985 and
automatic windows, without new threshold tuning.

## Measured results

Per-image means across all 98 held-out predictions:

| Prompt and method | SSIM | Dice | IoU |
| --- | ---: | ---: | ---: |
| Full-image MedSAM | 0.3475 | 0.4771 | 0.3179 |
| Training-mean box + MedSAM | 0.8409 | 0.7917 | 0.6693 |
| CNN box + MedSAM | 0.8513 | 0.8121 | 0.6933 |
| Reference box + MedSAM, assisted | 0.8746 | 0.8510 | 0.7438 |
| Full-image + Combined GLCM | 0.7063 | 0.4565 | 0.3840 |
| Training-mean box + Combined GLCM | 0.8347 | 0.6960 | 0.5602 |
| CNN box + Combined GLCM | 0.8404 | 0.7107 | 0.5810 |

Raw predicted-box IoU is 0.8137, versus 0.7456 for the training-mean box.
Padded CNN prompts cover 98.73% of reference foreground pixels on average,
versus 96.70% for mean prompts. Padding intentionally trades tighter box IoU
for greater coverage.

The CNN improves localization beyond the mean-box positional prior, although
much of the improvement over full-image prompting also occurs with the mean
box. Combined GLCM zero-overlap cases fall from 38 to 1 with CNN prompts
(2 with mean prompts). However, the unchanged texture refinement lowers Dice
relative to localized MedSAM alone. Localized MedSAM is therefore the stronger
baseline in this experiment; automatic texture refinement remains optional.

## Small CNN

`clarius_localizer.py` defines a 423,604-parameter network trained from scratch.
It takes a 128 by 224 grayscale image. Four convolution blocks have 16, 32,
64 and 96 channels, followed by fixed average pooling and a 128-unit head.
Group normalization supports small batches. Sigmoid coordinates parameterize
an ordered box within the image. Initialization starts at the training mean
box; no pretrained weights or external training data are downloaded.

Settings: AdamW learning rate 0.001, weight decay 0.001, batch 16, dropout 0.2,
GIoU plus SmoothL1 loss (beta 0.05). The limit is 160 epochs, with a minimum
40 epochs and stopping after 30 epochs without validation improvement.
Augmentation uses horizontal flips with corresponding box transforms, gain
0.9–1.1, offset ±0.03 and Gaussian noise SD 0.01. Seed 2026 plus the fold index
and deterministic Torch operations make the local experiment reproducible
within the installed environment.

These choices bound this initial experiment; they are not optimal settings
established by a hyperparameter search.

## Local outputs and review

The completed experiment is under
`clarius_annotations/liver_localization_20261117/`:

- `REPORT.md`: mask scores, box metrics and paired improvements/regressions.
- `review.html`: all images with verified contours, prompts and predictions.
- `scores/per_image.csv`: complete image/method scores.
- `scores/summary.csv`, `scores/per_exam.csv`, `scores/paired_dice.csv`: summaries.
- `training.json`: exam splits, validation histories and out-of-fold boxes.
- `fold_0.pth` through `fold_4.pth`: models used for held-out evaluation.
- `liver_box_final.pth`: separate all-data model, trained for the median
  selected duration of 43 epochs, for trying new images only.
- `parents/`, `refinements/`: original-size predictions and audit metadata.

The final all-data model does not contribute to the cross-validation scores.
Model files, source images and per-image artifacts stay in the ignored local
annotation directory.

## Try automatic boxes in the viewer

From the repository directory, point `-i` at new clean MATLAB exports:

```powershell
python MedSAM_Clarius.py --checkpoint .\medsam_vit_b.pth --device cuda:0 `
  --localizer-checkpoint .\clarius_annotations\liver_localization_20261117\liver_box_final.pth `
  -i "C:\path\to\new\exported_images" --output .\clarius_annotations\localizer_review
```

The viewer predicts a box and runs MedSAM for new unannotated frames. Review
the mask before using **Split latest region** to compare texture methods.
Existing saved annotations load without automatic replacement. Saved JSON
records the localizer checkpoint hash and prompt margin. Undo and manual box
drawing remain available. Omitting `--localizer-checkpoint` preserves the
existing manual workflow.

The model always predicts one box. Training included liver-present images
only, so it has no liver-absence decision or calibrated confidence score.
Testing other probes, acquisition settings or liver-free views requires new
labels and evaluation. Liver localization also does not by itself solve
vessel-lumen exclusion.

## Reproduce in a fresh output folder

The scripts use the same audited reference manifest and clean source lookup
as the earlier verified evaluation. The initial training command refuses to
overwrite a completed training experiment. Use a new output directory for
different settings.

```powershell
python -B tools/train_liver_localizer.py --output .\clarius_annotations\localizer_experiment
python -B tools/evaluate_liver_localizer.py --output .\clarius_annotations\localizer_experiment
python -B tools/refine_localized_liver.py --output .\clarius_annotations\localizer_experiment --workers 8
python -B tools/train_liver_localizer.py --output .\clarius_annotations\localizer_experiment --fit-final-only
python -B tools/report_liver_localizer.py --output .\clarius_annotations\localizer_experiment
```

GPU training and MedSAM inference use CUDA. The existing GLCM and spatial
refinement still run on the CPU. Worker count affects memory and throughput.
Evaluation caches are checked against their inputs; use a fresh directory
when changing the experimental setup.
