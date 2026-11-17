# Hoyt research presentation

[Segmentation Presentation v2](https://docs.google.com/presentation/d/12M2oSVrSDxxy5LOPk8Q2qi9au5VoCi3_OdNhWBhh1iU/edit?usp=drivesdk)
uses the provided Texas A&M template in the shared research folder. It has
21 main slides for a 15–20 minute talk and 10 appendix slides.

The deck covers segmentation background, MATLAB image formation, MedSAM,
local brightness/GLCM/GLSZM refinement, the completed 98-image verified
comparison, localization failures, PI reference revisions and next steps.
Earlier seven-frame provisional experiments remain in the appendix with
their original limitations. Tables and narrative text are editable. Notes
include sources and protocol details.

`hoyt_content_v1.json` retains the prior deck's content and source references.
`tools/build_hoyt_presentation_v2.mjs` generates native Slides requests using
the inspected template and actual saved evaluation results.
`tools/prepare_verified_presentation_assets.py` creates scientific overlays
from saved prediction/reference arrays without rerunning inference. Example
selection is deterministic: the upper-quartile Combined GLCM Dice case and
the first zero-overlap case. These examples do not replace aggregate results.

Build inputs and renders are local under
`clarius_annotations/hoyt_presentation_v2_20261117/`. The build requires full
Drive snapshots `raw-source.json` and `raw-template.json`, the matching
evaluation output, and the earlier local scientific image assets. Snapshots
must wrap the full native presentation in `structuredContent`, as returned
by the connected Google Drive reader.

```powershell
python -B tools/prepare_verified_presentation_assets.py
node tools/build_hoyt_presentation_v2.mjs
```

The builder writes `requests.json`, `image-jobs.json` and `slide-plan.json`.
It does not perform network writes. A connected Slides client must apply the
structured requests, upload each image sidecar, insert the notes into the
generated notes object IDs, remove the unused template exemplars, and reorder
slides. Read request files as UTF-8, including `Get-Content -Encoding UTF8`
when using Windows PowerShell. Inspect the final native deck after changes.

The original A&M PowerPoint template and previous presentation remain intact.
See [verified evaluation](../clarius/VERIFIED_EVALUATION.md) for the protocol,
scores and the limitations that the deck reports.
