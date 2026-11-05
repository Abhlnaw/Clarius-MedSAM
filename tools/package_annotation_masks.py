"""Package provisional reference masks without overwriting later manual edits."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_unchanged(source,target):
    if target.exists():
        if digest(source)!=digest(target):
            raise ValueError(f'Refusing to overwrite an existing edited file: {target}')
    else:shutil.copy2(source,target)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    root=args.root.resolve();review=root/'review'
    manual=json.loads((review/'manual_conversion_audit.json').read_text())
    drafts=json.loads((review/'draft_audit.json').read_text());records=manual+drafts
    assert len(records)==157 and len({r['id'] for r in records})==157
    for folder in ('model_inputs','outlines','human_drawings'):(root/folder).mkdir(exist_ok=True)
    for r in records:
        name=r['id']+'.png';human=r['source']=='user red drawing'
        original=root/'images'/name;clean=root/'clean_source_export'/'images'/name
        if human:
            assert digest(original)==r['source_sha256']
            copy_unchanged(original,root/'human_drawings'/name)
        else:
            # MATLAB can embed different PNG metadata; compare pixels.
            assert np.array_equal(np.asarray(Image.open(original)),np.asarray(Image.open(clean)))
        copy_unchanged(clean,root/'model_inputs'/name)
        mask_source=root/('converted_manual_masks' if human else 'draft_masks')/name
        destination=root/'masks'/name;copy_unchanged(mask_source,destination)
        overlay=review/(r['id']+('.manual.png' if human else '.draft.png'))
        copy_unchanged(overlay,root/'outlines'/name)
        with Image.open(clean) as image:
            a=np.asarray(image);assert image.size==(1418,853) and image.mode=='L' and a.max()>a.min()
        with Image.open(destination) as image:
            mask=np.asarray(image);assert image.size==(1418,853) and image.mode=='L'
            assert set(np.unique(mask))=={0,255}
            assert ndi.label(mask>0,structure=np.ones((3,3)))[1]==1,(r['id'],'disconnected foreground')
        r.update(mask='masks/'+name,model_input='model_inputs/'+name,outline='outlines/'+name,
                 mask_sha256=digest(destination),model_input_sha256=digest(clean),
                 independent_review_status='pending')
    manifest=dict(image_width=1418,image_height=853,foreground_label=255,
        reference_status='provisional: no anatomical ground-truth approval inferred',
        annotation_assistance='25 user drawings converted with logged gap closure; 49 visually authored draft anchors, 83 registered captures, 64 individual visual corrections',cases=records)
    (root/'mask_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    bridged=sum(bool(r['bridged_endpoints']) for r in manual)
    report=f'''# Annotation review

## First 25 drawings

All 25 are 1418 x 853. Their broad outlines and intended internal exclusions were preserved in conversion. The original drawings are unchanged (verified by hashes) and backed up in human_drawings/. This is technical and visual review, not expert confirmation of anatomical boundaries.

{bridged} drawings needed small endpoint bridges (at most 20 pixels), including extensions to the image boundary. Every bridge is recorded in manual_conversion_audit.json. Check those closures and dark-region exclusions before approval. Images 014, 023 and 024 have no enclosed hole because none was drawn; none was invented during conversion.

[First 25 conversion review sheet](first25_converted.png)

## Remaining 132

These are approximate drafts: 49 visually authored anchor contours and 83 within-exam optical-flow transfers, with individual visual revisions to 64 images. They are not 132 individually hand-drawn annotations. No MedSAM/GLCM/GLSZM prediction was used to construct them. All remain pending independent review in ../mask_manifest.json.

All 132 were visually inspected on contact sheets. Review the superficial/deep borders, image-edge contact, and every internal exclusion at full resolution. Some source boundaries and dark regions are ambiguous. Elliptical lumen approximations and transferred boundaries need correction where they do not follow the visible anatomy. Images 035/036 received independent outer-boundary corrections after transfer drift was found. Contact sheets do not establish fine boundary accuracy.

Use ../outlines/ for full-resolution red overlays, ../model_inputs/ for clean inputs, and ../masks/ for editable binary masks. The earlier images/ directory includes your red drawings and must not be used as model input. Clean model inputs were reconstructed from unmodified archives without cropping.

Have the independent reviewer correct masks and record approval before final scoring. Keep model predictions hidden while deciding reference boundaries. Preserve filenames, the full 1418 x 853 canvas, and binary values 0/255. Annotation and testing use the same grid.

'''
    for i in range(1,12):report+=f'- [Draft page {i}](drafts_{i:02d}.png)\n'
    (review/'README.md').write_text(report,encoding='utf-8')
    (root/'README.md').write_text('''# Manual reference annotation set

157 full-frame 1418 x 853 ultrasound images; original exam## folders excluded.

- model_inputs/: CLEAN inputs for model testing; reconstructed from unchanged archives without annotation pixels or cropping.
- masks/: 157 binary candidates. 001-025 converted from your drawings; 026-157 assisted drafts awaiting review.
- outlines/: full-resolution red contours on clean ultrasound for review.
- human_drawings/: backups of your 25 original drawings.
- images/: earlier exports, including your red markup on 001-025; retained untouched. Use model_inputs/ for testing.
- mask_manifest.json: provenance, pending review state, hashes, and paths.
- review/README.md: findings and contact sheets.

Preserve filenames, 1418 x 853 dimensions, and mask values 0/255. Independent review/correction is required before treating these candidates as ground truth.
''',encoding='utf-8')
    print(f'Validated 157 clean inputs, masks and overlays; 25 originals unchanged; {bridged}/25 need logged closure review.')


if __name__=='__main__':main()
