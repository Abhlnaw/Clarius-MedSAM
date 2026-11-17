"""Prepare scientific overlays from saved evaluation arrays, without inference."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]


def main():
    root = ROOT / 'clarius_annotations/verified_evaluation_20261117'
    output = ROOT / 'clarius_annotations/hoyt_presentation_v2_20261117/assets'
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / 'reference_manifest.json').read_text())
    scores = json.loads((root / 'scores/metrics.json').read_text())['rows']
    ordered = sorted((row for row in scores if row['method'] == 'combined_glcm'), key=lambda r: (r['dice'], r['case']))
    # Show an upper-quartile example and the first zero-overlap case, selected
    # by a recorded rule rather than manual visual preference.
    selected = [('upper_quartile', ordered[round(.75 * (len(ordered) - 1))]['case']),
                ('zero_overlap', next(row['case'] for row in scores if row['method'] == 'combined_glcm' and row['dice'] == 0))]
    records = []
    for role, key in selected:
        case = next(case for case in manifest['cases'] if case['id'] == key)
        clean = np.asarray(Image.open(case['clean_image']).convert('RGB'))
        reference = np.asarray(Image.open(case['reference'])) > 0
        border = ndi.binary_dilation(reference ^ ndi.binary_erosion(reference))
        with np.load(root / 'predictions' / (key + '.npz'), allow_pickle=False) as masks:
            images = []
            for method in ('combined_glcm', 'brightness_glszm'):
                prediction = masks[method].astype(bool)
                rgb = clean.copy()
                rgb[prediction] = (clean[prediction] * .6 + np.array([255, 140, 0]) * .4).astype('uint8')
                rgb[border] = [30, 255, 80]
                filename = output / f'{role}_{method}.png'
                Image.fromarray(rgb).save(filename)
                images.append(str(filename))
        rows = {row['method']: row for row in scores if row['case'] == key}
        records.append(dict(role=role, id=key, images=images, scores=rows))
    (output.parent / 'example-selection.json').write_text(json.dumps(records, indent=2))
    print(json.dumps([dict(role=r['role'], id=r['id']) for r in records]))


if __name__ == '__main__':
    main()
