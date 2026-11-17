"""Package verified-reference scores and native-grid visual comparisons."""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import statistics
import sys
from urllib.parse import quote

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_verified_outlines import METHODS


def package(root):
    root = Path(root).resolve()
    reference_manifest = json.loads((root / 'reference_manifest.json').read_text())
    result = json.loads((root / 'scores/metrics.json').read_text())
    rows = {(row['case'], row['method']): row for row in result['rows']}
    expected_pairs = {(case['id'], method) for case in reference_manifest['cases'] for method in METHODS}
    if set(rows) != expected_pairs or len(result['rows']) != len(expected_pairs):
        raise ValueError('Scores must contain exactly one row per reference/method pair')
    summary_rows = []
    method_failures = {name: 0 for name in METHODS}
    records = []
    durations = {name: [] for name in METHODS}
    validation = []
    (root / 'visual_masks').mkdir(exist_ok=True)
    for case in reference_manifest['cases']:
        key = case['id']
        detail = json.loads((root / 'details' / (key + '.json')).read_text())
        prediction_path = root / 'predictions' / (key + '.npz')
        if hashlib.sha256(prediction_path.read_bytes()).hexdigest() != detail['prediction_sha256']:
            raise ValueError(f'Prediction changed after evaluation: {key}')
        for method in METHODS:
            method_failures[method] += int(method in detail['failures'] or
                                           (method != 'medsam' and 'medsam' in detail['failures']))
            if method in detail['timing'] and 'total_seconds' in detail['timing'][method]:
                durations[method].append(detail['timing'][method]['total_seconds'])
        reference = np.asarray(Image.open(case['reference'])) > 0
        reference_rgba = np.zeros((*reference.shape, 4), np.uint8)
        border = ndi.binary_dilation(reference ^ ndi.binary_erosion(reference), iterations=1)
        reference_rgba[border] = [30, 255, 80, 255]
        reference_name = key + '.reference.png'
        Image.fromarray(reference_rgba).save(root / 'visual_masks' / reference_name)
        item = dict(id=key, image=quote(os.path.relpath(case['clean_image'], root).replace('\\', '/'), safe='/'),
                    reference='visual_masks/' + quote(reference_name), methods={})
        with np.load(root / 'predictions' / (key + '.npz'), allow_pickle=False) as predictions:
            for method in METHODS:
                prediction = predictions[method].astype(bool)
                if prediction.shape != reference.shape:
                    raise ValueError('Reference and prediction dimensions differ')
                rgba = np.zeros((*prediction.shape, 4), np.uint8)
                rgba[prediction] = [255, 140, 0, 100]
                name = key + '.' + method + '.png'
                Image.fromarray(rgba).save(root / 'visual_masks' / name)
                item['methods'][method] = dict(image='visual_masks/' + quote(name), **rows[(key, method)])
        records.append(item)
        for field, digest_field in [('annotated_image', 'annotated_sha256'), ('clean_image', 'clean_sha256'),
                                    ('reference', 'reference_sha256')]:
            actual = hashlib.sha256(Path(case[field]).read_bytes()).hexdigest()
            if actual != case[digest_field]:
                raise ValueError(f'Input changed during evaluation: {case[field]}')
        validation.append(dict(id=key, shape_hw=list(reference.shape), input_hashes_match=True))
    for excluded in reference_manifest['excluded']:
        if hashlib.sha256(Path(excluded['source']).read_bytes()).hexdigest() != excluded['sha256']:
            raise ValueError(f'Excluded original changed: {excluded["source"]}')
    for method in METHODS:
        summary = result['summary'][method]
        summary_rows.append(dict(method=method, n_images=summary['n_images'],
            **{metric: summary[metric]['mean'] for metric in ('ssim', 'roi_mean_ssim', 'dice', 'iou', 'precision', 'recall')},
            zero_overlap_images=sum(rows[(case['id'], method)]['dice'] == 0 for case in reference_manifest['cases']),
            algorithm_failures=method_failures[method],
            median_refinement_seconds=statistics.median(durations[method]) if durations[method] else None))
    with (root / 'scores/summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    groups = sorted({case['exam'] for case in reference_manifest['cases']})
    grouped = []
    for group in groups:
        cases = [case for case in reference_manifest['cases'] if case['exam'] == group]
        for method in METHODS:
            grouped.append(dict(exam=group, method=method, n_images=len(cases),
                **{metric: statistics.mean(rows[(case['id'], method)][metric] for case in cases)
                   for metric in ('ssim', 'roi_mean_ssim', 'dice', 'iou')}))
    with (root / 'scores/per_exam.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grouped[0]))
        writer.writeheader()
        writer.writerows(grouped)
    report = ['# Verified-reference evaluation', '',
        f"Scored {len(records)} annotated images from {len(groups)} recorded examinations.",
        'References were verified by the PhD student according to the user. Red contours were rasterized separately with an audit log.',
        'Files 006 and 045 were excluded at the user’s request. Image 021 received the explicitly authorized straight endpoint connector.',
        'Model inputs are the matching clean MATLAB env_gamma08 images, at 1418 by 853 pixels.',
        'All images use the complete-image box [0, 0, 1418, 853]. Reference boundaries and image location labels do not guide inference or refinement.',
        'All texture settings remain at their existing defaults: eight levels, q0.985, GLSZM step (8,2), up to three source groups.', '',
        '| Method | SSIM | ROI SSIM | Dice | IoU | Precision | Recall | Zero overlap | Failures |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in summary_rows:
        metrics = ' | '.join(f"{row[name]:.4f}" if row[name] is not None else 'N/A'
                             for name in ('ssim', 'roi_mean_ssim', 'dice', 'iou', 'precision', 'recall'))
        report.append(f"| {row['method']} | {metrics} | {row['zero_overlap_images']} | {row['algorithm_failures']} |")
    report += ['', 'Values are per-image macro means. Repeated captures within an examination are correlated. Per-examination means are in `scores/per_exam.csv`.',
        'Zero overlap counts valid predictions with no reference intersection. Failures count algorithm exceptions or an empty parent, retained as empty masks in scoring.',
        'Brightness_auto uses brightness proposals with the same automatic region selector. The other refinements reuse their corresponding current source methodology.',
        'Full-frame SSIM includes shared background. ROI SSIM uses the same union of parent, reference and all predictions for every method.',
        'The full-image prompt evaluates the requested automatic workflow. Earlier targeted-box development results use a different protocol.', '',
        '## Outputs', '',
        '- `review.html` compares predictions with green verified contours and orange predicted regions.',
        '- `scores/per_image.csv` contains every image/method score.',
        '- `scores/summary.csv` contains aggregate metrics and zero-overlap counts.',
        '- `reference_manifest.json` records the original drawing hashes and all seam closures.',
        '- `parents/` and `predictions/` retain the evaluated masks.',
        '- `details/` records method diagnostics, failures and runtime.', '',
        'CPU texture calculations include a serial pilot and parallel runs with four and eight workers. Per-case wall times include contention and are not directly comparable with earlier serial laptop timings.',
        'No parameter selection used these reference scores. Patient identity across examinations has not been established.']
    (root / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    (root / 'validation.json').write_text(json.dumps(dict(images=len(records), examinations=len(groups),
        methods=list(METHODS), expected_score_rows=len(records) * len(METHODS), actual_score_rows=len(result['rows']),
        prompt='full-image', reference_provenance=reference_manifest['reference_provenance'],
        inputs_unchanged=True, cases=validation), indent=2), encoding='utf-8')
    options = ''.join(f'<option value="{i}">{html.escape(case["id"])}</option>' for i, case in enumerate(records))
    method_options = ''.join(f'<option value="{method}">{method}</option>' for method in METHODS)
    page = '''<!doctype html><html><head><meta charset="utf-8"><title>Verified segmentation comparison</title><style>body{font:16px Arial;margin:18px;background:#181818;color:#eee}nav{display:flex;gap:8px;flex-wrap:wrap}button,select{font:inherit;padding:6px}#case{max-width:75vw}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.frame{position:relative}.frame img{display:block;width:100%;height:auto}.frame .layer{position:absolute;inset:0}h2{font-size:21px}a{color:#8bcfff}p{line-height:1.45}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style></head><body><h2>Verified reference comparison</h2><p>Green outlines show the verified reference. Orange fill shows the prediction. Fixed full-image prompts, current defaults.</p><nav><button onclick="step(-1)">Previous</button><select id="case" onchange="show()">CASES</select><button onclick="step(1)">Next</button><label><input id="reference" type="checkbox" checked onchange="show()">Show verified outline</label></nav><div class="grid"><section><p><select id="left" onchange="show()">METHODS</select></p><div class="frame"><img id="leftImage"><img class="layer" id="leftPrediction"><img class="layer" id="leftReference"></div><p id="leftScore"></p></section><section><p><select id="right" onchange="show()">METHODS</select></p><div class="frame"><img id="rightImage"><img class="layer" id="rightPrediction"><img class="layer" id="rightReference"></div><p id="rightScore"></p></section></div><p><a href="REPORT.md">Report</a> · <a href="scores/summary.csv">Summary CSV</a> · <a href="scores/per_image.csv">Per-image CSV</a></p><script>const cases=DATA;const choice=document.getElementById('case');document.getElementById('left').value='combined_glcm';document.getElementById('right').value='brightness_glszm';function show(){const c=cases[+choice.value];for(const side of ['left','right']){const m=c.methods[document.getElementById(side).value];document.getElementById(side+'Image').src=c.image;document.getElementById(side+'Prediction').src=m.image;document.getElementById(side+'Reference').src=c.reference;document.getElementById(side+'Reference').style.display=document.getElementById('reference').checked?'block':'none';document.getElementById(side+'Score').textContent=(+choice.value+1)+'/'+cases.length+'   Dice '+m.dice.toFixed(4)+'   IoU '+m.iou.toFixed(4)+'   SSIM '+m.ssim.toFixed(4)+'   ROI SSIM '+m.roi_mean_ssim.toFixed(4);}}function step(n){choice.value=Math.max(0,Math.min(cases.length-1,+choice.value+n));show();}document.addEventListener('keydown',e=>{if(e.target.tagName==='SELECT')return;if(e.key==='ArrowRight')step(1);if(e.key==='ArrowLeft')step(-1);});show();</script></body></html>'''
    page = page.replace('CASES', options).replace('METHODS', method_options).replace('DATA', json.dumps(records).replace('</', '<\\/'))
    (root / 'review.html').write_text(page, encoding='utf-8')
    print(json.dumps(dict(images=len(records), examinations=len(groups), summary=summary_rows), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    package(parser.parse_args().root)
