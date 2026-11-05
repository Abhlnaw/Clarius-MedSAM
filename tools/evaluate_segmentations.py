"""Score multiple methods against the SAME per-image reference masks.

python -B tools/evaluate_segmentations.py --manifest path/to/manifest.json
Paths in a manifest are relative to that manifest, unless absolute. The manifest
must label reference provenance (e.g. AI provisional versus expert-reviewed).
No model inference or segmentation changes are made.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from clarius_metrics import evaluate_mask, binary_mask, SSIM_SETTINGS


def load_mask(specification, root):
    spec = {'path': specification} if isinstance(specification, str) else specification
    path = Path(spec['path'])
    if not path.is_absolute():
        path = root / path
    if spec.get('sha256') and hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
        raise ValueError(f"Mask changed since manifest creation: {path}")
    if path.suffix.lower() == '.npz':
        if 'key' not in spec:
            raise ValueError(f"NPZ mask needs an explicit array key: {path}")
        with np.load(path, allow_pickle=False) as saved:
            values = saved[spec['key']].copy()
    elif path.suffix.lower() == '.npy':
        values = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            # Preserve class IDs in MATLAB/indexed label images.
            if image.mode not in ('1', 'L', 'P', 'I', 'I;16'):
                raise ValueError(f"Expected a single-channel/indexed label mask: {path}")
            values = np.asarray(image).copy()
    if 'label' in spec:
        if values.ndim != 2:
            raise ValueError(f"Label mask must be 2D: {path}")
        values = values == spec['label']
    return binary_mask(values, str(path))


def evaluate_manifest(path, output):
    manifest = json.loads(path.read_text(encoding='utf-8-sig'))
    if not manifest.get('reference_provenance'):
        raise ValueError("Manifest needs reference_provenance; AI annotations must not be called ground truth")
    cases = manifest.get('cases', [])
    if not cases:
        raise ValueError("Manifest has no cases")
    names = None
    rows = []
    seen = set()
    for case in cases:
        if case['id'] in seen:
            raise ValueError(f"Duplicate case: {case['id']}")
        seen.add(case['id'])
        reference = load_mask(case['reference'], path.parent)
        predictions = {name: load_mask(spec, path.parent) for name, spec in case['predictions'].items()}
        if not predictions:
            raise ValueError(f"No predictions for {case['id']}")
        if names is None:
            names = set(predictions)
        elif set(predictions) != names:
            raise ValueError("All cases must contain the same methods for a paired comparison")
        # Use one common domain for every method; do not favor either outline.
        domain = load_mask(case['evaluation_mask'], path.parent) if 'evaluation_mask' in case else reference.copy()
        if domain.shape != reference.shape:
            raise ValueError(f"Evaluation mask shape mismatch: {case['id']}")
        domain = domain | reference
        for mask in predictions.values():
            if mask.shape != reference.shape:
                raise ValueError(f"Native shape mismatch: {case['id']}")
            domain |= mask
        for name, prediction in predictions.items():
            rows.append(dict(case=case['id'], exam=case.get('exam','unspecified'), method=name,
                             **evaluate_mask(prediction, reference, domain)))
    summary = {}
    for name in sorted(names):
        selected = [row for row in rows if row['method'] == name]
        summary[name] = {'n_images': len(selected)}
        for metric in ['ssim', 'roi_mean_ssim', 'dice', 'iou', 'precision', 'recall']:
            values = np.array([row[metric] for row in selected if row[metric] is not None])
            summary[name][metric] = dict(n=int(values.size), mean=float(values.mean()) if values.size else None,
                sample_sd=float(values.std(ddof=1)) if values.size > 1 else None,
                minimum=float(values.min()) if values.size else None,
                maximum=float(values.max()) if values.size else None)
    result = dict(reference_provenance=manifest['reference_provenance'],
        reference_review_status=manifest.get('reference_review_status','unspecified'),
        ssim_settings=SSIM_SETTINGS,
        evaluation_note='Reference-mask agreement. Full-frame SSIM includes shared background; ROI mean is a custom SSIM-map aggregation. Native grids, no resizing. Frame-level SD is not uncertainty across independent patients.',
        rows=rows, summary=summary)
    output.mkdir(parents=True, exist_ok=True)
    (output/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (output/'per_image.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = ['# Segmentation scores against reference masks', '',
        f"Reference provenance: **{manifest['reference_provenance']}**.",
        f"Review status: **{result['reference_review_status']}**.", '',
        'Each prediction is compared with the same per-image reference. Scores are not method-to-method comparisons.',
        'SSIM uses binary native-grid arrays, data_range=1, Gaussian sigma=1.5, 11x11 windows, and population covariance.',
        'ROI SSIM means the local SSIM map over a common ROI (parent + reference + every prediction), excluding image-border window centers.', '',
        '| Image | Method | SSIM | ROI mean SSIM | Dice | IoU |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        roi = f"{row['roi_mean_ssim']:.4f}" if row['roi_mean_ssim'] is not None else 'N/A'
        lines.append(f"| {row['case']} | {row['method']} | {row['ssim']:.4f} | {roi} | {row['dice']:.4f} | {row['iou']:.4f} |")
    lines += ['', '## Per-image macro averages', '', '| Method | SSIM mean +/- SD | ROI mean SSIM | Dice | IoU |',
              '| --- | ---: | ---: | ---: | ---: |']
    for name, stats in summary.items():
        sd = f"{stats['ssim']['sample_sd']:.4f}" if stats['ssim']['sample_sd'] is not None else 'N/A'
        roi = f"{stats['roi_mean_ssim']['mean']:.4f}" if stats['roi_mean_ssim']['mean'] is not None else 'N/A'
        lines.append(f"| {name} | {stats['ssim']['mean']:.4f} +/- {sd} | {roi} | {stats['dice']['mean']:.4f} | {stats['iou']['mean']:.4f} |")
    lines += ['', 'Full-frame SSIM can look high because both masks share a large background.',
        'Dice/IoU measure foreground overlap. AI-drawn references require expert review before anatomical-accuracy claims.',
        'These frames may share an examination and are not independent patient samples.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'clarius_annotations/reference_evaluation')
    args=parser.parse_args()
    result=evaluate_manifest(args.manifest.resolve(),args.output)
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
