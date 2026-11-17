"""Evaluate the existing methods against user-supplied verified red contours.

The default prompt is the complete clean image. Reference contours never enter
model inference or refinement. Only the explicit reference-box option uses
reference coordinates for a prompted-assistance experiment.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

for thread_variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[thread_variable] = '1'

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clarius_auto import automatic_tissue_region
from clarius_glcm import local_glcm_features
from clarius_glszm import local_glszm_features
from clarius_texture import analyze_texture, localized_window
from tools.evaluate_segmentations import evaluate_manifest
from tools.review_drawn_outlines import overlay

METHODS = ('medsam', 'brightness_auto', 'glcm', 'combined_glcm', 'glszm', 'brightness_glszm')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def convert_outline(rgb, connect_open_ends=False):
    """Fill the largest enclosed contour compartment while retaining holes.

    Nearby skeleton endpoints permit at most 10-pixel technical seam closure.
    A longer direct connector requires an explicit per-image authorization.
    Boundary pixels inherit the nearest compartment, as in the existing
    contour-conversion helper. No previous model/reference mask supplies seeds.
    """
    values = np.asarray(rgb, dtype=np.int16)
    ink = ((values[..., 0] > 120) & (values[..., 0] - values[..., 1] > 35)
           & (values[..., 0] - values[..., 2] > 35))
    if ink.sum() < 20:
        raise ValueError('No red outline')
    skeleton = skeletonize(ink)
    neighbors = ndi.convolve(skeleton.astype(int), np.ones((3, 3), int)) - skeleton
    endpoints = np.argwhere(skeleton & (neighbors == 1))
    canvas = Image.fromarray(ink.astype('uint8') * 255)
    pen = ImageDraw.Draw(canvas)
    bridges = []
    if len(endpoints):
        if connect_open_ends:
            if len(endpoints) != 2:
                raise ValueError('Authorized straight closure requires exactly two endpoints')
            pairs = [(0, 1)]
        else:
            distances = cdist(endpoints, endpoints)
            pairs = sorted(((i, j) for i in range(len(endpoints))
                            for j in range(i + 1, len(endpoints)) if distances[i, j] <= 10),
                           key=lambda pair: distances[pair])
        used = set()
        for first, second in pairs:
            if first in used or second in used:
                continue
            start = tuple(map(int, endpoints[first][::-1]))
            end = tuple(map(int, endpoints[second][::-1]))
            pen.line([start, end], fill=255, width=3)
            bridges.append(dict(start_xy=start, end_xy=end,
                                distance_pixels=float(np.linalg.norm(endpoints[first] - endpoints[second])),
                                reason='user-authorized straight closure' if connect_open_ends
                                else 'technical endpoint seam of at most 10 pixels'))
            used.update((first, second))
    drawn = np.asarray(canvas) > 0
    barrier = ndi.binary_dilation(ndi.binary_closing(drawn, iterations=1) | drawn, iterations=1)
    labels, _ = ndi.label(~barrier)
    exterior = np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    areas = np.bincount(labels.ravel())
    areas[exterior] = 0
    areas[0] = 0
    interior = labels == int(areas.argmax())
    if areas.max() < ink.size * 0.02:
        raise ValueError('Outer outline does not enclose a usable region')
    nearest = ndi.distance_transform_edt(barrier, return_distances=False, return_indices=True)
    mask = interior.copy()
    mask[barrier] = interior[tuple(nearest[:, barrier])]
    holes = ndi.label(ndi.binary_fill_holes(mask) & ~mask)[1]
    return mask, ink, dict(ink_pixels=int(ink.sum()), bridges=bridges,
                           holes=int(holes), foreground_pixels=int(mask.sum()),
                           boundary_rule='Nearest compartment assignment across detected stroke and 1-pixel closure/dilation')


def prepare(args):
    source = json.loads(args.sources.read_text(encoding='utf-8-sig'))
    lookup = {case['id']: case for case in source['cases']}
    records, excluded = [], []
    for folder in ('references', 'conversion_review', 'parents', 'predictions', 'details', 'comparisons'):
        (args.output / folder).mkdir(parents=True, exist_ok=True)
    for path in sorted(args.references.iterdir()):
        if path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        key = path.stem[:3]
        if key in args.exclude:
            excluded.append(dict(id=path.stem, reason='User excluded image without an outline', source=str(path.resolve()), sha256=digest(path)))
            continue
        entry = lookup[path.stem]
        clean_path = Path(entry['source'])
        with Image.open(path) as handle:
            rgb = ImageOps.exif_transpose(handle).convert('RGB')
        with Image.open(clean_path) as handle:
            clean = handle.convert('RGB')
        if rgb.size != clean.size:
            raise ValueError(f'{key}: reference and clean source dimensions differ')
        mask, ink, conversion = convert_outline(rgb, key in args.connect)
        clean_gray = np.asarray(clean).mean(-1)
        marked_gray = np.asarray(rgb).mean(-1)
        valid = ~ndi.binary_dilation(ink, iterations=4)
        mae = float(np.abs(clean_gray[valid] - marked_gray[valid]).mean())
        if mae > 5:
            raise ValueError(f'{key}: source appearance mismatch (mean absolute difference {mae:.3f})')
        reference_path = args.output / 'references' / (path.stem + '.png')
        Image.fromarray(mask.astype('uint8') * 255).save(reference_path)
        overlay(clean, mask).save(args.output / 'conversion_review' / (path.stem + '.png'))
        records.append(dict(id=path.stem, clean_image=str(clean_path.resolve()), clean_sha256=digest(clean_path),
            annotated_image=str(path.resolve()), annotated_sha256=digest(path),
            reference=str(reference_path.resolve()), reference_sha256=digest(reference_path),
            shape_hw=list(mask.shape), source_agreement_mae=mae,
            exam=re.search(r'\(exam (\d+)\)', path.stem).group(1), conversion=conversion))
    manifest = dict(reference_provenance='PhD-student-verified red contours supplied by the user in final_ground_truths; raster conversion is separately audited',
        reference_review_status='Verified by PhD student according to the user; 021 straight closure explicitly authorized',
        clean_input_processing='MATLAB env_gamma08, full 1418 by 853 grid',
        prompt_protocol=args.prompt, box_padding=args.box_padding,
        levels=8, groups=3, quantile=.985, glszm_step=[8, 2],
        excluded=excluded, cases=records)
    write_json(args.output / 'reference_manifest.json', manifest)
    for page in range((len(records) + 3) // 4):
        sheet = Image.new('RGB', (1450, 934), 'white')
        pen = ImageDraw.Draw(sheet)
        for slot, case in enumerate(records[page * 4:page * 4 + 4]):
            with Image.open(args.output / 'conversion_review' / (case['id'] + '.png')) as image:
                thumbnail = image.resize((709, 427), Image.Resampling.BILINEAR)
            x, y = slot % 2 * 725, slot // 2 * 467
            sheet.paste(thumbnail, (x, y + 28))
            pen.text((x + 5, y + 6), f"{case['id'][:3]} holes={case['conversion']['holes']} bridges={len(case['conversion']['bridges'])}", fill='black')
        sheet.save(args.output / 'conversion_review' / f'page_{page + 1:02d}.png')
    print(f'Prepared {len(records)} references, excluded {len(excluded)} images', flush=True)
    return manifest


def parent_masks(args, cases):
    import torch
    from torch.nn import functional as F
    from skimage import transform
    from segment_anything import sam_model_registry
    from MedSAM_Clarius import load_rgb
    torch.set_num_threads(1)
    device = torch.device(args.device)
    checkpoint_hash = digest(args.checkpoint)
    model = None
    for index, case in enumerate(cases):
        destination = args.output / 'parents' / (case['id'] + '.npz')
        metadata_path = args.output / 'parents' / (case['id'] + '.json')
        reference = np.asarray(Image.open(case['reference'])) > 0
        height, width = reference.shape
        if args.prompt == 'full-image':
            box = np.array([0, 0, width, height], dtype=np.float32)
        else:
            rows, columns = np.nonzero(reference)
            padding = args.box_padding
            box = np.array([max(0, columns.min() - padding), max(0, rows.min() - padding),
                            min(width, columns.max() + 1 + padding), min(height, rows.max() + 1 + padding)], dtype=np.float32)
        signature = dict(clean_sha256=case['clean_sha256'], checkpoint_sha256=checkpoint_hash,
                         prompt_protocol=args.prompt, box_xyxy=box.tolist())
        if destination.exists() and metadata_path.exists():
            saved = json.loads(metadata_path.read_text())
            if saved['signature'] != signature:
                raise ValueError('Parent cache configuration differs. Choose a fresh output folder.')
            continue
        if model is None:
            print(f'Loading MedSAM on {device}', flush=True)
            model = sam_model_registry['vit_b'](checkpoint=str(args.checkpoint)).to(device).eval()
        start = time.perf_counter()
        image = load_rgb(Path(case['clean_image']))
        resized = transform.resize(image, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True)
        resized = (resized - resized.min()) / max(float(np.ptp(resized)), 1e-8)
        tensor = torch.from_numpy(resized.astype(np.float32)).permute(2, 0, 1)[None].to(device)
        prompt = torch.as_tensor(box / [width, height, width, height] * 1024, dtype=torch.float32, device=device)[None, None]
        with torch.inference_mode():
            embedding = model.image_encoder(tensor)
            sparse, dense = model.prompt_encoder(points=None, boxes=prompt, masks=None)
            logits, _ = model.mask_decoder(image_embeddings=embedding, image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False)
            probabilities = F.interpolate(torch.sigmoid(logits), (height, width), mode='bilinear', align_corners=False)
            mask = probabilities[0, 0].cpu().numpy() > .5
        np.savez_compressed(destination, parent=mask)
        write_json(metadata_path, dict(signature=signature, device=str(device), seconds=time.perf_counter() - start,
                                      foreground_pixels=int(mask.sum()), preprocessing='Existing MedSAM_Clarius.load normalization and 1024-square encoder preparation'))
        print(f'Parent {index + 1}/{len(cases)} {case["id"][:3]} area={mask.sum()}', flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def refine_case(case, output):
    output = Path(output)
    destination = output / 'predictions' / (case['id'] + '.npz')
    detail_path = output / 'details' / (case['id'] + '.json')
    if destination.exists() and detail_path.exists():
        cached = json.loads(detail_path.read_text())
        if (cached['prediction_sha256'] == digest(destination)
                and cached['parent_sha256'] == digest(output / 'parents' / (case['id'] + '.npz'))
                and cached['clean_sha256'] == case['clean_sha256']):
            return cached
    gray = np.asarray(Image.open(case['clean_image']).convert('L'), dtype=np.float32)
    gray = (gray - gray.min()) / max(float(np.ptp(gray)), 1e-8)
    with np.load(output / 'parents' / (case['id'] + '.npz')) as saved:
        parent = saved['parent'].astype(bool)
    masks = {'medsam': parent}
    timing = {'medsam': dict(refinement_seconds=0)}
    failures = {}
    if parent.any():
        window = localized_window(parent)
        for family, methods in [('brightness', [('brightness_auto', 'brightness')]),
                                ('glcm', [('glcm', 'glcm'), ('combined_glcm', 'combined')]),
                                ('glszm', [('glszm', 'glszm'), ('brightness_glszm', 'combined_glszm')])]:
            started = time.perf_counter()
            maps = None
            try:
                if family == 'glcm':
                    maps = local_glcm_features(gray, parent, window, levels=8)
                elif family == 'glszm':
                    maps = local_glszm_features(gray, parent, window, levels=8, step=(8, 2))
            except Exception as error:
                failures[family + '_features'] = str(error)
            map_seconds = time.perf_counter() - started
            for name, method in methods:
                started = time.perf_counter()
                try:
                    if family != 'brightness' and maps is None:
                        raise ValueError('Feature extraction failed')
                    proposal = analyze_texture(gray, parent, method=method, levels=8,
                        glcm_data=maps if family == 'glcm' else None,
                        glszm_data=maps if family == 'glszm' else None, glszm_step=(8, 2))
                    mask, diagnostics = automatic_tissue_region(gray, parent, proposal, correlation_quantile=.985)
                    masks[name] = mask
                    timing[name] = dict(map_seconds=map_seconds, proposal_and_refinement_seconds=time.perf_counter() - started,
                                        total_seconds=map_seconds + time.perf_counter() - started, diagnostics=diagnostics)
                    del proposal
                except Exception as error:
                    masks[name] = np.zeros(parent.shape, bool)
                    failures[name] = str(error)
            del maps
    else:
        failures['medsam'] = 'Empty parent prediction'
    for method in METHODS:
        masks.setdefault(method, np.zeros(parent.shape, bool))
    temporary = destination.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **masks)
    temporary.replace(destination)
    record = dict(id=case['id'], timing=timing, failures=failures,
                  prediction_sha256=digest(destination), parent_sha256=digest(output / 'parents' / (case['id'] + '.npz')),
                  clean_sha256=case['clean_sha256'], reference_sha256=case['reference_sha256'])
    write_json(detail_path, record)
    return record


def score(args, manifest, cases):
    evaluation = dict(reference_provenance=manifest['reference_provenance'], reference_review_status=manifest['reference_review_status'], cases=[])
    for case in cases:
        file = args.output / 'predictions' / (case['id'] + '.npz')
        evaluation['cases'].append(dict(id=case['id'], exam=case['exam'],
            reference=dict(path=case['reference'], label=255, sha256=case['reference_sha256']),
            predictions={method: dict(path=str(file.resolve()), key=method) for method in METHODS},
            evaluation_mask=dict(path=str(file.resolve()), key='medsam')))
    evaluation['protocol'] = dict(prompt=args.prompt, clean_input='MATLAB env_gamma08', levels=8, quantile=.985,
        glszm_step=[8, 2], brightness_baseline='Brightness proposals with the same automatic selector',
        failed_prediction_rule='Retain failed methods as empty masks in the paired score set and log failures')
    write_json(args.output / 'evaluation_manifest.json', evaluation)
    result = evaluate_manifest(args.output / 'evaluation_manifest.json', args.output / 'scores')
    print(json.dumps({name: {metric: stats[metric]['mean'] for metric in ('ssim', 'roi_mean_ssim', 'dice', 'iou')}
                      for name, stats in result['summary'].items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', type=Path, default=ROOT / 'final_ground_truths')
    parser.add_argument('--sources', type=Path, default=ROOT / 'clarius_annotations/liver_env_gamma08_v2/source_manifest.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'clarius_annotations/verified_evaluation_20261117')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'medsam_vit_b.pth')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--prompt', choices=('full-image', 'reference-box'), default='full-image')
    parser.add_argument('--box-padding', type=int, default=20)
    parser.add_argument('--exclude', nargs='*', default=['006', '045'])
    parser.add_argument('--connect', nargs='*', default=['021'])
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--reuse-prepared', action='store_true')
    parser.add_argument('--parents-only', action='store_true')
    parser.add_argument('--refine-only', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'run_configuration.json', dict(workers=args.workers, prompt=args.prompt,
        levels=8, quantile=.985, glszm_step=[8, 2], methods=list(METHODS),
        algorithm_sha256={name: digest(ROOT / name) for name in
            ('clarius_auto.py', 'clarius_glcm.py', 'clarius_glszm.py', 'clarius_texture.py', 'MedSAM_Clarius.py')}))
    if args.reuse_prepared:
        manifest = json.loads((args.output / 'reference_manifest.json').read_text())
        if manifest['prompt_protocol'] != args.prompt:
            raise ValueError('Prepared prompt protocol differs')
        for case in manifest['cases']:
            for path_key, hash_key in [('annotated_image', 'annotated_sha256'),
                                       ('clean_image', 'clean_sha256'), ('reference', 'reference_sha256')]:
                if digest(case[path_key]) != case[hash_key]:
                    raise ValueError(f'Prepared input changed: {case[path_key]}')
    else:
        manifest = prepare(args)
    cases = [c for c in manifest['cases'] if not args.cases or c['id'][:3] in args.cases]
    if args.prepare_only:
        return
    if args.refine_only:
        checkpoint_hash = digest(args.checkpoint)
        for case in cases:
            cached = json.loads((args.output / 'parents' / (case['id'] + '.json')).read_text())
            if (cached['signature']['clean_sha256'] != case['clean_sha256']
                    or cached['signature']['checkpoint_sha256'] != checkpoint_hash
                    or cached['signature']['prompt_protocol'] != args.prompt):
                raise ValueError('Cached parent configuration differs')
    else:
        parent_masks(args, cases)
    if args.parents_only:
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(refine_case, case, str(args.output)): case['id'] for case in cases}
        for index, future in enumerate(as_completed(jobs), 1):
            result = future.result()
            print(f'Refined {index}/{len(cases)} {result["id"][:3]} failures={list(result["failures"])}', flush=True)
    score(args, manifest, cases)


if __name__ == '__main__':
    main()
