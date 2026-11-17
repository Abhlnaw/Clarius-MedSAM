"""Apply unchanged Combined/automatic texture refinement to localized parents."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from clarius_texture import analyze_texture
from tools.evaluate_segmentations import evaluate_manifest


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def refine(case,root):
    root=Path(root)
    parent_path=root/'parents'/(case['id']+'.npz')
    destination=root/'refinements'/(case['id']+'.npz')
    metadata_path=destination.with_suffix('.json')
    signature=dict(parent_sha256=digest(parent_path),clean_sha256=digest(case['clean_image']),
                   algorithm_sha256={name:digest(ROOT/name) for name in ('clarius_texture.py','clarius_glcm.py','clarius_auto.py')})
    if signature['clean_sha256']!=case['clean_sha256']:raise ValueError('Clean image changed')
    if destination.exists() and metadata_path.exists():
        detail=json.loads(metadata_path.read_text())
        if detail['signature']==signature and detail['prediction_sha256']==digest(destination):return detail
        raise ValueError('Refinement cache changed')
    gray=np.asarray(Image.open(case['clean_image']).convert('L'),dtype=np.float32)
    gray=(gray-gray.min())/max(float(np.ptp(gray)),1e-8)
    masks,methods={},{}
    with np.load(parent_path,allow_pickle=False) as parents:
        for method in ('predicted','training_mean'):
            start=time.perf_counter()
            name=method+'_combined'
            try:
                result=analyze_texture(gray,parents[method],method='variation',auto_source='combined',levels=8,auto_correlation_quantile=.985)
                masks[name]=result['selected_mask']
                methods[name]=dict(seconds=time.perf_counter()-start,diagnostics=result['automatic_refinement'],error=None)
                del result
            except Exception as error:
                masks[name]=np.zeros(gray.shape,bool)
                methods[name]=dict(seconds=time.perf_counter()-start,error=repr(error))
    np.savez_compressed(destination,**masks)
    detail=dict(id=case['id'],signature=signature,prediction_sha256=digest(destination),methods=methods)
    metadata_path.write_text(json.dumps(detail,indent=2))
    return detail


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'clarius_annotations/liver_localization_20261117')
    p.add_argument('--reference-root',type=Path,default=ROOT/'clarius_annotations/verified_evaluation_20261117')
    p.add_argument('--workers',type=int,default=8)
    args=p.parse_args()
    args.output=args.output.resolve()
    (args.output/'refinements').mkdir(exist_ok=True)
    references=json.loads((args.reference_root/'reference_manifest.json').read_text())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(refine,case,str(args.output)) for case in references['cases']]
        for number,job in enumerate(as_completed(jobs),1):
            detail=job.result()
            print(f'Refined {number}/{len(jobs)} {detail["id"][:3]} errors={[m for m,v in detail["methods"].items() if v["error"]]}',flush=True)
    evaluation=json.loads((args.output/'parent_evaluation_manifest.json').read_text())
    for case in evaluation['cases']:
        destination=args.output/'refinements'/(case['id']+'.npz')
        for method in ('predicted_combined','training_mean_combined'):
            case['predictions'][method]=dict(path=str(destination),key=method)
    path=args.output/'evaluation_manifest.json'
    path.write_text(json.dumps(evaluation,indent=2))
    result=evaluate_manifest(path,args.output/'scores')
    print(json.dumps({m:{k:s[k]['mean'] for k in ('ssim','dice','iou')} for m,s in result['summary'].items()},indent=2),flush=True)


if __name__=='__main__':main()
