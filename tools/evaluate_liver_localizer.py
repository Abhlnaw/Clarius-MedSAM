"""Compare exam-held-out predicted boxes with full, training-mean and oracle prompts."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from skimage import transform
from segment_anything import sam_model_registry
from MedSAM_Clarius import load_rgb
from clarius_localizer import pad_box
from tools.train_liver_localizer import digest
from tools.evaluate_segmentations import evaluate_manifest


def box_iou(a,b):
    a,b = np.asarray(a),np.asarray(b)
    intersection = np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
    return float(intersection/((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-intersection))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root',type=Path,default=ROOT/'clarius_annotations/verified_evaluation_20261117')
    parser.add_argument('--output',type=Path,default=ROOT/'clarius_annotations/liver_localization_20261117')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'medsam_vit_b.pth')
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    torch.set_num_threads(1)
    source=args.reference_root
    manifest_path=source/'reference_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    training=json.loads((args.output/'training.json').read_text())
    if digest(manifest_path)!=training['manifest_sha256']:
        raise ValueError('Reference manifest changed after training')
    records={r['id']:r for r in training['predictions']}
    checkpoint_sha=digest(args.checkpoint)
    model=None
    evaluation=dict(reference_provenance=manifest['reference_provenance'],reference_review_status=manifest['reference_review_status'],cases=[])
    localization=[]
    (args.output/'parents').mkdir(exist_ok=True)
    for number,case in enumerate(manifest['cases'],1):
        record=records[case['id']]
        fold=training['folds'][record['fold']]
        if case['id'] not in fold['ids']['test'] or case['exam'] in fold['exams']['train']+fold['exams']['validation']:
            raise ValueError('Outer test leakage')
        if digest(args.output/f'fold_{record["fold"]}.pth')!=record['checkpoint_sha256']:
            raise ValueError('Localizer checkpoint changed')
        if digest(case['clean_image'])!=case['clean_sha256'] or digest(case['reference'])!=case['reference_sha256']:
            raise ValueError('Image or reference changed')
        h,w=case['shape_hw']
        prompts={name:pad_box(record[key],.03) for name,key in
                 [('predicted','predicted_box'),('training_mean','training_mean_box'),('oracle','reference_box')]}
        signature=dict(clean_sha256=case['clean_sha256'],medsam_checkpoint_sha256=checkpoint_sha,
                       localizer_checkpoint_sha256=record['checkpoint_sha256'],prompts={k:v.tolist() for k,v in prompts.items()})
        destination=args.output/'parents'/(case['id']+'.npz')
        metadata_path=destination.with_suffix('.json')
        if destination.exists() and metadata_path.exists():
            metadata=json.loads(metadata_path.read_text())
            if metadata['signature']!=signature or metadata['prediction_sha256']!=digest(destination):
                raise ValueError('Cached prompt inference differs')
        else:
            if model is None:
                model=sam_model_registry['vit_b'](checkpoint=str(args.checkpoint)).to(args.device).eval()
            start=time.perf_counter()
            image=load_rgb(Path(case['clean_image']))
            resized=transform.resize(image,(1024,1024),order=3,preserve_range=True,anti_aliasing=True)
            resized=(resized-resized.min())/max(float(np.ptp(resized)),1e-8)
            tensor=torch.from_numpy(resized.astype(np.float32)).permute(2,0,1)[None].to(args.device)
            baseline_meta=json.loads((source/'parents'/(case['id']+'.json')).read_text())
            if (baseline_meta['signature']['clean_sha256']!=case['clean_sha256'] or
                baseline_meta['signature']['checkpoint_sha256']!=checkpoint_sha or
                baseline_meta['signature']['prompt_protocol']!='full-image'):
                raise ValueError('Full-image baseline does not match')
            with np.load(source/'predictions'/(case['id']+'.npz'),allow_pickle=False) as saved:
                masks=dict(full_image=saved['medsam'],full_image_combined=saved['combined_glcm'])
            with torch.inference_mode():
                embedding=model.image_encoder(tensor)
                for name,prompt in prompts.items():
                    box=torch.as_tensor(prompt*1024,dtype=torch.float32,device=args.device)[None,None]
                    sparse,dense=model.prompt_encoder(points=None,boxes=box,masks=None)
                    logits,_=model.mask_decoder(image_embeddings=embedding,image_pe=model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense,multimask_output=False)
                    probability=F.interpolate(torch.sigmoid(logits),(h,w),mode='bilinear',align_corners=False)
                    masks[name]=probability[0,0].cpu().numpy()>.5
            np.savez_compressed(destination,**masks)
            metadata=dict(signature=signature,prediction_sha256=digest(destination),seconds=time.perf_counter()-start)
            metadata_path.write_text(json.dumps(metadata,indent=2))
        reference=np.asarray(Image.open(case['reference']))>0
        for name,key in [('predicted','predicted_box'),('training_mean','training_mean_box')]:
            b=prompts[name]
            x1,y1,x2,y2=np.round(b*[w,h,w,h]).astype(int)
            localization.append(dict(case=case['id'],fold=record['fold'],method=name,
                raw_box_iou=box_iou(record[key],record['reference_box']),prompt_box_iou=box_iou(b,record['reference_box']),
                reference_pixel_coverage=float(reference[y1:y2,x1:x2].sum()/reference.sum())))
        evaluation['cases'].append(dict(id=case['id'],exam=case['exam'],
            reference=dict(path=case['reference'],label=255,sha256=case['reference_sha256']),
            predictions={name:dict(path=str(destination.resolve()),key=name) for name in
                         ('full_image','training_mean','predicted','oracle','full_image_combined')},
            evaluation_mask=dict(path=str(destination.resolve()),key='full_image')))
        print(f'Prompt inference {number}/{len(manifest["cases"])} {case["id"][:3]}',flush=True)
    path=args.output/'parent_evaluation_manifest.json'
    path.write_text(json.dumps(evaluation,indent=2))
    result=evaluate_manifest(path,args.output/'parent_scores')
    (args.output/'box_metrics.json').write_text(json.dumps(localization,indent=2))
    summary={m:{k:float(np.mean([r[k] for r in localization if r['method']==m])) for k in
               ('raw_box_iou','prompt_box_iou','reference_pixel_coverage')} for m in ('predicted','training_mean')}
    print(json.dumps(dict(localization=summary,segmentation={m:{k:s[k]['mean'] for k in ('ssim','dice','iou')} for m,s in result['summary'].items()}),indent=2),flush=True)


if __name__=='__main__':main()
