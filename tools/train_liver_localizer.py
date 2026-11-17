"""Train exam-disjoint out-of-fold liver box models from verified masks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.ops import generalized_box_iou_loss
from clarius_localizer import INPUT_HW, LiverBoxNet, grouped_folds, mask_box, read_image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_iou(a,b):
    intersection = (torch.minimum(a[:,2:],b[:,2:])-torch.maximum(a[:,:2],b[:,:2])).clamp(min=0).prod(1)
    return intersection/((a[:,2:]-a[:,:2]).prod(1)+(b[:,2:]-b[:,:2]).prod(1)-intersection).clamp(min=1e-8)


def train(images, targets, training, validation, args, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = LiverBoxNet().to(args.device)
    mean_box = targets[training].mean(0)
    model.initialize_box(mean_box)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    history, best, best_epoch, state = [], -1., 0, None
    for epoch in range(1,args.epochs+1):
        model.train(); losses = []
        permutation = torch.randperm(len(training)).tolist()
        for start in range(0,len(training),16):
            indices = [training[i] for i in permutation[start:start+16]]
            x, y = images[indices].clone(), targets[indices].clone()
            flip = torch.rand(len(indices),device=args.device)<.5
            x[flip] = x[flip].flip(-1)
            old = y[flip].clone()
            y[flip,0], y[flip,2] = 1-old[:,2], 1-old[:,0]
            gain = torch.empty(len(indices),1,1,1,device=args.device).uniform_(.9,1.1)
            offset = torch.empty_like(gain).uniform_(-.03,.03)
            x = (x*gain+offset+torch.randn_like(x)*.01).clamp(0,1)
            predicted = model(x)
            loss = generalized_box_iou_loss(predicted,y,reduction='mean') + F.smooth_l1_loss(predicted,y,beta=.05)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            validation_iou = float(paired_iou(model(images[validation]),targets[validation]).mean()) if validation else None
        history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_box_iou=validation_iou))
        if not validation or validation_iou > best:
            best, best_epoch = validation_iou, epoch
            state = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
        if epoch%20 == 0:
            message = f'validation_box_iou={validation_iou:.4f} best={best:.4f}' if validation else f'train_loss={np.mean(losses):.4f}'
            print(f'  epoch={epoch} {message}',flush=True)
        if validation and epoch>=40 and epoch-best_epoch>=30:
            break
    model.load_state_dict(state)
    return model, state, history, best_epoch, best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'clarius_annotations/verified_evaluation_20261117/reference_manifest.json')
    parser.add_argument('--output',type=Path,default=ROOT/'clarius_annotations/liver_localization_20261117')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--epochs',type=int,default=160)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--fit-final-only',action='store_true',help='Fit all images for the median CV-selected epoch count; never use this model for CV scores')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    if (args.output/'training.json').exists() and not args.fit_final_only:
        parser.error('Training output already exists; choose a fresh --output folder to preserve the completed experiment')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True,exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    cases = manifest['cases']
    for case in cases:
        for key in ('clean','reference'):
            path = case['clean_image'] if key=='clean' else case['reference']
            if digest(path) != case[key+'_sha256']:
                raise ValueError('Input hash differs: '+path)
    images = torch.stack([read_image(c['clean_image']) for c in cases]).to(args.device)
    targets = torch.tensor(np.stack([mask_box(np.asarray(Image.open(c['reference']))>0) for c in cases]),device=args.device)
    if args.fit_final_only:
        training = json.loads((args.output/'training.json').read_text())
        if training['manifest_sha256'] != digest(args.manifest):
            raise ValueError('Final-fit inputs differ from the completed CV experiment')
        args.epochs = int(np.median([fold['best_epoch'] for fold in training['folds']]))
        _,state,history,_,_ = train(images,targets,list(range(len(cases))),[],args,args.seed+100)
        torch.save(dict(state_dict=state,input_hw=list(INPUT_HW),epochs=args.epochs,
            training_ids=[c['id'] for c in cases],padding_fraction=.03,seed=args.seed+100,
            purpose='Final all-data fit for new images only; not used in out-of-fold evaluation'),args.output/'liver_box_final.pth')
        (args.output/'final_fit.json').write_text(json.dumps(dict(epochs=args.epochs,history=history,
            checkpoint_sha256=digest(args.output/'liver_box_final.pth'),purpose='All-data fit, not a held-out evaluation'),indent=2))
        print(f'Final all-data checkpoint saved after {args.epochs} epochs',flush=True)
        return
    splits = grouped_folds(cases,5,args.seed)
    predictions, fold_records = [], []
    start = time.perf_counter()
    for fold,split in enumerate(splits):
        print(f'Fold {fold+1}/5 train={len(split["train"])} validation={len(split["validation"])} test={len(split["test"])}',flush=True)
        model,state,history,best_epoch,best = train(images,targets,split['train'],split['validation'],args,args.seed+fold)
        record = dict(fold=fold,ids={role:[cases[i]['id'] for i in indices] for role,indices in split.items()},
                      exams={role:sorted({cases[i]['exam'] for i in indices}) for role,indices in split.items()},
                      best_epoch=best_epoch,validation_box_iou=best,history=history)
        checkpoint = args.output/f'fold_{fold}.pth'
        torch.save(dict(state_dict=state,input_hw=list(INPUT_HW),fold=fold,training_ids=record['ids']['train'],
                        validation_ids=record['ids']['validation'],seed=args.seed+fold),checkpoint)
        model.eval()
        with torch.inference_mode():
            boxes = model(images[split['test']]).cpu().numpy()
        mean_box = targets[split['train']].mean(0).cpu().numpy()
        for index,box in zip(split['test'],boxes):
            predictions.append(dict(id=cases[index]['id'],exam=cases[index]['exam'],fold=fold,
                predicted_box=box.tolist(),training_mean_box=mean_box.tolist(),reference_box=targets[index].cpu().tolist(),
                checkpoint_sha256=digest(checkpoint)))
        fold_records.append(record)
        del model
    output = dict(protocol='Five-fold exam-disjoint outer test; inner exam-disjoint validation selects epoch. No outer-test optimization.',
        caveat='Exam grouping does not prove independent patient identities. This dataset was previously inspected during algorithm development.',
        manifest_sha256=digest(args.manifest),input_hw=list(INPUT_HW),parameters=sum(p.numel() for p in LiverBoxNet().parameters()),
        settings=dict(epochs=args.epochs,seed=args.seed,optimizer='AdamW',lr=.001,weight_decay=.001,batch_size=16,
                      loss='GIoU + SmoothL1(beta=.05)',dropout=.2,prompt_padding_fraction=.03),
        folds=fold_records,predictions=sorted(predictions,key=lambda r:r['id']),seconds=time.perf_counter()-start)
    (args.output/'training.json').write_text(json.dumps(output,indent=2))
    print(f'Training complete in {output["seconds"]:.1f}s; parameters={output["parameters"]}',flush=True)


if __name__=='__main__':
    main()
