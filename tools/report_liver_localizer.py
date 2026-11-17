"""Package localization results, paired comparisons and native-image review."""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import statistics
from urllib.parse import quote

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT=Path(__file__).resolve().parents[1]
LABELS=dict(full_image='Full-image MedSAM',training_mean='Training-mean box + MedSAM',
    predicted='CNN box + MedSAM',oracle='Reference box + MedSAM (assisted)',
    full_image_combined='Full-image + Combined GLCM',
    training_mean_combined='Mean box + MedSAM + Combined GLCM',
    predicted_combined='CNN box + MedSAM + Combined GLCM')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'clarius_annotations/liver_localization_20261117')
    parser.add_argument('--reference-root',type=Path,default=ROOT/'clarius_annotations/verified_evaluation_20261117')
    args=parser.parse_args();root=args.output.resolve()
    references=json.loads((args.reference_root/'reference_manifest.json').read_text())
    training=json.loads((root/'training.json').read_text())
    results=json.loads((root/'scores/metrics.json').read_text())
    manifest=json.loads((root/'evaluation_manifest.json').read_text())
    boxes=json.loads((root/'box_metrics.json').read_text())
    rows={(r['case'],r['method']):r for r in results['rows']}
    expected={(c['id'],m) for c in references['cases'] for m in LABELS}
    if set(rows)!=expected or len(rows)!=len(results['rows']):raise ValueError('Missing or duplicate score rows')
    summaries=[]
    for method,label in LABELS.items():
        summary=results['summary'][method]
        summaries.append(dict(method=method,label=label,n=summary['n_images'],
            **{key:summary[key]['mean'] for key in ('ssim','roi_mean_ssim','dice','iou','precision','recall')},
            zero_overlap_images=sum(row['dice']==0 for row in results['rows'] if row['method']==method)))
    paired=[]
    for a,b in [('predicted','full_image'),('predicted','training_mean'),
                ('predicted_combined','full_image_combined'),('predicted_combined','training_mean_combined'),
                ('predicted_combined','predicted')]:
        differences=[rows[(c['id'],a)]['dice']-rows[(c['id'],b)]['dice'] for c in references['cases']]
        paired.append(dict(method=a,baseline=b,mean_dice_difference=statistics.mean(differences),
            improved_images=sum(d>1e-8 for d in differences),worse_images=sum(d< -1e-8 for d in differences),
            tied_images=sum(abs(d)<=1e-8 for d in differences)))
    box_summary={method:{key:statistics.mean(row[key] for row in boxes if row['method']==method)
        for key in ('raw_box_iou','prompt_box_iou','reference_pixel_coverage')} for method in ('predicted','training_mean')}
    # Validate original inputs and fold separation again after all computations.
    for case in references['cases']:
        if digest(case['clean_image'])!=case['clean_sha256'] or digest(case['reference'])!=case['reference_sha256'] or digest(case['annotated_image'])!=case['annotated_sha256']:
            raise ValueError('Evaluation input changed')
    for fold in training['folds']:
        sets=[set(fold['exams'][role]) for role in ('train','validation','test')]
        if sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]:raise ValueError('Exam leakage')
    failures={name:0 for name in ('predicted_combined','training_mean_combined')}
    for case in references['cases']:
        record=json.loads((root/'refinements'/(case['id']+'.json')).read_text())
        for name in failures:failures[name]+=int(record['methods'][name]['error'] is not None)
        if record['prediction_sha256']!=digest(root/'refinements'/(case['id']+'.npz')):raise ValueError('Refinement changed')
        parent=json.loads((root/'parents'/(case['id']+'.json')).read_text())
        if parent['prediction_sha256']!=digest(root/'parents'/(case['id']+'.npz')):raise ValueError('Parent changed')
    grouped=[]
    for exam in sorted({c['exam'] for c in references['cases']}):
        cases=[c for c in references['cases'] if c['exam']==exam]
        for method in LABELS:
            grouped.append(dict(exam=exam,method=method,n_images=len(cases),
                **{key:statistics.mean(rows[(c['id'],method)][key] for c in cases) for key in ('ssim','dice','iou')}))
    for filename,data in [('summary.csv',summaries),('paired_dice.csv',paired),('per_exam.csv',grouped)]:
        with (root/'scores'/filename).open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    final_fit=json.loads((root/'final_fit.json').read_text())
    lines=['# Liver-localization experiment','',
        f'{len(references["cases"])} images, {len({c["exam"] for c in references["cases"]})} examination IDs, five outer folds. Every CNN-scored image comes from an exam excluded from both training and inner validation for that model.',
        'Inner validation selects the epoch using box IoU only. The training-mean box uses the exact same training cases. Padding is fixed at 0.03 of image width/height on every side.',
        'References: supplied PhD-student-verified liver contours. Original 006 and 045 exclusions and authorized 021 closure are unchanged.',
        'Inputs remain the clean MATLAB env_gamma08 images. Only CNN input resizing/normalization happens in Python. Frozen MedSAM uses the existing 1024 encoder preparation.', '',
        '| Prompt / method | SSIM | Dice | IoU | Zero overlap |',
        '| --- | ---: | ---: | ---: | ---: |']
    for row in summaries:lines.append(f'| {row["label"]} | {row["ssim"]:.4f} | {row["dice"]:.4f} | {row["iou"]:.4f} | {row["zero_overlap_images"]} |')
    lines+=['','## Localization','', '| Method | Raw box IoU | Padded prompt IoU | Reference-pixel coverage |','| --- | ---: | ---: | ---: |']
    for method,values in box_summary.items():lines.append(f'| {method} | {values["raw_box_iou"]:.4f} | {values["prompt_box_iou"]:.4f} | {values["reference_pixel_coverage"]:.4f} |')
    lines+=['','## Paired changes','', '| Method versus baseline | Mean Dice change | Improved | Worse | Tied |','| --- | ---: | ---: | ---: | ---: |']
    for row in paired:lines.append(f'| {row["method"]} vs {row["baseline"]} | {row["mean_dice_difference"]:+.4f} | {row["improved_images"]} | {row["worse_images"]} | {row["tied_images"]} |')
    lines+=['','## Model and settings','',
        f'The CNN has {training["parameters"]:,} parameters and a 128 by 224 grayscale input. Four stride-two convolution blocks use 16, 32, 64 and 96 channels, GroupNorm and SiLU. Fixed average pooling and a 128-unit head output four ordered box coordinates.',
        'Training starts from scratch, initialized to the training mean box. AdamW learning rate 0.001, weight decay 0.001, batch 16, dropout 0.2. Loss: GIoU plus SmoothL1 with beta 0.05. Maximum 160 epochs; early stopping after 30 epochs without validation improvement, with at least 40 epochs run. Seed 2026, plus fold index.',
        'Training augmentation: horizontal flips, gain 0.9–1.1, offset -0.03–0.03 and Gaussian noise SD 0.01. Labels transform with flips. No augmentation or label-derived prompt is applied to held-out CNN test images.',
        'Texture refinement is unchanged: Combined brightness + GLCM, 8 levels, q0.985, existing automatic windows and spatial cleanup. Exceptions, if any, remain empty masks in scoring.',
        f'A separate all-data model uses the median fold-selected epoch count ({final_fit["epochs"]}) for trying new images. It is never used for these out-of-fold scores.', '',
        '## Interpretation limits','',
        'The oracle/reference-box row uses held-out reference information and is an assisted diagnostic comparison, not automatic performance.',
        'All metrics are per-image means. The same acquisition can have multiple captures. Exam grouping does not establish patient independence. This is an exploratory cross-validation experiment on a previously inspected dataset, not a pristine external test.',
        'The model always predicts one liver box because all training examples contain liver. It has no liver-absence classifier or calibrated confidence score. New probes, acquisition settings and liver-free views need separate testing.',
        'Foreground references exclude visible lumens. Good outer localization alone does not establish accurate lumen exclusion. Full-frame SSIM includes shared background; Dice/IoU and visual review remain necessary.', '',
        '## Files','',
        '- `review.html`: all image/prompt/mask comparisons, verified outline in green, prediction in orange and prompt in blue.',
        '- `scores/`: full scores, aggregate summary, paired changes and examination summaries.',
        '- `training.json`: fold assignments, learning histories and out-of-fold boxes.',
        '- `fold_*.pth`: evaluated fold models; `liver_box_final.pth`: separate all-data model.',
        '- `parents/`, `refinements/`: saved native masks, settings, hashes and diagnostics.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    aggregate=dict(n_images=len(references['cases']),n_exams=len({c['exam'] for c in references['cases']}),n_scores=len(rows),summary=summaries,box_summary=box_summary,paired=paired,
        refinement_exceptions=failures,model_parameters=training['parameters'],training_seconds=training['seconds'],
        best_epochs=[fold['best_epoch'] for fold in training['folds']],exam_disjoint=True,inputs_unchanged=True)
    (root/'summary.json').write_text(json.dumps(aggregate,indent=2))
    (root/'visual_masks').mkdir(exist_ok=True)
    ui=[];lookup={r['id']:r for r in training['predictions']}
    for case,spec in zip(references['cases'],manifest['cases']):
        if case['id']!=spec['id']:raise ValueError('Review order mismatch')
        reference=np.asarray(Image.open(case['reference']))>0
        rgba=np.zeros((*reference.shape,4),np.uint8)
        rgba[ndi.binary_dilation(reference ^ ndi.binary_erosion(reference))]=[30,255,80,255]
        reference_file=case['id']+'.reference.png';Image.fromarray(rgba).save(root/'visual_masks'/reference_file)
        item=dict(id=case['id'],image=quote(os.path.relpath(case['clean_image'],root).replace('\\','/'),safe='/'),
                  reference='visual_masks/'+quote(reference_file),methods={})
        arrays={}
        for name,prediction in spec['predictions'].items():
            if prediction['path'] not in arrays:
                with np.load(prediction['path'],allow_pickle=False) as saved:
                    arrays[prediction['path']]={key:saved[key] for key in saved.files}
            mask=arrays[prediction['path']][prediction['key']].astype(bool)
            if mask.shape!=reference.shape:raise ValueError('Review dimensions differ')
            rgba=np.zeros((*mask.shape,4),np.uint8);rgba[mask]=[255,140,0,100]
            filename=case['id']+'.'+name+'.png';Image.fromarray(rgba).save(root/'visual_masks'/filename)
            key='predicted_box' if name.startswith('predicted') else 'training_mean_box' if name.startswith('training_mean') else 'reference_box' if name=='oracle' else None
            box=np.clip(np.array(lookup[case['id']][key])+[-.03,-.03,.03,.03],0,1).tolist() if key else [0,0,1,1]
            item['methods'][name]=dict(image='visual_masks/'+quote(filename),box=box,**rows[(case['id'],name)])
        ui.append(item)
    options=''.join(f'<option value="{i}">{html.escape(c["id"])}</option>' for i,c in enumerate(ui))
    method_options=''.join(f'<option value="{key}">{html.escape(label)}</option>' for key,label in LABELS.items())
    page='''<!doctype html><meta charset="utf-8"><title>Liver localization review</title><style>body{font:16px Arial;background:#181818;color:#eee;margin:18px}nav{display:flex;gap:8px;flex-wrap:wrap}button,select{font:inherit;padding:7px}#case{max-width:75vw}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.frame{position:relative}.frame img{display:block;width:100%;height:auto}.layer{position:absolute;inset:0}.box{position:absolute;border:2px solid #35bfff;box-sizing:border-box;pointer-events:none}a{color:#8bcfff}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style><h2>Exam-held-out liver localization</h2><p>Green: verified reference. Orange: predicted mask. Blue: prompt box. Reference-box results use the reference and are assisted.</p><nav><button onclick="step(-1)">Previous</button><select id="case" onchange="show()">CASES</select><button onclick="step(1)">Next</button></nav><div class="grid"><section><p><select id="left" onchange="show()">METHODS</select></p><div class="frame"><img id="leftBase"><img class="layer" id="leftMask"><img class="layer" id="leftReference"><div class="box" id="leftBox"></div></div><p id="leftScore"></p></section><section><p><select id="right" onchange="show()">METHODS</select></p><div class="frame"><img id="rightBase"><img class="layer" id="rightMask"><img class="layer" id="rightReference"><div class="box" id="rightBox"></div></div><p id="rightScore"></p></section></div><p><a href="REPORT.md">Report</a> · <a href="scores/per_image.csv">All scores</a></p><script>const cases=DATA;const choice=document.getElementById('case');document.getElementById('left').value='full_image';document.getElementById('right').value='predicted';function show(){const c=cases[+choice.value];for(const s of ['left','right']){const m=c.methods[document.getElementById(s).value];document.getElementById(s+'Base').src=c.image;document.getElementById(s+'Mask').src=m.image;document.getElementById(s+'Reference').src=c.reference;const b=m.box;Object.assign(document.getElementById(s+'Box').style,{left:b[0]*100+'%',top:b[1]*100+'%',width:(b[2]-b[0])*100+'%',height:(b[3]-b[1])*100+'%'});document.getElementById(s+'Score').textContent='Dice '+m.dice.toFixed(4)+'  IoU '+m.iou.toFixed(4)+'  SSIM '+m.ssim.toFixed(4);}}function step(n){choice.value=Math.max(0,Math.min(cases.length-1,+choice.value+n));show();}document.addEventListener('keydown',e=>{if(e.target.tagName==='SELECT')return;if(e.key==='ArrowLeft')step(-1);if(e.key==='ArrowRight')step(1);});show();</script>'''
    page=page.replace('CASES',options).replace('METHODS',method_options).replace('DATA',json.dumps(ui).replace('</','<\\/'))
    (root/'review.html').write_text(page,encoding='utf-8')
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
