"""Refine a correlation threshold and compare fixed settings on other frames."""
import json
import numpy as np
from threshold_experiment import ROOT, OUT, BASE, load_case, refine
import matplotlib.pyplot as plt

def candidate(s,q):
    if s['features'].shape[-1] != 12:
        raise ValueError('This experiment requires a 12-channel Combined proposal')
    distance=np.minimum(np.abs((s['native_features'][...,[7,11]]-s['center'][[7,11]])/s['scale'][[7,11]]),3)**2
    distance=distance.mean(-1)
    threshold=float(np.quantile(distance[s['seed']],q))
    s['experimental_map']=distance<=threshold
    return refine(s,dict(BASE,gate='experimental_map')),threshold

def main():
    image,parent,s,baseline,_=load_case(ROOT/'clarius_annotations/exam05_glcm_test')
    with np.load(OUT/'candidates.npz') as z:reference=z['reference']
    good=baseline&reference;spill=baseline&~reference
    results=[]
    for q in np.linspace(.95,.999,50):
        mask,threshold=candidate(s,q)
        tp=int((mask&reference).sum());fp=int((mask&~reference).sum());fn=int((~mask&reference).sum())
        results.append(dict(q=float(q),threshold=threshold,pixels=int(mask.sum()),
            dice=2*tp/(2*tp+fp+fn),keep=float((mask&good).sum()/good.sum()),
            remove=float((~mask&spill).sum()/spill.sum())))
    best=max(results,key=lambda r:r['dice']);print('best',json.dumps(best),flush=True)
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot([r['q'] for r in results],[r['dice'] for r in results],label='Overlap with rough outline')
    ax.plot([r['q'] for r in results],[r['keep'] for r in results],label='Wanted baseline retained')
    ax.plot([r['q'] for r in results],[r['remove'] for r in results],label='Baseline spill removed')
    ax.set(xlabel='Seed correlation-distance quantile',ylabel='Fraction');ax.legend();fig.tight_layout()
    fig.savefig(OUT/'threshold_tradeoff.png',dpi=150);plt.close(fig)

    cases=[(ROOT/'clarius_annotations/exam05_glcm_test','frame_001'),
        (ROOT/'clarius_annotations/exam05_glcm_test','frame_002')]
    holdout=ROOT/'clarius_annotations/glcm_validation/automatic_evaluation/holdout'
    cases.extend((holdout,f'frame_{i:03d}') for i in [3,5,10])
    fig,axes=plt.subplots(len(cases),3,figsize=(14,4*len(cases)))
    cross=[]
    for row,(folder,frame) in enumerate(cases):
        image,parent,s,baseline,_=load_case(folder,frame)
        entry={'frame':frame,'baseline_pixels':int(baseline.sum())}
        for col,q in enumerate([None,.98,best['q']]):
            mask=baseline if q is None else candidate(s,q)[0]
            assert mask.any() and not mask[~parent].any()
            assert not mask[s['crop']][s['mean']<s['lower']].any()
            if q is not None:
                entry[str(q)]={'pixels':int(mask.sum()),'baseline_retained':float((mask&baseline).sum()/baseline.sum()),
                    'new_pixels':int((mask&~baseline).sum())}
            ax=axes[row,col];ax.imshow(image,cmap='gray',vmin=0,vmax=1,aspect='auto')
            overlay=np.zeros((*image.shape,4));overlay[mask]=[1,.5,0,.45];ax.imshow(overlay,aspect='auto')
            title='Current application' if q is None else f'Correlation q{q:.3f}'
            ax.set_title(f'{frame}: {title}');ax.axis('off')
            if row==0:ax.contour(reference,levels=[.5],colors=['red'],linewidths=.6)
        cross.append(entry);print(json.dumps(entry),flush=True)
    fig.tight_layout();fig.savefig(OUT/'cross_frame_comparison.png',dpi=110);plt.close(fig)
    (OUT/'crosscheck.json').write_text(json.dumps({'refined_sweep':results,'best_rough_outline':best,'cross_frames':cross},indent=2))


if __name__ == '__main__':
    main()
