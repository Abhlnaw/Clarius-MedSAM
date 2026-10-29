"""Additional feature-specific thresholds; local validation outputs only."""
import json
import numpy as np
from threshold_experiment import ROOT, OUT, BASE, load_case, refine
import matplotlib.pyplot as plt

def main():
    image,parent,s,baseline,diagnostics=load_case(ROOT/'clarius_annotations/exam05_glcm_test')
    if s['features'].shape[-1] != 12:
        raise ValueError('This experiment requires a 12-channel Combined proposal')
    with np.load(OUT/'candidates.npz') as saved:reference=saved['reference']
    good=baseline&reference; spill=baseline&~reference
    results=[];masks=[]
    names=['half_mean','half_std','full_mean','full_std','half_contrast','half_homogeneity',
        'half_energy','half_correlation','full_contrast','full_homogeneity','full_energy','full_correlation']

    def evaluate(name,compatible):
        s['experimental_map']=compatible
        mask=refine(s,dict(BASE,gate='experimental_map'))
        tp=int((mask&reference).sum());fp=int((mask&~reference).sum());fn=int((~mask&reference).sum())
        rec=dict(index=len(results),name=name,pixels=int(mask.sum()),dice=2*tp/max(2*tp+fp+fn,1),
            keep=float((mask&good).sum()/good.sum()),remove=float((~mask&spill).sum()/spill.sum()))
        results.append(rec);masks.append(mask)

    features=s['native_features'];seed=s['seed']
    scaled=np.minimum(np.abs((features-s['center'])/s['scale']),3)**2
    families={'all':range(12),'glcm':range(4,12),'contrast':[4,8],'homogeneity':[5,9],
        'energy':[6,10],'correlation':[7,11],'local_std':[1,3]}
    for family,indices in families.items():
        distance=scaled[...,list(indices)].mean(-1)
        for quantile in [.7,.8,.9,.95,.98,.995]:
            threshold=float(np.quantile(distance[seed],quantile))
            evaluate(f'{family} distance seed q{quantile}',distance<=threshold)
        print('family',family,flush=True)
    for i,name in enumerate(names):
        for quantile in [.7,.8,.9,.95,.98,.995]:
            low,high=np.quantile(features[...,i][seed],[(1-quantile)/2,1-(1-quantile)/2])
            evaluate(f'{name} central seed {quantile:.1%}',(features[...,i]>=low)&(features[...,i]<=high))
        print('feature',name,flush=True)
    ranked=sorted(results,key=lambda r:r['dice'],reverse=True)
    (OUT/'feature_metrics.json').write_text(json.dumps({'results':results,'best':ranked[:12]},indent=2))
    choices=[{'index':-1,'name':'Current application','keep':1.,'remove':0.,'dice':0.91566}]
    for r in ranked:
        if all(x['index']==-1 or np.count_nonzero(masks[r['index']]^masks[x['index']])>2000 for x in choices):choices.append(r)
        if len(choices)==6:break
    fig,axes=plt.subplots(2,3,figsize=(16,8))
    for ax,r in zip(axes.ravel(),choices):
        mask=baseline if r['index']==-1 else masks[r['index']]
        ax.imshow(image,cmap='gray',aspect='auto',vmin=0,vmax=1)
        overlay=np.zeros((*image.shape,4));overlay[mask]=[1,.5,0,.45];ax.imshow(overlay,aspect='auto')
        ax.contour(reference,levels=[.5],colors=['red'],linewidths=.6)
        ax.set_title(f"{r['name']}\nkeep {r['keep']:.1%}; spill removed {r['remove']:.1%}",fontsize=10);ax.axis('off')
    fig.tight_layout();fig.savefig(OUT/'feature_comparison.png',dpi=150);plt.close(fig)
    np.savez_compressed(OUT/'feature_candidates.npz',**{f'mask_{r["index"]}':masks[r['index']] for r in choices if r['index']>=0})
    print(json.dumps({'top':ranked[:8], 'best_keep_98':next((r for r in ranked if r['keep']>=.98),None)}),flush=True)


if __name__ == '__main__':
    main()
