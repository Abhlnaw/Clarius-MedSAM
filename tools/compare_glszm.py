"""Compare local GLSZM with the q0.985 Combined/GLCM baseline on saved parents.

Run from the repo root. No model checkpoint or CUDA inference is required:
every method receives the same saved MedSAM parent mask. Generated data stay
under the ignored clarius_annotations directory by default.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from scipy import ndimage as ndi

from clarius_auto import automatic_tissue_region
from clarius_glcm import local_glcm_features
from clarius_glszm import local_glszm_features
from clarius_texture import analyze_texture, localized_window


def image_gray(path):
    image=io.imread(path)
    if image.ndim==2:image=np.repeat(image[...,None],3,axis=-1)
    else:image=image[...,:3]
    image=image.astype(np.float32)
    image=(image-image.min())/max(float(np.ptp(image)),1e-8)
    return image.mean(-1)


def load_case(folder,frame):
    metadata=json.loads((folder/f'{frame}.png.regions.json').read_text())
    with np.load(folder/f'{frame}.png.regions.npz',allow_pickle=False) as saved:
        parent=saved['masks'][0].astype(bool)
    return image_gray(metadata['source']),parent


def reference_metrics(mask,reference,baseline):
    tp=int((mask&reference).sum());fp=int((mask&~reference).sum());fn=int((~mask&reference).sum())
    good=baseline&reference;spill=baseline&~reference
    return dict(dice=2*tp/max(2*tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),
        outside_reference_pixels=fp,missed_reference_pixels=fn,
        wanted_baseline_retained=int((mask&good).sum())/max(int(good.sum()),1),
        baseline_spill_removed=int((~mask&spill).sum())/max(int(spill.sum()),1))


def plot_case(output,name,gray,parent,proposals,masks,reference):
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    panels=[('Original / same MedSAM parent',parent),('Current Combined + GLCM q0.985',masks['combined']),
        ('GLSZM q0.985',masks['glszm']),('Brightness + GLSZM q0.985',masks['combined_glszm'])]
    for ax,(title,mask) in zip(axes.ravel(),panels):
        ax.imshow(gray,cmap='gray',vmin=0,vmax=1,aspect='auto')
        if mask is parent:ax.contour(parent,levels=[.5],colors=['cyan'],linewidths=.7)
        else:
            rgba=np.zeros((*gray.shape,4));rgba[mask]=[1,.5,0,.4];ax.imshow(rgba,aspect='auto')
        if reference is not None:ax.contour(reference,levels=[.5],colors=['red'],linewidths=.7)
        ax.set_title(title);ax.axis('off')
    fig.suptitle(name+' — reference outline, where shown, is approximate',fontsize=11)
    fig.tight_layout();fig.savefig(output/f'{name}.comparison.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,(method,source) in zip(axes,proposals.items()):
        ax.imshow(gray,cmap='gray',aspect='auto',vmin=0,vmax=1)
        rgba=np.zeros((*gray.shape,4))
        for group in np.unique(source['labels']):
            if group:
                rgba[source['labels']==group]=plt.get_cmap('tab10')(int(group)-1)
                rgba[source['labels']==group,3]=.4
        ax.imshow(rgba,aspect='auto');ax.set_title(method+' source groups');ax.axis('off')
    fig.tight_layout();fig.savefig(output/f'{name}.proposals.png',dpi=110);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'clarius_annotations/glszm_comparison')
    parser.add_argument('--step',type=int,nargs=2,default=(8,2))
    parser.add_argument('--levels',type=int,default=8,choices=[8,16,32,64])
    parser.add_argument('--case',nargs=3,action='append',metavar=('EXAM','ANNOTATIONS','FRAME'),
        help='Repeat to override default saved exam04/05 cases')
    parser.add_argument('--reference',type=Path,default=ROOT/'clarius_annotations/glcm_validation/threshold_experiment/candidates.npz')
    parser.add_argument('--reference-case',default='exam05_frame_001',help='EXAM_FRAME identifier matching the optional reference')
    parser.add_argument('--skip-sensitivity',action='store_true')
    parser.add_argument('--quantiles',type=float,nargs='+',default=[.90,.95,.98,.985,.99,.995,1.],
        help='Descriptor quantiles checked on cases with an outline; never selected as new defaults')
    args=parser.parse_args()
    if any(n<1 for n in args.step):parser.error('Step must be positive')
    if any(not np.isfinite(q) or not 0<q<=1 for q in args.quantiles):parser.error('Quantiles must be in (0,1]')
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    if args.case:cases=[(exam,Path(folder),frame) for exam,folder,frame in args.case]
    else:
        cases=[('exam05',ROOT/'clarius_annotations/exam05_glcm_test',f'frame_{i:03d}') for i in [1,2]]
        cases += [('exam05',ROOT/'clarius_annotations/glcm_validation/automatic_evaluation/holdout',f'frame_{i:03d}') for i in [3,5,10]]
        cases += [('exam04',ROOT/'clarius_annotations/exam04_glcm_test',f'frame_{i:03d}') for i in [1,2]]
    reference=None
    if args.reference.is_file():
        with np.load(args.reference,allow_pickle=False) as saved:reference=saved['reference'].astype(bool)
    records=[];sensitivity={};quantile_records=[]
    for exam,folder,frame in cases:
        name=exam+'_'+frame;print('START',name,flush=True)
        gray,parent=load_case(folder,frame);window=localized_window(parent)
        sources={};masks={};entries={}
        start=time.perf_counter();glcm=local_glcm_features(gray,parent,window,levels=args.levels)
        glcm_seconds=time.perf_counter()-start
        start=time.perf_counter();glszm=local_glszm_features(gray,parent,window,levels=args.levels,step=tuple(args.step))
        glszm_seconds=time.perf_counter()-start
        for method in ['combined','glszm','combined_glszm']:
            start=time.perf_counter()
            source=analyze_texture(gray,parent,window=window,levels=args.levels,method=method,
                glcm_data=glcm if method=='combined' else None,
                glszm_data=glszm if method!='combined' else None,glszm_step=tuple(args.step))
            proposal_seconds=time.perf_counter()-start
            start=time.perf_counter();mask,diag=automatic_tissue_region(gray,parent,source)
            refinement_seconds=time.perf_counter()-start
            assert mask.any() and not mask[~parent].any()
            components,count=ndi.label(mask,structure=np.ones((3,3)));assert count==1
            sources[method]=source;masks[method]=mask
            entries[method]=dict(selected_pixels=int(mask.sum()),parent_fraction=float(mask.sum()/parent.sum()),
                map_seconds=glcm_seconds if method=='combined' else glszm_seconds,
                proposal_seconds=proposal_seconds,refinement_seconds=refinement_seconds,diagnostics=diag)
            entries[method]['total_seconds']=sum(entries[method][key] for key in ['map_seconds','proposal_seconds','refinement_seconds'])
        baseline=masks['combined']
        # Compare everyone against the baseline's SAME dark exclusion, avoiding
        # claiming that changing a method's own brightness model preserves it.
        held={}
        def capture(f,e,a):
            if f.f_code is automatic_tissue_region.__code__ and e=='return':held.update(f.f_locals)
        previous=sys.getprofile();sys.setprofile(capture)
        try:automatic_tissue_region(gray,parent,sources['combined'])
        finally:sys.setprofile(previous)
        dark=np.zeros(parent.shape,bool);dark[held['crop']]=(held['mean']<held['lower'])&parent[held['crop']]
        this_reference=reference if name==args.reference_case else None
        if this_reference is not None and this_reference.shape!=parent.shape:raise ValueError('Reference shape mismatch')
        for method,mask in masks.items():
            entries[method].update(dict(jaccard_to_current=float((mask&baseline).sum()/max((mask|baseline).sum(),1)),
                baseline_pixels_removed=int((baseline&~mask).sum()),new_pixels_added=int((mask&~baseline).sum()),
                baseline_dark_pixels_selected=int((mask&dark).sum())))
            if this_reference is not None:entries[method]['rough_reference']=reference_metrics(mask,this_reference,baseline)
        record=dict(case=name,shape_hw=list(parent.shape),parent_pixels=int(parent.sum()),window_hw=list(window),methods=entries)
        records.append(record)
        print(json.dumps({'case':name,'methods':{method:{key:data[key] for key in
            ['selected_pixels','total_seconds','jaccard_to_current','baseline_dark_pixels_selected']}
            for method,data in entries.items()}}),flush=True)
        plot_case(output,name,gray,parent,sources,masks,this_reference)
        np.savez_compressed(output/f'{name}.masks.npz',parent=parent,**masks)
        if this_reference is not None:
            for method in ['glszm','combined_glszm']:
                for q in args.quantiles:
                    mask,_=automatic_tissue_region(gray,parent,sources[method],correlation_quantile=q)
                    quantile_records.append(dict(case=name,method=method,quantile=q,
                        selected_pixels=int(mask.sum()),rough_reference=reference_metrics(mask,this_reference,baseline)))
            (output/'quantile_sensitivity.json').write_text(json.dumps(quantile_records,indent=2))
        if exam=='exam05' and frame=='frame_001' and not args.skip_sensitivity:
            # Sampling approximation is independently measured on a native-grid
            # patch. It is not validated solely by looking at pretty overlays.
            crop=(slice(300,428),slice(48,80));patch_gray=gray[crop];patch_mask=parent[crop]
            patch_window=(65,11)
            start=time.perf_counter();dense,support=local_glszm_features(patch_gray,patch_mask,patch_window,levels=args.levels,step=(1,1))
            dense_seconds=time.perf_counter()-start
            start=time.perf_counter();sparse,_=local_glszm_features(patch_gray,patch_mask,patch_window,levels=args.levels,step=tuple(args.step))
            sparse_seconds=time.perf_counter()-start
            scale=np.quantile(dense[support],.95,axis=0)-np.quantile(dense[support],.05,axis=0)
            relative=np.sqrt(np.mean((sparse[support]-dense[support])**2,axis=0))/np.maximum(scale,1e-6)
            sensitivity=dict(patch_shape=list(patch_gray.shape),patch_window=list(patch_window),
                dense_seconds=dense_seconds,sparse_seconds=sparse_seconds,
                rmse_over_dense_5_to_95_range={name:float(value) for name,value in zip(sources['glszm']['feature_names'],relative)})
            fine_step=(max(1,args.step[0]//2),max(1,args.step[1]//2))
            start=time.perf_counter();fine_maps=local_glszm_features(gray,parent,window,levels=args.levels,step=fine_step)
            sensitivity['fine_map_seconds']=time.perf_counter()-start
            sensitivity['fine_step_hw']=list(fine_step)
            sensitivity['fine_methods']={}
            for method in ['glszm','combined_glszm']:
                source=analyze_texture(gray,parent,window=window,levels=args.levels,method=method,glszm_data=fine_maps,glszm_step=fine_step)
                mask,diag=automatic_tissue_region(gray,parent,source)
                sensitivity['fine_methods'][method]=dict(selected_pixels=int(mask.sum()),
                    jaccard_to_default_step=float((mask&masks[method]).sum()/max((mask|masks[method]).sum(),1)),
                    rough_reference=reference_metrics(mask,this_reference,baseline) if this_reference is not None else None)
            print('SENSITIVITY',json.dumps(sensitivity),flush=True)
        (output/'metrics.json').write_text(json.dumps(dict(levels=args.levels,step_hw=list(args.step),
            quantile=.985,reference_note='Approximate screenshot outline on exam05 frame001 only; not expert ground truth.',
            records=records,sensitivity=sensitivity,quantile_sensitivity=quantile_records),indent=2))
    lines=['# GLSZM comparison','',
        'Same saved parent masks, eight-level quantization (unless overridden), two window scales, q0.985.',
        'Runtime includes fresh texture maps, clustering, and automatic selection; no GPU encoding or image I/O.',
        'Reference metrics apply only to the approximate red outline on exam05 frame 001.','',
        '| Case | Method | Selected pixels | IoU to current | Total seconds | Rough-outline Dice |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for record in records:
        for method,data in record['methods'].items():
            dice=f"{data['rough_reference']['dice']:.4f}" if 'rough_reference' in data else '—'
            lines.append(f"| {record['case']} | {method} | {data['selected_pixels']} | {data['jaccard_to_current']:.3f} | {data['total_seconds']:.2f} | {dice} |")
    lines += ['', 'IoU measures similarity to the current method, not anatomical accuracy.',
        'Each method learns its own seed and brightness bounds. The JSON also counts selections',
        'below the CURRENT method\'s dark cutoff, which is only a consistency check, not a vessel annotation.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
