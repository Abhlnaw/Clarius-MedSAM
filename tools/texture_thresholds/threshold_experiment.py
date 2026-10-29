"""Local experiments only; production modules and saved annotations stay unchanged."""
import sys
import json
import itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy import ndimage as ndi
from skimage import io, transform, draw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from clarius_auto import automatic_tissue_region, _disk, _largest

OUT = ROOT / 'clarius_annotations' / 'glcm_validation' / 'threshold_experiment'
OUT.mkdir(parents=True, exist_ok=True)


def load_case(folder, frame='frame_001'):
    meta_path = folder / (frame + '.png.regions.json')
    meta = json.loads(meta_path.read_text())
    image = io.imread(meta['source']).astype(np.float32)
    if image.ndim == 3:
        image = image[..., :3].mean(-1)
    image = (image - image.min()) / max(float(np.ptp(image)), 1e-8)
    with np.load(meta_path.with_suffix('.npz'), allow_pickle=False) as saved:
        parent = saved['masks'][0].astype(bool)
        if 'texture_0_source_labels' in saved:
            proposal = {'labels': saved['texture_0_source_labels'], 'features': saved['texture_0_features']}
        else:
            from clarius_texture import analyze_texture
            proposal = analyze_texture(image, parent, method='combined', levels=8)
        saved_mask = saved['texture_0_selected_mask'].astype(bool) if 'texture_0_source_labels' in saved else None
    state = {}
    def capture(frame, event, arg):
        if frame.f_code is automatic_tissue_region.__code__ and event == 'return':
            state.update(frame.f_locals)
    previous_profiler = sys.getprofile()
    sys.setprofile(capture)
    try:
        baseline, diagnostics = automatic_tissue_region(image, parent, proposal)
    finally:
        sys.setprofile(previous_profiler)
    if saved_mask is not None:
        assert np.array_equal(baseline, saved_mask), 'Baseline differs from saved annotation'
    assert np.array_equal(refine(state, BASE), baseline), 'Experimental reconstruction is not exact'
    return image, parent, state, baseline, diagnostics


def refine(s, cfg):
    lower = s['median'] - max(cfg['low'] * s['mad'], .05)
    upper = s['median'] + max(cfg['high'] * s['mad'], cfg.get('high_floor', .08))
    feature_ok = s['distance'] <= s['threshold'] * cfg['distance']
    same = s['labels'][s['crop']] == s['group']
    speckle = ((s['local_std'] >= s['std_low'] * cfg['std_low']) &
               (s['local_std'] <= s['std_high'] * cfg['std_high'] + .005))
    if cfg['gate'] == 'or':
        compatible = feature_ok | same | speckle
    elif cfg['gate'] == 'required':
        compatible = feature_ok
    elif cfg['gate'] == 'guarded':
        compatible = feature_ok & (same | speckle)
    elif cfg['gate'] == 'speckle_required':
        compatible = (feature_ok | same) & speckle
    elif cfg['gate'] == 'experimental_map':
        compatible = s['experimental_map'].copy()
    else:
        raise ValueError(f"Unknown experimental gate: {cfg['gate']}")
    compatible &= (s['mean'] >= lower) & (s['mean'] <= upper) & s['parent'][s['crop']]
    def small(array):
        return transform.resize(array.astype(float), s['shape'], order=0,
            preserve_range=True, anti_aliasing=False) > .5
    small_compatible = small(compatible)
    scaffold = small_compatible | (ndi.binary_fill_holes(small_compatible) & small(s['mean'] < lower))
    opened = ndi.binary_opening(scaffold & s['small_parent'], structure=s['structure'])
    components, count = ndi.label(opened, structure=np.ones((3, 3)))
    # The overlap anchor must be the INITIAL source seed, not the expanded body.
    body = small(s['native_body'])
    if count:
        overlap = np.bincount(components[body], minlength=count + 1)
        overlap[0] = 0
        if overlap.max() > 0:
            body = components == int(overlap.argmax())
    envelope = ndi.binary_dilation(body, structure=_disk(5)) & s['small_parent']
    result = envelope & small_compatible
    result = ndi.binary_closing(np.pad(result, 3, mode='edge'), structure=_disk(3))[3:-3, 3:-3] & envelope
    result = _largest(result)
    native = transform.resize(result.astype(float), s['parent'][s['crop']].shape,
        order=0, preserve_range=True, anti_aliasing=False) > .5
    native &= s['parent'][s['crop']] & (s['mean'] >= lower)
    mask = np.zeros(s['parent'].shape, bool)
    mask[s['crop']] = _largest(native)
    return mask


BASE = dict(gate='or', low=2.5, high=3., distance=1., std_low=.7, std_high=1.4)
if __name__ == '__main__':
    image, parent, state, baseline, diagnostics = load_case(ROOT/'clarius_annotations/exam05_glcm_test')
    assert np.array_equal(refine(state, BASE), baseline), 'Experimental reconstruction is not exact'
    # Approximate tracing of the USER'S red outline, used for evaluation only.
    # No coordinates from this reference are used by the segmentation algorithm.
    points = np.array([(58,274),(65,252),(103,195),(147,177),(273,139),(500,125),
        (653,140),(800,141),(910,154),(1232,155),(1284,161),(1340,185),(1397,198),
        (1366,266),(1344,292),(1284,309),(1190,345),(1137,356),(1024,356),
        (957,371),(914,391),(894,439),(839,475),(744,500),(666,512),(528,515),
        (461,505),(408,490),(345,484),(261,466),(175,447),(139,427),
        (112,391),(84,342),(61,296)], float)
    rows, cols = draw.polygon(points[:,1]/853*image.shape[0], points[:,0]/1418*image.shape[1], image.shape)
    reference = np.zeros(image.shape, bool); reference[rows,cols] = True
    # Exclude the already protected dark cavity from the rough reference.
    protected = np.zeros(image.shape, bool)
    protected[state['crop']] = state['mean'] < state['lower']
    reference &= parent & ~protected
    baseline_good = baseline & reference
    baseline_spill = baseline & ~reference
    configs = [BASE]
    # Test a tighter feature threshold alone, then make the feature gate binding.
    for gate, factor, lo, hi in itertools.product(
        ['or','required','guarded','speckle_required'], [.35,.5,.65,.8,1.], [.7,.9,1.], [1.,1.2,1.4]):
        configs.append(dict(BASE, gate=gate, distance=factor, std_low=lo, std_high=hi))
    # Separate intensity narrowing from the texture-only experiment.
    for low, high, floor in itertools.product([1.,1.5,2.,2.5], [1.,1.5,2.,3.], [.04,.08]):
        configs.append(dict(BASE, low=low, high=high, high_floor=floor))
    records=[]; masks=[]
    for i,cfg in enumerate(configs):
        mask = refine(state,cfg)
        tp=int((mask & reference).sum()); fp=int((mask & ~reference).sum()); fn=int((~mask & reference).sum())
        rec=dict(index=i, config=cfg, pixels=int(mask.sum()), rough_reference_dice=2*tp/max(2*tp+fp+fn,1),
            desired_baseline_retained=float((mask & baseline_good).sum()/baseline_good.sum()),
            unwanted_baseline_removed=float((~mask & baseline_spill).sum()/baseline_spill.sum()),
            reference_recall=tp/int(reference.sum()))
        records.append(rec); masks.append(mask)
        if i%40==0: print('sweep',i,'/',len(configs),flush=True)
    ranked=sorted(records,key=lambda r:r['rough_reference_dice'], reverse=True)
    (OUT/'metrics.json').write_text(json.dumps(dict(note='Rough screenshot outline, not an expert annotation; ranking only.',
        baseline_diagnostics=diagnostics, results=records, best=ranked[:12]),indent=2))
    selected=[records[0],next(r for r in records if r['config']==dict(BASE,distance=.35)),ranked[0]]
    texture_only=[r for r in ranked if r['config']['low']==2.5 and r['config']['high']==3.]
    if texture_only[0]['index'] not in [r['index'] for r in selected]:selected.append(texture_only[0])
    for r in ranked:
        if len(selected)>=6:break
        if all(abs(r['pixels']-x['pixels'])>1500 for x in selected):selected.append(r)
    fig,axes=plt.subplots(2,3,figsize=(16,8))
    for ax,r in zip(axes.ravel(),selected):
        mask=masks[r['index']]; ax.imshow(image,cmap='gray',vmin=0,vmax=1,aspect='auto')
        overlay=np.zeros((*image.shape,4));overlay[mask]=[1,.5,0,.45];ax.imshow(overlay,aspect='auto')
        ax.plot(np.r_[points[:,0],points[0,0]]/1418*image.shape[1],np.r_[points[:,1],points[0,1]]/853*image.shape[0],color='red',lw=1)
        cfg=r['config'];ax.set_title(f"#{r['index']}: {cfg['gate']}, feature x{cfg['distance']}\n"
            f"keep {r['desired_baseline_retained']:.1%}, spill removed {r['unwanted_baseline_removed']:.1%}",fontsize=10)
        ax.axis('off')
    fig.tight_layout(); fig.savefig(OUT/'comparison.png',dpi=150); plt.close(fig)
    np.savez_compressed(OUT/'candidates.npz', **{f'mask_{r["index"]}':masks[r['index']] for r in selected}, reference=reference)
    print(json.dumps({'baseline':records[0], 'top':ranked[:5], 'texture_only':texture_only[:3]}),flush=True)
