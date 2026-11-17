"""Render full-image versus localized scientific comparisons on native grids."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def main():
    root=ROOT/'clarius_annotations/liver_localization_20261117'
    references=json.loads((ROOT/'clarius_annotations/verified_evaluation_20261117/reference_manifest.json').read_text())['cases']
    results=json.loads((root/'scores/metrics.json').read_text())['rows']
    scores={(r['case'],r['method']):r for r in results}
    boxes={r['id']:r for r in json.loads((root/'training.json').read_text())['predictions']}
    worst=min((r for r in results if r['method']=='predicted'),key=lambda r:r['dice'])['case']
    keys=list(dict.fromkeys([references[0]['id'],worst]))
    folder=root/'comparisons';folder.mkdir(exist_ok=True)
    for key in keys:
        case=next(c for c in references if c['id']==key)
        image=np.asarray(Image.open(case['clean_image']).convert('RGB'))
        reference=np.asarray(Image.open(case['reference']))>0
        with np.load(root/'parents'/(key+'.npz')) as saved:parents={k:saved[k] for k in ('full_image','predicted')}
        with np.load(root/'refinements'/(key+'.npz')) as saved:parents['predicted_combined']=saved['predicted_combined']
        fig,axes=plt.subplots(1,3,figsize=(18,4.8))
        for axis,method,label in zip(axes,['full_image','predicted','predicted_combined'],
                ['Full-image MedSAM','CNN box + MedSAM','CNN box + MedSAM + Combined GLCM']):
            axis.imshow(image)
            rgba=np.zeros((*reference.shape,4));rgba[parents[method]]=[1,.55,0,.4]
            axis.imshow(rgba);axis.contour(reference.astype(float),levels=[.5],colors=['#20ff50'],linewidths=1)
            b=np.array([0,0,1,1]) if method=='full_image' else np.clip(np.array(boxes[key]['predicted_box'])+[-.03,-.03,.03,.03],0,1)
            h,w=reference.shape;x1,y1,x2,y2=b*[w,h,w,h]
            axis.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='#35bfff',linewidth=1))
            axis.set_title(f'{label}\nDice {scores[(key,method)]["dice"]:.3f}',fontsize=12)
            axis.set_xlim(-.5,w-.5);axis.set_ylim(h-.5,-.5);axis.axis('off')
        fig.suptitle(f'Image {key[:3]} | Green: verified contour, orange: prediction, blue: prompt',fontsize=13)
        fig.tight_layout();fig.savefig(folder/f'comparison_{key[:3]}.png',dpi=160);plt.close(fig)
    (folder/'selection.json').write_text(json.dumps(dict(first_case=references[0]['id'],worst_localized_medsam_case=worst,
        note='Illustrative first case and worst CNN-parent Dice; review.html contains all images'),indent=2))
    print('Rendered comparisons:',[key[:3] for key in keys])


if __name__=='__main__':main()
