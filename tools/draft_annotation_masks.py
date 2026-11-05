"""Render visually authored draft anchors and register within capture groups.

The output manifest distinguishes manual drawings from AI-assisted drafts.
No evaluated segmentation methods are used to construct these references.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.registration import optical_flow_tvl1
from create_reference_masks import smooth_contour
from propagate_reference_contours import registration_image, advect
from review_drawn_outlines import overlay


def raster(paths, shape=(853,1418)):
    canvas=Image.new('L',(shape[1],shape[0]),0);draw=ImageDraw.Draw(canvas)
    for i,path in enumerate(paths):
        draw.polygon([tuple(map(float,p)) for p in path],fill=255 if i==0 else 0)
    return np.asarray(canvas)>0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    root=args.root.resolve();work=root/'draft_work'
    groups=json.loads((work/'groups.json').read_text())
    recipe=json.loads((work/'anchors.json').read_text())
    output=root/'draft_masks';output.mkdir(exist_ok=True)
    review=root/'review';review.mkdir(exist_ok=True)
    source=root/'clean_source_export'/'images'
    records=[]
    correction_path=work/'visual_corrections.json'
    corrections=json.loads(correction_path.read_text()) if correction_path.exists() else {}
    for group in groups:
        anchor=group[0];spec=recipe['anchors'][anchor['id'][:3]]
        scale=np.array([1418/444,853/267])
        paths=[smooth_contour(spec['outer'],1000)*scale]
        theta=np.linspace(0,2*np.pi,120,endpoint=False)
        for x,y,rx,ry in spec['holes_ellipses']:
            paths.append(np.column_stack((x+rx*np.cos(theta),y+ry*np.sin(theta)))*scale)
        paths=[np.clip(p,[0,0],[1417,852]) for p in paths]
        anchor_image=registration_image(source/(anchor['id']+'.png'),(256,426))
        for job in group:
            transferred=job['id']!=anchor['id'];current_paths=paths
            stats={}
            if transferred:
                target=registration_image(source/(job['id']+'.png'),(256,426))
                flow=optical_flow_tvl1(anchor_image,target,attachment=12,tightness=.3,
                                     num_warp=5,num_iter=10,prefilter=True)
                yy,xx=np.indices(target.shape)
                aligned=ndi.map_coordinates(target,[yy+flow[0],xx+flow[1]],order=1,mode='nearest')
                before=float(np.abs(anchor_image-target).mean())
                after=float(np.abs(anchor_image-aligned).mean())
                current_paths=advect(paths,flow,(1418,853))
                stats=dict(unaligned_mae=before,aligned_mae=after,
                           displacement_p95=float(np.percentile(np.hypot(flow[0],flow[1]),95)))
            mask=raster(current_paths)
            correction=corrections.get(job['id'][:3])
            if correction is not None:
                mask=ndi.binary_fill_holes(mask)
                if 'outer' in correction:
                    mask=raster([smooth_contour(correction['outer'],1000)*scale])
                holes=Image.new('L',(1418,853));draw=ImageDraw.Draw(holes)
                for x,y,rx,ry in correction['holes_ellipses']:
                    draw.ellipse(((x-rx)*scale[0],(y-ry)*scale[1],
                                  (x+rx)*scale[0],(y+ry)*scale[1]),fill=255)
                mask=mask&~(np.asarray(holes)>0)
            components,_=ndi.label(mask,structure=np.ones((3,3)))
            sizes=np.bincount(components.ravel());sizes[0]=0
            tiny=(sizes>0)&(sizes<=9)
            removed=int(sizes[tiny].sum())
            if removed:mask=mask&~tiny[components]
            assert .1<mask.mean()<.9
            Image.fromarray(mask.astype('uint8')*255).save(output/(job['id']+'.png'))
            with Image.open(source/(job['id']+'.png')) as image:
                overlay(image,mask).save(review/(job['id']+'.draft.png'))
            records.append(dict(id=job['id'],source='AI-assisted draft',
                method='visually authored anchor with lumen exclusions' if not transferred else 'within-exam optical-flow transfer of visually authored anchor',
                anchor_id=anchor['id'],review_status='pending_independent_review',
                review_priority='high' if transferred else 'normal',
                foreground_pixels=int(mask.sum()),
                holes=int(ndi.label(ndi.binary_fill_holes(mask)&~mask)[1]),
                registration=stats,
                removed_raster_artifact_pixels=removed,
                source_sha256=hashlib.sha256((source/(job['id']+'.png')).read_bytes()).hexdigest()))
            if correction is not None:
                records[-1]['visual_correction']='Individually revised approximate lumen exclusions and/or boundary; requires independent review'
                records[-1]['review_priority']='high'
            print('Draft',job['id'][:3],flush=True)
    (review/'draft_audit.json').write_text(json.dumps(records,indent=2))
    for page in range((len(records)+11)//12):
        sheet=Image.new('RGB',(1350,1200),'white');draw=ImageDraw.Draw(sheet)
        for k,record in enumerate(records[page*12:page*12+12]):
            with Image.open(review/(record['id']+'.draft.png')) as image:im=image.resize((444,267))
            x=k%3*450;y=k//3*300;sheet.paste(im,(x,y+24))
            draw.text((x+4,y+5),record['id'][:3],fill='black')
        sheet.save(review/f'drafts_{page+1:02d}.png')
    print('Completed',len(records),'draft masks')


if __name__=='__main__':main()
