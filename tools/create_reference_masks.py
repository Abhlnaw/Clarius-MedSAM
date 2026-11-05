"""Rasterize reviewable provisional reference contours, without reading predictions.

Contours are authored in the declared display coordinate frame. Generated masks
retain the source image's native shape. No model, intensity thresholds, or
MedSAM clipping are used to construct a reference.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from skimage.draw import polygon
from skimage.measure import find_contours
from skimage.transform import resize
from scipy.interpolate import CubicSpline


def smooth_contour(points, samples=1200):
    """Interpolate authored vertices with a closed chord-parameterized curve."""
    points=np.asarray(points,float)
    if len(points)>1 and np.array_equal(points[0],points[-1]):points=points[:-1]
    closed=np.vstack((points,points[0]))
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(closed,axis=0),axis=1))]
    if len(points)<3 or np.any(np.diff(distance)<=0):
        raise ValueError('Closed contour requires distinct neighboring vertices')
    return CubicSpline(distance,closed,bc_type='periodic')(np.linspace(0,distance[-1],samples,endpoint=False))


def render_reference_overlay(image, mask, display_size=(1418,853), longest_edge=2048, curves=None):
    """RGB ultrasound review image; the label array itself is never resized.

    Display proportions reproduce the user's reference view, not calibrated
    physical geometry. Curves may preserve the original subpixel annotation;
    otherwise boundaries are extracted from a linearly interpolated display.
    """
    width,height=display_size
    factor=longest_edge/max(width,height)
    size=(int(round(width*factor)),int(round(height*factor)))
    source=Image.fromarray(np.asarray(image)).convert('RGB')
    canvas=source.resize(size,Image.Resampling.BILINEAR)
    scale=2
    drawing=canvas.resize((size[0]*scale,size[1]*scale),Image.Resampling.BILINEAR)
    pen=ImageDraw.Draw(drawing)
    if curves is None:
        displayed=resize(mask.astype(float),(size[1],size[0]),order=1,preserve_range=True,anti_aliasing=False)
        paths=[np.stack((contour[:,1],contour[:,0]),axis=-1) for contour in find_contours(np.pad(displayed,1),.5)]
        paths=[path-1 for path in paths]
    else:
        paths=[np.asarray(path,float)*np.array(size)/np.array(display_size) for path in curves]
    for path in paths:
        vertices=[tuple(point*scale) for point in path]
        if len(vertices)>2:pen.line(vertices+[vertices[0]],fill=(245,20,30),width=3*scale,joint='curve')
    return drawing.resize(size,Image.Resampling.LANCZOS)


def rasterize(points, shape, display_size):
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or not np.isfinite(points).all():
        raise ValueError('A contour must contain at least three finite (x,y) vertices')
    width,height=display_size
    if np.any(points<0) or np.any(points[:,0]>width) or np.any(points[:,1]>height):
        raise ValueError('Contour vertex lies outside its declared display coordinate frame')
    rows,cols=polygon(points[:,1]/height*shape[0]-.5,points[:,0]/width*shape[1]-.5,shape)
    result=np.zeros(shape,bool);result[rows,cols]=True
    return result


def create(contours_path, output, predictions):
    contours=json.loads(contours_path.read_text(encoding='utf-8-sig'))
    if not contours.get('reference_provenance') or not contours.get('review_status'):
        raise ValueError('Contour file must state provenance and review status')
    output.mkdir(parents=True,exist_ok=True)
    (output/'labels_native').mkdir(exist_ok=True)
    (output/'overlays').mkdir(exist_ok=True)
    cases=[];review=[]
    for case in contours['cases']:
        source=Path(case['source'])
        if not source.is_absolute():source=contours_path.parent/source
        with Image.open(source) as handle:image=np.asarray(handle).copy()
        shape=image.shape[:2]
        display_size=case.get('display_size_wh',contours['display_size_wh'])
        paths=[np.asarray(case['outer'],float)]+[np.asarray(hole,float) for hole in case.get('holes',[])]
        if case.get('curve_mode')=='closed_spline':
            paths=[np.clip(smooth_contour(path),[0,0],display_size) for path in paths]
        mask=rasterize(paths[0],shape,display_size)
        for hole in paths[1:]:mask &= ~rasterize(hole,shape,display_size)
        filename=case['id']+'.labels.png'
        target=output/'labels_native'/filename
        Image.fromarray(mask.astype(np.uint8)*255).save(target)
        overlay=render_reference_overlay(image,mask,display_size=display_size,curves=paths)
        overlay_path=output/'overlays'/(case['id']+'.reference.png')
        overlay.save(overlay_path)
        # These are paths only: predictions are first opened by the evaluator,
        # after the reference masks and hashes have been frozen here.
        predicted=(predictions/(case['id']+'.masks.npz')).resolve()
        cases.append(dict(id=case['id'],exam=case['exam'],
            reference=dict(path='labels_native/'+filename,label=255,
                sha256=hashlib.sha256(target.read_bytes()).hexdigest()),
            predictions={'combined_glcm':dict(path=str(predicted),key='combined'),
                'brightness_glszm':dict(path=str(predicted),key='combined_glszm')},
            evaluation_mask=dict(path=str(predicted),key='parent'),
            notes=case.get('notes','')))
        review.append(dict(case=case['id'],source=str(source.resolve()),shape_hw=list(shape),
            reference_pixels=int(mask.sum()),reference_sha256=cases[-1]['reference']['sha256'],
            overlay_path=str(overlay_path.resolve()),overlay_shape_hw=[overlay.height,overlay.width],
            display_proportions_wh=list(display_size),annotation_method=case.get('annotation_method','authored_contour'),
            notes=case.get('notes','')))
    manifest=dict(reference_provenance=contours['reference_provenance'],
        reference_review_status=contours['review_status'],cases=cases)
    (output/'evaluation_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (output/'reference_audit.json').write_text(json.dumps(dict(annotation_protocol=contours.get('annotation_protocol',''),
        prior_exposure_to_model_outputs=contours.get('prior_exposure_to_model_outputs',True),references=review),indent=2),encoding='utf-8')
    fig,axes=plt.subplots((len(cases)+1)//2,2,figsize=(16,4.8*((len(cases)+1)//2)),squeeze=False)
    for ax,case,record in zip(axes.ravel(),contours['cases'],review):
        with Image.open(record['overlay_path']) as handle:image=np.asarray(handle).copy()
        ax.imshow(image)
        ax.set_title(case['id']+' — provisional reference');ax.axis('off')
    for ax in axes.ravel()[len(cases):]:ax.axis('off')
    fig.suptitle('Unreviewed AI-drawn liver-region outlines — not verified ground truth',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.98));fig.savefig(output/'reference_review.png',dpi=120);plt.close(fig)
    (output/'labels_native'/'README.txt').write_text('Machine-readable binary labels for scoring only. Native RF sampling is tall/narrow. Open ../overlays/ for ultrasound review images in the presentation proportions of the user red-outline example. Do not resize these labels before scoring.\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contours',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--predictions',type=Path,required=True)
    args=parser.parse_args();manifest=create(args.contours.resolve(),args.output.resolve(),args.predictions)
    print(f"Created {len(manifest['cases'])} reviewable native-grid references; no predictions were read.")


if __name__=='__main__':main()
