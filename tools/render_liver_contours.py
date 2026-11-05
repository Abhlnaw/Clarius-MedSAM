"""Render individually authored liver contours without prediction or transfer.

This rasterizer does not identify anatomy. Its inputs are provisional visual
annotations requiring expert review; interpolation cannot establish accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import PchipInterpolator
from scipy import ndimage as ndi
from skimage.measure import find_contours


def curve(points):
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or not np.isfinite(points).all():
        raise ValueError('Expected at least three finite XY vertices')
    closed=np.vstack((points,points[0]))
    lengths=np.linalg.norm(np.diff(closed,axis=0),axis=1)
    if np.any(lengths==0):raise ValueError('Repeated neighboring contour vertex')
    distance=np.r_[0,np.cumsum(lengths)]
    # Coordinate-wise monotone interpolation avoids spline overshoot beyond
    # authored extrema. It supplies no anatomy or image-derived adjustments.
    return PchipInterpolator(distance,closed,axis=0)(np.linspace(0,distance[-1],2400))


def raster(points,shape,coordinate_size):
    path=curve(points)
    scale=np.array([shape[1]/coordinate_size[0],shape[0]/coordinate_size[1]])
    path=path*scale
    canvas=Image.new('L',(shape[1],shape[0]),0)
    ImageDraw.Draw(canvas).polygon([tuple(point) for point in path],fill=255)
    return np.asarray(canvas)>0


def render(root):
    root=Path(root)
    source=json.loads((root/'source_manifest.json').read_text())
    contours=json.loads((root/'contours.json').read_text())
    expected={entry['id'][:3] for entry in source['cases']}
    if len(expected)!=len(source['cases']) or set(contours['cases'])!=expected:
        raise ValueError('Contours must match every source ID exactly once')
    for folder in ('masks','outer_masks','overlays','review'):(root/folder).mkdir(exist_ok=True)
    records=[]
    for entry in source['cases']:
        key=entry['id'][:3]
        item=contours['cases'][key]
        source_path=Path(entry['source'])
        if hashlib.sha256(source_path.read_bytes()).hexdigest()!=entry['source_sha256']:
            raise ValueError('Source image changed')
        with Image.open(source_path) as handle:image=handle.convert('RGB')
        shape=(image.height,image.width)
        outer=raster(item['outer'],shape,contours['coordinate_size_wh'])
        mask=outer.copy();holes=[]
        boundary_exclusions=[]
        for exclusion_index,vertices in enumerate(item['holes']):
            hole=raster(vertices,shape,contours['coordinate_size_wh'])
            outside=int((hole&~outer).sum())
            if outside and exclusion_index not in item.get('boundary_exclusion_indices',[]):
                raise ValueError(f'{key}: exclusion extends {outside} pixels outside liver envelope')
            if outside:boundary_exclusions.append(dict(index=exclusion_index,outside_pixels=outside))
            mask &= ~hole;holes.append(int(hole.sum()))
        if not mask.any():raise ValueError('Empty mask')
        components,n=ndi.label(mask,structure=np.ones((3,3)))
        if n!=1:raise ValueError(f'{key}: {n} foreground components')
        name=entry['id']+'.png'
        path=root/'masks'/name;Image.fromarray(mask.astype('uint8')*255).save(path)
        Image.fromarray(outer.astype('uint8')*255).save(root/'outer_masks'/name)
        pen=ImageDraw.Draw(image)
        for boundary in find_contours(np.pad(mask.astype(float),1),.5):
            pen.line([(float(x-1),float(y-1)) for y,x in boundary],fill=(240,45,50),width=3)
        image.save(root/'overlays'/name)
        records.append(dict(id=entry['id'],source=str(source_path),source_sha256=entry['source_sha256'],
            method='Independent AI visual contour tracing, no model masks or inter-image transfer',
            expert_review_status='pending',notes=item.get('notes',[]),
            visual_revisions=item.get('visual_revisions',[]),
            mask=str(path.resolve()),mask_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            shape_hw=list(shape),foreground_pixels=int(mask.sum()),lumen_count=len(holes),
            exclusion_pixels=holes,boundary_intersecting_exclusions=boundary_exclusions,
            vertex_count=len(item['outer'])))
    manifest=dict(target=contours['target'],provenance=contours['provenance'],
        precision_status='Provisional annotations; expert anatomical and boundary verification required',
        expected_images=len(source['cases']),rendered_images=len(records),cases=records)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for page in range((len(records)+3)//4):
        canvas=Image.new('RGB',(1450,934),'white');pen=ImageDraw.Draw(canvas)
        for slot,record in enumerate(records[page*4:page*4+4]):
            with Image.open(root/'overlays'/(record['id']+'.png')) as handle:
                small=handle.resize((709,427),Image.Resampling.BILINEAR)
            x=slot%2*725;y=slot//2*467;canvas.paste(small,(x,y+28))
            pen.text((x+6,y+7),record['id'][:3]+f" | exclusions: {record['lumen_count']}",fill='black')
        canvas.save(root/'review'/f'page_{page+1:02d}.png')
    print(f"Rendered {len(records)}/{len(source['cases'])} independently authored contours")


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    render(parser.parse_args().root)
