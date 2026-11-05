"""Convert explicitly seeded user red contours to reviewable binary masks.

Small endpoint gaps and endpoints within 20 pixels of the image border are
bridged explicitly and logged. The original drawings are never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize
from skimage.measure import find_contours

SEEDS = [(800,300),(900,450),(900,300),(900,200),(900,250),
         (900,200),(700,350),(700,250),(700,300),(900,200),
         (700,200),(700,200),(900,300),(700,350),(800,300),
         (900,200),(900,200),(900,250),(700,300),(700,250),
         (700,250),(900,200),(900,200),(900,250),(900,300)]


def convert(rgb, seed):
    a = np.asarray(rgb, dtype=np.int16)
    ink = (a[:,:,0]>130)&(a[:,:,0]>a[:,:,1]+40)&(a[:,:,0]>a[:,:,2]+40)
    skeleton = skeletonize(ink)
    neighbors = ndi.convolve(skeleton.astype(int), np.ones((3,3),int))-skeleton
    endpoints = np.argwhere(skeleton & (neighbors==1))
    canvas = Image.fromarray(ink.astype('uint8')*255)
    draw = ImageDraw.Draw(canvas)
    bridges = []
    h,w = ink.shape
    used = set()
    if len(endpoints):
        distances = cdist(endpoints,endpoints)
        np.fill_diagonal(distances,np.inf)
        for i,(y,x) in enumerate(endpoints):
            if i in used: continue
            edge_points = [(x,0),(x,h-1),(0,y),(w-1,y)]
            edge = min(edge_points,key=lambda q:(q[0]-x)**2+(q[1]-y)**2)
            edge_distance = np.hypot(edge[0]-x,edge[1]-y)
            nearest = int(np.argmin(distances[i]))
            if edge_distance <= 20 and edge_distance <= distances[i,nearest]:
                target = edge; used.add(i)
            elif distances[i,nearest] <= 20 and nearest not in used:
                yy,xx=endpoints[nearest];target=(int(xx),int(yy));used.update((i,nearest))
            else: continue
            start=(int(x),int(y));target=tuple(map(int,target))
            draw.line([start,target],fill=255,width=3)
            bridges.append(dict(start=start,end=target))
    barrier = np.asarray(canvas)>0
    barrier = ndi.binary_closing(barrier,iterations=2)|barrier
    barrier = ndi.binary_dilation(barrier,iterations=1)
    barrier[[0,-1],:]=True;barrier[:,[0,-1]]=True
    labels,n=ndi.label(~barrier)
    x,y=seed; component=int(labels[y,x])
    if component==0: raise ValueError('Seed lies on ink')
    interior=labels==component
    if not (.1<interior.mean()<.8): raise ValueError('Unclosed or implausible outline')
    # Assign line pixels to their nearest region rather than erasing holes.
    nearest=ndi.distance_transform_edt(barrier,return_distances=False,return_indices=True)
    mask=interior.copy();mask[barrier]=interior[tuple(nearest[:,barrier])]
    # The image boundary itself belongs to the adjacent region.
    return mask,bridges,ink


def overlay(image,mask):
    result=image.convert('RGB');draw=ImageDraw.Draw(result)
    for contour in find_contours(np.pad(mask.astype(float),1),.5):
        points=[(float(x-1),float(y-1)) for y,x in contour]
        if len(points)>2:draw.line(points,fill=(235,35,45),width=3)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    root=args.root
    review=root/'review';review.mkdir(exist_ok=True)
    output=root/'converted_manual_masks';output.mkdir(exist_ok=True)
    records=[]
    for i,path in enumerate(sorted((root/'images').glob('*.png'))[:25]):
        with Image.open(path) as image: rgb=image.convert('RGB')
        mask,bridges,ink=convert(rgb,SEEDS[i])
        Image.fromarray(mask.astype('uint8')*255).save(output/path.name)
        clean=Image.open(root/'clean_source_export'/'images'/path.name).convert('RGB')
        rendered=overlay(clean,mask);rendered.save(review/(path.stem+'.manual.png'))
        records.append(dict(id=path.stem,source='user red drawing',
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            review_status='conversion_requires_review',foreground_pixels=int(mask.sum()),
            holes=int(ndi.label(ndi.binary_fill_holes(mask)&~mask)[1]),
            edge_contact=bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any()),
            bridged_endpoints=bridges,seed_xy=SEEDS[i]))
    (review/'manual_conversion_audit.json').write_text(json.dumps(records,indent=2))
    sheet=Image.new('RGB',(1500,1050),'white');draw=ImageDraw.Draw(sheet)
    for i,record in enumerate(records):
        im=Image.open(review/(record['id']+'.manual.png'));im.thumbnail((296,178))
        x=i%5*300;y=i//5*210;sheet.paste(im,(x,y+24))
        draw.text((x+3,y+3),f"{i+1:03d}; holes {record['holes']}; bridges {len(record['bridged_endpoints'])}",fill='black')
    sheet.save(review/'first25_converted.png')
    print([(r['id'][:3],r['holes'],len(r['bridged_endpoints'])) for r in records])


if __name__=='__main__':main()
