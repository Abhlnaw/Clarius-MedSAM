"""Propagate reviewable anchor contours between consecutive ORIGINAL frames.

This is provisional annotation assistance, not a segmentation ground-truth
generator. No prediction masks or intensity-based segmentation rules are read.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.registration import optical_flow_tvl1
from skimage.transform import resize
from create_reference_masks import smooth_contour


def advect(paths, flow, display_size):
    """Move display-coordinate vertices with forward (row,column) flow."""
    height,width=flow.shape[1:]
    result=[]
    for path in paths:
        points=np.asarray(path,float)
        x=points[:,0]/display_size[0]*width-.5
        y=points[:,1]/display_size[1]*height-.5
        dx=ndi.map_coordinates(flow[1],[y,x],order=1,mode='nearest')
        dy=ndi.map_coordinates(flow[0],[y,x],order=1,mode='nearest')
        moved=np.stack(((x+dx+.5)/width*display_size[0],
                        (y+dy+.5)/height*display_size[1]),axis=-1)
        result.append(np.clip(moved,[0,0],display_size))
    return result


def registration_image(path,shape):
    with Image.open(path) as handle:gray=np.asarray(handle.convert('L'),dtype=np.float32)/255
    return ndi.gaussian_filter(resize(gray,shape,order=1,preserve_range=True,anti_aliasing=True),1.2).astype(np.float32)


def propagate(recipe_path,output):
    recipe=json.loads(recipe_path.read_text(encoding='utf-8-sig'))
    display_size=recipe['display_size_wh'];shape=tuple(recipe.get('registration_shape_hw',[256,426]))
    cases=[];registration=[]
    for anchor in recipe['anchors']:
        source=Path(anchor['source'])
        if not source.is_absolute():source=recipe_path.parent/source
        paths=[smooth_contour(anchor['outer'],1200)]
        paths += [smooth_contour(hole,400) for hole in anchor.get('holes',[])]
        paths=[np.clip(path,[0,0],display_size) for path in paths]
        def snapshot(frame,source_path,method):
            cases.append(dict(id=f"{anchor['exam']}_frame_{frame:03d}",exam=anchor['exam'],source=str(source_path.resolve()),
                outer=paths[0].round(4).tolist(),holes=[path.round(4).tolist() for path in paths[1:]],
                annotation_method=method,anchor_case=f"{anchor['exam']}_frame_{anchor['frame']:03d}",
                notes=anchor.get('notes','')+' Neighboring frames use provisional optical-flow contour propagation; review required.'))
        snapshot(anchor['frame'],source,anchor['annotation_method'])
        previous=registration_image(source,shape)
        targets=set(anchor['target_frames'])
        for frame in range(anchor['frame']+1,max(targets,default=anchor['frame'])+1):
            source_path=source.parent/f'frame_{frame:03d}.png'
            current=registration_image(source_path,shape)
            flow=optical_flow_tvl1(previous,current,attachment=12,tightness=.3,num_warp=5,num_iter=10,prefilter=True)
            yy,xx=np.indices(shape,dtype=float)
            aligned=ndi.map_coordinates(current,[yy+flow[0],xx+flow[1]],order=1,mode='nearest')
            record=dict(exam=anchor['exam'],from_frame=frame-1,to_frame=frame,
                unaligned_mae=float(np.mean(np.abs(previous-current))),
                aligned_mae=float(np.mean(np.abs(previous-aligned))),
                flow_magnitude_p95=float(np.quantile(np.sqrt((flow**2).sum(axis=0)),.95)))
            registration.append(record);print(json.dumps(record),flush=True)
            paths=advect(paths,flow,display_size)
            if frame in targets:snapshot(frame,source_path,'image_registration_from_anchor')
            previous=current
    result={key:recipe[key] for key in ['reference_provenance','review_status','annotation_protocol','prior_exposure_to_model_outputs','display_size_wh']}
    result['cases']=cases
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    output.with_name('registration_audit.json').write_text(json.dumps(dict(
        registration_shape_hw=list(shape),method='TV-L1 on smoothed original images; forward vertex advection; no prediction masks',
        parameters=dict(attachment=12,tightness=.3,num_warp=5,num_iter=10,prefilter=True),steps=registration),indent=2),encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();result=propagate(args.recipe.resolve(),args.output)
    print(f"Generated {len(result['cases'])} provisional contour sets. Inspect the rendered ultrasound overlays.")


if __name__=='__main__':main()
