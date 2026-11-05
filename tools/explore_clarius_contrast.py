"""Historical Python contrast experiment, without using masks.

The active export pipeline is now prepare_clarius_frames.py (decompression
only), followed by MATLAB export_clarius_bmode. This experiment is retained
for reproducing the original comparison, not for creating current inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import tarfile
import tempfile

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clarius_contrast import rf_envelope, rf_legacy, rf_window, envelope_tone, mild_clahe

VARIANTS = ('rf_current', 'rf_dr60', 'rf_dr50', 'rf_dr40',
            'env_native', 'env_gamma08', 'env_window', 'env_clahe')
LABELS = dict(rf_current='Current RF: per-frame min/max', rf_dr60='RF: 60 dB window',
    rf_dr50='RF: 50 dB window', rf_dr40='RF: 40 dB window',
    env_native='Clarius B-mode: original 0-255', env_gamma08='Clarius B-mode: gamma 0.8',
    env_window='Clarius B-mode: window 5-210, gamma 0.9',
    env_clahe='Clarius B-mode: mild CLAHE (exploratory)')


def first_frame(tar, member, lzop, temporary, target_timestamp=None):
    """Read first RF frame or nearest-timestamp B-mode frame, not whole cines."""
    compressed = temporary/'stream.lzo'
    with tar.extractfile(member) as stream, compressed.open('wb') as handle:
        import shutil
        shutil.copyfileobj(stream, handle)
    process = subprocess.Popen([str(lzop), '-dc', str(compressed)],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        raw = process.stdout.read(20)
        if len(raw) != 20:
            raise ValueError('Truncated header')
        rid, count, lines, samples, size = struct.unpack('<5I', raw)
        if count < 1 or not 0 < lines < 10000 or not 0 < samples < 100000 or size not in (1,2):
            raise ValueError(f'Unexpected raw header: {(rid,count,lines,samples,size)}')
        best = None
        for index in range(count):
            timestamp_bytes = process.stdout.read(8)
            if len(timestamp_bytes) != 8:
                raise ValueError('Truncated timestamp')
            timestamp = struct.unpack('<Q', timestamp_bytes)[0]
            payload = process.stdout.read(lines*samples*size)
            if len(payload) != lines*samples*size:
                raise ValueError('Truncated frame')
            delta = 0 if target_timestamp is None else abs(timestamp-target_timestamp)
            if best is None or delta < best[0]:
                best = (delta,index,timestamp,payload)
            if target_timestamp is None or timestamp >= target_timestamp:
                break
        _,index,timestamp,payload = best
        data = np.frombuffer(payload,dtype='uint8' if size==1 else '<i2').reshape(lines,samples).T.copy()
        return data,dict(id=rid,frames=count,lines=lines,samples=samples,sample_size=size,
                         selected_frame=index+1,timestamp_ns=str(timestamp))
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        process.wait()


def metadata(tar, name):
    member = next((m for m in tar.getmembers() if m.name == name),None)
    if member is None:return dict(available=False)
    text = tar.extractfile(member).read().decode('utf-8',errors='replace')
    fields = {}
    for key in ('software version','imaging depth','sampling rate','delay samples','mla','focal depth'):
        match = re.search(r'^'+re.escape(key)+r':\s*(.+)$',text,re.M)
        if match:fields[key]=match.group(1)
    rx = re.findall(r'rx element:\s*([-\d.]+)',text)
    if rx:fields['rx_element_extent']=[float(rx[0]),float(rx[-1])]
    fields['yaml_sha256']=hashlib.sha256(text.encode()).hexdigest()
    return fields


def geometry_checks(acquisition):
    rf,env=acquisition['rf'],acquisition['env']
    lateral=None
    if 'rx_element_extent' in rf and 'rx_element_extent' in env:
        lateral=bool(np.allclose(rf['rx_element_extent'],env['rx_element_extent'],rtol=0,atol=1e-6))
    return dict(same_recorded_depth=rf.get('imaging depth')==env.get('imaging depth'),
                same_recorded_lateral_extent=lateral,
                note='Matching extents and timestamps do not prove pixel correspondence across different acquisition streams.')


def panel(output, case, variants):
    tile_w,tile_h = 480,289
    columns=4;rows=(len(variants)+columns-1)//columns
    sheet=Image.new('RGB',(columns*tile_w,rows*(tile_h+30)),'white');draw=ImageDraw.Draw(sheet)
    for i,variant in enumerate(variants):
        with Image.open(output/variant/(case+'.png')) as im:tile=im.resize((tile_w,tile_h))
        x=i%columns*tile_w;y=i//columns*(tile_h+30)
        draw.text((x+6,y+8),LABELS[variant],fill='black');sheet.paste(tile,(x,y+30))
    sheet.save(output/'comparisons'/(case+'.png'))


def run(args):
    manifest=json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    jobs=manifest['jobs']
    if args.cases:jobs=[job for job in jobs if job['id'][:3] in args.cases]
    if not jobs:raise ValueError('No selected cases')
    args.output.mkdir(parents=True,exist_ok=True)
    for variant in args.variants:(args.output/variant).mkdir(exist_ok=True)
    (args.output/'comparisons').mkdir(exist_ok=True)
    records=[];failures=[]
    with tempfile.TemporaryDirectory(prefix='clarius_contrast_') as temporary:
        for job in jobs:
            try:
                with tarfile.open(job['source_archive']) as tar:
                    rf_member=next(m for m in tar.getmembers() if m.name==job['source_member'])
                    env_name=rf_member.name.replace('_rf.raw.lzo','_env.raw.lzo')
                    env_member=next(m for m in tar.getmembers() if m.name==env_name)
                    rf,rf_header=first_frame(tar,rf_member,args.lzop,Path(temporary))
                    env,env_header=first_frame(tar,env_member,args.lzop,Path(temporary),int(rf_header['timestamp_ns']))
                    if rf_header['sample_size']!=2 or env_header['sample_size']!=1:
                        raise ValueError('Wrong RF/B-mode sample types')
                    acquisition=dict(rf=metadata(tar,rf_member.name.removesuffix('.raw.lzo')+'.yml'),
                        env=metadata(tar,env_name.removesuffix('.raw.lzo')+'.yml'))
                envelope=rf_envelope(rf)
                record=dict(id=job['id'],source_archive=job['source_archive'],rf=rf_header,env=env_header,
                    timestamp_delta_ns=int(env_header['timestamp_ns'])-int(rf_header['timestamp_ns']),
                    metadata=acquisition,geometry_checks=geometry_checks(acquisition),variants={})
                for variant in args.variants:
                    settings={}
                    if variant=='rf_current':pixels=rf_legacy(envelope)
                    elif variant.startswith('rf_dr'):
                        pixels,settings=rf_window(envelope,float(variant[5:]))
                    elif variant=='env_native':pixels=env.copy()
                    elif variant=='env_gamma08':pixels=envelope_tone(env,gamma=.8);settings=dict(gamma=.8)
                    elif variant=='env_window':pixels=envelope_tone(env,5,210,.9);settings=dict(black=5,white=210,gamma=.9)
                    else:pixels=mild_clahe(env);settings=dict(clip_limit=.01,tiles='approximately 8 x 8')
                    # Each method uses the complete source frame; no crop, warp,
                    # mask, local anatomical selection, or AI synthesis is used.
                    displayed=Image.fromarray(pixels).resize((1418,853),Image.Resampling.BILINEAR)
                    path=args.output/variant/(job['id']+'.png');displayed.save(path)
                    quantiles=np.percentile(pixels,[0,1,5,25,50,75,95,99,100]).tolist()
                    record['variants'][variant]=dict(settings=settings,native_shape=list(pixels.shape),
                        output_shape=[853,1418],native_percentiles=quantiles,
                        native_black_fraction=float((pixels==0).mean()),
                        native_white_fraction=float((pixels==255).mean()),
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),path=str(path.resolve()))
                if args.panels:panel(args.output,job['id'],args.variants)
                records.append(record)
                print(f"Contrast {len(records)}/{len(jobs)}: {job['id'][:3]}; B/RF delta={record['timestamp_delta_ns']/1e6:.3f} ms",flush=True)
            except Exception as error:
                failures.append(dict(id=job['id'],error=str(error)))
                print(f"FAILED {job['id']}: {error}",flush=True)
    result=dict(description='Image-formation/contrast experiment only; no reference masks or segmentation used',
        quantile_order=[0,1,5,25,50,75,95,99,100],
        display_size_wh=[1418,853],cropped=False,
        note='RF and B streams have different acquisition settings and can differ temporally and geometrically; this is not a matched pixel-level contrast-only comparison.',
        cases=records,failures=failures)
    (args.output/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if failures:raise SystemExit(f'{len(failures)} failed cases; see manifest')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--lzop',type=Path,required=True)
    parser.add_argument('--cases',nargs='+')
    parser.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    parser.add_argument('--panels',action='store_true')
    run(parser.parse_args())
