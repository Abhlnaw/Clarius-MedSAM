"""Prepare unchanged selected RF bytes and YAML for MATLAB physical GLCM."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare_clarius_frames import read_member


def prepare(manifest_path,output,lzop,cases):
    document=json.loads(Path(manifest_path).read_text(encoding='utf-8-sig'))
    selected=[job for job in document['jobs'] if job['id'][:3] in cases]
    if set(cases)-{job['id'][:3] for job in selected}:raise ValueError('Unknown capture ID')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    records=[]
    with tempfile.TemporaryDirectory(prefix='clarius_rf_bytes_') as temporary:
        for job in selected:
            key=job['id']
            if any(c in key for c in '/\\'):raise ValueError('Unsafe capture ID')
            with tarfile.open(job['source_archive']) as archive:
                members={m.name:m for m in archive.getmembers() if m.isfile()}
                name=job['source_member']
                raw,info=read_member(archive,members[name],lzop,Path(temporary),2)
                yaml_name=name.removesuffix('.lzo').removesuffix('.raw')+'.yml'
                if yaml_name not in members:raise ValueError('RF YAML missing')
                with archive.extractfile(members[yaml_name]) as stream:yaml=stream.read()
            raw_path=output/(key+'_rf.raw');yaml_path=output/(key+'_rf.yml')
            for path,data in [(raw_path,raw),(yaml_path,yaml)]:
                if path.exists() and path.read_bytes()!=data:raise ValueError('Refusing differing prepared bytes')
                path.write_bytes(data)
            records.append(dict(id=key,raw=str(raw_path),metadata_yml=str(yaml_path),
                raw_sha256=hashlib.sha256(raw).hexdigest(),yaml_sha256=hashlib.sha256(yaml).hexdigest(),
                source_archive=job['source_archive'],source_member=name,**info))
            print('Prepared RF bytes',key[:3],flush=True)
    result=dict(processing='Byte-preserving RF extraction/decompression/frame selection only; MATLAB owns image processing',jobs=records)
    (output/'frames.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=Path('clarius_annotations/manual_segmentation/manifest.json'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--lzop',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',default=['001','004','010','070'])
    args=parser.parse_args();prepare(args.manifest,args.output,args.lzop,args.cases)
