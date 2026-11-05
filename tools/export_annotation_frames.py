"""Prepare first RF frames from non-exam## archives, then export with MATLAB."""
import argparse
import json
from pathlib import Path
import re
import struct
import subprocess
import tarfile
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lzop', type=Path, required=True)
    parser.add_argument('--matlab', type=Path, required=True)
    args = parser.parse_args()
    archives = sorted(p for folder in args.data.iterdir()
                      if folder.is_dir() and not re.fullmatch(r'exam\d+', folder.name, re.I)
                      for p in folder.rglob('*.tar'))
    args.output.mkdir(parents=True, exist_ok=True)
    jobs, failures = [], []
    with tempfile.TemporaryDirectory(prefix='clarius_first_frames_') as temp:
        temp = Path(temp)
        for archive in archives:
            try:
                with tarfile.open(archive) as tar:
                    members = [m for m in tar.getmembers() if m.isfile() and
                               m.name.lower().endswith(('_rf.raw.lzo', '_rf.raw'))]
                    if not members:
                        raise ValueError('No RF recording in archive')
                    for index, member in enumerate(members):
                        key = f'{len(jobs)+1:03d}_{archive.parent.name}_{archive.stem}'
                        if len(members) > 1:
                            key += f'_rf{index+1}'
                        destination = temp / f'{len(jobs):03d}.raw'
                        with tar.extractfile(member) as source:
                            process = None
                            if member.name.lower().endswith('.lzo'):
                                compressed = temp / 'current.raw.lzo'
                                with compressed.open('wb') as target:
                                    import shutil
                                    shutil.copyfileobj(source, target)
                                process = subprocess.Popen([str(args.lzop), '-dc', str(compressed)],
                                                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                                stream = process.stdout
                            else:
                                stream = source
                            try:
                                header = stream.read(20)
                                rid, frames, lines, samples, size = struct.unpack('<5i', header)
                                if rid != 2 or frames < 1 or size != 2 or not (0 < lines < 10000 and 0 < samples < 100000):
                                    raise ValueError(f'Invalid RF header: {(rid, frames, lines, samples, size)}')
                                payload_size = 8 + lines * samples * size
                                payload = stream.read(payload_size)
                                if len(payload) != payload_size:
                                    raise ValueError('Truncated first frame')
                                destination.write_bytes(struct.pack('<5i', rid, 1, lines, samples, size) + payload)
                            finally:
                                if process:
                                    process.stdout.close()
                                    if process.poll() is None:
                                        process.terminate()
                                    process.wait()
                        jobs.append(dict(id=key, raw=str(destination), source_archive=str(archive),
                                         source_member=member.name, frame=1, original_frames=frames,
                                         height=samples, width=lines,
                                         timestamp_raw=str(struct.unpack('<q', payload[:8])[0])))
                print(f'Prepared {len(jobs)}/{len(archives)}: {archive.parent.name}/{archive.name}', flush=True)
            except Exception as error:
                failures.append(dict(archive=str(archive), error=str(error)))
                print(f'FAILED: {archive}: {error}', flush=True)
        manifest = args.output / 'manifest.json'
        manifest.write_text(json.dumps(dict(jobs=jobs, failures=failures), indent=2), encoding='utf-8')
        matlab_script = Path(__file__).resolve().parents[1] / 'clarius' / 'export_annotation_frames.m'
        quote = lambda value: str(value).replace("'", "''")
        command = f"addpath('{quote(matlab_script.parent)}'); export_annotation_frames('{quote(manifest)}','{quote(args.output)}');"
        subprocess.run([str(args.matlab), '-batch', command], check=True)
        for job in jobs:
            job.pop('raw')
            metadata = json.loads((args.output / 'images' / (job['id'] + '.json')).read_text())
            for key in ('height', 'width', 'native_height', 'native_width', 'resize_method'):
                job[key] = metadata[key]
        manifest.write_text(json.dumps(dict(jobs=jobs, failures=failures), indent=2), encoding='utf-8')
        if failures:
            raise SystemExit(f'{len(failures)} archives failed; inspect manifest.json')


if __name__ == '__main__':
    main()
