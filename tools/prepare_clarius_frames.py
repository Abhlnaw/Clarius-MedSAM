"""Decompress selected Clarius frame bytes for MATLAB; no image processing.

Uses the existing capture manifest to preserve IDs and source selection.
Only standard-library archive, binary-format and subprocess utilities are used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tarfile
import tempfile


def select_frame(stream, expected_id, target_timestamp=None):
    """Return an intact one-frame RAW stream; never transform sample values."""
    header = stream.read(20)
    if len(header) != 20:
        raise ValueError('Truncated RAW header')
    rid, count, lines, samples, size = struct.unpack('<5I', header)
    expected_size = {1: 1, 2: 2}[expected_id]
    if (rid != expected_id or size != expected_size or count < 1 or
            not 0 < lines < 10000 or not 0 < samples < 100000):
        raise ValueError(f'Unexpected RAW header: {(rid, count, lines, samples, size)}')
    best = None
    previous = None
    for index in range(count):
        stamp = stream.read(8)
        if len(stamp) != 8:
            raise ValueError('Truncated frame timestamp')
        timestamp = struct.unpack('<Q', stamp)[0]
        if previous is not None and timestamp < previous:
            raise ValueError('Non-monotonic frame timestamps')
        previous = timestamp
        payload = stream.read(lines * samples * size)
        if len(payload) != lines * samples * size:
            raise ValueError('Truncated frame samples')
        delta = 0 if target_timestamp is None else abs(timestamp - target_timestamp)
        if best is None or delta < best[0]:
            best = (delta, index, stamp, payload)
        if target_timestamp is None or timestamp >= target_timestamp:
            break
    _, index, stamp, payload = best
    metadata = dict(original_frames=count, selected_frame=index + 1,
                    timestamp_raw=str(struct.unpack('<Q', stamp)[0]),
                    lines=lines, samples=samples, sample_size=size)
    return struct.pack('<5I', rid, 1, lines, samples, size) + stamp + payload, metadata


def read_member(archive, member, lzop, temporary, expected_id, target=None):
    with archive.extractfile(member) as stream:
        if not member.name.lower().endswith('.lzo'):
            return select_frame(stream, expected_id, target)
        compressed = temporary / 'stream.lzo'
        with compressed.open('wb') as handle:
            shutil.copyfileobj(stream, handle)
    process = subprocess.Popen([str(lzop), '-dc', str(compressed)],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        return select_frame(process.stdout, expected_id, target)
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        process.wait()


def prepare(manifest_path, output, lzop, cases=None):
    document = json.loads(Path(manifest_path).read_text(encoding='utf-8-sig'))
    jobs = document['jobs']
    if cases:
        missing = set(cases) - {job['id'][:3] for job in jobs}
        if missing:
            raise ValueError(f'Unknown capture IDs: {sorted(missing)}')
        jobs = [job for job in jobs if job['id'][:3] in cases]
    if not jobs:
        raise ValueError('No captures selected')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records, failures = [], []
    with tempfile.TemporaryDirectory(prefix='clarius_bytes_') as temporary:
        for job in jobs:
            try:
                key = job['id']
                if any(ch in key for ch in '/\\') or key in ('.', '..'):
                    raise ValueError('Unsafe capture ID')
                source = Path(job['source_archive'])
                if re.fullmatch(r'exam\d+', source.parent.name, re.I):
                    raise ValueError('exam## folders are excluded from this dataset')
                rf_name = job['source_member']
                env_name = rf_name.replace('_rf.raw', '_env.raw')
                if env_name == rf_name:
                    raise ValueError('Expected an RF source member')
                with tarfile.open(source) as archive:
                    members = {m.name: m for m in archive.getmembers() if m.isfile()}
                    _, rf_info = read_member(archive, members[rf_name], lzop,
                                             Path(temporary), 2)
                    raw, env_info = read_member(archive, members[env_name], lzop,
                        Path(temporary), 1, int(rf_info['timestamp_raw']))
                    yaml_name = env_name.removesuffix('.lzo').removesuffix('.raw') + '.yml'
                    yaml_path = ''
                    if yaml_name in members:
                        yaml_path = str(output / (key + '_env.yml'))
                        with archive.extractfile(members[yaml_name]) as stream:
                            Path(yaml_path).write_bytes(stream.read())
                destination = output / (key + '_env.raw')
                # This writes header/timestamp/sample bytes only. MATLAB owns
                # array interpretation, tone mapping, resizing and PNG export.
                if destination.exists() and destination.read_bytes() != raw:
                    raise ValueError('Refusing to overwrite differing prepared RAW bytes')
                destination.write_bytes(raw)
                records.append(dict(id=key, raw=str(destination),
                    raw_sha256=hashlib.sha256(raw).hexdigest(),
                    source_archive=str(source), source_member=env_name,
                    metadata_yml=yaml_path, rf_timestamp_raw=rf_info['timestamp_raw'],
                    timestamp_delta_ns=str(int(env_info['timestamp_raw']) - int(rf_info['timestamp_raw'])),
                    **env_info))
                print(f'Prepared bytes {len(records)}/{len(jobs)}: {key[:3]}', flush=True)
            except Exception as error:
                failures.append(dict(id=job['id'], error=str(error)))
                print(f"FAILED {job['id'][:3]}: {error}", flush=True)
    result = dict(processing='Archive extraction, LZO decompression and byte-preserving frame selection only',
                  image_processing='None; render with MATLAB export_clarius_bmode',
                  expected_captures=len(jobs), jobs=records, failures=failures)
    (output / 'frames.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    if failures:
        raise RuntimeError(f'{len(failures)} captures failed; inspect frames.json')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lzop', type=Path, required=True)
    parser.add_argument('--cases', nargs='+')
    args = parser.parse_args()
    prepare(args.manifest, args.output, args.lzop, args.cases)
