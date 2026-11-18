"""Byte-only preparation tests; MATLAB separately tests the image analysis."""
import io
import json
from pathlib import Path
import struct
import tarfile
import tempfile
import unittest

from tools.prepare_clarius_glcm import prepare


class PrepareGlcmTests(unittest.TestCase):
    def fixture(self,root,include_yaml=True):
        timestamp=(1<<53)+17
        payload=bytes(range(24))
        raw=struct.pack('<5I',2,2,3,4,2)+struct.pack('<Q',timestamp)+payload+struct.pack('<Q',timestamp+100)+bytes(24)
        archive=root/'capture.tar';yaml=b'sampling rate: 60 MHz\n'
        with tarfile.open(archive,'w') as stream:
            for name,data in [('./fixture_rf.raw',raw)]+([('./fixture_rf.yml',yaml)] if include_yaml else []):
                info=tarfile.TarInfo(name);info.size=len(data);stream.addfile(info,io.BytesIO(data))
        manifest=root/'manifest.json'
        manifest.write_text(json.dumps(dict(jobs=[dict(id='001_fixture',source_archive=str(archive),source_member='./fixture_rf.raw')])))
        return manifest,struct.pack('<5I',2,1,3,4,2)+struct.pack('<Q',timestamp)+payload,yaml

    def test_exact_samples_metadata_and_integer_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest,expected,yaml=self.fixture(root)
            prepare(manifest,root/'prepared',Path('unused'),['001'])
            self.assertEqual((root/'prepared/001_fixture_rf.raw').read_bytes(),expected)
            self.assertEqual((root/'prepared/001_fixture_rf.yml').read_bytes(),yaml)
            job=json.loads((root/'prepared/frames.json').read_text())['jobs'][0]
            self.assertEqual(job['timestamp_raw'],str((1<<53)+17))
            self.assertEqual(job['selected_frame'],1)

    def test_differing_prepared_bytes_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest,_,_=self.fixture(root)
            output=root/'prepared';output.mkdir();raw=output/'001_fixture_rf.raw';raw.write_bytes(b'existing')
            with self.assertRaisesRegex(ValueError,'Refusing differing'):
                prepare(manifest,output,Path('unused'),['001'])
            self.assertEqual(raw.read_bytes(),b'existing')

    def test_missing_yaml_and_unknown_id_are_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest,_,_=self.fixture(root,False)
            with self.assertRaisesRegex(ValueError,'YAML missing'):
                prepare(manifest,root/'prepared',Path('unused'),['001'])
            with self.assertRaisesRegex(ValueError,'Unknown capture'):
                prepare(manifest,root/'other',Path('unused'),['999'])


if __name__=='__main__':unittest.main()
