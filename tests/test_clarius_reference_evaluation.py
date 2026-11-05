import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

module_spec=importlib.util.spec_from_file_location('reference_evaluator',Path(__file__).resolve().parents[1]/'tools/evaluate_segmentations.py')
evaluator=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(evaluator)


class ReferenceEvaluationTests(unittest.TestCase):
    def test_indexed_label_ids_and_hash_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);values=np.zeros((20,24),np.uint8);values[3:8,4:10]=1;values[10:15,4:10]=2
            image=Image.fromarray(values).convert('P');image.save(root/'labels.png')
            spec=dict(path='labels.png',label=1,sha256=hashlib.sha256((root/'labels.png').read_bytes()).hexdigest())
            np.testing.assert_array_equal(evaluator.load_mask(spec,root),values==1)
            spec['sha256']='invalid'
            with self.assertRaisesRegex(ValueError,'changed'):evaluator.load_mask(spec,root)

    def test_batch_scores_references_and_counts_foreground_outside_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);reference=np.zeros((30,40),bool);reference[10:20,15:25]=True
            Image.fromarray(reference.astype(np.uint8)*255).save(root/'reference.png')
            np.savez(root/'predictions.npz',matching=reference,empty=np.zeros_like(reference),parent=np.zeros_like(reference))
            case=dict(id='frame',reference=dict(path='reference.png',label=255),
                predictions={'method_a':dict(path='predictions.npz',key='matching'),
                             'method_b':dict(path='predictions.npz',key='empty')},
                evaluation_mask=dict(path='predictions.npz',key='parent'))
            manifest=dict(reference_provenance='Synthetic unit-test reference',reference_review_status='synthetic',cases=[case])
            path=root/'manifest.json';path.write_text(json.dumps(manifest))
            result=evaluator.evaluate_manifest(path,root/'output')
            self.assertEqual(result['summary']['method_a']['ssim']['mean'],1)
            row=next(row for row in result['rows'] if row['method']=='method_b')
            self.assertEqual(row['false_negative_pixels'],100);self.assertEqual(row['dice'],0)
            self.assertTrue((root/'output/per_image.csv').is_file())
            self.assertEqual(json.loads((root/'output/metrics.json').read_text())['ssim_settings']['data_range'],1)

    def test_incomplete_method_pairs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);mask=np.ones((20,20),bool);np.save(root/'mask.npy',mask)
            first=dict(id='first',reference='mask.npy',predictions={'a':'mask.npy','b':'mask.npy'})
            second=dict(id='second',reference='mask.npy',predictions={'a':'mask.npy'})
            manifest=dict(reference_provenance='synthetic',cases=[first,second])
            path=root/'manifest.json';path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'same methods'):evaluator.evaluate_manifest(path,root/'output')


if __name__=='__main__':unittest.main()
