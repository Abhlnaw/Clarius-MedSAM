"""Raster geometry and provenance checks; these do not validate anatomy."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.render_liver_contours import curve, raster, render


class LiverContourTests(unittest.TestCase):
    def test_curve_stays_inside_authored_extrema(self):
        points = [[10, 20], [85, 10], [90, 70], [30, 80]]
        path = curve(points)
        self.assertTrue((path >= [10, 10]).all())
        self.assertTrue((path <= [90, 80]).all())

    def test_xy_scaling_and_lumen_subtraction(self):
        shape, coordinates = (100, 300), (100, 50)
        outer = raster([[10, 5], [90, 5], [90, 45], [10, 45]], shape, coordinates)
        hole = raster([[40, 20], [60, 20], [60, 30], [40, 30]], shape, coordinates)
        mask = outer & ~hole
        self.assertEqual(mask.shape, shape)
        self.assertTrue(mask[20, 60])
        self.assertFalse(mask[50, 150])
        self.assertFalse(mask[95, 150])

    def test_invalid_vertices(self):
        for points in ([[0, 0], [1, 1]], [[0, 0], [0, 0], [1, 1]],
                       [[0, 0], [1, 0], [np.nan, 1]]):
            with self.assertRaises(ValueError):
                curve(points)

    def test_render_preserves_source_and_requires_complete_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.png'
            Image.fromarray(np.full((100, 200), 100, np.uint8)).save(source)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            (root / 'source_manifest.json').write_text(json.dumps({'cases': [
                {'id': '001_case', 'source': str(source), 'source_sha256': digest}]}))
            document = {'coordinate_size_wh': [200, 100], 'target': 'test',
                        'provenance': 'synthetic geometry', 'cases': {'001': {
                            'outer': [[10, 10], [190, 10], [190, 90], [10, 90]],
                            'holes': [[[90, 40], [110, 40], [110, 60], [90, 60]]]}}}
            (root / 'contours.json').write_text(json.dumps(document))
            render(root)
            with Image.open(root / 'masks/001_case.png') as handle:
                self.assertEqual(handle.size, (200, 100))
                self.assertEqual(handle.mode, 'L')
                self.assertEqual(handle.getpixel((100, 50)), 0)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)
            document['cases'] = {}
            (root / 'contours.json').write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, 'every source ID'):
                render(root)


if __name__ == '__main__':
    unittest.main()
