"""Check annotation conversion, hole preservation and protection of user edits."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw

TOOLS = Path(__file__).resolve().parents[1] / 'tools'


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(TOOLS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


outlines = load_tool('review_drawn_outlines')
packaging = load_tool('package_annotation_masks')
drafts = load_tool('draft_annotation_masks')


class AnnotationToolsTests(unittest.TestCase):
    def test_drawn_outline_preserves_lumen_and_input(self):
        image = Image.new('RGB', (240, 160), (100, 100, 100))
        draw = ImageDraw.Draw(image)
        draw.rectangle((15, 30, 220, 130), outline=(235, 30, 40), width=4)
        draw.ellipse((90, 60, 120, 90), outline=(235, 30, 40), width=4)
        original = np.asarray(image).copy()
        mask, _, _ = outlines.convert(image, (60, 60))
        self.assertTrue(mask[60, 60])
        self.assertFalse(mask[75, 105])
        self.assertFalse(mask[10, 10])
        np.testing.assert_array_equal(np.asarray(image), original)

    def test_open_contour_can_meet_image_edge(self):
        image = Image.new('RGB', (240, 160), (100, 100, 100))
        ImageDraw.Draw(image).line([(233, 30), (20, 30), (20, 130), (239, 130)],
                                  fill=(235, 30, 40), width=4)
        mask, bridges, _ = outlines.convert(image, (60, 60))
        self.assertTrue(mask[60, 60])
        self.assertFalse(mask[10, 10])
        self.assertTrue(bridges)

    def test_packaging_refuses_to_overwrite_edited_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'candidate'
            target = Path(directory) / 'reviewed'
            source.write_bytes(b'candidate')
            packaging.copy_unchanged(source, target)
            packaging.copy_unchanged(source, target)
            target.write_bytes(b'human correction')
            with self.assertRaises(ValueError):
                packaging.copy_unchanged(source, target)
            self.assertEqual(target.read_bytes(), b'human correction')

    def test_raster_keeps_holes_and_coordinate_order(self):
        mask = drafts.raster([
            np.array([[10, 20], [80, 20], [80, 70], [10, 70]]),
            np.array([[30, 30], [50, 30], [50, 50], [30, 50]])], shape=(100, 120))
        self.assertEqual(mask.shape, (100, 120))
        self.assertTrue(mask[60, 60])
        self.assertFalse(mask[40, 40])
        self.assertFalse(mask[90, 90])


if __name__ == '__main__':
    unittest.main()
