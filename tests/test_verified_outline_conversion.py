"""Regression checks for rasterizing supplied red reference outlines."""
import unittest

import numpy as np
from PIL import Image, ImageDraw

from tools.evaluate_verified_outlines import convert_outline


class VerifiedOutlineTests(unittest.TestCase):
    def drawing(self, hole=True):
        image = Image.new('RGB', (200, 120), (45, 45, 45))
        pen = ImageDraw.Draw(image)
        pen.rectangle((20, 20, 180, 100), outline=(235, 35, 45), width=3)
        if hole:
            pen.ellipse((85, 45, 115, 75), outline=(235, 35, 45), width=3)
        return image

    def test_outer_region_retains_internal_exclusion(self):
        image = self.drawing()
        original = np.array(image)
        mask, _, audit = convert_outline(image)
        self.assertTrue(mask[35, 50])
        self.assertFalse(mask[60, 100])
        self.assertFalse(mask[5, 5])
        self.assertEqual(audit['holes'], 1)
        np.testing.assert_array_equal(np.asarray(image), original)

    def test_reject_unmarked_image(self):
        with self.assertRaisesRegex(ValueError, 'No red outline'):
            convert_outline(Image.new('RGB', (200, 120), (45, 45, 45)))

    def test_small_endpoint_seam_is_logged(self):
        image = self.drawing(False)
        # Erase the complete three-pixel stroke across the seam.
        ImageDraw.Draw(image).rectangle((178, 55, 180, 59), fill=(45, 45, 45))
        mask, _, audit = convert_outline(image)
        self.assertTrue(mask[60, 50])
        self.assertTrue(audit['bridges'])
        self.assertLessEqual(audit['bridges'][0]['distance_pixels'], 10)

    def test_long_gap_needs_explicit_straight_closure(self):
        image = self.drawing(False)
        ImageDraw.Draw(image).rectangle((178, 40, 180, 80), fill=(45, 45, 45))
        with self.assertRaisesRegex(ValueError, 'does not enclose'):
            convert_outline(image)
        mask, _, audit = convert_outline(image, connect_open_ends=True)
        self.assertTrue(mask[60, 50])
        self.assertEqual(len(audit['bridges']), 1)
        self.assertEqual(audit['bridges'][0]['reason'], 'user-authorized straight closure')


if __name__ == '__main__':
    unittest.main()
