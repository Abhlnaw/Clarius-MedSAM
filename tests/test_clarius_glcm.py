import unittest

import numpy as np
from skimage.feature import graycomatrix, graycoprops

from clarius_glcm import (DEFAULT_OFFSETS, FEATURE_NAMES, PROPERTIES,
                          analysis_windows, local_glcm_features)


def reference_patch(gray, mask, center, window, levels, offsets):
    """Independent explicit pair enumeration, including holes in a mask."""
    y, x = center
    h, w = window
    y0, y1 = max(0, y - h // 2), min(len(gray), y + h // 2 + 1)
    x0, x1 = max(0, x - w // 2), min(gray.shape[1], x + w // 2 + 1)
    quantized = np.minimum((gray * levels).astype(int), levels - 1)
    features = []
    for dy, dx in offsets:
        counts = np.zeros((levels, levels))
        for row in range(y0, y1):
            for col in range(x0, x1):
                if (y0 <= row + dy < y1 and x0 <= col + dx < x1
                        and mask[row, col] and mask[row + dy, col + dx]):
                    counts[quantized[row, col], quantized[row + dy, col + dx]] += 1
        if counts.sum():
            symmetric = (counts + counts.T)[..., None, None]
            features.append([float(graycoprops(symmetric, p)[0, 0]) for p in PROPERTIES])
    return np.mean(features, axis=0) if features else np.zeros(4)


class GlcmTests(unittest.TestCase):
    def test_matches_skimage_unmasked_patch(self):
        gray = np.random.default_rng(7).integers(0, 4, (11, 13)) / 3
        mask = np.ones(gray.shape, dtype=bool)
        maps, support = local_glcm_features(gray, mask, (7, 9), levels=4)
        for k, (h, w) in enumerate(analysis_windows((7, 9))):
            patch = gray[5 - h // 2:6 + h // 2, 6 - w // 2:7 + w // 2]
            quantized = np.minimum((patch * 4).astype(np.uint8), 3)
            matrix = graycomatrix(quantized, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                  levels=4, symmetric=True, normed=True)
            expected = [graycoprops(matrix, prop).mean() for prop in PROPERTIES]
            np.testing.assert_allclose(maps[5, 6, k * 4:k * 4 + 4], expected, atol=1e-6)
        self.assertTrue(support.all())

    def test_holes_edges_and_all_offset_signs_match_enumeration(self):
        gray = np.random.default_rng(12).random((9, 12))
        mask = np.ones(gray.shape, dtype=bool)
        mask[2:5, 4:8] = False
        mask[0, :3] = False
        offsets = DEFAULT_OFFSETS + ((-2, 1), (0, -2))
        maps, _ = local_glcm_features(gray, mask, (7, 5), levels=8, offsets=offsets)
        for point in ((0, 5), (8, 11), (5, 4), (1, 3)):
            expected = np.concatenate([reference_patch(gray, mask, point, window, 8, offsets)
                                       for window in analysis_windows((7, 5))])
            np.testing.assert_allclose(maps[point], expected, atol=2e-6)

    def test_constant_has_known_features(self):
        maps, support = local_glcm_features(np.full((7, 9), 0.4), np.ones((7, 9), bool), (9, 11))
        np.testing.assert_allclose(maps[support], np.tile([0, 1, 1, 1] * 2, (63, 1)))

    def test_checkerboard_horizontal_contrast(self):
        gray = (np.indices((9, 9)).sum(axis=0) % 2).astype(float)
        maps, _ = local_glcm_features(gray, np.ones_like(gray, bool), (7, 7),
                                      levels=2, offsets=((0, 1),))
        np.testing.assert_allclose(maps[4, 4, :4], [1, 0.5, np.sqrt(0.5), -1], atol=1e-6)

    def test_background_is_excluded_even_if_nonfinite(self):
        gray = np.random.default_rng(2).random((15, 20))
        mask = np.zeros(gray.shape, bool)
        mask[3:12, 4:17] = True
        expected, _ = local_glcm_features(gray, mask, (7, 5))
        gray[~mask] = np.nan
        actual, _ = local_glcm_features(gray, mask, (7, 5))
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(np.all(actual[~mask] == 0))

    def test_isolated_pixels_are_unsupported(self):
        mask = np.eye(5, dtype=bool)
        maps, support = local_glcm_features(np.zeros((5, 5)), mask, (5, 5), offsets=((0, 1),))
        self.assertFalse(support.any())
        self.assertFalse(maps.any())

    def test_invalid_input(self):
        mask = np.ones((5, 5), bool)
        for kwargs in ({"levels": 1}, {"offsets": ((0, 0),)}, {"window": (2, 5)}):
            options = {"window": (5, 5), **kwargs}
            with self.assertRaises(ValueError):
                local_glcm_features(np.zeros((5, 5)), mask, **options)
        for value in (-0.1, 1.1, np.nan):
            with self.assertRaises(ValueError):
                local_glcm_features(np.full((5, 5), value), mask, (5, 5))
        self.assertEqual(len(FEATURE_NAMES), 8)


if __name__ == "__main__":
    unittest.main()
