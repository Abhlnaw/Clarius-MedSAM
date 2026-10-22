import unittest

import numpy as np

from clarius_texture import automatic_window, texture_groups


class TextureTests(unittest.TestCase):
    def test_automatic_window_uses_region_extent_not_canvas(self):
        mask = np.zeros((200, 300), dtype=bool)
        mask[40:120, 60:100] = True
        self.assertEqual(automatic_window(mask), (21, 11))
        padded = np.pad(mask, ((100, 200), (80, 90)))
        self.assertEqual(automatic_window(padded), (21, 11))
        image = np.zeros(mask.shape)
        expected, _ = texture_groups(image, mask, window=(21, 11))
        actual, _ = texture_groups(image, mask)
        np.testing.assert_array_equal(actual, expected)

    def test_small_automatic_window_is_valid(self):
        self.assertEqual(automatic_window(np.ones((1, 2), dtype=bool)), (3, 3))

    def test_outside_intensities_do_not_affect_groups(self):
        image = np.zeros((90, 100))
        image[:, 50:] = 0.8
        mask = np.zeros_like(image, dtype=bool)
        mask[10:80, 10:90] = True
        labels, features = texture_groups(image, mask, (21, 15))
        image[~mask] = 1000
        other_labels, other_features = texture_groups(image, mask, (21, 15))
        np.testing.assert_array_equal(labels, other_labels)
        np.testing.assert_allclose(features, other_features)
        self.assertTrue(np.all(labels[~mask] == 0))
        self.assertNotEqual(labels[40, 25], labels[40, 75])

    def test_constant_region_and_large_windows(self):
        mask = np.ones((8, 9), dtype=bool)
        labels, features = texture_groups(np.full(mask.shape, 0.5), mask, (301, 151))
        np.testing.assert_array_equal(labels, np.ones(mask.shape))
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_allclose(features[..., 1::2], 0, atol=1e-6)

    def test_empty_mask_rejected(self):
        with self.assertRaises(ValueError):
            texture_groups(np.zeros((5, 5)), np.zeros((5, 5), dtype=bool))


if __name__ == "__main__":
    unittest.main()
