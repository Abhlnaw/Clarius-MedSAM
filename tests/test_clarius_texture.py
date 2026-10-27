import unittest

import numpy as np

from clarius_texture import analyze_texture, automatic_window, localized_window, texture_groups


class TextureTests(unittest.TestCase):
    def test_automatic_window_uses_region_extent_not_canvas(self):
        mask = np.zeros((200, 300), dtype=bool)
        mask[40:120, 60:100] = True
        self.assertEqual(automatic_window(mask), (21, 11))
        padded = np.pad(mask, ((100, 200), (80, 90)))
        self.assertEqual(automatic_window(padded), (21, 11))
        image = np.zeros(mask.shape)
        expected, _ = texture_groups(image, mask, window=localized_window(mask))
        actual, _ = texture_groups(image, mask)
        np.testing.assert_array_equal(actual, expected)

    def test_small_automatic_window_is_valid(self):
        self.assertEqual(automatic_window(np.ones((1, 2), dtype=bool)), (3, 3))

    def test_outside_intensities_do_not_affect_groups(self):
        image = np.zeros((90, 100))
        image[:, 50:] = (np.indices((90, 50)).sum(axis=0) % 2) * 0.8
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
        np.testing.assert_allclose(features, np.tile([0, 1, 1, 1] * 2, (8, 9, 1)), atol=1e-6)

    def test_same_brightness_different_spatial_pattern(self):
        # Both interiors have equal mean and variance; co-occurrence differs.
        image = np.zeros((80, 160))
        image[:, :80] = np.indices((80, 80)).sum(axis=0) % 2
        image[:, 80:] = (np.indices((80, 80))[1] // 4) % 2
        mask = np.ones(image.shape, bool)
        labels, maps = texture_groups(image, mask, window=(15, 15), groups=2)
        self.assertNotEqual(labels[40, 30], labels[40, 120])
        self.assertGreater(abs(float(maps[40, 30, 0] - maps[40, 120, 0])), 1)
        again, _ = texture_groups(image, mask, window=(15, 15), groups=2)
        np.testing.assert_array_equal(labels, again)

    def test_metadata_and_unsupported_pixels(self):
        mask = np.zeros((20, 20), bool)
        mask[2:10, 2:10] = True
        mask[18, 18] = True
        result = analyze_texture(np.zeros(mask.shape), mask)
        self.assertEqual(result["method"], "local_glcm_v2")
        self.assertEqual(len(result["feature_names"]), 8)
        self.assertEqual(result["labels"][18, 18], 0)
        self.assertEqual(result["unsupported_mask_pixels"], 1)
        self.assertEqual(result["actual_groups"], 1)

    def test_isolated_region_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "neighboring pixel pairs"):
            texture_groups(np.zeros((1, 1)), np.ones((1, 1), bool))

    def test_smaller_bounded_windows_for_elongated_frame(self):
        mask = np.ones((1520, 192), bool)
        self.assertEqual(localized_window(mask), (129, 25))
        self.assertLess(localized_window(mask)[0], automatic_window(mask)[0])

    def test_combined_restores_brightness_discrimination(self):
        image = np.full((80, 100), 0.2)
        image[:, 50:] = 0.8
        mask = np.ones(image.shape, bool)
        result = analyze_texture(image, mask, groups=2, method="combined")
        self.assertNotEqual(result["labels"][40, 20], result["labels"][40, 80])
        self.assertEqual(result["features"].shape[-1], 12)
        self.assertEqual(len(result["feature_weights_after_zscore"]), 12)

    def test_brightness_mode_uses_original_moments_and_background_masking(self):
        image = np.random.default_rng(5).random((40, 60))
        mask = np.zeros(image.shape, bool)
        mask[5:35, 10:50] = True
        result = analyze_texture(image, mask, method="brightness")
        self.assertEqual(result["window_hw_pixels"], list(automatic_window(mask)))
        self.assertEqual(result["features"].shape[-1], 4)
        image[~mask] = 1000
        other = analyze_texture(image, mask, method="brightness")
        np.testing.assert_array_equal(result["labels"], other["labels"])
        self.assertNotIn("gray_levels", result)

    def test_empty_mask_rejected(self):
        with self.assertRaises(ValueError):
            texture_groups(np.zeros((5, 5)), np.zeros((5, 5), dtype=bool))


if __name__ == "__main__":
    unittest.main()
