import unittest

import numpy as np

from clarius_texture import analyze_texture, automatic_window, largest_labeled_region, localized_window, merge_variation_groups, texture_groups


class TextureTests(unittest.TestCase):
    def test_glszm_proposals_and_automatic_mode_keep_raw_maps_and_metadata(self):
        gray=np.random.default_rng(12).random((22,28)); mask=np.zeros(gray.shape,bool);mask[2:20,3:25]=True
        pure=analyze_texture(gray,mask,window=(9,7),method='glszm',glszm_step=(2,2))
        self.assertEqual(pure['features'].shape,(22,28,16))
        self.assertEqual(pure['zone_connectivity'],8)
        self.assertNotIn('offsets_dy_dx',pure)
        self.assertFalse(pure['features'][~mask].any())
        combined=analyze_texture(gray,mask,window=(9,7),method='combined_glszm',glszm_step=(2,2),
            glszm_data=(pure['features'],pure['labels']>0))
        np.testing.assert_array_equal(combined['features'][...,-16:],pure['features'])
        self.assertEqual(combined['features'].shape[-1],20)
        self.assertEqual(combined['feature_transform_before_clustering'],'log1p_glszm_channels_then_zscore')
        auto=analyze_texture(gray,mask,method='variation',auto_source='glszm',source_result=pure)
        info=auto['automatic_refinement']
        self.assertTrue(info['distance_gate_applied'])
        self.assertEqual(info['distance_gate_family'],'glszm')
        self.assertFalse(info['correlation_gate_applied'])
        self.assertEqual(info['distance_gate_quantile'],.985)
        self.assertFalse(auto['selected_mask'][~mask].any())

    def test_merges_two_low_variation_groups_despite_different_brightness(self):
        labels = np.repeat(np.arange(1, 4), 100).reshape(10, 30)
        gray = np.zeros(labels.shape)
        for group, mean, amplitude in ((1, 0.2, 0.02), (2, 0.8, 0.025), (3, 0.5, 0.2)):
            gray[labels == group] = mean + amplitude * np.tile([-1, 1], 50)
        merged, mapping, _, threshold = merge_variation_groups(gray, labels)
        self.assertEqual(mapping, {1: 1, 2: 1, 3: 2})
        self.assertIsNotNone(threshold)
        np.testing.assert_array_equal(merged == 1, (labels == 1) | (labels == 2))

    def test_similar_variation_is_not_forced_into_two_classes(self):
        labels = np.tile([1, 1, 2, 2, 3, 3], (5, 1))
        gray = np.tile([0.2, 0.22, 0.5, 0.52, 0.8, 0.82], (5, 1))
        merged, _, _, threshold = merge_variation_groups(gray, labels)
        self.assertIsNone(threshold)
        self.assertTrue(np.all(merged == 1))

    def test_variation_result_preserves_parent_coverage_and_metadata(self):
        gray = np.random.default_rng(99).random((40, 50))
        mask = np.zeros(gray.shape, bool)
        mask[5:35, 5:45] = True
        result = analyze_texture(gray, mask, method="variation")
        self.assertEqual(result["method"], "automatic_tissue_v1")
        self.assertLessEqual(result["actual_groups"], 2)
        self.assertEqual(result["gray_levels"], 8)
        self.assertFalse(result["selected_mask"][~mask].any())
        np.testing.assert_array_equal(result["labels"] > 0, result["selected_mask"])
        self.assertEqual(result["actual_groups"], 1)
        self.assertIn("automatic_refinement", result)

    def test_largest_region_removes_detached_islands_and_other_class(self):
        labels = np.zeros((12, 18), np.uint8)
        labels[1:6, 1:7] = 1
        labels[9:11, 1:3] = 1
        labels[1:5, 10:15] = 2
        retained, group = largest_labeled_region(labels)
        self.assertEqual(group, 1)
        self.assertEqual(np.count_nonzero(retained), 30)
        self.assertFalse(retained[9:11].any())
        self.assertFalse(np.any(retained == 2))

    def test_largest_region_can_be_high_variation_even_if_low_total_is_larger(self):
        labels = np.zeros((15, 20), np.uint8)
        labels[0:4, 0:4] = 1
        labels[10:14, 0:4] = 1
        labels[1:6, 10:15] = 2
        retained, group = largest_labeled_region(labels)
        self.assertEqual(group, 2)
        self.assertEqual(np.count_nonzero(retained), 25)
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
