import unittest

import numpy as np

from clarius_auto import automatic_tissue_region


def tissue_case():
    gray = np.full((250, 210), 0.5)
    parent = np.zeros(gray.shape, bool)
    parent[25:145, 25:185] = True
    parent[145:205, 98:106] = True  # thin unwanted lower connection
    parent[205:230, 50:170] = True  # unwanted lower lobe
    labels = parent.astype(np.uint8)
    # Same tissue, split into a different appearance group by the source.
    labels[35:55, 50:80] = 2
    labels[70:95, 30:48] = 2
    y, x = np.ogrid[:250, :210]
    cavity = (x - 110) ** 2 + (y - 90) ** 2 < 12 ** 2
    labels[cavity] = 3
    gray[cavity] = 0.15
    features = np.stack([gray, np.full_like(gray, 0.05)], axis=-1)
    return gray, parent, {"labels": labels, "features": features}, cavity


class AutomaticTests(unittest.TestCase):
    def test_can_merge_compatible_tissue_across_different_texture_labels(self):
        gray, parent, proposal, _ = tissue_case()
        proposal["features"][35:55, 50:80] = 10  # intentionally incompatible source descriptor
        selected, _ = automatic_tissue_region(gray, parent, proposal)
        self.assertGreater(selected[35:55, 50:80].mean(), 0.9)

    def test_large_mixed_population_does_not_outscore_coherent_tissue_seed(self):
        parent = np.zeros((220, 200), bool)
        parent[10:85, 20:180] = True
        parent[105:215, 5:195] = True
        gray = np.full(parent.shape, 0.55)
        gray[105:215] = np.linspace(0.1, 0.95, 200)[None, :]
        labels = np.zeros(parent.shape, np.uint8)
        labels[105:215, 5:195] = 1
        labels[10:85, 20:180] = 2
        features = np.stack([gray, gray], axis=-1)
        selected, diagnostics = automatic_tissue_region(gray, parent, {"labels": labels, "features": features})
        self.assertEqual(diagnostics["source_group"], 2)
        self.assertGreater(selected[10:85, 20:180].mean(), 0.9)
        self.assertFalse(selected[105:].any())

    def test_recovers_compatible_tissue_preserves_cavity_and_removes_lower_lobe(self):
        gray, parent, proposal, cavity = tissue_case()
        selected, diagnostics = automatic_tissue_region(gray, parent, proposal)
        self.assertGreater(selected[35:55, 50:80].mean(), 0.9)
        self.assertGreater(selected[70:95, 30:48].mean(), 0.9)
        self.assertFalse(selected[90, 110])
        self.assertLess(selected[cavity].mean(), 0.1)
        self.assertFalse(selected[205:].any())
        self.assertFalse(selected[~parent].any())
        self.assertEqual(diagnostics["source_group"], 1)

    def test_does_not_assume_target_is_always_at_top(self):
        gray, parent, proposal, _ = tissue_case()
        rotated = {key: np.flip(value, axis=0) for key, value in proposal.items()}
        selected, _ = automatic_tissue_region(np.flip(gray, 0), np.flip(parent, 0), rotated)
        self.assertGreater(selected[120:220].sum(), selected[:50].sum())
        self.assertFalse(selected[:20].any())

    def test_parent_exclusions_are_never_filled(self):
        gray, parent, proposal, _ = tissue_case()
        parent[100:110, 60:80] = False
        selected, _ = automatic_tissue_region(gray, parent, proposal)
        self.assertFalse(selected[100:110, 60:80].any())

    def test_constant_region_is_finite_and_deterministic(self):
        gray = np.full((30, 40), 0.5)
        parent = np.ones(gray.shape, bool)
        source = {"labels": parent.astype(np.uint8), "features": np.zeros((*gray.shape, 4))}
        first, diagnostics = automatic_tissue_region(gray, parent, source)
        second, _ = automatic_tissue_region(gray, parent, source)
        np.testing.assert_array_equal(first, second)
        self.assertGreater(first.mean(), 0.95)
        self.assertTrue(np.isfinite(diagnostics["texture_distance_threshold"]))


if __name__ == "__main__":
    unittest.main()
