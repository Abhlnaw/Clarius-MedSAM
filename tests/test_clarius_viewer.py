"""Headless viewer integration; model loading/CUDA are tested separately."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import torch

from MedSAM_Clarius import Viewer


class StubModel:
    def image_encoder(self, tensor):
        return torch.zeros((1, 1, 1, 1))


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "frame.png"
        image = np.random.default_rng(11).integers(0, 256, (40, 60), dtype=np.uint8)
        io.imsave(self.path, image, check_contrast=False)
        self.viewer = Viewer(StubModel(), "cpu", [self.path], self.root / "output")
        self.addCleanup(plt.close, self.viewer.fig)
        mask = np.zeros(image.shape, bool)
        mask[5:35, 10:50] = True
        self.viewer.regions.append(([10, 5, 50, 35], mask))

    def test_glcm_click_save_reload_and_undo(self):
        viewer = self.viewer
        viewer.refinement_mode.set_active(2)
        viewer.split_texture()
        self.assertEqual(viewer.texture["features"].shape, (40, 60, 8))
        self.assertFalse(viewer.selector.active)
        group = int(viewer.texture["labels"][20, 30])
        viewer.pick_texture(SimpleNamespace(inaxes=viewer.ax, button=1, xdata=30, ydata=20))
        self.assertEqual(viewer.texture["selected_group"], group)
        self.assertFalse(viewer.texture["selected_mask"][~viewer.regions[-1][1]].any())
        expected = viewer.texture["selected_mask"].copy()
        viewer.save()
        metadata = json.loads((viewer.output / "frame.png.regions.json").read_text())
        self.assertEqual(metadata["regions"][0]["texture"]["method"], "local_glcm_v2")
        self.assertEqual(len(metadata["regions"][0]["texture"]["feature_names"]), 8)
        with np.load(viewer.output / "frame.png.regions.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(data["texture_0_selected_mask"], expected)
        viewer.load()
        np.testing.assert_array_equal(viewer.refinements[0]["selected_mask"], expected)
        self.assertEqual(viewer.refinements[0]["method"], "local_glcm_v2")
        viewer.undo()
        self.assertFalse(viewer.refinements)
        viewer.save()
        with np.load(viewer.output / "frame.png.regions.npz", allow_pickle=False) as data:
            self.assertNotIn("texture_0_labels", data.files)

    def test_existing_four_feature_annotations_remain_readable(self):
        self.viewer.refinement_mode.set_active(0)
        mask = self.viewer.regions[0][1]
        self.viewer.refinements[0] = {
            "labels": mask.astype(np.uint8), "features": np.zeros((*mask.shape, 4)),
            "selected_mask": mask.copy(), "selected_group": 1,
            "feature_names": ["half_window_mean", "half_window_std", "full_window_mean", "full_window_std"]}
        self.viewer.save()
        self.viewer.load()
        self.assertEqual(self.viewer.refinements[0]["features"].shape[-1], 4)
        self.viewer.split_texture()
        self.assertEqual(self.viewer.refinements[0]["features"].shape[-1], 12)

    def test_switch_methods_recomputes_without_changing_parent(self):
        self.viewer.refinement_mode.set_active(0)
        parent = self.viewer.regions[0][1].copy()
        self.viewer.split_texture()
        self.assertEqual(self.viewer.texture["method"], "local_hybrid_v1")
        self.viewer.refinement_mode.set_active(1)
        self.assertEqual(self.viewer.texture["method"], "local_intensity_v1")
        self.assertEqual(self.viewer.texture["features"].shape[-1], 4)
        self.viewer.refinement_mode.set_active(2)
        self.assertEqual(self.viewer.texture["features"].shape[-1], 8)
        np.testing.assert_array_equal(self.viewer.regions[0][1], parent)

    def test_automatic_mode_needs_no_click_and_saves_original_source_labels(self):
        viewer = self.viewer
        self.assertEqual(viewer.refinement_mode.value_selected, "Low / High variation")
        viewer.split_texture()
        self.assertEqual(viewer.texture["method"], "automatic_tissue_v1")
        self.assertEqual(viewer.texture["source_method"], "local_hybrid_v1")
        self.assertTrue(viewer.texture["selected_mask"].any())
        self.assertFalse(viewer.texture["selected_mask"][~viewer.regions[0][1]].any())
        viewer.save()
        expected = viewer.texture["selected_mask"].copy()
        viewer.load()
        np.testing.assert_array_equal(viewer.refinements[0]["selected_mask"], expected)
        self.assertIn("source_labels", viewer.refinements[0])
        self.assertFalse(hasattr(viewer, "selection_tool"))

    def test_automatic_uses_last_source_view_and_cached_glcm_maps(self):
        import clarius_texture
        viewer = self.viewer
        viewer.refinement_mode.set_active(0)
        with patch("clarius_texture.local_glcm_features", wraps=clarius_texture.local_glcm_features) as calculate:
            viewer.split_texture()
            viewer.refinement_mode.set_active(3)
            self.assertEqual(viewer.texture["source_method"], "local_hybrid_v1")
            viewer.refinement_mode.set_active(2)
            viewer.refinement_mode.set_active(3)
            self.assertEqual(viewer.texture["source_method"], "local_glcm_v2")
            self.assertEqual(calculate.call_count, 1)

    def test_click_keeps_local_patch_instead_of_disconnected_matches(self):
        labels = np.zeros((40, 60), dtype=np.uint8)
        labels[10:15, 10:15] = 1
        labels[25:30, 40:45] = 1
        self.viewer.texture = {"labels": labels, "selected_group": None,
                               "selected_mask": np.zeros_like(labels, dtype=bool)}
        event = SimpleNamespace(inaxes=self.viewer.ax, button=1, xdata=12, ydata=12)
        self.viewer.pick_texture(event)
        self.assertEqual(int(self.viewer.texture["selected_mask"].sum()), 25)
        self.assertFalse(self.viewer.texture["selected_mask"][27, 42])
        self.viewer.connected_only.set_active(0)
        self.viewer.pick_texture(event)
        self.assertEqual(int(self.viewer.texture["selected_mask"].sum()), 50)

    def test_unsupported_region_reports_error_without_creating_refinement(self):
        mask = np.zeros((40, 60), bool)
        mask[20, 30] = True
        self.viewer.regions = [([30, 20, 31, 21], mask)]
        self.viewer.split_texture()
        self.assertIsNone(self.viewer.texture)
        self.assertIn("no valid neighboring pixel pairs", self.viewer.status.get_text())
        self.assertFalse(self.viewer.refinements)

    def test_tall_rf_overlay_uses_viewer_proportions_without_resizing_masks(self):
        viewer = self.viewer
        viewer.height, viewer.width = 1520, 192
        viewer.image = np.full((1520, 192, 3), 0.5, dtype=np.float32)
        mask = np.zeros((1520, 192), dtype=bool)
        mask[200:600, 20:100] = True
        viewer.regions = [([20, 200, 100, 600], mask)]
        viewer.redraw()
        # A zoom must not crop the saved full-frame overlay.
        viewer.ax.set_xlim(20, 100)
        viewer.ax.set_ylim(600, 200)
        viewer.save()
        overlay = io.imread(viewer.output / "frame.png.overlay.png")
        bounds = viewer.ax.get_window_extent()
        self.assertAlmostEqual(overlay.shape[1] / overlay.shape[0], bounds.width / bounds.height, places=2)
        self.assertGreater(overlay.shape[1], overlay.shape[0])
        with np.load(viewer.output / "frame.png.regions.npz", allow_pickle=False) as saved:
            self.assertEqual(saved["masks"].shape, (1, 1520, 192))
            np.testing.assert_array_equal(saved["masks"][0], mask)
        metadata = json.loads((viewer.output / "frame.png.regions.json").read_text())
        self.assertEqual(metadata["shape_hw"], [1520, 192])
        self.assertEqual(metadata["overlay_shape_hw"], list(overlay.shape[:2]))
        # Unselected full-frame top/bottom remain gray, even when zoomed in.
        np.testing.assert_allclose(overlay[0, 0, :3], [127] * 3, atol=1)
        np.testing.assert_allclose(overlay[-1, -1, :3], [127] * 3, atol=1)


if __name__ == "__main__":
    unittest.main()
