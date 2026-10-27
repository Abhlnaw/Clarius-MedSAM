"""Headless viewer integration; model loading/CUDA are tested separately."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

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
        self.assertEqual(metadata["regions"][0]["texture"]["method"], "local_glcm_v1")
        self.assertEqual(len(metadata["regions"][0]["texture"]["feature_names"]), 8)
        with np.load(viewer.output / "frame.png.regions.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(data["texture_0_selected_mask"], expected)
        viewer.load()
        np.testing.assert_array_equal(viewer.refinements[0]["selected_mask"], expected)
        self.assertEqual(viewer.refinements[0]["method"], "local_glcm_v1")
        viewer.undo()
        self.assertFalse(viewer.refinements)
        viewer.save()
        with np.load(viewer.output / "frame.png.regions.npz", allow_pickle=False) as data:
            self.assertNotIn("texture_0_labels", data.files)

    def test_existing_four_feature_annotations_remain_readable(self):
        mask = self.viewer.regions[0][1]
        self.viewer.refinements[0] = {
            "labels": mask.astype(np.uint8), "features": np.zeros((*mask.shape, 4)),
            "selected_mask": mask.copy(), "selected_group": 1,
            "feature_names": ["half_window_mean", "half_window_std", "full_window_mean", "full_window_std"]}
        self.viewer.save()
        self.viewer.load()
        self.assertEqual(self.viewer.refinements[0]["features"].shape[-1], 4)
        self.viewer.split_texture()
        self.assertEqual(self.viewer.refinements[0]["features"].shape[-1], 8)

    def test_unsupported_region_reports_error_without_creating_refinement(self):
        mask = np.zeros((40, 60), bool)
        mask[20, 30] = True
        self.viewer.regions = [([30, 20, 31, 21], mask)]
        self.viewer.split_texture()
        self.assertIsNone(self.viewer.texture)
        self.assertIn("no valid neighboring pixel pairs", self.viewer.status.get_text())
        self.assertFalse(self.viewer.refinements)


if __name__ == "__main__":
    unittest.main()
