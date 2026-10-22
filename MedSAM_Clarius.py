"""Interactive box annotation of exported B-mode images (not raw RF)."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RectangleSelector
import numpy as np
from skimage import io, transform
import torch
from torch.nn import functional as F

from segment_anything import sam_model_registry
from clarius_texture import automatic_window, texture_groups


def load_rgb(path):
    image = io.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] in (3, 4):
        image = image[..., :3]
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    image = image.astype(np.float32)
    if not np.isfinite(image).all():
        raise ValueError("Image contains non-finite intensities")
    return (image - image.min()) / max(float(np.ptp(image)), 1e-8)


class Viewer:
    def __init__(self, model, device, paths, output):
        self.model, self.device = model, device
        self.paths, self.output = paths, output
        self.index = 0
        self.fig, self.ax = plt.subplots(figsize=(11, 8))
        self.fig.subplots_adjust(bottom=0.23)
        self.buttons = []
        for i, (label, callback) in enumerate([
            ("Undo", self.undo), ("Save", self.save),
            ("Previous", lambda _: self.navigate(-1)),
            ("Next", lambda _: self.navigate(1)),
        ]):
            button = Button(self.fig.add_axes([0.12 + i * 0.19, 0.04, 0.16, 0.05]), label)
            button.on_clicked(callback)
            self.buttons.append(button)
        self.status = self.fig.text(0.12, 0.11, "")
        self.texture_button = Button(self.fig.add_axes([0.12, 0.145, 0.23, 0.04]), "Split latest by texture")
        self.texture_button.on_clicked(self.split_texture)
        self.mask_button = Button(self.fig.add_axes([0.39, 0.145, 0.23, 0.04]), "Show MedSAM masks")
        self.mask_button.on_clicked(self.show_masks)
        self.fig.canvas.mpl_connect("button_press_event", self.pick_texture)
        self.fig.canvas.mpl_connect("close_event", lambda _: self.save())
        self.load()

    @torch.inference_mode()
    def load(self):
        self.image = load_rgb(self.paths[self.index])
        self.height, self.width = self.image.shape[:2]
        resized = transform.resize(self.image, (1024, 1024), order=3,
                                   preserve_range=True, anti_aliasing=True)
        resized = (resized - resized.min()) / max(float(np.ptp(resized)), 1e-8)
        tensor = torch.from_numpy(resized.astype(np.float32)).permute(2, 0, 1)[None]
        print(f"Encoding {self.paths[self.index]} ...", flush=True)
        self.embedding = self.model.image_encoder(tensor.to(self.device))
        self.regions = []
        self.texture = None
        self.refinements = {}
        saved_masks = self.output / f"{self.paths[self.index].name}.regions.npz"
        saved_json = self.output / f"{self.paths[self.index].name}.regions.json"
        if saved_masks.exists() or saved_json.exists():
            metadata = json.loads(saved_json.read_text())
            if metadata["source"] != str(self.paths[self.index].resolve()):
                raise ValueError("Output contains annotations for a different source. Choose another --output folder.")
            with np.load(saved_masks, allow_pickle=False) as saved:
                masks = saved["masks"]
                if masks.shape != (len(metadata["regions"]), self.height, self.width):
                    raise ValueError("Saved annotations do not match this image")
                self.regions = [(record["box_xyxy"], mask.copy())
                                for record, mask in zip(metadata["regions"], masks)]
                for i, record in enumerate(metadata["regions"]):
                    if "texture" in record:
                        self.refinements[i] = dict(record["texture"])
                        for key in ("labels", "selected_mask", "features"):
                            self.refinements[i][key] = saved[f"texture_{i}_{key}"].copy()
        if hasattr(self, "selector"):
            self.selector.disconnect_events()
        self.ax.clear()
        self.ax.imshow(self.image, aspect="auto")
        self.overlay = self.ax.imshow(np.zeros((self.height, self.width, 4)), aspect="auto")
        self.ax.set_title(f"{self.paths[self.index].name} — drag a tight box around one region")
        self.ax.set_xlim(-0.5, self.width - 0.5)
        self.ax.set_ylim(self.height - 0.5, -0.5)
        self.selector = RectangleSelector(self.ax, self.select, useblit=False,
                                          button=[1], minspanx=3, minspany=3,
                                          spancoords="pixels", interactive=False)
        self.status.set_text("Drag to segment. Save writes masks; navigation and closing also save.")
        self.redraw()

    @torch.inference_mode()
    def select(self, press, release):
        if None in (press.xdata, press.ydata, release.xdata, release.ydata):
            return
        xs = np.clip(sorted([press.xdata, release.xdata]), 0, self.width)
        ys = np.clip(sorted([press.ydata, release.ydata]), 0, self.height)
        box = np.array([xs[0], ys[0], xs[1], ys[1]], dtype=np.float32)
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            return
        try:
            scaled = box / [self.width, self.height, self.width, self.height] * 1024
            prompts = torch.as_tensor(scaled, dtype=torch.float32, device=self.device)[None, None]
            sparse, dense = self.model.prompt_encoder(points=None, boxes=prompts, masks=None)
            logits, _ = self.model.mask_decoder(
                image_embeddings=self.embedding, image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                multimask_output=False)
            probabilities = F.interpolate(torch.sigmoid(logits), (self.height, self.width),
                                          mode="bilinear", align_corners=False)
            mask = probabilities[0, 0].cpu().numpy() > 0.5
            if not mask.any():
                self.status.set_text("Empty prediction. Try a different box.")
            else:
                self.regions.append((box.tolist(), mask))
                self.texture = None
                self.status.set_text(f"Region {len(self.regions)} added. Review its boundary; Undo removes it.")
            self.redraw()
        except Exception as error:
            self.status.set_text(f"Segmentation failed: {error}")
            self.fig.canvas.draw_idle()

    def redraw(self):
        rgba = np.zeros((self.height, self.width, 4))
        for i, (_, mask) in enumerate(self.regions):
            rgba[mask] = plt.get_cmap("tab10")(i % 10)
            rgba[mask, 3] = 0.4
        if self.texture is not None:
            labels = self.texture["labels"]
            rgba[:] = 0
            for group in range(1, int(labels.max()) + 1):
                rgba[labels == group] = plt.get_cmap("tab10")(group - 1)
                rgba[labels == group, 3] = 0.45
            selected = self.texture.get("selected_group")
            if selected is not None:
                rgba[(labels > 0) & (labels != selected), 3] = 0.08
        self.overlay.set_data(rgba)
        self.fig.canvas.draw_idle()

    def undo(self, _=None):
        if self.regions:
            self.regions.pop()
            self.refinements.pop(len(self.regions), None)
            self.texture = None
            self.status.set_text("Last region removed. Save to update the annotation bundle.")
            self.redraw()

    def save(self, _=None):
        # Separate boolean masks preserve overlaps; JSON supplies region IDs and boxes.
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.paths[self.index]
        masks = np.stack([m for _, m in self.regions]) if self.regions else np.zeros(
            (0, self.height, self.width), dtype=bool)
        arrays = {"masks": masks}
        for i, refinement in self.refinements.items():
            for key in ("labels", "selected_mask", "features"):
                arrays[f"texture_{i}_{key}"] = refinement[key]
        np.savez_compressed(self.output / f"{path.name}.regions.npz", **arrays)
        gray = self.image.mean(axis=-1)
        records = []
        for i, (box, mask) in enumerate(self.regions):
            values = gray[mask]
            records.append({"id": i + 1, "label": f"region_{i + 1}", "box_xyxy": box,
                            "area_pixels": int(mask.sum()),
                            "mean_display_intensity": float(values.mean()),
                            "std_display_intensity": float(values.std())})
            if i in self.refinements:
                records[-1]["texture"] = {key: value for key, value in self.refinements[i].items()
                                           if key not in ("labels", "selected_mask", "features")}
        metadata = {"source": str(path.resolve()), "shape_hw": [self.height, self.width],
                    "note": "Display intensities normalized to 0..1; no physical calibration or diagnostic labels.",
                    "regions": records}
        self.output.joinpath(f"{path.name}.regions.json").write_text(json.dumps(metadata, indent=2))
        plt.imsave(self.output / f"{path.name}.overlay.png",
                   self.image * (1 - self.overlay.get_array()[..., 3:4])
                   + self.overlay.get_array()[..., :3] * self.overlay.get_array()[..., 3:4])
        self.status.set_text(f"Saved {len(records)} regions to {self.output}")
        self.fig.canvas.draw_idle()

    def navigate(self, step):
        index = self.index + step
        if 0 <= index < len(self.paths):
            self.save()
            self.index = index
            self.load()

    def split_texture(self, _=None):
        if not self.regions:
            self.status.set_text("Draw a MedSAM box first.")
            self.fig.canvas.draw_idle()
            return
        window = list(automatic_window(self.regions[-1][1]))
        labels, features = texture_groups(self.image.mean(axis=-1), self.regions[-1][1], window)
        self.texture = {"labels": labels, "features": features, "window_hw_pixels": window,
                        "window_selection": "automatic_quarter_mask_extent",
                        "feature_names": ["half_window_mean", "half_window_std", "full_window_mean", "full_window_std"],
                        "selected_group": None, "selected_mask": np.zeros(labels.shape, dtype=bool)}
        self.refinements[len(self.regions) - 1] = self.texture
        self.selector.set_active(False)
        self.status.set_text("Texture scales chosen automatically. Click a group to keep it; Show MedSAM masks to draw.")
        self.redraw()

    def pick_texture(self, event):
        if self.texture is None or event.inaxes != self.ax or event.button != 1:
            return
        if self.fig.canvas.toolbar and self.fig.canvas.toolbar.mode:
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        group = int(self.texture["labels"][y, x])
        if group:
            self.texture["selected_group"] = group
            self.texture["selected_mask"] = self.texture["labels"] == group
            self.status.set_text(f"Keeping texture group {group}: {int(self.texture['selected_mask'].sum())} pixels. Save to export.")
            self.redraw()

    def show_masks(self, _=None):
        self.texture = None
        self.selector.set_active(True)
        self.status.set_text("MedSAM masks shown. Drag another box, or split the latest region by texture.")
        self.redraw()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", type=Path, required=True, help="Image or folder of exported frames")
    parser.add_argument("--checkpoint", type=Path, default=Path("medsam_vit_b.pth"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-o", "--output", type=Path, default=Path("clarius_annotations"))
    args = parser.parse_args()
    paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in
                   {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}) if args.input.is_dir() else [args.input]
    if not paths or any(not p.is_file() for p in paths):
        parser.error("No input images found")
    model = sam_model_registry["vit_b"](checkpoint=str(args.checkpoint)).to(args.device).eval()
    viewer = Viewer(model, args.device, paths, args.output)
    plt.show()


if __name__ == "__main__":
    main()
