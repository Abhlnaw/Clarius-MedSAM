"""Interactive box annotation of exported B-mode images (not raw RF)."""

import argparse
import json
from pathlib import Path
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons, RectangleSelector
from matplotlib.patches import Patch
import numpy as np
from skimage import io, transform
from scipy.ndimage import label as connected_components
import torch
from torch.nn import functional as F

from segment_anything import sam_model_registry
from clarius_texture import analyze_texture
from clarius_auto import DEFAULT_CORRELATION_QUANTILE, validate_correlation_quantile

ARRAY_KEYS = ("labels", "selected_mask", "features", "source_labels")


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
    def __init__(self, model, device, paths, output, glcm_levels=8, auto_source="combined",
                 auto_correlation_quantile=DEFAULT_CORRELATION_QUANTILE, glszm_step=(8, 2)):
        self.model, self.device = model, device
        self.paths, self.output = paths, output
        self.index = 0
        self.glcm_levels, self.auto_source = glcm_levels, auto_source
        self.auto_correlation_quantile = validate_correlation_quantile(auto_correlation_quantile)
        if len(glszm_step) != 2 or any(not isinstance(n, (int, np.integer)) or n < 1 for n in glszm_step):
            raise ValueError("GLSZM step must contain two positive integers")
        self.glszm_step = tuple(glszm_step)
        self.embedding_cache = OrderedDict()
        self._busy = False
        self.fig, self.ax = plt.subplots(figsize=(11, 8))
        self.fig.subplots_adjust(bottom=0.31)
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
        self.texture_button = Button(self.fig.add_axes([0.12, 0.145, 0.23, 0.04]), "Split latest region")
        self.texture_button.on_clicked(self.split_texture)
        self.mask_button = Button(self.fig.add_axes([0.39, 0.145, 0.23, 0.04]), "Show MedSAM masks")
        self.mask_button.on_clicked(self.show_masks)
        self.refinement_mode = RadioButtons(self.fig.add_axes([0.67, 0.13, 0.28, 0.14]),
            ("Combined", "Brightness", "GLCM", "Low / High variation", "GLSZM", "Combined GLSZM"), active=3)
        self.refinement_mode.on_clicked(self.change_mode)
        self.connected_only = CheckButtons(self.fig.add_axes([0.12, 0.195, 0.49, 0.025]),
                                            ["Keep clicked connected patch only"], [True])
        self.fig.canvas.mpl_connect("button_press_event", self.pick_texture)
        self.fig.canvas.mpl_connect("close_event", lambda _: self.save() if self._dirty else None)
        self.load()

    @torch.inference_mode()
    def load(self):
        self.texture_cache = {}
        self._dirty = False
        self.image = load_rgb(self.paths[self.index])
        self.height, self.width = self.image.shape[:2]
        key = (str(self.paths[self.index].resolve()), self.paths[self.index].stat().st_mtime_ns)
        if key in self.embedding_cache:
            self.embedding = self.embedding_cache[key]
            self.embedding_cache.move_to_end(key)
        else:
            resized = transform.resize(self.image, (1024, 1024), order=3,
                                       preserve_range=True, anti_aliasing=True)
            resized = (resized - resized.min()) / max(float(np.ptp(resized)), 1e-8)
            tensor = torch.from_numpy(resized.astype(np.float32)).permute(2, 0, 1)[None]
            print(f"Encoding {self.paths[self.index]} on {self.device} ...", flush=True)
            self.embedding = self.model.image_encoder(tensor.to(self.device))
            self.embedding_cache[key] = self.embedding
            if len(self.embedding_cache) > 3:
                self.embedding_cache.popitem(last=False)
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
                        for key in ARRAY_KEYS:
                            if f"texture_{i}_{key}" in saved:
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
                self.texture_cache.clear()
                self._dirty = True
                self.status.set_text(f"Region {len(self.regions)} added. Review its boundary; Undo removes it.")
            self.redraw()
        except Exception as error:
            self.status.set_text(f"Segmentation failed: {error}")
            self.fig.canvas.draw_idle()

    def redraw(self):
        if self.ax.get_legend() is not None:
            self.ax.get_legend().remove()
        rgba = np.zeros((self.height, self.width, 4))
        for i, (_, mask) in enumerate(self.regions):
            rgba[mask] = plt.get_cmap("tab10")(i % 10)
            rgba[mask, 3] = 0.4
        if self.texture is not None:
            labels = self.texture["labels"]
            rgba[:] = 0
            for group in range(1, int(labels.max()) + 1):
                color_index = 1 if self.texture.get("method") == "automatic_tissue_v1" else group - 1
                rgba[labels == group] = plt.get_cmap("tab10")(color_index)
                rgba[labels == group, 3] = 0.45
            selected = self.texture.get("selected_group")
            if selected is not None:
                rgba[(labels > 0) & ~self.texture["selected_mask"], 3] = 0.08
            summaries = self.texture.get("group_summaries", [])
            if summaries:
                self.ax.legend(handles=[Patch(facecolor=plt.get_cmap("tab10")(1 if self.texture.get("method") == "automatic_tissue_v1" else item["group"] - 1),
                    label=f"{item['group']}: {item['description']} ({item['fraction_of_parent']:.0%})")
                    for item in summaries], loc="upper right", fontsize=8, framealpha=0.85)
        self.overlay.set_data(rgba)
        self.fig.canvas.draw_idle()

    def undo(self, _=None):
        if self.regions:
            self.regions.pop()
            self.texture_cache.clear()
            self.refinements.pop(len(self.regions), None)
            self.texture = None
            self._dirty = True
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
            for key in ARRAY_KEYS:
                if key in refinement:
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
                                           if key not in ARRAY_KEYS}
        # Match the full image's presentation in the axes, not the RF sample
        # grid's aspect ratio. Use 2x canvas resolution for a sharper PNG.
        # Never resample annotation arrays or crop the export to a zoomed view.
        self.fig.canvas.draw()
        bounds = self.ax.get_window_extent()
        overlay_shape = (max(1, round(bounds.height * 2)), max(1, round(bounds.width * 2)))
        if max(overlay_shape) > 2048:
            factor = 2048 / max(overlay_shape)
            overlay_shape = tuple(max(1, round(n * factor)) for n in overlay_shape)
        rgba = np.asarray(self.overlay.get_array())
        composite = self.image * (1 - rgba[..., 3:4]) + rgba[..., :3] * rgba[..., 3:4]
        presentation = transform.resize(composite, (*overlay_shape, 3), order=1,
                                        preserve_range=True, anti_aliasing=True)
        metadata = {"source": str(path.resolve()), "shape_hw": [self.height, self.width],
                    "overlay_shape_hw": list(overlay_shape),
                    "overlay_geometry": "full_frame_resized_to_viewer_axes; presentation_only_not_scan_conversion",
                    "note": "Display intensities normalized to 0..1; no physical calibration or diagnostic labels.",
                    "regions": records}
        self.output.joinpath(f"{path.name}.regions.json").write_text(json.dumps(metadata, indent=2))
        plt.imsave(self.output / f"{path.name}.overlay.png", np.clip(presentation, 0, 1))
        self.status.set_text(f"Saved {len(records)} regions to {self.output}")
        self._dirty = False
        self.fig.canvas.draw_idle()

    def navigate(self, step):
        index = self.index + step
        if 0 <= index < len(self.paths):
            if self._dirty:
                self.save()
            self.index = index
            self.load()

    def split_texture(self, _=None):
        if self._busy:
            return
        if not self.regions:
            self.status.set_text("Draw a MedSAM box first.")
            self.fig.canvas.draw_idle()
            return
        mode = self.refinement_mode.value_selected.lower()
        if mode == "low / high variation":
            mode = "variation"
        mode = mode.replace(" ", "_")
        base_mode = self.auto_source if mode == "variation" else mode
        self.status.set_text(f"Calculating {mode} features inside the latest mask ...")
        self.fig.canvas.draw()
        self._busy = True
        start = time.perf_counter()
        try:
            key = (len(self.regions) - 1, base_mode, self.glcm_levels)
            if base_mode in ("glszm", "combined_glszm"):
                key += self.glszm_step
            if key not in self.texture_cache:
                shared = None
                shared_glszm = None
                if base_mode in ("combined", "glcm"):
                    for other in ("combined", "glcm"):
                        existing = self.texture_cache.get((len(self.regions) - 1, other, self.glcm_levels))
                        if existing is not None:
                            maps = existing["features"][..., -8:]
                            shared = (maps, (maps[..., 2] > 0) & (maps[..., 6] > 0) & self.regions[-1][1])
                            break
                if base_mode in ("glszm", "combined_glszm"):
                    for other in ("glszm", "combined_glszm"):
                        existing = self.texture_cache.get((len(self.regions) - 1, other, self.glcm_levels, *self.glszm_step))
                        if existing is not None:
                            shared_glszm = (existing["features"][..., -16:], existing["labels"] > 0)
                            break
                self.texture_cache[key] = analyze_texture(self.image.mean(axis=-1), self.regions[-1][1],
                    method=base_mode, levels=self.glcm_levels, glcm_data=shared,
                    glszm_data=shared_glszm, glszm_step=self.glszm_step)
            source = self.texture_cache[key]
            if mode == "variation":
                auto_key = (*key, "automatic", self.auto_correlation_quantile)
                if auto_key not in self.texture_cache:
                    self.texture_cache[auto_key] = analyze_texture(self.image.mean(axis=-1), self.regions[-1][1],
                        method="variation", levels=self.glcm_levels, auto_source=base_mode, source_result=source,
                        auto_correlation_quantile=self.auto_correlation_quantile)
                refinement = self.texture_cache[auto_key]
            else:
                refinement = source
        except ValueError as error:
            self.status.set_text(f"Refinement unavailable: {error}")
            self.fig.canvas.draw_idle()
            return
        finally:
            self._busy = False
        print(f"{mode} ({base_mode}, {self.glcm_levels} levels): {time.perf_counter() - start:.2f}s", flush=True)
        self.texture = refinement
        self.refinements[len(self.regions) - 1] = self.texture
        self.selector.set_active(False)
        self._dirty = True
        groups = self.texture["actual_groups"]
        if mode == "variation":
            gate = self.texture["automatic_refinement"]
            setting = f", q{gate['distance_gate_quantile']:.3f}" if gate["distance_gate_applied"] else ", legacy texture rule"
            self.status.set_text(f"Automatically kept {int(self.texture['selected_mask'].sum())} pixels from {base_mode}{setting}. Save to export.")
        else:
            self.status.set_text(f"{mode.capitalize()}: {groups} group(s). Click one to keep it; switch mode to compare.")
        self.redraw()

    def change_mode(self, _=None):
        if self._busy:
            return
        mode = self.refinement_mode.value_selected.lower().replace(" ", "_")
        if mode in ("combined", "glcm", "glszm", "combined_glszm"):
            self.auto_source = mode
        if self.texture is not None:
            self.split_texture()

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
            selected = self.texture["labels"] == group
            if self.connected_only.get_status()[0]:
                components, _ = connected_components(selected, structure=np.ones((3, 3)))
                selected = components == components[y, x]
                self.texture["selection_mode"] = "clicked_component_8_connected"
            else:
                self.texture["selection_mode"] = "whole_group"
            self.texture["selected_mask"] = selected
            self._dirty = True
            self.status.set_text(f"Keeping {int(selected.sum())} pixels from group {group}. Save to export.")
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
    parser.add_argument("--glcm-levels", type=int, choices=(8, 16, 32, 64), default=8)
    parser.add_argument("--auto-source", choices=("combined", "glcm", "glszm", "combined_glszm"), default="combined",
                        help="Source proposal for automatic Low / High variation refinement")
    parser.add_argument("--auto-correlation-quantile", "--auto-texture-quantile", dest="auto_correlation_quantile", type=validate_correlation_quantile,
                        default=DEFAULT_CORRELATION_QUANTILE,
                        help="Automatic GLCM-correlation or GLSZM-feature distance quantile (default: 0.985; 0 disables it)")
    parser.add_argument("--glszm-step", type=int, nargs=2, default=(8, 2), metavar=("DEPTH", "LATERAL"),
                        help="GLSZM sampling step in native pixels; 1 1 is exact dense mode (default: 8 2)")
    args = parser.parse_args()
    if any(n < 1 for n in args.glszm_step):
        parser.error("GLSZM sampling steps must be positive")
    paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in
                   {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}) if args.input.is_dir() else [args.input]
    if not paths or any(not p.is_file() for p in paths):
        parser.error("No input images found")
    model = sam_model_registry["vit_b"](checkpoint=str(args.checkpoint)).to(args.device).eval()
    viewer = Viewer(model, args.device, paths, args.output, glcm_levels=args.glcm_levels, auto_source=args.auto_source,
                    auto_correlation_quantile=args.auto_correlation_quantile, glszm_step=args.glszm_step)
    plt.show()


if __name__ == "__main__":
    main()
