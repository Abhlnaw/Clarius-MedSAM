# Clarius-MedSAM

Adaptations of [MedSAM](https://github.com/bowang-lab/MedSAM) for exploring and annotating Clarius ultrasound B-mode images. This README describes the Clarius-specific changes in this repository. The pretrained MedSAM model and architecture are unchanged.

## Changes added for Clarius ultrasound

### Clean B-mode export from MATLAB

`clarius/runme2_export.m` adapts the existing MATLAB RF/IQ/envelope viewer. For RF frames, it exports the underlying log-compressed, envelope-derived B-mode image as a grayscale PNG, rather than capturing a screenshot of the plotted axes. Each frame is scaled using its own intensity range and receives a JSON file with source, frame, dimensions, timestamp, and intensity limits.

This keeps axes, titles, and screen-resolution scaling out of the segmentation input. The original external `runme2.m` is unchanged.

### Interactive segmentation of exported frames

`MedSAM_Clarius.py` adds a Matplotlib viewer for a single image or a folder of frames:

- Drag a bounding box to generate a MedSAM mask.
- Draw multiple boxes to annotate separate regions.
- Undo the last region and browse frames with Previous/Next.
- Save separate masks, an overlay, and JSON measurements; navigation and closing also save.
- Reload existing annotations when returning to a frame.

Grayscale, RGB, and RGBA inputs are converted to three channels. `MedSAM_Inference.py` also handles RGBA and single-channel images so PNG alpha channels do not cause the encoder's channel-count error.

### Automatic texture-guided region selection

The **Low / High variation** mode now automatically finds a coherent tissue region from a Combined or GLCM proposal. It replaces the earlier whole-color standard-deviation merge with compact-body seed selection, robust local appearance matching across source colors, spatial cleanup, and dark-cavity protection. It retains the main component while suppressing narrow lower spillover. No tissue click, painting, or second crop is required after the MedSAM box.

Combined is the default source; select GLCM before returning to the automatic mode, or use `--auto-source glcm`. The default quantization is eight gray levels; `--glcm-levels 16`, `32`, or `64` is available. Higher levels cost substantially more CPU time.

Automatic refinement now uses the tested **q0.985 correlation-distance** gate.
Use `--auto-correlation-quantile 0.980` for a tighter comparison, or `0` for the
previous acceptance rule. Existing saved selections need **Split latest
region** again to apply the setting; a separate output folder keeps comparisons
separate. See [settings and limits](clarius/README.md#settings-and-performance).

An experimental [local GLSZM implementation](clarius/GLSZM.md) is available via
`--auto-source glszm` or `--auto-source combined_glszm`, with selectable inspection
views and a reproducible comparison script. The default remains Combined/GLCM.

The original MedSAM masks and source labels are retained alongside the automatic result. GLCM maps, automatic results, and recent image embeddings are cached. The manual correction tools from the rejected iteration have been removed. See [the current workflow](clarius/README.md) and [algorithm details](clarius/GLCM.md).

## Run the Clarius workflow

Use a Python environment with the MedSAM dependencies, including PyTorch, NumPy, SciPy, scikit-image, and Matplotlib, and obtain the MedSAM ViT-B checkpoint separately.

1. Set the exam directory near the top of `clarius/runme2_export.m`.
2. Run the MATLAB script to create the exam's `exported_images` folder. From PowerShell in this repository:

   ```powershell
   matlab -batch "run('clarius/runme2_export.m')"
   ```

3. Launch the viewer, substituting your exported frame directory:

   ```powershell
   python MedSAM_Clarius.py --checkpoint .\medsam_vit_b.pth --device cuda:0 -i "C:\path\to\exam\exported_images" --output .\clarius_annotations\exam04
   ```

   Use `--device cpu` without CUDA. The input may also be a single image. Use a separate output folder for each exam.

4. Draw a tight box, review the mask, optionally split it by texture, and save. Turn off the Matplotlib zoom/pan tools before drawing or selecting a texture group.

See [the detailed workflow and output format](clarius/README.md) for additional usage notes.

## Saved outputs

Each image produces:

- **NPZ:** individual MedSAM masks, plus texture labels, feature maps, and selected refined masks when texture analysis has been run. Separate masks preserve overlaps.
- **JSON:** source image, bounding boxes, region IDs, pixel areas, display brightness statistics, and texture settings.
- **PNG:** the full-frame overlay resized to the viewer axes proportions at twice the canvas resolution, so tall RF sample grids do not appear as narrow strips. This is a presentation image; masks remain on the original grid.

Generated annotations, raw RF data, exported images, and model checkpoints are excluded from new Git commits by `.gitignore`.

## Current limits and checks

The exports retain the RF sample/line grid without scan conversion. Measurements are in pixels; physical measurements need verified acquisition geometry. Per-frame normalization means brightness statistics are not calibrated across frames.

Texture groups describe appearance, not anatomy or lesions. This workflow does not add automatic lesion detection, manual boundary painting, or tracking between frames. It can refine an oversized candidate mask but cannot recover tissue outside that mask.

The feature, grouping, and headless viewer tests check GLCM calculations against scikit-image, mask isolation, spatial patterns, automatic scales, selection, save/reload, and legacy annotations:

```powershell
python -m unittest discover -s tests -p "test_clarius_*.py" -v
```

The regression suite checks both numerical features and automatic-region behavior. Qualitative evaluation also uses exam04/exam05 frames and the actual CUDA checkpoint with no tissue-selection clicks. These checks do not establish anatomical accuracy without expert reference masks. See [validation details](clarius/README.md#validation).

## Attribution

Built on [MedSAM by bowang-lab](https://github.com/bowang-lab/MedSAM). See [LICENSE](LICENSE) for the repository license.
