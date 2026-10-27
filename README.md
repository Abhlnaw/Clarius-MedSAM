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

### Texture refinement inside oversized masks

The viewer now offers **Combined** (default), **Brightness**, and **GLCM** modes. Combined restores the earlier local mean/standard-deviation features and adds lower-weight GLCM features. Brightness uses the earlier four-feature approach and larger original windows. GLCM uses co-occurrence features alone. Switch modes after splitting to compare the same parent mask. Combined/GLCM use smaller automatic windows, capped at 129 depth pixels by 33 lateral pixels, to reduce excessive spatial averaging.

A legend describes each group relative to the parent mask (darker/brighter and lower/higher intensity variation), and those measurements are saved in JSON. These are appearance summaries, not tissue labels.

`clarius_glcm.py` extracts local gray-level co-occurrence matrix (GLCM) features inside the latest MedSAM region. It measures contrast, homogeneity, energy, and correlation at two automatic neighborhood sizes, using 16 gray levels and four pixel-pair directions. `clarius_texture.py` standardizes the eight resulting features and uses fixed-seed k-means clustering to create up to three appearance groups.

Neighborhood sizes are automatic: Brightness uses approximately one quarter of the selected region's occupied dimensions; Combined/GLCM use approximately one eighth with the caps above. Each method also uses a second window at half that size. Only pairs with both endpoints inside the mask and local window contribute. Pixels without pair support are excluded from grouping.

Click **Split latest region**, then click a colored group to save it as a refined mask. The original MedSAM mask is retained. By default a click keeps only the connected patch under the cursor; disable **Keep clicked connected patch only** to select the entire group. **Show MedSAM masks** returns to box drawing.

Three groups is a configurable code default, not an anatomical assumption or an automatically estimated number of tissue types. Existing four-feature annotations remain readable; splitting again explicitly replaces the latest refinement with the selected method. See [the GLCM explanation](clarius/GLCM.md) for pair counting, feature formulas, parameters, output format, and limitations.

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

All 24 tests pass, including mode switching and brightness-sensitive hybrid grouping. A headless end-to-end smoke test also ran the actual checkpoint on CUDA with an exported Clarius frame; GLCM refinement took about 2.7 seconds for a 49,097-pixel candidate mask. This verifies execution, not anatomical accuracy. See [validation details](clarius/GLCM.md#validation).

## Attribution

Built on [MedSAM by bowang-lab](https://github.com/bowang-lab/MedSAM). See [LICENSE](LICENSE) for the repository license.
