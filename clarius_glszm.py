"""2D mask-aware GLSZM matrices, standard features, and sliding-window maps.

Zones are 8-connected pixels with the same discretized intensity. Every window
relabels its own clipped ROI; global component sizes are never used as a proxy.
Definitions follow PyRadiomics' GLSZM feature documentation. No radiomics package
or compiled extension is required beyond NumPy/SciPy/scikit-image.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from skimage.measure import label

from clarius_glcm import DEFAULT_LEVELS, analysis_windows


PROPERTIES = (
    "small_area_emphasis", "large_area_emphasis",
    "gray_level_non_uniformity", "gray_level_non_uniformity_normalized",
    "size_zone_non_uniformity", "size_zone_non_uniformity_normalized",
    "zone_percentage", "gray_level_variance", "zone_variance", "zone_entropy",
    "low_gray_level_zone_emphasis", "high_gray_level_zone_emphasis",
    "small_area_low_gray_level_emphasis", "small_area_high_gray_level_emphasis",
    "large_area_low_gray_level_emphasis", "large_area_high_gray_level_emphasis",
)
# Avoid duplicating normalized and unnormalized nonuniformity in the local maps.
LOCAL_PROPERTIES = (
    "small_area_emphasis", "large_area_emphasis",
    "gray_level_non_uniformity_normalized", "size_zone_non_uniformity_normalized",
    "zone_percentage", "gray_level_variance", "zone_variance", "zone_entropy",
)
FEATURE_NAMES = tuple(f"{scale}_glszm_{prop}" for scale in ("half_window", "full_window")
                      for prop in LOCAL_PROPERTIES)
DEFAULT_STEP = (8, 2)


def _quantize(gray, mask, levels):
    gray, mask = np.asarray(gray, float), np.asarray(mask, bool)
    if gray.ndim != 2 or gray.shape != mask.shape or not mask.any():
        raise ValueError("Expected a matching image and nonempty 2D mask")
    if not isinstance(levels, (int, np.integer)) or not 2 <= levels <= 64:
        raise ValueError("Gray levels must be an integer from 2 to 64")
    values = gray[mask]
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Masked intensities must be finite and normalized to [0,1]")
    # One-based gray-level indices keep zero available for excluded pixels.
    return np.where(mask, np.minimum((np.where(mask, gray, 0) * levels).astype(np.int32),
                                     levels - 1) + 1, 0), mask


def _zones(quantized):
    components, count = label(quantized, background=0, connectivity=2, return_num=True)
    sizes = np.bincount(components.ravel(), minlength=count + 1)[1:]
    gray_levels = np.zeros(count + 1, np.int32)
    np.maximum.at(gray_levels, components.ravel(), quantized.ravel())
    return gray_levels[1:], sizes


def glszm_matrix(gray, mask, levels=DEFAULT_LEVELS):
    """Return integer P[gray_level-1, zone_size-1], including empty gray bins."""
    quantized, _ = _quantize(gray, mask, levels)
    gray_levels, sizes = _zones(quantized)
    matrix = np.zeros((levels, int(sizes.max())), np.int64)
    np.add.at(matrix, (gray_levels - 1, sizes - 1), 1)
    return matrix


def _features(gray_levels, sizes):
    gray_levels, sizes = gray_levels.astype(float), sizes.astype(float)
    nz = len(sizes)
    gray_counts = np.unique(gray_levels, return_counts=True)[1].astype(float)
    size_counts = np.unique(sizes, return_counts=True)[1].astype(float)
    pairs = np.stack((gray_levels, sizes), axis=1)
    probabilities = np.unique(pairs, axis=0, return_counts=True)[1] / nz
    g2, s2 = gray_levels ** 2, sizes ** 2
    return np.array([
        np.mean(1 / s2), np.mean(s2),
        np.sum(gray_counts ** 2) / nz, np.sum(gray_counts ** 2) / nz ** 2,
        np.sum(size_counts ** 2) / nz, np.sum(size_counts ** 2) / nz ** 2,
        nz / sizes.sum(), np.var(gray_levels), np.var(sizes),
        -np.sum(probabilities * np.log2(probabilities)),
        np.mean(1 / g2), np.mean(g2), np.mean(1 / (g2 * s2)),
        np.mean(g2 / s2), np.mean(s2 / g2), np.mean(g2 * s2),
    ], dtype=float)


def glszm_features(gray, mask, levels=DEFAULT_LEVELS):
    """Return all 16 standard GLSZM features for one nonempty ROI."""
    quantized, _ = _quantize(gray, mask, levels)
    return dict(zip(PROPERTIES, _features(*_zones(quantized)).tolist()))


def local_glszm_features(gray, mask, window, levels=DEFAULT_LEVELS, step=DEFAULT_STEP):
    """Return (H,W,16) maps and support at half/full window scales.

    Step (1,1) computes an exact independent GLSZM at every masked pixel.
    Default (8,2) evaluates exact window matrices on a native-sample lattice,
    then mask-normalized bilinear interpolation approximates intervening maps.
    Windows clip at image edges and parent-mask exclusions split zones. No image
    resizing or physical-space interpretation is performed. Outside maps are 0.
    """
    quantized, mask = _quantize(gray, mask, levels)
    windows = analysis_windows(window)
    if len(step) != 2 or any(not isinstance(n, (int, np.integer)) or n < 1 for n in step):
        raise ValueError("GLSZM step must contain two positive integers")
    rows, cols = np.nonzero(mask)
    crop = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
    quantized = quantized[crop]
    height, width = quantized.shape
    grid = [np.unique(np.r_[np.arange(0, n, stride), n - 1]) for n, stride in zip((height, width), step)]
    maps = np.zeros((len(grid[0]), len(grid[1]), 16), float)
    supported = np.zeros((*maps.shape[:2], 2), float)
    property_indices = [PROPERTIES.index(prop) for prop in LOCAL_PROPERTIES]
    for iy, y in enumerate(grid[0]):
        for ix, x in enumerate(grid[1]):
            # Exact dense mode need not evaluate centers outside the ROI.
            if tuple(step) == (1, 1) and not mask[crop][y, x]:
                continue
            for scale_index, (h, w) in enumerate(windows):
                patch = quantized[max(0, y - h // 2):min(height, y + h // 2 + 1),
                                  max(0, x - w // 2):min(width, x + w // 2 + 1)]
                if np.any(patch):
                    features = _features(*_zones(patch))[property_indices]
                    maps[iy, ix, scale_index * 8:(scale_index + 1) * 8] = features
                    supported[iy, ix, scale_index] = 1
    yy, xx = np.indices((height, width))
    queries = np.stack((yy.ravel(), xx.ravel()), axis=-1)
    interpolated = RegularGridInterpolator(grid, maps, bounds_error=False, fill_value=None)(queries)
    weights = RegularGridInterpolator(grid, supported, bounds_error=False, fill_value=None)(queries)
    support = mask[crop] & (weights.reshape(height, width, 2) > 1e-12).all(axis=-1)
    interpolated /= np.maximum(np.repeat(weights, 8, axis=-1), 1e-12)
    interpolated = interpolated.reshape(height, width, 16)
    interpolated[~support] = 0
    result = np.zeros((*mask.shape, 16), np.float32)
    result[crop] = interpolated
    full_support = np.zeros(mask.shape, bool)
    full_support[crop] = support
    return result, full_support
