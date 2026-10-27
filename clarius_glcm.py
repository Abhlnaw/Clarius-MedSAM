"""Exact local, symmetric GLCM features with mask-aware pixel pairs.

Histograms are streamed one gray-level pair at a time using summed-area tables;
no height x width x levels x levels tensor is allocated.
"""

import numpy as np


PROPERTIES = ("contrast", "homogeneity", "energy", "correlation")
DEFAULT_OFFSETS = ((0, 1), (1, 1), (1, 0), (1, -1))
DEFAULT_LEVELS = 8
FEATURE_NAMES = tuple(f"{scale}_glcm_{prop}" for scale in ("half_window", "full_window")
                      for prop in PROPERTIES)


def analysis_windows(window):
    """Odd half/full windows in (row, column) image pixels."""
    if len(window) != 2 or any(not isinstance(n, (int, np.integer)) or n < 3 for n in window):
        raise ValueError("Window must contain two integer sizes >= 3")
    return [tuple(max(3, int(n * scale) // 2 * 2 + 1) for n in window)
            for scale in (0.5, 1.0)]


def _bounds(length, size, delta):
    centers = np.arange(length)
    # The source AND destination must be within the centered window.
    lo = np.clip(np.maximum(centers - size // 2, centers - size // 2 - delta), 0, length)
    hi = np.clip(np.minimum(centers + size // 2 + 1,
                            centers + size // 2 + 1 - delta), 0, length)
    return lo, np.maximum(lo, hi)


def _local_sum(image, row_bounds, col_bounds):
    integral = np.pad(image, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    r0, r1 = row_bounds
    c0, c1 = col_bounds
    return (integral[r1[:, None], c1] - integral[r0[:, None], c1]
            - integral[r1[:, None], c0] + integral[r0[:, None], c0])


def local_glcm_features(gray, mask, window, levels=DEFAULT_LEVELS, offsets=DEFAULT_OFFSETS):
    """Return (H,W,8) maps and (H,W) supported-pixel mask.

    Input intensities inside mask must be finite and in [0,1]. Fixed quantization
    floor(gray * levels), clipped at levels-1, avoids using background statistics.
    Offsets are (row delta, column delta); opposite directions are made symmetric.
    Features are averaged equally over directions with at least one valid pair.
    A pixel needs pair support at both window scales; unsupported maps stay zero.
    Correlation for a constant marginal is defined as 1, matching scikit-image.
    """
    gray = np.asarray(gray, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if gray.ndim != 2 or gray.shape != mask.shape or not mask.any():
        raise ValueError("Expected a matching image and nonempty 2D mask")
    if not isinstance(levels, (int, np.integer)) or not 2 <= levels <= 64:
        raise ValueError("Gray levels must be an integer from 2 to 64")
    offsets = tuple(tuple(offset) for offset in offsets)
    if not offsets or any(len(o) != 2 or any(not isinstance(d, (int, np.integer)) for d in o)
                          or o == (0, 0) for o in offsets):
        raise ValueError("Offsets must be nonzero integer (row, column) pairs")
    windows = analysis_windows(window)
    inside = gray[mask]
    if not np.isfinite(inside).all() or inside.min() < 0 or inside.max() > 1:
        raise ValueError("Masked intensities must be finite and normalized to [0,1]")
    rows, cols = np.nonzero(mask)
    crop = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
    cropped_mask = mask[crop]
    quantized = np.minimum((np.where(cropped_mask, gray[crop], 0) * levels).astype(np.int32), levels - 1)
    height, width = cropped_mask.shape
    sums = [np.zeros((height, width, 4), dtype=np.float64) for _ in windows]
    directions = [np.zeros((height, width), dtype=np.int32) for _ in windows]
    for dy, dx in offsets:
        if abs(dy) >= height or abs(dx) >= width:
            continue
        source = (slice(max(0, -dy), min(height, height - dy)),
                  slice(max(0, -dx), min(width, width - dx)))
        destination = (slice(max(0, dy), min(height, height + dy)),
                       slice(max(0, dx), min(width, width + dx)))
        valid = cropped_mask[source] & cropped_mask[destination]
        first, second = quantized[source], quantized[destination]
        codes = np.full((height, width), -1, dtype=np.int32)
        unordered = np.minimum(first, second) * levels + np.maximum(first, second)
        codes[source] = np.where(valid, unordered, -1)
        bounds = [(_bounds(height, h, dy), _bounds(width, w, dx)) for h, w in windows]
        totals = [_local_sum((codes >= 0).astype(np.int64), *b) for b in bounds]
        accum = [np.zeros((height, width, 6), dtype=np.float64) for _ in windows]
        for code in np.unique(codes[codes >= 0]):
            i, j = divmod(int(code), levels)
            pair = (codes == code).astype(np.int64)
            for total, values, bound in zip(totals, accum, bounds):
                probability = _local_sum(pair, *bound) / np.maximum(total, 1)
                values[..., 0] += probability * (i - j) ** 2
                values[..., 1] += probability / (1 + (i - j) ** 2)
                values[..., 2] += probability ** 2 * (1 if i == j else 0.5)
                values[..., 3] += probability * (i + j) / 2
                values[..., 4] += probability * (i * i + j * j) / 2
                values[..., 5] += probability * i * j
        for index, (total, values) in enumerate(zip(totals, accum)):
            supported = total > 0
            variance = np.maximum(values[..., 4] - values[..., 3] ** 2, 0)
            correlation = np.ones((height, width))
            np.divide(values[..., 5] - values[..., 3] ** 2, variance,
                      out=correlation, where=variance > 1e-12)
            properties = np.stack((values[..., 0], values[..., 1],
                                   np.sqrt(values[..., 2]), np.clip(correlation, -1, 1)), axis=-1)
            sums[index] += properties * supported[..., None]
            directions[index] += supported
    supported = cropped_mask & (directions[0] > 0) & (directions[1] > 0)
    combined = np.concatenate([s / np.maximum(n[..., None], 1)
                               for s, n in zip(sums, directions)], axis=-1)
    combined[~supported] = 0
    maps = np.zeros((*mask.shape, len(FEATURE_NAMES)), dtype=np.float32)
    maps[crop] = combined
    support = np.zeros(mask.shape, dtype=bool)
    support[crop] = supported
    return maps, support
