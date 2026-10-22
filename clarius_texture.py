"""Mask-aware, multiscale local-intensity texture grouping."""

import numpy as np
from scipy.ndimage import uniform_filter


def automatic_window(mask):
    """Use one quarter of the mask's occupied extent in each image direction.

    This is a relative-scale heuristic, not an estimate of physical spacing.
    The second analysis scale uses half this window.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("Select a nonempty 2D MedSAM region first")
    rows, columns = np.nonzero(mask)
    extents = (int(rows.max() - rows.min() + 1), int(columns.max() - columns.min() + 1))
    return tuple(max(3, int(round(extent / 4)) // 2 * 2 + 1) for extent in extents)


def texture_groups(gray, mask, window=None, groups=3):
    """Return labels 1..K inside mask, zero outside, and local feature maps.

    Windows are (depth pixels, lateral pixels), not physical measurements.
    Only masked pixels contribute to local moments. Groups describe appearance,
    not tissue identity. IDs are ordered by local mean brightness.
    """
    gray = np.asarray(gray, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if gray.shape != mask.shape or gray.ndim != 2:
        raise ValueError("Expected matching 2D image and mask")
    if not mask.any():
        raise ValueError("Select a nonempty MedSAM region first")
    if window is None:
        window = automatic_window(mask)
    if min(window) < 3 or groups < 2:
        raise ValueError("Windows must be >=3 pixels and groups >=2")
    features = []
    for scale in (0.5, 1.0):
        size = tuple(max(3, int(n * scale) // 2 * 2 + 1) for n in window)
        weight = uniform_filter(mask.astype(float), size, mode="constant")
        denominator = np.maximum(weight, 1e-12)
        mean = uniform_filter(np.where(mask, gray, 0), size, mode="constant") / denominator
        second = uniform_filter(np.where(mask, gray * gray, 0), size, mode="constant") / denominator
        features.extend([mean, np.sqrt(np.maximum(second - mean * mean, 0))])
    maps = np.stack(features, axis=-1).astype(np.float32)
    values = maps[mask].astype(float)
    center = values.mean(axis=0)
    spread = values.std(axis=0)
    spread[spread < 1e-6] = 1.0
    values = (values - center) / spread
    # Bounded deterministic training sample; classify the full region in chunks.
    sample = values[::max(1, len(values) // 20000)][:20000]
    centers = [sample[np.argmin(np.sum(sample * sample, axis=1))]]
    for _ in range(1, min(groups, len(sample))):
        distance = np.min([np.sum((sample - c) ** 2, axis=1) for c in centers], axis=0)
        if distance.max() < 1e-10:
            break
        centers.append(sample[np.argmax(distance)])
    centers = np.array(centers)
    for _ in range(40):
        labels = np.argmin(((sample[:, None] - centers) ** 2).sum(axis=2), axis=1)
        updated = np.array([sample[labels == i].mean(axis=0) if np.any(labels == i) else c
                            for i, c in enumerate(centers)])
        if np.allclose(updated, centers, atol=1e-4):
            break
        centers = updated
    order = np.argsort(centers[:, 0])
    centers = centers[order]
    assigned = np.empty(len(values), dtype=np.uint8)
    for start in range(0, len(values), 50000):
        assigned[start:start + 50000] = 1 + np.argmin(
            ((values[start:start + 50000, None] - centers) ** 2).sum(axis=2), axis=1)
    result = np.zeros(mask.shape, dtype=np.uint8)
    result[mask] = assigned
    maps[~mask] = 0
    return result, maps
