"""Group local GLCM features within a MedSAM candidate region."""

import numpy as np
from clarius_glcm import (DEFAULT_LEVELS, DEFAULT_OFFSETS, FEATURE_NAMES,
                          analysis_windows, local_glcm_features)


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


def _cluster_features(maps, support, groups):
    if not isinstance(groups, (int, np.integer)) or not 1 <= groups <= 255:
        raise ValueError("Groups must be an integer from 1 to 255")
    if not support.any():
        raise ValueError("Region has no valid neighboring pixel pairs at both scales; draw a larger region")
    values = maps[support].astype(float)
    center = values.mean(axis=0)
    spread = values.std(axis=0)
    spread[spread < 1e-6] = 1.0
    values = (values - center) / spread
    # Bounded deterministic training sample; classify the full region in chunks.
    sample = values[np.linspace(0, len(values) - 1, min(20000, len(values)), dtype=int)]
    # Multiple deterministic starts reduce the chance of isolating edge outliers
    # while merging two substantial texture populations into a single group.
    starts = [int(np.argmin(np.sum(sample * sample, axis=1))),
              int(np.argmin(sample[:, 0])), int(np.argmax(sample[:, 0]))]
    rng = np.random.default_rng(2023)
    best_error, best_centers = np.inf, None
    for start in dict.fromkeys(starts):
        centers = [sample[start]]
        for _ in range(1, min(groups, len(sample))):
            distance = np.min([np.sum((sample - c) ** 2, axis=1) for c in centers], axis=0)
            if distance.max() < 1e-10:
                break
            centers.append(sample[rng.choice(len(sample), p=distance / distance.sum())])
        centers = np.array(centers)
        for _ in range(40):
            labels = np.argmin(((sample[:, None] - centers) ** 2).sum(axis=2), axis=1)
            updated = np.array([sample[labels == i].mean(axis=0) if np.any(labels == i) else c
                                for i, c in enumerate(centers)])
            converged = np.allclose(updated, centers, atol=1e-4)
            centers = updated
            if converged:
                break
        error = np.min(((sample[:, None] - centers) ** 2).sum(axis=2), axis=1).sum()
        if error < best_error:
            best_error, best_centers = error, centers
    centers = best_centers
    order = np.argsort(centers[:, 0])
    centers = centers[order]
    assigned = np.empty(len(values), dtype=np.uint8)
    for start in range(0, len(values), 50000):
        assigned[start:start + 50000] = 1 + np.argmin(
            ((values[start:start + 50000, None] - centers) ** 2).sum(axis=2), axis=1)
    result = np.zeros(support.shape, dtype=np.uint8)
    result[support] = assigned
    return result


def analyze_texture(gray, mask, window=None, groups=3, levels=DEFAULT_LEVELS):
    """Produce the arrays and reproducibility metadata saved by the viewer.

    Groups are a requested maximum, not automatically inferred tissue classes.
    Labels are ordered by standardized half-window contrast of cluster centers.
    Zero means outside the mask or without local pair support.
    """
    automatic = window is None
    if automatic:
        window = automatic_window(mask)
    features, support = local_glcm_features(gray, mask, window, levels=levels)
    labels = _cluster_features(features, support, groups)
    return {"method": "local_glcm_v1", "labels": labels, "features": features,
            "feature_names": list(FEATURE_NAMES), "window_hw_pixels": list(window),
            "analysis_windows_hw_pixels": [list(w) for w in analysis_windows(window)],
            "window_selection": "automatic_quarter_mask_extent" if automatic else "explicit",
            "gray_levels": int(levels), "quantization": "floor(clip(gray,0,1)*levels), capped at levels-1",
            "offsets_dy_dx": [list(o) for o in DEFAULT_OFFSETS],
            "symmetric": True, "direction_aggregation": "equal_mean_over_supported_directions",
            "clustering": "deterministic_kmeans_zscore", "requested_groups": int(groups),
            "actual_groups": int(len(np.unique(labels[support]))),
            "supported_pixels": int(support.sum()),
            "unsupported_mask_pixels": int(np.count_nonzero(mask) - support.sum()),
            "selected_group": None, "selected_mask": np.zeros(labels.shape, dtype=bool)}


def texture_groups(gray, mask, window=None, groups=3):
    """Compatibility wrapper returning (labels, eight local GLCM feature maps)."""
    result = analyze_texture(gray, mask, window=window, groups=groups)
    return result["labels"], result["features"]
