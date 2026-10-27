"""Group local GLCM features within a MedSAM candidate region."""

import numpy as np
from scipy.ndimage import uniform_filter
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


def localized_window(mask):
    """Smaller windows for hybrid/GLCM refinement, capped on the sample grid."""
    large = automatic_window(mask)
    return tuple(min(limit, max(3, (n // 2) // 2 * 2 + 1))
                 for n, limit in zip(large, (129, 33)))


def _intensity_features(gray, mask, window):
    features = []
    for size in analysis_windows(window):
        weight = uniform_filter(mask.astype(float), size, mode="constant")
        denominator = np.maximum(weight, 1e-12)
        mean = uniform_filter(np.where(mask, gray, 0), size, mode="constant") / denominator
        second = uniform_filter(np.where(mask, gray * gray, 0), size, mode="constant") / denominator
        features.extend((mean, np.sqrt(np.maximum(second - mean * mean, 0))))
    maps = np.stack(features, axis=-1).astype(np.float32)
    maps[~mask] = 0
    return maps


def _cluster_features(maps, support, groups, weights=None):
    if not isinstance(groups, (int, np.integer)) or not 1 <= groups <= 255:
        raise ValueError("Groups must be an integer from 1 to 255")
    if not support.any():
        raise ValueError("Region has no valid neighboring pixel pairs at both scales; draw a larger region")
    values = maps[support].astype(float)
    center = values.mean(axis=0)
    spread = values.std(axis=0)
    spread[spread < 1e-6] = 1.0
    values = (values - center) / spread
    if weights is not None:
        values *= weights
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


def analyze_texture(gray, mask, window=None, groups=3, levels=DEFAULT_LEVELS, method="glcm"):
    """Produce the arrays and reproducibility metadata saved by the viewer.

    Groups are a requested maximum, not automatically inferred tissue classes.
    Labels are ordered by the first standardized feature of cluster centers:
    half-window contrast for GLCM, half-window mean for brightness/combined.
    Zero means outside the mask or without local pair support.
    """
    if method not in ("glcm", "brightness", "combined"):
        raise ValueError("Unknown refinement method")
    gray, mask = np.asarray(gray, dtype=float), np.asarray(mask, dtype=bool)
    if gray.ndim != 2 or gray.shape != mask.shape or not mask.any():
        raise ValueError("Expected matching image and nonempty 2D mask")
    if not np.isfinite(gray[mask]).all() or np.any((gray[mask] < 0) | (gray[mask] > 1)):
        raise ValueError("Masked intensities must be finite and normalized to [0,1]")
    automatic = window is None
    if automatic:
        window = automatic_window(mask) if method == "brightness" else localized_window(mask)
    names = [f"{scale}_{prop}" for scale in ("half_window", "full_window") for prop in ("mean", "std")]
    weights = None
    if method == "brightness":
        features, support = _intensity_features(gray, mask, window), mask.copy()
    else:
        features, support = local_glcm_features(gray, mask, window, levels=levels)
        if method == "combined":
            features = np.concatenate((_intensity_features(gray, mask, window), features), axis=-1)
            features[~support] = 0
            names += list(FEATURE_NAMES)
            # Give the brightness block more influence; avoid eight GLCM
            # channels outvoting four intensity channels simply by their count.
            weights = np.array([1 / np.sqrt(2), 0.25] * 2 + [0.35 / np.sqrt(8)] * 8)
        else:
            names = list(FEATURE_NAMES)
    labels = _cluster_features(features, support, groups, weights=weights)
    result = {"method": {"glcm": "local_glcm_v2", "brightness": "local_intensity_v1",
                          "combined": "local_hybrid_v1"}[method],
            "labels": labels, "features": features,
            "feature_names": names, "window_hw_pixels": list(window),
            "analysis_windows_hw_pixels": [list(w) for w in analysis_windows(window)],
            "window_selection": ("automatic_quarter_mask_extent" if method == "brightness"
                                  else "automatic_eighth_mask_extent_capped_129x33") if automatic else "explicit",
            "gray_levels": int(levels), "quantization": "floor(clip(gray,0,1)*levels), capped at levels-1",
            "offsets_dy_dx": [list(o) for o in DEFAULT_OFFSETS],
            "symmetric": True, "direction_aggregation": "equal_mean_over_supported_directions",
            "clustering": "deterministic_kmeans_zscore", "requested_groups": int(groups),
            "actual_groups": int(len(np.unique(labels[support]))),
            "supported_pixels": int(support.sum()),
            "unsupported_mask_pixels": int(np.count_nonzero(mask) - support.sum()),
            "selected_group": None, "selected_mask": np.zeros(labels.shape, dtype=bool)}
    result["feature_weights_after_zscore"] = weights.tolist() if weights is not None else [1.0] * len(names)
    if method == "brightness":
        for key in ("gray_levels", "quantization", "offsets_dy_dx", "symmetric", "direction_aggregation"):
            result.pop(key)
    parent_mean = float(gray[mask].mean())
    parent_std = float(gray[mask].std())
    summaries = []
    for group in np.unique(labels[support]):
        values = gray[labels == group]
        mean, std = float(values.mean()), float(values.std())
        brightness = "darker" if mean < parent_mean - 0.03 else "brighter" if mean > parent_mean + 0.03 else "similar brightness"
        variation = "higher variation" if std > parent_std * 1.15 + 0.01 else "lower variation" if std < parent_std * 0.85 - 0.01 else "similar variation"
        summaries.append({"group": int(group), "area_pixels": int(len(values)),
                          "fraction_of_parent": float(len(values) / mask.sum()),
                          "mean_display_intensity": mean, "std_display_intensity": std,
                          "description": f"{brightness}, {variation}"})
    result["group_summaries"] = summaries
    return result


def texture_groups(gray, mask, window=None, groups=3):
    """Compatibility wrapper returning (labels, eight local GLCM feature maps)."""
    result = analyze_texture(gray, mask, window=window, groups=groups)
    return result["labels"], result["features"]
