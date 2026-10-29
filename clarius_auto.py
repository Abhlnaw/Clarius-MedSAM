"""Automatic spatial refinement of Combined/GLCM tissue proposals.

Select a compact dominant tissue population, then recover locally compatible
boundary pixels. This is an appearance heuristic, not an anatomical detector.
"""

import numpy as np
from scipy import ndimage as ndi
from skimage import transform


DEFAULT_CORRELATION_QUANTILE = 0.985


def validate_correlation_quantile(value):
    """Validate a seed-distance quantile; zero/None disables the gate."""
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Correlation quantile must be between 0 and 1 (0 disables it)")
    return value if value > 0 else None


def _largest(mask):
    components, count = ndi.label(mask, structure=np.ones((3, 3)))
    if not count:
        return np.zeros(mask.shape, bool)
    areas = np.bincount(components.ravel())
    areas[0] = 0
    return components == int(areas.argmax())


def _disk(radius):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return x * x + y * y <= radius * radius


def automatic_tissue_region(gray, parent, proposal,
                            correlation_quantile=DEFAULT_CORRELATION_QUANTILE):
    """Return a native-grid mask and reproducibility diagnostics, without clicks.

    Opening suppresses narrow bridges/tails before candidate selection. Texture
    evidence can recover missing compatible tissue in a limited spatial envelope.
    Dark cavities are explicitly protected from closing; holes are never filled
    unconditionally. All output remains inside the original MedSAM mask.
    The correlation gate uses the seed-distance quantile, not a raw correlation
    coefficient cutoff. Pass None or 0 to recover the earlier acceptance rule.
    """
    quantile = validate_correlation_quantile(correlation_quantile)
    gray, parent = np.asarray(gray, float), np.asarray(parent, bool)
    labels = np.asarray(proposal["labels"])
    features = np.asarray(proposal["features"])
    if gray.ndim != 2 or gray.shape != parent.shape or labels.shape != parent.shape or not parent.any():
        raise ValueError("Expected matching image, parent mask, and proposal")
    if features.shape[:2] != parent.shape or not np.isfinite(gray[parent]).all():
        raise ValueError("Invalid proposal features or image values")
    rows, cols = np.nonzero(parent)
    crop = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
    # Work in a normalized display grid for shape operations, avoiding a square
    # pixel kernel on the strongly unequal RF row/column sampling.
    shape = (256, 256)
    small_parent = transform.resize(parent[crop].astype(float), shape, order=0,
                                    preserve_range=True, anti_aliasing=False) > 0.5
    small_labels = transform.resize(labels[crop], shape, order=0,
                                    preserve_range=True, anti_aliasing=False).astype(int)
    radius = 9
    structure = _disk(radius)
    smooth_sigma = tuple(max(0.5, n / 256 * 1.5) for n in parent[crop].shape)
    weight = ndi.gaussian_filter(parent[crop].astype(float), smooth_sigma)
    mean = ndi.gaussian_filter(np.where(parent[crop], gray[crop], 0), smooth_sigma) / np.maximum(weight, 1e-12)
    candidates = []
    candidate_stats = []
    for group in (int(g) for g in np.unique(small_labels) if g > 0):
        original = (small_labels == group) & small_parent
        opened = ndi.binary_opening(original, structure=structure)
        body = _largest(opened)
        if body.any():
            native_candidate = transform.resize(body.astype(float), parent[crop].shape, order=0,
                                                preserve_range=True, anti_aliasing=False) > 0.5
            sample = mean[native_candidate & parent[crop]]
            center = float(np.median(sample))
            dispersion = float(np.median(np.abs(sample - center))) * 1.4826
            score = int(body.sum()) / (1 + (dispersion / 0.05) ** 2)
            candidates.append((score, group, body))
            candidate_stats.append({"group": group, "compact_area": int(body.sum()),
                                    "local_mean_mad": dispersion, "score": score})
    fallback = not candidates
    if fallback:
        for group in (int(g) for g in np.unique(small_labels) if g > 0):
            body = _largest((small_labels == group) & small_parent)
            candidates.append((int(body.sum()), group, body))
    if not candidates:
        raise ValueError("No supported source proposal to refine")
    _, group, body = max(candidates, key=lambda item: (item[0], -item[1]))
    # Normalize smoothing by the parent to avoid introducing black borders.
    native_body = transform.resize(body.astype(float), parent[crop].shape, order=0,
                                    preserve_range=True, anti_aliasing=False) > 0.5
    seed = native_body & parent[crop] & (labels[crop] == group)
    values = mean[seed]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) * 1.4826
    lower = median - max(2.5 * mad, 0.05)
    upper = median + max(3.0 * mad, 0.08)
    native_features = features[crop]
    reference = native_features[seed].astype(float)
    center = np.median(reference, axis=0)
    scale = np.quantile(np.abs(reference - center), 0.85, axis=0)
    scale = np.maximum(scale, 1e-4)
    # Bounded influence prevents one redundant GLCM channel from dominating.
    distance = np.mean(np.minimum(np.abs((native_features - center) / scale), 3) ** 2, axis=-1)
    seed_distance = distance[seed]
    threshold = max(float(np.quantile(seed_distance, 0.95)) * 1.5, 1.0)
    # Co-occurrence groups can split one tissue population by orientation. Use
    # local speckle variation as a second, independent source of compatibility,
    # rather than trapping recovery inside one arbitrary source label.
    texture_sigma = tuple(max(0.5, n / 256 * 3) for n in parent[crop].shape)
    tw = np.maximum(ndi.gaussian_filter(parent[crop].astype(float), texture_sigma), 1e-12)
    tm = ndi.gaussian_filter(np.where(parent[crop], gray[crop], 0), texture_sigma) / tw
    ts = ndi.gaussian_filter(np.where(parent[crop], gray[crop] ** 2, 0), texture_sigma) / tw
    local_std = np.sqrt(np.maximum(ts - tm * tm, 0))
    std_low, std_high = np.quantile(local_std[seed], (0.05, 0.95))
    similar_speckle = (local_std >= std_low * 0.7) & (local_std <= std_high * 1.4 + 0.005)
    compatible = ((distance <= threshold) | (labels[crop] == group) | similar_speckle)
    names = list(proposal.get("feature_names", []))
    if names and len(names) != features.shape[-1]:
        raise ValueError("Feature names do not match the proposal channels")
    correlation_names = [f"{scale}_glcm_correlation" for scale in ("half_window", "full_window")]
    correlation_threshold = None
    correlation_applied = quantile is not None and all(name in names for name in correlation_names)
    if correlation_applied:
        indices = [names.index(name) for name in correlation_names]
        # Match the tested two-scale correlation distance. Source labels and
        # speckle similarity cannot bypass this appearance gate.
        correlation_distance = np.mean(np.minimum(np.abs(
            (native_features[..., indices] - center[indices]) / scale[indices]), 3) ** 2, axis=-1)
        correlation_threshold = float(np.quantile(correlation_distance[seed], quantile))
        compatible = correlation_distance <= correlation_threshold
    compatible &= (mean >= lower) & (mean <= upper) & parent[crop]
    small_compatible = transform.resize(compatible.astype(float), shape, order=0,
                                        preserve_range=True, anti_aliasing=False) > 0.5
    # Build the shape from compatible evidence across ALL source groups. Fill
    # dark cavities only in this temporary scaffold so a thin tissue rim does
    # not disappear during opening; reapply dark exclusion on the final mask.
    small_dark = transform.resize((mean < lower).astype(float), shape, order=0,
                                   preserve_range=True, anti_aliasing=False) > 0.5
    scaffold = small_compatible | (ndi.binary_fill_holes(small_compatible) & small_dark)
    opened = ndi.binary_opening(scaffold & small_parent, structure=structure)
    components, count = ndi.label(opened, structure=np.ones((3, 3)))
    if count:
        overlap = np.bincount(components[body], minlength=count + 1)
        overlap[0] = 0
        if overlap.max() > 0:
            body = components == int(overlap.argmax())
    envelope = ndi.binary_dilation(body, structure=_disk(5)) & small_parent
    result = envelope & small_compatible
    # Close only small classification gaps; independently protected dark pixels
    # will be removed again on the native grid below.
    padded = np.pad(result, 3, mode="edge")
    result = ndi.binary_closing(padded, structure=_disk(3))[3:-3, 3:-3] & envelope
    result = _largest(result)
    native = transform.resize(result.astype(float), parent[crop].shape, order=0,
                               preserve_range=True, anti_aliasing=False) > 0.5
    native &= parent[crop] & (mean >= lower)
    native = _largest(native)
    selected = np.zeros(parent.shape, bool)
    selected[crop] = native
    return selected, {"source_group": group, "analysis_shape_hw": list(shape),
        "opening_radius": radius, "boundary_recovery_radius": 5,
        "intensity_median": median, "intensity_mad": mad,
        "protected_dark_threshold": lower, "bright_rejection_threshold": upper,
        "texture_distance_threshold": threshold, "opening_fallback": fallback,
        "seed_local_std_range": [float(std_low), float(std_high)],
        "correlation_quantile": quantile,
        "correlation_gate_applied": correlation_applied,
        "correlation_distance_threshold": correlation_threshold,
        "correlation_feature_names": correlation_names if correlation_applied else [],
        "compatibility_rule": "seed_correlation_quantile" if correlation_applied else "feature_or_source_or_speckle",
        "candidate_scores": candidate_stats,
        "selected_pixels": int(selected.sum())}
