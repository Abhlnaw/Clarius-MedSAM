"""Reference-mask evaluation on native binary arrays, independent of the viewer."""
import numpy as np
from skimage.metrics import structural_similarity


SSIM_SETTINGS = dict(data_range=1.0, win_size=11, gaussian_weights=True,
                     sigma=1.5, use_sample_covariance=False, K1=0.01, K2=0.03)


def binary_mask(values, name="mask"):
    values = np.asarray(values)
    if values.ndim != 2 or not values.size:
        raise ValueError(f"{name} must be a nonempty 2D array")
    if not np.isfinite(values).all() or not np.all((values == 0) | (values == 1)):
        raise ValueError(f"{name} must contain only binary 0/1 values")
    return values.astype(bool)


def evaluate_mask(prediction, reference, evaluation_mask=None):
    """Evaluate a prediction against a reference; never clip either to the ROI.

    Standard full-frame SSIM follows the Wang/scikit-image Gaussian recipe.
    roi_mean_ssim is a separately named, custom mean of the SAME local SSIM map
    over valid centers in a common evaluation mask. Frame-border centers are
    excluded consistently with standard SSIM. Dice/IoU always use full arrays.
    Shared background can inflate full-frame SSIM. Neither score certifies the
    reference's anatomical correctness. No resizing or intensity overlays occur.
    """
    prediction, reference = binary_mask(prediction, 'prediction'), binary_mask(reference, 'reference')
    if prediction.shape != reference.shape:
        raise ValueError("Prediction and reference must have identical native shapes")
    if min(reference.shape) < SSIM_SETTINGS['win_size']:
        raise ValueError("SSIM requires at least 11 pixels along each image dimension")
    if evaluation_mask is None:
        evaluation_mask = np.ones(reference.shape, bool)
    else:
        evaluation_mask = binary_mask(evaluation_mask, 'evaluation mask')
        if evaluation_mask.shape != reference.shape:
            raise ValueError("Evaluation mask must have the same shape as the reference")
    # Do not allow a chosen ROI to silently hide prediction/reference positives.
    if np.any((prediction | reference) & ~evaluation_mask):
        raise ValueError("Evaluation mask must include all prediction and reference foreground")
    score, local = structural_similarity(reference.astype(float), prediction.astype(float),
                                         full=True, **SSIM_SETTINGS)
    valid = np.zeros(reference.shape, bool)
    pad = SSIM_SETTINGS['win_size'] // 2
    valid[pad:-pad, pad:-pad] = True
    centers = valid & evaluation_mask
    tp = int(np.count_nonzero(prediction & reference))
    fp = int(np.count_nonzero(prediction & ~reference))
    fn = int(np.count_nonzero(~prediction & reference))
    total = 2 * tp + fp + fn
    union = tp + fp + fn
    return dict(ssim=float(score),
        roi_mean_ssim=float(local[centers].mean()) if centers.any() else None,
        roi_ssim_centers=int(centers.sum()), dice=2*tp/total if total else 1.0,
        iou=tp/union if union else 1.0,
        precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None,
        true_positive_pixels=tp, false_positive_pixels=fp, false_negative_pixels=fn,
        prediction_pixels=int(prediction.sum()), reference_pixels=int(reference.sum()))
