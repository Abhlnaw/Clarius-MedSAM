"""Display-only ultrasound contrast experiments; no segmentation or synthesis."""
import numpy as np
from scipy.signal import hilbert
from skimage import exposure


def _image(values):
    values = np.asarray(values)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError('Expected a nonempty finite 2D image')
    return values


def rf_envelope(rf):
    """Analytic-signal magnitude along depth (rows), independently per line."""
    return np.abs(hilbert(_image(rf).astype(np.float64), axis=0))


def rf_legacy(envelope):
    """Reproduce the existing per-frame log1p/min-max gray mapping."""
    db = 20 * np.log10(1 + _image(envelope))
    span = float(np.ptp(db))
    return np.zeros(db.shape, np.uint8) if span == 0 else np.floor(
        np.clip((db-db.min())/span, 0, 1)*255+.5).astype(np.uint8)


def rf_window(envelope, dynamic_range_db=50, white_percentile=99.9, gamma=1):
    """Window amplitude dB below a robust bright reference; gamma >1 darkens.

    Zero amplitude maps to black. The bright reference is per-frame, so this
    is a display normalization, not calibrated backscatter or fixed gain.
    """
    envelope = _image(envelope).astype(np.float64)
    if (envelope < 0).any():
        raise ValueError('Envelope cannot be negative')
    if not np.isfinite(dynamic_range_db) or dynamic_range_db <= 0:
        raise ValueError('Dynamic range must be positive and finite')
    if not np.isfinite(gamma) or gamma <= 0 or not 0 < white_percentile <= 100:
        raise ValueError('Invalid gamma or white percentile')
    reference = float(np.percentile(envelope, white_percentile))
    if reference <= 0:
        return np.zeros(envelope.shape, np.uint8), dict(white_amplitude=0,
            dynamic_range_db=float(dynamic_range_db), gamma=float(gamma))
    relative = np.maximum(envelope/reference, np.finfo(float).tiny)
    db = 20*np.log10(relative)
    normalized = np.clip((db+dynamic_range_db)/dynamic_range_db, 0, 1)**gamma
    pixels = np.floor(normalized*255+.5).astype(np.uint8)
    return pixels, dict(white_amplitude=reference, white_percentile=float(white_percentile),
        dynamic_range_db=float(dynamic_range_db), gamma=float(gamma),
        black_relative_db=-float(dynamic_range_db), white_relative_db=0)


def envelope_tone(gray, black=0, white=255, gamma=1):
    """Fixed pointwise tone curve on an already 8-bit B-mode stream."""
    gray = _image(gray)
    if gray.dtype != np.uint8:
        raise ValueError('Expected the original uint8 B-mode samples')
    if not 0 <= black < white <= 255 or not np.isfinite(gamma) or gamma <= 0:
        raise ValueError('Invalid tone window or gamma')
    normalized = np.clip((gray.astype(float)-black)/(white-black), 0, 1)**gamma
    return np.floor(normalized*255+.5).astype(np.uint8)


def mild_clahe(gray):
    """Exploratory local enhancement, not the default or recovered Clarius LUT."""
    gray = _image(gray)
    if gray.dtype != np.uint8:
        raise ValueError('Expected uint8 image')
    if np.ptp(gray) == 0:
        return gray.copy()
    result = exposure.equalize_adapthist(gray,
        kernel_size=(max(8, gray.shape[0]//8), max(8, gray.shape[1]//8)),
        clip_limit=.01, nbins=256)
    return np.floor(result*255+.5).astype(np.uint8)
