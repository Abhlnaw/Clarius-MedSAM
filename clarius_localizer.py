"""Small image-only liver bounding-box regressor; no texture or mask inference."""
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

INPUT_HW = (128, 224)


def mask_box(mask):
    """Normalized xyxy box of nonempty foreground, with exclusive max edges."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError('Expected a nonempty 2D reference mask')
    y, x = np.nonzero(mask)
    h, w = mask.shape
    return np.array([x.min()/w, y.min()/h, (x.max()+1)/w, (y.max()+1)/h], np.float32)


def decode_box(raw):
    """Differentiable ordered boxes entirely inside the normalized image."""
    start = raw[..., :2].sigmoid()
    end = start + (1-start) * raw[..., 2:].sigmoid()
    return torch.cat((start, end), -1)


def pad_box(box, fraction=.03):
    box = np.asarray(box, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError('Expected a finite ordered xyxy box')
    if not 0 <= fraction < .5:
        raise ValueError('Padding must be a fraction in [0, .5)')
    return np.clip(box + [-fraction, -fraction, fraction, fraction], 0, 1)


def read_image(path):
    # MATLAB already formed the B-mode image. Only model input normalization
    # and resizing happen here; the source export is never changed.
    with Image.open(path) as image:
        array = np.asarray(image.convert('L').resize(INPUT_HW[::-1], Image.Resampling.BILINEAR), dtype=np.float32).copy()
    array = (array-array.min()) / max(float(np.ptp(array)), 1e-8)
    return torch.from_numpy(array)[None]


class LiverBoxNet(nn.Module):
    def __init__(self):
        super().__init__()
        blocks, channels = [], 1
        for width in (16, 32, 64, 96):
            blocks += [nn.Conv2d(channels, width, 3, stride=2, padding=1, bias=False),
                       nn.GroupNorm(8, width), nn.SiLU()]
            channels = width
        self.features = nn.Sequential(*blocks, nn.AvgPool2d(2))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(96*4*7, 128), nn.SiLU(),
                                  nn.Dropout(.2), nn.Linear(128, 4))

    def initialize_box(self, box):
        box = torch.as_tensor(box).clamp(.001, .999)
        parameters = torch.cat((box[:2], (box[2:]-box[:2])/(1-box[:2]))).clamp(.001, .999)
        with torch.no_grad():
            self.head[-1].weight.zero_()
            self.head[-1].bias.copy_(torch.logit(parameters))

    def forward(self, image):
        return decode_box(self.head(self.features(image)))


def grouped_folds(cases, folds=5, seed=2026):
    """Disjoint exam groups, one outer test appearance per case."""
    groups = np.array(sorted({str(case['exam']) for case in cases}))
    if len(groups) < folds + 2 or folds < 3:
        raise ValueError('Need at least folds+2 exam groups and at least 3 folds')
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    chunks = np.array_split(groups, folds)
    result = []
    for index, held_out in enumerate(chunks):
        remaining = groups[~np.isin(groups, held_out)]
        # Keep inner validation independent of outer test labels and scores.
        val_count = max(1, round(len(remaining)*.2))
        validation = remaining[:val_count]
        training = remaining[val_count:]
        role = {exam:name for name, exams in [('train',training),('validation',validation),('test',held_out)] for exam in exams}
        result.append({name:[i for i,c in enumerate(cases) if role[str(c['exam'])] == name]
                       for name in ('train','validation','test')})
    return result


def load_localizer(checkpoint, device='cpu'):
    saved = torch.load(Path(checkpoint), map_location=device, weights_only=True)
    if tuple(saved['input_hw']) != INPUT_HW:
        raise ValueError('Localizer input dimensions differ')
    model = LiverBoxNet().to(device)
    model.load_state_dict(saved['state_dict'])
    return model.eval(), saved


@torch.inference_mode()
def predict_box(model, path, device='cpu', padding=.03):
    box = model(read_image(path)[None].to(device))[0].cpu().numpy()
    with Image.open(path) as image:
        w, h = image.size
    return pad_box(box, padding) * [w, h, w, h]
