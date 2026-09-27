"""Предсказание вероятности поворота текста на 180°: предобработка, два прогона, стэкинг."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageClassification

from .config import MODEL_ID, MODEL_REVISION, INPUT_HEIGHT, INPUT_WIDTH, IMAGE_MEAN, IMAGE_STD

_MEAN = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGE_STD).view(3, 1, 1)


def to_tensor(im, w=INPUT_WIDTH, h=INPUT_HEIGHT):
    """Повторяет PPLCNetImageProcessor с точностью до округления.

    Ресайз до h×w без сохранения пропорций (сохранение пропорций на тесте ухудшает качество),
    нормализация в порядке RGB, затем разворот каналов в BGR — так модель обучалась в PaddleOCR.
    """
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255).permute(2, 0, 1)
    x = F.interpolate(x[None], size=(h, w), mode='bilinear', align_corners=False, antialias=True)[0]
    return ((x - _MEAN) / _STD)[[2, 1, 0]]


def get_logits(out):
    
    return out.logits if getattr(out, 'logits', None) is not None else out.last_hidden_state


def load_model(device, weights=None):
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(device)
    if weights is not None:
        model.load_state_dict(torch.load(weights, map_location=device))
    return model.eval()


def class_180_index(model):
    return next(int(k) for k, v in model.config.id2label.items() if '180' in str(v))


@torch.no_grad()
def predict_p180(model, imgs, device):
    x = torch.stack([to_tensor(im) for im in imgs]).to(device)
    return get_logits(model(pixel_values=x)).softmax(-1)[:, class_180_index(model)].cpu().numpy()


def two_passes(model, paths, aspects, device, batch=256):
    """Вероятность поворота для каждого кропа и для его копии, повёрнутой на 180°.

    Кропы обрабатываются в порядке возрастания отношения сторон: порядок сохранён,
    чтобы численно в точности повторить отправленный сабмит.
    """
    order = np.argsort(aspects)
    p_orig, p_rot = np.empty(len(paths)), np.empty(len(paths))
    for i in range(0, len(paths), batch):
        idx = order[i:i + batch]
        imgs = [Image.open(paths[j]).convert('RGB') for j in idx]
        p_orig[idx] = predict_p180(model, imgs, device)
        p_rot[idx] = predict_p180(model, [im.transpose(Image.Transpose.ROTATE_180) for im in imgs], device)
    return p_orig, p_rot


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def orientation_feature(p_orig, p_rot):
    """Признак для стэкинга: разность логитов исходного и повёрнутого кропа.

    Антисимметричен: если поменять кроп и его поворот местами, признак меняет знак,
    и итоговая вероятность превращается в 1 − p.
    """
    return logit(p_orig) - logit(p_rot)


def stack_proba(coefs, features):
    return 1 / (1 + np.exp(-np.clip(features @ coefs, -50, 50)))
