"""Оценка на валидациях и подбор коэффициентов стэкинга."""
import numpy as np
import torch
from PIL import Image
from scipy.optimize import minimize

from .inference import get_logits, class_180_index, to_tensor, orientation_feature, stack_proba
from .utils import brier_score

# Вес синтетики в смеси валидаций. Подобран по результатам сабмитов на тесте:
# с ним смесь совпала с тестом для ансамбля (0.960) и для исходной модели (0.925 против 0.922)
ALPHA = 0.54


def stretch_predictor(model, device):
    idx = class_180_index(model)

    @torch.no_grad()
    def predict(imgs):
        x = torch.stack([to_tensor(im) for im in imgs]).to(device)
        return get_logits(model(pixel_values=x)).softmax(-1)[:, idx].cpu().numpy()
    return predict


def two_passes_frame(predict_fn, frame, load_fn, batch=256):
    """Два прогона (кроп и его поворот) для кропов из таблицы; load_fn(row) -> PIL.Image."""
    order = np.argsort(frame.aspect.values)
    p_orig, p_rot = np.empty(len(frame)), np.empty(len(frame))
    for i in range(0, len(frame), batch):
        idx = order[i:i + batch]
        imgs = [load_fn(r) for r in frame.iloc[idx].itertuples()]
        p_orig[idx] = predict_fn(imgs)
        p_rot[idx] = predict_fn([im.transpose(Image.Transpose.ROTATE_180) for im in imgs])
    return p_orig, p_rot


def score(frame, p, mask=None):
    """Взвешенный 1 − Brier на части валидации."""
    mask = np.ones(len(frame), bool) if mask is None else mask
    return 1 - brier_score(frame.y.values[mask], p[mask], frame.weight.values[mask])


def split_halves(rus_val, syn_val, seed):
    """Половины A (подбор параметров) и B (проверка).

    RusTitW делится по фотографиям, чтобы кропы одного фото не оказались в обеих половинах.
    """
    names = rus_val.image_name.unique()
    half_a = set(np.random.default_rng(seed).permutation(names)[:len(names) // 2])
    rus_a = rus_val.image_name.isin(half_a).values
    syn_a = np.random.default_rng(seed).random(len(syn_val)) < 0.5
    return {'A': (rus_a, syn_a), 'B': (~rus_a, ~syn_a)}


def mixed_score(coefs, feats, rus_val, syn_val, masks, alpha=ALPHA):
    """Смесь валидаций: alpha × синтетика + (1 − alpha) × RusTitW для стэкинга с коэффициентами coefs."""
    rus_mask, syn_mask = masks
    s = score(syn_val, stack_proba(coefs, feats['синтетика']), syn_mask)
    r = score(rus_val, stack_proba(coefs, feats['RusTitW']), rus_mask)
    return alpha * s + (1 - alpha) * r


def fit_stacking(feats, rus_val, syn_val, halves, x0=(0.25, 0.25)):
    res = minimize(lambda c: -mixed_score(c, feats, rus_val, syn_val, halves['A']),
                   x0=list(x0), method='Nelder-Mead')
    return res.x
