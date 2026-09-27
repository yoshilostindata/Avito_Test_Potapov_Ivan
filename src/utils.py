"""Воспроизводимость, метрика и интервалы пропорций кропов."""
import os
import random

import numpy as np
import torch

# Интервалы отношения ширины к высоте вида (a, b]
ASPECT_BINS = np.array([1.5, 2, 3, 5, 8, 12, np.inf])


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def brier_score(y, p, weights=None):
    return float(np.average((np.asarray(p) - np.asarray(y)) ** 2, weights=weights))


def bin_index(aspect):
    """Номер интервала пропорций; всё, что не больше 1.5, относится к первому."""
    return max(int(np.searchsorted(ASPECT_BINS, aspect, side='left')) - 1, 0)


def test_shares(aspects):
    """Доли интервалов пропорций среди тестовых кропов.

    Кропы с отношением сторон не больше 1.5 (0.3% теста) в долях не учитываются.
    """
    aspects = np.asarray(aspects)
    bins = [bin_index(a) for a in aspects[aspects > 1.5]]
    return np.bincount(bins, minlength=len(ASPECT_BINS) - 1) / len(bins)
