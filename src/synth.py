"""Генератор синтетических кропов: текст с кириллицей, плотная обрезка как у детектора, деградации."""
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from wordfreq import top_n_list

from .inference import to_tensor
from .utils import bin_index

REQUIRED_CHARS = 'АБВЖЯабвжяёЁ' + 'ABCGQabgqy' + '0123456789'
UNITS = ['р', 'руб', 'шт', '%', 'кг', 'м²']
# Настройки текста для обучения: короткие фразы, чтобы реже отбраковывать кропы по пропорциям
TRAIN_TEXT = dict(max_ru=3, max_en=2, max_mixed=1)


@dataclass
class SynthResources:
    fonts: list
    ru_words: list
    en_words: list
    test_heights: np.ndarray  # высоты тестовых кропов: из них выбирается высота синтетического


def font_supports(path, chars=REQUIRED_CHARS):
    try:
        cmap = TTFont(path, fontNumber=0, lazy=True).getBestCmap()
        return all(ord(c) in cmap for c in chars)
    except Exception:
        return False


def load_resources(test_heights, fonts_root='/usr/share/fonts'):
    all_fonts = sorted({p for ext in ('*.ttf', '*.otf') for p in Path(fonts_root).rglob(ext)})
    fonts = [str(p) for p in all_fonts if font_supports(p)]
    ru = [w for w in top_n_list('ru', 30000) if w.isalpha() and len(w) > 1]
    en = [w for w in top_n_list('en', 10000) if w.isalpha() and len(w) > 1]
    return SynthResources(fonts, ru, en, np.asarray(test_heights))


def random_color(rng):
    return tuple(int(c) for c in rng.integers(0, 256, 3))


def luminance(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def contrasting_colors(rng, min_diff=80):
    while True:
        bg, fg = random_color(rng), random_color(rng)
        if abs(luminance(bg) - luminance(fg)) >= min_diff:
            return bg, fg


def make_text(rng, res, max_ru=4, max_en=3, max_mixed=2):
    kind = rng.random()
    if kind < 0.60:
        words = list(rng.choice(res.ru_words, rng.integers(1, max_ru + 1)))
    elif kind < 0.75:
        words = list(rng.choice(res.en_words, rng.integers(1, max_en + 1)))
    else:
        words = list(rng.choice(res.ru_words, rng.integers(1, max_mixed + 1)))
        words.insert(int(rng.integers(0, len(words) + 1)), f"{rng.integers(1, 100000)} {rng.choice(UNITS)}")
    text = ' '.join(words)
    case = rng.random()
    return text.upper() if case < 0.4 else text.title() if case < 0.7 else text


def make_background(rng, w, h, bg):
    mode = rng.random()
    base = np.array(bg, dtype=np.float32)
    if mode < 0.5:
        arr = np.broadcast_to(base, (h, w, 3))
    elif mode < 0.8:
        # Сдвиг второго цвета ограничен, чтобы градиент не съел контраст с текстом
        end = np.clip(base + rng.uniform(-60, 60, 3), 0, 255)
        t = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
        arr = np.broadcast_to(base * (1 - t) + end * t, (h, w, 3))
    else:
        noise = rng.normal(0, 25, (h, w, 3)).astype(np.float32)
        arr = base + np.asarray(Image.fromarray(np.clip(noise + 128, 0, 255).astype(np.uint8))
                                .filter(ImageFilter.GaussianBlur(2)), dtype=np.float32) - 128
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def render_crop(rng, res, accept_fn=None, **text_kw):
    """Строка текста на фоне, обрезанная с маленькими случайными полями.

    Поля сверху, снизу и по бокам выбираются независимо, чтобы их размер не выдавал ориентацию.
    Если задан accept_fn, пропорции проверяются до рисования: отбракованный кроп почти ничего не стоит.
    """
    text = make_text(rng, res, **text_kw)
    font = ImageFont.truetype(str(rng.choice(res.fonts)), int(rng.integers(28, 97)))
    bg, fg = contrasting_colors(rng)
    stroke = int(rng.integers(1, 4)) if rng.random() < 0.15 else 0

    x0, y0, x1, y1 = font.getbbox(text, stroke_width=stroke)
    tw, th = x1 - x0, y1 - y0
    mt, mb = (rng.uniform(-0.05, 0.20, 2) * th).astype(int)
    ml, mr = (rng.uniform(-0.03, 0.15, 2) * th).astype(int)
    W, H = max(tw + ml + mr, 8), max(th + mt + mb, 8)

    if accept_fn is not None and not accept_fn(W / H):
        return None, text

    img = make_background(rng, W, H, bg)
    ImageDraw.Draw(img).text((ml - x0, mt - y0), text, font=font, fill=fg,
                             stroke_width=stroke, stroke_fill=random_color(rng))
    return img, text


def degrade(img, rng, test_heights):
    h = int(rng.choice(test_heights))
    w = max(8, round(img.width * h / img.height))
    img = img.resize((w, h), Image.BILINEAR)
    if rng.random() < 0.3:
        f = rng.uniform(1.5, 3.0)
        small = img.resize((max(4, round(w / f)), max(4, round(h / f))), Image.BILINEAR)
        img = small.resize((w, h), Image.BILINEAR)
    if rng.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    if rng.random() < 0.25:
        arr = np.asarray(img, dtype=np.float32) + rng.normal(0, rng.uniform(3, 12), (h, w, 3))
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.7:
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=int(rng.integers(30, 96)))
        img = Image.open(io.BytesIO(buf.getvalue())).convert('RGB')
    return img


def make_sample(rng, res, accept_fn=None, **text_kw):
    img = None
    while img is None:
        img, text = render_crop(rng, res, accept_fn, **text_kw)
    y = int(rng.integers(0, 2))
    # Поворот до деградаций: иначе сетка JPEG у перевёрнутых кропов смещена и выдаёт класс
    if y == 1:
        img = img.transpose(Image.Transpose.ROTATE_180)
    return degrade(img, rng, res.test_heights), y, text


def build_synth_validation(res, out_dir, shares, n, seed):
    """Синтетическая валидация: n кропов, доли интервалов пропорций как в тесте (по квотам)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    quota = (np.asarray(shares) * n).round().astype(int)
    filled = np.zeros(len(quota), dtype=int)
    records = []
    while (filled < quota).any():
        img, y, text = make_sample(rng, res)
        b = bin_index(img.width / img.height)
        if filled[b] >= quota[b]:
            continue
        filled[b] += 1
        crop_id = f"syn_{len(records):05d}.png"
        img.save(out_dir / crop_id)
        records.append({'crop_id': crop_id, 'y': y, 'text': text, 'w': img.width, 'h': img.height, 'bin': b})
    crops = pd.DataFrame(records)
    crops['aspect'] = crops.w / crops.h
    crops['weight'] = 1.0
    return crops


def estimate_acceptance(res, shares, seed, n=5000, **text_kw):
    """Вероятность принять кроп из каждого интервала пропорций, чтобы доли совпали с тестом."""
    rng = np.random.default_rng(seed)
    counts = np.zeros(len(shares))

    def record(aspect):
        counts[bin_index(aspect)] += 1
        return False

    for _ in range(n):
        render_crop(rng, res, accept_fn=record, **text_kw)
    accept = np.asarray(shares) / np.maximum(counts / counts.sum(), 1e-6)
    return accept / accept.max()


class SynthOrientationDataset(torch.utils.data.Dataset):
    """Обучающие примеры, которые генерируются на лету: (тензор 3×80×160, метка)."""

    def __init__(self, res, accept, n, seed, text_kw=TRAIN_TEXT):
        self.res, self.accept, self.n, self.seed, self.text_kw = res, accept, n, seed, text_kw
        self.epoch = 0

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        # Генератор на каждый пример: картинка зависит только от (seed, epoch, i),
        # а не от порядка загрузки и числа процессов DataLoader
        rng = np.random.default_rng([self.seed, self.epoch, i])
        img, y, _ = make_sample(rng, self.res, lambda a: rng.random() < self.accept[bin_index(a)], **self.text_kw)
        return to_tensor(img), y
