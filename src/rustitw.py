"""Кропы из датасета RusTitW (реальные фото с кириллицей) для валидации и обучения."""
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .utils import bin_index, test_shares

# Перенос строки в разметке записан либо как \n, либо как \/n
LINE_SEP_RE = r'\n|\\/n'
MIN_SIDE = 8


def parse_boxes(info, min_aspect=1.5):
    """Однострочные прямоугольные рамки с отношением ширины к высоте не меньше min_aspect.

    Узкие рамки отбрасываются: в RusTitW это вертикальный текст (каждый столбец размечен
    отдельно) и почти квадратные рамки с большим запасом фона — в тесте таких кропов нет.
    """
    rows = []
    for r in info.itertuples():
        for group in json.loads(r.box_and_label):
            for b in group:
                rows.append({'image_name': r.image_name, 'shape': b.get('shape'), 'label': b.get('label', ''),
                             'x': b['left'] * r.width, 'y': b['top'] * r.height,
                             'w': b['width'] * r.width, 'h': b['height'] * r.height})
    boxes = pd.DataFrame(rows)
    boxes['n_lines'] = boxes.label.str.count(LINE_SEP_RE) + 1
    boxes['aspect'] = boxes.w / boxes.h
    keep = (boxes.n_lines == 1) & (boxes['shape'] == 'rectangle') & (boxes.aspect >= min_aspect)
    return boxes[keep].reset_index(drop=True)


def cut_crops(info, boxes, images_dir, out_dir, prefix, min_side=MIN_SIDE):
    out_dir.mkdir(parents=True, exist_ok=True)
    img_size = info.set_index('image_name')[['width', 'height']]
    records = []
    for img_name, group in boxes.groupby('image_name'):
        im = Image.open(images_dir / img_name)
        expected = tuple(img_size.loc[img_name])
        if im.size != expected:
            # Разметка сделана по фото с учётом EXIF-поворота, а PIL сам его не применяет
            im = ImageOps.exif_transpose(im)
            if im.size != expected:
                continue
        im = im.convert('RGB')
        W, H = im.size
        for b in group.itertuples():
            x0, y0 = max(0, round(b.x)), max(0, round(b.y))
            x1, y1 = min(W, round(b.x + b.w)), min(H, round(b.y + b.h))
            if x1 - x0 < min_side or y1 - y0 < min_side:
                continue
            crop_id = f"{prefix}_{len(records):05d}.png"
            im.crop((x0, y0, x1, y1)).save(out_dir / crop_id)
            records.append({'crop_id': crop_id, 'image_name': img_name, 'text': b.label,
                            'w': x1 - x0, 'h': y1 - y0, 'box_aspect': b.aspect})
    crops = pd.DataFrame(records)
    crops['aspect'] = crops.w / crops.h
    return crops


def build_validation(info, images_dir, out_dir, test_aspects, seed):
    """Валидация из RusTitW: кропы, метки поворота и веса под распределение пропорций теста.

    Метка 1 означает, что кроп при чтении поворачивается на 180° (см. load_crop).
    Метки назначаются до удаления рамок с отношением сторон ровно 1.5: в таком порядке
    валидация в точности совпадает с той, на которой подбирались параметры решения.
    """
    crops = cut_crops(info, parse_boxes(info), images_dir, out_dir, 'val')
    crops['y'] = np.random.default_rng(seed).permutation(np.arange(len(crops)) % 2)
    crops = crops[crops.box_aspect > 1.5].reset_index(drop=True)

    crops['bin'] = [bin_index(a) for a in crops.box_aspect]
    crop_share = np.bincount(crops.bin, minlength=len(test_shares(test_aspects))) / len(crops)
    weights = (test_shares(test_aspects) / crop_share)[crops.bin]
    crops['weight'] = weights / weights.mean()
    return crops


def load_crop(crops_dir, row):
    im = Image.open(crops_dir / row.crop_id).convert('RGB')
    return im.transpose(Image.Transpose.ROTATE_180) if row.y == 1 else im
