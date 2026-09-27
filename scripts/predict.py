"""Предсказание для тестовых кропов.

Пример: python scripts/predict.py --data test.zip --out submission.csv
"""
import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference import load_model, two_passes, orientation_feature, stack_proba

IMG_EXT = {'.png', '.jpg', '.jpeg', '.webp'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='zip-архив или папка с кропами и sample_submission.csv')
    parser.add_argument('--weights', default=ROOT / 'weights', type=Path)
    parser.add_argument('--out', default='submission.csv')
    parser.add_argument('--batch', default=256, type=int)
    args = parser.parse_args()

    data = Path(args.data)
    if data.suffix == '.zip':
        extracted = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(data) as z:
            z.extractall(extracted)
        data = extracted

    images = sorted(p for p in data.rglob('*') if p.suffix.lower() in IMG_EXT)
    sample = pd.read_csv(next(data.rglob('sample_submission.csv')))
    aspects = np.array([w / h for w, h in (Image.open(p).size for p in images)])
    print(f"Картинок: {len(images)}")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    models = [load_model(device), load_model(device, args.weights / 'ft_best.pt')]
    features = np.stack([orientation_feature(*two_passes(m, images, aspects, device, args.batch))
                         for m in models], axis=1)
    p_180 = stack_proba(np.load(args.weights / 'stacking_coefs.npy'), features)

    preds = pd.DataFrame({'image_id': [p.stem for p in images], 'p_180': p_180})
    sub = sample[['image_id']].merge(preds, on='image_id', how='left')
    assert sub.p_180.notna().all(), "Для части image_id из sample_submission нет картинок"
    sub.to_csv(args.out, index=False, float_format='%.6f')
    print(f"Сохранено: {args.out}")


if __name__ == '__main__':
    main()
