"""Дообучение PP-LCNet на сгенерированных примерах."""
import numpy as np
import torch
from tqdm.auto import tqdm

from .inference import get_logits, load_model
from .utils import set_seed


def train_model(dataset, device, ckpt_path, seed, epochs=4, lr=3e-4, batch=128, num_workers=2):
    set_seed(seed)
    model = load_model(device)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, num_workers=num_workers, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(loader), pct_start=0.1)
    # Сглаживание меток снижает переуверенность модели, которую сильно штрафует Brier Score
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    for epoch in range(1, epochs + 1):
        # Датасет копируется в процессы загрузки при каждом новом проходе, поэтому номер эпохи доходит до них
        dataset.epoch = epoch
        model.train()
        losses = []
        for x, y in tqdm(loader, desc=f'эпоха {epoch}'):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = loss_fn(get_logits(model(pixel_values=x)), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        print(f"эпоха {epoch}: loss {np.mean(losses):.4f}")

    torch.save(model.state_dict(), ckpt_path)
    return model.eval()
