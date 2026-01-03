from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, device) -> Dict[str, float]:
    model.train()
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0

    for x_img, x_txt, y, _, _ in tqdm(loader, leave=False, desc="train"):
        x_img, x_txt, y = x_img.to(device), x_txt.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_img, x_txt)
        loss = ce(logits, y)
        loss.backward()
        optimizer.step()

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item() * y.size(0)

    return {"loss": loss_sum / total, "acc": correct / total}


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0

    for x_img, x_txt, y, _, _ in tqdm(loader, leave=False, desc="val"):
        x_img, x_txt, y = x_img.to(device), x_txt.to(device), y.to(device)
        logits = model(x_img, x_txt)
        loss = ce(logits, y)

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item() * y.size(0)

    return {"loss": loss_sum / total, "acc": correct / total}


@torch.no_grad()
def collect_predictions(
    model, loader, device
) -> Tuple[List[int], List[int], List[str], List[str]]:
    """
    Returns:
      y_true, y_pred, image_names, captions
    """
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    img_names: List[str] = []
    caps: List[str] = []

    for x_img, x_txt, y, names, captions in loader:
        x_img, x_txt = x_img.to(device), x_txt.to(device)
        logits = model(x_img, x_txt)
        pred = logits.argmax(dim=1).cpu().tolist()

        y_true.extend(y.tolist())
        y_pred.extend(pred)
        img_names.extend(names)
        caps.extend(captions)

    return y_true, y_pred, img_names, caps
