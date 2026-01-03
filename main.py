import os
from typing import Dict, List

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from torch.utils.data import DataLoader

from src.utils import set_seed, get_device
from src.data import (
    Flickr8kDataset,
    collate_batch,
    DEFAULT_CLASSES,
    read_captions,
    pick_label,
)
from src.model import MultiModalClassifier
from src.train import train_one_epoch, evaluate, collect_predictions


def plot_curves(history: Dict[str, List[float]], out_path: str, title: str):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title + " (loss)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.replace(".png", "_loss.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(title + " (accuracy)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.replace(".png", "_acc.png"))
    plt.close()


def save_error_analysis(
    y_true: List[int],
    y_pred: List[int],
    img_names: List[str],
    caps: List[str],
    idx_to_class: Dict[int, str],
    out_path: str,
    max_items: int = 30,
):
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for yt, yp, name, cap in zip(y_true, y_pred, img_names, caps):
            if yt != yp:
                f.write(f"image: {name}\n")
                f.write(f"true: {idx_to_class[yt]}\n")
                f.write(f"pred: {idx_to_class[yp]}\n")
                f.write(f"caption: {cap}\n")
                f.write("-" * 60 + "\n")
                n += 1
                if n >= max_items:
                    break


def plot_confusion_matrix(cm, class_names: List[str], out_path: str, title: str):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_experiment(
    mode: str,
    images_dir: str,
    captions_path: str,
    device: torch.device,
    out_dir: str,
    epochs: int = 5,
    batch_train: int = 16,
    batch_val: int = 32,
    seed: int = 42,
):
    # Sanity check (caption parsing + label hits)
    pairs = read_captions(captions_path)
    sample_n = min(2000, len(pairs))
    hits = 0
    for _, cap in pairs[:sample_n]:
        if pick_label(cap, DEFAULT_CLASSES) is not None:
            hits += 1
    print(f"[{mode}] raw_pairs={len(pairs)} hits_in_first_{sample_n}={hits}")

    # Strict image split datasets
    train_ds = Flickr8kDataset(
        images_dir=images_dir,
        captions_path=captions_path,
        split="train",
        train_ratio=0.8,
        seed=seed,
    )
    val_ds = Flickr8kDataset(
        images_dir=images_dir,
        captions_path=captions_path,
        split="val",
        train_ratio=0.8,
        seed=seed,
    )

    print(f"[{mode}] train_samples={len(train_ds)} val_samples={len(val_ds)} vocab={len(train_ds.vocab.itos)}")

    # Class distribution
    print(f"[{mode}] train_class_counts={train_ds.class_counts}")
    print(f"[{mode}] val_class_counts={val_ds.class_counts}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_train,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_batch,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_val,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_batch,
        pin_memory=True,
    )

    model = MultiModalClassifier(
        vocab_size=len(train_ds.vocab.itos),
        num_classes=len(DEFAULT_CLASSES),
        mode=mode,
        freeze_cnn=True,
    ).to(device)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    best_path = os.path.join(out_dir, f"best_{mode}.pt")

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device)
        va = evaluate(model, val_loader, device)

        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["acc"])
        history["val_loss"].append(va["loss"])
        history["val_acc"].append(va["acc"])

        print(
            f"[{mode}][epoch {epoch}] "
            f"train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
            f"val loss {va['loss']:.4f} acc {va['acc']:.4f}"
        )

        if va["acc"] > best_acc:
            best_acc = va["acc"]
            torch.save(model.state_dict(), best_path)

    print(f"[{mode}] best_val_acc={best_acc:.6f} saved={best_path}")

    # Reload best for analysis
    model.load_state_dict(torch.load(best_path, map_location=device))

    y_true, y_pred, img_names, caps = collect_predictions(model, val_loader, device)

    # Error analysis
    error_path = os.path.join(out_dir, f"errors_{mode}.txt")
    save_error_analysis(
        y_true=y_true,
        y_pred=y_pred,
        img_names=img_names,
        caps=caps,
        idx_to_class=val_ds.idx_to_class,
        out_path=error_path,
        max_items=30,
    )
    print(f"[{mode}] error_examples_saved={error_path}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(DEFAULT_CLASSES))))
    cm_path = os.path.join(out_dir, f"confusion_{mode}.png")
    plot_confusion_matrix(cm, DEFAULT_CLASSES, cm_path, title=f"Confusion Matrix ({mode})")
    print(f"[{mode}] confusion_matrix_saved={cm_path}")

    # Curves
    curves_base = os.path.join(out_dir, f"curves_{mode}.png")
    plot_curves(history, curves_base, title=f"Flickr8k ({mode})")
    print(f"[{mode}] curves_saved={curves_base.replace('.png','_loss.png')} and _acc.png")

    return best_acc


def main():
    set_seed(42)
    device = get_device()
    print("device:", device)

    images_dir = r"D:\MachineLearning\flickr8k\Images"
    captions_path = r"D:\MachineLearning\flickr8k\captions.txt"

    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    modes = ["image", "text", "fusion"]
    results = {}
    for m in modes:
        results[m] = run_experiment(
            mode=m,
            images_dir=images_dir,
            captions_path=captions_path,
            device=device,
            out_dir=out_dir,
            epochs=5,
            batch_train=16,
            batch_val=32,
            seed=42,
        )

    print("final_results:", results)


if __name__ == "__main__":
    main()
