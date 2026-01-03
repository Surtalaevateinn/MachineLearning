import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


CLASS_ALIASES: Dict[str, List[str]] = {
    "dog": ["dog", "dogs", "puppy", "puppies"],
    "man": ["man", "men", "guy", "guys"],
    "woman": ["woman", "women", "lady", "ladies"],
    "child": ["child", "children", "kid", "kids"],
    "boy": ["boy", "boys"],
    "girl": ["girl", "girls"],
    "bike": ["bike", "bikes", "bicycle", "bicycles"],
    "car": ["car", "cars"],
    "ball": ["ball", "balls"],
    "horse": ["horse", "horses"],
}

DEFAULT_CLASSES: List[str] = list(CLASS_ALIASES.keys())


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def simple_tokenize(s: str) -> List[str]:
    return normalize_text(s).split()


def pick_label(caption: str, classes: List[str]) -> Optional[str]:
    cap = " " + normalize_text(caption) + " "
    for cls in classes:
        for alias in CLASS_ALIASES.get(cls, [cls]):
            if f" {alias} " in cap:
                return cls
    return None


def read_captions(captions_path: str) -> List[Tuple[str, str]]:
    """
    Supports:
      1) image.jpg<TAB>caption
      2) image.jpg,caption (CSV; may have a header)
      3) image.jpg#0<TAB>caption (removes #0 suffix)
    """
    pairs: List[Tuple[str, str]] = []
    with open(captions_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            low = line.lower()
            if low.startswith("image") and ("caption" in low or "comment" in low):
                continue

            img, cap = None, None
            if "\t" in line:
                img, cap = line.split("\t", 1)
            elif "," in line:
                img, cap = line.split(",", 1)
            else:
                continue

            img = img.strip()
            cap = cap.strip()

            if "#" in img:
                img = img.split("#", 1)[0]

            if not img.lower().endswith(".jpg"):
                continue
            if not cap:
                continue

            pairs.append((img, cap))

    return pairs


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad_id: int
    unk_id: int


def build_vocab(token_lists: List[List[str]], min_freq: int = 2) -> Vocab:
    freq: Dict[str, int] = {}
    for toks in token_lists:
        for t in toks:
            freq[t] = freq.get(t, 0) + 1

    itos = ["<pad>", "<unk>"]
    for w, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
        if c >= min_freq:
            itos.append(w)

    stoi = {w: i for i, w in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos, pad_id=stoi["<pad>"], unk_id=stoi["<unk>"])


def encode(tokens: List[str], vocab: Vocab, max_len: int) -> torch.Tensor:
    ids = [vocab.stoi.get(t, vocab.unk_id) for t in tokens][:max_len]
    if len(ids) < max_len:
        ids += [vocab.pad_id] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def split_images_strict(
    image_names: List[str],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[set, set]:
    uniq = sorted(set(image_names))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(uniq), generator=g).tolist()
    cut = int(len(uniq) * train_ratio)
    train_imgs = {uniq[i] for i in perm[:cut]}
    val_imgs = {uniq[i] for i in perm[cut:]}
    return train_imgs, val_imgs


class Flickr8kDataset(Dataset):
    """
    Strict image-level split:
      - all captions of an image belong to the same split
    """
    def __init__(
        self,
        images_dir: str,
        captions_path: str,
        split: str = "train",
        train_ratio: float = 0.8,
        max_len: int = 25,
        min_freq: int = 2,
        seed: int = 42,
        classes: Optional[List[str]] = None,
    ):
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")

        self.images_dir = images_dir
        self.classes = classes if classes is not None else DEFAULT_CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        self.max_len = max_len

        pairs = read_captions(captions_path)

        # First pass: keep only caption samples that match a label
        kept_all = []
        tokens_all: List[List[str]] = []
        image_list_for_split = []

        for img, cap in pairs:
            lbl = pick_label(cap, self.classes)
            if lbl is None:
                continue
            toks = simple_tokenize(cap)
            if not toks:
                continue
            kept_all.append((img, cap, toks, lbl))
            tokens_all.append(toks)
            image_list_for_split.append(img)

        if not kept_all:
            raise RuntimeError("No samples matched labels. Check caption format or label aliases.")

        # Build vocab from all kept tokens (simple & stable)
        self.vocab = build_vocab(tokens_all, min_freq=min_freq)

        # Strict image split
        train_imgs, val_imgs = split_images_strict(image_list_for_split, train_ratio=train_ratio, seed=seed)
        target_imgs = train_imgs if split == "train" else val_imgs

        # Filter samples by target image set
        self.samples = []
        for img, cap, toks, lbl in kept_all:
            if img in target_imgs:
                self.samples.append((img, cap, toks, lbl))

        if not self.samples:
            raise RuntimeError("Split produced zero samples. Try different seed or train_ratio.")

        # Class counts (for reporting)
        self.class_counts: Dict[str, int] = {c: 0 for c in self.classes}
        for _, _, _, lbl in self.samples:
            self.class_counts[lbl] += 1

        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        img_name, cap, toks, lbl = self.samples[i]
        y = self.class_to_idx[lbl]

        img_path = f"{self.images_dir}\\{img_name}"
        with Image.open(img_path) as im:
            im = im.convert("RGB")
        x_img = self.tf(im)

        x_txt = encode(toks, self.vocab, self.max_len)
        return x_img, x_txt, torch.tensor(y, dtype=torch.long), img_name, cap


def collate_batch(batch):
    x_img = torch.stack([b[0] for b in batch], dim=0)
    x_txt = torch.stack([b[1] for b in batch], dim=0)
    y = torch.stack([b[2] for b in batch], dim=0)
    img_names = [b[3] for b in batch]
    caps = [b[4] for b in batch]
    return x_img, x_txt, y, img_names, caps
