import torch
import torch.nn as nn
from torchvision import models


class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 256, freeze: bool = True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # (B,512,1,1)
        self.fc = nn.Linear(512, out_dim)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x).flatten(1)  # (B,512)
        return self.fc(x)                # (B,out_dim)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 200, hid_dim: int = 256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(x_ids)      # (B,T,E)
        _, h = self.gru(x)       # (1,B,H)
        return h.squeeze(0)      # (B,H)


class MultiModalClassifier(nn.Module):
    """
    mode:
      - "image": image-only
      - "text":  text-only
      - "fusion": concat(image,text)
    """
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        mode: str = "fusion",
        freeze_cnn: bool = True,
        img_dim: int = 256,
        txt_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        if mode not in ("image", "text", "fusion"):
            raise ValueError("mode must be one of: image, text, fusion")
        self.mode = mode

        self.img_enc = ImageEncoder(out_dim=img_dim, freeze=freeze_cnn)
        self.txt_enc = TextEncoder(vocab_size=vocab_size, emb_dim=200, hid_dim=txt_dim)

        in_dim = img_dim if mode == "image" else txt_dim if mode == "text" else (img_dim + txt_dim)

        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x_img: torch.Tensor, x_txt: torch.Tensor) -> torch.Tensor:
        if self.mode == "image":
            v = self.img_enc(x_img)
            return self.head(v)
        if self.mode == "text":
            v = self.txt_enc(x_txt)
            return self.head(v)

        v_img = self.img_enc(x_img)
        v_txt = self.txt_enc(x_txt)
        z = torch.cat([v_img, v_txt], dim=1)
        return self.head(z)
