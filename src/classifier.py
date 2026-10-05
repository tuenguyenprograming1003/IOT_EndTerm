"""Task 2.1 — reference CNN classifier on normalised log-mel (1 x 64 x 32) -> 10 logits."""
import torch
from torch import nn


class DigitCNN(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).mean(dim=(2, 3))  # global average pooling
        return self.head(h)                     # logits


class DigitCNNv2(nn.Module):
    """Follow-up classifier: same 3-conv + GAP + linear design as DigitCNN, but with configurable
    per-layer strides (e.g. (2,1) keeps time resolution longer) and dropout before the linear head."""

    def __init__(self, strides=((2, 2), (2, 2), (2, 2)), dropout: float = 0.0, n_classes: int = 10):
        super().__init__()
        chans = (1, 32, 64, 128)
        layers = []
        for i, s in enumerate(strides):
            layers += [nn.Conv2d(chans[i], chans[i + 1], 3, stride=tuple(s), padding=1), nn.ReLU()]
        self.features = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.features(x).mean(dim=(2, 3))))


ARCHS = {
    "v1": dict(strides=((2, 2), (2, 2), (2, 2))),   # spec architecture
    "t1": dict(strides=((2, 1), (2, 2), (2, 2))),   # keep time resolution in layer 1
    "t2": dict(strides=((2, 1), (2, 1), (2, 2))),   # keep time resolution in layers 1-2
}


def build_classifier(arch: dict | None = None) -> nn.Module:
    if not arch:
        return DigitCNN()
    return DigitCNNv2(strides=ARCHS[arch["name"]]["strides"], dropout=arch.get("dropout", 0.0))


def load_classifier(path, device="cpu") -> nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_classifier(ckpt.get("arch"))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def predict(model: nn.Module, x, batch: int = 500) -> torch.Tensor:
    """x: tensor/ndarray (N, 64, 32) or (N, 1, 64, 32). Returns predicted labels (N,)."""
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.dim() == 3:
        x = x.unsqueeze(1)
    dev = next(model.parameters()).device
    return torch.cat([model(x[i:i + batch].to(dev)).argmax(1).cpu() for i in range(0, len(x), batch)])
