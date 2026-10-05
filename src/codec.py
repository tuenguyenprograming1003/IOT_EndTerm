"""Task 4.1 — convolutional autoencoder codec with an INT8 latent of d = 32 r values (r x 8 x 4).

No BatchNorm and no skip connections across the bottleneck.
"""
import torch
from torch import nn

from quantizer import dequantize, fake_quantize, quantize

R_FOR_D = {32: 1, 64: 2, 128: 4, 256: 8, 512: 16}


class Encoder(nn.Module):
    def __init__(self, r: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, r, 1), nn.Tanh(),
        )

    def forward(self, x):  # (B,1,64,32) -> (B,r,8,4) in [-1,1]
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, r: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(r, 128, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # linear output
        )

    def forward(self, z):  # (B,r,8,4) -> (B,1,64,32)
        return self.net(z)


class Codec(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        if d not in R_FOR_D:
            raise ValueError(f"d must be one of {list(R_FOR_D)}")
        self.d, self.r = d, R_FOR_D[d]
        self.encoder = Encoder(self.r)
        self.decoder = Decoder(self.r)

    def forward(self, x):
        """Training/eval path with fake quantisation (STE). Returns (x_hat, z)."""
        z = self.encoder(x)
        return self.decoder(fake_quantize(z)), z

    # --- real INT8 path (node / gateway split) ---
    @torch.no_grad()
    def encode_int8(self, x) -> torch.Tensor:
        """Node side: (B,1,64,32) -> int8 (B, d), flattened in (r, 8, 4) C-order."""
        return quantize(self.encoder(x)).reshape(len(x), -1)

    @torch.no_grad()
    def decode_int8(self, q) -> torch.Tensor:
        """Gateway side: int8 (B, d) -> x_hat (B,1,64,32)."""
        q = torch.as_tensor(q)
        return self.decoder(dequantize(q).reshape(len(q), self.r, 8, 4))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
