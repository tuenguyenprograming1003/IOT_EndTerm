"""Task 4.2 — symmetric INT8 quantisation of the tanh latent z in [-1, 1].

q = clip(round(127 z), -127, 127)  (int8),   z_hat = q / 127
Training uses fake quantisation with a straight-through estimator (identity gradient).
"""
import numpy as np
import torch

QMAX = 127


def quantize(z: torch.Tensor) -> torch.Tensor:
    """Real quantisation: float latent -> torch.int8 tensor."""
    return torch.clamp(torch.round(QMAX * z), -QMAX, QMAX).to(torch.int8)


def dequantize(q) -> torch.Tensor:
    return torch.as_tensor(q).to(torch.float32) / QMAX


def fake_quantize(z: torch.Tensor) -> torch.Tensor:
    """Forward: dequantize(quantize(z)); backward: identity (STE)."""
    zq = torch.clamp(torch.round(QMAX * z), -QMAX, QMAX) / QMAX
    return z + (zq - z).detach()


# Direct-INT8 baseline (whole normalised log-mel, no dimensionality reduction).
# A per-packet scale would need extra header bytes, so a fixed clipping range is used:
# x in [-X_CLIP, X_CLIP] -> 255 levels. X_CLIP is a fixed constant shared by node and gateway.
X_CLIP = 8.0


def quantize_direct(x: np.ndarray, clip: float = X_CLIP) -> np.ndarray:
    return np.clip(np.round(x / clip * QMAX), -QMAX, QMAX).astype(np.int8)


def dequantize_direct(q: np.ndarray, clip: float = X_CLIP) -> np.ndarray:
    return q.astype(np.float32) / QMAX * clip
