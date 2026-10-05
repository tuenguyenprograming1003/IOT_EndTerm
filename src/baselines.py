"""Tasks 3.2–3.4 — non-learned baselines operating on normalised log-mel x (N, 64, 32).

Each function returns the reconstruction x_hat that is fed to the classifier.
DCT and Low-Mel keep d coefficients/values; the kept values are sent as INT8 with a fixed
per-method clipping range so that the payload is exactly d bytes (same as the AE codecs).
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dctn, idctn

from quantizer import QMAX, X_CLIP, dequantize_direct, quantize_direct

H, W = 64, 32


def zigzag_indices(h: int = H, w: int = W) -> np.ndarray:
    """Fixed JPEG-style zig-zag order over an h x w grid; returns (h*w, 2) [row, col]."""
    order = []
    for s in range(h + w - 1):
        diag = [(i, s - i) for i in range(max(0, s - w + 1), min(h, s + 1))]
        order.extend(diag if s % 2 else diag[::-1])
    return np.array(order)


ZIGZAG = zigzag_indices()


def direct_int8(x: np.ndarray) -> np.ndarray:
    return dequantize_direct(quantize_direct(x))


def dct_coeffs(x: np.ndarray) -> np.ndarray:
    return dctn(x, type=2, norm="ortho", axes=(-2, -1))


def dct_keep(x: np.ndarray, d: int, clip: np.ndarray | None = None) -> np.ndarray:
    """Keep the first d zig-zag DCT-II coefficients, zero the rest, inverse DCT.
    If clip (shape (d,)) is given, kept coefficient k is INT8-quantised with range [-clip[k], clip[k]]."""
    C = dct_coeffs(x)
    rows, cols = ZIGZAG[:d, 0], ZIGZAG[:d, 1]
    kept = C[..., rows, cols]
    if clip is not None:
        kept = np.clip(np.round(kept / clip * QMAX), -QMAX, QMAX).astype(np.int8).astype(np.float32) / QMAX * clip
    C2 = np.zeros_like(C)
    C2[..., rows, cols] = kept
    return idctn(C2, type=2, norm="ortho", axes=(-2, -1)).astype(np.float32)


def lowmel_keep(x: np.ndarray, d: int, quantize: bool = True) -> np.ndarray:
    """Keep the first m = d/32 mel rows (all 32 frames); other rows = 0 in the normalised domain."""
    m = d // W
    assert m * W == d
    out = np.zeros_like(x)
    kept = x[..., :m, :]
    out[..., :m, :] = direct_int8(kept) if quantize else kept
    return out


def fit_dct_clip(x_train: np.ndarray, d: int, pct: float = 99.9) -> np.ndarray:
    """Per-coefficient clipping ranges (d,) for DCT quantisation, fitted on Train only.
    This fixed table is shared by node and gateway, so it costs no payload bytes."""
    C = dct_coeffs(x_train)[..., ZIGZAG[:d, 0], ZIGZAG[:d, 1]]
    return np.maximum(np.percentile(np.abs(C), pct, axis=0), 1e-6).astype(np.float32)


__all__ = ["direct_int8", "dct_keep", "lowmel_keep", "fit_dct_clip", "ZIGZAG", "X_CLIP"]
