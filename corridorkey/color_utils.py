import functools

import torch
import numpy as np

HDR_ASINH_KNEE = 0.008
HDR_ASINH_REF = 16.0

def _is_tensor(x):
    return isinstance(x, torch.Tensor)


@functools.cache
def _side_stream(device: int) -> torch.cuda.Stream:
    return torch.cuda.Stream(device)


def _on_gpu(fn, x: np.ndarray) -> np.ndarray:
    """``fn`` (a tensor transform) of numpy ``x``, on the GPU when there is one, on a side stream so it never
    waits for the model."""
    t = torch.from_numpy(np.ascontiguousarray(x))
    if not torch.cuda.is_available():
        return fn(t).numpy()
    with torch.cuda.stream(_side_stream(torch.cuda.current_device())):
        return fn(t.cuda()).cpu().numpy()

def _signed_pow(x, gamma):
    """Sign-preserving power. For x ≥ 0 this is identical to `x.pow(gamma)`.
    For x < 0 it returns -|x|^gamma — keeps the operation well-defined (no
    NaN) and lets gradient flow through negative inputs.

    For the physically-valid [0, 1] inputs produced by the dataset the
    behavior is bit-for-bit identical to the previous clamp-based form.
    `to_srgb` uses its own epsilon-shifted branch because gamma < 1 has an
    infinite mathematical slope at zero.
    """
    if _is_tensor(x):
        pos = x.clamp(min=0.0).pow(gamma)
        neg = -((-x).clamp(min=0.0).pow(gamma))
        return torch.where(x >= 0, pos, neg)
    if isinstance(x, np.ndarray):
        return _on_gpu(lambda t: _signed_pow(t, gamma), x)
    return np.sign(x) * np.power(np.abs(x), gamma)


def linear_to_srgb(x):
    """Scene-linear RGB → sRGB (IEC 61966-2-1). Numpy or torch input.

    Uses standard IEC 61966-2-1 equations:
    - x <= 0.0031308: 12.92 * x
    - x > 0.0031308: 1.055 * x^(1/2.4) - 0.055

    To prevent gradient NaN at zero during backpropagation in PyTorch,
    we clamp the power branch input to a minimum positive threshold.
    """
    if _is_tensor(x):
        pos_x = x.clamp(min=0.0)
        pos_pow = pos_x.clamp(min=0.0031308).pow(1.0 / 2.4)
        pos = torch.where(pos_x <= 0.0031308, pos_x * 12.92, 1.055 * pos_pow - 0.055)
        
        neg_x = (-x).clamp(min=0.0)
        neg_pow = neg_x.clamp(min=0.0031308).pow(1.0 / 2.4)
        neg = -torch.where(neg_x <= 0.0031308, neg_x * 12.92, 1.055 * neg_pow - 0.055)
        return torch.where(x >= 0, pos, neg)
    return _on_gpu(linear_to_srgb, x)


def srgb_to_linear(x):
    """sRGB → Scene-linear RGB (IEC 61966-2-1). Numpy or torch input.

    Uses standard IEC 61966-2-1 equations:
    - x <= 0.04045: x / 12.92
    - x > 0.04045: ((x + 0.055) / 1.055)^2.4
    """
    if _is_tensor(x):
        pos_x = x.clamp(min=0.0)
        pos_pow = ((pos_x + 0.055) / 1.055).pow(2.4)
        pos = torch.where(pos_x <= 0.04045, pos_x / 12.92, pos_pow)
        
        neg_x = (-x).clamp(min=0.0)
        neg_pow = ((neg_x + 0.055) / 1.055).pow(2.4)
        neg = -torch.where(neg_x <= 0.04045, neg_x / 12.92, neg_pow)
        return torch.where(x >= 0, pos, neg)
    return _on_gpu(srgb_to_linear, x)


# Backwards compatibility aliases
to_srgb = linear_to_srgb
to_linear = srgb_to_linear


def linear_to_asinh(x, knee=HDR_ASINH_KNEE, ref=HDR_ASINH_REF):
    """Signed asinh encoding for scene-linear RGB.

    `ref` is a normalization reference, not a clamp. Values above `ref`
    encode above 1.0 and remain reversible. This is the model RGB contract for
    HDR-capable training: physical compositing stays in decoded linear light,
    while the network sees a compressed but exact signed representation.

    Native asinh is already odd and has finite nonzero derivative at zero, so
    this intentionally avoids sign/abs branch construction.
    """
    knee = float(knee)
    ref = float(ref)
    denom = np.arcsinh(ref / knee)
    if _is_tensor(x):
        scale = x.new_tensor(denom)
        k = x.new_tensor(knee)
        return torch.asinh(x / k) / scale
    if isinstance(x, np.ndarray):
        return _on_gpu(lambda t: linear_to_asinh(t, knee, ref), x)
    return np.arcsinh(x / knee) / denom


def asinh_to_linear(y, knee=HDR_ASINH_KNEE, ref=HDR_ASINH_REF):
    """Inverse of `linear_to_asinh`."""
    knee = float(knee)
    ref = float(ref)
    denom = np.arcsinh(ref / knee)
    if _is_tensor(y):
        scale = y.new_tensor(denom)
        k = y.new_tensor(knee)
        return torch.sinh(y * scale) * k
    return np.sinh(y * denom) * knee


def linear_to_vfm_sdr(x):
    """Tone-limited proxy for foundation models that expect SDR RGB [0, 1]."""
    return np.clip(linear_to_srgb(np.maximum(x, 0.0)), 0.0, 1.0)


def linear_to_vfm_sdr_torch(x):
    """Torch version of `linear_to_vfm_sdr`."""
    return linear_to_srgb(x.clamp(min=0.0)).clamp(0.0, 1.0)

def premultiply(fg, alpha):
    """
    Premultiplies foreground by alpha.
    fg: Color [..., C] or [C, ...]
    alpha: Alpha [..., 1] or [1, ...]
    """
    return fg * alpha

def unpremultiply(fg, alpha, eps=1e-6):
    """
    Un-premultiplies foreground by alpha.
    Ref: fg_straight = fg_premul / (alpha + eps)

    Uses max(alpha, 0) as the denominator floor so slightly-negative alpha
    (e.g. from Lanczos ringing) can't land on alpha == -eps → denom == 0.
    """
    if _is_tensor(fg):
        return fg / (alpha.clamp(min=0.0) + eps)
    return fg / (np.maximum(alpha, 0.0) + eps)

def composite_straight(fg, bg, alpha):
    """
    Composites Straight FG over BG.
    Formula: FG * Alpha + BG * (1 - Alpha)
    """
    return fg * alpha + bg * (1.0 - alpha)

def composite_premul(fg, bg, alpha):
    """
    Composites Premultiplied FG over BG.
    Formula: FG + BG * (1 - Alpha)
    """
    return fg + bg * (1.0 - alpha)

def rgb_to_yuv(image):
    """
    Converts RGB to YUV (Rec. 601).
    Input: [..., 3, H, W] or [..., 3] depending on layout. 
    Supports standard PyTorch BCHW.
    """
    if not _is_tensor(image):
        raise TypeError("rgb_to_yuv only supports dict/tensor inputs currently")

    # Weights for RGB -> Y
    # Rec. 601: 0.299, 0.587, 0.114
    
    # Assume BCHW layout if 4 dims
    if image.dim() == 4:
        r = image[:, 0:1, :, :]
        g = image[:, 1:2, :, :]
        b = image[:, 2:3, :, :]
    elif image.dim() == 3 and image.shape[0] == 3: # CHW
        r = image[0:1, :, :]
        g = image[1:2, :, :]
        b = image[2:3, :, :]
    else:
        # Last dim conversion
        r = image[..., 0]
        g = image[..., 1]
        b = image[..., 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = 0.492 * (b - y)
    v = 0.877 * (r - y)
    
    if image.dim() >= 3 and image.shape[-3] == 3: # Concatenate along Channel dim
         return torch.cat([y, u, v], dim=-3)
    else:
         return torch.stack([y, u, v], dim=-1)
