"""Grad-CAM heatmaps. The PyTorch part is imported lazily; the image helpers need only NumPy and Pillow."""
import base64
import io

import numpy as np
from PIL import Image


def compute_cam(model, target_layer, x, class_idx):
    """Grad-CAM for one image tensor x (1x3xHxW). Returns an HxW array in [0, 1]."""
    import torch
    import torch.nn.functional as F

    store = {}

    def forward_hook(_module, _inputs, output):
        store["act"] = output
        output.register_hook(lambda grad: store.__setitem__("grad", grad))

    handle = target_layer.register_forward_hook(forward_hook)
    try:
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = model(x)
            model.zero_grad(set_to_none=True)
            logits[0, class_idx].backward()
    finally:
        handle.remove()

    act, grad = store["act"].detach(), store["grad"].detach()
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.cpu().numpy()


def jet(values):
    """Jet-like colormap for values in [0, 1]; returns an array of shape values.shape + (3,), uint8."""
    x = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a heatmap over the image (the image is resized to the heatmap size)."""
    h, w = cam.shape
    base = np.asarray(image.convert("RGB").resize((w, h)), dtype=np.float32)
    heat = jet(cam).astype(np.float32)
    mixed = (1 - alpha) * base + alpha * heat
    return Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8))


def to_png_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
