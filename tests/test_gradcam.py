import base64
import io

import numpy as np
from PIL import Image

from app.gradcam import jet, overlay, to_png_base64


def test_jet_range_and_shape():
    colors = jet(np.linspace(0, 1, 11))
    assert colors.shape == (11, 3) and colors.dtype == np.uint8
    assert colors[0][2] > colors[0][0]      # low values are blue
    assert colors[-1][0] > colors[-1][2]    # high values are red


def test_overlay_matches_heatmap_size_and_png_roundtrip():
    img = Image.new("RGB", (60, 40), (200, 100, 50))
    cam = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    out = overlay(img, cam)
    assert out.size == (32, 32) and out.mode == "RGB"
    decoded = Image.open(io.BytesIO(base64.b64decode(to_png_base64(out))))
    assert decoded.format == "PNG" and decoded.size == (32, 32)
