import importlib.util
import json

import pytest
from PIL import Image

from app.errors import ModelUnavailable
from app.inference import Engine
from app.registry import MODEL_SPECS


def test_models_info_reflects_checkpoints_and_calibration(tmp_path):
    (tmp_path / "convnext_tiny_seed0.pt").write_bytes(b"x")
    (tmp_path / "resnet50.pt").write_bytes(b"x")
    (tmp_path / "calibration.json").write_text(json.dumps({"temperatures": {"convnext_tiny": 1.3}}))
    info = {m["id"]: m for m in Engine(model_dir=tmp_path).models_info()}
    assert set(info) == set(MODEL_SPECS)
    assert info["convnext_tiny"]["available"] and info["convnext_tiny"]["temperature"] == 1.3
    assert info["convnext_tiny"]["explainable"]
    assert info["resnet50"]["available"] and info["resnet50"]["temperature"] is None
    assert not info["swin_t"]["available"]


def test_predict_without_checkpoints_is_a_clean_error(tmp_path):
    with pytest.raises(ModelUnavailable):
        Engine(model_dir=tmp_path).predict(Image.new("RGB", (30, 30)))


@pytest.mark.skipif(importlib.util.find_spec("torch") is not None, reason="only meaningful when PyTorch is missing")
def test_predict_without_torch_is_a_clean_error(tmp_path):
    (tmp_path / "convnext_tiny_seed0.pt").write_bytes(b"x")
    with pytest.raises(ModelUnavailable, match="not installed"):
        Engine(model_dir=tmp_path).predict(Image.new("RGB", (30, 30)))
