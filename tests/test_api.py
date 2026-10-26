import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.errors import ModelUnavailable
from app.main import create_app


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.fail = False

    def models_info(self):
        if self.fail:
            raise ModelUnavailable("PyTorch is missing.")
        return [{"id": "convnext_tiny", "name": "ConvNeXt-Tiny", "available": True, "explainable": True, "temperature": 1.2}]

    def predict(self, image, model_id, calibrated, threshold, explain):
        if self.fail:
            raise ModelUnavailable("No checkpoint.")
        self.calls.append((image.size, model_id, calibrated, threshold, explain))
        return {"verdict": "abnormal", "p_abnormal": 0.9}


def png_bytes(size=(20, 20), fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 40, 90)).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def setup():
    engine = FakeEngine()
    return TestClient(create_app(engine)), engine


def post(client, data=None, content=None, name="cell.png", **form):
    files = {"file": (name, content if content is not None else png_bytes(), "image/png")}
    return client.post("/api/predict", files=files, data=form)


def test_pages_and_health(setup):
    client, _ = setup
    assert client.get("/api/health").json() == {"status": "ok"}
    page = client.get("/")
    assert page.status_code == 200 and "CytoTrust" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/api/results").json()["order"][0] == "convnext_tiny"
    assert page.headers["x-content-type-options"] == "nosniff"
    assert client.get("/api/health").headers["cache-control"] == "no-store"


def test_models_endpoint(setup):
    client, engine = setup
    body = client.get("/api/models").json()
    assert body["ensemble_available"] is True and body["models"][0]["id"] == "convnext_tiny"
    engine.fail = True
    assert client.get("/api/models").status_code == 503


def test_predict_passes_options_to_the_engine(setup):
    client, engine = setup
    r = post(client, model="convnext_tiny", calibrated="false", threshold="0.8", explain="false")
    assert r.status_code == 200 and r.json()["verdict"] == "abnormal"
    assert engine.calls[-1] == ((20, 20), "convnext_tiny", False, 0.8, False)


def test_predict_defaults(setup):
    client, engine = setup
    assert post(client).status_code == 200
    assert engine.calls[-1] == ((20, 20), "ensemble", True, 0.7, True)


@pytest.mark.parametrize("form", [{"threshold": "0.2"}, {"threshold": "1.5"}, {"model": "gpt"}])
def test_predict_rejects_bad_options(setup, form):
    client, engine = setup
    assert post(client, **form).status_code == 422
    assert not engine.calls


def test_predict_rejects_bad_files(setup, monkeypatch):
    client, engine = setup
    assert post(client, content=b"").status_code == 400
    assert post(client, content=b"not an image at all").status_code == 400
    monkeypatch.setattr("app.main.MAX_UPLOAD", 50)
    assert post(client).status_code == 413
    monkeypatch.setattr("app.main.MAX_UPLOAD", 5 * 1024 * 1024)
    monkeypatch.setattr("app.main.MAX_PIXELS", 10)
    assert post(client).status_code == 413
    assert not engine.calls


def test_predict_accepts_jpeg_and_bmp(setup):
    client, engine = setup
    assert post(client, content=png_bytes(fmt="JPEG"), name="a.jpg").status_code == 200
    assert post(client, content=png_bytes(fmt="BMP"), name="a.bmp").status_code == 200
    assert len(engine.calls) == 2


def test_engine_failure_becomes_503(setup):
    client, engine = setup
    engine.fail = True
    r = post(client)
    assert r.status_code == 503 and r.json()["detail"] == "No checkpoint."
