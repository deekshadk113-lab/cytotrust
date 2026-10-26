"""Model loading and prediction. PyTorch is imported lazily so importing this module never fails."""
import json
import threading
import time
from pathlib import Path

from PIL import Image

from . import gradcam
from .calibration import average, calibrate, decide, sigmoid
from .errors import ModelUnavailable
from .registry import CAM_PREFERENCE, ENSEMBLE_ID, MODEL_SPECS

IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_arch(name):
    """Rebuild the architecture exactly as in the training notebook. Returns (model, cam_target_layer)."""
    import torch.nn as nn
    from torchvision import models

    if name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, 2))
        return m, m.layer4[-1]
    if name == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, 2))
        return m, m.features[-1]
    if name == "convnext_tiny":
        m = models.convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, 2)
        return m, m.features[-1]
    if name == "swin_t":
        m = models.swin_t(weights=None)
        m.head = nn.Linear(m.head.in_features, 2)
        return m, None
    if name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
        return m, m.features[-1]
    raise ValueError(f"unknown model: {name}")


class Engine:
    def __init__(self, model_dir="models", calibration_file=None):
        self.model_dir = Path(model_dir)
        cal = Path(calibration_file) if calibration_file else self.model_dir / "calibration.json"
        self.temperatures = {}
        if cal.exists():
            self.temperatures = json.loads(cal.read_text()).get("temperatures", {})
        self._cache = {}
        self._lock = threading.Lock()
        self._device = None
        self._transform = None

    # ---- discovery -------------------------------------------------------------------------
    def checkpoint_path(self, model_id):
        for name in (f"{model_id}_seed0.pt", f"{model_id}.pt"):
            p = self.model_dir / name
            if p.exists():
                return p
        return None

    def is_available(self, model_id):
        return self.checkpoint_path(model_id) is not None

    def models_info(self):
        info = []
        for mid, spec in MODEL_SPECS.items():
            ok = self.is_available(mid)
            info.append({
                "id": mid,
                "name": spec["name"],
                "available": ok,
                "explainable": bool(ok and spec["cam"]),
                "temperature": self.temperatures.get(mid),
            })
        return info

    # ---- loading ---------------------------------------------------------------------------
    def _setup_torch(self):
        try:
            import torch
            from torchvision import transforms
        except ImportError as exc:
            raise ModelUnavailable("PyTorch and torchvision are not installed on the server.") from exc
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])
        return torch

    def _load(self, model_id):
        if model_id in self._cache:
            return self._cache[model_id]
        torch = self._setup_torch()
        path = self.checkpoint_path(model_id)
        if path is None:
            raise ModelUnavailable(f"No checkpoint for {model_id} in {self.model_dir}.")
        model, layer = build_arch(model_id)
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model.to(self._device).eval()
        self._cache[model_id] = (model, layer)
        return self._cache[model_id]

    # ---- prediction ------------------------------------------------------------------------
    def predict(self, image: Image.Image, model_id=ENSEMBLE_ID, calibrated=True, threshold=0.7, explain=True):
        started = time.perf_counter()
        if model_id == ENSEMBLE_ID:
            ids = [m for m in MODEL_SPECS if self.is_available(m)]
        elif model_id in MODEL_SPECS and self.is_available(model_id):
            ids = [model_id]
        else:
            ids = []
        if not ids:
            raise ModelUnavailable("The requested model has no checkpoint on the server. See the README for setup.")

        with self._lock:
            torch = self._setup_torch()
            image = image.convert("RGB")
            x = self._transform(image).unsqueeze(0).to(self._device)
            per_model, warnings = [], []
            with torch.no_grad():
                for mid in ids:
                    model, _ = self._load(mid)
                    logits = model(x)[0]
                    z = float(logits[1] - logits[0])
                    temp = self.temperatures.get(mid)
                    if temp is None:
                        temp = 1.0
                        if calibrated:
                            warnings.append(f"No calibration temperature found for {MODEL_SPECS[mid]['name']}; T = 1 was used.")
                    per_model.append({
                        "id": mid, "name": MODEL_SPECS[mid]["name"], "temperature": temp,
                        "p_raw": sigmoid(z), "p_cal": calibrate(z, temp),
                    })

            key = "p_cal" if calibrated else "p_raw"
            p = average(r[key] for r in per_model)
            decision = decide(p, threshold)
            if model_id == ENSEMBLE_ID and len(ids) < len(MODEL_SPECS):
                warnings.append(f"The ensemble used {len(ids)} of {len(MODEL_SPECS)} models (the others have no checkpoint).")

            cam_png, cam_model = None, None
            if explain:
                candidates = CAM_PREFERENCE if model_id == ENSEMBLE_ID else [model_id]
                cam_id = next((m for m in candidates if m in ids and MODEL_SPECS[m]["cam"]), None)
                if cam_id is None:
                    warnings.append("No heatmap is available for this model (Grad-CAM supports the convolutional models).")
                else:
                    model, layer = self._load(cam_id)
                    cls = 1 if decision["label"] == "abnormal" else 0
                    cam = gradcam.compute_cam(model, layer, x, cls)
                    cam_png = gradcam.to_png_base64(gradcam.overlay(image.resize((IMG_SIZE, IMG_SIZE)), cam))
                    cam_model = MODEL_SPECS[cam_id]["name"]

        return {
            "verdict": decision["verdict"],
            "label": decision["label"],
            "p_abnormal": p,
            "confidence": decision["confidence"],
            "threshold": threshold,
            "calibrated": bool(calibrated),
            "models": per_model,
            "gradcam": cam_png,
            "gradcam_model": cam_model,
            "warnings": warnings,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        }
