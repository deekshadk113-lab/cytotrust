import io
import json
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .errors import ModelUnavailable
from .registry import ENSEMBLE_ID, MODEL_SPECS

BASE = Path(__file__).parent
STATIC = BASE / "static"
MAX_UPLOAD = int(float(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024)
MAX_PIXELS = 25_000_000


def create_app(engine=None):
    """Build the app. Tests pass a fake engine; production creates the real one on first use."""
    app = FastAPI(
        title="CytoTrust API",
        version="1.0.0",
        description="Cervical cell classification with calibrated confidence and referral. Research prototype, not a medical device.",
    )
    state = {"engine": engine}

    def get_engine():
        if state["engine"] is None:
            from .inference import Engine

            state["engine"] = Engine(
                model_dir=os.getenv("MODEL_DIR", "models"),
                calibration_file=os.getenv("CALIBRATION_FILE"),
            )
        return state["engine"]

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/models")
    def models():
        try:
            info = get_engine().models_info()
        except ModelUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        return {"models": info, "ensemble_available": any(m["available"] for m in info)}

    @app.get("/api/results")
    def results():
        path = STATIC / "data" / "results.json"
        return json.loads(path.read_text(encoding="utf-8"))

    @app.post("/api/predict")
    def predict(
        file: UploadFile = File(...),
        model: str = Form(ENSEMBLE_ID),
        calibrated: bool = Form(True),
        threshold: float = Form(0.7),
        explain: bool = Form(True),
    ):
        if model != ENSEMBLE_ID and model not in MODEL_SPECS:
            raise HTTPException(status_code=422, detail=f"Unknown model '{model}'.")
        if not 0.5 <= threshold <= 0.999:
            raise HTTPException(status_code=422, detail="The confidence limit must be between 0.5 and 0.999.")

        raw = file.file.read(MAX_UPLOAD + 1)
        if not raw:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        if len(raw) > MAX_UPLOAD:
            raise HTTPException(status_code=413, detail=f"The image is larger than {MAX_UPLOAD // (1024 * 1024)} MB.")

        try:
            image = Image.open(io.BytesIO(raw))
            if image.width * image.height > MAX_PIXELS:
                raise HTTPException(status_code=413, detail="The image has too many pixels. Crop it to a single cell first.")
            image = image.convert("RGB")
        except HTTPException:
            raise
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError):
            raise HTTPException(status_code=400, detail="This file could not be read as an image. Use PNG, JPEG, BMP or TIFF.")

        try:
            return get_engine().predict(image, model_id=model, calibrated=calibrated, threshold=threshold, explain=explain)
        except ModelUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
