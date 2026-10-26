"""Quick self-check for a machine that has PyTorch and the checkpoints.

    python tools/check_setup.py

For every checkpoint found in models/ it loads the weights, runs a prediction on a random test image and, for the
convolutional models, a Grad-CAM heatmap. It prints OK or FAIL for each step, so problems show up before the demo.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from app.errors import ModelUnavailable  # noqa: E402
from app.inference import Engine  # noqa: E402
from app.registry import MODEL_SPECS  # noqa: E402


def main():
    engine = Engine(model_dir=Path(__file__).resolve().parents[1] / "models")
    present = [m for m in MODEL_SPECS if engine.is_available(m)]
    if not present:
        print("FAIL  no checkpoints found in models/ (see models/README.md)")
        return 1
    print("calibration temperatures:", engine.temperatures or "none found (run tools/fit_temperatures.py)")

    rng = np.random.default_rng(0)
    image = Image.fromarray((rng.random((96, 96, 3)) * 255).astype("uint8"))
    failed = 0
    for mid in present:
        try:
            started = time.perf_counter()
            result = engine.predict(image, model_id=mid, calibrated=True, threshold=0.7, explain=True)
            took = time.perf_counter() - started
            cam = "heatmap ok" if result["gradcam"] else "no heatmap (expected for Swin-T)"
            print(f"OK    {mid}: p(abnormal) = {result['p_abnormal']:.3f}, {cam}, {took:.1f} s")
        except ModelUnavailable as exc:
            print(f"FAIL  {mid}: {exc}")
            failed += 1
        except Exception as exc:  # noqa: BLE001  report any loading problem clearly
            print(f"FAIL  {mid}: {type(exc).__name__}: {exc}")
            failed += 1
    if len(present) > 1:
        try:
            result = engine.predict(image, model_id="ensemble", explain=True)
            print(f"OK    ensemble of {len(result['models'])}: p(abnormal) = {result['p_abnormal']:.3f}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL  ensemble: {type(exc).__name__}: {exc}")
            failed += 1
    print("All checks passed." if not failed else f"{failed} check(s) failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
