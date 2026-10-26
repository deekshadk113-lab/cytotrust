import json

import numpy as np

from tools.fit_temperatures import fit_temperature, main


def _overconfident(n=4000, factor=3.0, seed=0):
    rng = np.random.default_rng(seed)
    signal = rng.normal(0, 1.5, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-signal))).astype(int)   # true probability is sigmoid(signal)
    probs = 1 / (1 + np.exp(-factor * signal))                    # the model is `factor` times too confident
    return probs, y


def test_fit_recovers_overconfidence():
    probs, y = _overconfident(factor=3.0)
    assert 2.4 < fit_temperature(probs, y) < 3.8


def test_fit_is_near_one_for_calibrated_model():
    probs, y = _overconfident(factor=1.0)
    assert 0.8 < fit_temperature(probs, y) < 1.25


def test_cli_writes_calibration_file(tmp_path):
    probs, y = _overconfident(factor=2.0)
    np.savez(tmp_path / "preds_convnext_tiny_seed0_val.npz", probs=probs, y=y)
    np.savez(tmp_path / "preds_convnext_tiny_seed1_val.npz", probs=probs, y=y)   # other seeds are ignored
    out = tmp_path / "models" / "calibration.json"
    assert main(["--preds", str(tmp_path), "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert list(data["temperatures"]) == ["convnext_tiny"]
    assert 1.5 < data["temperatures"]["convnext_tiny"] < 2.7
    assert data["n_val"]["convnext_tiny"] == 4000


def test_cli_fails_cleanly_without_files(tmp_path):
    assert main(["--preds", str(tmp_path), "--out", str(tmp_path / "x.json")]) == 1
