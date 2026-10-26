import math

import pytest

from app.calibration import average, calibrate, decide, sigmoid


def test_sigmoid_is_stable_and_symmetric():
    assert sigmoid(0) == 0.5
    assert sigmoid(800) == pytest.approx(1.0)
    assert sigmoid(-800) == pytest.approx(0.0)
    assert sigmoid(2) + sigmoid(-2) == pytest.approx(1.0)


def test_temperature_above_one_softens_probabilities():
    z = 3.0
    assert calibrate(z, 1.0) == pytest.approx(sigmoid(3.0))
    assert 0.5 < calibrate(z, 2.0) < calibrate(z, 1.0)
    assert calibrate(0.0, 5.0) == 0.5
    with pytest.raises(ValueError):
        calibrate(1.0, 0)


def test_temperature_never_flips_the_label():
    for z in (-4, -0.5, 0.5, 4):
        for t in (0.3, 1, 3):
            assert (calibrate(z, t) >= 0.5) == (z >= 0)


def test_decide_refers_below_the_confidence_limit():
    assert decide(0.95, 0.7) == {"label": "abnormal", "confidence": 0.95, "verdict": "abnormal"}
    assert decide(0.05, 0.7)["verdict"] == "normal"
    refer = decide(0.62, 0.7)
    assert refer["verdict"] == "refer" and refer["label"] == "abnormal"
    assert decide(0.35, 0.7)["verdict"] == "refer" and decide(0.35, 0.7)["label"] == "normal"
    assert decide(0.7, 0.7)["verdict"] == "abnormal"  # exactly at the limit is answered
    assert math.isclose(decide(0.3, 0.5)["confidence"], 0.7)


def test_average():
    assert average([0.2, 0.4, 0.9]) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        average([])
