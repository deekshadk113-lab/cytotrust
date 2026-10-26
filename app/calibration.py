"""Probability helpers used at prediction time (pure Python, no dependencies)."""
import math


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def calibrate(z: float, temperature: float) -> float:
    """Temperature scaling for a binary model: sigmoid(z / T), where z = logit(abnormal) - logit(normal)."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return sigmoid(z / temperature)


def average(values):
    values = list(values)
    if not values:
        raise ValueError("nothing to average")
    return sum(values) / len(values)


def decide(p_abnormal: float, threshold: float) -> dict:
    """Turn a probability into a verdict. Below the confidence threshold the cell is referred."""
    label = "abnormal" if p_abnormal >= 0.5 else "normal"
    confidence = max(p_abnormal, 1.0 - p_abnormal)
    verdict = label if confidence >= threshold else "refer"
    return {"label": label, "confidence": confidence, "verdict": verdict}
