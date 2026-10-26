"""Fit one temperature per model on the SIPaKMeD validation predictions saved by the training notebook.

Usage:
    python tools/fit_temperatures.py --preds path/to/notebook_output --out models/calibration.json

It looks (recursively) for files named preds_<model>_seed<k>_val.npz. The checkpoint used by the app is the
seed-0 checkpoint, so seed 0 is used by default. Only validation data from the SOURCE dataset is used,
exactly as in the study (no target labels).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.registry import MODEL_SPECS  # noqa: E402

EPS = 1e-6


def to_logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def nll(logit, y, T):
    p = 1.0 / (1.0 + np.exp(-logit / T))
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(probs, y):
    lg = to_logit(np.asarray(probs, dtype=float))
    y = np.asarray(y, dtype=float)
    res = minimize_scalar(lambda T: nll(lg, y, T), bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True, help="folder that contains the preds_*_val.npz files (searched recursively)")
    ap.add_argument("--out", default="models/calibration.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    temps, n_val = {}, {}
    for mid in MODEL_SPECS:
        matches = sorted(Path(args.preds).rglob(f"preds_{mid}_seed{args.seed}_val.npz"))
        if not matches:
            print(f"  {mid}: no validation predictions found, skipped")
            continue
        d = np.load(matches[0])
        temps[mid] = round(fit_temperature(d["probs"], d["y"]), 4)
        n_val[mid] = int(len(d["y"]))
        print(f"  {mid}: T = {temps[mid]} (n = {n_val[mid]})")
    if not temps:
        print("No models found. Check --preds.")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "temperatures": temps,
        "n_val": n_val,
        "source": f"SIPaKMeD validation split, seed {args.seed}",
    }, indent=2))
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
