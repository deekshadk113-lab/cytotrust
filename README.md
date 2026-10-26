# CytoTrust

Cervical cell classification with calibrated confidence and referral. Final-year major project.

> **Research prototype. Not a medical device. Do not use it for clinical decisions.**

CytoTrust takes one Pap smear cell image and answers **normal**, **abnormal**, or **refer to a pathologist**. The models are
ImageNet-pretrained networks fine-tuned on SIPaKMeD. What makes the project different from a plain classifier is that it is built
around the findings of the accompanying study: models that look excellent on their own dataset lose a lot on a new one, and their
confidence has to be calibrated and used carefully.

## What the app does

- **Analyze page:** upload a cell crop, pick one model or the ensemble, and get a verdict, the probability of abnormal, the
  confidence, a per-model breakdown, and a Grad-CAM heatmap.
- **Referral rule:** if the confidence (the larger of the two class probabilities) is below a limit you set, the cell is
  referred instead of labeled. The result shows a ruler with the referral zone, so the effect of the limit is visible.
- **Calibration:** probabilities are corrected with one temperature per model, fitted on the SIPaKMeD validation set only.
- **Ensemble:** averages the calibrated probabilities of every model whose checkpoint is present.
- **Study results page:** the benchmark from the paper (accuracy gap, calibration, ensembles, training on Herlev), with charts.
- **REST API** with automatic documentation at `/docs`.

## How it works

```
Browser (HTML, CSS, JavaScript)
   |  POST /api/predict  (image, model, limit, options)
   v
FastAPI  (app/main.py)         validates the file: type, size, pixel count
   |
   v
Engine   (app/inference.py)    resize to 224, ImageNet normalization
   |   for each model:  z = logit(abnormal) - logit(normal)
   |                    p_raw = sigmoid(z),  p_cal = sigmoid(z / T)     (temperature scaling)
   |   ensemble:        average of the p_cal values
   |   decision:        confidence = max(p, 1 - p);  refer if confidence < limit
   |   explanation:     Grad-CAM on a convolutional model (app/gradcam.py)
   v
JSON result  ->  verdict, probabilities, heatmap, warnings
```

## Quick start

Requires Python 3.10 or newer.

**1. Get the trained model weights.** Checkpoint files are large and are not included in this repository. They come from the
Part 1 training notebook on Kaggle (`deekshadk/part-a`). Download the `<model>_seed0.pt` files for the models you want from
the notebook's Output tab, or with the Kaggle command line tool (`pip install kaggle`, put your API token in
`~/.kaggle/kaggle.json`):

```bash
kaggle kernels output deekshadk/part-a -p kaggle_output
```

This downloads every output file, which can be over 1 GB. You only need the seed-0 checkpoint files. The app uses the
calibration temperatures in `models/calibration.json`, which is included in this repository; validation prediction files
are not needed to run it.

**2. Install.**

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**3. Add the checkpoints.** Copy the `*_seed0.pt` files into `models/`. One model is enough to run the app.

**4. (Optional) Refit the calibration temperatures.** Skip this if you are using the included `models/calibration.json`.
To refit, download the matching `preds_<model>_seed0_val.npz` validation prediction files from the notebook output into
`kaggle_output`, then run:

```bash
python tools/fit_temperatures.py --preds kaggle_output --out models/calibration.json
```

**5. Check the setup.** This loads every checkpoint, makes a prediction and a heatmap, and prints OK or FAIL for each step.

```bash
python tools/check_setup.py
```

**6. Run.**

```bash
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. The interactive API documentation is at http://127.0.0.1:8000/docs.

## API

| Method and path | Purpose |
|---|---|
| `GET /api/health` | Liveness check |
| `GET /api/models` | Which models have checkpoints, their temperatures, whether a heatmap is available |
| `GET /api/results` | The study benchmark shown on the results page |
| `POST /api/predict` | Analyze one image |

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -F "file=@cell.png" -F "model=ensemble" -F "calibrated=true" -F "threshold=0.7" -F "explain=true"
```

Form fields: `file` (PNG, JPEG, BMP or TIFF), `model` (`ensemble`, `convnext_tiny`, `efficientnet_v2_s`, `mobilenet_v3_large`,
`resnet50`, `swin_t`), `calibrated` (default true), `threshold` (0.5 to 0.999, default 0.7), `explain` (default true).

The response contains `verdict` (`normal`, `abnormal` or `refer`), `p_abnormal`, `confidence`, per-model probabilities,
an optional base64 PNG heatmap, and any warnings. Errors use standard status codes: 400 unreadable image, 413 too large,
422 invalid option, 503 models or PyTorch missing.

## Configuration

| Environment variable | Default | Meaning |
|---|---|---|
| `MODEL_DIR` | `models` | Folder holding the checkpoints and `calibration.json` |
| `CALIBRATION_FILE` | `<MODEL_DIR>/calibration.json` | Temperature file |
| `MAX_UPLOAD_MB` | `5` | Largest accepted upload |

## Docker

```bash
docker build -t cytotrust .
docker run -p 8000:8000 cytotrust
```

The image copies `models/` at build time, so add your checkpoints and `calibration.json` first, or mount the folder:
`docker run -p 8000:8000 -v "$PWD/models:/srv/models" cytotrust`. The Dockerfile installs CPU-only PyTorch.
To host it online (for example a Docker Space on Hugging Face or a small cloud VM), a few models are easier than all five:
each checkpoint is roughly 20 to 120 MB, and the ensemble loads every model into memory.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests cover the calibration and referral logic, temperature fitting, heatmap helpers, model discovery, and the whole API
(uploads, validation, error codes) using a stand-in engine. The PyTorch parts (loading weights, forward pass, Grad-CAM)
need real checkpoints, so they are covered by `python tools/check_setup.py` instead.

## Project structure

```
app/
  main.py          FastAPI app, validation, routes
  inference.py     model loading, ensemble, calibration, decision
  calibration.py   sigmoid, temperature scaling, referral rule (pure Python)
  gradcam.py       Grad-CAM and heatmap overlay
  registry.py      model names and which ones support heatmaps
  static/          the web interface (index.html, style.css, app.js, data/results.json)
models/            checkpoints and calibration.json (not committed)
tools/             fit_temperatures.py, check_setup.py
tests/             pytest suite
```

## What the study found (summary)

Five backbones (ResNet-50, EfficientNetV2-S, ConvNeXt-Tiny, Swin-T, MobileNetV3-Large) were fine-tuned on SIPaKMeD and tested
on Herlev without further training.

- Accuracy on SIPaKMeD was 94% to 97%. On Herlev it fell to 51% to 79%, and AUC to 0.65 to 0.71.
- Temperature scaling fitted on the source dataset reduced calibration error on Herlev only partly.
- Averaging models improved calibration but not discrimination.
- Trained on Herlev itself, the same models reached 92% to 94% accuracy, so the drop is dataset shift, not a hard dataset.

The numbers on the results page come from `app/static/data/results.json`. Update that file if you rerun the experiments.

## Limits and responsible use

- The models only know single-cell crops like those in SIPaKMeD and Herlev, with two labels. Whole slides, clusters and other
  stains are out of scope.
- Calibration helps on data like the training set. On other datasets confidence is a weak signal of which answers are wrong,
  so a referral limit is a convenience, not a safety guarantee.
- Heatmaps show influence on the prediction, not pathology.
- Uploaded images are processed in memory and never stored. Do not upload patient-identifiable data to a public deployment.
- This is a student research prototype. It has had no clinical validation.

## Data and credits

- SIPaKMeD: M. E. Plissiti et al., "SIPaKMeD: A new dataset for feature and image based classification of normal and
  pathological cervical cells in Pap smear images," IEEE ICIP 2018.
- Herlev Pap smear database: J. Jantzen et al., "Pap-smear benchmark data for pattern classification," NiSIS 2005
  (data from the MDE-Lab, University of the Aegean).
- Backbones are the torchvision implementations with their original architectures.

Before presenting, edit the footer in `app/static/index.html` to add your name, your guide's name and the year.

## Troubleshooting

- **"No model checkpoints were found":** the files must be in `models/` (or `MODEL_DIR`) and named `<model>_seed0.pt` or `<model>.pt`.
- **`size mismatch` or `Missing key(s)` when loading:** the checkpoint was trained with a different architecture than its name.
  The app rebuilds the networks exactly as in the training notebook.
- **Probabilities look too confident:** `models/calibration.json` is missing, so T = 1 is used. Run step 4.
- **Heatmap missing for Swin-T:** expected. Grad-CAM here supports the convolutional models.
