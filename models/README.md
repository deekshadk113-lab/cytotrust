# Model files

Put the trained checkpoints here, named `<model>_seed0.pt` (or `<model>.pt`):

```
convnext_tiny_seed0.pt
efficientnet_v2_s_seed0.pt
mobilenet_v3_large_seed0.pt
resnet50_seed0.pt
swin_t_seed0.pt
calibration.json          <- created by tools/fit_temperatures.py
```

The app uses whichever checkpoints it finds. One model is enough to run it, and the ensemble option averages
all the models that are present. Checkpoints are git-ignored because they are large.
