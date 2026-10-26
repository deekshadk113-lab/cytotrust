"""Static facts about the models. No heavy imports here, so the API can start without PyTorch."""

MODEL_SPECS = {
    "convnext_tiny": {"name": "ConvNeXt-Tiny", "cam": True},
    "efficientnet_v2_s": {"name": "EfficientNetV2-S", "cam": True},
    "mobilenet_v3_large": {"name": "MobileNetV3-Large", "cam": True},
    "resnet50": {"name": "ResNet-50", "cam": True},
    "swin_t": {"name": "Swin-T", "cam": False},
}

# Order in which the explanation (Grad-CAM) model is chosen when the ensemble is used.
CAM_PREFERENCE = ["convnext_tiny", "resnet50", "efficientnet_v2_s", "mobilenet_v3_large"]

ENSEMBLE_ID = "ensemble"
