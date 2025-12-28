from config.base_config import BaseConfig

class ResNet50Config(BaseConfig):
    """ResNet50 backbone"""
    experiment_name = "backbone_resnet50"
    backbone = "resnet50"


class EfficientNetB0Config(BaseConfig):
    """EfficientNet-B0 backbone"""
    experiment_name = "backbone_efficientnet_b0"
    backbone = "efficientnet_b0"


class MobileNetV3Config(BaseConfig):
    """MobileNetV3 backbone"""
    experiment_name = "backbone_mobilenet_v3"
    backbone = "mobilenet_v3"


class ViTSmallConfig(BaseConfig):
    """Vision Transformer Small"""
    experiment_name = "backbone_vit_small"
    backbone = "vit_small_patch16_224"
    batch_size = 32  